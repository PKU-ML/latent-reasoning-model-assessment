"""
Test script for SkiLa-7B model on local MMVP and VSTAR datasets.
Uses Sketch-in-Latents approach for unified multimodal reasoning.

Reference: SkiLa training code and Monet/test_org.py for evaluation approach
"""

import sys
import os
from pathlib import Path
import argparse

# Set CUDA device before any imports
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import json
import csv
import re
import gc
from tqdm import tqdm

# Add SkiLa src to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

# Set HuggingFace cache to local path (可通过 SKILA_HF_CACHE 覆盖，默认使用 ~/.cache/huggingface)
os.environ.setdefault("HF_HOME", os.environ.get("SKILA_HF_HOME", os.path.expanduser("~/.cache/huggingface")))
os.environ.setdefault("HF_HUB_CACHE", os.environ.get("SKILA_HF_HUB_CACHE", os.path.expanduser("~/.cache/huggingface/hub")))

# Apply monkey patch to enable SkiLa sketch mode with latent vector storage
from src.train.monkey_patch_forward_skila_test import replace_qwen2_5_with_skila_forward
replace_qwen2_5_with_skila_forward()

from transformers import AutoProcessor, AutoConfig, CLIPProcessor, SiglipImageProcessor
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

from src.model.skila import SkiLa

# Model configuration
# 通过 SKILA_MODEL_PATH / SKILA_SKETCH_ENCODER 环境变量注入；默认使用相对路径
MODEL_PATH = os.environ.get("SKILA_MODEL_PATH", "./SkiLa-7B")
SKETCH_ENCODER = os.environ.get(
    "SKILA_SKETCH_ENCODER",
    "./models--google--siglip2-so400m-patch14-384/snapshots/e8e487298228002f3d8a82e0cd5c8ea9c567f57f",
)
SKETCH_TOKEN_NUM = 54  # Default from training script

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Local dataset paths — 走 DATA_DIR + 相对子目录
_DATA_ROOT = os.environ.get("DATA_DIR", str(Path(__file__).resolve().parent.parent / "data"))
MMVP_DATA_DIR = os.environ.get("SKILA_MMVP_DATA_DIR", str(Path(_DATA_ROOT) / "MMVP"))
MMVP_IMAGE_DIR = os.environ.get("SKILA_MMVP_IMAGE_DIR", str(Path(_DATA_ROOT) / "MMVP" / "MMVP Images"))
MMVP_CSV = os.environ.get("SKILA_MMVP_CSV", str(Path(_DATA_ROOT) / "MMVP" / "Questions.csv"))

VSTAR_DATA_DIR = os.environ.get("SKILA_VSTAR_DATA_DIR", str(Path(_DATA_ROOT) / "vstar"))
VSTAR_JSONL = os.environ.get("SKILA_VSTAR_JSONL", str(Path(_DATA_ROOT) / "vstar" / "test_questions.jsonl"))

MMSTAR_DATA_DIR = os.environ.get("SKILA_MMSTAR_DATA_DIR", str(Path(_DATA_ROOT) / "MMStar"))
MMSTAR_METADATA = os.environ.get("SKILA_MMSTAR_METADATA", str(Path(_DATA_ROOT) / "MMStar" / "metadata.json"))

# Output directory — 默认 <SkiLa>/test_results
OUTPUT_DIR = os.environ.get("SKILA_OUTPUT_DIR", str(Path(__file__).resolve().parent / "test_results"))

os.makedirs(OUTPUT_DIR, exist_ok=True)


def extract_answer(response: str) -> str:
    """Extract the answer letter from model response."""
    if not response:
        return ""

    # First try to find boxed answer
    boxed_match = re.search(r'\\boxed\{([^}]+)\}', response)
    if boxed_match:
        answer = boxed_match.group(1).strip()
        if len(answer) > 1:
            answer = answer[0]
        return answer.upper()

    # Look for <answer> tag first (contains final answer)
    if '<answer>' in response:
        given_answer = response.split('<answer>')[-1]
        given_answer = given_answer.split('</answer')[0].strip()
        if given_answer:
            if " " in given_answer:
                given_answer = given_answer.split(" ")[0]
            if len(given_answer) > 1:
                given_answer = given_answer[0]
            return given_answer.upper()

    # Look for answer in parentheses pattern like "(D)" or "(C) silver" anywhere in response
    paren_match = re.search(r'\(([A-Z])\)', response)
    if paren_match:
        return paren_match.group(1)

    # Fallback: look for answer at the end
    lines = [l.strip() for l in response.split('\n') if l.strip()]
    given_answer = lines[-1] if lines else ""

    if " " in given_answer:
        given_answer = given_answer.split(" ")[0]
    if len(given_answer) > 1:
        given_answer = given_answer[0]

    return given_answer.upper() if given_answer else ""


def accuracy_reward(response: str, ground_truth: str) -> float:
    """Check if the response matches the ground truth answer."""
    given_answer = extract_answer(response)
    ground_truth = ground_truth.upper().strip()
    if ground_truth.startswith('('):
        ground_truth = ground_truth[1]
    return 1.0 if given_answer == ground_truth else 0.0


def get_task_instruction(bench_name):
    """Get task instruction for different benchmarks."""
    return "\nAnswer with the option's letter from the given choices directly."


def get_passed_ids_from_summary(summary_path, data):
    """Load passed_ids from summary.json."""
    if not os.path.exists(summary_path):
        print(f"Warning: Summary file not found: {summary_path}")
        return None
    with open(summary_path, 'r') as f:
        summary = json.load(f)
    # Try to get passed_ids from mmvp_results first, then vstar
    return summary[data]["passed_ids"]


def load_model():
    """Load the SkiLa model with sketch extractor."""
    print(f"Loading SkiLa model from {MODEL_PATH}...")

    # Load config
    config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)

    # Load SkiLa model
    model = SkiLa.from_pretrained(
        MODEL_PATH,
        config=config,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    # Load sketch extractor (Siglip-based)
    print(f"Loading sketch extractor: {SKETCH_ENCODER}")
    from src.model.sketch_extractor import SketchExtractor_Siglip
    sketch_config = AutoConfig.from_pretrained(SKETCH_ENCODER)
    sketch_processor = SiglipImageProcessor.from_pretrained(SKETCH_ENCODER)

    sketch_extractor = SketchExtractor_Siglip(
        SKETCH_ENCODER,
        sketch_token_num=SKETCH_TOKEN_NUM,
        llm_hidden_dim=config.hidden_size,
        config=sketch_config,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    model.sketch_extractor = sketch_extractor

    # Load processor
    processor = AutoProcessor.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
    )

    # Add special tokens for sketch mode
    sketch_tokens = ["<|skila|>", "<|sketch_start|>", "<|sketch_end|>"]
    processor.tokenizer.add_tokens(sketch_tokens, special_tokens=False)

    # Set sketch token IDs in model config
    skila_id = processor.tokenizer.convert_tokens_to_ids("<|skila|>")
    sketch_start_id = processor.tokenizer.convert_tokens_to_ids("<|sketch_start|>")
    sketch_end_id = processor.tokenizer.convert_tokens_to_ids("<|sketch_end|>")

    model.config.skila_id = skila_id
    model.config.sketch_start_id = sketch_start_id
    model.config.sketch_end_id = sketch_end_id
    model.config.sketch_token_num = SKETCH_TOKEN_NUM
    model.config.compress_strategy = "average"

    print(f"skila_id: {skila_id}, sketch_start_id: {sketch_start_id}, sketch_end_id: {sketch_end_id}")
    print(f"sketch_token_num: {SKETCH_TOKEN_NUM}")
    print(f"Model loaded successfully!")

    model.eval()

    return model, processor, sketch_processor


def create_messages(img_path, question):
    """Create messages format for the model."""
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img_path},
                {"type": "text", "text": question},
            ],
        }
    ]


@torch.no_grad()
def run_inference(model, processor, img_path, text, max_new_tokens=512, question_id=None):
    """
    Run inference using standard model.generate().
    Force the first token to be <sketch_start> to trigger sketch mode.
    """
    messages = create_messages(img_path, text)
    text_formatted = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text_formatted],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    # Move inputs to device
    inputs = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

    # Move vision encoder to GPU
    if hasattr(model, 'visual') and next(model.visual.parameters(), None) is not None:
        if next(model.visual.parameters()).device.type != 'cuda':
            model.visual = model.visual.to(DEVICE)

    # Force first generated token to be <think> to trigger sketch mode
    # Get the token ID for <think>
    think_start_id = processor.tokenizer.encode("<think>", add_special_tokens=False)[0]
    input_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask")

    # Append <think> token to force explicit thinking mode
    think_start_tensor = torch.tensor([[think_start_id]], device=input_ids.device)
    input_ids = torch.cat([input_ids, think_start_tensor], dim=-1)
    attention_mask = torch.cat([attention_mask, torch.ones_like(think_start_tensor)], dim=-1)

    # Set question ID for replacement mode if applicable
    if question_id is not None and hasattr(model, 'set_current_question_id'):
        model.set_current_question_id(question_id)

    # Enable latent vector storage (for saving, not replacement mode)
    store_mode = hasattr(model, '_use_replacement_latent') and model._use_replacement_latent
    if not store_mode:
        model.store_latent_vectors = True
        model._latent_vectors_list = []

    # Use standard generate
    generated_ids = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        pixel_values=inputs.get("pixel_values"),
        image_grid_thw=inputs.get("image_grid_thw"),
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=processor.tokenizer.pad_token_id,
        eos_token_id=processor.tokenizer.eos_token_id,
    )

    # Retrieve and clear latent vectors (only for save mode)
    latent_vectors = []
    if not store_mode:
        latent_vectors = model._latent_vectors_list if hasattr(model, '_latent_vectors_list') else []
        model.store_latent_vectors = False
        model._latent_vectors_list = []

    # Decode response - skip the forced sketch_start token
    original_len = inputs["input_ids"].shape[1]
    trimmed_ids = generated_ids[0][original_len:]
    output_text = processor.tokenizer.decode(trimmed_ids, skip_special_tokens=True)

    return output_text, latent_vectors


def load_mmvp_dataset():
    """Load MMVP dataset from local CSV and images."""
    data = []
    with open(MMVP_CSV, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            idx = int(row["Index"])
            text = row["Question"] + '\nOptions:\n' + row["Options"]
            label = row["Correct Answer"].strip().upper()
            if label.startswith('('):
                label = label[1]
            if label not in ['A', 'B']:
                label = label[0] if label else ""

            item = {
                "question_id": idx,
                "image": f"{idx}.jpg",
                "query": text,
                "label": label
            }
            data.append(item)
    return data


def load_vstar_dataset():
    """Load VSTAR dataset from local JSONL file."""
    data = []
    with open(VSTAR_JSONL, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line.strip())
            label = item["label"].upper().strip()
            if label.startswith('('):
                label = label[1]
            item["label"] = label
            data.append(item)
    return data


def load_mmstar_dataset():
    """Load MMStar dataset from local metadata JSON."""
    with open(MMSTAR_METADATA, 'r', encoding='utf-8') as f:
        records = json.load(f)
    data = []
    for item in records:
        label = item["answer"].upper().strip()
        if label.startswith('('):
            label = label[1]
        data.append({
            "question_id": item["index"],
            "image": item["image_path"],
            "query": item["question"],
            "label": label,
            "category": item.get("category", ""),
            "l2_category": item.get("l2_category", "")
        })
    return data


def evaluate_mmvp(model, processor, dataset, image_dir, out_dir, max_new_tokens=512,
                  replace_images=False, adv_images_dir=None, filter_ids=None, num_samples=None,
                  save_latent_vectors=False):
    """Evaluate on MMVP dataset."""
    print(f"\nEvaluating MMVP")
    if replace_images:
        print(f"  Using adversarial images from: {adv_images_dir}")
    if filter_ids is not None:
        print(f"  Filtering to {len(filter_ids)} passed IDs from summary")
    if num_samples is not None:
        print(f"  Limiting to first {num_samples} samples")
    os.makedirs(out_dir, exist_ok=True)
    task_instruction = get_task_instruction("mmvp")

    out_file = os.path.join(out_dir, "mmvp_results.json")
    latent_npz_file = os.path.join(out_dir, "mmvp_latent.npz")
    total, correct = 0, 0
    results = []
    passed_ids = []
    evaluated_count = 0
    latent_dict = {}  # {question_id: [vectors]}

    for dat in tqdm(dataset, desc="MMVP"):
        # Filter by passed_ids if specified
        if filter_ids is not None and dat['question_id'] not in filter_ids:
            continue

        # Limit number of samples if specified
        if num_samples is not None and evaluated_count >= num_samples:
            break

        evaluated_count += 1

        # Use adversarial image if replace_images is enabled
        if replace_images and adv_images_dir:
            adv_img_path = os.path.join(f"{adv_images_dir}/mmvp", f"{dat['question_id']}_adv.png")
            if os.path.exists(adv_img_path):
                img_path = adv_img_path
            else:
                img_path = os.path.join(image_dir, dat['image'])
        else:
            img_path = os.path.join(image_dir, dat['image'])

        text = dat['query'].replace('(a)', 'A.').replace('(b)', 'B.')
        text = text + task_instruction

        try:
            qid = dat['question_id']
            prediction, latent_vectors = run_inference(
                model, processor, img_path, text,
                max_new_tokens=max_new_tokens, question_id=qid
            )
            latent_dict[qid] = []
            for lv in latent_vectors:
                lv_tensor = lv.squeeze(0) if lv.dim() > 1 else lv
                latent_dict[qid].append(lv_tensor.float().numpy())
        except Exception as e:
            print(f"\nError on sample {dat['question_id']}: {e}")
            import traceback
            traceback.print_exc()
            prediction = ""
            latent_dict[dat['question_id']] = []

        is_correct = accuracy_reward(prediction, dat['label'])
        if is_correct:
            passed_ids.append(dat['question_id'])

        res = {
            'id': dat['question_id'],
            'prediction': prediction,
            'label': dat['label'],
            'correct': bool(is_correct)
        }
        results.append(res)

        if is_correct:
            correct += 1
        total += 1

    accuracy = correct / total if total > 0 else 0
    print(f"MMVP Accuracy: {correct}/{total} = {accuracy*100:.2f}%")

    # Save latent vectors to .npz file if requested
    if save_latent_vectors and latent_dict:
        import numpy as np
        # Convert keys to strings as np.savez requires string keywords
        latent_dict_str = {str(k): v for k, v in latent_dict.items()}
        np.savez(latent_npz_file, **latent_dict_str)
        total_vectors = sum(len(v) for v in latent_dict.values())
        print(f"Saved latent vectors for {len(latent_dict)} questions ({total_vectors} total vectors) to {latent_npz_file}")
    elif latent_dict:
        total_vectors = sum(len(v) for v in latent_dict.values())
        print(f"Collected latent vectors for {len(latent_dict)} questions ({total_vectors} total, not saved)")
    else:
        print("No latent vectors collected for MMVP")

    step_result = {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "passed_ids": passed_ids,
        "results": results,
        "mean_l2_length": 0.0
    }
    json.dump(results, open(out_file, 'w+'), indent=2)

    # Clear GPU cache after MMVP evaluation
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        print(f"GPU cache cleared. Free memory: {torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")

    return step_result


def evaluate_vstar(model, processor, dataset, image_dir, out_dir, max_new_tokens=512,
                   replace_images=False, adv_images_dir=None, filter_ids=None, num_samples=None,
                   save_latent_vectors=False):
    """Evaluate on VSTAR dataset."""
    print(f"\nEvaluating VSTAR")
    if replace_images:
        print(f"  Using adversarial images from: {adv_images_dir}")
    if filter_ids is not None:
        print(f"  Filtering to {len(filter_ids)} passed IDs from summary")
    if num_samples is not None:
        print(f"  Limiting to first {num_samples} samples")
    os.makedirs(out_dir, exist_ok=True)
    task_instruction = get_task_instruction("vstar")

    out_file = os.path.join(out_dir, "vstar_results.json")
    latent_npz_file = os.path.join(out_dir, "vstar_latent.npz")
    total, correct = 0, 0
    results = []
    passed_ids = []
    evaluated_count = 0
    latent_dict = {}  # {question_id: [vectors]}

    for dat in tqdm(dataset, desc="VSTAR"):
        # Filter by passed_ids if specified
        if filter_ids is not None and int(dat['question_id']) not in filter_ids:
            continue

        # Limit number of samples if specified
        if num_samples is not None and evaluated_count >= num_samples:
            break

        evaluated_count += 1

        # Use adversarial image if replace_images is enabled
        if replace_images and adv_images_dir:
            adv_img_path = os.path.join(f"{adv_images_dir}/vstar", f"{dat['question_id']}_adv.png")
            if os.path.exists(adv_img_path):
                img_path = adv_img_path
            else:
                img_path = os.path.join(image_dir, dat['image'])
        else:
            img_path = os.path.join(image_dir, dat['image'])

        text = dat['text']

        try:
            qid = dat['question_id']
            prediction, latent_vectors = run_inference(
                model, processor, img_path, text,
                max_new_tokens=max_new_tokens, question_id=qid
            )
            latent_dict[qid] = []
            for lv in latent_vectors:
                lv_tensor = lv.squeeze(0) if lv.dim() > 1 else lv
                latent_dict[qid].append(lv_tensor.float().numpy())
        except Exception as e:
            print(f"\nError on sample {dat['question_id']}: {e}")
            import traceback
            traceback.print_exc()
            prediction = ""
            latent_dict[dat['question_id']] = []

        is_correct = accuracy_reward(prediction, dat['label'])
        if is_correct:
            passed_ids.append(dat['question_id'])

        res = {
            'id': dat['question_id'],
            'prediction': prediction,
            'label': dat['label'],
            'correct': bool(is_correct),
            'category': dat.get('category', 'unknown')
        }
        results.append(res)

        if is_correct:
            correct += 1
        total += 1

    accuracy = correct / total if total > 0 else 0
    print(f"VSTAR Accuracy: {correct}/{total} = {accuracy*100:.2f}%")

    # Save latent vectors to .npz file if requested
    if save_latent_vectors and latent_dict:
        import numpy as np
        # Convert keys to strings as np.savez requires string keywords
        latent_dict_str = {str(k): v for k, v in latent_dict.items()}
        np.savez(latent_npz_file, **latent_dict_str)
        total_vectors = sum(len(v) for v in latent_dict.values())
        print(f"Saved latent vectors for {len(latent_dict)} questions ({total_vectors} total vectors) to {latent_npz_file}")
    elif latent_dict:
        total_vectors = sum(len(v) for v in latent_dict.values())
        print(f"Collected latent vectors for {len(latent_dict)} questions ({total_vectors} total, not saved)")
    else:
        print("No latent vectors collected for VSTAR")

    step_result = {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "passed_ids": passed_ids,
        "results": results,
        "mean_l2_length": 0.0
    }
    json.dump(results, open(out_file, 'w+'), indent=2)

    return step_result


def evaluate_mmstar(model, processor, dataset, image_dir, out_dir, max_new_tokens=512,
                    replace_images=False, adv_images_dir=None, filter_ids=None, num_samples=None,
                    save_latent_vectors=False):
    """Evaluate on MMStar dataset."""
    print(f"\nEvaluating MMStar")
    if replace_images:
        print(f"  Using adversarial images from: {adv_images_dir}")
    if filter_ids is not None:
        print(f"  Filtering to {len(filter_ids)} passed IDs from summary")
    if num_samples is not None:
        print(f"  Limiting to first {num_samples} samples")
    os.makedirs(out_dir, exist_ok=True)
    task_instruction = get_task_instruction("mmstar")

    out_file = os.path.join(out_dir, "mmstar_results.json")
    latent_npz_file = os.path.join(out_dir, "mmstar_latent.npz")
    total, correct = 0, 0
    results = []
    passed_ids = []
    evaluated_count = 0
    latent_dict = {}  # {question_id: [vectors]}

    for dat in tqdm(dataset, desc="MMStar"):
        # Filter by passed_ids if specified
        if filter_ids is not None and dat['question_id'] not in filter_ids:
            continue

        # Limit number of samples if specified
        if num_samples is not None and evaluated_count >= num_samples:
            break

        evaluated_count += 1

        # Use adversarial image if replace_images is enabled
        if replace_images and adv_images_dir:
            adv_img_path = os.path.join(f"{adv_images_dir}/mmstar", f"{dat['question_id']}_adv.png")
            if os.path.exists(adv_img_path):
                img_path = adv_img_path
            else:
                img_path = os.path.join(image_dir, dat['image'])
        else:
            img_path = os.path.join(image_dir, dat['image'])

        text = dat['query'] + task_instruction

        try:
            qid = dat['question_id']
            prediction, latent_vectors = run_inference(
                model, processor, img_path, text,
                max_new_tokens=max_new_tokens, question_id=qid
            )
            latent_dict[qid] = []
            for lv in latent_vectors:
                lv_tensor = lv.squeeze(0) if lv.dim() > 1 else lv
                latent_dict[qid].append(lv_tensor.float().numpy())
        except Exception as e:
            print(f"\nError on sample {dat['question_id']}: {e}")
            import traceback
            traceback.print_exc()
            prediction = ""
            latent_dict[dat['question_id']] = []

        is_correct = accuracy_reward(prediction, dat['label'])
        if is_correct:
            passed_ids.append(dat['question_id'])

        res = {
            'id': dat['question_id'],
            'prediction': prediction,
            'label': dat['label'],
            'correct': bool(is_correct),
            'category': dat.get('category', ''),
            'l2_category': dat.get('l2_category', '')
        }
        results.append(res)

        if is_correct:
            correct += 1
        total += 1

    accuracy = correct / total if total > 0 else 0
    print(f"MMStar Accuracy: {correct}/{total} = {accuracy*100:.2f}%")

    # Save latent vectors to .npz file if requested
    if save_latent_vectors and latent_dict:
        import numpy as np
        # Convert keys to strings as np.savez requires string keywords
        latent_dict_str = {str(k): v for k, v in latent_dict.items()}
        np.savez(latent_npz_file, **latent_dict_str)
        total_vectors = sum(len(v) for v in latent_dict.values())
        print(f"Saved latent vectors for {len(latent_dict)} questions ({total_vectors} total vectors) to {latent_npz_file}")
    elif latent_dict:
        total_vectors = sum(len(v) for v in latent_dict.values())
        print(f"Collected latent vectors for {len(latent_dict)} questions ({total_vectors} total, not saved)")
    else:
        print("No latent vectors collected for MMStar")

    step_result = {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "passed_ids": passed_ids,
        "results": results,
        "mean_l2_length": 0.0
    }
    json.dump(results, open(out_file, 'w+'), indent=2)

    return step_result


def collect_examples(results, n=5):
    """Collect some correct and incorrect examples."""
    correct_examples = []
    incorrect_examples = []

    for res in results:
        if res['correct'] and len(correct_examples) < n:
            correct_examples.append(res)
        elif not res['correct'] and len(incorrect_examples) < n:
            incorrect_examples.append(res)

        if len(correct_examples) >= n and len(incorrect_examples) >= n:
            break

    return {
        "correct_examples": correct_examples,
        "incorrect_examples": incorrect_examples
    }


def save_summary(mmvp_result, vstar_result, mmstar_result, output_path):
    """Save summary of evaluation results."""
    summary = {
        "model_path": MODEL_PATH,
        "sketch_encoder": SKETCH_ENCODER,
        "sketch_token_num": SKETCH_TOKEN_NUM,
    }

    if mmvp_result is not None:
        summary["mmvp_result"] = {
            "accuracy": mmvp_result["accuracy"],
            "correct": mmvp_result["correct"],
            "total": mmvp_result["total"],
            "passed_ids": mmvp_result["passed_ids"],
            "mean_l2_length": mmvp_result.get("mean_l2_length", 0.0),
            "examples": collect_examples(mmvp_result["results"])
        }

    if vstar_result is not None:
        summary["vstar_result"] = {
            "accuracy": vstar_result["accuracy"],
            "correct": vstar_result["correct"],
            "total": vstar_result["total"],
            "passed_ids": vstar_result["passed_ids"],
            "mean_l2_length": vstar_result.get("mean_l2_length", 0.0),
            "examples": collect_examples(vstar_result["results"])
        }

    if mmstar_result is not None:
        summary["mmstar_result"] = {
            "accuracy": mmstar_result["accuracy"],
            "correct": mmstar_result["correct"],
            "total": mmstar_result["total"],
            "passed_ids": mmstar_result["passed_ids"],
            "mean_l2_length": mmstar_result.get("mean_l2_length", 0.0),
            "examples": collect_examples(mmstar_result["results"])
        }

    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nSummary saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="SkiLa-7B Evaluation")
    parser.add_argument("--dataset", type=str, default="all",
                        choices=["all", "mmvp", "vstar", "mmstar"],
                        help="Which dataset to evaluate: 'all', 'mmvp', 'vstar', or 'mmstar' (default: all)")
    parser.add_argument("--max_new_tokens", type=int, default=1024,
                        help="Maximum number of tokens to generate (default: 512)")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Model path (default: JosephTong/SkiLa-7B)")
    parser.add_argument("--sketch_token_num", type=int, default=27,
                        help="Number of sketch tokens (default: 27)")
    parser.add_argument("--adv_images_dir", type=str, default=None,
                        help="Directory containing adversarial images to replace original images")
    parser.add_argument("--use_passed_ids", action="store_true",
                        help="Only test IDs that are in the passed_ids from summary.json")
    parser.add_argument("--num_samples", type=int, default=None,
                        help="Number of samples to evaluate (default: all)")
    parser.add_argument("--summary_path", type=str, default="test_results/org/summary.json",
                        help="Path to summary.json for loading passed_ids (default: test_results/skila_evaluation/summary.json)")
    parser.add_argument("--save_latent_vectors", action="store_true",
                        help="Save latent vectors to .npz files (default: disabled)")
    parser.add_argument("--load_latent_vectors", type=str, default=None,
                        help="Path to .npz file with latent vectors to replace during inference")
    args = parser.parse_args()

    # Examples:
    #   python test.py --dataset mmvp
    #   python test.py --dataset mmvp --num_samples 20
    #   python test.py --dataset vstar --use_passed_ids --adv_images_dir adv_images/vstar --load_latent_vectors 
    #   python test.py --dataset mmstar --use_passed_ids --load_latent_vectors test_results/clean/mmstar/mmstar_latent.npz --adv_images_dir adv_images
    ''' python test.py --dataset mmvp --use_passed_ids --load_latent_vectors test_results/adv/mmvp/mmvp_latent.npz ; python test.py --dataset mmvp --use_passed_ids --load_latent_vectors test_results/clean/mmvp/mmvp_latent.npz --adv_images_dir adv_images ; python test.py --dataset vstar --use_passed_ids --load_latent_vectors test_results/adv/vstar/vstar_latent.npz ; python test.py --dataset vstar --use_passed_ids --load_latent_vectors test_results/clean/vstar/vstar_latent.npz --adv_images_dir adv_images
    '''

    global MODEL_PATH, SKETCH_TOKEN_NUM
    if args.model_path:
        MODEL_PATH = args.model_path
    SKETCH_TOKEN_NUM = args.sketch_token_num

    # Determine summary path for loading passed_ids
    default_summary_path = os.path.join(OUTPUT_DIR, "skila_evaluation", "summary.json")
    summary_file = args.summary_path if args.summary_path else default_summary_path

    print("=" * 64)
    print("SkiLa-7B Evaluation on Local Datasets")
    print(f"Model: {MODEL_PATH}")
    print(f"Sketch encoder: {SKETCH_ENCODER}")
    print(f"Sketch token num: {SKETCH_TOKEN_NUM}")
    print(f"Mode: {args.dataset.upper()}")
    print(f"Max new tokens: {args.max_new_tokens}")
    print(f"Adversarial images dir: {args.adv_images_dir}")
    print(f"Use passed IDs: {args.use_passed_ids}")
    print(f"Num samples: {args.num_samples}")
    print(f"Summary path: {summary_file}")
    print(f"Save latent vectors: {args.save_latent_vectors}")
    print("=" * 64)

    # Load passed_ids if requested
    filter_ids = None
    if args.use_passed_ids:
        filter_ids = get_passed_ids_from_summary(summary_file, f"{args.dataset}_result")
        if filter_ids:
            print(f"Loaded {len(filter_ids)} passed IDs for filtering")
            # If num_samples specified, take only first N from passed_ids
            if args.num_samples is not None and args.num_samples < len(filter_ids):
                filter_ids = filter_ids[:args.num_samples]
                print(f"Limited to first {len(filter_ids)} passed IDs")
        else:
            print("Warning: No passed_ids found in summary.json")

    # Load model
    model, processor, sketch_processor = load_model()

    # Load latent vectors for replacement if specified
    if args.load_latent_vectors:
        import numpy as np
        print(f"Loading latent vectors from {args.load_latent_vectors}...")
        latent_data = np.load(args.load_latent_vectors)
        latent_dict = {int(k): v for k, v in latent_data.items()}
        print(f"Loaded latent vectors for {len(latent_dict)} questions")
        model.set_replacement_latent_vectors(latent_dict)

    # Create output directory
    run_output_dir = os.path.join(OUTPUT_DIR, f"skila_evaluation")
    os.makedirs(run_output_dir, exist_ok=True)

    mmvp_result = None
    vstar_result = None
    mmstar_result = None

    if args.dataset in ["all", "mmvp"]:
        # Load MMVP dataset
        print("\nLoading MMVP dataset...")
        mmvp_dataset = load_mmvp_dataset()
        print(f"MMVP: {len(mmvp_dataset)} samples")

        # Evaluate MMVP
        mmvp_result = evaluate_mmvp(
            model, processor, mmvp_dataset,
            MMVP_IMAGE_DIR, run_output_dir,
            max_new_tokens=args.max_new_tokens,
            replace_images=args.adv_images_dir is not None,
            adv_images_dir=args.adv_images_dir,
            filter_ids=filter_ids,
            num_samples=args.num_samples,
            save_latent_vectors=args.save_latent_vectors
        )

        # Clear GPU cache after MMVP to free memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            gc.collect()
            print(f"GPU cache cleared after MMVP. Free: {torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")

    if args.dataset in ["all", "vstar"]:
        # Load VSTAR dataset
        print("\nLoading VSTAR dataset...")
        vstar_dataset = load_vstar_dataset()
        print(f"VSTAR: {len(vstar_dataset)} samples")

        # Clear GPU cache before VSTAR
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            print(f"GPU memory before VSTAR: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB allocated, "
                  f"{torch.cuda.memory_reserved(0) / 1024**3:.2f} GB reserved")

        # Evaluate VSTAR
        vstar_result = evaluate_vstar(
            model, processor, vstar_dataset,
            VSTAR_DATA_DIR, run_output_dir,
            max_new_tokens=args.max_new_tokens,
            replace_images=args.adv_images_dir is not None,
            adv_images_dir=args.adv_images_dir,
            filter_ids=filter_ids,
            num_samples=args.num_samples,
            save_latent_vectors=args.save_latent_vectors
        )

    if args.dataset in ["all", "mmstar"]:
        # Load MMStar dataset
        print("\nLoading MMStar dataset...")
        mmstar_dataset = load_mmstar_dataset()
        print(f"MMStar: {len(mmstar_dataset)} samples")

        # Clear GPU cache before MMStar
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            print(f"GPU memory before MMStar: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB allocated, "
                  f"{torch.cuda.memory_reserved(0) / 1024**3:.2f} GB reserved")

        # Evaluate MMStar
        mmstar_result = evaluate_mmstar(
            model, processor, mmstar_dataset,
            MMSTAR_DATA_DIR, run_output_dir,
            max_new_tokens=args.max_new_tokens,
            replace_images=args.adv_images_dir is not None,
            adv_images_dir=args.adv_images_dir,
            filter_ids=filter_ids,
            num_samples=args.num_samples,
            save_latent_vectors=args.save_latent_vectors
        )

    # Save summary
    summary_path = os.path.join(run_output_dir, "summary.json")
    save_summary(mmvp_result, vstar_result, mmstar_result, summary_path)

    # Print final summary
    print("\n" + "=" * 64)
    print("Final Results Summary")
    print("=" * 64)
    if mmvp_result is not None:
        print(f"MMVP: {mmvp_result['accuracy']*100:.2f}% ({mmvp_result['correct']}/{mmvp_result['total']}), mean L2: {mmvp_result.get('mean_l2_length', 0):.6f}")
    if vstar_result is not None:
        print(f"VSTAR: {vstar_result['accuracy']*100:.2f}% ({vstar_result['correct']}/{vstar_result['total']}), mean L2: {vstar_result.get('mean_l2_length', 0):.6f}")
    if mmstar_result is not None:
        print(f"MMStar: {mmstar_result['accuracy']*100:.2f}% ({mmstar_result['correct']}/{mmstar_result['total']}), mean L2: {mmstar_result.get('mean_l2_length', 0):.6f}")
    print("\nDone!")


if __name__ == "__main__":
    main()
