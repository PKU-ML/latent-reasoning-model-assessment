"""
Test script for Monet-7B model on local MMVP and VSTAR datasets.
Uses implicit latent visual reasoning via MonetModel class (no vLLM required).

This version implements a custom generation loop similar to SkiLa's approach,
where the model's own hidden states are fed back as latent representations
to enable implicit visual reasoning.
"""

import sys
import os
from pathlib import Path
import argparse


import torch
import json
import csv
import re
import gc
import numpy as np
from tqdm import tqdm

# Add Monet src to path for model class (local Monet directory)
monet_src_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
sys.path.insert(0, monet_src_path)

# Apply monkey patch for latent token support BEFORE importing model
from src.train.monkey_patch_forward_monet_test import replace_qwen2_5_with_monet_forward
replace_qwen2_5_with_monet_forward()

from transformers import AutoProcessor, AutoConfig
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

# Import MonetModel after patch is applied
from src.model.monet import MonetModel

# Model configuration — 通过环境变量 MONET_MODEL_PATH 注入；不设置时使用 HuggingFace 仓库名
MODEL_PATH = os.environ.get("MONET_MODEL_PATH", "NOVAglow646/Monet-7B")

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Local dataset paths — 走 DATA_DIR + 相对子目录
_DATA_ROOT = os.environ.get("DATA_DIR", str(Path(__file__).resolve().parent.parent / "data"))
MMVP_DATA_DIR = os.environ.get("MONET_MMVP_DATA_DIR", str(Path(_DATA_ROOT) / "MMVP"))
MMVP_IMAGE_DIR = os.environ.get("MONET_MMVP_IMAGE_DIR", str(Path(_DATA_ROOT) / "MMVP" / "MMVP Images"))
MMVP_CSV = os.environ.get("MONET_MMVP_CSV", str(Path(_DATA_ROOT) / "MMVP" / "Questions.csv"))

VSTAR_DATA_DIR = os.environ.get("MONET_VSTAR_DATA_DIR", str(Path(_DATA_ROOT) / "vstar"))
VSTAR_JSONL = os.environ.get("MONET_VSTAR_JSONL", str(Path(_DATA_ROOT) / "vstar" / "test_questions.jsonl"))

MMSTAR_DATA_DIR = os.environ.get("MONET_MMSTAR_DATA_DIR", str(Path(_DATA_ROOT) / "MMStar"))
MMSTAR_METADATA = os.environ.get("MONET_MMSTAR_METADATA", str(Path(_DATA_ROOT) / "MMStar" / "metadata.json"))

# Output directory — 默认 <Monet>/test_results
OUTPUT_DIR = os.environ.get("MONET_OUTPUT_DIR", str(Path(__file__).resolve().parent / "test_results"))

# Latent reasoning settings
MAX_LATENT_STEPS = int(os.environ.get("MAX_LATENT_STEPS", "20"))

# Default summary path for loading passed_ids
DEFAULT_SUMMARY_PATH = os.path.join(OUTPUT_DIR, "summary.json")

os.makedirs(OUTPUT_DIR, exist_ok=True)


def replace_abs_vis_token_content(s: str) -> str:
    """Replace latent token content with readable placeholder."""
    pattern = re.compile(r'(<abs_vis_token>)(.*?)(</abs_vis_token>)', flags=re.DOTALL)
    return pattern.sub(r'\1<latent>\3', s)


def extract_answer(response: str) -> str:
    """Extract the answer letter from model response."""
    if not response:
        return ""

    response = response.strip()

    # First try to find boxed answer
    boxed_match = re.search(r'\\boxed\{([^}]+)\}', response)
    if boxed_match:
        answer = boxed_match.group(1).strip()
        if len(answer) > 1:
            answer = answer[0]
        return answer.upper()

    # Try to find pattern like "(A)" or "(B)" anywhere - take the LAST one
    paren_matches = re.findall(r'\(([A-Z])\)', response)
    if paren_matches:
        return paren_matches[-1]

    # Try to find standalone single letter A/B/C/D at start of first non-empty line
    lines = [l.strip() for l in response.split('\n') if l.strip()]
    if lines:
        first_line = lines[0]
        # Check if first line is just a single letter A/B/C/D (not part of a word)
        if len(first_line) == 1 and first_line in 'ABCD':
            return first_line
        # Check if first line is like "A." or "B." at very start
        option_match = re.match(r'^([A-Z])\.', first_line)
        if option_match:
            return option_match.group(1)

    # Look for known answer phrases
    answer_phrases = [
        'the correct answer is', 'the answer is', 'answer:',
        'so the answer is', 'therefore the answer'
    ]
    for phrase in answer_phrases:
        idx = response.lower().find(phrase)
        if idx != -1:
            # Get text after phrase
            after = response[idx + len(phrase):].strip()
            # Try to find option in parentheses
            m = re.match(r'\(([A-Z])\)', after)
            if m:
                return m.group(1)
            # Try to find letter at start
            m = re.match(r'([A-Z])\.?', after)
            if m:
                return m.group(1)

    # Look for standalone single letter A/B/C/D anywhere in response (for "C" or "D" outputs)
    standalone_match = re.search(r'\b([A-D])\b', response)
    if standalone_match:
        return standalone_match.group(1)

    return ""


def accuracy_reward(response: str, ground_truth: str) -> float:
    """Check if the response matches the ground truth answer."""
    given_answer = extract_answer(response)
    ground_truth = ground_truth.upper().strip()
    if ground_truth.startswith('('):
        ground_truth = ground_truth[1]
    return 1.0 if given_answer == ground_truth else 0.0


def get_task_instruction(bench_name):
    """Get task instruction for different benchmarks."""
    return "\nAnswer with the option's letter from the given choices directly. Only output the letter (e.g., A, B, C, or D), do not include any explanation."


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


def load_model():
    """Load the Monet model and processor using MonetModel class."""
    print(f"Loading Monet model from {MODEL_PATH}...")

    config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)

    # Use MonetModel instead of standard Qwen2_5_VLForConditionalGeneration
    model = MonetModel.from_pretrained(
        MODEL_PATH,
        config=config,
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )
    model = model.to(DEVICE)

    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)

    # Add special tokens for latent reasoning
    processor.tokenizer.add_tokens("<abs_vis_token>", special_tokens=True)
    processor.tokenizer.add_tokens("</abs_vis_token>", special_tokens=True)

    # Set latent token IDs
    latent_start_idx = processor.tokenizer("<abs_vis_token>", return_tensors="pt")["input_ids"][0]
    model.config.latent_token_id = int(latent_start_idx[0]) if len(latent_start_idx) > 0 else 151666
    model.config.latent_start_id = int(latent_start_idx[0]) if len(latent_start_idx) > 0 else 151666
    model.config.max_latent_steps = MAX_LATENT_STEPS

    print(f"latent_token_id: {model.config.latent_start_id}")
    print(f"max_latent_steps: {model.config.max_latent_steps}")

    model.eval()
    # Enable gradient checkpointing to save memory during inference
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        print("Gradient checkpointing enabled")

    # Offload vision encoder to CPU to save GPU memory
    # Note: device_map="auto" already handles device placement, so we don't call .to(DEVICE)
    if hasattr(model, 'visual'):
        # Just set the flag, don't move the model again
        model.visual_requires_grad = False
        print("Vision encoder available for CPU offloading if needed")

    return model, processor


@torch.no_grad()
def run_inference(model, processor, img_path, text, max_new_tokens=512, latent_replace_vectors=None):
    """
    Run inference using MonetModel's custom _sample method for latent reasoning.

    This uses the custom generation loop implemented in MonetModel._sample()
    which handles latent mode switching automatically.
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
    inputs = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

    # Note: with device_map="auto", vision encoder is already on the right device
    input_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask")

    # Insert <abs_vis_token> after <|assistant|> token in the input
    # This triggers latent reasoning mode in the model
    # <|im_start|> = 151644, assistant = 77091, \n = 198
    im_start_id = 151644
    assistant_id = 77091
    latent_token_id = model.config.latent_token_id

    # Find position of <|im_start|>assistant sequence in input_ids
    input_ids_list = input_ids[0].tolist()

    # Insert latent token at the end of all tokens
    input_ids = input_ids[0].tolist()
    input_ids.append(latent_token_id)
    insert_pos = len(input_ids) - 1
    input_ids = torch.tensor([input_ids], dtype=torch.long, device=DEVICE)
    attention_mask = torch.tensor([[1] * len(input_ids[0])], dtype=torch.long, device=DEVICE)
    #print(f"[DEBUG] Inserted <abs_vis_token> at position {insert_pos} after assistant token")

    # Set latent replace vectors if provided
    if latent_replace_vectors is not None:
        model._latent_replace_vectors = latent_replace_vectors
        model._latent_replace_idx = 0

    # Use standard generate (now uses MonetModel's custom _sample)
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

    # Retrieve latent hidden states from MonetModel._sample
    latent_vectors = model._latent_hidden_state_list if hasattr(model, '_latent_hidden_state_list') else []
    model._latent_hidden_state_list = []

    # Clear replace vectors
    if latent_replace_vectors is not None:
        model._latent_replace_vectors = None
        model._latent_replace_idx = 0

    # Decode response
    trimmed_ids = generated_ids[0][len(input_ids[0]):]
    output_text = processor.tokenizer.decode(trimmed_ids, skip_special_tokens=True)
    cleaned_output = replace_abs_vis_token_content(output_text)

    return cleaned_output, latent_vectors


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


def get_passed_ids_from_summary(summary_path, key):
    """Load passed_ids from summary.json."""
    if not os.path.exists(summary_path):
        print(f"Warning: Summary file not found: {summary_path}")
        return None
    with open(summary_path, 'r') as f:
        summary = json.load(f)
    return summary[key]["passed_ids"]


def evaluate_mmvp(model, processor, dataset, image_dir, out_dir, max_new_tokens=512,
                  replace_images=False, adv_images_dir=None, filter_ids=None, num_samples=None,
                  save_latent_vectors=False, latent_replace_vectors=None):
    """Evaluate on MMVP dataset."""
    print(f"\nEvaluating MMVP (max_latent_steps={MAX_LATENT_STEPS})")
    if replace_images:
        print(f"  Using adversarial images from: {adv_images_dir}")
    if latent_replace_vectors is not None:
        print(f"  Using replacement latent vectors for {len(latent_replace_vectors)} questions")
    if filter_ids is not None:
        print(f"  Filtering to {len(filter_ids)} passed IDs from summary")
    if num_samples is not None:
        print(f"  Limiting to first {num_samples} samples")
    os.makedirs(out_dir, exist_ok=True)
    task_instruction = get_task_instruction("mmvp")

    out_file = os.path.join(out_dir, f"mmvp_latent{MAX_LATENT_STEPS:03d}.json")
    latent_npz_file = os.path.join(out_dir, f"mmvp_latent{MAX_LATENT_STEPS:03d}.npz")
    total, correct = 0, 0
    results = []
    passed_ids = []
    evaluated_count = 0
    latent_vectors_dict = {}  # {question_id: list of latent vectors}
    l2_lengths = []  # Collect L2 lengths of each latent vector

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
            adv_img_path = os.path.join(adv_images_dir, "mmvp", f"{dat['question_id']}_adv.png")
            if os.path.exists(adv_img_path):
                img_path = adv_img_path
            else:
                img_path = os.path.join(image_dir, dat['image'])
        else:
            img_path = os.path.join(image_dir, dat['image'])

        text = dat['query'].replace('(a)', 'A.').replace('(b)', 'B.')
        text = text + task_instruction

        # Get replacement vectors for this question if available
        replace_vec = None
        if latent_replace_vectors is not None and str(dat['question_id']) in latent_replace_vectors:
            replace_vec = latent_replace_vectors[str(dat['question_id'])]

        try:
            prediction, latent_vectors = run_inference(model, processor, img_path, text, max_new_tokens=max_new_tokens, latent_replace_vectors=replace_vec)
            # Store latent vectors by question_id
            if latent_vectors:
                latent_vectors_dict[dat['question_id']] = [lv.numpy() for lv in latent_vectors]
            for lv in latent_vectors:
                lv_tensor = lv.squeeze(0) if lv.dim() > 1 else lv
                l2_length = torch.sqrt(torch.sum(lv_tensor ** 2)).item()
                l2_lengths.append(l2_length)
        except Exception as e:
            print(f"\nError on sample {dat['question_id']}: {e}")
            import traceback
            traceback.print_exc()
            prediction = ""
            latent_vectors = []

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

    # Save latent vectors to .npz file
    if save_latent_vectors and latent_vectors_dict:
        # Save as dictionary: {question_id: array of latent vectors}
        latent_vectors_dict_str = {str(k): v for k, v in latent_vectors_dict.items()}
        np.savez(latent_npz_file, **latent_vectors_dict_str)
        # Compute mean L2 across all vectors
        all_l2 = [torch.sqrt(torch.sum(torch.tensor(lv) ** 2)).item() for lv_dict in latent_vectors_dict.values() for lv in lv_dict]
        mean_l2 = np.mean(all_l2) if all_l2 else 0.0
        total_vectors = sum(len(lv_list) for lv_list in latent_vectors_dict.values())
        print(f"Saved {total_vectors} latent vectors for {len(latent_vectors_dict)} questions to {latent_npz_file}")
        print(f"Mean L2 length: {mean_l2:.6f}")
    else:
        mean_l2 = 0.0
        if save_latent_vectors:
            print("No latent vectors collected for MMVP")

    accuracy = correct / total if total > 0 else 0
    print(f"MMVP Accuracy: {correct}/{total} = {accuracy*100:.2f}%")

    step_result = {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "passed_ids": passed_ids,
        "results": results,
        "mean_l2_length": mean_l2
    }
    json.dump(results, open(out_file, 'w+'), indent=2)

    # Clear GPU cache after MMVP evaluation to free memory for VSTAR
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        print(f"GPU cache cleared. Free memory: {torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")

    return step_result


def evaluate_vstar(model, processor, dataset, image_dir, out_dir, max_new_tokens=512,
                  replace_images=False, adv_images_dir=None, filter_ids=None, num_samples=None,
                  save_latent_vectors=False, latent_replace_vectors=None):
    """Evaluate on VSTAR dataset."""
    print(f"\nEvaluating VSTAR (max_latent_steps={MAX_LATENT_STEPS})")
    if replace_images:
        print(f"  Using adversarial images from: {adv_images_dir}")
    if latent_replace_vectors is not None:
        print(f"  Using replacement latent vectors for {len(latent_replace_vectors)} questions")
    if filter_ids is not None:
        print(f"  Filtering to {len(filter_ids)} passed IDs from summary")
    if num_samples is not None:
        print(f"  Limiting to first {num_samples} samples")
    os.makedirs(out_dir, exist_ok=True)
    task_instruction = get_task_instruction("vstar")

    out_file = os.path.join(out_dir, f"vstar_latent{MAX_LATENT_STEPS:03d}.json")
    latent_npz_file = os.path.join(out_dir, f"vstar_latent{MAX_LATENT_STEPS:03d}.npz")
    total, correct = 0, 0
    results = []
    passed_ids = []
    evaluated_count = 0
    latent_vectors_dict = {}  # {question_id: list of latent vectors}
    l2_lengths = []

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
            adv_img_path = os.path.join(adv_images_dir, "vstar", f"{dat['question_id']}_adv.png")
            if os.path.exists(adv_img_path):
                img_path = adv_img_path
            else:
                img_path = os.path.join(image_dir, dat['image'])
        else:
            img_path = os.path.join(image_dir, dat['image'])

        text = dat['text'] + task_instruction

        # Get replacement vectors for this question if available
        replace_vec = None
        if latent_replace_vectors is not None and str(dat['question_id']) in latent_replace_vectors:
            replace_vec = latent_replace_vectors[str(dat['question_id'])]

        try:
            prediction, latent_vectors = run_inference(model, processor, img_path, text, max_new_tokens=max_new_tokens, latent_replace_vectors=replace_vec)
            # Store latent vectors by question_id
            if latent_vectors:
                latent_vectors_dict[dat['question_id']] = [lv.numpy() for lv in latent_vectors]
            for lv in latent_vectors:
                lv_tensor = lv.squeeze(0) if lv.dim() > 1 else lv
                l2_length = torch.sqrt(torch.sum(lv_tensor ** 2)).item()
                l2_lengths.append(l2_length)
        except Exception as e:
            print(f"\nError on sample {dat['question_id']}: {e}")
            import traceback
            traceback.print_exc()
            prediction = ""
            latent_vectors = []

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

    # Save latent vectors to .npz file
    if save_latent_vectors and latent_vectors_dict:
        # Convert integer keys to strings for np.savez compatibility
        latent_vectors_dict_str = {str(k): v for k, v in latent_vectors_dict.items()}
        np.savez(latent_npz_file, **latent_vectors_dict_str)
        all_l2 = [torch.sqrt(torch.sum(torch.tensor(lv) ** 2)).item() for lv_dict in latent_vectors_dict.values() for lv in lv_dict]
        mean_l2 = np.mean(all_l2) if all_l2 else 0.0
        total_vectors = sum(len(lv_list) for lv_list in latent_vectors_dict.values())
        print(f"Saved {total_vectors} latent vectors for {len(latent_vectors_dict)} questions to {latent_npz_file}")
        print(f"Mean L2 length: {mean_l2:.6f}")
    else:
        mean_l2 = 0.0
        if save_latent_vectors:
            print("No latent vectors collected for VSTAR")

    accuracy = correct / total if total > 0 else 0
    print(f"VSTAR Accuracy: {correct}/{total} = {accuracy*100:.2f}%")

    step_result = {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "passed_ids": passed_ids,
        "results": results,
        "mean_l2_length": mean_l2
    }
    json.dump(results, open(out_file, 'w+'), indent=2)

    return step_result


def evaluate_mmstar(model, processor, dataset, image_dir, out_dir, max_new_tokens=512,
                    replace_images=False, adv_images_dir=None, filter_ids=None, num_samples=None,
                    save_latent_vectors=False, latent_replace_vectors=None):
    """Evaluate on MMStar dataset."""
    print(f"\nEvaluating MMStar (max_latent_steps={MAX_LATENT_STEPS})")
    if replace_images:
        print(f"  Using adversarial images from: {adv_images_dir}/mmstar")
    if latent_replace_vectors is not None:
        print(f"  Using replacement latent vectors for {len(latent_replace_vectors)} questions")
    if filter_ids is not None:
        print(f"  Filtering to {len(filter_ids)} passed IDs from summary")
    if num_samples is not None:
        print(f"  Limiting to first {num_samples} samples")
    os.makedirs(out_dir, exist_ok=True)
    task_instruction = get_task_instruction("mmstar")

    out_file = os.path.join(out_dir, f"mmstar_latent{MAX_LATENT_STEPS:03d}.json")
    latent_npz_file = os.path.join(out_dir, f"mmstar_latent{MAX_LATENT_STEPS:03d}.npz")
    total, correct = 0, 0
    results = []
    passed_ids = []
    evaluated_count = 0
    latent_vectors_dict = {}  # {question_id: list of latent vectors}
    l2_lengths = []

    for dat in tqdm(dataset, desc="MMStar"):
        if filter_ids is not None and dat['question_id'] not in filter_ids:
            continue
        if num_samples is not None and evaluated_count >= num_samples:
            break

        evaluated_count += 1

        if replace_images and adv_images_dir:
            adv_img_path = os.path.join(adv_images_dir, "mmstar", f"{dat['question_id']}_adv.png")
            if os.path.exists(adv_img_path):
                img_path = adv_img_path
            else:
                img_path = os.path.join(image_dir, dat['image'])
        else:
            img_path = os.path.join(image_dir, dat['image'])

        text = dat['query'] + task_instruction

        # Get replacement vectors for this question if available
        replace_vec = None
        if latent_replace_vectors is not None and str(dat['question_id']) in latent_replace_vectors:
            replace_vec = latent_replace_vectors[str(dat['question_id'])]

        try:
            prediction, latent_vectors = run_inference(model, processor, img_path, text, max_new_tokens=max_new_tokens, latent_replace_vectors=replace_vec)
            # Store latent vectors by question_id
            if latent_vectors:
                latent_vectors_dict[dat['question_id']] = [lv.numpy() for lv in latent_vectors]
            for lv in latent_vectors:
                lv_tensor = lv.squeeze(0) if lv.dim() > 1 else lv
                l2_length = torch.sqrt(torch.sum(lv_tensor ** 2)).item()
                l2_lengths.append(l2_length)
        except Exception as e:
            print(f"\nError on sample {dat['question_id']}: {e}")
            import traceback
            traceback.print_exc()
            prediction = ""
            latent_vectors = []

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

    # Save latent vectors to .npz file
    if save_latent_vectors and latent_vectors_dict:
        # Convert integer keys to strings for np.savez compatibility
        latent_vectors_dict_str = {str(k): v for k, v in latent_vectors_dict.items()}
        np.savez(latent_npz_file, **latent_vectors_dict_str)
        all_l2 = [torch.sqrt(torch.sum(torch.tensor(lv) ** 2)).item() for lv_dict in latent_vectors_dict.values() for lv in lv_dict]
        mean_l2 = np.mean(all_l2) if all_l2 else 0.0
        total_vectors = sum(len(lv_list) for lv_list in latent_vectors_dict.values())
        print(f"Saved {total_vectors} latent vectors for {len(latent_vectors_dict)} questions to {latent_npz_file}")
        print(f"Mean L2 length: {mean_l2:.6f}")
    else:
        mean_l2 = 0.0
        if save_latent_vectors:
            print("No latent vectors collected for MMStar")

    accuracy = correct / total if total > 0 else 0
    print(f"MMStar Accuracy: {correct}/{total} = {accuracy*100:.2f}%")

    step_result = {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "passed_ids": passed_ids,
        "results": results,
        "mean_l2_length": mean_l2
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
        "max_latent_steps": MAX_LATENT_STEPS,
        "mode": "MonetModel with custom _sample (no vLLM required)",
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
    parser = argparse.ArgumentParser(description="Monet-7B Evaluation")
    parser.add_argument("--dataset", type=str, default="all",
                        choices=["all", "mmvp", "vstar", "mmstar"],
                        help="Which dataset to evaluate: 'all', 'mmvp', 'vstar', or 'mmstar' (default: all)")
    parser.add_argument("--max_new_tokens", type=int, default=13,
                        help="Maximum number of tokens to generate (default: 2)")
    parser.add_argument("--max_latent_steps", type=int, default=10,
                        help="Maximum number of latent reasoning steps (default: 10)")
    parser.add_argument("--adv_images_dir", type=str, default=None,
                        help="Directory containing adversarial images (mmvp/, vstar/, mmstar/ subdirectories)")
    parser.add_argument("--use_passed_ids", action="store_true",
                        help="Only test IDs that are in the passed_ids from summary.json")
    parser.add_argument("--num_samples", type=int, default=None,
                        help="Number of samples to evaluate (default: all)")
    parser.add_argument("--summary_path", type=str, default=None,
                        help="Path to summary.json for loading passed_ids (default: test_results/summary.json)")
    parser.add_argument("--save_latent_vectors", action="store_true", default=False,
                        help="Save latent vectors to .npz file (default: False)")
    parser.add_argument("--latent_vectors_replace", type=str, default=None,
                        help="Path to .npz file containing latent vectors to replace with")
    args = parser.parse_args()
    #--adv_images_dir random_images_monet
    #--latent_vectors_replace test_results/adv/mmstar/mmstar_latent010.npz
    #python test.py --dataset vstar --summary_path test_results/org/monet_latent010/summary.json --use_passed_ids  --adv_images_dir adv_images_white --save_latent_vectors
    '''python test.py --dataset vstar --summary_path test_results/org/monet_latent010/summary.json --use_passed_ids  --adv_images_dir adv_images_white ; python test.py --dataset vstar --summary_path test_results/org/monet_latent010/summary.json --use_passed_ids  --adv_images_dir adv_imgs_lvr ; python test.py --dataset vstar --summary_path test_results/org/monet_latent010/summary.json --use_passed_ids  --adv_images_dir adv_imgs_skila ; python test.py --dataset mmvp --summary_path test_results/org/monet_latent010/summary.json --use_passed_ids  --adv_images_dir adv_images_white ; python test.py --dataset mmvp --summary_path test_results/org/monet_latent010/summary.json --use_passed_ids  --adv_images_dir adv_imgs_lvr ; python test.py --dataset mmvp --summary_path test_results/org/monet_latent010/summary.json --use_passed_ids  --adv_images_dir adv_imgs_skila ; python test.py --dataset mmstar --summary_path test_results/org/monet_latent010/summary.json --use_passed_ids  --adv_images_dir adv_images_white ; python test.py --dataset mmstar --summary_path test_results/org/monet_latent010/summary.json --use_passed_ids  --adv_images_dir adv_imgs_lvr ; python test.py --dataset mmstar --summary_path test_results/org/monet_latent010/summary.json --use_passed_ids  --adv_images_dir adv_imgs_skila  
    '''

    global MAX_LATENT_STEPS
    MAX_LATENT_STEPS = args.max_latent_steps

    print("=" * 64)
    print("Monet-7B Evaluation on Local Datasets")
    print(f"Mode: {args.dataset.upper()}")
    print(f"Using MonetModel with latent reasoning (no vLLM)")
    print(f"Max latent steps: {MAX_LATENT_STEPS}")
    print(f"Max new tokens: {args.max_new_tokens}")
    print(f"Adversarial images dir: {args.adv_images_dir}/{{mmvp,vstar,mmstar}}")
    print(f"Use passed IDs: {args.use_passed_ids}")
    print(f"Num samples: {args.num_samples}")
    if args.latent_vectors_replace:
        print(f"Replace latent vectors from: {args.latent_vectors_replace}")
    print("=" * 64)

    # Use custom summary path if provided, otherwise use default
    summary_file = args.summary_path if args.summary_path else DEFAULT_SUMMARY_PATH

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

    # Load model using MonetModel class
    model, processor = load_model()

    # Load replacement latent vectors if provided
    latent_replace_vectors = None
    if args.latent_vectors_replace:
        print(f"Loading replacement latent vectors from {args.latent_vectors_replace}...")
        latent_npz = np.load(args.latent_vectors_replace, allow_pickle=True)
        latent_replace_vectors = {k: latent_npz[k] for k in latent_npz.files}
        print(f"Loaded replacement vectors for {len(latent_replace_vectors)} questions")

    # Create output directory
    run_output_dir = os.path.join("test_results", "adv", "monet_latent010")
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
            save_latent_vectors=args.save_latent_vectors,
            latent_replace_vectors=latent_replace_vectors
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
            save_latent_vectors=args.save_latent_vectors,
            latent_replace_vectors=latent_replace_vectors
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

        # Evaluate MMStar
        mmstar_result = evaluate_mmstar(
            model, processor, mmstar_dataset,
            MMSTAR_DATA_DIR, run_output_dir,
            max_new_tokens=args.max_new_tokens,
            replace_images=args.adv_images_dir is not None,
            adv_images_dir=args.adv_images_dir,
            filter_ids=filter_ids,
            num_samples=args.num_samples,
            save_latent_vectors=args.save_latent_vectors,
            latent_replace_vectors=latent_replace_vectors
        )

    # Save summary
    summary_path = os.path.join(run_output_dir, "summary.json")
    save_summary(mmvp_result, vstar_result, mmstar_result, summary_path)

    # Print final summary
    print("\n" + "=" * 64)
    print("Final Results Summary")
    print("=" * 64)
    if mmvp_result is not None:
        l2_str = f", mean L2: {mmvp_result.get('mean_l2_length', 0):.6f}" if args.save_latent_vectors else ""
        print(f"MMVP: {mmvp_result['accuracy']*100:.2f}% ({mmvp_result['correct']}/{mmvp_result['total']}){l2_str}")
    if vstar_result is not None:
        l2_str = f", mean L2: {vstar_result.get('mean_l2_length', 0):.6f}" if args.save_latent_vectors else ""
        print(f"VSTAR: {vstar_result['accuracy']*100:.2f}% ({vstar_result['correct']}/{vstar_result['total']}){l2_str}")
    if mmstar_result is not None:
        l2_str = f", mean L2: {mmstar_result.get('mean_l2_length', 0):.6f}" if args.save_latent_vectors else ""
        print(f"MMStar: {mmstar_result['accuracy']*100:.2f}% ({mmstar_result['correct']}/{mmstar_result['total']}){l2_str}")
    print("\nDone!")


if __name__ == "__main__":
    main()
