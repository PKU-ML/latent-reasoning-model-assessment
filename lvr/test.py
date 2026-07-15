"""
Test script for LVR-7B model on local MMVP and VSTAR datasets.
Loading IDs logic consistent with attack_white.py.
Adversarial images loaded from mmvp/vstar subdirectories (like Monet/test.py).
"""

import sys
import os
from pathlib import Path
import torch
import json
import csv
import argparse
from tqdm import tqdm

import numpy as np

from transformers import AutoTokenizer, AutoProcessor, AutoConfig, Qwen2_5_VLForConditionalGeneration

from src.model.qwen_lvr_model import QwenWithLVR
from src.train.monkey_patch_forward_lvr_test import replace_qwen2_5_with_mixed_modality_forward_lvr
from qwen_vl_utils import process_vision_info

# ==== Config ====
# 模型路径通过 LVR_MODEL_PATH 环境变量注入；不设置时使用 HuggingFace 仓库名
MODEL_PATH = os.environ.get("LVR_MODEL_PATH", "vincentleebang/LVR-7B")

# Auto-detect available GPUs from environment
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Local dataset paths — 全部走 DATA_DIR 环境变量 + 相对子目录
_DATA_ROOT = os.environ.get("DATA_DIR", str(Path(__file__).resolve().parent.parent / "data"))
MMVP_DATA_DIR = os.environ.get("LVR_MMVP_DATA_DIR", str(Path(_DATA_ROOT) / "MMVP"))
MMVP_IMAGE_DIR = os.environ.get("LVR_MMVP_IMAGE_DIR", str(Path(_DATA_ROOT) / "MMVP" / "MMVP Images"))
MMVP_CSV = os.environ.get("LVR_MMVP_CSV", str(Path(_DATA_ROOT) / "MMVP" / "Questions.csv"))

VSTAR_DATA_DIR = os.environ.get("LVR_VSTAR_DATA_DIR", str(Path(_DATA_ROOT) / "vstar"))
VSTAR_JSONL = os.environ.get("LVR_VSTAR_JSONL", str(Path(_DATA_ROOT) / "vstar" / "test_questions.jsonl"))

MMSTAR_DATA_DIR = os.environ.get("LVR_MMSTAR_DATA_DIR", str(Path(_DATA_ROOT) / "MMStar"))
MMSTAR_METADATA = os.environ.get("LVR_MMSTAR_METADATA", str(Path(_DATA_ROOT) / "MMStar" / "metadata.json"))

# Output directory — 默认指向 <lvr>/results
OUTPUT_DIR = os.environ.get("LVR_OUTPUT_DIR", str(Path(__file__).resolve().parent / "results"))
LVR_STEPS = [4]
DECODING_STRATEGY = "steps"

# Default summary paths for loading passed_ids (same as attack_white.py)
MMVP_SUMMARY_PATH = "test_results/org/run_steps/summary.json"
VSTAR_SUMMARY_PATH = "test_results/org/run_steps/summary.json"
MMSTAR_SUMMARY_PATH = "test_results/org/run_steps/summary.json"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def accuracy_reward(response: str, ground_truth: str) -> float:
    """Check if the response matches the ground truth answer."""
    given_answer = response.split('<answer>')[-1]
    given_answer = given_answer.split('</answer')[0].strip()
    if " " in given_answer:
        given_answer = given_answer.split(" ")[0]
    if len(given_answer) > 1:
        given_answer = given_answer[0]
    return given_answer == ground_truth


def get_task_instruction(bench_name):
    if bench_name.lower() == "vstar":
        return "\nAnswer with the option's letter from the given choices directly."
    elif bench_name.lower() == "mmvp":
        return "\nAnswer with the option's letter from the given choices directly."
    else:
        return "\nAnswer with the option's letter from the given choices directly."


def create_messages(img_path, question):
    if not isinstance(img_path, list):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img_path},
                    {"type": "text", "text": question},
                ],
            }
        ]
    else:
        vision_content = []
        for ip in img_path:
            vision_content.append({"type": "image", "image": ip})
        vision_content.append({"type": "text", "text": question})
        messages = [{"role": "user", "content": vision_content}]
    return messages


def load_model_and_processor():
    """Load the LVR-7B model and processor."""
    print(f"Loading model from {MODEL_PATH}...")

    config = AutoConfig.from_pretrained(MODEL_PATH)

    replace_qwen2_5_with_mixed_modality_forward_lvr(
        inference_mode=True, lvr_head=config.lvr_head
    )

    model = QwenWithLVR.from_pretrained(
        MODEL_PATH,
        config=config,
        trust_remote_code=True,
        dtype="auto",
        device_map="auto",
    )

    processor = AutoProcessor.from_pretrained(MODEL_PATH)

    return model, processor


def run_inference(model, processor, img_path, text, steps, decoding_strategy, question_id=None):
    """Run inference on a single sample with latent vector recording."""
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
    inputs = inputs.to(DEVICE)

    # Set question ID for replacement mode if applicable
    if question_id is not None and hasattr(model, 'set_current_question_id'):
        model.set_current_question_id(question_id)

    # Enable latent vector storage (for saving, not replacement mode)
    store_mode = hasattr(model, '_use_replacement_latent') and model._use_replacement_latent
    if not store_mode:
        model.store_latent_vectors = True
        model._latent_vectors_list = []

    lvr_steps = [steps]
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=16,  # Answers are single letters, no need for long generation
            decoding_strategy=decoding_strategy,
            lvr_steps=lvr_steps
        )
        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        #print("Generated IDs:", generated_ids_trimmed)
        output_text = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False
        )

    # Retrieve and clear latent vectors (only for save mode)
    latent_vectors = []
    if not store_mode:
        latent_vectors = model._latent_vectors_list if hasattr(model, '_latent_vectors_list') else []
        model.store_latent_vectors = False
        model._latent_vectors_list = []

    return output_text, latent_vectors


def load_mmvp_dataset():
    """Load MMVP dataset from local CSV and images."""
    data = []
    with open(MMVP_CSV, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            idx = int(row["Index"])
            text = row["Question"] + '\nOptions:\n' + row["Options"]
            label = row["Correct Answer"].strip().upper()[1] if row["Correct Answer"] in ['(a)', '(b)'] else row["Correct Answer"].strip().upper()
            if label in ['A', 'B']:
                label = label
            else:
                label = row["Correct Answer"].strip().upper()
                if len(label) > 1:
                    label = label[1]

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
            item["question_id"] = item["question_id"]
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


def get_passed_ids_from_summary(summary_path, passed_ids_key="mmvp_results"):
    """Load passed_ids from summary.json (consistent with attack_white.py)."""
    if not os.path.exists(summary_path):
        print(f"Warning: Summary file not found: {summary_path}")
        return None
    with open(summary_path, 'r') as f:
        summary = json.load(f)
    # mmvp_results/vstar_results structure: key -> steps_X -> passed_ids
    if passed_ids_key in summary:
        for step_key, result in summary[passed_ids_key].items():
            if "passed_ids" in result:
                return set(result["passed_ids"])
    print(f"Warning: No passed_ids found for '{passed_ids_key}' in {summary_path}")
    return None


def evaluate_mmvp(model, processor, dataset, image_dir, out_dir, decoding_strategy="steps",
                  replace_images=False, adv_images_dir=None, filter_ids=None, num_samples=None,
                  save_latent_vectors=False):
    """Evaluate on MMVP dataset."""
    print(f"\nEvaluating MMVP with decoding strategy: {decoding_strategy}")
    if replace_images:
        print(f"  Using adversarial images from: {adv_images_dir}/mmvp")
    if filter_ids is not None:
        print(f"  Filtering to {len(filter_ids)} passed IDs from summary")
    if num_samples is not None:
        print(f"  Limiting to first {num_samples} samples")
    if save_latent_vectors:
        print(f"  Saving latent vectors enabled")
    os.makedirs(out_dir, exist_ok=True)
    task_instruction = get_task_instruction("mmvp")

    all_results = {}
    for steps in LVR_STEPS:
        out_file = os.path.join(out_dir, f"mmvp_{decoding_strategy}{steps:03d}.json")
        latent_npz_file = os.path.join(out_dir, f"mmvp_{decoding_strategy}{steps:03d}_latent.npz")
        total, correct = 0, 0
        results = []
        passed_ids = []
        evaluated_count = 0
        latent_dict = {}  # {question_id: [vectors]}
        l2_lengths = []

        for dat in tqdm(dataset, desc=f"MMVP steps={steps}"):
            # Filter by passed_ids if specified
            if filter_ids is not None and dat['question_id'] not in filter_ids:
                continue

            # Limit number of samples if specified
            if num_samples is not None and evaluated_count >= num_samples:
                break

            evaluated_count += 1

            # Use adversarial image if replace_images is enabled (Monet/test.py style)
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

            qid = dat['question_id']
            outputs, latent_vectors = run_inference(model, processor, img_path, text, steps, decoding_strategy, question_id=qid)
            prediction = outputs[0]

            # Store latent vectors by question_id (only if save_latent_vectors is enabled)
            if save_latent_vectors:
                latent_dict[qid] = []
                for lv in latent_vectors:
                    lv_tensor = lv.squeeze(0) if lv.dim() > 1 else lv
                    latent_dict[qid].append(lv_tensor.float().numpy())
                    l2_length = torch.sqrt(torch.sum(lv_tensor ** 2)).item()
                    l2_lengths.append(l2_length)

            is_correct = accuracy_reward(prediction, dat['label'])
            if is_correct:
                passed_ids.append(dat['question_id'])

            res = {
                'id': dat['question_id'],
                'prediction': prediction,
                'label': dat['label'],
                'correct': is_correct
            }
            results.append(res)

            if is_correct:
                correct += 1
            total += 1

        accuracy = correct / total if total > 0 else 0
        print(f"MMVP Steps: {steps} - Accuracy: {correct}/{total} = {accuracy*100:.2f}%")

        # Save latent vectors to .npz file (only if save_latent_vectors is enabled)
        if save_latent_vectors and latent_dict:
            latent_dict_str = {str(k): v for k, v in latent_dict.items()}
            np.savez(latent_npz_file, **latent_dict_str)
            total_vectors = sum(len(v) for v in latent_dict.values())
            mean_l2 = np.mean(l2_lengths) if l2_lengths else 0.0
            print(f"Saved latent vectors for {len(latent_dict)} questions ({total_vectors} total vectors) to {latent_npz_file}")
            print(f"Mean L2 length: {mean_l2:.6f}")
        else:
            mean_l2 = 0.0
            if not save_latent_vectors:
                print("Latent vector saving disabled")
            else:
                print("No latent vectors collected for MMVP")

        # Save results for this step
        step_result = {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "passed_ids": passed_ids,
            "results": results,
            "mean_l2_length": mean_l2
        }
        all_results[f"steps_{steps}"] = step_result

        json.dump(results, open(out_file, 'w+'), indent=2)

    return all_results


def evaluate_vstar(model, processor, dataset, image_dir, out_dir, decoding_strategy="steps",
                   replace_images=False, adv_images_dir=None, filter_ids=None, num_samples=None,
                   save_latent_vectors=False):
    """Evaluate on VSTAR dataset."""
    print(f"\nEvaluating VSTAR with decoding strategy: {decoding_strategy}")
    if replace_images:
        print(f"  Using adversarial images from: {adv_images_dir}/vstar")
    if filter_ids is not None:
        print(f"  Filtering to {len(filter_ids)} passed IDs from summary")
    if num_samples is not None:
        print(f"  Limiting to first {num_samples} samples")
    if save_latent_vectors:
        print(f"  Saving latent vectors enabled")
    os.makedirs(out_dir, exist_ok=True)
    task_instruction = get_task_instruction("vstar")

    all_results = {}
    for steps in LVR_STEPS:
        out_file = os.path.join(out_dir, f"vstar_{decoding_strategy}{steps:03d}.json")
        latent_npz_file = os.path.join(out_dir, f"vstar_{decoding_strategy}{steps:03d}_latent.npz")
        total, correct = 0, 0
        results = []
        passed_ids = []
        evaluated_count = 0
        latent_dict = {}  # {question_id: [vectors]}
        l2_lengths = []

        for dat in tqdm(dataset, desc=f"VSTAR steps={steps}"):
            # Filter by passed_ids if specified
            if filter_ids is not None and int(dat['question_id']) not in filter_ids:
                continue

            # Limit number of samples if specified
            if num_samples is not None and evaluated_count >= num_samples:
                break

            evaluated_count += 1

            # Use adversarial image if replace_images is enabled (Monet/test.py style)
            if replace_images and adv_images_dir:
                adv_img_path = os.path.join(adv_images_dir, "vstar", f"{dat['question_id']}_adv.png")
                if os.path.exists(adv_img_path):
                    img_path = adv_img_path
                else:
                    img_path = os.path.join(image_dir, dat['image'])
            else:
                img_path = os.path.join(image_dir, dat['image'])

            text = dat['text'] + task_instruction

            qid = dat['question_id']
            outputs, latent_vectors = run_inference(model, processor, img_path, text, steps, decoding_strategy, question_id=qid)
            prediction = outputs[0]

            # Store latent vectors by question_id (only if save_latent_vectors is enabled)
            if save_latent_vectors:
                latent_dict[qid] = []
                for lv in latent_vectors:
                    lv_tensor = lv.squeeze(0) if lv.dim() > 1 else lv
                    latent_dict[qid].append(lv_tensor.float().numpy())
                    l2_length = torch.sqrt(torch.sum(lv_tensor ** 2)).item()
                    l2_lengths.append(l2_length)

            is_correct = accuracy_reward(prediction, dat['label'])
            if is_correct:
                passed_ids.append(dat['question_id'])

            res = {
                'id': dat['question_id'],
                'prediction': prediction,
                'label': dat['label'],
                'correct': is_correct,
                'category': dat.get('category', 'unknown')
            }
            results.append(res)

            if is_correct:
                correct += 1
            total += 1

        accuracy = correct / total if total > 0 else 0
        print(f"VSTAR Steps: {steps} - Accuracy: {correct}/{total} = {accuracy*100:.2f}%")

        # Save latent vectors to .npz file (only if save_latent_vectors is enabled)
        if save_latent_vectors and latent_dict:
            latent_dict_str = {str(k): v for k, v in latent_dict.items()}
            np.savez(latent_npz_file, **latent_dict_str)
            total_vectors = sum(len(v) for v in latent_dict.values())
            mean_l2 = np.mean(l2_lengths) if l2_lengths else 0.0
            print(f"Saved latent vectors for {len(latent_dict)} questions ({total_vectors} total vectors) to {latent_npz_file}")
            print(f"Mean L2 length: {mean_l2:.6f}")
        else:
            mean_l2 = 0.0
            if not save_latent_vectors:
                print("Latent vector saving disabled")
            else:
                print("No latent vectors collected for VSTAR")

        # Save results for this step
        step_result = {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "passed_ids": passed_ids,
            "results": results,
            "mean_l2_length": mean_l2
        }
        all_results[f"steps_{steps}"] = step_result

        json.dump(results, open(out_file, 'w+'), indent=2)

    return all_results


def evaluate_mmstar(model, processor, dataset, image_dir, out_dir, decoding_strategy="steps",
                    replace_images=False, adv_images_dir=None, filter_ids=None, num_samples=None,
                    save_latent_vectors=False):
    """Evaluate on MMStar dataset."""
    print(f"\nEvaluating MMStar with decoding strategy: {decoding_strategy}")
    if replace_images:
        print(f"  Using adversarial images from: {adv_images_dir}/mmstar")
    if filter_ids is not None:
        print(f"  Filtering to {len(filter_ids)} passed IDs from summary")
    if num_samples is not None:
        print(f"  Limiting to first {num_samples} samples")
    if save_latent_vectors:
        print(f"  Saving latent vectors enabled")
    os.makedirs(out_dir, exist_ok=True)
    task_instruction = get_task_instruction("mmstar")

    all_results = {}
    for steps in LVR_STEPS:
        out_file = os.path.join(out_dir, f"mmstar_{decoding_strategy}{steps:03d}.json")
        latent_npz_file = os.path.join(out_dir, f"mmstar_{decoding_strategy}{steps:03d}_latent.npz")
        total, correct = 0, 0
        results = []
        passed_ids = []
        evaluated_count = 0
        latent_dict = {}  # {question_id: [vectors]}
        l2_lengths = []

        for dat in tqdm(dataset, desc=f"MMStar steps={steps}"):
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

            qid = dat['question_id']
            outputs, latent_vectors = run_inference(model, processor, img_path, text, steps, decoding_strategy, question_id=qid)
            prediction = outputs[0]

            # Store latent vectors by question_id (only if save_latent_vectors is enabled)
            if save_latent_vectors:
                latent_dict[qid] = []
                for lv in latent_vectors:
                    lv_tensor = lv.squeeze(0) if lv.dim() > 1 else lv
                    latent_dict[qid].append(lv_tensor.float().numpy())
                    l2_length = torch.sqrt(torch.sum(lv_tensor ** 2)).item()
                    l2_lengths.append(l2_length)

            is_correct = accuracy_reward(prediction, dat['label'])
            if is_correct:
                passed_ids.append(dat['question_id'])

            res = {
                'id': dat['question_id'],
                'prediction': prediction,
                'label': dat['label'],
                'correct': is_correct,
                'category': dat.get('category', ''),
                'l2_category': dat.get('l2_category', '')
            }
            results.append(res)

            if is_correct:
                correct += 1
            total += 1

        accuracy = correct / total if total > 0 else 0
        print(f"MMStar Steps: {steps} - Accuracy: {correct}/{total} = {accuracy*100:.2f}%")

        # Save latent vectors to .npz file (only if save_latent_vectors is enabled)
        if save_latent_vectors and latent_dict:
            latent_dict_str = {str(k): v for k, v in latent_dict.items()}
            np.savez(latent_npz_file, **latent_dict_str)
            total_vectors = sum(len(v) for v in latent_dict.values())
            mean_l2 = np.mean(l2_lengths) if l2_lengths else 0.0
            print(f"Saved latent vectors for {len(latent_dict)} questions ({total_vectors} total vectors) to {latent_npz_file}")
            print(f"Mean L2 length: {mean_l2:.6f}")
        else:
            mean_l2 = 0.0
            if not save_latent_vectors:
                print("Latent vector saving disabled")
            else:
                print("No latent vectors collected for MMStar")

        step_result = {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "passed_ids": passed_ids,
            "results": results,
            "mean_l2_length": mean_l2
        }
        all_results[f"steps_{steps}"] = step_result

        json.dump(results, open(out_file, 'w+'), indent=2)

    return all_results


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


def save_summary(mmvp_results, vstar_results, mmstar_results, output_path):
    """Save summary of evaluation results."""
    summary = {
        "model_path": MODEL_PATH,
        "mmvp_results": {},
        "vstar_results": {},
        "mmstar_results": {}
    }

    # MMVP summary
    for step_key, result in mmvp_results.items():
        summary["mmvp_results"][step_key] = {
            "accuracy": result["accuracy"],
            "correct": result["correct"],
            "total": result["total"],
            "passed_ids": result["passed_ids"],
            "mean_l2_length": result.get("mean_l2_length", 0.0),
            "examples": collect_examples(result["results"])
        }

    # VSTAR summary
    for step_key, result in vstar_results.items():
        summary["vstar_results"][step_key] = {
            "accuracy": result["accuracy"],
            "correct": result["correct"],
            "total": result["total"],
            "passed_ids": result["passed_ids"],
            "mean_l2_length": result.get("mean_l2_length", 0.0),
            "examples": collect_examples(result["results"])
        }

    # MMStar summary
    for step_key, result in mmstar_results.items():
        summary["mmstar_results"][step_key] = {
            "accuracy": result["accuracy"],
            "correct": result["correct"],
            "total": result["total"],
            "passed_ids": result["passed_ids"],
            "mean_l2_length": result.get("mean_l2_length", 0.0),
            "examples": collect_examples(result["results"])
        }

    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nSummary saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="LVR-7B Evaluation on Local Datasets")
    parser.add_argument("--dataset", type=str, default="all",
                        choices=["all", "mmvp", "vstar", "mmstar"],
                        help="Dataset to evaluate on: all, mmvp, vstar, or mmstar")
    parser.add_argument("--adv_images_dir", type=str, default=None,
                        help="Directory containing adversarial images (mmvp/, vstar/, mmstar/ subdirectories)")
    parser.add_argument("--use_passed_ids", action="store_true",
                        help="Only test IDs that are in the passed_ids from summary.json")
    parser.add_argument("--num_samples", type=int, default=None,
                        help="Number of samples to evaluate (default: all)")
    parser.add_argument("--summary_path", type=str, default=None,
                        help="Path to summary.json for loading passed_ids")
    parser.add_argument("--save_latent_vectors", action="store_true",
                        help="Save latent vectors to .npz files")
    parser.add_argument("--load_latent_vectors", type=str, default=None,
                        help="Path to .npz file with latent vectors to replace during inference")
    args = parser.parse_args()
    # Examples:
    #   python test.py --dataset all
    #   python test.py --dataset mmvp
    #   python test.py --dataset vstar --use_passed_ids --adv_images_dir random_images
    #   python test.py --dataset mmvp --num_samples 20
    #   python test.py --dataset mmvp --use_passed_ids --num_samples 20
    #   python test.py --dataset mmstar --use_passed_ids --adv_images_dir adv_images_white --save_latent_vectors
    #   python test.py --dataset mmvp --save_latent_vectors  --use_passed_ids --adv_images_dir adv_images_white
    #   --load_latent_vectors results/vstar/vstar_steps004_latent.npz
    # Determine summary path based on dataset (consistent with attack_white.py)
    '''  python test.py --dataset mmvp --use_passed_ids --adv_images_dir adv_images_monet ; python test.py --dataset mmvp --use_passed_ids --adv_images_dir adv_images_skila ; python test.py --dataset mmvp --use_passed_ids --adv_images_dir adv_images_white ;  python test.py --dataset mmstar --use_passed_ids --adv_images_dir adv_images_white ; python test.py --dataset mmstar --use_passed_ids --adv_images_dir adv_images_monet ;  python test.py --dataset mmstar --use_passed_ids --adv_images_dir adv_images_skila ; python test.py --dataset vstar --use_passed_ids --adv_images_dir adv_images_white ; python test.py --dataset vstar --use_passed_ids --adv_images_dir adv_images_monet ; python test.py --dataset vstar --use_passed_ids --adv_images_dir adv_images_skila  '''
    if args.dataset == "vstar":
        summary_file = args.summary_path if args.summary_path else VSTAR_SUMMARY_PATH
        passed_ids_key = "vstar_results"
    elif args.dataset == "mmstar":
        summary_file = args.summary_path if args.summary_path else MMSTAR_SUMMARY_PATH
        passed_ids_key = "mmstar_results"
    else:
        summary_file = args.summary_path if args.summary_path else MMVP_SUMMARY_PATH
        passed_ids_key = "mmvp_results"

    print("=" * 64)
    print("LVR-7B Evaluation on Local Datasets")
    print("=" * 64)
    print(f"Dataset: {args.dataset}")
    print(f"Adversarial images dir: {args.adv_images_dir}")
    print(f"Use passed IDs: {args.use_passed_ids}")
    print(f"Num samples: {args.num_samples}")
    print(f"Summary path: {summary_file}")
    print(f"Save latent vectors: {args.save_latent_vectors}")

    # Load passed_ids if requested (consistent with attack_white.py)
    filter_ids = None
    if args.use_passed_ids:
        filter_ids = get_passed_ids_from_summary(summary_file, passed_ids_key)
        if filter_ids:
            print(f"Loaded {len(filter_ids)} passed IDs for filtering")
            # If num_samples specified, take only first N from passed_ids
            if args.num_samples is not None and args.num_samples < len(filter_ids):
                filter_ids = list(filter_ids)[:args.num_samples]
                print(f"Limited to first {len(filter_ids)} passed IDs")
        else:
            print("Warning: No passed_ids found in summary.json")

    # Set random seed for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)

    # Load model
    model, processor = load_model_and_processor()
    model.eval()

    # Load latent vectors for replacement if specified
    if args.load_latent_vectors:
        print(f"Loading latent vectors from {args.load_latent_vectors}...")
        latent_data = np.load(args.load_latent_vectors)
        # Convert to dict mapping question_id -> list of vectors
        latent_dict = {}
        for key in latent_data.files:
            latent_dict[int(key)] = latent_data[key]
        print(f"Loaded latent vectors for {len(latent_dict)} questions")
        model.set_replacement_latent_vectors(latent_dict)

    # Create output directory
    run_output_dir = os.path.join(OUTPUT_DIR, f"run_{DECODING_STRATEGY}")
    os.makedirs(run_output_dir, exist_ok=True)

    mmvp_results = {}
    vstar_results = {}
    mmstar_results = {}

    if args.dataset in ["all", "mmvp"]:
        print("\nLoading MMVP dataset...")
        mmvp_dataset = load_mmvp_dataset()
        print(f"MMVP: {len(mmvp_dataset)} samples")

        mmvp_results = evaluate_mmvp(
            model, processor, mmvp_dataset,
            MMVP_IMAGE_DIR, run_output_dir, DECODING_STRATEGY,
            replace_images=args.adv_images_dir is not None,
            adv_images_dir=args.adv_images_dir,
            filter_ids=filter_ids,
            num_samples=args.num_samples,
            save_latent_vectors=args.save_latent_vectors
        )

    if args.dataset in ["all", "vstar"]:
        print("\nLoading VSTAR dataset...")
        vstar_dataset = load_vstar_dataset()
        print(f"VSTAR: {len(vstar_dataset)} samples")

        vstar_results = evaluate_vstar(
            model, processor, vstar_dataset,
            VSTAR_DATA_DIR, run_output_dir, DECODING_STRATEGY,
            replace_images=args.adv_images_dir is not None,
            adv_images_dir=args.adv_images_dir,
            filter_ids=filter_ids,
            num_samples=args.num_samples,
            save_latent_vectors=args.save_latent_vectors
        )

    if args.dataset in ["all", "mmstar"]:
        print("\nLoading MMStar dataset...")
        mmstar_dataset = load_mmstar_dataset()
        print(f"MMStar: {len(mmstar_dataset)} samples")

        mmstar_results = evaluate_mmstar(
            model, processor, mmstar_dataset,
            MMSTAR_DATA_DIR, run_output_dir, DECODING_STRATEGY,
            replace_images=args.adv_images_dir is not None,
            adv_images_dir=args.adv_images_dir,
            filter_ids=filter_ids,
            num_samples=args.num_samples,
            save_latent_vectors=args.save_latent_vectors
        )

    # Save summary only if results exist
    if mmvp_results or vstar_results or mmstar_results:
        summary_path = os.path.join(run_output_dir, "summary.json")
        save_summary(mmvp_results, vstar_results, mmstar_results, summary_path)

    # Print final summary
    print("\n" + "=" * 64)
    print("Final Results Summary")
    print("=" * 64)

    if mmvp_results:
        print("\nMMVP:")
        for step_key, result in mmvp_results.items():
            print(f"  {step_key}: {result['accuracy']*100:.2f}% ({result['correct']}/{result['total']}), mean L2: {result.get('mean_l2_length', 0):.6f}")

    if vstar_results:
        print("\nVSTAR:")
        for step_key, result in vstar_results.items():
            print(f"  {step_key}: {result['accuracy']*100:.2f}% ({result['correct']}/{result['total']}), mean L2: {result.get('mean_l2_length', 0):.6f}")

    if mmstar_results:
        print("\nMMStar:")
        for step_key, result in mmstar_results.items():
            print(f"  {step_key}: {result['accuracy']*100:.2f}% ({result['correct']}/{result['total']}), mean L2: {result.get('mean_l2_length', 0):.6f}")

    print("\nDone!")


if __name__ == "__main__":
    main()