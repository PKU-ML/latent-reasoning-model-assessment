"""
Random Noise Attack on Qwen2.5-VL-7B-Instruct.

Mirrors ``lvr/attack_random_noise.py`` but evaluates the vanilla
Qwen2.5-VL-7B-Instruct model (no LVR / monkey patches) and additionally
supports the MMStar benchmark.

攻击方法：
- 对图像添加随机扰动，最大像素差为 +-epsilon/255
- 测试模型在扰动图片上能否做对题目
- 如果某个扰动导致模型做错，保存该图片并提前结束
- 如果所有扰动都未导致模型做错，保存最后一张图片

支持数据集：MMVP, VSTAR, MMStar
保存目录：<output_dir>/{mmvp,vstar,mmstar}/{question_id}_adv.png

用法：
  python attack_random_noise.py --dataset all       # 攻击所有数据集
  python attack_random_noise.py --dataset mmvp     # 只攻击 MMVP
  python attack_random_noise.py --dataset vstar    # 只攻击 VSTAR
  python attack_random_noise.py --dataset mmstar   # 只攻击 MMStar
  python attack_random_noise.py --dataset vstar --num_samples 20
"""

import os
import torch
import json
import csv
import argparse
from PIL import Image
import numpy as np
from tqdm import tqdm

from transformers import AutoTokenizer, AutoProcessor, AutoConfig
from transformers import Qwen2_5_VLForConditionalGeneration

from qwen_vl_utils import process_vision_info


# ==== Config ====
MODEL_PATH = "/root/autodl-tmp/attack/models--Qwen--Qwen2.5-VL-7B-Instruct"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Local dataset paths
MMVP_DATA_DIR = "/root/autodl-tmp/attack/data/MMVP"
MMVP_IMAGE_DIR = "/root/autodl-tmp/attack/data/MMVP/MMVP Images"
MMVP_CSV = "/root/autodl-tmp/attack/data/MMVP/Questions.csv"

VSTAR_DATA_DIR = "/root/autodl-tmp/attack/data/vstar"
VSTAR_JSONL = "/root/autodl-tmp/attack/data/vstar/test_questions.jsonl"

MMSTAR_DATA_DIR = "/root/autodl-tmp/attack/data/MMStar"
MMSTAR_METADATA = "/root/autodl-tmp/attack/data/MMStar/metadata.json"

OUTPUT_DIR = "random_images"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Random noise attack parameters
EPSILON = 4 / 255.0  # Maximum pixel perturbation
MAX_ATTEMPTS = 20    # Maximum number of random perturbations to try per image
SEED = 42            # Random seed for reproducibility

# Default summary paths for loading passed_ids.
# These match the layout written by qwen/test.py
# (results/org/summary.json, flat structure: key -> {accuracy, passed_ids, ...}).
MMVP_SUMMARY_PATH = "results/org/summary.json"
VSTAR_SUMMARY_PATH = "results/org/summary.json"
MMSTAR_SUMMARY_PATH = "results/org/summary.json"


# ---------------------------------------------------------------------------
# Prompt template (same as qwen/test.py)
# ---------------------------------------------------------------------------
REASONING_SYSTEM_PROMPT = (
    "You are a careful visual reasoning assistant. "
    "You MUST always think step-by-step before giving a final answer. "
    "Never skip the reasoning step."
)

REASONING_USER_PROMPT = (
    "Look carefully at the image, noting fine-grained details and the "
    "spatial location of every object mentioned in the question. "
    "Write your reasoning inside <think>...</think> tags, then give your "
    "final answer inside <answer>X</answer> tags, where X is the letter of "
    "the correct option.\n\n"
    "Format:\n"
    "<think>\n"
    "[your reasoning]\n"
    "</think>\n"
    "<answer>X</answer>\n\n"
    "Now answer the following question.\n"
)


def extract_reasoning_and_answer(response: str):
    """Split the model output into (reasoning, answer_letter_or_None)."""
    reasoning = response
    if "<think>" in response and "</think>" in response:
        last_think = response.rsplit("<think>", 1)[-1]
        if "</think>" in last_think:
            reasoning = last_think.split("</think>")[0].strip()

    answer = None
    if "<answer>" in response:
        last_answer_split = response.rsplit("<answer>", 1)
        if len(last_answer_split) >= 2:
            given_answer = last_answer_split[-1]
            if "</answer>" in given_answer:
                given_answer = given_answer.split("</answer>")[0]
            given_answer = given_answer.strip()
            if " " in given_answer:
                given_answer = given_answer.split(" ")[0]
            if given_answer:
                answer = given_answer[0]
    return reasoning, answer


def accuracy_reward(response: str, ground_truth: str) -> float:
    """Check if the response matches the ground truth answer (last <answer>...</answer>)."""
    _, answer = extract_reasoning_and_answer(response)
    if answer is None:
        return False
    return answer == ground_truth


def get_task_instruction():
    return "\nAnswer with the option's letter from the given choices directly."


def create_messages(img_path, question):
    """Build chat-template messages containing the image(s) and question text."""
    if not isinstance(img_path, list):
        user_content = [
            {"type": "image", "image": img_path},
            {"type": "text", "text": question},
        ]
    else:
        user_content = []
        for ip in img_path:
            user_content.append({"type": "image", "image": ip})
        user_content.append({"type": "text", "text": question})
    return [
        {"role": "system", "content": REASONING_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def load_model_and_processor():
    """Load Qwen2.5-VL-7B-Instruct model and processor."""
    print(f"Loading model from {MODEL_PATH}...")

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype="auto",
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(MODEL_PATH)

    return model, processor


def run_inference(model, processor, img_path, text, max_new_tokens=2048):
    """Run inference on a single sample.

    Mirrors qwen/test.py.run_inference: prepends the reasoning user prompt
    (so the model emits ``<think>...</think><answer>X</answer>``) and adds
    the reasoning system prompt via ``create_messages``.
    """
    full_user_text = REASONING_USER_PROMPT + text
    messages = create_messages(img_path, full_user_text)
    text_formatted = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    image_inputs, video_inputs = process_vision_info(messages)

    # Only pass videos when actually present (avoids
    # `IndexError: list index out of range` in transformers' video processor).
    if video_inputs:
        inputs = processor(
            text=[text_formatted],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
    else:
        inputs = processor(
            text=[text_formatted],
            images=image_inputs,
            padding=True,
            return_tensors="pt",
        )
    inputs = inputs.to(DEVICE)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
        )
        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
    return output_text[0]


# ---------------------------------------------------------------------------
# Dataset loaders (mirror qwen/test.py)
# ---------------------------------------------------------------------------
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
        })
    return data


def get_passed_ids_from_summary(summary_path, passed_ids_key="mmvp_results"):
    """Load passed_ids from summary.json, supporting flat and nested layouts.

    Flat (qwen/test.py output):
        { "mmvp_results": { "accuracy": .., "passed_ids": [...], ... } }
    Nested (lvr/test.py style):
        { "mmvp_results": { "steps_4": { "accuracy": .., "passed_ids": [...] } } }
    """
    if not os.path.exists(summary_path):
        print(f"Warning: Summary file not found: {summary_path}")
        return None
    with open(summary_path, 'r') as f:
        summary = json.load(f)
    if passed_ids_key not in summary:
        print(f"Warning: Key '{passed_ids_key}' not found in {summary_path}")
        return None
    sub = summary[passed_ids_key]
    if not isinstance(sub, dict):
        return None
    if "passed_ids" in sub and isinstance(sub["passed_ids"], list):
        return set(sub["passed_ids"])
    for step_key, result in sub.items():
        if isinstance(result, dict) and "passed_ids" in result:
            return set(result["passed_ids"])
    print(f"Warning: No passed_ids found for '{passed_ids_key}' in {summary_path}")
    return None


# ---------------------------------------------------------------------------
# Random noise attack primitives
# ---------------------------------------------------------------------------
def add_random_noise(image: Image.Image, epsilon: float = 4/255.0) -> Image.Image:
    """Add uniform [-epsilon, epsilon] noise to each pixel, then clamp to [0,1]."""
    img_array = np.array(image).astype(np.float32) / 255.0
    noise = np.random.uniform(-epsilon, epsilon, img_array.shape).astype(np.float32)
    perturbed = np.clip(img_array + noise, 0, 1)
    perturbed_uint8 = (perturbed * 255).astype(np.uint8)
    return Image.fromarray(perturbed_uint8)


def build_text_for_bench(bench_name, question):
    """Construct the user prompt for a given bench (MMVP rewrites (a)/(b) → A./B.)."""
    task_instruction = get_task_instruction()
    if bench_name == "mmvp":
        text = question.replace('(a)', 'A.').replace('(b)', 'B.')
    else:
        text = question
    return text + task_instruction


def verify_original_correct(model, processor, img_path, question, label, bench_name="mmvp"):
    """Verify the model gives the correct answer on the original (clean) image."""
    text = build_text_for_bench(bench_name, question)
    prediction = run_inference(model, processor, img_path, text)
    _, answer = extract_reasoning_and_answer(prediction)
    is_correct = accuracy_reward(prediction, label)
    return is_correct, answer, prediction


def test_perturbed_image(model, processor, perturbed_img: Image.Image, question, label, bench_name="mmvp"):
    """Save perturbed image to a temp PNG and test the model on it."""
    import tempfile

    text = build_text_for_bench(bench_name, question)
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        tmp_path = tmp.name
    try:
        perturbed_img.save(tmp_path)
        prediction = run_inference(model, processor, tmp_path, text)
        _, answer = extract_reasoning_and_answer(prediction)
        is_correct = accuracy_reward(prediction, label)
        return is_correct, answer, prediction
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


# ---------------------------------------------------------------------------
# Per-dataset attack driver
# ---------------------------------------------------------------------------
def run_attack_on_dataset(model, processor, dataset, image_dir, out_dir, bench_name):
    """Run random noise attack on a dataset.

    Returns (results, attack_success_count, attack_fail_count).
    """
    print(f"\n{'='*64}")
    print(f"Running attack on {bench_name.upper()} dataset")
    print(f"{'='*64}")

    os.makedirs(out_dir, exist_ok=True)
    print(f"Total samples to attack: {len(dataset)}")

    results = []
    attack_success_count = 0
    attack_fail_count = 0

    print("\nStarting attack...")

    for item in tqdm(dataset, desc=f"Attacking {bench_name}"):
        question_id = item['question_id']

        img_path = os.path.join(image_dir, item['image'])
        if bench_name == "vstar":
            question = item['text']
        else:  # mmvp / mmstar both use 'query' / 'question'
            question = item['query']

        label = item['label']

        # Verify original image is correct
        original_correct, original_answer, original_pred = verify_original_correct(
            model, processor, img_path, question, label, bench_name
        )

        if not original_correct:
            print(f"\n[Warning] Question {question_id}: Model originally gives wrong answer on clean image (answer={original_answer}, GT={label})")
            results.append({
                'question_id': question_id,
                'gt': label,
                'original_answer': original_answer,
                'attack_success': False,
                'reason': 'original_incorrect',
                'attempts': 0,
                'saved_image': None
            })
            continue

        # Try random perturbations
        original_img = Image.open(img_path).convert('RGB')
        attack_success = False
        final_perturbed_img = None
        wrong_answer = None
        attempts_made = 0

        for attempt in range(MAX_ATTEMPTS):
            attempts_made = attempt + 1

            perturbed_img = add_random_noise(original_img, epsilon=EPSILON)
            is_correct, answer, _ = test_perturbed_image(
                model, processor, perturbed_img, question, label, bench_name
            )

            final_perturbed_img = perturbed_img

            if not is_correct:
                attack_success = True
                wrong_answer = answer
                attack_success_count += 1
                print(f"\n[Success] Question {question_id}: Attack successful at attempt {attempts_made}")
                print(f"          GT={label}, Wrong answer={answer}")
                break

        if not attack_success:
            attack_fail_count += 1
            print(f"\n[Fail] Question {question_id}: No perturbation caused error after {MAX_ATTEMPTS} attempts")

        save_path = os.path.join(out_dir, f"{question_id}_adv.png")
        if final_perturbed_img is not None:
            final_perturbed_img.save(save_path)

        results.append({
            'question_id': question_id,
            'gt': label,
            'original_answer': original_answer,
            'attack_success': attack_success,
            'wrong_answer': wrong_answer,
            'attempts': attempts_made,
            'saved_image': save_path if attack_success else f"{question_id}_adv.png (last attempt)"
        })

    print("\n" + "=" * 64)
    print(f"Attack Summary for {bench_name.upper()}")
    print("=" * 64)
    print(f"Total samples tested: {len(results)}")
    print(f"Attack success (model wrong): {attack_success_count}")
    print(f"Attack failed (model still correct): {attack_fail_count}")
    rate = attack_success_count / len(results) if results else 0
    print(f"Attack success rate: {rate * 100:.2f}%")
    print(f"\nImages saved to: {os.path.abspath(out_dir)}")

    return results, attack_success_count, attack_fail_count


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Random Noise Attack on Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--dataset", type=str, default="all",
                        choices=["all", "mmvp", "vstar", "mmstar"],
                        help="Which dataset to attack: 'all', 'mmvp', 'vstar', or 'mmstar' (default: all)")
    parser.add_argument("--max_attempts", type=int, default=20,
                        help="Maximum number of random perturbations to try per image")
    parser.add_argument("--epsilon", type=float, default=4/255.0,
                        help="Maximum pixel perturbation (default: 4/255)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--num_samples", type=int, default=None,
                        help="Number of samples to test (default: all passed_ids)")
    parser.add_argument("--summary_path", type=str, default=None,
                        help="Path to summary.json for loading passed_ids")
    parser.add_argument("--output_dir", type=str, default="random_images",
                        help="Output directory for adversarial images (default: random_images)")
    args = parser.parse_args()

    global OUTPUT_DIR, MAX_ATTEMPTS, EPSILON, SEED

    MAX_ATTEMPTS = args.max_attempts
    EPSILON = args.epsilon
    SEED = args.seed
    OUTPUT_DIR = args.output_dir
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 64)
    print("Random Noise Attack on Qwen2.5-VL-7B-Instruct")
    print("=" * 64)
    print(f"Dataset: {args.dataset}")
    print(f"Max attempts per image: {MAX_ATTEMPTS}")
    print(f"Epsilon (max pixel perturbation): {EPSILON:.6f} ({int(EPSILON*255)}/255)")
    print(f"Random seed: {SEED}")
    print(f"Output directory: {OUTPUT_DIR}")

    np.random.seed(SEED)
    torch.manual_seed(SEED)

    print("\nLoading model...")
    model, processor = load_model_and_processor()
    model.eval()
    print("Model loaded")

    datasets_to_process = ["mmvp", "vstar", "mmstar"] if args.dataset == "all" else [args.dataset]

    all_results = []

    for bench_name in datasets_to_process:
        # Per-dataset config
        if bench_name == "vstar":
            summary_file = args.summary_path if args.summary_path else VSTAR_SUMMARY_PATH
            passed_ids_key = "vstar_results"
            output_subdir = "vstar"
        elif bench_name == "mmstar":
            summary_file = args.summary_path if args.summary_path else MMSTAR_SUMMARY_PATH
            passed_ids_key = "mmstar_results"
            output_subdir = "mmstar"
        else:
            summary_file = args.summary_path if args.summary_path else MMVP_SUMMARY_PATH
            passed_ids_key = "mmvp_results"
            output_subdir = "mmvp"

        bench_output_dir = os.path.join(OUTPUT_DIR, output_subdir)
        os.makedirs(bench_output_dir, exist_ok=True)

        print(f"\nLoading passed_ids from {summary_file} (key={passed_ids_key})...")
        passed_ids = get_passed_ids_from_summary(summary_file, passed_ids_key)
        if not passed_ids:
            print(f"Warning: No passed_ids found for {bench_name}. Skipping.")
            continue
        print(f"Loaded {len(passed_ids)} passed_ids for {bench_name}")

        if args.num_samples is not None and args.num_samples < len(passed_ids):
            samples_to_process = list(passed_ids)[:args.num_samples]
            print(f"Limited to first {args.num_samples} samples")
        else:
            samples_to_process = list(passed_ids)

        # Load dataset + image dir
        if bench_name == "mmvp":
            dataset = load_mmvp_dataset()
            image_dir = MMVP_IMAGE_DIR
            key_fn = lambda x: x['question_id']
        elif bench_name == "vstar":
            dataset = load_vstar_dataset()
            image_dir = VSTAR_DATA_DIR
            key_fn = lambda x: int(x['question_id'])
        else:  # mmstar
            dataset = load_mmstar_dataset()
            image_dir = MMSTAR_DATA_DIR
            key_fn = lambda x: x['question_id']

        print(f"{bench_name.upper()}: {len(dataset)} samples in dataset")
        dataset_dict = {key_fn(item): item for item in dataset}

        filtered_dataset = [dataset_dict[qid] for qid in samples_to_process if qid in dataset_dict]
        print(f"Testing on {len(filtered_dataset)} samples (model-correct only)")

        results, sc, fc = run_attack_on_dataset(
            model=model,
            processor=processor,
            dataset=filtered_dataset,
            image_dir=image_dir,
            out_dir=bench_output_dir,
            bench_name=bench_name
        )

        all_results.append({
            'benchmark': bench_name,
            'results': results,
            'attack_success_count': sc,
            'attack_fail_count': fc
        })

    # Save combined summary
    combined_summary = {
        'config': {
            'max_attempts': MAX_ATTEMPTS,
            'epsilon': EPSILON,
            'epsilon_pixel': int(EPSILON * 255),
            'seed': SEED,
            'model_path': MODEL_PATH,
        },
        'datasets': {}
    }
    for result in all_results:
        bench_name = result['benchmark']
        combined_summary['datasets'][bench_name] = {
            'total_samples': len(result['results']),
            'attack_success_count': result['attack_success_count'],
            'attack_fail_count': result['attack_fail_count'],
            'attack_success_rate': result['attack_success_count'] / len(result['results']) if result['results'] else 0
        }

    summary_path = os.path.join(OUTPUT_DIR, "attack_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(combined_summary, f, indent=2)

    print("\n" + "=" * 64)
    print("Final Attack Summary")
    print("=" * 64)
    for result in all_results:
        bench_name = result['benchmark']
        total = len(result['results'])
        sc = result['attack_success_count']
        rate = sc / total if total else 0
        print(f"\n{bench_name.upper()}:")
        print(f"  Total samples: {total}")
        print(f"  Attack success: {sc}")
        print(f"  Attack failed:  {result['attack_fail_count']}")
        print(f"  Success rate:   {rate * 100:.2f}%")

    print(f"\nAll results saved to: {os.path.abspath(summary_path)}")
    print("\nDone!")


if __name__ == "__main__":
    main()