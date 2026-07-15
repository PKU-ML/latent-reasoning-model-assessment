"""
Random Noise Attack on MMVP and VSTAR datasets for SkiLa model.

攻击方法：
- 对图像添加随机扰动，最大像素差为 +-epsilon
- 测试模型在扰动图片上能否做对题目
- 每个样本完整跑完 MAX_ATTEMPTS 次扰动（无早停机制）
- 每个样本保存最后一张扰动图片

保存目录：random_images/mmvp/ 和 random_images/vstar/
图片格式：{question_id}_adv.png

结果格式：
- attack_success: 该样本在MAX_ATTEMPTS次中是否至少成功一次
- success_count: 成功次数（有多少次扰动导致模型答错）
- wrong_answers: 所有成功时模型给出的错误答案列表
- first_success_attempt: 第一次成功的attempt编号（1-indexed）
"""

import os
from pathlib import Path
import sys
import re
import torch
import json
import csv
import argparse
import tempfile
import gc
from PIL import Image
import numpy as np
from tqdm import tqdm

# Set CUDA device before any imports
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Add SkiLa src to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

# Set HuggingFace cache to local path

# Apply monkey patch to enable SkiLa sketch mode
from src.train.monkey_patch_forward_skila_test import replace_qwen2_5_with_skila_forward
replace_qwen2_5_with_skila_forward()

from transformers import AutoProcessor, AutoConfig, SiglipImageProcessor
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

from src.model.skila import SkiLa

# ==== Config ====
MODEL_PATH = os.environ.get("SKILA_MODEL_PATH", "./SkiLa-7B")
SKETCH_ENCODER = os.environ.get(
    "SKILA_SKETCH_ENCODER",
    "./models--google--siglip2-so400m-patch14-384/snapshots/e8e487298228002f3d8a82e0cd5c8ea9c567f57f",
)
SKETCH_TOKEN_NUM = 54

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_DATA_ROOT = os.environ.get("DATA_DIR", str(Path(__file__).resolve().parent.parent / "data"))
# MMVP dataset paths
MMVP_IMAGE_DIR = os.environ.get("SKILA_MMVP_IMAGE_DIR", str(Path(_DATA_ROOT) / "MMVP" / "MMVP Images"))
MMVP_CSV = os.environ.get("SKILA_MMVP_CSV", str(Path(_DATA_ROOT) / "MMVP" / "Questions.csv"))

VSTAR_DATA_DIR = os.environ.get("SKILA_VSTAR_DATA_DIR", str(Path(_DATA_ROOT) / "vstar"))
VSTAR_JSONL = os.environ.get("SKILA_VSTAR_JSONL", str(Path(_DATA_ROOT) / "vstar" / "test_questions.jsonl"))

MMSTAR_DATA_DIR = os.environ.get("SKILA_MMSTAR_DATA_DIR", str(Path(_DATA_ROOT) / "MMStar"))
MMSTAR_METADATA = os.environ.get("SKILA_MMSTAR_METADATA", str(Path(_DATA_ROOT) / "MMStar" / "metadata.json"))

# Output directory (parent directory for mmvp and vstar subdirectories)
OUTPUT_DIR = os.environ.get("SKILA_OUTPUT_DIR", "random_images")

# Random noise attack parameters
EPSILON = 4 / 255.0  # Maximum pixel perturbation
MAX_ATTEMPTS = 60    # Maximum number of random perturbations to try per image
SEED = 42            # Random seed for reproducibility

# Summary file path for loading passed_ids
SUMMARY_PATH = "test_results/org/summary.json"
MMSTAR_SUMMARY_PATH = "test_results/org/summary.json"


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


def get_task_instruction(bench_name="mmvp"):
    return "\nAnswer with the option's letter from the given choices directly."


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

    return model, processor


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
def run_inference(model, processor, img_path, text, max_new_tokens=512):
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

    # Decode response - skip the forced sketch_start token
    original_len = inputs["input_ids"].shape[1]
    trimmed_ids = generated_ids[0][original_len:]
    output_text = processor.tokenizer.decode(trimmed_ids, skip_special_tokens=True)

    return output_text


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


def get_passed_ids_from_summary(summary_path, bench_name):
    """Load passed_ids from summary.json for specified benchmark.

    Args:
        summary_path: path to summary.json
        bench_name: dataset name ("mmvp", "vstar", or "mmstar")

    Returns:
        passed_ids list, or None if not found
    """
    if not os.path.exists(summary_path):
        print(f"Warning: Summary file not found: {summary_path}")
        return None
    with open(summary_path, 'r') as f:
        summary = json.load(f)

    # Directly access the result using the benchmark name + "_result" suffix
    data_key = f"{bench_name}_result"
    if data_key not in summary:
        print(f"Warning: '{data_key}' not found in {summary_path}")
        return None

    passed_ids = summary[data_key].get("passed_ids")
    if passed_ids is None:
        print(f"Warning: 'passed_ids' not found in {data_key} of {summary_path}")
        return None

    return passed_ids


def add_random_noise(image: Image.Image, epsilon: float = 4/255.0) -> Image.Image:
    """
    Add random noise to image within epsilon bounds.
    image: PIL Image in RGB
    epsilon: maximum perturbation per pixel (in [0, 1] scale)
    Returns: perturbed PIL Image
    """
    img_array = np.array(image).astype(np.float32) / 255.0

    # Generate random noise in [-epsilon, epsilon]
    noise = np.random.uniform(-epsilon, epsilon, img_array.shape).astype(np.float32)

    # Add noise and clamp to [0, 1]
    perturbed = np.clip(img_array + noise, 0, 1)

    # Convert back to uint8
    perturbed_uint8 = (perturbed * 255).astype(np.uint8)

    return Image.fromarray(perturbed_uint8)


def verify_original_correct(model, processor, img_path, question, label, bench_name="mmvp"):
    """Verify that the model gives the correct answer on the original image."""
    task_instruction = get_task_instruction(bench_name)

    if bench_name == "mmvp":
        text = question.replace('(a)', 'A.').replace('(b)', 'B.')
        text = text + task_instruction
    elif bench_name == "mmstar":
        text = question + task_instruction
    else:  # vstar
        text = question + task_instruction

    prediction = run_inference(model, processor, img_path, text)
    is_correct = accuracy_reward(prediction, label)
    answer = extract_answer(prediction)

    return is_correct, answer


def test_perturbed_image(model, processor, perturbed_img: Image.Image, question, label, bench_name="mmvp"):
    """Test if model gives correct answer on perturbed image."""
    task_instruction = get_task_instruction(bench_name)

    if bench_name == "mmvp":
        text = question.replace('(a)', 'A.').replace('(b)', 'B.')
        text = text + task_instruction
    elif bench_name == "mmstar":
        text = question + task_instruction
    else:  # vstar
        text = question + task_instruction

    # Save perturbed image to a temporary file
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        tmp_path = tmp.name

    try:
        perturbed_img.save(tmp_path)
        prediction = run_inference(model, processor, tmp_path, text)
        is_correct = accuracy_reward(prediction, label)
        answer = extract_answer(prediction)
        return is_correct, answer
    finally:
        # Clean up temp file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def run_attack_on_dataset(model, processor, dataset, image_dir, out_dir, bench_name,
                           filter_ids=None, num_samples=None):
    """
    Run random noise attack on a dataset.
    For MMVP: uses query field, image from MMVP_IMAGE_DIR
    For VSTAR: uses text field, image from VSTAR_DATA_DIR
    """
    print(f"\n{'='*64}")
    print(f"Running attack on {bench_name.upper()} dataset")
    print(f"{'='*64}")

    os.makedirs(out_dir, exist_ok=True)

    # Filter to passed_ids only if specified
    if filter_ids is not None:
        if bench_name == "mmvp":
            filtered_dataset = [item for item in dataset if item['question_id'] in filter_ids]
        elif bench_name == "mmstar":
            filtered_dataset = [item for item in dataset if item['question_id'] in filter_ids]
        else:  # vstar
            filtered_dataset = [item for item in dataset if int(item['question_id']) in filter_ids]
        print(f"Filtered to {len(filtered_dataset)} samples from passed_ids")
    else:
        filtered_dataset = dataset

    # Limit samples if specified
    if num_samples is not None and num_samples < len(filtered_dataset):
        filtered_dataset = filtered_dataset[:num_samples]
        print(f"Limited to first {num_samples} samples")

    print(f"Total samples to attack: {len(filtered_dataset)}")

    # Results storage
    results = []
    attack_success_count = 0
    attack_fail_count = 0

    print("\nStarting attack...")

    for item in tqdm(filtered_dataset, desc=f"Attacking {bench_name}"):
        question_id = item['question_id']

        # Get image path based on bench_name
        if bench_name == "mmvp":
            img_path = os.path.join(image_dir, item['image'])
            question = item['query']
        elif bench_name == "mmstar":
            img_path = os.path.join(image_dir, item['image'])
            question = item['query']
        else:  # vstar
            img_path = os.path.join(image_dir, item['image'])
            question = item['text']

        label = item['label']

        # Verify original image is correct
        original_correct, original_answer = verify_original_correct(
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

        # Try random perturbations (run ALL attempts, no early stopping)
        original_img = Image.open(img_path).convert('RGB')
        success_count = 0
        first_success_attempt = None
        wrong_answers = []
        # Save the perturbation that FIRST caused the model to give a wrong answer,
        # so that re-testing this saved image reproduces the attack. If no attempt
        # succeeds, leave this as None (no adversarial image is saved).
        saved_perturbed_img = None
        # last_seen_perturbed_img is only kept as a fallback in case the saved
        # directory is missing — it is never written to disk in the success path.
        last_seen_perturbed_img = None

        for attempt in range(MAX_ATTEMPTS):
            # Generate perturbed image
            perturbed_img = add_random_noise(original_img, epsilon=EPSILON)

            # Test on perturbed image
            is_correct, answer = test_perturbed_image(
                model, processor, perturbed_img, question, label, bench_name
            )

            last_seen_perturbed_img = perturbed_img

            if not is_correct:
                # Attack succeeded on this attempt
                success_count += 1
                wrong_answers.append(answer)
                # Remember the FIRST successful perturbation so the saved image
                # actually reproduces the attack when re-tested later.
                if first_success_attempt is None:
                    first_success_attempt = attempt + 1
                    saved_perturbed_img = perturbed_img

        # Determine if attack succeeded (at least once in all attempts)
        attack_success = success_count > 0
        if attack_success:
            attack_success_count += 1
            print(f"\n[Result] Question {question_id}: succeeded {success_count}/{MAX_ATTEMPTS} times, first at attempt {first_success_attempt}")
        else:
            attack_fail_count += 1
            print(f"\n[Result] Question {question_id}: failed all {MAX_ATTEMPTS} attempts")

        # Save the FIRST successful perturbed image so test.py can reproduce it.
        # If no attempt succeeded, fall back to the last perturbed image so that
        # downstream tools (e.g. test.py) still have an image to load — that
        # image just won't reproduce an attack, which is fine because the
        # attack_success flag already records the failure.
        save_path = os.path.join(out_dir, f"{question_id}_adv.png")
        if saved_perturbed_img is not None:
            saved_perturbed_img.save(save_path)
        elif last_seen_perturbed_img is not None:
            last_seen_perturbed_img.save(save_path)

        results.append({
            'question_id': question_id,
            'gt': label,
            'original_answer': original_answer,
            'attack_success': attack_success,
            'success_count': success_count,
            'first_success_attempt': first_success_attempt,
            'wrong_answers': wrong_answers,
            'total_attempts': MAX_ATTEMPTS,
            'saved_image': save_path
        })

    # Summary statistics
    print("\n" + "=" * 64)
    print(f"Attack Summary for {bench_name.upper()}")
    print("=" * 64)
    print(f"Total samples tested: {len(results)}")
    print(f"Attack success (at least one wrong answer): {attack_success_count}")
    print(f"Attack failed (all attempts correct): {attack_fail_count}")
    print(f"Attack success rate: {attack_success_count / len(results) * 100:.2f}% if len(results) > 0 else 0%")

    # Calculate total success count across all attempts
    total_successes = sum(r['success_count'] for r in results)
    total_attempts = len(results) * MAX_ATTEMPTS
    print(f"Total successful perturbations: {total_successes}/{total_attempts} ({total_successes/total_attempts*100:.2f}% if total_attempts > 0 else 0%)")

    print(f"\nImages saved to: {os.path.abspath(out_dir)}")

    # Save results JSON
    results_summary = {
        'config': {
            'model_path': MODEL_PATH,
            'sketch_encoder': SKETCH_ENCODER,
            'sketch_token_num': SKETCH_TOKEN_NUM,
            'benchmark': bench_name,
            'max_attempts': MAX_ATTEMPTS,
            'epsilon': EPSILON,
            'epsilon_pixel': int(EPSILON * 255),
            'seed': SEED,
            'total_samples': len(results),
            'attack_success_count': attack_success_count,
            'attack_fail_count': attack_fail_count,
            'attack_success_rate': attack_success_count / len(results) if results else 0,
            'total_successful_perturbations': total_successes,
            'total_possible_perturbations': total_attempts
        },
        'results': results
    }

    summary_path = os.path.join(out_dir, "attack_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(results_summary, f, indent=2)
    print(f"\nSummary saved to: {os.path.abspath(summary_path)}")

    return results_summary


def main():
    parser = argparse.ArgumentParser(description="Random Noise Attack on MMVP/VSTAR datasets for SkiLa")
    parser.add_argument("--dataset", type=str, default="all",
                        choices=["all", "mmvp", "vstar", "mmstar"],
                        help="Which dataset to attack: 'all', 'mmvp', 'vstar', or 'mmstar' (default: all)")
    parser.add_argument("--max_attempts", type=int, default=50,
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
                        help="Parent output directory for adversarial images")
    parser.add_argument("--max_new_tokens", type=int, default=1024,
                        help="Maximum number of tokens to generate (default: 1024)")
    args = parser.parse_args()

    global OUTPUT_DIR, MAX_ATTEMPTS, EPSILON, SEED, SUMMARY_PATH

    MAX_ATTEMPTS = args.max_attempts
    EPSILON = args.epsilon
    SEED = args.seed
    OUTPUT_DIR = args.output_dir

    summary_file = args.summary_path if args.summary_path else SUMMARY_PATH

    print("=" * 64)
    print("Random Noise Attack on MMVP/VSTAR datasets (SkiLa)")
    print("=" * 64)
    print(f"Model: {MODEL_PATH}")
    print(f"Dataset: {args.dataset}")
    print(f"Max attempts per image: {MAX_ATTEMPTS}")
    print(f"Epsilon (max pixel perturbation): {EPSILON:.6f} ({int(EPSILON*255)}/255)")
    print(f"Random seed: {SEED}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Summary path: {summary_file}")

    # Set random seed
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # Load model
    print("\nLoading model...")
    model, processor = load_model()
    model.eval()
    print("Model loaded")

    # Determine which datasets to process
    datasets_to_process = []
    if args.dataset == "all":
        datasets_to_process = ["mmvp", "vstar", "mmstar"]
    else:
        datasets_to_process = [args.dataset]

    for bench_name in datasets_to_process:
        # Get output subdirectory for this benchmark
        bench_output_dir = os.path.join(OUTPUT_DIR, bench_name)
        os.makedirs(bench_output_dir, exist_ok=True)

        # Load passed_ids
        print(f"\nLoading passed_ids from summary for {bench_name}...")
        passed_ids = get_passed_ids_from_summary(summary_file, bench_name)
        if not passed_ids:
            print(f"Warning: No passed_ids found for {bench_name}. Skipping.")
            continue
        print(f"Loaded {len(passed_ids)} passed_ids for {bench_name}")

        # Limit samples if specified
        samples_to_process = passed_ids
        if args.num_samples is not None and args.num_samples < len(passed_ids):
            samples_to_process = passed_ids[:args.num_samples]
            print(f"Limited to first {args.num_samples} samples")

        # Load dataset
        if bench_name == "mmvp":
            print(f"\nLoading {bench_name} dataset...")
            dataset = load_mmvp_dataset()
            image_dir = MMVP_IMAGE_DIR
        elif bench_name == "mmstar":
            print(f"\nLoading {bench_name} dataset...")
            dataset = load_mmstar_dataset()
            image_dir = MMSTAR_DATA_DIR
        else:  # vstar
            print(f"\nLoading {bench_name} dataset...")
            dataset = load_vstar_dataset()
            image_dir = VSTAR_DATA_DIR

        print(f"{bench_name.upper()}: {len(dataset)} samples in dataset")

        # Run attack
        run_attack_on_dataset(
            model=model,
            processor=processor,
            dataset=dataset,
            image_dir=image_dir,
            out_dir=bench_output_dir,
            bench_name=bench_name,
            filter_ids=samples_to_process,
            num_samples=None  # Already limited via filter_ids
        )

        # Clear GPU cache between datasets
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            gc.collect()
            print(f"GPU cache cleared after {bench_name}. "
                  f"Free memory: {torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")

    # Clear GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        gc.collect()

    print("\n" + "=" * 64)
    print("All attacks completed!")
    print(f"Results saved to: {os.path.abspath(OUTPUT_DIR)}")
    print("=" * 64)


if __name__ == "__main__":
    main()
