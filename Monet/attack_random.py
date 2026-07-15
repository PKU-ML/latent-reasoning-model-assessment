"""
Random Noise Attack on Monet-7B model.

攻击方法：
- 对图像添加随机扰动，最大像素差为 +-epsilon/255
- 测试模型在扰动图片上能否做对题目
- 如果某个扰动导致模型做错，保存该图片并提前结束
- 如果所有扰动都未导致模型做错，保存最后一张图片

保存目录：random_images_monet/
图片格式：{question_id}_adv.png
"""

import os
from pathlib import Path
import sys
import torch
import json
import csv
import argparse
import gc
from PIL import Image
import numpy as np
from tqdm import tqdm
import re

# Add Monet src to path
monet_src_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
sys.path.insert(0, monet_src_path)

# Apply monkey patch for latent token support
from src.train.monkey_patch_forward_monet_test import replace_qwen2_5_with_monet_forward
replace_qwen2_5_with_monet_forward()

from transformers import AutoProcessor, AutoConfig
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
from src.model.monet import MonetModel

# ==== Config ====
MODEL_PATH = os.environ.get("MONET_MODEL_PATH", "NOVAglow646/Monet-7B")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_DATA_ROOT = os.environ.get("DATA_DIR", str(Path(__file__).resolve().parent.parent / "data"))
MMVP_IMAGE_DIR = os.environ.get("MONET_MMVP_IMAGE_DIR", str(Path(_DATA_ROOT) / "MMVP" / "MMVP Images"))
MMVP_CSV = os.environ.get("MONET_MMVP_CSV", str(Path(_DATA_ROOT) / "MMVP" / "Questions.csv"))

VSTAR_DATA_DIR = os.environ.get("MONET_VSTAR_DATA_DIR", str(Path(_DATA_ROOT) / "vstar"))
VSTAR_JSONL = os.environ.get("MONET_VSTAR_JSONL", str(Path(_DATA_ROOT) / "vstar" / "test_questions.jsonl"))

MMSTAR_DATA_DIR = os.environ.get("MONET_MMSTAR_DATA_DIR", str(Path(_DATA_ROOT) / "MMStar"))
MMSTAR_METADATA = os.environ.get("MONET_MMSTAR_METADATA", str(Path(_DATA_ROOT) / "MMStar" / "metadata.json"))

OUTPUT_DIR = os.environ.get("MONET_OUTPUT_DIR", "random_images_monet")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Random noise attack parameters
EPSILON = 4 / 255.0  # Maximum pixel perturbation
MAX_ATTEMPTS = 20    # Maximum number of random perturbations to try per image
SEED = 42            # Random seed for reproducibility
MAX_LATENT_STEPS = int(os.environ.get("MAX_LATENT_STEPS", "10"))

# Summary file path for loading passed_ids
MMVP_SUMMARY_PATH = "test_results/org/monet_latent010/summary.json"
VSTAR_SUMMARY_PATH = "test_results/org/monet_latent010/summary.json"
MMSTAR_SUMMARY_PATH = "test_results/org/monet_latent010/summary.json"


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
        if len(first_line) == 1 and first_line in 'ABCD':
            return first_line
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
            after = response[idx + len(phrase):].strip()
            m = re.match(r'\(([A-Z])\)', after)
            if m:
                return m.group(1)
            m = re.match(r'([A-Z])\.?', after)
            if m:
                return m.group(1)

    # Look for standalone single letter A/B/C/D anywhere in response
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

    print(f"latent_token_id: {model.config.latent_token_id}")
    print(f"max_latent_steps: {model.config.max_latent_steps}")

    model.eval()
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        print("Gradient checkpointing enabled")

    if hasattr(model, 'visual'):
        model.visual_requires_grad = False
        print("Vision encoder available for CPU offloading if needed")

    return model, processor


@torch.no_grad()
def run_inference(model, processor, img_path, text, max_new_tokens=512):
    """
    Run inference using MonetModel's custom _sample method for latent reasoning.
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

    input_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask")

    # Insert <abs_vis_token> after assistant token to trigger latent reasoning
    im_start_id = 151644
    assistant_id = 77091
    latent_token_id = model.config.latent_token_id

    input_ids_list = input_ids[0].tolist()
    input_ids = input_ids[0].tolist()
    input_ids.append(latent_token_id)
    insert_pos = len(input_ids) - 1
    input_ids = torch.tensor([input_ids], dtype=torch.long, device=DEVICE)
    attention_mask = torch.tensor([[1] * len(input_ids[0])], dtype=torch.long, device=DEVICE)

    # Use standard generate (uses MonetModel's custom _sample)
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

    # Decode response
    trimmed_ids = generated_ids[0][len(input_ids[0]):]
    output_text = processor.tokenizer.decode(trimmed_ids, skip_special_tokens=True)
    cleaned_output = replace_abs_vis_token_content(output_text)

    return cleaned_output


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


def get_passed_ids_from_summary(summary_path, passed_ids_key="mmvp_result"):
    """Load passed_ids from summary.json."""
    if not os.path.exists(summary_path):
        print(f"Warning: Summary file not found: {summary_path}")
        return None
    with open(summary_path, 'r') as f:
        summary = json.load(f)
    if passed_ids_key in summary and "passed_ids" in summary[passed_ids_key]:
        return set(summary[passed_ids_key]["passed_ids"])
    print(f"Warning: No passed_ids found for '{passed_ids_key}' in {summary_path}")
    return None


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
    answer = extract_answer(prediction)

    is_correct = accuracy_reward(prediction, label)
    return is_correct, answer


def test_perturbed_image(model, processor, perturbed_img: Image.Image, question, label, bench_name="mmvp"):
    """Test if model gives correct answer on perturbed image."""
    import tempfile

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
        answer = extract_answer(prediction)

        is_correct = accuracy_reward(prediction, label)
        return is_correct, answer
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def run_attack_on_dataset(model, processor, dataset, image_dir, out_dir, bench_name, max_new_tokens=2):
    """Run random noise attack on a dataset."""
    print(f"\n{'='*64}")
    print(f"Running attack on {bench_name.upper()} dataset")
    print(f"{'='*64}")

    os.makedirs(out_dir, exist_ok=True)

    print(f"Total samples to attack: {len(dataset)}")

    # Results storage
    results = []
    attack_success_count = 0
    attack_fail_count = 0

    print("\nStarting attack...")

    for item in tqdm(dataset, desc=f"Attacking {bench_name}"):
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

        # Try random perturbations
        original_img = Image.open(img_path).convert('RGB')
        attack_success = False
        final_perturbed_img = None
        wrong_answer = None
        attempts_made = 0

        for attempt in range(MAX_ATTEMPTS):
            attempts_made = attempt + 1

            # Generate perturbed image
            perturbed_img = add_random_noise(original_img, epsilon=EPSILON)

            # Test on perturbed image
            is_correct, answer = test_perturbed_image(
                model, processor, perturbed_img, question, label, bench_name
            )

            final_perturbed_img = perturbed_img

            if not is_correct:
                # Attack success! Model answered incorrectly
                attack_success = True
                wrong_answer = answer
                attack_success_count += 1
                print(f"\n[Success] Question {question_id}: Attack successful at attempt {attempts_made}")
                print(f"          GT={label}, Wrong answer={answer}")
                #break

        if not attack_success:
            attack_fail_count += 1
            print(f"\n[Fail] Question {question_id}: No perturbation caused error after {MAX_ATTEMPTS} attempts")

        # Save the image (last perturbed image, whether successful or not)
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

    # Summary statistics
    print("\n" + "=" * 64)
    print(f"Attack Summary for {bench_name.upper()}")
    print("=" * 64)
    print(f"Total samples tested: {len(results)}")
    print(f"Attack success (model wrong): {attack_success_count}")
    print(f"Attack failed (model still correct): {attack_fail_count}")
    print(f"Attack success rate: {attack_success_count / len(results) * 100:.2f}%" if results else "N/A")
    print(f"\nImages saved to: {os.path.abspath(out_dir)}")

    # Save results JSON
    results_summary = {
        'config': {
            'model_path': MODEL_PATH,
            'max_attempts': MAX_ATTEMPTS,
            'epsilon': EPSILON,
            'epsilon_pixel': int(EPSILON * 255),
            'seed': SEED,
            'max_latent_steps': MAX_LATENT_STEPS,
            'max_new_tokens': max_new_tokens,
            'benchmark': bench_name,
            'total_samples': len(results),
            'attack_success_count': attack_success_count,
            'attack_fail_count': attack_fail_count,
            'attack_success_rate': attack_success_count / len(results) if results else 0
        },
        'results': results
    }

    summary_path = os.path.join(out_dir, "attack_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(results_summary, f, indent=2)
    print(f"\nSummary saved to: {os.path.abspath(summary_path)}")

    return results_summary


def main():
    parser = argparse.ArgumentParser(description="Random Noise Attack on Monet-7B")
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
    parser.add_argument("--output_dir", type=str, default="random_images_monet",
                        help="Output directory for adversarial images")
    parser.add_argument("--max_new_tokens", type=int, default=2,
                        help="Maximum number of tokens to generate (default: 2)")
    parser.add_argument("--max_latent_steps", type=int, default=10,
                        help="Maximum number of latent reasoning steps (default: 10)")
    args = parser.parse_args()

    global OUTPUT_DIR, MAX_ATTEMPTS, EPSILON, SEED, MAX_LATENT_STEPS

    MAX_ATTEMPTS = args.max_attempts
    EPSILON = args.epsilon
    SEED = args.seed
    MAX_LATENT_STEPS = args.max_latent_steps
    OUTPUT_DIR = args.output_dir
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 64)
    print("Random Noise Attack on Monet-7B")
    print("=" * 64)
    print(f"Dataset: {args.dataset}")
    print(f"Max attempts per image: {MAX_ATTEMPTS}")
    print(f"Epsilon (max pixel perturbation): {EPSILON:.6f} ({int(EPSILON*255)}/255)")
    print(f"Random seed: {SEED}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Max new tokens: {args.max_new_tokens}")
    print(f"Max latent steps: {MAX_LATENT_STEPS}")

    # Set random seed
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # Load model
    print("\nLoading Monet model...")
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
        # Determine summary path and passed_ids key based on dataset type
        if bench_name == "vstar":
            summary_file = args.summary_path if args.summary_path else VSTAR_SUMMARY_PATH
            passed_ids_key = "vstar_result"
            output_subdir = "vstar"
        elif bench_name == "mmstar":
            summary_file = args.summary_path if args.summary_path else MMSTAR_SUMMARY_PATH
            passed_ids_key = "mmstar_result"
            output_subdir = "mmstar"
        else:
            summary_file = args.summary_path if args.summary_path else MMVP_SUMMARY_PATH
            passed_ids_key = "mmvp_result"
            output_subdir = "mmvp"

        # Get output subdirectory for this benchmark
        bench_output_dir = os.path.join(OUTPUT_DIR, output_subdir)
        os.makedirs(bench_output_dir, exist_ok=True)

        # Load passed_ids
        print(f"\nLoading passed_ids from {summary_file}...")
        passed_ids = get_passed_ids_from_summary(summary_file, passed_ids_key)
        if not passed_ids:
            print(f"Warning: No passed_ids found for {bench_name}. Skipping.")
            continue
        print(f"Loaded {len(passed_ids)} passed_ids for {bench_name}")

        # Limit samples if specified
        samples_to_process = passed_ids
        if args.num_samples is not None and args.num_samples < len(passed_ids):
            samples_to_process = list(passed_ids)[:args.num_samples]
            print(f"Limited to first {args.num_samples} samples")
        else:
            samples_to_process = list(passed_ids)

        # Load dataset and image directory
        if bench_name == "mmvp":
            print(f"\nLoading {bench_name} dataset...")
            dataset = load_mmvp_dataset()
            image_dir = MMVP_IMAGE_DIR
            dataset_dict = {item['question_id']: item for item in dataset}
        elif bench_name == "mmstar":
            print(f"\nLoading {bench_name} dataset...")
            dataset = load_mmstar_dataset()
            image_dir = MMSTAR_DATA_DIR
            dataset_dict = {item['question_id']: item for item in dataset}
        else:  # vstar
            print(f"\nLoading {bench_name} dataset...")
            dataset = load_vstar_dataset()
            image_dir = VSTAR_DATA_DIR
            dataset_dict = {int(item['question_id']): item for item in dataset}

        print(f"{bench_name.upper()}: {len(dataset)} samples in dataset")

        # Filter to passed_ids only
        filtered_dataset = [dataset_dict[qid] for qid in samples_to_process if int(qid) in dataset_dict]
        print(f"Testing on {len(filtered_dataset)} samples (model-correct only)")

        # Run attack
        run_attack_on_dataset(
            model=model,
            processor=processor,
            dataset=filtered_dataset,
            image_dir=image_dir,
            out_dir=bench_output_dir,
            bench_name=bench_name,
            max_new_tokens=args.max_new_tokens
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
    print("\nDone!")


if __name__ == "__main__":
    main()
