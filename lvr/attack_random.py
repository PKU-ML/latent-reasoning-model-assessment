"""
Random Noise Attack on LVR-7B model.

攻击方法：
- 对图像添加随机扰动，最大像素差为 +-epsilon/255
- 测试模型在扰动图片上能否做对题目
- 如果某个扰动导致模型做错，保存该图片并提前结束
- 如果所有扰动都未导致模型做错，保存最后一张图片

支持数据集：MMVP 和 VSTAR
保存目录：random_images/
图片格式：{question_id}_adv.png

用法：
  python attack_random_noise.py --dataset all       # 攻击所有数据集
  python attack_random_noise.py --dataset mmvp     # 只攻击 MMVP
  python attack_random_noise.py --dataset vstar    # 只攻击 VSTAR
  python attack_random_noise.py --dataset vstar --num_samples 20
"""

import os
from pathlib import Path
import torch
import json
import csv
import argparse
from PIL import Image
import numpy as np
from tqdm import tqdm

from transformers import AutoTokenizer, AutoProcessor, AutoConfig
from src.model.qwen_lvr_model import QwenWithLVR
from src.train.monkey_patch_forward_lvr import replace_qwen2_5_with_mixed_modality_forward_lvr
from qwen_vl_utils import process_vision_info

# ==== Config ====
MODEL_PATH = os.environ.get("LVR_MODEL_PATH", "vincentleebang/LVR-7B")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据集路径全部走 DATA_DIR + 相对子目录，避免硬编码绝对路径
_DATA_ROOT = os.environ.get("DATA_DIR", str(Path(__file__).resolve().parent.parent / "data"))
MMVP_IMAGE_DIR = os.environ.get("LVR_MMVP_IMAGE_DIR", str(Path(_DATA_ROOT) / "MMVP" / "MMVP Images"))
MMVP_CSV = os.environ.get("LVR_MMVP_CSV", str(Path(_DATA_ROOT) / "MMVP" / "Questions.csv"))

VSTAR_DATA_DIR = os.environ.get("LVR_VSTAR_DATA_DIR", str(Path(_DATA_ROOT) / "vstar"))
VSTAR_JSONL = os.environ.get("LVR_VSTAR_JSONL", str(Path(_DATA_ROOT) / "vstar" / "test_questions.jsonl"))

MMSTAR_DATA_DIR = os.environ.get("LVR_MMSTAR_DATA_DIR", str(Path(_DATA_ROOT) / "MMStar"))
MMSTAR_METADATA = os.environ.get("LVR_MMSTAR_METADATA", str(Path(_DATA_ROOT) / "MMStar" / "metadata.json"))

OUTPUT_DIR = os.environ.get("LVR_OUTPUT_DIR", "random_images")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Random noise attack parameters
EPSILON = 4 / 255.0  # Maximum pixel perturbation
MAX_ATTEMPTS = 20    # Maximum number of random perturbations to try per image
SEED = 42            # Random seed for reproducibility

# Summary file path for loading passed_ids
MMVP_SUMMARY_PATH = os.environ.get(
    "LVR_MMVP_SUMMARY_PATH",
    "test_results/org/run_steps/summary.json",
)
VSTAR_SUMMARY_PATH = os.environ.get(
    "LVR_VSTAR_SUMMARY_PATH",
    "test_results/org/run_steps/summary.json",
)
MMSTAR_SUMMARY_PATH = os.environ.get(
    "LVR_MMSTAR_SUMMARY_PATH",
    "test_results/org/run_steps/summary.json",
)


def accuracy_reward(response: str, ground_truth: str) -> float:
    """Check if the response matches the ground truth answer."""
    given_answer = response.split('<answer>')[-1]
    given_answer = given_answer.split('</answer')[0].strip()
    if " " in given_answer:
        given_answer = given_answer.split(" ")[0]
    if len(given_answer) > 1:
        given_answer = given_answer[0]
    return given_answer == ground_truth


def extract_answer_tagged(response: str) -> str:
    """Extract answer from model response."""
    given = response.split('<answer>')[-1].split('</answer>')[0].strip()
    if " " in given:
        given = given.split(" ")[0]
    if len(given) > 1:
        given = given[0]
    return given


def get_task_instruction():
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


def run_inference(model, processor, img_path, text, steps=4, decoding_strategy="steps"):
    """Run inference on a single sample."""
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

    lvr_steps = [steps]
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=16,
            decoding_strategy=decoding_strategy,
            lvr_steps=lvr_steps
        )
        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False
        )
    return output_text


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
    """Load passed_ids from summary.json."""
    if not os.path.exists(summary_path):
        print(f"Warning: Summary file not found: {summary_path}")
        return None
    with open(summary_path, 'r') as f:
        summary = json.load(f)
    if passed_ids_key in summary:
        for step_key, result in summary[passed_ids_key].items():
            if "passed_ids" in result:
                return set(result["passed_ids"])
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
    task_instruction = get_task_instruction()
    if bench_name == "mmvp":
        text = question.replace('(a)', 'A.').replace('(b)', 'B.')
        text = text + task_instruction
    elif bench_name == "mmstar":
        text = question + task_instruction
    else:  # vstar
        text = question + task_instruction

    outputs = run_inference(model, processor, img_path, text)
    prediction = outputs[0]
    answer = extract_answer_tagged(prediction)

    is_correct = accuracy_reward(prediction, label)
    return is_correct, answer


def test_perturbed_image(model, processor, perturbed_img: Image.Image, question, label, bench_name="mmvp"):
    """Test if model gives correct answer on perturbed image."""
    import tempfile

    task_instruction = get_task_instruction()
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
        outputs = run_inference(model, processor, tmp_path, text)
        prediction = outputs[0]
        answer = extract_answer_tagged(prediction)

        is_correct = accuracy_reward(prediction, label)
        return is_correct, answer
    finally:
        # Clean up temp file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def run_attack_on_dataset(model, processor, dataset, image_dir, out_dir, bench_name):
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
    dataset.sort(key = lambda x : int(x['question_id']))
    print([i['question_id'] for i in dataset])
    for item in tqdm(dataset, desc=f"Attacking {bench_name}"):
        question_id = item['question_id']

        # Get image path and question based on bench_name
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
                break

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

    return results, attack_success_count, attack_fail_count


def main():
    parser = argparse.ArgumentParser(description="Random Noise Attack on LVR-7B")
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
                        help="Output directory for adversarial images")
    args = parser.parse_args()

    global OUTPUT_DIR, MAX_ATTEMPTS, EPSILON, SEED

    MAX_ATTEMPTS = args.max_attempts
    EPSILON = args.epsilon
    SEED = args.seed
    OUTPUT_DIR = args.output_dir
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 64)
    print("Random Noise Attack on LVR-7B")
    print("=" * 64)
    print(f"Dataset: {args.dataset}")
    print(f"Max attempts per image: {MAX_ATTEMPTS}")
    print(f"Epsilon (max pixel perturbation): {EPSILON:.6f} ({int(EPSILON*255)}/255)")
    print(f"Random seed: {SEED}")
    print(f"Output directory: {OUTPUT_DIR}")

    # Set random seed
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # Load model
    print("\nLoading model...")
    model, processor = load_model_and_processor()
    model.eval()
    print("Model loaded")

    # Determine which datasets to process
    datasets_to_process = []
    if args.dataset == "all":
        datasets_to_process = ["mmvp", "vstar", "mmstar"]
    else:
        datasets_to_process = [args.dataset]

    all_results = []

    for bench_name in datasets_to_process:
        # Determine summary path and passed_ids key based on dataset type
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
        filtered_dataset = [dataset_dict[qid] for qid in samples_to_process if qid in dataset_dict]
        print(f"Testing on {len(filtered_dataset)} samples (model-correct only)")

        # Run attack
        results, attack_success_count, attack_fail_count = run_attack_on_dataset(
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
            'attack_success_count': attack_success_count,
            'attack_fail_count': attack_fail_count
        })

    # Save combined summary
    combined_summary = {
        'config': {
            'max_attempts': MAX_ATTEMPTS,
            'epsilon': EPSILON,
            'epsilon_pixel': int(EPSILON * 255),
            'seed': SEED,
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

    # Print final summary
    print("\n" + "=" * 64)
    print("Final Attack Summary")
    print("=" * 64)
    for result in all_results:
        bench_name = result['benchmark']
        print(f"\n{bench_name.upper()}:")
        print(f"  Total samples: {len(result['results'])}")
        print(f"  Attack success: {result['attack_success_count']}")
        print(f"  Attack failed: {result['attack_fail_count']}")
        rate = result['attack_success_count'] / len(result['results']) if result['results'] else 0
        print(f"  Success rate: {rate * 100:.2f}%")

    print(f"\nAll results saved to: {os.path.abspath(summary_path)}")
    print("\nDone!")


if __name__ == "__main__":
    main()
