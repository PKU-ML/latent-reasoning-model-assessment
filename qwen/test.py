"""
Test script for Qwen2.5-VL-7B-Instruct on local MMVP, VSTAR and MMStar datasets.

This script mirrors the structure of `lvr/test.py`, but evaluates the vanilla
Qwen2.5-VL-7B-Instruct model (no LVR / monkey patches). The key differences are:

1. Model: ``Qwen/Qwen2.5-VL-7B-Instruct`` loaded via ``Qwen2_5_VLForConditionalGeneration``.
2. Prompt: the model is asked to reason step-by-step first, then output the
   final answer wrapped in ``<answer>...</answer>`` tags at the very end.
3. Output extraction: the result extractor only looks at the content inside
   the last ``<answer>...</answer>`` block, so the reasoning text never affects
   the parsing of the final letter answer.
"""

import sys
import os
from pathlib import Path
import re
import torch
import json
import csv
import argparse
from tqdm import tqdm

import numpy as np

from transformers import AutoTokenizer, AutoProcessor, AutoConfig
from transformers import Qwen2_5_VLForConditionalGeneration

from qwen_vl_utils import process_vision_info


# ==== Config ====
# 模型路径通过 QWEN_MODEL_PATH 环境变量注入；不设置时使用 HuggingFace 仓库名
MODEL_PATH = os.environ.get("QWEN_MODEL_PATH", "Qwen/Qwen2.5-VL-7B-Instruct")

# Auto-detect available GPUs from environment
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Local dataset paths — 走 DATA_DIR + 相对子目录
_DATA_ROOT = os.environ.get("DATA_DIR", str(Path(__file__).resolve().parent.parent / "data"))
MMVP_DATA_DIR = os.environ.get("QWEN_MMVP_DATA_DIR", str(Path(_DATA_ROOT) / "MMVP"))
MMVP_IMAGE_DIR = os.environ.get("QWEN_MMVP_IMAGE_DIR", str(Path(_DATA_ROOT) / "MMVP" / "MMVP Images"))
MMVP_CSV = os.environ.get("QWEN_MMVP_CSV", str(Path(_DATA_ROOT) / "MMVP" / "Questions.csv"))

VSTAR_DATA_DIR = os.environ.get("QWEN_VSTAR_DATA_DIR", str(Path(_DATA_ROOT) / "vstar"))
VSTAR_JSONL = os.environ.get("QWEN_VSTAR_JSONL", str(Path(_DATA_ROOT) / "vstar" / "test_questions.jsonl"))

MMSTAR_DATA_DIR = os.environ.get("QWEN_MMSTAR_DATA_DIR", str(Path(_DATA_ROOT) / "MMStar"))
MMSTAR_METADATA = os.environ.get("QWEN_MMSTAR_METADATA", str(Path(_DATA_ROOT) / "MMStar" / "metadata.json"))

# Output directory — 默认 <qwen>/results
OUTPUT_DIR = os.environ.get("QWEN_OUTPUT_DIR", str(Path(__file__).resolve().parent / "results"))

# Default summary paths for loading passed_ids. The summaries are written
# by this same script to <output_dir>/summary.json, where output_dir defaults
# to /root/autodl-tmp/attack/qwen/results/run_qwen_vl on the server.
MMVP_SUMMARY_PATH = "results/org/summary.json"
VSTAR_SUMMARY_PATH = "results/org/summary.json"
MMSTAR_SUMMARY_PATH = "results/org/summary.json"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------
# The model is asked to:
#   1. Reason step-by-step inside a <think>...</think> block.
#   2. End with the final answer wrapped in <answer>X</answer> tags.
# This guarantees that even when the model produces long reasoning, the
# extractor can always pull the final letter from the last <answer> block.
#
# The prompt is intentionally emphatic: Qwen2.5-VL-Instruct is NOT a native
# thinking model, so we have to coerce it into producing reasoning text by
# (a) putting a strong system-style instruction, (b) showing an explicit
# response format, and (c) using Qwen3-style <think> tags that the model is
# familiar with from related model variants.
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
    """Split the model output into (reasoning, answer_letter_or_None).

    - reasoning: the text inside the LAST ``<think>...</think>`` block, or the
      full response if no think block is found.
    - answer: the first character inside the LAST ``<answer>...</answer>``
      block, or ``None`` if no answer tag is found.
    """
    reasoning = response
    if "<think>" in response and "</think>" in response:
        # take the LAST <think>...</think> block
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
    """Check if the response matches the ground truth answer.

    Only looks at the content inside the LAST ``<answer>...</answer>`` block,
    so any preceding reasoning text is ignored. If the model outputs more than
    one character inside the tags (e.g. "A."), we only take the first letter.
    """
    _, answer = extract_reasoning_and_answer(response)
    if answer is None:
        return False
    return answer == ground_truth


def get_task_instruction(bench_name):
    if bench_name.lower() == "vstar":
        return ("\nAnswer with the option's letter from the given choices directly.")
    elif bench_name.lower() == "mmvp":
        return ("\nAnswer with the option's letter from the given choices directly.")
    else:
        return ("\nAnswer with the option's letter from the given choices directly.")


def _extract_clean_thinking(prediction: str) -> str:
    """Extract the body of the LAST ``<think>...</think>`` block from a model
    prediction, stripping any trailing ``Final answer: X`` line that some
    predictions embed inside the block.

    Returns the inner body text (no ``<think>`` / ``</think>`` tags). Returns
    an empty string when there is nothing safe to inject.
    """
    if not prediction:
        return ""
    matches = re.findall(r'<think>(.*?)</think>', prediction, re.DOTALL)
    if not matches:
        return ""
    body = matches[-1].strip()
    # Strip a trailing "Final answer: X" line (case-insensitive, optional dot).
    body = re.sub(
        r'\n*\s*final\s*answer\s*:\s*[A-D]\.?\s*$',
        '',
        body,
        flags=re.IGNORECASE,
    ).strip()
    return body


def load_thinking_from_json(json_path: str, dataset_name: str) -> dict:
    """Build a ``{(dataset_name, str(question_id)): '<think>\\n<body>\\n'}`` dict
    from a per-dataset ``*_results.json`` file (as written by
    ``evaluate_mmvp`` / ``evaluate_vstar`` / ``evaluate_mmstar``).

    Expected layout (one dataset only):
      {
        "accuracy": ..., "correct": ..., "total": ...,
        "passed_ids": [...],
        "results": [
          {"id": ..., "prediction": "<think>...</think><answer>X</answer>",
           "reasoning": ..., "extracted_answer": ..., "label": ..., ...},
          ...
        ]
      }

    Returns ``{('mmvp'|'vstar'|'mmstar', str(id)): '<think>\\n<body>\\n'}``.
    Each example's stored thinking is the LAST ``<think>...</think>`` body
    from its ``prediction`` field, with any trailing ``Final answer: X`` line
    already stripped (see ``_extract_clean_thinking``).
    """
    thinking_dict = {}
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # results.json format: top-level 'results' is a list of per-sample dicts.
    # We deliberately do NOT support the older summary.json three-dataset
    # layout here — pass --dataset to pick which dataset this file represents.
    if not isinstance(data, dict) or "results" not in data \
            or not isinstance(data["results"], list):
        print(f"Warning: {json_path} does not look like a per-dataset "
              f"results.json (missing top-level 'results' list); "
              f"thinking injection disabled.")
        return thinking_dict

    for ex in data["results"]:
        qid = ex.get("id")
        if qid is None:
            continue
        body = _extract_clean_thinking(ex.get("prediction", ""))
        if not body:
            continue
        thinking_dict[(dataset_name, str(qid))] = f"<think>\n{body}\n"
    return thinking_dict


def create_messages(img_path, question, system_prompt=None):
    """Build chat-template messages containing the image(s) and question text.

    If ``system_prompt`` is provided, it is inserted as a system message at
    the start of the conversation. This is used to force the model into a
    "always reason before answering" mode.
    """
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

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_content})
    return messages


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


def run_inference(model, processor, img_path, text, max_new_tokens=2048,
                  question_id=None, thinking_dict=None, dataset_name=None,
                  debug=False):
    """Run inference on a single sample.

    The ``text`` argument is the bare question text (multiple-choice question +
    task instruction). The function prepends the reasoning prompt and adds a
    system message to force step-by-step reasoning.

    If ``thinking_dict`` and ``dataset_name`` are provided, the stored
    ``<think>...</think>`` body for ``(dataset_name, question_id)`` (if any)
    is appended to the chat-templated prompt so the model continues from
    that thinking instead of reasoning from scratch. The image placeholder
    tokens emitted by ``apply_chat_template`` are unaffected; the appended
    thinking only adds new tokens after the assistant generation prompt.

    If ``debug`` is True, prints four sections per call: chat template
    before injection, the injected thinking, chat template after injection
    (i.e. what the model actually sees), and the model's raw output.

    Returns ``(model_output, injected_thinking)``:
      - ``model_output`` is whatever the model generated (typically
        ``\n</think>\n<answer>X</answer>`` after injected thinking).
      - ``injected_thinking`` is the ``<think>...</think>`` body that was
        appended to the prompt, or ``""`` if none. Callers can stitch the
        two together for debug logging.
    """
    full_user_text = REASONING_USER_PROMPT + text
    messages = create_messages(img_path, full_user_text,
                               system_prompt=REASONING_SYSTEM_PROMPT)
    text_formatted = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    if debug:
        # Snapshot the un-injected prompt so we can print it BEFORE we mutate
        # ``text_formatted`` below. We render both side-by-side for the user.
        text_formatted_before = text_formatted
        tag = "qid=%s dataset=%s" % (question_id, dataset_name)
        print("\n" + "#" * 72)
        print("# DEBUG %s" % tag)
        print("#" * 72)
        print("\n----- [1/4] text_formatted BEFORE injection (with image placeholders) -----")
        print(text_formatted_before)

    # Optionally inject a stored <think>...</think> body for this question,
    # mirroring the --thinking_from_json behavior of qwen_org/test.py.
    injected = ""
    if (thinking_dict is not None and dataset_name is not None
            and question_id is not None):
        injected = thinking_dict.get((dataset_name, str(question_id)), "")
        if injected:
            text_formatted = text_formatted + injected

    if debug:
        print("\n----- [2/4] injected thinking that will be appended -----")
        print(repr(injected) if injected else "(none)")
        print("\n----- [3/4] text_formatted AFTER injection (what the model actually sees) -----")
        print(text_formatted)
        print("#" * 72 + "\n", flush=True)

    image_inputs, video_inputs = process_vision_info(messages)

    # Only pass videos to the processor when we actually have any.
    # Passing an empty list (the default from process_vision_info when the
    # messages contain no videos) crashes inside
    # `transformers.video_utils.convert_pil_frames_to_video` with
    # `IndexError: list index out of range`.
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

    if debug:
        print("----- [4/4] MODEL OUTPUT (raw, decoded) -----")
        print(repr(output_text[0]))
        print("----- [4b] MODEL OUTPUT (rendered, with the injected <think> prepended) -----")
        full_trace = (injected if injected else "") + output_text[0]
        print(full_trace)
        print("#" * 72 + "\n", flush=True)

    return output_text[0], injected


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
    """Load passed_ids from summary.json, supporting two summary layouts.

    1. Flat (this test.py's own output):
        { "mmvp_results": { "accuracy": .., "passed_ids": [...], ... } }
    2. Nested (lvr/test.py style):
        { "mmvp_results": { "steps_4": { "accuracy": .., "passed_ids": [...] } } }

    Either way, return a set of passed_ids, or None if not found.
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
        print(f"Warning: '{passed_ids_key}' is not a dict in {summary_path}")
        return None

    # 1) flat layout: passed_ids is a direct child of the per-dataset dict
    if "passed_ids" in sub and isinstance(sub["passed_ids"], list):
        return set(sub["passed_ids"])

    # 2) nested layout: step_key -> { ..., "passed_ids": [...] }
    for step_key, result in sub.items():
        if isinstance(result, dict) and "passed_ids" in result:
            return set(result["passed_ids"])

    print(f"Warning: No passed_ids found for '{passed_ids_key}' in {summary_path}")
    return None


# ---------------------------------------------------------------------------
# Per-dataset evaluation
# ---------------------------------------------------------------------------
def _make_adv_img_path(adv_images_dir, sub, question_id):
    """Build the path to an adversarial image with the standard naming scheme."""
    # VSTAR ids may have suffixes (e.g. "1_2"); keep the original id string.
    return os.path.join(adv_images_dir, sub, f"{question_id}_adv.png")


def evaluate_mmvp(model, processor, dataset, image_dir, out_dir,
                  replace_images=False, adv_images_dir=None,
                  filter_ids=None, num_samples=None, max_new_tokens=2048,
                  thinking_dict=None, debug=False):
    """Evaluate on MMVP dataset."""
    print(f"\nEvaluating MMVP")
    if replace_images:
        print(f"  Using adversarial images from: {adv_images_dir}/mmvp")
    if filter_ids is not None:
        print(f"  Filtering to {len(filter_ids)} passed IDs from summary")
    if num_samples is not None:
        print(f"  Limiting to first {num_samples} samples")
    os.makedirs(out_dir, exist_ok=True)
    task_instruction = get_task_instruction("mmvp")

    out_file = os.path.join(out_dir, "mmvp_results.json")
    total, correct = 0, 0
    results = []
    passed_ids = []
    evaluated_count = 0

    for dat in tqdm(dataset, desc="MMVP"):
        if filter_ids is not None and dat['question_id'] not in filter_ids:
            continue
        if num_samples is not None and evaluated_count >= num_samples:
            break
        evaluated_count += 1

        if replace_images and adv_images_dir:
            adv_img_path = _make_adv_img_path(adv_images_dir, "mmvp", dat['question_id'])
            if os.path.exists(adv_img_path):
                img_path = adv_img_path
            else:
                img_path = os.path.join(image_dir, dat['image'])
        else:
            img_path = os.path.join(image_dir, dat['image'])

        # Build prompt: the multiple-choice question + answer-format instruction
        # (reasoning prompt + system message are added inside run_inference).
        text = dat['query'].replace('(a)', 'A.').replace('(b)', 'B.')
        text = text + task_instruction

        prediction, injected_thinking = run_inference(
            model, processor, img_path, text,
            max_new_tokens=max_new_tokens, question_id=dat['question_id'],
            thinking_dict=thinking_dict, dataset_name="mmvp",
            debug=debug,
        )
        # If a thinking body was injected into the prompt, stitch it into the
        # stored prediction so debug output shows the full "prompt reasoning +
        # model answer" trace. ``prediction`` alone is only the model's own
        # generation (typically "</think><answer>X</answer>").
        if injected_thinking:
            full_prediction = injected_thinking + prediction
        else:
            full_prediction = prediction

        reasoning, extracted = extract_reasoning_and_answer(full_prediction)
        is_correct = accuracy_reward(full_prediction, dat['label'])
        if is_correct:
            passed_ids.append(dat['question_id'])

        res = {
            'id': dat['question_id'],
            'prediction': full_prediction,
            'injected_thinking': injected_thinking,
            'reasoning': reasoning,
            'extracted_answer': extracted,
            'has_think_tag': '<think>' in full_prediction and '</think>' in full_prediction,
            'has_answer_tag': '<answer>' in full_prediction,
            'label': dat['label'],
            'correct': is_correct
        }
        results.append(res)

        if is_correct:
            correct += 1
        total += 1

    accuracy = correct / total if total > 0 else 0
    # Stats: how many samples actually contained a <think> block (proof of reasoning)
    n_think = sum(1 for r in results if r['has_think_tag'])
    n_answer_tag = sum(1 for r in results if r['has_answer_tag'])
    n_injected = sum(1 for r in results if r.get('injected_thinking'))
    print(f"MMVP - Accuracy: {correct}/{total} = {accuracy*100:.2f}%  "
          f"(think-tag: {n_think}/{total}, answer-tag: {n_answer_tag}/{total}, "
          f"injected: {n_injected}/{total})")

    final_result = {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "passed_ids": passed_ids,
        "results": results,
    }

    json.dump(final_result, open(out_file, 'w+'), indent=2)
    return final_result


def evaluate_vstar(model, processor, dataset, image_dir, out_dir,
                   replace_images=False, adv_images_dir=None,
                   filter_ids=None, num_samples=None, max_new_tokens=2048,
                   thinking_dict=None, debug=False):
    """Evaluate on VSTAR dataset."""
    print(f"\nEvaluating VSTAR")
    if replace_images:
        print(f"  Using adversarial images from: {adv_images_dir}/vstar")
    if filter_ids is not None:
        print(f"  Filtering to {len(filter_ids)} passed IDs from summary")
    if num_samples is not None:
        print(f"  Limiting to first {num_samples} samples")
    os.makedirs(out_dir, exist_ok=True)
    task_instruction = get_task_instruction("vstar")

    out_file = os.path.join(out_dir, "vstar_results.json")
    total, correct = 0, 0
    results = []
    passed_ids = []
    evaluated_count = 0

    for dat in tqdm(dataset, desc="VSTAR"):
        if filter_ids is not None and int(dat['question_id']) not in filter_ids:
            continue
        if num_samples is not None and evaluated_count >= num_samples:
            break
        evaluated_count += 1

        if replace_images and adv_images_dir:
            adv_img_path = _make_adv_img_path(adv_images_dir, "vstar", dat['question_id'])
            if os.path.exists(adv_img_path):
                img_path = adv_img_path
            else:
                img_path = os.path.join(image_dir, dat['image'])
        else:
            img_path = os.path.join(image_dir, dat['image'])

        text = dat['text'] + task_instruction

        prediction, injected_thinking = run_inference(
            model, processor, img_path, text,
            max_new_tokens=max_new_tokens, question_id=dat['question_id'],
            thinking_dict=thinking_dict, dataset_name="vstar",
            debug=debug,
        )
        if injected_thinking:
            full_prediction = injected_thinking + prediction
        else:
            full_prediction = prediction

        reasoning, extracted = extract_reasoning_and_answer(full_prediction)
        is_correct = accuracy_reward(full_prediction, dat['label'])
        if is_correct:
            passed_ids.append(dat['question_id'])

        res = {
            'id': dat['question_id'],
            'prediction': full_prediction,
            'injected_thinking': injected_thinking,
            'reasoning': reasoning,
            'extracted_answer': extracted,
            'has_think_tag': '<think>' in full_prediction and '</think>' in full_prediction,
            'has_answer_tag': '<answer>' in full_prediction,
            'label': dat['label'],
            'correct': is_correct,
            'category': dat.get('category', 'unknown')
        }
        results.append(res)

        if is_correct:
            correct += 1
        total += 1

    accuracy = correct / total if total > 0 else 0
    n_think = sum(1 for r in results if r['has_think_tag'])
    n_answer_tag = sum(1 for r in results if r['has_answer_tag'])
    n_injected = sum(1 for r in results if r.get('injected_thinking'))
    print(f"VSTAR - Accuracy: {correct}/{total} = {accuracy*100:.2f}%  "
          f"(think-tag: {n_think}/{total}, answer-tag: {n_answer_tag}/{total}, "
          f"injected: {n_injected}/{total})")

    final_result = {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "passed_ids": passed_ids,
        "results": results,
    }

    json.dump(final_result, open(out_file, 'w+'), indent=2)
    return final_result


def evaluate_mmstar(model, processor, dataset, image_dir, out_dir,
                    replace_images=False, adv_images_dir=None,
                    filter_ids=None, num_samples=None, max_new_tokens=2048,
                    thinking_dict=None, debug=False):
    """Evaluate on MMStar dataset."""
    print(f"\nEvaluating MMStar")
    if replace_images:
        print(f"  Using adversarial images from: {adv_images_dir}/mmstar")
    if filter_ids is not None:
        print(f"  Filtering to {len(filter_ids)} passed IDs from summary")
    if num_samples is not None:
        print(f"  Limiting to first {num_samples} samples")
    os.makedirs(out_dir, exist_ok=True)
    task_instruction = get_task_instruction("mmstar")

    out_file = os.path.join(out_dir, "mmstar_results.json")
    total, correct = 0, 0
    results = []
    passed_ids = []
    evaluated_count = 0

    for dat in tqdm(dataset, desc="MMStar"):
        if filter_ids is not None and dat['question_id'] not in filter_ids:
            continue
        if num_samples is not None and evaluated_count >= num_samples:
            break
        evaluated_count += 1

        if replace_images and adv_images_dir:
            adv_img_path = _make_adv_img_path(adv_images_dir, "mmstar", dat['question_id'])
            if os.path.exists(adv_img_path):
                img_path = adv_img_path
            else:
                img_path = os.path.join(image_dir, dat['image'])
        else:
            img_path = os.path.join(image_dir, dat['image'])

        text = dat['query'] + task_instruction

        prediction, injected_thinking = run_inference(
            model, processor, img_path, text,
            max_new_tokens=max_new_tokens, question_id=dat['question_id'],
            thinking_dict=thinking_dict, dataset_name="mmstar",
            debug=debug,
        )
        if injected_thinking:
            full_prediction = injected_thinking + prediction
        else:
            full_prediction = prediction

        reasoning, extracted = extract_reasoning_and_answer(full_prediction)
        is_correct = accuracy_reward(full_prediction, dat['label'])
        if is_correct:
            passed_ids.append(dat['question_id'])

        res = {
            'id': dat['question_id'],
            'prediction': full_prediction,
            'injected_thinking': injected_thinking,
            'reasoning': reasoning,
            'extracted_answer': extracted,
            'has_think_tag': '<think>' in full_prediction and '</think>' in full_prediction,
            'has_answer_tag': '<answer>' in full_prediction,
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
    n_think = sum(1 for r in results if r['has_think_tag'])
    n_answer_tag = sum(1 for r in results if r['has_answer_tag'])
    n_injected = sum(1 for r in results if r.get('injected_thinking'))
    print(f"MMStar - Accuracy: {correct}/{total} = {accuracy*100:.2f}%  "
          f"(think-tag: {n_think}/{total}, answer-tag: {n_answer_tag}/{total}, "
          f"injected: {n_injected}/{total})")

    final_result = {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "passed_ids": passed_ids,
        "results": results,
    }

    json.dump(final_result, open(out_file, 'w+'), indent=2)
    return final_result


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

    if mmvp_results:
        summary["mmvp_results"] = {
            "accuracy": mmvp_results["accuracy"],
            "correct": mmvp_results["correct"],
            "total": mmvp_results["total"],
            "passed_ids": mmvp_results["passed_ids"],
            "examples": collect_examples(mmvp_results["results"])
        }

    if vstar_results:
        summary["vstar_results"] = {
            "accuracy": vstar_results["accuracy"],
            "correct": vstar_results["correct"],
            "total": vstar_results["total"],
            "passed_ids": vstar_results["passed_ids"],
            "examples": collect_examples(vstar_results["results"])
        }

    if mmstar_results:
        summary["mmstar_results"] = {
            "accuracy": mmstar_results["accuracy"],
            "correct": mmstar_results["correct"],
            "total": mmstar_results["total"],
            "passed_ids": mmstar_results["passed_ids"],
            "examples": collect_examples(mmstar_results["results"])
        }

    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nSummary saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Qwen2.5-VL-7B-Instruct Evaluation on Local Datasets"
    )
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
    parser.add_argument("--max_new_tokens", type=int, default=2048,
                        help="Max new tokens for generation (default 2048 for reasoning)")
    parser.add_argument("--thinking_from_json", type=str, default=None,
                        help="Path to a per-dataset *_results.json file "
                             "(e.g. mmvp_results.json as written by this "
                             "script). The JSON must be for the same dataset "
                             "passed via --dataset (one of mmvp/vstar/mmstar; "
                             "not 'all'). The LAST <think>...</think> body "
                             "from each sample's 'prediction' is appended "
                             "after the chat template so the model continues "
                             "from that thinking instead of reasoning from "
                             "scratch. Any trailing 'Final answer: X' line "
                             "inside the block is stripped; the "
                             "<answer>X</answer> tag is never injected.")
    parser.add_argument("--debug_thinking", action="store_true",
                        help="When set, prints the chat template, the "
                             "injected thinking, the final prompt the model "
                             "sees, and the model's raw output for each "
                             "sample as it is evaluated.")
    args = parser.parse_args()

    # Examples:
    #   python test.py --dataset all
    #   python test.py --dataset mmvp
    #   python test.py --dataset vstar --use_passed_ids --adv_images_dir random_images
    #   python test.py --dataset mmvp --num_samples 20
    #   python test.py --dataset mmvp --use_passed_ids --num_samples 20
    #   python test.py --dataset vstar --use_passed_ids --adv_images_dir adv_images_white
    # Determine summary path based on dataset (consistent with attack_white.py)
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
    print("Qwen2.5-VL-7B-Instruct Evaluation on Local Datasets")
    print("=" * 64)
    print(f"Model path: {MODEL_PATH}")
    print(f"Dataset: {args.dataset}")
    print(f"Adversarial images dir: {args.adv_images_dir}")
    print(f"Use passed IDs: {args.use_passed_ids}")
    print(f"Num samples: {args.num_samples}")
    print(f"Max new tokens: {args.max_new_tokens}")
    print(f"Summary path: {summary_file}")

    # Load passed_ids if requested (consistent with attack_white.py)
    filter_ids = None
    if args.use_passed_ids:
        filter_ids = get_passed_ids_from_summary(summary_file, passed_ids_key)
        if filter_ids:
            print(f"Loaded {len(filter_ids)} passed IDs for filtering")
            if args.num_samples is not None and args.num_samples < len(filter_ids):
                filter_ids = list(filter_ids)[:args.num_samples]
                print(f"Limited to first {len(filter_ids)} passed IDs")
        else:
            print("Warning: No passed_ids found in summary.json")

    # Load thinking_dict from a per-dataset results.json if --thinking_from_json
    # is set. The file's dataset identity comes from --dataset (must be a
    # single dataset, not 'all', since the per-dataset format does not encode
    # which dataset the entries belong to).
    thinking_dict = None
    if args.thinking_from_json:
        if not os.path.exists(args.thinking_from_json):
            print(f"Warning: --thinking_from_json path not found: "
                  f"{args.thinking_from_json}")
        elif args.dataset == "all":
            print("Warning: --thinking_from_json requires --dataset to be a "
                  "single dataset (mmvp/vstar/mmstar), not 'all'. "
                  "Run each dataset separately.")
        else:
            thinking_dict = load_thinking_from_json(
                args.thinking_from_json, args.dataset)
            print(f"Loaded thinking for {len(thinking_dict)} (dataset={args.dataset},"
                  f"id) pairs from {args.thinking_from_json}")

    # Set random seed for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)

    # Load model
    model, processor = load_model_and_processor()
    model.eval()

    # Create output directory
    run_output_dir = os.path.join(OUTPUT_DIR, "run_qwen_vl")
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
            MMVP_IMAGE_DIR, run_output_dir,
            replace_images=args.adv_images_dir is not None,
            adv_images_dir=args.adv_images_dir,
            filter_ids=filter_ids,
            num_samples=args.num_samples,
            max_new_tokens=args.max_new_tokens,
            thinking_dict=thinking_dict,
            debug=args.debug_thinking,
        )

    if args.dataset in ["all", "vstar"]:
        print("\nLoading VSTAR dataset...")
        vstar_dataset = load_vstar_dataset()
        print(f"VSTAR: {len(vstar_dataset)} samples")

        vstar_results = evaluate_vstar(
            model, processor, vstar_dataset,
            VSTAR_DATA_DIR, run_output_dir,
            replace_images=args.adv_images_dir is not None,
            adv_images_dir=args.adv_images_dir,
            filter_ids=filter_ids,
            num_samples=args.num_samples,
            max_new_tokens=args.max_new_tokens,
            thinking_dict=thinking_dict,
            debug=args.debug_thinking,
        )

    if args.dataset in ["all", "mmstar"]:
        print("\nLoading MMStar dataset...")
        mmstar_dataset = load_mmstar_dataset()
        print(f"MMStar: {len(mmstar_dataset)} samples")

        mmstar_results = evaluate_mmstar(
            model, processor, mmstar_dataset,
            MMSTAR_DATA_DIR, run_output_dir,
            replace_images=args.adv_images_dir is not None,
            adv_images_dir=args.adv_images_dir,
            filter_ids=filter_ids,
            num_samples=args.num_samples,
            max_new_tokens=args.max_new_tokens,
            thinking_dict=thinking_dict,
            debug=args.debug_thinking,
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
        print(f"\nMMVP: {mmvp_results['accuracy']*100:.2f}% "
              f"({mmvp_results['correct']}/{mmvp_results['total']})")

    if vstar_results:
        print(f"\nVSTAR: {vstar_results['accuracy']*100:.2f}% "
              f"({vstar_results['correct']}/{vstar_results['total']})")

    if mmstar_results:
        print(f"\nMMStar: {mmstar_results['accuracy']*100:.2f}% "
              f"({mmstar_results['correct']}/{mmstar_results['total']})")

    print("\nDone!")


if __name__ == "__main__":
    main()