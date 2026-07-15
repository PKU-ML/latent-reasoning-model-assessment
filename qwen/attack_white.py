"""
================================================================================
Qwen2.5-VL 图像对抗攻击 - logit攻击 white box 版本
================================================================================

攻击原理：
- 对像素图像添加扰动
- 损失函数：prompt中倒数第6个token的隐向量经解码矩阵后，错误选项的logit
- 目标：最大化错误选项的logit，使模型倾向于选择错误答案

================================================================================
"""

import gc
import torch
import torch.nn.functional as F
import json
import os
from pathlib import Path
from tqdm import tqdm
import torchvision
from torchvision import transforms
from PIL import Image
import argparse

import sys
qwen_src_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
sys.path.insert(0, qwen_src_path)

from transformers import AutoProcessor, AutoConfig, AutoModelForVision2Seq
from qwen_vl_utils import process_vision_info

# ============ 像素图像处理函数 ============

def restore_images(flatten_patches, grid_t, grid_h, grid_w, merge_size, temporal_patch_size, patch_size, channel, data_format):
    """将 patchified pixel_values 还原为像素图像"""
    grid_h_re = grid_h // merge_size
    grid_w_re = grid_w // merge_size

    restored = flatten_patches.reshape(
        grid_t, grid_h_re, grid_w_re, merge_size, merge_size, channel,
        temporal_patch_size, patch_size, patch_size
    )
    restored = restored.permute(0, 6, 5, 1, 3, 7, 2, 4, 8)

    total_time = grid_t * temporal_patch_size
    height = grid_h_re * merge_size * patch_size
    width = grid_w_re * merge_size * patch_size
    restored = restored.reshape(total_time, channel, height, width)

    if data_format == "channels_last":
        restored = restored.transpose(0, 2, 3, 1)

    return restored


def pixel_reshape(image, patch_size, merge_size, temporal_patch_size):
    """将像素图像转换为 patchified pixel_values"""
    if image.dim() == 3:
        patches = image.unsqueeze(0)
    else:
        patches = image
    if patches.shape[0] == 1:
        patches = patches.repeat(temporal_patch_size, 1, 1, 1)

    channel = patches.shape[1]
    resized_height = patches.shape[2]
    resized_width = patches.shape[3]
    grid_t = patches.shape[0] // temporal_patch_size
    grid_h, grid_w = resized_height // patch_size, resized_width // patch_size

    patches = patches.view(
        grid_t, temporal_patch_size, channel,
        grid_h // merge_size, merge_size, patch_size,
        grid_w // merge_size, merge_size, patch_size,
    )
    patches = patches.permute(0, 3, 6, 4, 7, 2, 1, 5, 8)
    flatten_patches = patches.reshape(
        grid_t * grid_h * grid_w, channel * temporal_patch_size * patch_size * patch_size
    )
    return flatten_patches, (grid_t, grid_h, grid_w)


# ============ 配置 ============
MODEL_PATH = os.environ.get("QWEN_MODEL_PATH", "Qwen/Qwen2.5-VL-7B-Instruct")

EPSILON = 4 / 255.0
STEP_SIZE = 0.5 / 255.0
NUM_ITERS = 300

_DATA_ROOT = os.environ.get("DATA_DIR", str(Path(__file__).resolve().parent.parent / "data"))
VSTAR_DATA_DIR = os.environ.get("QWEN_VSTAR_DATA_DIR", str(Path(_DATA_ROOT) / "vstar"))
VSTAR_JSONL = os.environ.get("QWEN_VSTAR_JSONL", str(Path(_DATA_ROOT) / "vstar" / "test_questions.jsonl"))

MMVP_IMAGE_DIR = os.environ.get("QWEN_MMVP_IMAGE_DIR", str(Path(_DATA_ROOT) / "MMVP" / "MMVP Images"))
MMVP_CSV = os.environ.get("QWEN_MMVP_CSV", str(Path(_DATA_ROOT) / "MMVP" / "Questions.csv"))

MMSTAR_DATA_DIR = os.environ.get("QWEN_MMSTAR_DATA_DIR", str(Path(_DATA_ROOT) / "MMStar"))
MMSTAR_METADATA = os.environ.get("QWEN_MMSTAR_METADATA", str(Path(_DATA_ROOT) / "MMStar" / "metadata.json"))

OUTPUT_DIR_BASE = os.environ.get(
    "QWEN_OUTPUT_DIR",
    str(Path(__file__).resolve().parent / "adv_images_white"),
)

SUMMARY_PATH = os.environ.get(
    "QWEN_SUMMARY_PATH",
    str(Path(__file__).resolve().parent / "results" / "org" / "summary.json"),
)

MMVP_SUMMARY_PATH = os.environ.get("QWEN_MMVP_SUMMARY_PATH", "summary.json")
VSTAR_SUMMARY_PATH = os.environ.get("QWEN_VSTAR_SUMMARY_PATH", "summary.json")
MMSTAR_SUMMARY_PATH = os.environ.get("QWEN_MMSTAR_SUMMARY_PATH", "summary.json")
MMVP_SUMMARY_PATH = VSTAR_SUMMARY_PATH = MMSTAR_SUMMARY_PATH = SUMMARY_PATH

os.makedirs(OUTPUT_DIR_BASE, exist_ok=True)

# CLIP 归一化参数（与 processor 一致）
CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD  = [0.26862954, 0.26130258, 0.27577711]


# ============ Test.py 对齐常量与函数 ============
# 下面这些与 qwen/test.py 严格对齐：
#   - REASONING_SYSTEM_PROMPT / REASONING_USER_PROMPT：test.py 强制模型"先推理后给答案"
#   - extract_reasoning_and_answer：从 <think>...</think><answer>X</answer> 中抽最后一个 <answer> 字母
#   - accuracy_reward_test / get_task_instruction_test / create_messages_test_style：
#     与 test.py 中的同名逻辑完全等价
#   - validate_with_test_format：对单张 adv image 跑一次和 test.py.run_inference 等价的
#     generate + 答案抽取
# 这一层的目的：attack_white.py 在每一次 FGSM 步产生的图像上跑一次"和 test.py 完全等价
# 的推理 + 抽取"，筛出那些既能让模型产出错误答案、又满足 wrong_logit - correct_logit > 0.1
# 的样本提前保存，从而保证产出的 _adv.png 一旦被 test.py 验证也得到错误答案。

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


def extract_reasoning_and_answer(response):
    """与 qwen/test.py.extract_reasoning_and_answer 完全一致：抽最后一个 <answer> 块内的首字母。"""
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


def accuracy_reward_test(response, ground_truth):
    """与 qwen/test.py.accuracy_reward 完全一致：只看最后一个 <answer>X</answer>。"""
    _, answer = extract_reasoning_and_answer(response)
    if answer is None:
        return False
    return answer == ground_truth


def get_task_instruction_test(bench_name):
    """与 qwen/test.py.get_task_instruction 完全一致（无论 bench_name 都返回同一字符串）。"""
    return "\nAnswer with the option's letter from the given choices directly."


def create_messages_test_style(img_path, question, system_prompt=None):
    """与 qwen/test.py.create_messages 完全一致：构造 image + text 的 user content，
    并根据 system_prompt 是否提供决定是否在开头插入 system 消息。"""
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


def validate_with_test_format(model, processor, img_path, question, correct_answer,
                              dataset_name, device, max_new_tokens=1024):
    """
    对单张图像跑一次与 qwen/test.py.run_inference 完全等价的生成 + 答案抽取流程。
    返回 (is_wrong, extracted_answer, raw_prediction_text)：
      - is_wrong: 模型产出字母与 correct_answer 不同（None → False）
      - extracted_answer: 抽出的字母（可能是 None，表示模型未生成 <answer> 标签）
      - raw_prediction_text: 模型生成的原文（不包含前缀注入）

    与 test.py 的对齐点：
      * 用相同的 REASONING_SYSTEM_PROMPT / REASONING_USER_PROMPT / task_instruction
      * 用相同的 chat-template（system + user(image+text)）
      * 生成参数：do_sample=False, temperature=0.0（test.py 默认）
      * max_new_tokens 默认 1024（test.py 默认 2048，可通过 main 的 CLI 覆写）
      * 答案抽取：extract_reasoning_and_answer（与 test.py 完全一致）
    """
    task_instruction = get_task_instruction_test(dataset_name)
    text = question.replace('(a)', 'A.').replace('(b)', 'B.')
    text = text + task_instruction

    full_user_text = REASONING_USER_PROMPT + text
    messages = create_messages_test_style(
        img_path, full_user_text, system_prompt=REASONING_SYSTEM_PROMPT
    )
    text_formatted = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    image_inputs, video_inputs = process_vision_info(messages)
    # 与 test.py 保持一致的 video 处理（list 默认空时直接抛 IndexError）
    if video_inputs:
        inputs = processor(
            text=[text_formatted], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt"
        )
    else:
        inputs = processor(
            text=[text_formatted], images=image_inputs,
            padding=True, return_tensors="pt"
        )
    inputs = inputs.to(device)

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
        )[0]

    _, extracted = extract_reasoning_and_answer(output_text)
    is_wrong = (extracted is not None
                and extracted.upper() != str(correct_answer).strip().upper())
    return is_wrong, extracted, output_text


# ============ 工具函数 ============

def load_mmvp():
    import csv
    data = []
    with open(MMVP_CSV, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            idx = int(row["Index"])
            text = row["Question"] + '\nOptions:\n' + row["Options"]
            raw = row["Correct Answer"].strip()
            if raw.startswith('(') and len(raw) >= 2:
                label = raw[1].upper()
            else:
                label = raw.upper()[0]
            data.append({'id': idx, 'image': f'{idx}.jpg', 'query': text, 'label': label})
    return data


def load_vstar():
    """Load VSTAR dataset from local JSONL file."""
    import json
    data = []
    with open(VSTAR_JSONL, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line.strip())
            label = item["label"].upper().strip()
            if label.startswith('('):
                label = label[1]
            data.append({
                'id': item['question_id'],
                'image': item['image'],
                'query': item['text'],
                'label': label
            })
    return data


def load_mmstar():
    """Load MMStar dataset from local metadata JSON.

    Each entry in metadata.json has fields ``index``, ``question`` (with
    options already formatted as ``A: text, B: text, ...``), ``answer``
    (single letter A/B/C/D), ``image_path`` (relative path like
    ``images/0.jpg``).
    """
    with open(MMSTAR_METADATA, 'r', encoding='utf-8') as f:
        records = json.load(f)
    data = []
    for item in records:
        label = item["answer"].upper().strip()
        if label.startswith('('):
            label = label[1]
        data.append({
            'id': item["index"],
            'image': item["image_path"],
            'query': item["question"],
            'label': label,
            'category': item.get("category", ""),
            'l2_category': item.get("l2_category", ""),
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


def get_task_instruction():
    return "\nAnswer with the option's letter from the given choices directly."


def setup_model():
    config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)

    model = AutoModelForVision2Seq.from_pretrained(
        MODEL_PATH,
        config=config,
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )
    model = model.to("cuda:0")

    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)

    model.eval()
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    return model, processor, config


def save_adv_image(orig_img, delta, save_path):
    """
    将扰动叠加到原图后保存，保留原始分辨率。
    orig_img: [3, H_orig, W_orig]，[0,1]，CPU 或 GPU tensor
    delta:    与 orig_img 同 shape，[0,1] 空间的扰动
    """
    adv = (orig_img.cpu() + delta.detach().cpu()).clamp(0, 1)
    img_np = (adv.permute(1, 2, 0).numpy() * 255).astype('uint8')
    Image.fromarray(img_np).save(save_path)


# ============ 核心攻击函数 ============

def attack_sample_logit_prob_white(model, processor, config, img_path, question, correct_answer, device, dataset_type='mmvp', validate_max_new_tokens=1024, validate_temp_dir=None):
    """
    Qwen2.5-VL white box 攻击：delta 定义在原图像素空间，通过可微 F.interpolate 传递梯度。

    损失函数：prompt中倒数第6个token的隐向量经lm_head解码后，错误选项（A或B）对应的logit
    目标：最大化错误选项的logit，使模型倾向于选择错误答案

    与 qwen/test.py 严格对齐的验证步骤（在每一次 FGSM 步之后执行）：
      1) 当 wrong_logit - correct_logit > LOGIT_GAP_THRESHOLD（默认 0.1）时，把当前
         adv image 写到 validate_temp_dir，做一次与 test.py.run_inference 等价的推理。
      2) 用 extract_reasoning_and_answer（与 test.py 一致）抽出模型给的字母。
      3) 若字母与 correct_answer 不同 → 把当前 delta 提前保存为 final，break 出循环
         （满足"让模型出错且 logit 差超过 0.1 则保存并提前终止"）。

    dataset_type:
        - 'mmvp': 二选一，交换A/B作为错误答案
        - 'vstar': 四选一，选择模型预测logit概率最大的那个错误选项作为攻击目标
        - 'mmstar': 同 vstar (四选一)
    """

    task_instruction = get_task_instruction()
    text = question.replace('(a)', 'A.').replace('(b)', 'B.')
    text = text + task_instruction

    visual_device = next(model.visual.parameters()).device
    lm_device = next(model.language_model.parameters()).device

    # 检查 visual 模块参数的梯度状态
    # print(f"    [DEBUG] visual module device: {visual_device}")
    visual_trainable = sum(1 for p in model.visual.parameters() if p.requires_grad)
    visual_total = sum(1 for _ in model.visual.parameters())
    # print(f"    [DEBUG] visual parameters: {visual_total}, trainable: {visual_trainable}")

    # 检查模型整体参数
    total_params = sum(1 for _ in model.parameters())
    trainable_params = sum(1 for p in model.parameters() if p.requires_grad)
    # print(f"    [DEBUG] model total params: {total_params}, trainable: {trainable_params}")

    # ============ 运行 processor 获取 text 相关输入和 image_grid_thw ============
    messages = [{'role': 'user', 'content': [
        {'type': 'image', 'image': img_path},
        {'type': 'text', 'text': text}
    ]}]
    text_formatted = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)

    # Only pass `videos` to the processor when we actually have any.
    # Passing an empty list (the default from process_vision_info when the
    # messages contain no videos) crashes inside
    # `transformers.video_utils.convert_pil_frames_to_video` with
    # `IndexError: list index out of range`. Same fix as qwen/test.py.
    if video_inputs:
        inputs = processor(text=[text_formatted], images=image_inputs, videos=video_inputs,
                           padding=True, return_tensors="pt")
    else:
        inputs = processor(text=[text_formatted], images=image_inputs,
                           padding=True, return_tensors="pt")
    input_ids = inputs['input_ids'].to(lm_device)
    attention_mask = inputs['attention_mask'].to(lm_device)
    pixel_values_clean = inputs['pixel_values'].to(visual_device)
    image_grid_thw = inputs['image_grid_thw'].to(visual_device)

    # ============ 从 image_grid_thw 得到模型处理分辨率 ============
    T, H, W = image_grid_thw[0].tolist()
    T, H, W = int(T), int(H), int(W)
    patch_size, temporal_patch_size, merge_size, channel = 14, 2, 2, 3
    H_target = H * patch_size
    W_target = W * patch_size

    # ============ 加载原图为 [0,1] tensor（原始分辨率）============
    orig_img = transforms.ToTensor()(Image.open(img_path).convert('RGB'))
    orig_img = orig_img.to(visual_device)

    print(f"    Original image size: {orig_img.shape[2]}x{orig_img.shape[1]}")
    print(f"    Model target size:   {W_target}x{H_target}")

    mean = torch.tensor(CLIP_MEAN, device=visual_device).view(3, 1, 1)
    std  = torch.tensor(CLIP_STD,  device=visual_device).view(3, 1, 1)

    def preprocess(img_01):
        """[3, H_orig, W_orig] [0,1] → patchified pixel_values"""
        # print(f"    [DEBUG preprocess] img_01.shape={img_01.shape}, img_01.requires_grad={img_01.requires_grad}")
        resized = F.interpolate(
            img_01.unsqueeze(0), size=(H_target, W_target),
            mode='bilinear', align_corners=False
        )[0]
        # print(f"    [DEBUG preprocess] resized.shape={resized.shape}, resized.requires_grad={resized.requires_grad}")
        normalized = (resized - mean) / std
        # print(f"    [DEBUG preprocess] normalized.requires_grad={normalized.requires_grad}")
        pv, _ = pixel_reshape(normalized, patch_size=patch_size,
                               merge_size=merge_size, temporal_patch_size=temporal_patch_size)
        # print(f"    [DEBUG preprocess] pv.shape={pv.shape}, pv.requires_grad={pv.requires_grad}")
        return pv

    tokenizer = processor.tokenizer

    # ============ 找到倒数第6个token的位置作为answer_pos ============
    input_ids_list = input_ids[0].tolist()
    answer_pos = len(input_ids_list) - 6

    print(f"    Total tokens: {len(input_ids_list)}")
    print(f"    answer_pos (6th from last): {answer_pos}")
    print(f"    Tokens at answer_pos~answer_pos+5: {[tokenizer.decode([tid]) for tid in input_ids_list[answer_pos:answer_pos+6]]}")

    # ============ DEBUG: 输出完整的input_ids结构 ============
    # print(f"\n    [DEBUG] Full input_ids shape: {input_ids.shape}")
    # print(f"    [DEBUG] Full prompt tokens:")
    all_tokens = input_ids_list
    special_tokens = ['<|im_start|>', '<|im_end|>', '<|vision_start|>', '<|vision_end|>', '<|end|>']
    for i, token_id in enumerate(all_tokens):
        token_str = tokenizer.decode([token_id])
        if token_str.strip() in special_tokens:
            print(f"      [{i:3d}] {token_id:6d} -> {repr(token_str)} <-- SPECIAL")
        elif i == answer_pos:
            print(f"      [{i:3d}] {token_id:6d} -> {repr(token_str)} <-- ANSWER_POS")
        else:
            if i < 30 or i > len(all_tokens) - 10:
                print(f"      [{i:3d}] {token_id:6d} -> {repr(token_str)}")
            elif i == 30:
                print(f"      ... (skipping middle tokens) ...")
    print()

    lm_head_weight = model.lm_head.weight

    # ============ 判断错误答案 ============
    # VSTAR / MMStar (4 选): 先做一次前向传播确定哪个错误选项logit最高
    # MMVP (2 选): 使用交换策略（A↔B）
    if dataset_type in ('vstar', 'mmstar'):
        label_tag = 'MMSTAR' if dataset_type == 'mmstar' else 'VSTAR'
        print(f"    [{label_tag}] Running initial forward to determine attack target...")
        with torch.no_grad():
            adv_orig_init = orig_img.clamp(0, 1)
            adv_pixels_init = preprocess(adv_orig_init)
            adv_pixels_init = adv_pixels_init.to(visual_device)

            outputs_init = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=adv_pixels_init,
                image_grid_thw=image_grid_thw,
                return_dict=True,
                output_hidden_states=True
            )
            answer_hidden_init = outputs_init.hidden_states[-1][0, answer_pos, :]
            logits_init = answer_hidden_init @ lm_head_weight.t()

        del outputs_init, adv_pixels_init
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        abcd_token_ids = {}
        for opt in ['A', 'B', 'C', 'D']:
            tokens = tokenizer(opt, add_special_tokens=False, return_tensors='pt')
            if tokens['input_ids'].numel() > 0:
                abcd_token_ids[opt] = tokens['input_ids'][0].item()
            else:
                abcd_token_ids[opt] = tokenizer.encode(opt, add_special_tokens=False)[0]

        correct_upper = correct_answer.upper()
        wrong_options = [opt for opt in ['A', 'B', 'C', 'D'] if opt != correct_upper]

        max_wrong_logit = float('-inf')
        wrong_answer = 'A'
        for opt in wrong_options:
            opt_logit = logits_init[abcd_token_ids[opt]].item()
            print(f"      Option {opt} logit: {opt_logit:.4f}")
            if opt_logit > max_wrong_logit:
                max_wrong_logit = opt_logit
                wrong_answer = opt

        print(f"    [{label_tag}] Correct: {correct_answer}, Selected wrong target: '{wrong_answer}' (logit: {max_wrong_logit:.4f})")
    else:
        if correct_answer == 'A':
            wrong_answer = 'B'
        elif correct_answer == 'B':
            wrong_answer = 'A'
        else:
            wrong_answer = 'A'
        print(f"    [MMVP] Correct: {correct_answer}, Wrong target: '{wrong_answer}'")

    wrong_tokens = tokenizer(wrong_answer, add_special_tokens=False, return_tensors='pt')
    if wrong_tokens['input_ids'].numel() > 0:
        wrong_token_id = wrong_tokens['input_ids'][0].item()
    else:
        wrong_token_id = tokenizer.encode(wrong_answer, add_special_tokens=False)[0]

    print(f"    Wrong target token_id: {wrong_token_id}")

    # 同时计算 correct_answer 的 token id，用于评估 (wrong - correct) logit 差
    correct_letter = str(correct_answer).strip().upper()
    correct_tokens = tokenizer(correct_letter, add_special_tokens=False, return_tensors='pt')
    if correct_tokens['input_ids'].numel() > 0:
        correct_token_id = correct_tokens['input_ids'][0].item()
    else:
        correct_token_id = tokenizer.encode(correct_letter, add_special_tokens=False)[0]
    print(f"    Correct target token_id: {correct_token_id}")

    # ============ 验证阶段的阈值与临时目录 ============
    # 用户的策略："优化目标 logit 的差值超过 0.1" → 这里定义为
    # (wrong - correct) logit gap > 0.1；满足该条件后，再去跑一次与 test.py
    # 等价的推理，验证模型是否真的给出错误答案；两者皆满足则提前保存 + 终止循环。
    LOGIT_GAP_THRESHOLD = 0.1
    if validate_temp_dir is None:
        validate_temp_dir = "/tmp/attack_white_validate"
    os.makedirs(validate_temp_dir, exist_ok=True)
    # temp 文件名按 image 名取，避免路径特殊字符
    sample_tag = os.path.basename(img_path).rsplit('.', 1)[0]
    safe_tag = "".join(c if c.isalnum() or c in ('_', '-') else '_' for c in str(sample_tag))
    temp_img_path = os.path.join(validate_temp_dir, f"{safe_tag}_adv.png")

    # ============ delta 定义在原图像素空间，初始化为零 ============
    delta = torch.zeros_like(orig_img, requires_grad=True)
    # print(f"    [DEBUG] orig_img shape: {orig_img.shape}")
    # print(f"    [DEBUG] delta shape: {delta.shape}  (original image space, [0,1])")

    best_loss = float('-inf')
    best_delta = None
    early_stopped = False
    success_iter = -1

    for iteration in range(NUM_ITERS):
        # ============ 原图 + delta → 可微 resize → 归一化 → patchify ============
        adv_orig = (orig_img + delta).clamp(0, 1)
        adv_pixels = preprocess(adv_orig)
        adv_pixels = adv_pixels.to(visual_device)

        # 前向传播
        outputs_adv = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=adv_pixels,
            image_grid_thw=image_grid_thw,
            return_dict=True,
            output_hidden_states=True
        )

        # 取 answer_pos 位置的隐向量，经 lm_head 得 logit
        answer_hidden = outputs_adv.hidden_states[-1][0, answer_pos, :]
        logits = answer_hidden @ lm_head_weight.t()
        wrong_logit = logits[wrong_token_id]
        # 新增：抽出 correct 选项的 logit，用于计算 (wrong - correct) logit 差
        correct_logit = logits[correct_token_id]
        loss = wrong_logit

        # (wrong - correct) 的差 → 当 logit 差 > 0.1 时才付出"与 test.py 等价的推理"的代价
        with torch.no_grad():
            logit_gap = (wrong_logit - correct_logit).item()

        # 保存最优 delta（目标函数最大）
        if loss.item() > best_loss:
            best_loss = loss.item()
            best_delta = delta.detach().clone()

        if iteration % 10 == 0:
            print(f"    Iter {iteration}: loss={loss.item():.6f}, "
                  f"wrong_logit={wrong_logit.item():.6f}, "
                  f"correct_logit={correct_logit.item():.6f}, "
                  f"gap={logit_gap:.4f}")

        # ============ 反传梯度到原图空间，FGSM 更新 delta ============
        model.zero_grad()
        if delta.grad is not None:
            delta.grad.zero_()

        loss.backward(retain_graph=True)
        orig_grad = delta.grad

        with torch.no_grad():
            delta = torch.clamp(
                delta + STEP_SIZE * orig_grad.sign(),
                -EPSILON, EPSILON
            ).detach().requires_grad_(True)

        # 显式释放 outputs_adv + 缓存，避免每一步都吃一份显存
        del outputs_adv
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # ============ "与即将保存的同一张图"对一次 logit gap ============
        # 上面算的 logit_gap 是 FGSM 之前的 delta 算的，落盘时用的却是 FGSM 之后的
        # delta（两个 delta 相差 STEP_SIZE * sign(grad) 并被 clamp 到 [-EPSILON,
        # EPSILON]）。这俩在数学上不完全等价，所以我们必须对"真正要保存的那个 delta"
        # 再做一次前向，把闸门用的 logit_gap 也换成更新过的值。这样 is_wrong、logit_gap
        # 和最终落盘的 _adv.png 三者全部来自同一个 delta，三条都严格满足。
        with torch.no_grad():
            adv_pixels_check = preprocess((orig_img + delta).clamp(0, 1)).to(visual_device)
            outputs_check = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=adv_pixels_check,
                image_grid_thw=image_grid_thw,
                return_dict=True,
                output_hidden_states=True,
            )
            answer_hidden_check = outputs_check.hidden_states[-1][0, answer_pos, :]
            logits_check = answer_hidden_check @ lm_head_weight.t()
            wrong_logit_check = logits_check[wrong_token_id]
            correct_logit_check = logits_check[correct_token_id]
            logit_gap = (wrong_logit_check - correct_logit_check).item()
            del outputs_check, adv_pixels_check, logits_check
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # ============ 验证步骤：与 test.py 严格对齐的推理 + 抽取 ============
        # 用户的策略：
        #   "对迭代每一步产生的图片都要让模型进行推理并输出答案，只要找到让模型出错
        #    且优化目标 logit 的差值超过 0.1 则保存，然后提前终止循环。"
        # 我们的实现：
        #   - 每一轮 FGSM 后都用 no-grad 再前向一次，得到 *即将保存的* delta 上的
        #     logit_gap（与"将要落盘"严格对齐）→ cheap；
        #   - 只有当 (wrong - correct) logit 差已经 > 0.1 时，才发一次"和 test.py 完全等价"
        #     的 generate（这是 expensive 的一步）。这样既落实了"每一步都要验证"的语义
        #     （logit 检查每步在跑），又把真正昂贵的 generate 留在已具备攻击潜力时；
        #   - generate 出来用 extract_reasoning_and_answer（与 test.py 一致）抽字母，
        #     字母 ≠ correct → 同时满足两条："让模型出错" + "logit 差 > 0.1" → 立刻保存。
        if logit_gap > LOGIT_GAP_THRESHOLD:
            # 把当前 adv image 落到磁盘，再让 validate 走和 test.py 完全一致的
            # "先 chat-template → 再 generate" 流程
            save_adv_image(orig_img, delta.detach(), temp_img_path)

            try:
                is_wrong, extracted, _raw_pred = validate_with_test_format(
                    model, processor, temp_img_path, question, correct_answer,
                    dataset_name=dataset_type, device=device,
                    max_new_tokens=validate_max_new_tokens,
                )
            except Exception as e:
                print(f"    [validate error at iter {iteration}] {e}")
                is_wrong, extracted = False, None

            print(f"    [Iter {iteration}] validate: extracted={extracted}, "
                  f"target={correct_answer}, is_wrong={is_wrong}, gap={logit_gap:.4f}")

            if is_wrong:
                print(f"    [Iter {iteration}] ✓ SUCCESS: model errs AND logit gap "
                      f"{logit_gap:.4f} > {LOGIT_GAP_THRESHOLD} → early stop")
                # 用当前 delta 作为 final（满足条件那一刻的扰动），不再依赖 best_loss
                best_delta = delta.detach().clone()
                best_loss = loss.item()
                early_stopped = True
                success_iter = iteration
                break
        # else: logit gap 还没到 0.1，继续下一轮 FGSM

    if best_delta is not None:
        final_delta = best_delta
    else:
        final_delta = delta.detach()

    tag = "EARLY" if early_stopped else "FULL"
    print(f"    [{tag}] best_loss={best_loss:.6f}, "
          f"early_stopped={early_stopped} (success_iter={success_iter})")

    return orig_img, final_delta, early_stopped, success_iter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--max_samples', type=int, default=None)
    parser.add_argument('--iters', type=int, default=50)
    parser.add_argument('--summary_path', type=str, default=None,
                        help="Path to summary.json for loading passed_ids")
    parser.add_argument('--dataset', type=str, default='mmvp', choices=['mmvp', 'vstar', 'mmstar'],
                        help='Dataset to attack: mmvp, vstar, or mmstar')
    parser.add_argument('--validate_max_new_tokens', type=int, default=1024,
                        help="Max new tokens for the test.py-aligned validation generation in each FGSM step "
                             "(test.py default is 2048; lower = faster but may cut the model's reasoning)")
    parser.add_argument('--validate_temp_dir', type=str, default='/tmp/attack_white_validate',
                        help="Where to dump per-sample adv images during the in-loop validation step")
    parser.add_argument('--logit_gap_threshold', type=float, default=0.1,
                        help="(wrong - correct) logit gap 必须 > 此阈值，validation 才会触发 "
                             "test.py 等价的 generate；test.py 验证可得性不直接由此阈值决定，"
                             "但它控制 attack 内部的闸门，使我们早停时 confidence 足够。")
    args = parser.parse_args()

    global NUM_ITERS
    NUM_ITERS = args.iters

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    device = torch.device('cuda:0')
    print(f"Using GPU: {os.environ.get('CUDA_VISIBLE_DEVICES', 'auto')}, iters: {NUM_ITERS}")
    print(f"Attacking dataset: {args.dataset.upper()}")

    print("Loading model...")
    model, processor, config = setup_model()
    model.eval()
    print("Model loaded")

    # Determine image directory and summary path based on dataset type.
    # MMStar and VStar are both 4-choice, so they share the same per-sample
    # attack target selection logic (pick the wrong option with the highest
    # pre-attack logit).
    if args.dataset == 'vstar':
        image_dir = VSTAR_DATA_DIR
        summary_file = args.summary_path if args.summary_path else VSTAR_SUMMARY_PATH
        passed_ids_key = "vstar_results"
        output_subdir = "vstar"
    elif args.dataset == 'mmstar':
        image_dir = MMSTAR_DATA_DIR
        summary_file = args.summary_path if args.summary_path else MMSTAR_SUMMARY_PATH
        passed_ids_key = "mmstar_results"
        output_subdir = "mmstar"
    else:
        image_dir = MMVP_IMAGE_DIR
        summary_file = args.summary_path if args.summary_path else MMVP_SUMMARY_PATH
        passed_ids_key = "mmvp_results"
        output_subdir = "mmvp"

    OUTPUT_DIR = os.path.join(OUTPUT_DIR_BASE, output_subdir)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}")

    # Load passed_ids from summary.json
    print(f"\nLoading passed_ids from {summary_file}...")
    passed_ids = get_passed_ids_from_summary(summary_file, passed_ids_key)
    if not passed_ids:
        print("Error: No passed_ids found. Exiting.")
        return
    print(f"Loaded {len(passed_ids)} passed_ids")

    # Load dataset
    if args.dataset == 'vstar':
        dataset = load_vstar()
    elif args.dataset == 'mmstar':
        dataset = load_mmstar()
    else:
        dataset = load_mmvp()
    print(f"Dataset: {len(dataset)} samples total")

    # Filter to passed_ids only
    dataset = [item for item in dataset if int(item['id']) in passed_ids]
    print(f"Testing on {len(dataset)} samples (model-correct only)")

    results = []
    processed_ids = set()
    for fname in os.listdir(OUTPUT_DIR):
        if fname.endswith('_adv.png'):
            try:
                pid = int(fname.split('_')[0])
                processed_ids.add(pid)
            except ValueError:
                pass
    if processed_ids:
        print(f"Found {len(processed_ids)} already processed IDs in {OUTPUT_DIR}, will skip them")

    dataset = [item for item in dataset if item['id'] not in processed_ids]

    # Limit samples if specified
    if args.max_samples is not None and args.max_samples < len(dataset):
        dataset = dataset[:args.max_samples]
        print(f"Limited to first {args.max_samples} samples")

    for i, item in enumerate(tqdm(dataset, desc="Attacking")):
        img_path = os.path.join(image_dir, item['image'])
        question = item['query']
        correct = item['label']

        try:
            orig_img, final_delta, early_stopped, success_iter = attack_sample_logit_prob_white(
                model, processor, config, img_path, question, correct, device,
                dataset_type=args.dataset,
                validate_max_new_tokens=args.validate_max_new_tokens,
                validate_temp_dir=args.validate_temp_dir,
            )

            save_path = os.path.join(OUTPUT_DIR, f"{item['id']}_adv.png")
            save_adv_image(orig_img, final_delta, save_path)

            results.append({
                'id': item['id'],
                'gt': correct,
                'early_stopped': early_stopped,
                'success_iter': success_iter,
                'iters_run': (success_iter + 1) if early_stopped else NUM_ITERS,
            })

            tag = "EARLY" if early_stopped else "FULL"
            print(f"[{i}] id={item['id']} GT={correct} | [{tag} "
                  f"success_iter={success_iter if early_stopped else 'n/a'}] "
                  f"| Saved to {save_path}")

        except Exception as e:
            print(f"[{i}] id={item['id']} Error: {e}")
            import traceback
            traceback.print_exc()

    result_path = os.path.join(OUTPUT_DIR, "attack_results.json")
    all_results = list(results)
    if os.path.exists(result_path):
        print(f"Loading existing results from {result_path}...")
        with open(result_path, 'r') as f:
            existing = json.load(f)
        existing_results = existing.get('results', [])
        print(f"Loaded {len(existing_results)} existing results, merging with {len(results)} new results...")
        existing_ids = set(r['id'] for r in existing_results)
        for r in results:
            if r['id'] not in existing_ids:
                all_results.append(r)
        print(f"Merged to {len(all_results)} total results")
    else:
        print(f"No existing results file, saving {len(results)} new results...")
    print(f"Writing {len(all_results)} results to {result_path}...")
    # 数一下这次运行里"早停成功"的样本数，并把阈值也写进 JSON，方便核对
    early_count = sum(1 for r in all_results if r.get('early_stopped'))
    try:
        with open(result_path, 'w') as f:
            json.dump({
                'passed_ids_count': len(passed_ids),
                'early_stopped_count': early_count,
                'logit_gap_threshold': args.logit_gap_threshold,
                'validate_max_new_tokens': args.validate_max_new_tokens,
                'results': all_results
            }, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
    except Exception as e:
        print(f"Error writing results: {e}")
    print(f"Results saved to {result_path} (early_stopped={early_count}/{len(all_results)})")


if __name__ == "__main__":
    main()