"""
================================================================================
Monet 模型图像对抗攻击 - 隐向量解码logit攻击 white box 版本
================================================================================

攻击原理：
- 对像素图像添加扰动
- 损失函数：prompt中倒数第6个token的隐向量经解码矩阵后，错误选项的logit概率
- 目标：最大化错误选项的logit概率，使模型倾向于选择错误答案
- 与 LVR v6 的区别：
    * Monet 使用 <abs_vis_token> 作为 latent token
    * Monet 的 _sample 方法实现了隐式推理循环
    * delta 定义在原图像素空间（原始分辨率，[0,1]），初始化为零
    * resize 改用 F.interpolate（可微），梯度可完整反传回原图像素
    * 归一化在 resize 之后手动完成（与 processor 等价）

================================================================================
"""

import gc
import io
from pathlib import Path
import torch
import torch.nn.functional as F
import json
import os
from tqdm import tqdm
import torchvision
from torchvision import transforms
from PIL import Image
import argparse

import sys
monet_src_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
sys.path.insert(0, monet_src_path)

from src.train.monkey_patch_forward_monet_test import replace_qwen2_5_with_monet_forward
replace_qwen2_5_with_monet_forward()

from transformers import AutoProcessor, AutoConfig
from qwen_vl_utils import process_vision_info

from src.model.monet import MonetModel

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
MODEL_PATH = os.environ.get("MONET_MODEL_PATH", "NOVAglow646/Monet-7B")

EPSILON = 4 / 255.0
STEP_SIZE = 0.5 / 255.0
NUM_ITERS = 300

_DATA_ROOT = os.environ.get("DATA_DIR", str(Path(__file__).resolve().parent.parent / "data"))
VSTAR_DATA_DIR = os.environ.get("MONET_VSTAR_DATA_DIR", str(Path(_DATA_ROOT) / "vstar"))
VSTAR_JSONL = os.environ.get("MONET_VSTAR_JSONL", str(Path(_DATA_ROOT) / "vstar" / "test_questions.jsonl"))

MMVP_IMAGE_DIR = os.environ.get("MONET_MMVP_IMAGE_DIR", str(Path(_DATA_ROOT) / "MMVP" / "MMVP Images"))
MMVP_CSV = os.environ.get("MONET_MMVP_CSV", str(Path(_DATA_ROOT) / "MMVP" / "Questions.csv"))

MMSTAR_DATA_DIR = os.environ.get("MONET_MMSTAR_DATA_DIR", str(Path(_DATA_ROOT) / "MMStar"))
MMSTAR_METADATA = os.environ.get("MONET_MMSTAR_METADATA", str(Path(_DATA_ROOT) / "MMStar" / "metadata.json"))

OUTPUT_DIR_BASE = os.environ.get("MONET_OUTPUT_DIR", str(Path(__file__).resolve().parent / "adv_images_white"))

SUMMARY_PATH = os.environ.get(
    "MONET_SUMMARY_PATH",
    str(Path(__file__).resolve().parent / "test_results" / "org" / "monet_latent010" / "summary.json"),
)

MMVP_SUMMARY_PATH = os.environ.get("MONET_MMVP_SUMMARY_PATH", "test_results/org/monet_latent010/summary.json")
VSTAR_SUMMARY_PATH = os.environ.get("MONET_VSTAR_SUMMARY_PATH", "test_results/org/monet_latent010/summary.json")
MMSTAR_SUMMARY_PATH = os.environ.get("MONET_MMSTAR_SUMMARY_PATH", "test_results/org/monet_latent010/summary.json")
os.makedirs(OUTPUT_DIR_BASE, exist_ok=True)

# CLIP 归一化参数（与 processor 一致）
CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD  = [0.26862954, 0.26130258, 0.27577711]


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
    """Load MMStar dataset from local metadata JSON."""
    import json
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
            'l2_category': item.get("l2_category", "")
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


def extract_answer(response):
    """Extract answer letter from model response."""
    if not response:
        return ""

    response = response.strip()

    boxed_match = re.search(r'\\boxed\{([^}]+)\}', response)
    if boxed_match:
        answer = boxed_match.group(1).strip()
        if len(answer) > 1:
            answer = answer[0]
        return answer.upper()

    paren_matches = re.findall(r'\(([A-Z])\)', response)
    if paren_matches:
        return paren_matches[-1]

    lines = [l.strip() for l in response.split('\n') if l.strip()]
    if lines:
        first_line = lines[0]
        if len(first_line) == 1 and first_line in 'ABCD':
            return first_line
        option_match = re.match(r'^([A-Z])\.', first_line)
        if option_match:
            return option_match.group(1)

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

    standalone_match = re.search(r'\b([A-D])\b', response)
    if standalone_match:
        return standalone_match.group(1)

    return ""


def get_task_instruction():
    return "\nAnswer with the option's letter from the given choices directly. Only output the letter (e.g., A, B, C, or D), do not include any explanation."


def setup_model():
    config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)

    model = MonetModel.from_pretrained(
        MODEL_PATH,
        config=config,
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )
    model = model.to("cuda:0")

    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)

    processor.tokenizer.add_tokens("<abs_vis_token>", special_tokens=True)
    processor.tokenizer.add_tokens("</abs_vis_token>", special_tokens=True)

    latent_start_idx = processor.tokenizer("<abs_vis_token>", return_tensors="pt")["input_ids"][0]
    model.config.latent_token_id = int(latent_start_idx[0]) if len(latent_start_idx) > 0 else 151666
    model.config.latent_start_id = int(latent_start_idx[0]) if len(latent_start_idx) > 0 else 151666
    model.config.max_latent_steps = 10

    model.eval()
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    if hasattr(model, 'visual'):
        model.visual_requires_grad = False

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


def make_verify_pixels(orig_img, delta, text_formatted, processor, visual_device):
    """
    与 test_org.py 完全对齐的预处理管道，用于循环内验证推理。

    流程：
      1. (orig + delta) → uint8 截断（模拟 PNG 保存的量化误差）
      2. 内存中 PNG 往返（BytesIO，无磁盘 I/O）→ PIL 图像
      3. processor 完整预处理：PIL smart-resize(bicubic) + CLIP 归一化 + patchify
    返回的 pixel_values 与 test_org 读取磁盘 PNG 后的结果数值一致。
    """
    adv = (orig_img.cpu() + delta.detach().cpu()).clamp(0, 1)
    img_np = (adv.permute(1, 2, 0).numpy() * 255).astype('uint8')
    pil_img = Image.fromarray(img_np)

    buf = io.BytesIO()
    pil_img.save(buf, format='PNG')
    buf.seek(0)
    pil_img = Image.open(buf).convert('RGB')

    inputs_tmp = processor(
        text=[text_formatted],
        images=[pil_img],
        padding=True,
        return_tensors="pt"
    )
    return inputs_tmp['pixel_values'].to(visual_device)


# ============ 核心攻击函数 ============

def attack_sample_logit_prob_white(model, processor, config, img_path, question, correct_answer, device, dataset_type='mmvp'):
    """
    Monet white box 攻击：delta 定义在原图像素空间，通过可微 F.interpolate 传递梯度。

    损失函数：prompt中倒数第6个token的隐向量经lm_head解码后，错误选项（A或B）对应的logit
    目标：最大化错误选项的logit，使模型倾向于选择错误答案

    dataset_type:
        - 'mmvp': 二选一，交换A/B作为错误答案
        - 'vstar': 四选一，选择模型预测logit概率最大的那个错误选项作为攻击目标
    """

    import re
    task_instruction = get_task_instruction()
    text = question.replace('(a)', 'A.').replace('(b)', 'B.')
    text = text + task_instruction

    visual_device = next(model.model.visual.parameters()).device
    lm_device = next(model.lm_head.parameters()).device

    # ============ 运行 processor 获取 text 相关输入和 image_grid_thw ============
    messages = [{'role': 'user', 'content': [
        {'type': 'image', 'image': img_path},
        {'type': 'text', 'text': text}
    ]}]
    text_formatted = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text_formatted], images=image_inputs, videos=video_inputs,
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
        resized = F.interpolate(
            img_01.unsqueeze(0), size=(H_target, W_target),
            mode='bilinear', align_corners=False
        )[0]
        normalized = (resized - mean) / std
        pv, _ = pixel_reshape(normalized, patch_size=patch_size,
                               merge_size=merge_size, temporal_patch_size=temporal_patch_size)
        return pv

    tokenizer = processor.tokenizer

    # ============ 找到倒数第6个token的位置作为answer_pos ============
    input_ids_list = input_ids[0].tolist()
    answer_pos = len(input_ids_list) - 6

    print(f"    Total tokens: {len(input_ids_list)}")
    print(f"    answer_pos (6th from last): {answer_pos}")
    print(f"    Tokens at answer_pos~answer_pos+5: {[tokenizer.decode([tid]) for tid in input_ids_list[answer_pos:answer_pos+6]]}")

    # ============ DEBUG: 输出完整的input_ids结构 ============
    print(f"\n    [DEBUG] Full input_ids shape: {input_ids.shape}")
    print(f"    [DEBUG] Full prompt tokens:")
    all_tokens = input_ids_list
    special_tokens = ['<|im_start|>', '<|im_end|>', '<|vision_start|>', '<|vision_end|>', '<abs_vis_token>', '</abs_vis_token>', '<|end|>']
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
    # VSTAR/MMStar: 先做一次前向传播确定哪个错误选项logit最高
    # MMVP: 使用交换策略（A↔B）
    if dataset_type in ['vstar', 'mmstar']:
        print(f"    [{dataset_type.upper()}] Running initial forward to determine attack target...")
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
            answer_hidden_init = outputs_init.hidden_states[-1][answer_pos, :]
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

        print(f"    [{dataset_type.upper()}] Correct: {correct_answer}, Selected wrong target: '{wrong_answer}' (logit: {max_wrong_logit:.4f})")
    else:
        # MMVP: 交换策略
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

    # ============ delta 定义在原图像素空间，初始化为零 ============
    delta = torch.zeros_like(orig_img, requires_grad=True)
    print(f"    [DEBUG] orig_img shape: {orig_img.shape}")
    print(f"    [DEBUG] delta shape: {delta.shape}  (original image space, [0,1])")

    best_loss = float('-inf')
    best_delta = None
    best_ans = None
    best_is_wrong = False

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
        answer_hidden = outputs_adv.hidden_states[-1][answer_pos, :]
        logits = answer_hidden @ lm_head_weight.t()
        wrong_logit = logits[wrong_token_id]
        loss = wrong_logit

        # ============ DEBUG: 打印 iteration=0 时 logits 的 top-k 概率 ============
        '''if iteration == 0:
            probs = torch.softmax(logits, dim=-1)
            top_k = 10
            top_values, top_indices = torch.topk(probs, top_k)
            print(f"    [DEBUG Iter 0] Top-{top_k} token probabilities:")
            for idx, (prob, token_id) in enumerate(zip(top_values.tolist(), top_indices.tolist())):
                token_str = tokenizer.decode([token_id]).replace('\n', '\\n')
                print(f"      {idx+1}. token_id={token_id}, prob={prob:.6f}, text={repr(token_str)}")
            print(f"    [DEBUG Iter 0] wrong_answer='{wrong_answer}', wrong_token_id={wrong_token_id}, wrong_prob={probs[wrong_token_id].item():.6f}")'''

        # ============ 验证推理：走与 test_org 对齐的完整 processor 管道 ============
        with torch.no_grad():
            pv_verify = make_verify_pixels(orig_img, delta, text_formatted, processor, visual_device)

            # 构建带 <abs_vis_token> 的输入（与 test_org.py 一致）
            latent_token_id = model.config.latent_token_id
            input_ids_list = input_ids[0].tolist()
            input_ids_list.append(latent_token_id)
            input_ids_with_latent = torch.tensor([input_ids_list], dtype=torch.long, device=lm_device)
            attention_mask_with_latent = torch.tensor([[1] * len(input_ids_list)], dtype=torch.long, device=lm_device)

            generated_ids = model.generate(
                input_ids=input_ids_with_latent,
                attention_mask=attention_mask_with_latent,
                pixel_values=pv_verify,
                image_grid_thw=image_grid_thw,
                max_new_tokens=16,
                do_sample=False,
                pad_token_id=processor.tokenizer.pad_token_id,
                eos_token_id=processor.tokenizer.eos_token_id,
                return_dict_in_generate=True
            )
            sequences = generated_ids[0]
            generated_ids_trimmed = sequences[-1][len(input_ids_with_latent[0]):]
            print(generated_ids_trimmed)
            answer_text = processor.tokenizer.decode(generated_ids_trimmed, skip_special_tokens=True)
            print(answer_text)
            ans = extract_answer(answer_text)

            if correct_answer in ['A', 'B', 'C', 'D']:
                is_correct = (ans == correct_answer)
            else:
                try:
                    is_correct = (abs(float(ans) - float(correct_answer)) < 0.01)
                except:
                    is_correct = False

            is_wrong = not is_correct

            # 早停：loss >= 25 且攻击成功
            if loss.item() >= 25 and is_correct == False:
                print(f"    [Early Stop] Iter {iteration}: loss={loss.item():.6f} >= 10, attack successful (ans='{ans.strip()}', GT={correct_answer})")
                best_loss = loss.item()
                best_delta = delta.detach().clone()
                best_ans = ans
                best_is_wrong = True
                break

            if iteration % 10 == 0:
                wrong_prob = torch.softmax(logits, dim=-1)[wrong_token_id].item()
                print(f"    Iter {iteration}: loss={loss.item():.6f}, wrong_logit={wrong_logit.item():.6f}, wrong_prob={wrong_prob:.4f}, ans='{ans.strip()}', GT={correct_answer}, correct={is_correct}")

        # 保存最优 delta（优先攻击成功 + loss 最大）
        if is_wrong and loss.item() > best_loss:
            best_loss = loss.item()
            best_delta = delta.detach().clone()
            best_ans = ans
            best_is_wrong = True

        if best_delta is None or (not best_is_wrong and loss.item() > best_loss):
            best_loss = loss.item()
            best_delta = delta.detach().clone()
            best_ans = ans
            best_is_wrong = is_wrong

        # ============ 反传梯度到原图空间，FGSM 更新 delta ============
        model.zero_grad()
        if delta.grad is not None:
            delta.grad.zero_()

        loss.backward()
        orig_grad = delta.grad

        if orig_grad is None:
            print(f"    Warning: orig_grad is None at iteration {iteration}")
            orig_grad = torch.zeros_like(delta)

        with torch.no_grad():
            delta = torch.clamp(
                delta + STEP_SIZE * orig_grad.sign(),
                -EPSILON, EPSILON
            ).detach().requires_grad_(True)

    if best_delta is not None:
        final_delta = best_delta
    else:
        final_delta = delta.detach()

    print(f"    Best result: best_loss={best_loss:.6f}, best_ans='{best_ans.strip() if best_ans else 'N/A'}', is_wrong={best_is_wrong}")

    # 验证：clean inference
    print(f"    Running clean inference...")
    with torch.no_grad():
        latent_token_id = model.config.latent_token_id
        input_ids_list = input_ids[0].tolist()
        input_ids_list.append(latent_token_id)
        input_ids_with_latent = torch.tensor([input_ids_list], dtype=torch.long, device=lm_device)
        attention_mask_with_latent = torch.tensor([[1] * len(input_ids_list)], dtype=torch.long, device=lm_device)

        generated_ids = model.generate(
            input_ids=input_ids_with_latent,
            attention_mask=attention_mask_with_latent,
            pixel_values=pixel_values_clean,
            image_grid_thw=image_grid_thw,
            max_new_tokens=16,
            do_sample=False,
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
            return_dict_in_generate=True
        )
        sequences = generated_ids[0]
        generated_ids_trimmed = sequences[-1][len(input_ids_with_latent[0]):]
        answer_clean = processor.tokenizer.decode(generated_ids_trimmed, skip_special_tokens=True)

    # 验证：adv inference
    print(f"    Running adv inference...")
    with torch.no_grad():
        pv_adv_final = make_verify_pixels(orig_img, final_delta, text_formatted, processor, visual_device)
        generated_ids = model.generate(
            input_ids=input_ids_with_latent,
            attention_mask=attention_mask_with_latent,
            pixel_values=pv_adv_final,
            image_grid_thw=image_grid_thw,
            max_new_tokens=16,
            do_sample=False,
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
            return_dict_in_generate=True
        )
        sequences = generated_ids[0]
        generated_ids_trimmed = sequences[-1][len(input_ids_with_latent[0]):]
        answer_adv = processor.tokenizer.decode(generated_ids_trimmed, skip_special_tokens=True)

    print(f"    Clean answer: '{answer_clean}'")
    print(f"    Adv answer: '{answer_adv}'")

    ans_clean = extract_answer(answer_clean)
    ans_adv = extract_answer(answer_adv)

    if correct_answer in ['A', 'B', 'C', 'D']:
        clean_correct = (ans_clean == correct_answer)
        adv_correct = (ans_adv == correct_answer)
    else:
        clean_correct = (abs(float(ans_clean) - float(correct_answer)) < 0.01)
        adv_correct = (abs(float(ans_adv) - float(correct_answer)) < 0.01)

    attack_success = clean_correct and not adv_correct

    return orig_img, final_delta, answer_clean, answer_adv, attack_success


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--max_samples', type=int, default=None)
    parser.add_argument('--iters', type=int, default=50)
    parser.add_argument('--summary_path', type=str, default=None,
                        help="Path to summary.json for loading passed_ids")
    parser.add_argument('--dataset', type=str, default='mmvp', choices=['mmvp', 'vstar', 'mmstar'],
                        help='Dataset to attack: mmvp, vstar, or mmstar')
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

    # Determine image directory and summary path based on dataset type
    if args.dataset == 'vstar':
        image_dir = VSTAR_DATA_DIR
        summary_file = args.summary_path if args.summary_path else VSTAR_SUMMARY_PATH
        passed_ids_key = "vstar_result"
        output_subdir = "vstar"
    elif args.dataset == 'mmstar':
        image_dir = MMSTAR_DATA_DIR
        summary_file = args.summary_path if args.summary_path else MMSTAR_SUMMARY_PATH
        passed_ids_key = "mmstar_result"
        output_subdir = "mmstar"
    else:
        image_dir = MMVP_IMAGE_DIR
        summary_file = args.summary_path if args.summary_path else MMVP_SUMMARY_PATH
        passed_ids_key = "mmvp_result"
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

    dataset = [item for item in dataset if int(item['id']) not in processed_ids]

    # Limit samples if specified
    if args.max_samples is not None and args.max_samples < len(dataset):
        dataset = dataset[:args.max_samples]
        print(f"Limited to first {args.max_samples} samples")

    for i, item in enumerate(tqdm(dataset, desc="Attacking")):
        img_path = os.path.join(image_dir, item['image'])
        question = item['query']
        correct = item['label']

        try:
            orig_img, final_delta, ans_clean, ans_adv, success = attack_sample_logit_prob_white(
                model, processor, config, img_path, question, correct, device,
                dataset_type=args.dataset
            )

            save_path = os.path.join(OUTPUT_DIR, f"{item['id']}_adv.png")
            save_adv_image(orig_img, final_delta, save_path)

            results.append({
                'id': item['id'],
                'gt': correct,
                'clean_answer': ans_clean,
                'adv_answer': ans_adv,
                'attack_success': success
            })

            print(f"[{i}] id={item['id']} GT={correct} | Clean={ans_clean.strip()} | Adv={ans_adv.strip()} | Success={success}")

        except Exception as e:
            print(f"[{i}] id={item['id']} Error: {e}")
            import traceback
            traceback.print_exc()

    success_rate = sum(1 for r in results if r['attack_success']) / len(results) if results else 0
    print(f"\nAttack success rate: {success_rate:.2%} ({sum(1 for r in results if r['attack_success'])}/{len(results)})")

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
    try:
        with open(result_path, 'w') as f:
            json.dump({
                'success_rate': success_rate,
                'passed_ids_count': len(passed_ids),
                'results': all_results
            }, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
    except Exception as e:
        print(f"Error writing results: {e}")
    print(f"Results saved to {result_path}")


if __name__ == "__main__":
    import re
    main()