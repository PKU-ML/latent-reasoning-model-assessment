"""
================================================================================
LVR 模型图像对抗攻击 - 隐向量解码logit攻击 v6
================================================================================

攻击原理：
- 对像素图像添加扰动
- 损失函数：prompt中倒数第6个token的隐向量经解码矩阵后，错误选项的logit概率
- 目标：最大化错误选项的logit概率，使模型倾向于选择错误答案
- 验证模式：仍然使用<|lvr_start|>开启多步推理，最后才输出答案
- 与v5的区别：
    * delta 定义在原图像素空间（原始分辨率，[0,1]），初始化为零
    * resize 改用 F.interpolate（可微），梯度可完整反传回原图像素
    * 归一化在 resize 之后手动完成（与 processor 等价）
    * 保存对抗图像时保留原始分辨率（不再是模型处理分辨率）

================================================================================
"""

import gc
import io
import torch
import torch.nn.functional as F
import json
import os
from tqdm import tqdm
import torchvision
from torchvision import transforms
from PIL import Image
import argparse

from transformers import AutoProcessor, AutoConfig

from src.model.qwen_lvr_model import QwenWithLVR
from src.train.monkey_patch_forward_lvr import replace_qwen2_5_with_mixed_modality_forward_lvr
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
MODEL_PATH = "/root/autodl-tmp/attack/lvr/models--vincentleebang--LVR-7B"

EPSILON = 4 / 255.0
STEP_SIZE = 0.5 / 255.0
NUM_ITERS = 300

MMVP_IMAGE_DIR = "/root/autodl-tmp/attack/data/MMVP/MMVP Images"
MMVP_CSV = "/root/autodl-tmp/attack/data/MMVP/Questions.csv"

VSTAR_DATA_DIR = "/root/autodl-tmp/attack/data/vstar"
VSTAR_JSONL = "/root/autodl-tmp/attack/data/vstar/test_questions.jsonl"

MMSTAR_DATA_DIR = "/root/autodl-tmp/attack/data/MMStar"
MMSTAR_METADATA = "/root/autodl-tmp/attack/data/MMStar/metadata.json"

OUTPUT_DIR_BASE = "/root/autodl-tmp/attack/lvr/adv_images_white"
os.makedirs(OUTPUT_DIR_BASE, exist_ok=True)

MMVP_SUMMARY_PATH = "test_results/org/run_steps/summary.json"
VSTAR_SUMMARY_PATH = "test_results/org/run_steps/summary.json"
MMSTAR_SUMMARY_PATH = "test_results/org/run_steps/summary.json"

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


def get_passed_ids_from_summary(summary_path, passed_ids_key="mmvp_results"):
    """Load passed_ids from summary.json."""
    if not os.path.exists(summary_path):
        print(f"Warning: Summary file not found: {summary_path}")
        return None
    with open(summary_path, 'r') as f:
        summary = json.load(f)
    if passed_ids_key in summary and "passed_ids" in summary[passed_ids_key]['steps_4']:
        return set(summary[passed_ids_key]['steps_4']["passed_ids"])
    print(f"Warning: No passed_ids found for '{passed_ids_key}' in {summary_path}")
    return None


def extract_answer_tagged(response):
    if '<answer>' in response:
        given = response.split('<answer>')[-1].split('</answer>')[0].strip()
    else:
        given = response.strip()
    if " " in given:
        given = given.split(" ")[0]
    if len(given) > 1:
        given = given[0]
    return given.upper()


def get_task_instruction():
    return "\nAnswer with the option's letter from the given choices directly."


def setup_model():
    config = AutoConfig.from_pretrained(MODEL_PATH)
    replace_qwen2_5_with_mixed_modality_forward_lvr(
        inference_mode=True, lvr_head=config.lvr_head
    )
    model = QwenWithLVR.from_pretrained(
        MODEL_PATH, config=config, trust_remote_code=True,
        dtype="auto", device_map="cuda:0"
    )
    processor = AutoProcessor.from_pretrained(MODEL_PATH)

    tokenizer = processor.tokenizer
    model.config.lvr_start_id = tokenizer.convert_tokens_to_ids("<|lvr_start|>")
    model.config.lvr_end_id = tokenizer.convert_tokens_to_ids("<|lvr_end|>")

    return model, processor, config


def save_adv_image_v6(orig_img, delta, save_path):
    """
    将扰动叠加到原图后保存，保留原始分辨率。
    orig_img: [3, H_orig, W_orig]，[0,1]，CPU 或 GPU tensor
    delta:    与 orig_img 同 shape，[0,1] 空间的扰动
    """
    adv = (orig_img.cpu() + delta.detach().cpu()).clamp(0, 1)
    img_np = (adv.permute(1, 2, 0).numpy() * 255).astype('uint8')
    Image.fromarray(img_np).save(save_path)


def make_verify_pixels(orig_img, delta, text_formatted, processor, visual_device, image_grid_thw):
    """
    验证推理：用 processor 处理 (orig_img + delta) 的 PIL 图片，不用 PNG 往返。
    返回 (pixel_values, image_grid_thw)。
    """
    # (orig + delta) → PIL 图片
    adv = (orig_img.cpu() + delta.detach().cpu()).clamp(0, 1)
    img_np = (adv.permute(1, 2, 0).numpy() * 255).astype('uint8')
    pil_img = Image.fromarray(img_np)

    # processor 处理
    inputs_tmp = processor(
        text=[text_formatted],
        images=[pil_img],
        padding=True,
        return_tensors="pt"
    )
    pixel_values = inputs_tmp['pixel_values'].to(visual_device)
    processor_grid_thw = inputs_tmp['image_grid_thw'].to(visual_device)

    return pixel_values, processor_grid_thw


# ============ 核心攻击函数 ============

def attack_sample_logit_prob_v6(model, processor, config, img_path, question, correct_answer, device, dataset_type='mmvp'):
    """
    v6：delta 定义在原图像素空间，通过可微 F.interpolate 传递梯度。

    损失函数：prompt中倒数第6个token的隐向量经lm_head解码后，错误选项对应的logit
    目标：最大化错误选项的logit，使模型倾向于选择错误答案

    dataset_type:
        - 'mmvp': 二选一，交换A/B作为错误答案
        - 'vstar': 四选一，选择模型预测logit概率最大的那个错误选项作为攻击目标
    """

    # 准备输入
    task_instruction = get_task_instruction()
    text = question.replace('(a)', 'A.').replace('(b)', 'B.')
    text = text + task_instruction

    # 获取设备
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
    # pixel_values 仅用于 clean inference，不参与攻击梯度计算
    pixel_values_clean = inputs['pixel_values'].to(visual_device)
    image_grid_thw = inputs['image_grid_thw'].to(visual_device)

    # ============ 从 image_grid_thw 得到模型处理分辨率 ============
    T, H, W = image_grid_thw[0].tolist()
    T, H, W = int(T), int(H), int(W)
    patch_size, temporal_patch_size, merge_size, channel = 14, 2, 2, 3
    # 模型处理分辨率（单帧像素尺寸）
    H_target = H * patch_size   # e.g. 336
    W_target = W * patch_size   # e.g. 336

    # ============ 加载原图为 [0,1] tensor（原始分辨率）============
    orig_img = transforms.ToTensor()(Image.open(img_path).convert('RGB'))  # [3, H_orig, W_orig]
    orig_img = orig_img.to(visual_device)

    print(f"    Original image size: {orig_img.shape[2]}x{orig_img.shape[1]}")
    print(f"    Model target size:   {W_target}x{H_target}")

    # CLIP 归一化参数（在 visual_device 上）
    mean = torch.tensor(CLIP_MEAN, device=visual_device).view(3, 1, 1)
    std  = torch.tensor(CLIP_STD,  device=visual_device).view(3, 1, 1)

    def preprocess(img_01):
        """
        [3, H_orig, W_orig] [0,1] → patchified pixel_values
        完全可微：F.interpolate + 归一化 + pixel_reshape
        """
        # resize（可微）
        resized = F.interpolate(
            img_01.unsqueeze(0), size=(H_target, W_target),
            mode='bilinear', align_corners=False
        )[0]  # [3, H_target, W_target]
        # CLIP 归一化
        normalized = (resized - mean) / std  # [3, H_target, W_target]
        # patchify
        pv, grid_info = pixel_reshape(normalized, patch_size=patch_size,
                               merge_size=merge_size, temporal_patch_size=temporal_patch_size)
        # 返回修正后的 image_grid_thw（因为 pixel_reshape 可能改变了 temporal 维度）
        grid_t, grid_h, grid_w = grid_info
        corrected_image_grid_thw = torch.tensor([[grid_t, grid_h, grid_w]], device=visual_device)
        return pv, corrected_image_grid_thw

    # 获取tokenizer
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
    special_tokens = ['<|im_start|>', '<|im_end|>', '<|vision_start|>', '<|vision_end|>', '<|lvr_start|>', '<|lvr_end|>', '<|lvr|>', '<|end|>']
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

    lm_head_weight = model.lm_head.weight  # [vocab_size, hidden_dim]

    # ============ 判断错误答案 ============
    # VSTAR/MMStar: 先做一次前向传播确定哪个错误选项logit最高
    # MMVP: 使用交换策略（A↔B）
    if dataset_type in ['vstar', 'mmstar']:
        print(f"    [{dataset_type.upper()}] Running initial forward to determine attack target...")
        with torch.no_grad():
            adv_orig_init = orig_img.clamp(0, 1)
            adv_pixels_init, corrected_image_grid_thw_init = preprocess(adv_orig_init)
            adv_pixels_init = adv_pixels_init.to(visual_device)

            outputs_init = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=adv_pixels_init,
                image_grid_thw=corrected_image_grid_thw_init,
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

        print(f"    [{dataset_type.upper()}] Correct: {correct_answer}, Selected wrong target: '{wrong_answer}' (logit: {max_wrong_logit:.4f})")
    else:
        # MMVP: 使用交换策略（A↔B）
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

    # ============ v6: delta 定义在原图像素空间，初始化为零 ============
    delta = torch.zeros_like(orig_img, requires_grad=True)
    print(f"    [DEBUG] orig_img shape: {orig_img.shape}")
    print(f"    [DEBUG] delta shape: {delta.shape}  (original image space, [0,1])")

    best_loss = float('-inf')
    best_delta = None
    best_ans = None
    best_is_wrong = False

    for iteration in range(NUM_ITERS):
        # ============ v6: 原图 + delta → 可微 resize → 归一化 → patchify ============
        adv_orig = (orig_img + delta).clamp(0, 1)   # [3, H_orig, W_orig]，[0,1]
        adv_pixels, corrected_image_grid_thw = preprocess(adv_orig)  # patchified，计算图连通 delta
        adv_pixels = adv_pixels.to(visual_device)

        # 前向传播
        outputs_adv = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=adv_pixels,
            image_grid_thw=corrected_image_grid_thw,
            return_dict=True,
            output_hidden_states=True
        )

        # 取 answer_pos 位置的隐向量，经 lm_head 得 logit
        answer_hidden = outputs_adv.hidden_states[-1][0, answer_pos, :]  # [hidden_dim]
        logits = answer_hidden @ lm_head_weight.t()
        wrong_logit = logits[wrong_token_id]
        loss = wrong_logit

        # ============ 验证推理：使用与攻击时相同的 preprocess 处理 (orig+delta) ============
        with torch.no_grad():
            adv_verify = (orig_img + delta).clamp(0, 1)
            pv_verify, corrected_image_grid_thw_verify = preprocess(adv_verify)
            pv_verify = pv_verify.to(visual_device)
            #print(f"    [DEBUG model.generate] pv_verify.shape={pv_verify.shape}, corrected_image_grid_thw_verify={corrected_image_grid_thw_verify[0].tolist()}")
            generated_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pv_verify,
                image_grid_thw=corrected_image_grid_thw_verify,
                max_new_tokens=16,
                decoding_strategy="steps",
                lvr_steps=[4],
                return_dict_in_generate=True
            )
            sequences = generated_ids.sequences[0]
            generated_ids_trimmed = sequences[len(input_ids[0]):]
            answer_text = processor.tokenizer.decode(generated_ids_trimmed, skip_special_tokens=True)
            ans = extract_answer_tagged(answer_text)

            if correct_answer in ['A', 'B', 'C', 'D']:
                is_correct = (ans == correct_answer)
            else:
                try:
                    is_correct = (abs(float(ans) - float(correct_answer)) < 0.01)
                except:
                    is_correct = False

            is_wrong = not is_correct

            # 早停：loss >= 10 且攻击成功
            if loss.item() >= 6 and is_correct == False:
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

        # ============ v6: 反传梯度到原图空间，FGSM 更新 delta ============
        model.zero_grad()
        if delta.grad is not None:
            delta.grad.zero_()

        loss.backward()
        orig_grad = delta.grad   # ∂loss/∂delta，原图像素空间的梯度

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
        generated_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values_clean,
            image_grid_thw=image_grid_thw,
            max_new_tokens=16,
            decoding_strategy="steps",
            lvr_steps=[4],
            return_dict_in_generate=True
        )
        sequences = generated_ids.sequences[0]
        generated_ids_trimmed = sequences[len(input_ids[0]):]
        answer_clean = processor.tokenizer.decode(generated_ids_trimmed, skip_special_tokens=True)

    # 验证：adv inference
    print(f"    Running adv inference...")
    with torch.no_grad():
        adv_final = (orig_img + final_delta).clamp(0, 1)
        pv_adv, corrected_image_grid_thw_adv = preprocess(adv_final)
        pv_adv = pv_adv.to(visual_device)
        print(f"    [DEBUG] pv_adv.shape={pv_adv.shape}, corrected_image_grid_thw_adv={corrected_image_grid_thw_adv[0].tolist()}")
        generated_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pv_adv,
            image_grid_thw=corrected_image_grid_thw_adv,
            max_new_tokens=16,
            decoding_strategy="steps",
            lvr_steps=[4],
            return_dict_in_generate=True
        )
        sequences = generated_ids.sequences[0]
        generated_ids_trimmed = sequences[len(input_ids[0]):]
        answer_adv = processor.tokenizer.decode(generated_ids_trimmed, skip_special_tokens=True)

    print(f"    Clean answer: '{answer_clean}'")
    print(f"    Adv answer: '{answer_adv}'")

    ans_clean = extract_answer_tagged(answer_clean)
    ans_adv = extract_answer_tagged(answer_adv)

    if correct_answer in ['A', 'B', 'C', 'D']:
        clean_correct = (ans_clean == correct_answer)
        adv_correct = (ans_adv == correct_answer)
    else:
        clean_correct = (abs(float(ans_clean) - float(correct_answer)) < 0.01)
        adv_correct = (abs(float(ans_adv) - float(correct_answer)) < 0.01)

    attack_success = clean_correct and not adv_correct

    # 返回 orig_img + final_delta（原图分辨率），供保存
    return orig_img, final_delta, answer_clean, answer_adv, attack_success


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=1)
    parser.add_argument('--max_samples', type=int, default=200)
    parser.add_argument('--iters', type=int, default=50)
    parser.add_argument('--summary_path', type=str, default=None)
    parser.add_argument('--dataset', type=str, default='mmvp', choices=['mmvp', 'vstar', 'mmstar'],
                        help='Dataset to attack: mmvp, vstar, or mmstar')
    args = parser.parse_args()

    global NUM_ITERS
    NUM_ITERS = args.iters

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
    # 获取已处理的问题id（从输出目录下的图片文件）
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

        # 跳过已处理的问题
        if item['id'] in processed_ids:
            print(f"[{i}] Skipping id={item['id']} (already processed)")
            continue

        try:
            orig_img, final_delta, ans_clean, ans_adv, success = attack_sample_logit_prob_v6(
                model, processor, config, img_path, question, correct, device,
                dataset_type=args.dataset
            )

            save_path = os.path.join(OUTPUT_DIR, f"{item['id']}_adv.png")
            save_adv_image_v6(orig_img, final_delta, save_path)

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

    result_path = os.path.join(OUTPUT_DIR, f"attack_results_{args.dataset}.json")
    # 合并新结果与已有结果
    all_results = results
    if os.path.exists(result_path):
        with open(result_path, 'r') as f:
            existing = json.load(f)
            existing_ids = [r['id'] for r in existing.get('results', [])]
            for r in results:
                if r['id'] not in existing_ids:
                    all_results.append(r)
    with open(result_path, 'w') as f:
        json.dump({'success_rate': success_rate, 'results': all_results}, f, indent=2)
    print(f"Results saved to {result_path}")


if __name__ == "__main__":
    import argparse as _ap
    _pre = _ap.ArgumentParser(add_help=False)
    _pre.add_argument('--gpu', type=int, default=1)
    _args, _ = _pre.parse_known_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(_args.gpu)
    main()
