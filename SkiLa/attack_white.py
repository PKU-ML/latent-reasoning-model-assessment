"""
================================================================================
SkiLa 模型图像对抗攻击 - 隐向量解码logit攻击
================================================================================

攻击原理（参照lvr/attack_lvr_l2_v6.py）：
- 对像素图像添加扰动
- 损失函数：prompt中倒数第6个token的隐向量经解码矩阵后，错误选项的logit概率
- 目标：最大化错误选项的logit概率，使模型倾向于选择错误答案

与SkiLa/test_org.py对齐的预处理管道
================================================================================
"""

import gc
import io
from pathlib import Path
import torch
import re
import torch.nn.functional as F
import json
import os
from tqdm import tqdm
import torchvision
from torchvision import transforms
from PIL import Image
import argparse

from transformers import AutoProcessor, AutoConfig, SiglipImageProcessor

from src.model.skila import SkiLa
from src.model.sketch_extractor import SketchExtractor_Siglip
from src.train.monkey_patch_forward_skila import replace_qwen2_5_with_skila_forward
from qwen_vl_utils import process_vision_info

# 设置CUDA内存分配配置，减少碎片化
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# ============ 配置 ============
MODEL_PATH = os.environ.get("SKILA_MODEL_PATH", "./SkiLa-7B")
SKETCH_ENCODER = os.environ.get(
    "SKILA_SKETCH_ENCODER",
    "./models--google--siglip2-so400m-patch14-384/snapshots/e8e487298228002f3d8a82e0cd5c8ea9c567f57f",
)
SKETCH_TOKEN_NUM = 54  # 默认值

EPSILON = 4 / 255.0
STEP_SIZE = 0.5 / 255.0
NUM_ITERS = 300

_DATA_ROOT = os.environ.get("DATA_DIR", str(Path(__file__).resolve().parent.parent / "data"))
VSTAR_DATA_DIR = os.environ.get("SKILA_VSTAR_DATA_DIR", str(Path(_DATA_ROOT) / "vstar"))
VSTAR_JSONL = os.environ.get("SKILA_VSTAR_JSONL", str(Path(_DATA_ROOT) / "vstar" / "test_questions.jsonl"))

MMVP_IMAGE_DIR = os.environ.get("SKILA_MMVP_IMAGE_DIR", str(Path(_DATA_ROOT) / "MMVP" / "MMVP Images"))
MMVP_CSV = os.environ.get("SKILA_MMVP_CSV", str(Path(_DATA_ROOT) / "MMVP" / "Questions.csv"))

MMSTAR_DATA_DIR = os.environ.get("SKILA_MMSTAR_DATA_DIR", str(Path(_DATA_ROOT) / "MMStar"))
MMSTAR_METADATA = os.environ.get("SKILA_MMSTAR_METADATA", str(Path(_DATA_ROOT) / "MMStar" / "metadata.json"))

OUTPUT_DIR_BASE = os.environ.get(
    "SKILA_OUTPUT_DIR",
    str(Path(__file__).resolve().parent / "adv_images"),
)

SUMMARY_PATH = os.environ.get(
    "SKILA_SUMMARY_PATH",
    str(Path(__file__).resolve().parent / "test_results" / "org" / "summary.json"),
)

MMVP_SUMMARY_PATH = os.environ.get("SKILA_MMVP_SUMMARY_PATH", "test_results/org/summary.json")
VSTAR_SUMMARY_PATH = os.environ.get("SKILA_VSTAR_SUMMARY_PATH", "test_results/org/summary.json")
MMSTAR_SUMMARY_PATH = os.environ.get("SKILA_MMSTAR_SUMMARY_PATH", "test_results/org/summary.json")

# CLIP 归一化参数（与 processor 一致）
CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD  = [0.26862954, 0.26130258, 0.27577711]


# ============ 工具函数 ============

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


def extract_answer(response: str) -> str:
    """Extract the answer letter from model response - matches test_org.py exactly."""
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


def extract_answer_tagged(response):
    """Legacy wrapper for compatibility."""
    return extract_answer(response)


def get_task_instruction():
    return "\nAnswer with the option's letter from the given choices directly."


def setup_model():
    """Load SkiLa model with sketch extractor."""
    print(f"Loading SkiLa model from {MODEL_PATH}...")

    # Apply monkey patch
    replace_qwen2_5_with_skila_forward()

    config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)

    # Load SkiLa model
    model = SkiLa.from_pretrained(
        MODEL_PATH,
        config=config,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
    )

    # Load sketch extractor (Siglip-based)
    print(f"Loading sketch extractor: {SKETCH_ENCODER}")
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
    return model, processor, sketch_processor


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
    # 1. uint8 量化（与 save_adv_image 写盘相同的操作）
    adv = (orig_img.cpu() + delta.detach().cpu()).clamp(0, 1)
    img_np = (adv.permute(1, 2, 0).numpy() * 255).astype('uint8')
    pil_img = Image.fromarray(img_np)

    # 2. 内存中 PNG 往返：写入 BytesIO 再读回，确保量化一致
    buf = io.BytesIO()
    pil_img.save(buf, format='PNG')
    buf.seek(0)
    pil_img = Image.open(buf).convert('RGB')

    # 3. processor 完整管道（与 test_org.run_inference 完全一致）
    image_inputs, video_inputs = process_vision_info([{
        'role': 'user',
        'content': [
            {'type': 'image', 'image': pil_img},
            {'type': 'text', 'text': text_formatted}
        ]
    }])
    inputs_tmp = processor(
        text=[text_formatted],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    )
    return inputs_tmp['pixel_values'].to(visual_device)


# ============ 核心攻击函数 ============

def attack_sample_logit_prob_skila(model, processor, sketch_processor, img_path, question, correct_answer, device, dataset_type='mmvp'):
    """
    SkiLa 攻击：delta 定义在原图像素空间，通过可微 F.interpolate 传递梯度。

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
        pv, _ = pixel_reshape(normalized, patch_size=patch_size,
                               merge_size=merge_size, temporal_patch_size=temporal_patch_size)
        return pv

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
    special_tokens = ['<|im_start|>', '<|im_end|>', '<|vision_start|>', '<|vision_end|>', '<|skila|>', '<|sketch_start|>', '<|sketch_end|>', '<|end|>']
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

    # ============ 判断错误答案 ============
    # VSTAR/MMStar: 先做一次前向传播确定哪个错误选项logit最高
    # MMVP: 使用交换策略（A↔B）
    answer_hidden_for_wrong = None  # 延迟确定wrong_answer

    # lm_head_weight 需要提前定义，因为VSTAR/MMStar初始前向传播会用到
    lm_head_weight = model.lm_head.weight  # [vocab_size, hidden_dim]

    if dataset_type in ['vstar', 'mmstar']:
        # 先做一次前向传播，获取logits来确定攻击目标
        # 使用 torch.no_grad() 避免创建不必要的计算图，节省显存
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
            answer_hidden_init = outputs_init.hidden_states[-1][0, answer_pos, :]
            logits_init = answer_hidden_init @ lm_head_weight.t()

        # 释放初始前向传播的中间变量
        del outputs_init, adv_pixels_init
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 获取A/B/C/D的token ids
        abcd_token_ids = {}
        for opt in ['A', 'B', 'C', 'D']:
            tokens = tokenizer(opt, add_special_tokens=False, return_tensors='pt')
            if tokens['input_ids'].numel() > 0:
                abcd_token_ids[opt] = tokens['input_ids'][0].item()
            else:
                abcd_token_ids[opt] = tokenizer.encode(opt, add_special_tokens=False)[0]

        # 找出correct_answer之外的选项中logit最高的
        correct_upper = correct_answer.upper()
        wrong_options = [opt for opt in ['A', 'B', 'C', 'D'] if opt != correct_upper]

        max_wrong_logit = float('-inf')
        wrong_answer = 'A'  # 默认值
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
        # 清理GPU缓存
        torch.cuda.empty_cache()
        gc.collect()

        # ============ 原图 + delta → 可微 resize → 归一化 → patchify ============
        adv_orig = (orig_img + delta).clamp(0, 1)   # [3, H_orig, W_orig]，[0,1]
        adv_pixels = preprocess(adv_orig)            # patchified，计算图连通 delta
        adv_pixels = adv_pixels.to(visual_device)

        # 前向传播 - 获取hidden_states
        outputs_adv = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=adv_pixels,
            image_grid_thw=image_grid_thw,
            return_dict=True,
            output_hidden_states=True
        )

        # 取 answer_pos 位置的隐向量
        answer_hidden = outputs_adv.hidden_states[-1][0, answer_pos, :].clone()
        logits = answer_hidden @ lm_head_weight.t()
        wrong_logit = logits[wrong_token_id]
        loss = wrong_logit

        # 释放outputs_adv（但保留answer_hidden/logits用于反向传播）
        del outputs_adv
        torch.cuda.empty_cache()

        # ============ 反向传播 ============
        model.zero_grad()
        if delta.grad is not None:
            delta.grad.zero_()

        loss.backward(retain_graph=False)

        # 获取梯度
        orig_grad = delta.grad
        if orig_grad is None:
            print(f"    Warning: orig_grad is None at iteration {iteration}")
            orig_grad = torch.zeros_like(delta)

        # FGSM更新
        with torch.no_grad():
            delta = torch.clamp(
                delta + STEP_SIZE * orig_grad.sign(),
                -EPSILON, EPSILON
            ).detach().requires_grad_(True)

        # 清理本轮不再需要的变量（反向传播完成后）
        del answer_hidden, logits, wrong_logit, loss, orig_grad
        torch.cuda.empty_cache()

        # ============ 验证推理（每10次迭代做一次）============
        if iteration % 10 == 0 or iteration == NUM_ITERS - 1:
            with torch.no_grad():
                pv_verify = make_verify_pixels(orig_img, delta, text_formatted, processor, visual_device)

                think_start_id = processor.tokenizer.encode("<think>", add_special_tokens=False)[0]
                think_start_tensor = torch.tensor([[think_start_id]], device=input_ids.device)
                input_ids_for_gen = torch.cat([input_ids, think_start_tensor], dim=-1)
                attention_mask_for_gen = torch.cat([attention_mask, torch.ones_like(think_start_tensor)], dim=-1)

                generated_ids = model.generate(
                    input_ids=input_ids_for_gen,
                    attention_mask=attention_mask_for_gen,
                    pixel_values=pv_verify,
                    image_grid_thw=image_grid_thw,
                    max_new_tokens=1024,
                    do_sample=False,
                    pad_token_id=processor.tokenizer.pad_token_id,
                    eos_token_id=processor.tokenizer.eos_token_id,
                )

                original_len = input_ids.shape[1]
                trimmed_ids = generated_ids[0][original_len:]
                answer_text = processor.tokenizer.decode(trimmed_ids, skip_special_tokens=True)
                ans = extract_answer_tagged(answer_text)

                if correct_answer.upper() in ['A', 'B', 'C', 'D']:
                    is_correct = (ans.upper() == correct_answer.upper())
                else:
                    try:
                        is_correct = (abs(float(ans) - float(correct_answer)) < 0.01)
                    except:
                        is_correct = False

                is_wrong = not is_correct
                print(f"    Iter {iteration}: ans='{ans.strip()}', GT={correct_answer}, correct={is_correct}")

                # 保存最优delta并早停
                if is_wrong:
                    best_delta = delta.detach().clone()
                    best_ans = ans
                    best_is_wrong = is_wrong
                    print(f"    [Early Stop] Attack successful at iter {iteration}")
                    break

                del generated_ids, pv_verify, input_ids_for_gen, attention_mask_for_gen
                torch.cuda.empty_cache()

    if best_delta is not None:
        final_delta = best_delta
    else:
        final_delta = delta.detach()

    print(f"    Best result: best_loss={best_loss:.6f}, best_ans='{best_ans.strip() if best_ans else 'N/A'}', is_wrong={best_is_wrong}")

    # 验证：clean inference
    print(f"    Running clean inference...")
    with torch.no_grad():
        think_start_id = processor.tokenizer.encode("<think>", add_special_tokens=False)[0]
        think_start_tensor = torch.tensor([[think_start_id]], device=input_ids.device)
        input_ids_for_gen = torch.cat([input_ids, think_start_tensor], dim=-1)
        attention_mask_for_gen = torch.cat([attention_mask, torch.ones_like(think_start_tensor)], dim=-1)

        generated_ids = model.generate(
            input_ids=input_ids_for_gen,
            attention_mask=attention_mask_for_gen,
            pixel_values=pixel_values_clean,
            image_grid_thw=image_grid_thw,
            max_new_tokens=1024,
            do_sample=False,
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
        )
        original_len = input_ids.shape[1]
        trimmed_ids = generated_ids[0][original_len:]
        answer_clean = processor.tokenizer.decode(trimmed_ids, skip_special_tokens=True)

    # 验证：adv inference（走与 test_org 对齐的完整 processor 管道）
    print(f"    Running adv inference...")
    with torch.no_grad():
        pv_adv_final = make_verify_pixels(orig_img, final_delta, text_formatted, processor, visual_device)

        think_start_id = processor.tokenizer.encode("<think>", add_special_tokens=False)[0]
        think_start_tensor = torch.tensor([[think_start_id]], device=input_ids.device)
        input_ids_for_gen = torch.cat([input_ids, think_start_tensor], dim=-1)
        attention_mask_for_gen = torch.cat([attention_mask, torch.ones_like(think_start_tensor)], dim=-1)

        generated_ids = model.generate(
            input_ids=input_ids_for_gen,
            attention_mask=attention_mask_for_gen,
            pixel_values=pv_adv_final,
            image_grid_thw=image_grid_thw,
            max_new_tokens=1024,
            do_sample=False,
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
        )
        original_len = input_ids.shape[1]
        trimmed_ids = generated_ids[0][original_len:]
        answer_adv = processor.tokenizer.decode(trimmed_ids, skip_special_tokens=True)

    print(f"    Clean answer: '{answer_clean}'")
    print(f"    Adv answer: '{answer_adv}'")

    ans_clean = extract_answer_tagged(answer_clean)
    ans_adv = extract_answer_tagged(answer_adv)

    if correct_answer.upper() in ['A', 'B', 'C', 'D']:
        clean_correct = (ans_clean.upper() == correct_answer.upper())
        adv_correct = (ans_adv.upper() == correct_answer.upper())
    else:
        try:
            clean_correct = (abs(float(ans_clean) - float(correct_answer)) < 0.01)
            adv_correct = (abs(float(ans_adv) - float(correct_answer)) < 0.01)
        except:
            clean_correct = False
            adv_correct = False

    attack_success = clean_correct and not adv_correct

    # 返回 orig_img + final_delta（原图分辨率），供保存
    return orig_img, final_delta, answer_clean, answer_adv, attack_success


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--max_samples', type=int, default=200)
    parser.add_argument('--iters', type=int, default=50)
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--sketch_token_num', type=int, default=54)
    parser.add_argument('--dataset', type=str, default='mmvp', choices=['mmvp', 'vstar', 'mmstar'],
                        help='Dataset to attack: mmvp, vstar, or mmstar')
    args = parser.parse_args()

    global MODEL_PATH, SKETCH_TOKEN_NUM, NUM_ITERS
    if args.model_path:
        MODEL_PATH = args.model_path
    SKETCH_TOKEN_NUM = args.sketch_token_num
    NUM_ITERS = args.iters

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    device = torch.device('cuda:0')
    print(f"Using GPU: {os.environ.get('CUDA_VISIBLE_DEVICES', 'auto')}, iters: {NUM_ITERS}")
    print(f"Attacking dataset: {args.dataset.upper()}")

    print("Loading SkiLa model...")
    model, processor, sketch_processor = setup_model()
    model.eval()
    print("Model loaded")

    # 根据数据集选择summary路径和图像目录
    if args.dataset == 'vstar':
        summary_path = os.environ.get(
            "SKILA_VSTAR_SUMMARY_PATH",
            "test_results/org/summary.json",
        )
        passed_ids_key = "vstar_result"
        image_dir = VSTAR_DATA_DIR
    elif args.dataset == 'mmstar':
        summary_path = os.environ.get(
            "SKILA_MMSTAR_SUMMARY_PATH",
            "test_results/org/summary.json",
        )
        passed_ids_key = "mmstar_result"
        image_dir = MMSTAR_DATA_DIR
    else:
        summary_path = os.environ.get(
            "SKILA_MMVP_SUMMARY_PATH",
            "test_results/org/summary.json",
        )
        passed_ids_key = "mmvp_result"
        image_dir = MMVP_IMAGE_DIR

    # 如果有之前的测试结果，使用passed_ids作为测试集
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            summary = json.load(f)
        if passed_ids_key in summary and "passed_ids" in summary[passed_ids_key]:
            passed_ids = set(summary[passed_ids_key]["passed_ids"])
            print(f"Using {len(passed_ids)} passed IDs from summary.json")
        else:
            passed_ids = None
            print("No passed_ids found in summary.json")
    else:
        passed_ids = None
        print("No summary.json found, using all samples")

    # 加载数据集
    if args.dataset == 'vstar':
        dataset = load_vstar()
    elif args.dataset == 'mmstar':
        dataset = load_mmstar()
    else:
        dataset = load_mmvp()

    if passed_ids is not None:
        dataset = [item for item in dataset if int(item['id']) in passed_ids]
    print(f"Dataset: {len(dataset)} samples")

    # 设置输出目录为 adv_images/{dataset}
    OUTPUT_DIR = os.path.join(OUTPUT_DIR_BASE, args.dataset)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}")

    # 获取已处理的问题id
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

    results = []
    for i, item in enumerate(tqdm(dataset[:args.max_samples], desc="Attacking")):
        img_path = os.path.join(image_dir, item['image'])
        question = item['query']
        correct = item['label']

        # 跳过已处理的问题
        if item['id'] in processed_ids:
            print(f"[{i}] Skipping id={item['id']} (already processed)")
            continue

        try:
            orig_img, final_delta, ans_clean, ans_adv, success = attack_sample_logit_prob_skila(
                model, processor, sketch_processor, img_path, question, correct, device,
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

    result_path = os.path.join(OUTPUT_DIR, f"attack_results_{args.dataset}.json")
    with open(result_path, 'w') as f:
        json.dump({'success_rate': success_rate, 'results': results}, f, indent=2)
    print(f"Results saved to {result_path}")


if __name__ == "__main__":
    import argparse as _ap
    _pre = _ap.ArgumentParser(add_help=False)
    _pre.add_argument('--gpu', type=int, default=0)
    _args, _ = _pre.parse_known_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(_args.gpu)
    main()