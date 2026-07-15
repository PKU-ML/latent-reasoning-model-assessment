#!/usr/bin/env python3
"""
================================================================================
RoT模型 白盒对抗攻击 - GCG (Greedy Coordinate Gradient)
================================================================================

一、算法概述
-----------
本代码实现了一种针对RoT模型的GCG对抗攻击，参考CODI的attack_white.py实现。
目标是通过优化一个问题前缀，使得模型在给定问题下输出错误的答案。

二、RoT模型推理流程
------------------
RoT是一种将CoT压缩为视觉token的模型，其推理过程如下：
1. 输入：问题(question) + 可选的前缀(prefix)
2. Question encoding: 使用tokenizer将question转为input_ids，再用embedding层得到question_embeds
3. Vision token generation (thinking process): 迭代生成vision embeddings
   - 通过language_model获取hidden states
   - 通过projection_head预测下一个vision embedding
   - 将预测的vision embedding添加到序列中
4. 添加<img_end> token
5. Answer generation: 自回归生成答案文本

三、攻击目标
-----------
模型输出格式为 "The answer is: {答案}"

攻击目标：
- 在答案token位置找到top-1 token（最大概率的token，记作t1）和top-2 token（次大概率，记作t2）
- 优化目标：让 t2 的logit - t1 的logit 差值最大化
- 当这个差值足够大时，t2的概率可能超过t1，导致模型输出错误答案

四、关键实现
-----------
1. 答案token位置确定：
   - "The answer is: " 包含4个token（The, answer, is, :）加上1个空格
   - 答案token（数字）出现在第5个位置（索引5）
   - logits[5] 预测的是位置6的token，即实际答案

2. BASELINE_TOKEN_ID 和 TARGET_TOKEN_ID：
   - 在无前缀时，答案位置最大的token是baseline_token（正确答案是t1）
   - 次大的token是target_token（我们想让模型选择的t2）
   - 这两个token ID在整个攻击过程中保持固定

3. 损失函数：
   Loss = logit[target_token] - logit[baseline_token]
   - 通过反向传播计算梯度
   - 梯度从答案位置 -> answer generation -> vision generation -> projection_head -> question embeddings -> prefix tokens

4. GCG优化：
   - 对前缀的每个位置，计算梯度
   - 选取梯度最大的TOP_K个候选token
   - 随机采样1到前缀长度个位置进行替换
   - 评估候选前缀的logit差值
   - 选择logit差值最大的前缀

================================================================================
"""

import json
import os
import sys
import torch
import numpy as np
import yaml
import re
import argparse
from pathlib import Path
from typing import Tuple

import random

# 设置路径
sys.path.insert(0, str(Path(__file__).parent) + "/RoT")

from models.cot_compressor import CoTCompressor
from scripts.evaluate import load_model

# ============ 命令行参数 ============
parser = argparse.ArgumentParser(description="RoT GCG 对抗前缀攻击")
parser.add_argument("--candidate-selection", type=str, choices=["gradient", "random"], default="gradient",
                    help="候选token选择方式: gradient(梯度选择) 或 random(随机选择)")
parser.add_argument("--problem-id", type=int, default=None,
                    help="指定攻击的问题ID (默认自动分配)")
parser.add_argument("--prefix-length", type=int, default=5,
                    help="前缀长度 (默认10)")
parser.add_argument("--n-iters", type=int, default=30,
                    help="迭代次数 (默认50)")
parser.add_argument("--top-k", type=int, default=300,
                    help="每个位置考虑top-k个候选 (默认300)")
parser.add_argument("--batch-size", type=int, default=40,
                    help="候选batch大小 (默认40)")
parser.add_argument("--num-vision-tokens", type=int, default=32,
                    help="vision token数量 (默认32)")
parser.add_argument("--output-dir", type=str,
                    default=os.environ.get(
                        "ROT_OUTPUT_DIR",
                        str(Path(__file__).resolve().parent / "gcg_results" / "white"),
                    ),
                    help="结果保存目录 (默认: <RoT>/gcg_results/white；可通过 ROT_OUTPUT_DIR 覆盖)")
parser.add_argument("--seed", type=int, default=42,
                    help="随机种子 (默认42)")
parser.add_argument("--dataset", type=str, choices=["gsm8k", "MultiArith", "SVAMP"], default="gsm8k",
                    help="指定要攻击的数据集 (默认 gsm8k)")
args = parser.parse_args()

# 设置随机种子
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)

# ============ 配置 ============
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用设备: {DEVICE}")

# RoT模型配置 — 通过环境变量注入
CHECKPOINT = os.environ.get(
    "ROT_CHECKPOINT",
    str(Path(__file__).resolve().parent / "RoT" / "rot_model" / "converted"),
)
STAGE1_CHECKPOINT = os.environ.get(
    "ROT_STAGE1_CHECKPOINT",
    str(Path(__file__).resolve().parent / "RoT" / "rot_model"),
)
CONFIG = os.environ.get(
    "ROT_CONFIG",
    str(Path(__file__).resolve().parent / "RoT" / "configs" / "stage2_config_qwen3vl_2b.yaml"),
)
if not Path(CHECKPOINT).exists():
    print(f"[警告] ROT_CHECKPOINT={CHECKPOINT} 不存在，请通过环境变量指向实际目录")
if not Path(CONFIG).exists():
    print(f"[警告] ROT_CONFIG={CONFIG} 不存在，请通过环境变量指向实际 yaml")

# 推理参数
MAX_NEW_TOKENS = 256
TEMPERATURE = 0.0
NUM_VISION_TOKENS = args.num_vision_tokens
STOP_THRESHOLD = 0.02

# GCG 参数
PREFIX_LENGTH = args.prefix_length
N_ITERS = args.n_iters
TOP_K = args.top_k
BATCH_SIZE = args.batch_size

# 评估数据配置
DATA_NAME = args.dataset
DATA_DIR = os.environ.get(
    "DATA_DIR",
    str(Path(__file__).resolve().parent.parent / "data"),
)
DATA_PATH = f"{DATA_DIR}/{DATA_NAME}.json"
OUTPUT_DIR = args.output_dir
os.makedirs(OUTPUT_DIR, exist_ok=True)

if args.problem_id is not None:
    PROBLEM_ID = args.problem_id
else:
    existing_files = list(Path(OUTPUT_DIR).glob("problem_*.json"))
    if existing_files:
        max_id = max([int(f.stem.split('_')[1]) for f in existing_files])
        PROBLEM_ID = max_id + 1
    else:
        PROBLEM_ID = 0

print("=" * 60)
print("RoT GCG 寻找对抗前缀")
print(f"模式: {'梯度选择候选' if args.candidate_selection == 'gradient' else '随机选择候选'}")
print("=" * 60)


# ============ 1. 加载数据 ============
print(f"\n[1] 加载{DATA_NAME}数据...")

def load_local_dataset(data_path: str):
    """从本地 JSON 文件加载数据集"""
    questions = []
    answers = []

    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for item in data:
        questions.append(item['question'])
        answer_text = item['answer'].replace(',', '')
        try:
            ans = float(answer_text)
        except ValueError:
            ans = float("inf")
        answers.append(ans)

    return questions, answers

dataset = load_local_dataset(DATA_PATH)
questions = dataset[0]
answers = dataset[1]

question = questions[PROBLEM_ID]
answer_str = str(answers[PROBLEM_ID])
print(f"问题: {question[:100]}...")
print(f"正确答案: {answer_str}")


# ============ 2. 加载RoT模型 ============
print("\n[2] 加载RoT模型...")

with open(CONFIG, "r") as f:
    config = yaml.safe_load(f)

model = load_model(
    checkpoint_path=CHECKPOINT,
    config=config,
    model_type="v2",
    verbose=True,
    stage1_checkpoint=STAGE1_CHECKPOINT
)

model = model.to(DEVICE)
model.eval()
print(f"RoT模型加载完成")


# ============ 3. 辅助函数 ============

def extract_answer_number(sentence: str) -> float:
    """从生成的文本中提取答案数字"""
    sentence = sentence.replace(',', '')
    pred = re.findall(r'-?\d+\.?\d*', sentence)
    if not pred:
        return float('inf')
    return float(pred[-1])


def get_embedding_layer():
    """获取模型的embedding层"""
    return model.language_model.get_input_embeddings()


def make_question_embeddings(question_str: str, prefix_str: str = ""):
    """
    构建问题的embeddings
    格式: [prefix_tokens] + [question_tokens] + [img_begin_token]
    """
    full_text = prefix_str.strip() + " " + question_str.strip() if prefix_str else question_str.strip()

    # 使用chat template格式化问题
    question_message = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": full_text},
    ]
    full_text_formatted = model.tokenizer.apply_chat_template(
        question_message, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )

    # Tokenize
    encoding = model.tokenizer([full_text_formatted], return_tensors="pt", padding=True)
    input_ids = encoding["input_ids"].to(DEVICE)

    # 获取embeddings
    embed_layer = get_embedding_layer()
    question_embeds = embed_layer(input_ids)

    # 添加 img_begin token
    img_begin_ids = torch.full((1, 1), model.img_begin_token_id, dtype=torch.long, device=DEVICE)
    img_begin_embeds = embed_layer(img_begin_ids)

    return question_embeds, img_begin_embeds, input_ids


def generate_vision_embeddings_with_grad(question_embeds, img_begin_embeds, num_vision_tokens: int):
    """
    生成vision embeddings (thinking process)
    这是可求导的关键过程 - 梯度必须能够流过
    """
    embed_layer = get_embedding_layer()

    # 初始化：question + img_begin
    current_embeds = torch.cat([question_embeds, img_begin_embeds], dim=1)  # [1, q_len+1, hidden_dim]

    # 用于KV cache的DynamicCache
    from transformers import DynamicCache
    kv_cache = None

    predicted_vision_embeds_list = []

    for i in range(num_vision_tokens):
        if kv_cache is None:
            lm_outputs = model.language_model(
                inputs_embeds=current_embeds,
                output_hidden_states=True,
                use_cache=True,
            )
        else:
            cache_obj = DynamicCache()
            for layer_idx, layer in enumerate(kv_cache.layers):
                k = layer.keys
                v = layer.values
                cache_obj.update(k, v, layer_idx)

            lm_outputs = model.language_model(
                inputs_embeds=current_embeds[:, -1:, :],
                past_key_values=cache_obj,
                output_hidden_states=True,
                use_cache=True,
            )

        kv_cache = lm_outputs.past_key_values
        last_hidden = lm_outputs.hidden_states[-1][:, -1:, :]  # [1, 1, hidden_dim]

        # 通过 projection_head 预测 vision embedding
        if model.projection_head is not None:
            predicted_vision_emb = model.projection_head(last_hidden)  # [1, 1, hidden_dim]
        else:
            predicted_vision_emb = last_hidden

        predicted_vision_embeds_list.append(predicted_vision_emb)

        # 关键：不detach！让梯度可以流过
        current_embeds = torch.cat([current_embeds, predicted_vision_emb], dim=1)

        del lm_outputs, last_hidden

    # 合并所有 vision embeddings
    predicted_vision_embeds = torch.cat(predicted_vision_embeds_list, dim=1)  # [1, num_tokens, hidden_dim]

    return predicted_vision_embeds, kv_cache


def run_forward_for_logits(question_embeds, img_begin_embeds, vision_embeds, kv_cache, max_tokens: int = 30):
    """
    运行前向传播，获取生成过程中的logits
    用于计算答案token位置的损失
    """
    embed_layer = get_embedding_layer()
    eos_id = model.tokenizer.eos_token_id

    # 构建完整输入: question + img_begin + vision + img_end
    img_end_ids = torch.full((1, 1), model.img_end_token_id, dtype=torch.long, device=DEVICE)
    img_end_embeds = embed_layer(img_end_ids)

    full_embeds = torch.cat([
        question_embeds,
        img_begin_embeds,
        vision_embeds,
        img_end_embeds
    ], dim=1)

    # 自回归生成并记录logits
    all_logits = []
    current_seq = full_embeds

    for step in range(max_tokens):
        lm_out = model.language_model(inputs_embeds=current_seq, output_hidden_states=True)
        last_hidden = lm_out.last_hidden_state[:, -1:, :]
        logits = model.llm_lm_head(last_hidden).squeeze(1)
        all_logits.append(logits)

        # 贪婪解码
        next_token_id = logits.argmax(dim=-1).item()

        if next_token_id == eos_id:
            break

        next_emb = embed_layer(torch.tensor([next_token_id], dtype=torch.long, device=DEVICE)).unsqueeze(0)
        current_seq = torch.cat([current_seq, next_emb], dim=1)

    return all_logits


# ============ 4. 获取baseline信息 ============
print("\n[3] 获取baseline信息...")

embed_layer = get_embedding_layer()

# 确定 "The answer is:" 前缀的token数量
prefix_text = "The answer is:"
prefix_ids = model.tokenizer.encode(prefix_text, add_special_tokens=False)
NUM_PREFIX_TOKENS = len(prefix_ids)
# "The answer is: " 包含4个token+1个空格，答案token位置在第5个
ANSWER_TOKEN_POS = NUM_PREFIX_TOKENS + 1
print(f"前缀 '{prefix_text}' 的token数量: {NUM_PREFIX_TOKENS}")
print(f"答案token位置: {ANSWER_TOKEN_POS}")

# 测试无前缀时的输出
with torch.no_grad():
    q_embeds, img_begin_embeds, _ = make_question_embeddings(question, "")
    vision_embeds, kv_cache = generate_vision_embeddings_with_grad(q_embeds, img_begin_embeds, NUM_VISION_TOKENS)

    img_end_ids = torch.full((1, 1), model.img_end_token_id, dtype=torch.long, device=DEVICE)
    img_end_embeds = embed_layer(img_end_ids)

    full_embeds = torch.cat([q_embeds, img_begin_embeds, vision_embeds, img_end_embeds], dim=1)

    # 生成答案
    answer_ids = []
    current_seq = full_embeds
    eos_id = model.tokenizer.eos_token_id

    for step in range(MAX_NEW_TOKENS):
        lm_out = model.language_model(inputs_embeds=current_seq, output_hidden_states=True)
        last_hidden = lm_out.last_hidden_state[:, -1:, :]
        logits = model.llm_lm_head(last_hidden).squeeze(1)

        next_token_id = logits.argmax(dim=-1).item()
        if next_token_id == eos_id:
            break
        answer_ids.append(next_token_id)
        next_emb = embed_layer(torch.tensor([next_token_id], dtype=torch.long, device=DEVICE)).unsqueeze(0)
        current_seq = torch.cat([current_seq, next_emb], dim=1)

    baseline_text = model.tokenizer.decode(answer_ids, skip_special_tokens=True)
    baseline_answer = extract_answer_number(baseline_text)

print(f"Baseline生成文本: '{baseline_text[:100]}...'")
print(f"Baseline提取答案: {baseline_answer}")
print(f"正确答案: {answer_str}")

baseline_correct = (abs(baseline_answer - float(answer_str)) < 1e-9 if baseline_answer != float('inf')
                   else baseline_answer == float(answer_str))
print(f"Baseline是否正确: {baseline_correct}")

# 获取baseline答案token的logits信息
print("\n获取baseline logits信息...")

with torch.no_grad():
    # 生成足够多的token来覆盖前缀和答案token
    all_logits = []
    current_seq = full_embeds

    for step in range(NUM_PREFIX_TOKENS + 10):
        lm_out = model.language_model(inputs_embeds=current_seq, output_hidden_states=True)
        last_hidden = lm_out.last_hidden_state[:, -1:, :]
        logits = model.llm_lm_head(last_hidden).squeeze(1)
        all_logits.append(logits.clone())

        next_token_id = logits.argmax(dim=-1).item()
        if next_token_id == eos_id:
            break

        next_emb = embed_layer(torch.tensor([next_token_id], dtype=torch.long, device=DEVICE)).unsqueeze(0)
        current_seq = torch.cat([current_seq, next_emb], dim=1)

    # 打印前几个生成的token帮助调试
    print(f"生成的前{len(all_logits)}个token:")
    for idx in range(min(8, len(all_logits))):
        tok_id = all_logits[idx].argmax().item()
        tok_str = model.tokenizer.decode([tok_id])
        print(f"  位置{idx}: '{tok_str}' (id={tok_id})")

    # 获取答案token位置的logits
    if len(all_logits) > ANSWER_TOKEN_POS:
        answer_token_logits = all_logits[ANSWER_TOKEN_POS]
    else:
        answer_token_logits = all_logits[-1]
        print(f"警告: 生成的token数量不足，使用最后一个位置的logits")

    sorted_logits, sorted_indices = torch.sort(answer_token_logits, descending=True)
    # sorted_indices shape: [1, vocab_size], 取第一个需要 [0, 0] 和 [0, 1]
    BASELINE_TOKEN_ID = int(sorted_indices[0, 0].item())  # 最大概率的token
    TARGET_TOKEN_ID = int(sorted_indices[0, 1].item())   # 第二大概率的token

    baseline_token_str = model.tokenizer.decode([BASELINE_TOKEN_ID])
    target_token_str = model.tokenizer.decode([TARGET_TOKEN_ID])

    baseline_logit = answer_token_logits[0, BASELINE_TOKEN_ID].item()
    target_logit = answer_token_logits[0, TARGET_TOKEN_ID].item()
    baseline_prob = torch.softmax(answer_token_logits, dim=-1)[0, BASELINE_TOKEN_ID].item()
    target_prob = torch.softmax(answer_token_logits, dim=-1)[0, TARGET_TOKEN_ID].item()
    logit_diff = target_logit - baseline_logit

    top5_tokens = [model.tokenizer.decode([int(sorted_indices[0, i].item())]) for i in range(min(5, sorted_indices.shape[1]))]
    print(f"\nBaseline答案token位置:")
    print(f"  最大token: '{baseline_token_str}' (id={BASELINE_TOKEN_ID}), logit={baseline_logit:.4f}, prob={baseline_prob:.4f}")
    print(f"  第二大token: '{target_token_str}' (id={TARGET_TOKEN_ID}), logit={target_logit:.4f}, prob={target_prob:.4f}")
    print(f"  Logit差值(次大-最大): {logit_diff:.4f}")
    print(f"  Top 5 tokens: {top5_tokens}")


# ============ 5. 初始化前缀 ============
print("\n[4] 初始化前缀...")

char_tokens = []
for i in range(15, 25):
    try:
        decoded = model.tokenizer.decode([i])
        if decoded.strip() and len(decoded.strip()) > 0:
            char_tokens.append(i)
    except:
        pass

print(f"有效token数量: {len(char_tokens)}")

# 随机初始化前缀
prefix_ids = np.random.choice(char_tokens, size=PREFIX_LENGTH, replace=True)
prefix_str = model.tokenizer.decode(prefix_ids)
print(f"初始前缀: '{prefix_str}'")


# ============ 6. 梯度计算函数 ============
print("\n[5] 定义梯度计算函数...")


def compute_gradients(question_str: str, prefix_ids_arr: np.ndarray):
    """
    计算损失相对于前缀的梯度
    损失函数: loss = logit[target] - logit[baseline]
    """
    prefix_len = len(prefix_ids_arr)

    # 构建输入
    prefix_str = model.tokenizer.decode(prefix_ids_arr)
    q_embeds, img_begin_embeds, input_ids = make_question_embeddings(question_str, prefix_str)

    # 让question_embeds需要梯度
    q_embeds = q_embeds.requires_grad_(True)

    # 生成 vision embeddings (关键：保留梯度)
    vision_embeds, kv_cache = generate_vision_embeddings_with_grad(q_embeds, img_begin_embeds, NUM_VISION_TOKENS)

    # 构建完整输入
    img_end_ids = torch.full((1, 1), model.img_end_token_id, dtype=torch.long, device=DEVICE)
    img_end_embeds = get_embedding_layer()(img_end_ids)

    full_embeds = torch.cat([q_embeds, img_begin_embeds, vision_embeds, img_end_embeds], dim=1)

    # 生成到答案token位置，获取该位置的logits
    all_logits = []
    current_seq = full_embeds

    for step in range(ANSWER_TOKEN_POS + 1):
        lm_out = model.language_model(inputs_embeds=current_seq, output_hidden_states=True)
        last_hidden = lm_out.last_hidden_state[:, -1:, :]
        logits = model.llm_lm_head(last_hidden).squeeze(1)
        all_logits.append(logits)

        # 贪婪解码
        next_token_id = logits.argmax(dim=-1).item()
        if next_token_id == model.tokenizer.eos_token_id:
            break

        next_emb = get_embedding_layer()(torch.tensor([next_token_id], dtype=torch.long, device=DEVICE)).unsqueeze(0)
        current_seq = torch.cat([current_seq, next_emb], dim=1)

    # 在答案token位置获取logits
    if len(all_logits) > ANSWER_TOKEN_POS:
        answer_token_logits = all_logits[ANSWER_TOKEN_POS]
    else:
        answer_token_logits = all_logits[-1]

    # 计算损失
    loss = answer_token_logits[0, TARGET_TOKEN_ID] - answer_token_logits[0, BASELINE_TOKEN_ID]
    print(f"[DEBUG] logit_loss: {loss.item():.4f}")

    # 反向传播
    loss.backward()

    # 获取梯度
    grad = q_embeds.grad  # [1, seq_len, hidden_dim]

    if grad is None:
        return np.zeros((prefix_len, get_embedding_layer().weight.shape[0]), dtype=np.float32), prefix_len

    # 前缀位置的梯度
    prefix_grad_emb = grad[0, :prefix_len]  # [prefix_len, hidden_dim]

    # 将embedding梯度转换为token logit梯度
    embed_weight = get_embedding_layer().weight.T  # [vocab_size, hidden_dim]
    token_grads = torch.matmul(prefix_grad_emb, embed_weight)  # [prefix_len, vocab_size]

    # 计算梯度的平均值（用于监控）
    grad_mean = token_grads.abs().mean().item()

    return token_grads.cpu().detach().to(torch.float32).numpy(), prefix_len, grad_mean


def compute_logits_diff(question_str: str, prefix_ids_arr: np.ndarray):
    """
    计算给定前缀的logits差值
    返回: (logit差值, baseline概率, target概率, baseline字符串, target字符串)
    """
    prefix_str = model.tokenizer.decode(prefix_ids_arr)

    with torch.no_grad():
        q_embeds, img_begin_embeds, _ = make_question_embeddings(question_str, prefix_str)
        vision_embeds, kv_cache = generate_vision_embeddings_with_grad(q_embeds, img_begin_embeds, NUM_VISION_TOKENS)

        img_end_ids = torch.full((1, 1), model.img_end_token_id, dtype=torch.long, device=DEVICE)
        img_end_embeds = get_embedding_layer()(img_end_ids)

        full_embeds = torch.cat([q_embeds, img_begin_embeds, vision_embeds, img_end_embeds], dim=1)

        # 生成到答案token位置
        all_logits = []
        current_seq = full_embeds

        for step in range(ANSWER_TOKEN_POS + 5):
            lm_out = model.language_model(inputs_embeds=current_seq, output_hidden_states=True)
            last_hidden = lm_out.last_hidden_state[:, -1:, :]
            logits = model.llm_lm_head(last_hidden).squeeze(1)
            all_logits.append(logits)

            next_token_id = logits.argmax(dim=-1).item()
            if next_token_id == model.tokenizer.eos_token_id:
                break

            next_emb = get_embedding_layer()(torch.tensor([next_token_id], dtype=torch.long, device=DEVICE)).unsqueeze(0)
            current_seq = torch.cat([current_seq, next_emb], dim=1)

        if len(all_logits) > ANSWER_TOKEN_POS:
            answer_token_logits = all_logits[ANSWER_TOKEN_POS]
        else:
            answer_token_logits = all_logits[-1]

        baseline_logit = answer_token_logits[0, BASELINE_TOKEN_ID].item()
        target_logit = answer_token_logits[0, TARGET_TOKEN_ID].item()
        baseline_prob = torch.softmax(answer_token_logits, dim=-1)[0, BASELINE_TOKEN_ID].item()
        target_prob = torch.softmax(answer_token_logits, dim=-1)[0, TARGET_TOKEN_ID].item()
        logit_diff = target_logit - baseline_logit

        baseline_str = model.tokenizer.decode([BASELINE_TOKEN_ID])
        target_str = model.tokenizer.decode([TARGET_TOKEN_ID])

    return logit_diff, baseline_prob, target_prob, baseline_str, target_str


def evaluate(question_str: str, prefix_ids_arr: np.ndarray):
    """评估当前前缀"""
    prefix_str = model.tokenizer.decode(prefix_ids_arr)

    with torch.no_grad():
        q_embeds, img_begin_embeds, _ = make_question_embeddings(question_str, prefix_str)
        vision_embeds, kv_cache = generate_vision_embeddings_with_grad(q_embeds, img_begin_embeds, NUM_VISION_TOKENS)

        embed_layer = get_embedding_layer()
        img_end_ids = torch.full((1, 1), model.img_end_token_id, dtype=torch.long, device=DEVICE)
        img_end_embeds = embed_layer(img_end_ids)

        full_embeds = torch.cat([q_embeds, img_begin_embeds, vision_embeds, img_end_embeds], dim=1)

        answer_ids = []
        current_seq = full_embeds
        eos_id = model.tokenizer.eos_token_id

        for step in range(MAX_NEW_TOKENS):
            lm_out = model.language_model(inputs_embeds=current_seq, output_hidden_states=True)
            last_hidden = lm_out.last_hidden_state[:, -1:, :]
            logits = model.llm_lm_head(last_hidden).squeeze(1)

            next_token_id = logits.argmax(dim=-1).item()
            if next_token_id == eos_id:
                break

            answer_ids.append(next_token_id)
            next_emb = embed_layer(torch.tensor([next_token_id], dtype=torch.long, device=DEVICE)).unsqueeze(0)
            current_seq = torch.cat([current_seq, next_emb], dim=1)

        text = model.tokenizer.decode(answer_ids, skip_special_tokens=True)

    pred_answer = extract_answer_number(text)
    return text, pred_answer


# ============ 7. GCG 迭代 ============
print("\n[6] 开始GCG迭代...")

current_text, current_answer = evaluate(question, prefix_ids)
current_correct = (abs(current_answer - float(answer_str)) < 1e-9 if current_answer != float('inf')
                   else current_answer == float(answer_str))

# 计算初始前缀的logits差值
initial_logit_diff, init_baseline_prob, init_target_prob, _, _ = compute_logits_diff(question, prefix_ids)
print(f"初始前缀 logits差值: {initial_logit_diff:.4f} (top1='{baseline_token_str}' prob={init_baseline_prob:.4f}, top2='{target_token_str}' prob={init_target_prob:.4f})")
print(f"固定token - BASELINE_TOKEN_ID={BASELINE_TOKEN_ID}, TARGET_TOKEN_ID={TARGET_TOKEN_ID}")

print(f"初始生成: '{current_text[:80]}...'")
print(f"初始答案: {current_answer}")
print(f"是否正确: {current_correct}")

best_prefix = prefix_ids.copy()
best_correct = current_correct
best_logit_diff = initial_logit_diff

scores_list = []
for iteration in range(N_ITERS):
    # 计算梯度
    gradients, actual_len, grad_mean = compute_gradients(question, prefix_ids)
    curr_len = len(prefix_ids)

    # 生成候选组合
    sampled_combinations = [{}]
    for _ in range(BATCH_SIZE):
        n_changes = np.random.randint(1, curr_len + 1)
        positions = np.random.choice(curr_len, size=n_changes, replace=False)

        combo = {}
        for pos in positions:
            if args.candidate_selection == "gradient":
                pos_grads = gradients[pos]
                top_k_indices = np.argsort(pos_grads)[-TOP_K:]
                combo[pos] = np.random.choice(top_k_indices)
            else:
                combo[pos] = np.random.choice(char_tokens)

        sampled_combinations.append(combo)

    # 评估候选
    scores = []
    for combo in sampled_combinations:
        new_prefix = prefix_ids.copy()
        for pos, tok_id in combo.items():
            new_prefix[pos] = tok_id

        logit_diff, _, _, _, _ = compute_logits_diff(question, new_prefix)
        scores.append((logit_diff, combo))

    # 选择最佳候选
    scores = sorted(scores, key=lambda x: x[0])
    best_combo = scores[-1][1]
    best_score = scores[-1][0]

    # 更新前缀
    for pos, tok_id in best_combo.items():
        prefix_ids[pos] = tok_id

    scores_list.append(best_score)
    if best_score > best_logit_diff:
        best_logit_diff = best_score
        best_prefix = prefix_ids.copy()

    # 获取当前状态
    current_text, current_answer = evaluate(question, prefix_ids)
    current_correct = (abs(current_answer - float(answer_str)) < 1e-9 if current_answer != float('inf')
                       else current_answer == float(answer_str))

    curr_logit_diff, curr_baseline_prob, curr_target_prob, _, _ = compute_logits_diff(question, prefix_ids)

    print(f"Iter {iteration+1}: logit差值={curr_logit_diff:.4f}, 梯度均值={grad_mean:.6f}, (top1='{baseline_token_str}' prob={curr_baseline_prob:.4f}, top2='{target_token_str}' prob={curr_target_prob:.4f}), 最佳差值={best_logit_diff:.4f}, 答案={current_answer}, 正确={current_correct}")
    #if best_logit_diff > 0.49 and current_correct == False:
    if current_correct == False:
        best_logit_diff = best_score
        best_prefix = prefix_ids.copy()
        break

# ============ 8. 结果 ============
print("\n" + "=" * 60)
print("最终结果")
print("=" * 60)

final_prefix_str = model.tokenizer.decode(best_prefix)
print(f"对抗前缀: '{final_prefix_str}'")

# 评估无前缀
clean_text, clean_answer = evaluate(question, np.array([]))
clean_correct = (abs(clean_answer - float(answer_str)) < 1e-9 if clean_answer != float('inf')
                 else clean_answer == float(answer_str))
clean_logit_diff, _, _, _, _ = compute_logits_diff(question, np.array([]))

# 评估有前缀
adv_text, adv_answer = evaluate(question, best_prefix)
adv_correct = (abs(adv_answer - float(answer_str)) < 1e-9 if adv_answer != float('inf')
               else adv_answer == float(answer_str))
adv_logit_diff, _, _, _, _ = compute_logits_diff(question, best_prefix)

print(f"\n无前缀:")
print(f"  生成文本: '{clean_text[:100]}...'")
print(f"  提取答案: {clean_answer}")
print(f"  是否正确: {clean_correct}")

print(f"\n有前缀 ('{final_prefix_str}'):")
print(f"  生成文本: '{adv_text[:100]}...'")
print(f"  提取答案: {adv_answer}")
print(f"  是否正确: {adv_correct}")

print(f"\nLogit差值信息:")
print(f"  Baseline差值: {logit_diff:.4f}")
print(f"  无前缀差值: {clean_logit_diff:.4f}")
print(f"  有前缀差值: {adv_logit_diff:.4f}")
print(f"  最佳差值: {best_logit_diff:.4f}")

print(f"\n攻击效果: {'成功' if not adv_correct and clean_correct else '失败'}")


# 保存结果
result = {
    "question": question,
    "ground_truth": answer_str,
    "prefix": final_prefix_str,
    "prefix_token_ids": [int(x) for x in best_prefix],
    "baseline_token_id": BASELINE_TOKEN_ID,
    "target_token_id": TARGET_TOKEN_ID,
    "baseline_logit_diff": logit_diff,
    "clean_logit_diff": clean_logit_diff,
    "adv_logit_diff": adv_logit_diff,
    "best_logit_diff": best_logit_diff,
    "clean_text": clean_text,
    "clean_answer": clean_answer,
    "clean_correct": clean_correct,
    "adv_text": adv_text,
    "adv_answer": adv_answer,
    "adv_correct": adv_correct,
    "attack_success": not adv_correct and clean_correct,
    "scores": scores_list
}

with open(f"{OUTPUT_DIR}/problem_{PROBLEM_ID}.json", "w") as f:
    json.dump(result, f, indent=2, ensure_ascii=False)

print(f"\n结果已保存到 {OUTPUT_DIR}/problem_{PROBLEM_ID}.json")