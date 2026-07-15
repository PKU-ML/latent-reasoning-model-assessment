"""
================================================================================
CODI模型 GCG (Greedy Coordinate Gradient) 对抗前缀攻击
================================================================================

一、算法概述
------------
本代码实现了一种针对CODI模型的GCG对抗攻击，目标是通过优化一个前缀字符串，
使得模型在给定问题下输出错误的答案。

二、CODI模型推理流程
--------------------
CODI是一种具有latent推理能力的模型，其推理过程如下：
1. 输入：问题(question) + 对抗前缀(prefix) + bot_token
2. Encoder编码：问题+前缀 -> 获取hidden states
3. Latent推理：迭代执行inf_latent_iterations次
   - 使用hidden states的最后一个token作为latent embedding
   - 通过投影层(PRJ)处理
   - 再次输入模型进行推理
4. 添加EOT token：表示 latent 推理完成
5. 自回归生成：模型开始生成答案文本

三、攻击目标
------------
模型输出格式为 "The answer is: {答案}"

例如问题 "What is 5+13?" -> 模型输出 "The answer is: 18"

攻击目标：
- 在答案token位置（"The answer is: "之后的第一个实际答案token）
- 找到top-1 token（最大概率的token，记作t1）和top-2 token（次大概率，记作t2）
- 优化目标：让 t2 的logit - t1 的logit 差值最大化
- 当这个差值足够大时，t2的概率可能超过t1，导致模型输出错误答案

四、关键实现
------------
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
   - 梯度从答案位置 -> latent推理 -> encoder -> 前缀token

4. GCG优化：
   - 对前缀的每个位置，计算梯度
   - 选取梯度最大的TOP_K个候选token
   - 随机采样1到前缀长度个位置进行替换
   - 评估候选前缀的logit差值
   - 选择logit差值最大的前缀

五、输出信息
------------
- Baseline: 无前缀时的模型输出和logits信息
- GCG迭代: 每次迭代的logit差值、概率、最佳差值
- 最终结果: 对抗前缀、攻击效果、logit差值变化

六、梯度流动
------------
答案token的梯度反向传播路径：
logits[答案位置] -> 输出层 -> 最后一层Transformer ->
... (多层) ... -> Latent推理层 -> Encoder ->
前缀token的embedding -> 前缀token

================================================================================
"""

import json
import torch
import torch.nn.functional as F
import numpy as np
import os
import math
import logging
import re
import argparse

from transformers import AutoTokenizer
from peft import LoraConfig, TaskType
from datasets import load_dataset

# 导入CODI模型
import sys
from pathlib import Path
_CODI_PROJECT_ROOT = Path(os.environ.get("CODI_PROJECT_ROOT", Path(__file__).resolve().parent / "codi"))
sys.path.insert(0, str(_CODI_PROJECT_ROOT))
from src.model import CODI, ModelArguments, DataArguments, TrainingArguments

# ============ 命令行参数 ============
parser = argparse.ArgumentParser(description="CODI GCG 对抗前缀攻击")
parser.add_argument("--candidate-selection", type=str, choices=["gradient", "random"], default="gradient",
                    help="候选token选择方式: gradient(梯度选择) 或 random(随机选择，仅根据logit分数爬山)")
parser.add_argument("--problem-id", type=int, default=None,
                    help="指定攻击的问题ID (默认自动分配)")
parser.add_argument("--prefix-length", type=int, default=3,
                    help="前缀长度 (默认3)")
parser.add_argument("--n-iters", type=int, default=50,
                    help="迭代次数 (默认50)")
parser.add_argument("--top-k", type=int, default=300,
                    help="每个位置考虑top-k个候选 (默认300)")
parser.add_argument("--batch-size", type=int, default=40,
                    help="候选batch大小 (默认40)")
parser.add_argument("--latent-gradient", type=str, choices=['true', 'false'], default='true',
                    help="梯度是否流经latent向量: true 或 false (默认true)")
parser.add_argument("--filter-digits", type=str, choices=['true', 'false'], default='true',
                    help="前缀候选是否过滤含数字的token: true 或 false (默认true，启用时候选数仍保持top-k)")
parser.add_argument("--output-dir", type=str,
                    default=os.environ.get(
                        "CODI_OUTPUT_DIR",
                        str(Path(__file__).resolve().parent / "gcg_results"),
                    ),
                    help="结果保存目录 (默认: <codi>/gcg_results；可通过 CODI_OUTPUT_DIR 覆盖)")
parser.add_argument("--seed", type=int, default=20,
                    help="随机种子 (默认42)")
parser.add_argument("--dataset", type=str, choices=["gsm8k", "MultiArith", "SVAMP"], default="gsm8k",
                    help="指定要攻击的数据集 (默认 gsm8k)")
args = parser.parse_args()
args.pass_gradient_through_latent = args.latent_gradient == 'true'
args.filter_digits = args.filter_digits == 'true'

# 设置随机种子
import random
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)
# python attack_prefix.py --candidate-selection gradient --latent-gradient false

# ============ 配置 ============
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用设备: {DEVICE}")

# 模型配置 (参考 test_llama1b.sh) — 通过环境变量注入；找不到时打印警告
MODEL_NAME = os.environ.get(
    "CODI_MODEL_NAME",
    "meta-llama/Llama-3.2-1B-Instruct",
)
CKPT_DIR = os.environ.get(
    "CODI_CKPT_DIR",
    str(Path(__file__).resolve().parent / "codi" / "codi_llama1b"),
)  # LoRA权重目录
if not Path(MODEL_NAME).exists() and not MODEL_NAME.startswith(("meta-llama/", "huggingface.co/", "http")):
    print(f"[警告] CODI_MODEL_NAME={MODEL_NAME} 在本地不存在，请检查环境变量或模型缓存路径")
if not Path(CKPT_DIR).exists():
    print(f"[警告] CODI_CKPT_DIR={CKPT_DIR} 不存在，请设置环境变量指向实际的 CODI checkpoint 目录")

# 推理参数
NUM_LATENT = 6  # latent推理步数
INF_LATENT_ITERATIONS = 6  # 推理时的latent步数
USE_PRJ = True  # 使用投影层
REMOVE_EOS = True

# GCG 参数
PREFIX_LENGTH = args.prefix_length  # 前缀长度
N_ITERS = args.n_iters  # 迭代次数
TOP_K = args.top_k  # 每个位置考虑top-k个候选
BATCH_SIZE = args.batch_size  # 候选batch大小
MAX_NEW_TOKENS = 30  # 最大生成token数

# 评估数据配置 — 数据目录通过 DATA_DIR 环境变量覆盖
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
    PROBLEM_ID = len(os.listdir(OUTPUT_DIR))  # 攻击第?个问题
#PROBLEM_ID = json.load(open("pass_ids.json"))[PROBLEM_ID]

print("=" * 60)
print("CODI GCG 寻找对抗前缀")
print(f"模式: {'梯度选择候选' if args.candidate_selection == 'gradient' else '随机选择候选 (仅根据logit分数爬山)'}")
print("=" * 60)


# ============ 1. 加载数据 ============
print(f"\n[1] 加载{DATA_NAME}数据...")

# 不同数据集的答案提取方式
ANSWER_PATTERNS = {
    "gsm8k": r'The answer is (\d+)',      # "The answer is 18"
    "MultiArith": r'(\d+)',                # 直接提取数字
    "SVAMP": r'The answer is (\d+)',      # "The answer is 18"
}

def load_local_dataset(data_path, dataset_name):
    questions = []
    answers = []
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    for item in data:
        questions.append(item['question'])
        answer_text = item['answer'].replace(',', '')
        # 根据数据集使用不同的答案提取方式
        pattern = ANSWER_PATTERNS.get(dataset_name, r'-?\d+\.?\d*')
        match = re.search(pattern, answer_text)
        if match:
            answer_text = match.group(1) if match.groups() else match.group(0)
        try:
            ans = float(answer_text)
        except ValueError:
            ans = float("inf")
        answers.append(ans)
    return questions, answers

dataset = load_local_dataset(DATA_PATH, DATA_NAME)
questions = dataset[0]
answers = dataset[1]

question = questions[PROBLEM_ID]
answer_str = str(answers[PROBLEM_ID])
print(f"问题: {question[:100]}...")
print(f"正确答案: {answer_str}")

# ============ 2. 配置模型参数 ============
print("\n[2] 配置模型...")

# ModelArguments
model_args = ModelArguments(
    model_name_or_path=MODEL_NAME,
    lora_init=True,
    lora_r=128,
    lora_alpha=32,
    full_precision=True,
    train=False,
)

# TrainingArguments
training_args = TrainingArguments(
    model_max_length=512,
    num_latent=NUM_LATENT,
    use_lora=True,
    use_prj=USE_PRJ,
    prj_dim=2048,
    prj_no_ln=False,
    prj_dropout=0.0,
    inf_latent_iterations=INF_LATENT_ITERATIONS,
    remove_eos=REMOVE_EOS,
    greedy=True,  # 使用贪婪解码便于梯度计算
    print_loss=False,
)

# LoRA配置
task_type = TaskType.CAUSAL_LM
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"]
lora_config = LoraConfig(
    task_type=task_type,
    inference_mode=False,
    r=model_args.lora_r,
    lora_alpha=model_args.lora_alpha,
    lora_dropout=0.1,
    target_modules=target_modules,
    init_lora_weights=True,
)


# ============ 3. 初始化CODI模型 ============
print("\n[3] 初始化CODI模型...")

model = CODI(model_args, training_args, lora_config)

# 加载LoRA权重
try:
    state_dict = torch.load(os.path.join(CKPT_DIR, "pytorch_model.bin"), map_location="cpu")
except:
    from safetensors.torch import load_file
    state_dict = load_file(os.path.join(CKPT_DIR, "model.safetensors"))

model.load_state_dict(state_dict, strict=False)
model.codi.tie_weights()

# 加载tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
if tokenizer.pad_token_id is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    tokenizer.pad_token_id = model.pad_token_id

# 移到GPU并转换为bfloat16
model = model.to(DEVICE)
model = model.to(torch.bfloat16)
model.eval()

print(f"模型加载完成")
print(f"num_latent: {model.num_latent}")
print(f"inf_latent_iterations: {training_args.inf_latent_iterations}")


# ============ 4. 辅助函数 ============

def make_prompt(question_str, prefix_str=""):
    """
    构建prompt
    格式: 问题 + 前缀 + bot_id
    """
    full_text = prefix_str.strip() + question_str.strip()

    # Tokenize
    inputs = tokenizer(full_text, return_tensors="pt", padding=True)

    # 添加bot token
    if REMOVE_EOS:
        bot_tensor = torch.tensor([model.bot_id], dtype=torch.long).expand(inputs["input_ids"].size(0), 1)
    else:
        bot_tensor = torch.tensor([tokenizer.eos_token_id, model.bot_id], dtype=torch.long).expand(inputs["input_ids"].size(0), 2)

    input_ids = torch.cat((inputs["input_ids"], bot_tensor), dim=1)
    attention_mask = torch.cat((inputs["attention_mask"], torch.ones_like(bot_tensor)), dim=1)

    return input_ids, attention_mask


def get_embedding_layer(model):
    """获取模型的embedding层"""
    return model.get_embd(model.codi, model.model_name)


def generate_answer(input_ids, attention_mask, max_tokens=MAX_NEW_TOKENS):
    """
    完整生成答案
    """
    input_ids = input_ids.to(DEVICE)
    attention_mask = attention_mask.to(DEVICE)

    # 1. 编码问题，获取初始latent embedding
    past_key_values = None
    outputs = model.codi(
        input_ids=input_ids,
        use_cache=True,
        output_hidden_states=True,
        past_key_values=past_key_values,
        attention_mask=attention_mask
    )
    past_key_values = outputs.past_key_values
    latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)

    if USE_PRJ:
        latent_embd = model.prj(latent_embd)

    # 2. 迭代latent推理
    for i in range(INF_LATENT_ITERATIONS):
        outputs = model.codi(
            inputs_embeds=latent_embd,
            use_cache=True,
            output_hidden_states=True,
            past_key_values=past_key_values
        )
        past_key_values = outputs.past_key_values
        latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)

        if USE_PRJ:
            latent_embd = model.prj(latent_embd)

    # 3. 添加EOT token
    eot_tensor = torch.tensor([model.eot_id], dtype=torch.long, device=DEVICE)
    eot_emb = get_embedding_layer(model)(eot_tensor).unsqueeze(0)
    eot_emb = eot_emb.expand(input_ids.size(0), -1, -1)

    # 4. 自回归生成
    output = eot_emb
    generated_ids = []

    for _ in range(max_tokens):
        out = model.codi(
            inputs_embeds=output,
            output_hidden_states=False,
            attention_mask=None,
            use_cache=True,
            past_key_values=past_key_values
        )
        past_key_values = out.past_key_values
        logits = out.logits[:, -1, :]

        # 贪婪解码
        next_token_id = logits.argmax(dim=-1).item()
        generated_ids.append(next_token_id)

        if next_token_id == tokenizer.eos_token_id:
            break

        # 转换为embedding作为下一步输入
        output = get_embedding_layer(model)(
            torch.tensor([next_token_id], dtype=torch.long, device=DEVICE)
        ).unsqueeze(0)

    # 解码
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return generated_text, generated_ids


def extract_answer_number(sentence):
    """从生成的文本中提取答案数字"""
    sentence = sentence.replace(',', '')
    pred = [s for s in re.findall(r'-?\d+\.?\d*', sentence)]
    if not pred:
        return float('inf')
    return float(pred[-1])


def contains_digit(token_id, tokenizer):
    """检查token的解码结果是否包含数字字符（0-9）"""
    try:
        decoded = tokenizer.decode([int(token_id)]).strip()
        return bool(re.search(r'\d', decoded))
    except Exception:
        return False


def select_topk_candidates(pos_grads, top_k, tokenizer, filter_digits, fallback_pool):
    """
    按梯度降序选择 top_k 个 token 作为候选。
    当 filter_digits=True 时，跳过含数字的 token，并用后续较低梯度的不含数字 token 补齐，
    保证最终候选数量仍为 top_k。

    参数:
        pos_grads: 该位置所有 token 的梯度（一维 numpy 数组，长度 = vocab_size）
        top_k: 目标候选数量
        tokenizer: 用于解码判断是否含数字
        filter_digits: 是否过滤含数字的 token
        fallback_pool: 若 vocab 中不含数字的 token 不足 top_k 时的备用候选池

    返回:
        numpy 数组，包含 top_k 个 token id
    """
    # 按梯度降序排列所有 token id
    sorted_indices = np.argsort(pos_grads)[::-1]

    if not filter_digits:
        return sorted_indices[:top_k]

    selected = []
    for idx in sorted_indices:
        if not contains_digit(int(idx), tokenizer):
            selected.append(int(idx))
            if len(selected) >= top_k:
                break

    if len(selected) < top_k:
        # 不含数字的 token 不足（极端情况），从备用池补齐
        remaining = top_k - len(selected)
        fallback_indices = np.random.choice(fallback_pool, size=remaining, replace=True)
        selected.extend(int(x) for x in fallback_indices)

    return np.array(selected)


# ============ 5. 初始化前缀 ============
print("\n[5] 初始化前缀...")


# ============ 5. 测试baseline ============
print("\n[4.6] 测试baseline...")

baseline_input_ids, baseline_attn = make_prompt(question, "")
with torch.no_grad():
    baseline_text, baseline_ids = generate_answer(baseline_input_ids, baseline_attn)

baseline_answer = extract_answer_number(baseline_text)

print(f"Baseline生成文本: '{baseline_text}'")
print(f"Baseline提取答案: {baseline_answer}")
print(f"正确答案: {answer_str}")

# 计算baseline是否正确
baseline_correct = (str(baseline_answer) == answer_str or
                   abs(baseline_answer - float(answer_str)) < 0.01 if baseline_answer != float('inf') and answer_str.lstrip('-').replace('.','',1).isdigit() else False)
print(f"Baseline是否正确: {baseline_correct}")

# 获取baseline答案token的logits信息
# 用于GCG攻击：在答案token位置，最大token是正确答案，目标是让第二大token概率超过它
print("\n获取baseline logits信息...")

# 先确定 "The answer is:" 这个前缀有多少个token
prefix_text = "The answer is:"
prefix_ids = tokenizer.encode(prefix_text, add_special_tokens=False)
NUM_PREFIX_TOKENS = len(prefix_ids)
# "The answer is:" 后面还有一个空格，所以答案token位置是 NUM_PREFIX_TOKENS + 1
# "The answer is: " 包含4个token+1个空格，答案"18"在第5个位置（索引5）
# logits[5] 预测的是位置6的token，也就是"18"
ANSWER_TOKEN_POS = NUM_PREFIX_TOKENS + 1
print(f"前缀 '{prefix_text}' 的token数量: {NUM_PREFIX_TOKENS}")
print(f"答案token位置（考虑空格）: {ANSWER_TOKEN_POS}")
print(f"前缀token ids: {prefix_ids}")

with torch.no_grad():
    # 重新运行获取答案token的logits
    input_ids = baseline_input_ids.to(DEVICE)
    attention_mask = baseline_attn.to(DEVICE)

    outputs = model.codi(input_ids=input_ids, use_cache=True, output_hidden_states=True, attention_mask=attention_mask)
    past_key_values = outputs.past_key_values
    latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)

    if USE_PRJ:
        latent_embd = model.prj(latent_embd)

    for i in range(INF_LATENT_ITERATIONS):
        outputs = model.codi(inputs_embeds=latent_embd, use_cache=True, output_hidden_states=True, past_key_values=past_key_values)
        past_key_values = outputs.past_key_values
        latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)
        if USE_PRJ:
            latent_embd = model.prj(latent_embd)

    # 添加EOT token
    eot_emb = model.get_embd(model.codi, model.model_name)(torch.tensor([model.eot_id], dtype=torch.long, device=DEVICE)).unsqueeze(0)

    # 逐个生成token，记录每个位置的logits
    embed_layer = get_embedding_layer(model)
    all_logits = []
    all_generated_tokens = []
    current_emb = eot_emb
    current_past = past_key_values

    # 生成足够的token来覆盖前缀和至少一个答案token
    for step in range(NUM_PREFIX_TOKENS + 10):
        outputs = model.codi(
            inputs_embeds=current_emb,
            use_cache=True,
            past_key_values=current_past
        )
        current_past = outputs.past_key_values
        logits = outputs.logits[0, -1].detach().clone()
        all_logits.append(logits)

        # 贪婪解码
        next_token_id = logits.argmax(dim=-1).item()
        next_token_str = tokenizer.decode([next_token_id])
        all_generated_tokens.append((step, next_token_id, next_token_str))

        if next_token_id == tokenizer.eos_token_id:
            break

        # 获取下一个token的embedding
        current_emb = embed_layer(torch.tensor([next_token_id], dtype=torch.long, device=DEVICE)).unsqueeze(0)

    # 打印前几个生成的token帮助调试
    print(f"生成的前{len(all_generated_tokens)}个token:")
    for idx, tok_id, tok_str in all_generated_tokens[:8]:
        print(f"  位置{idx}: '{tok_str}' (id={tok_id})")
    print(f"前缀token数: {NUM_PREFIX_TOKENS}, 答案token位置: {ANSWER_TOKEN_POS}")

    # 获取答案token位置的logits（ANSWER_TOKEN_POS位置，即空格之后的位置）
    if len(all_logits) > ANSWER_TOKEN_POS:
        answer_token_logits = all_logits[ANSWER_TOKEN_POS]
    else:
        # 如果生成的token不够，使用最后一个可用位置
        answer_token_logits = all_logits[-1]
        print(f"警告: 生成的token数量不足，使用最后一个位置的logits")

    sorted_logits, sorted_indices = torch.sort(answer_token_logits, descending=True)
    BASELINE_TOKEN_ID = sorted_indices[0].item()  # 最大概率的token（答案）- 固定使用
    TARGET_TOKEN_ID = sorted_indices[1].item()     # 第二大概率的token - 固定使用

    # 打印信息
    baseline_token_str = tokenizer.decode([BASELINE_TOKEN_ID])
    target_token_str = tokenizer.decode([TARGET_TOKEN_ID])

    # 计算logits和概率
    baseline_logit = answer_token_logits[BASELINE_TOKEN_ID].item()
    target_logit = answer_token_logits[TARGET_TOKEN_ID].item()
    baseline_prob = torch.softmax(answer_token_logits, dim=-1)[BASELINE_TOKEN_ID].item()
    target_prob = torch.softmax(answer_token_logits, dim=-1)[TARGET_TOKEN_ID].item()
    logit_diff = target_logit - baseline_logit  # 次大 - 最大化（目标是让这个差值越大越好）

    # 也打印一下前几个token帮助调试
    top5_tokens = [tokenizer.decode([sorted_indices[i].item()]) for i in range(min(5, len(sorted_indices)))]
    print(f"Baseline答案token位置:")
    print(f"  最大token: '{baseline_token_str}' (id={BASELINE_TOKEN_ID}), logit={baseline_logit:.4f}, prob={baseline_prob:.4f}")
    print(f"  第二大token: '{target_token_str}' (id={TARGET_TOKEN_ID}), logit={target_logit:.4f}, prob={target_prob:.4f}")
    print(f"  Logit差值(次大-最大): {logit_diff:.4f}")
    print(f"  Top 5 tokens: {top5_tokens}")


# ============ 6. 初始化前缀 ============
print("\n[5] 初始化前缀...")

# 找出有效的单token（用于初始化）
char_tokens = []
for i in range(8000, 200000):
    try:
        decoded = tokenizer.decode([i])
        if decoded.strip() and len(decoded.strip()) > 0:
            char_tokens.append(i)
    except:
        pass

print(f"有效token数量: {len(char_tokens)}")

# 用于候选过滤的备用池：去除含数字的token（仅在 --filter-digits 启用时用于 fallback）
char_tokens_no_digit = [t for t in char_tokens if not contains_digit(t, tokenizer)]
print(f"不含数字的token数量: {len(char_tokens_no_digit)}")

# 随机初始化前缀
prefix_ids = np.random.choice(char_tokens, size=PREFIX_LENGTH, replace=True)
prefix_str = tokenizer.decode(prefix_ids)
'''
prefix_org = "(Texture газIFICATE punishing Fantasy"
prefix_str = prefix_org
prefix_ids = tokenizer.encode(prefix_str)
'''
print(f"初始前缀: '{prefix_str}'")


# ============ 7. 梯度计算函数 ============
print("\n[6] 定义梯度计算函数...")

def compute_gradients(question_str, prefix_ids_arr, mode="both"):
    """
    计算损失相对于前缀的梯度

    mode 参数:
        - "both": 通过 KV cache 和 latent embedding 两者传递梯度 (默认)
        - "kv_only": 只通过 KV cache 传递梯度 (latent_embd detach)

    损失函数: loss = logit[target] - logit[baseline]
    """
    prefix_len = len(prefix_ids_arr)

    # 构建输入
    prefix_str = tokenizer.decode(prefix_ids_arr)
    input_ids, attention_mask = make_prompt(question_str, prefix_str)
    input_ids = input_ids.to(DEVICE)
    attention_mask = attention_mask.to(DEVICE)

    # 获取embedding层
    embed_layer = get_embedding_layer(model)

    # ===== 步骤1: 用 input_embeds 编码，获取 KV =====
    # 创建可训练的embedding
    input_embeds = embed_layer(input_ids).requires_grad_(True)

    # 前向传播 - 编码输入，获取 KV
    outputs = model.codi(
        inputs_embeds=input_embeds,
        use_cache=True,
        output_hidden_states=True,
        attention_mask=attention_mask
    )
    past_key_values = outputs.past_key_values
    latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)

    if USE_PRJ:
        latent_embd = model.prj(latent_embd)

    # kv_only 模式: 断开 latent_embd 的梯度连接
    if mode == "kv_only":
        latent_embd = latent_embd.detach()

    # ===== 步骤2: latent 推理 =====
    for _ in range(INF_LATENT_ITERATIONS):
        outputs = model.codi(
            inputs_embeds=latent_embd,
            use_cache=True,
            output_hidden_states=True,
            past_key_values=past_key_values
        )
        past_key_values = outputs.past_key_values
        latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)
        if USE_PRJ:
            latent_embd = model.prj(latent_embd)

    # kv_only 模式: 再次断开 latent_embd 梯度
    if mode == "kv_only":
        latent_embd = latent_embd.detach()

    # ===== 步骤3: 生成答案 =====
    # EOT
    eot_emb = embed_layer(torch.tensor([model.eot_id], dtype=torch.long, device=DEVICE)).unsqueeze(0)

    # 逐个生成token，在答案token位置获取logits
    current_emb = eot_emb
    current_past = past_key_values

    # 生成前缀token数量+1（包含空格），然后在答案token位置计算损失
    for step in range(ANSWER_TOKEN_POS):
        outputs = model.codi(
            inputs_embeds=current_emb,
            output_hidden_states=False,
            attention_mask=None,
            use_cache=True,
            past_key_values=current_past
        )
        current_past = outputs.past_key_values

        # 贪婪解码获取下一个token
        next_token_id = outputs.logits[0, -1].argmax(dim=-1).item()

        if next_token_id == tokenizer.eos_token_id:
            break

        # 获取下一个token的embedding继续生成
        current_emb = embed_layer(torch.tensor([next_token_id], dtype=torch.long, device=DEVICE)).unsqueeze(0)

    # 在答案token位置获取logits
    answer_outputs = model.codi(
        inputs_embeds=current_emb,
        output_hidden_states=False,
        attention_mask=None,
        use_cache=True,
        past_key_values=current_past
    )

    # ===== 计算损失 =====
    # 损失函数: logit差值
    answer_token_logits = answer_outputs.logits[0, -1]
    logit_loss = answer_token_logits[TARGET_TOKEN_ID] - answer_token_logits[BASELINE_TOKEN_ID]
    total_loss = logit_loss

    # 打印损失值
    print(f"[DEBUG] logit_loss: {logit_loss.item():.4f}, total_loss: {total_loss.item():.4f}")

    # 反向传播
    total_loss.backward()

    # 获取梯度
    input_grad = input_embeds.grad
    if input_grad is None:
        return np.zeros((prefix_len, embed_layer.weight.shape[0])), prefix_len

    # 前缀位置的梯度
    prefix_grad_emb = input_grad[0, :prefix_len]
    token_grads = torch.matmul(prefix_grad_emb, embed_layer.weight.T)
    return token_grads.cpu().detach().to(torch.float32).numpy(), prefix_len


def evaluate(question_str, prefix_ids_arr):
    """评估当前前缀"""
    prefix_str = tokenizer.decode(prefix_ids_arr)
    input_ids, attention_mask = make_prompt(question_str, prefix_str)

    with torch.no_grad():
        text, ids = generate_answer(input_ids, attention_mask)

    pred_answer = extract_answer_number(text)

    return text, pred_answer


def compute_logits_diff(question_str, prefix_ids_arr):
    """
    计算给定前缀的logits差值（次大token的logit - 最大token的logit）
    返回: (logit差值, 最大token_id, 次大token_id, 最大token字符串, 次大token字符串)
    """
    prefix_str = tokenizer.decode(prefix_ids_arr)
    input_ids, attention_mask = make_prompt(question_str, prefix_str)
    input_ids = input_ids.to(DEVICE)
    attention_mask = attention_mask.to(DEVICE)

    embed_layer = get_embedding_layer(model)

    with torch.no_grad():
        outputs = model.codi(
            input_ids=input_ids,
            use_cache=True,
            output_hidden_states=True,
            attention_mask=attention_mask
        )
        past_key_values = outputs.past_key_values
        latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)

        if USE_PRJ:
            latent_embd = model.prj(latent_embd)

        for _ in range(INF_LATENT_ITERATIONS):
            outputs = model.codi(
                inputs_embeds=latent_embd,
                use_cache=True,
                output_hidden_states=True,
                past_key_values=past_key_values
            )
            past_key_values = outputs.past_key_values
            latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)
            if USE_PRJ:
                latent_embd = model.prj(latent_embd)

        # EOT
        eot_emb = embed_layer(torch.tensor([model.eot_id], dtype=torch.long, device=DEVICE)).unsqueeze(0)

        # 逐个生成token，记录每个位置的logits（与baseline代码逻辑一致）
        current_emb = eot_emb
        current_past = past_key_values  # 使用encoder后的past_key_values
        all_logits = []

        for step in range(ANSWER_TOKEN_POS + 5):
            outputs = model.codi(
                inputs_embeds=current_emb,
                use_cache=True,
                past_key_values=current_past
            )
            current_past = outputs.past_key_values
            logits = outputs.logits[0, -1].detach().clone()
            all_logits.append(logits)

            # 贪婪解码
            next_token_id = logits.argmax(dim=-1).item()

            if next_token_id == tokenizer.eos_token_id:
                break

            current_emb = embed_layer(torch.tensor([next_token_id], dtype=torch.long, device=DEVICE)).unsqueeze(0)

        # 获取答案token位置的logits
        if len(all_logits) > ANSWER_TOKEN_POS:
            answer_token_logits = all_logits[ANSWER_TOKEN_POS]
        else:
            answer_token_logits = all_logits[-1]

        # 使用固定的 token id（baseline 时确定的）
        baseline_logit = answer_token_logits[BASELINE_TOKEN_ID].item()
        target_logit = answer_token_logits[TARGET_TOKEN_ID].item()

        # 计算概率
        baseline_prob = torch.softmax(answer_token_logits, dim=-1)[BASELINE_TOKEN_ID].item()
        target_prob = torch.softmax(answer_token_logits, dim=-1)[TARGET_TOKEN_ID].item()

        logit_diff = target_logit - baseline_logit  # 次大 - 最大

        baseline_str = tokenizer.decode([BASELINE_TOKEN_ID])
        target_str = tokenizer.decode([TARGET_TOKEN_ID])

    return logit_diff, baseline_prob, target_prob, baseline_str, target_str


# ============ 8. GCG 迭代 ============
print("\n[7] 开始GCG迭代...")

current_text, current_answer = evaluate(question, prefix_ids)
current_correct = (str(current_answer) == answer_str or
                  abs(current_answer - float(answer_str)) < 0.01 if current_answer != float('inf') and answer_str.lstrip('-').replace('.','',1).isdigit() else False)

# 计算初始前缀的logits差值
initial_logit_diff, init_baseline_prob, init_target_prob, baseline_str, target_str = compute_logits_diff(question, prefix_ids)
print(f"初始前缀 logits差值: {initial_logit_diff:.4f} (top1='{baseline_str}' prob={init_baseline_prob:.4f}, top2='{target_str}' prob={init_target_prob:.4f})")
print(f"固定token - BASELINE_TOKEN_ID={BASELINE_TOKEN_ID}, TARGET_TOKEN_ID={TARGET_TOKEN_ID}")

print(f"初始生成: '{current_text}'")
print(f"初始答案: {current_answer}")
print(f"是否正确: {current_correct}")

best_prefix = prefix_ids.copy()
best_correct = current_correct
best_logit_diff = initial_logit_diff  # 记录最佳的logits差值

scores_list = []
for iteration in range(N_ITERS):
    # 计算梯度
    mode = "both" if args.pass_gradient_through_latent else "kv_only"
    gradients, actual_len = compute_gradients(question, prefix_ids, mode=mode)
    curr_len = len(prefix_ids)

    # 生成候选组合
    sampled_combinations = [{}]
    for _ in range(BATCH_SIZE):
        # 随机选择替换1~curr_len个位置
        n_changes = np.random.randint(1, curr_len + 1)
        positions = np.random.choice(curr_len, size=n_changes, replace=False)

        combo = {}
        for pos in positions:
            if args.candidate_selection == "gradient":
                # 使用梯度：选择梯度最大的候选（启用 --filter-digits 时去掉含数字的 token，
                # 并用梯度较低的下一个不含数字的 token 补齐，使候选数仍为 TOP_K）
                pos_grads = gradients[pos]
                top_k_indices = select_topk_candidates(
                    pos_grads=pos_grads,
                    top_k=TOP_K,
                    tokenizer=tokenizer,
                    filter_digits=args.filter_digits,
                    fallback_pool=char_tokens_no_digit,
                )
                combo[pos] = np.random.choice(top_k_indices)
            else:
                # 随机选择：从不使用梯度，直接从所有有效token中随机选择
                # 启用 --filter-digits 时改为从不含数字的池中采样
                pool = char_tokens_no_digit if args.filter_digits else char_tokens
                combo[pos] = np.random.choice(pool)

        sampled_combinations.append(combo)

    # 评估候选 - 使用logits差值
    scores = []
    for combo in sampled_combinations:
        new_prefix = prefix_ids.copy()
        for pos, tok_id in combo.items():
            new_prefix[pos] = tok_id

        # 计算logits差值（目标是让这个差值最大化，即次大token的logit超过最大token）
        logit_diff, _, _, _, _ = compute_logits_diff(question, new_prefix)
        scores.append((logit_diff, combo))

    # 选择最佳候选（logits差值最大的）
    '''scores_np = np.array(scores)
    best_idx = np.argmax(scores_np)
    best_combo = sampled_combinations[best_idx]
    best_score = scores_np[best_idx]  # 最佳候选的logits差值'''
    scores = sorted(scores, key=lambda x: x[0])
    best_combo = scores[-1][1]
    best_score = scores[-1][0]

    # 更新前缀
    for pos, tok_id in best_combo.items():
        prefix_ids[pos] = tok_id
    
    scores_list.append(best_score)
    # 记录最佳差值
    if best_score > best_logit_diff:
        best_logit_diff = best_score
        best_prefix = prefix_ids.copy()

    if best_logit_diff > 1.5:
        break

    # 获取当前状态
    current_text, current_answer = evaluate(question, prefix_ids)
    current_correct = (str(current_answer) == answer_str or
                      abs(current_answer - float(answer_str)) < 0.01 if current_answer != float('inf') and answer_str.lstrip('-').replace('.','',1).isdigit() else False)

    # 获取当前logits信息（固定token id）
    curr_logit_diff, curr_baseline_prob, curr_target_prob, _, _ = compute_logits_diff(question, prefix_ids)

    print(f"Iter {iteration+1}: logit差值={curr_logit_diff:.4f} (top1='{baseline_str}' prob={curr_baseline_prob:.4f}, top2='{target_str}' prob={curr_target_prob:.4f}), 最佳差值={best_logit_diff:.4f}, 答案={current_answer}, 正确={current_correct}")



# ============ 9. 结果 ============
print("\n" + "=" * 60)
print("最终结果")
print("=" * 60)

final_prefix_str = tokenizer.decode(best_prefix)
print(f"对抗前缀: '{final_prefix_str}'")

# 评估无前缀
clean_text, clean_answer = evaluate(question, np.array([]))
clean_correct = (str(clean_answer) == answer_str or
                abs(clean_answer - float(answer_str)) < 0.01 if clean_answer != float('inf') and answer_str.lstrip('-').replace('.','',1).isdigit() else False)

# 计算无前缀时的logit差值
clean_logit_diff, _, _, _, _ = compute_logits_diff(question, np.array([]))

# 评估有前缀
adv_text, adv_answer = evaluate(question, best_prefix)
adv_correct = (str(adv_answer) == answer_str or
              abs(adv_answer - float(answer_str)) < 0.01 if adv_answer != float('inf') and answer_str.lstrip('-').replace('.','',1).isdigit() else False)

# 计算有前缀时的logit差值
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
    "baseline_logit_diff": logit_diff,  # baseline时的logit差值
    "clean_logit_diff": clean_logit_diff,  # 无前缀时的logit差值
    "adv_logit_diff": adv_logit_diff,  # 有前缀时的logit差值
    "best_logit_diff": best_logit_diff,  # 找到的最佳logit差值
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
