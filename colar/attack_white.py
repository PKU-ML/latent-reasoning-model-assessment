"""
================================================================================
CoLaR模型 GCG (Greedy Coordinate Gradient) 对抗前缀攻击
================================================================================

一、算法概述
------------
本代码实现了一种针对CoLaR模型的GCG对抗攻击，目标是通过优化一个前缀字符串，
使得模型在给定问题下输出错误的答案。

二、CoLaR模型推理流程
--------------------
CoLaR是一种具有连续思维（latent reasoning）能力的模型，其推理过程如下：

1. 输入编码：问题 + "(Thinking speed: N)###" 后缀
2. Question Forward: 将问题token序列通过LLM获取hidden states
3. Latent推理循环（最多max_n_latent_forward次）:
   - LatentPolicy (MLP) 将LLM最后一层的hidden states映射为潜在嵌入分布（均值+方差）
   - 从分布中采样得到新的嵌入向量: sampled = mean + std * epsilon
   - 将采样嵌入喂回LLM继续推理
   - 检查是否生成了 ### (end-of-thinking) token，若是则停止循环
4. 添加 ### token标记思考结束
5. Answer Generation: LLM基于完整序列自回归生成答案

三、攻击目标
------------
模型输出格式为 "... ### Answer: {答案}"

攻击目标：
- 在答案token位置，找到top-1 token（最大概率的token，记作t1）和top-2 token（次大概率，记作t2）
- 优化目标：让 t2 的logit - t1 的logit 差值最大化
- 当这个差值足够大时，t2的概率可能超过t1，导致模型输出错误答案

四、梯度流动
------------
答案token的梯度反向传播路径：
logits[答案位置] -> LLM输出层 -> ... -> Latent推理层 ->
LatentPolicy采样 -> Question Forward的hidden states -> 前缀token的embedding

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
from pathlib import Path

import sys
sys.path.insert(0, os.environ.get("COLAR_WORKSPACE", str(Path(__file__).resolve().parent / "colar")))

import yaml
from omegaconf import OmegaConf

# ============ 命令行参数 ============
parser = argparse.ArgumentParser(description="CoLaR GCG 对抗前缀攻击")
parser.add_argument("--candidate-selection", type=str, choices=["gradient", "random"], default="gradient",
                    help="候选token选择方式: gradient(梯度选择) 或 random(随机选择)")
parser.add_argument("--problem-id", type=int, default=None,
                    help="指定攻击的问题ID (默认自动分配)")
parser.add_argument("--prefix-length", type=int, default=5,
                    help="前缀长度 (默认10)")
parser.add_argument("--n-iters", type=int, default=30,
                    help="迭代次数 (默认500)")
parser.add_argument("--top-k", type=int, default=256,
                    help="每个位置考虑top-k个候选 (默认256)")
parser.add_argument("--batch-size", type=int, default=40,
                    help="候选batch大小 (默认64)")
parser.add_argument("--latent-gradient", type=str, choices=['true', 'false'], default='true',
                    help="梯度是否流经latent向量: true 或 false (默认true)")
parser.add_argument("--output-dir", type=str,
                    default=os.environ.get(
                        "COLAR_OUTPUT_DIR",
                        str(Path(__file__).resolve().parent / "adv_results"),
                    ),
                    help="结果保存目录 (默认: <colar>/adv_results；可通过 COLAR_OUTPUT_DIR 覆盖)")
parser.add_argument("--seed", type=int, default=20,
                    help="随机种子 (默认20)")
parser.add_argument("--dataset", type=str, choices=["gsm8k", "MultiArith", "SVAMP"], default="gsm8k",
                    help="指定要攻击的数据集 (默认 gsm8k)")
parser.add_argument("--checkpoint-path", type=str,
                    default=os.environ.get(
                        "COLAR_CHECKPOINT",
                        str(
                            Path(__file__).resolve().parent
                            / "colar"
                            / "models"
                            / "CoLaR"
                            / "logs"
                            / "colar"
                            / "qsa-gsm"
                            / "colar-final"
                            / "checkpoints"
                            / "colar_best.ckpt"
                        ),
                    ),
                    help="模型checkpoint路径 (默认读取 COLAR_CHECKPOINT 环境变量)")
parser.add_argument("--workspace-path", type=str,
                    default=os.environ.get(
                        "COLAR_WORKSPACE",
                        str(Path(__file__).resolve().parent / "colar"),
                    ),
                    help="工作目录路径 (默认读取 COLAR_WORKSPACE 环境变量)")
args = parser.parse_args()
args.pass_gradient_through_latent = args.latent_gradient == 'true'

# 设置随机种子
import random
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)

# ============ 配置 ============
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用设备: {DEVICE}")

# 模型配置
CHECKPOINT_PATH = args.checkpoint_path
WORKSPACE_PATH = args.workspace_path

# 推理参数
MAX_N_LATENT_FORWARD = 64  # latent推理最大步数
LATENT_TEMPERATURE = 1e-9  # 确定性模式
COMPRESSION_FACTOR = 5

# GCG 参数
PREFIX_LENGTH = args.prefix_length
N_ITERS = args.n_iters
TOP_K = args.top_k
BATCH_SIZE = args.batch_size

# 评估数据配置 — 数据目录通过 DATA_DIR 环境变量覆盖
DATA_NAME = args.dataset
DATA_DIR = Path(os.environ.get("DATA_DIR", str(Path(__file__).resolve().parent.parent / "data")))
DATA_PATH = DATA_DIR / f"{DATA_NAME}.json"
OUTPUT_DIR = Path(args.output_dir)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if args.problem_id is not None:
    PROBLEM_ID = args.problem_id
else:
    existing_files = list(OUTPUT_DIR.glob("problem_*.json"))
    if len(existing_files) > 0:
        existing_ids = [int(f.stem.split("_")[1]) for f in existing_files]
        PROBLEM_ID = max(existing_ids) + 1
    else:
        PROBLEM_ID = 0

print("=" * 60)
print("CoLaR GCG 寻找对抗前缀")
print(f"模式: {'梯度选择候选' if args.candidate_selection == 'gradient' else '随机选择候选'}")
print(f"梯度流经latent: {args.pass_gradient_through_latent}")
print("=" * 60)


# ============ 1. 加载数据 ============
print(f"\n[1] 加载{DATA_NAME}数据...")

def load_local_dataset(data_path):
    questions = []
    answers = []
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    for item in data:
        questions.append(item['question'])
        answers.append(item['answer'])
    return questions, answers

dataset = load_local_dataset(str(DATA_PATH))
questions = dataset[0]
answers = dataset[1]

question = questions[PROBLEM_ID]
answer_str = answers[PROBLEM_ID]

# 提取答案数字
def extract_answer_number(answer_str):
    """从答案字符串中提取数字"""
    answer_str = answer_str.strip("#\n ").rstrip(".").replace(",", "").lower()
    try:
        return float(answer_str)
    except ValueError:
        match = re.search(r'-?\d+\.?\d*', answer_str)
        if match:
            return float(match.group())
        return float('inf')

gt_answer_num = extract_answer_number(answer_str)
print(f"问题: {question[:100]}...")
print(f"正确答案: {answer_str} (提取数字: {gt_answer_num})")


# ============ 2. 加载模型 ============
print("\n[2] 加载CoLaR模型...")

os.chdir(WORKSPACE_PATH)

from colar.src.models.colar import LitCoLaR
from colar.src.utils.utils import get_position_ids_from_attention_mask


def set_deterministic_mode(model, deterministic=True):
    """设置或取消确定性模式"""
    if deterministic:
        # 保存原始配置
        model._orig_answer_config = model.model_kwargs.answer_generation_config.copy()
        model._orig_latent_temp = model.model_kwargs.latent_generation_config.get("latent_temperature", 1.0)

        # 设置确定性模式
        model.model_kwargs.answer_generation_config.do_sample = False
        model.model_kwargs.latent_generation_config.latent_temperature = 1e-9
    else:
        # 恢复原始配置
        if hasattr(model, '_orig_answer_config'):
            model.model_kwargs.answer_generation_config.do_sample = model._orig_answer_config.get("do_sample", True)
        if hasattr(model, '_orig_latent_temp'):
            model.model_kwargs.latent_generation_config.latent_temperature = model._orig_latent_temp

# Load hparams.yaml
hparams_path = os.path.dirname(CHECKPOINT_PATH).replace('/checkpoints', '') + '/hparams.yaml'
with open(hparams_path, 'r') as f:
    hparams_data = yaml.safe_load(f)

all_config = OmegaConf.create(hparams_data['all_config'])
all_config.args = OmegaConf.create({
    "workspace_path": WORKSPACE_PATH,
    "no_log": True,
})

# Create model
model = LitCoLaR(
    model_kwargs=all_config.model.model_kwargs,
    training_kwargs=all_config.model.training_kwargs,
    all_config=all_config,
)

# Load checkpoint
checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
state_dict = checkpoint["state_dict"]
model.load_state_dict(state_dict=state_dict, strict=False)

tokenizer = model.tokenizer
model.eval()
model.to(DEVICE)

print(f"模型加载完成")
print(f"max_n_latent_forward: {MAX_N_LATENT_FORWARD}")
print(f"latent_temperature: {LATENT_TEMPERATURE}")


# ============ 3. 辅助函数 ============

def extract_answer_from_output(output_string: str, answer_template="Answer:"):
    """从模型输出中提取答案"""
    try:
        return output_string.strip('#').split(answer_template)[-1]
    except (ValueError, IndexError):
        return output_string


def verify_answer(gt_answer: str, pred_answer: str) -> bool:
    """验证预测答案是否正确"""
    def get_pure_string(s: str):
        return s.strip("#\n ").rstrip(".").replace(",", "").lower()

    gt_answer = get_pure_string(gt_answer)
    pred_answer = get_pure_string(pred_answer)

    try:
        gt_num = float(gt_answer)
        pred_num = float(pred_answer)
        return abs(gt_num - pred_num) < 1e-6
    except ValueError:
        pass

    return gt_answer == pred_answer


def prepare_inputs(text_list, padding_side="left", suffix=""):
    """准备模型输入"""
    if isinstance(text_list, str):
        text_list = [text_list]

    batch_size = len(text_list)
    base_template = "Question: {} Let's think step by step:"
    speed_template = "(Thinking speed: {})"
    thinking_separator = "###"

    if suffix:
        full_texts = [base_template.format(text) + speed_template.format(COMPRESSION_FACTOR) + thinking_separator + suffix
                      for text in text_list]
    else:
        full_texts = [base_template.format(text) + speed_template.format(COMPRESSION_FACTOR) + thinking_separator
                      for text in text_list]

    inputs = tokenizer.batch_encode_plus(
        full_texts,
        return_tensors="pt",
        add_special_tokens=False,
        padding="longest",
        padding_side=padding_side
    )
    input_ids = inputs["input_ids"].to(DEVICE)
    attention_mask = inputs["attention_mask"].to(DEVICE)
    return input_ids, attention_mask


def make_prefix_prompt(question_str, prefix_str=""):
    """
    构建带前缀的prompt
    格式: 前缀 + Question: {question} Let's think step by step: (Thinking speed: N)###
    """
    full_text = prefix_str.strip() + " Question: " + question_str.strip() + " Let's think step by step:"

    inputs = tokenizer(full_text, return_tensors="pt", padding="longest", padding_side="left")
    input_ids = inputs["input_ids"].to(DEVICE)
    attention_mask = inputs["attention_mask"].to(DEVICE)

    # 添加 "(Thinking speed: N)###"
    suffix_ids, suffix_mask = prepare_inputs([""], suffix="")
    # 获取问题部分的长度（到 "###" 之前）
    return input_ids, attention_mask


def generate_answer_with_latent(question_str, prefix_str="", max_new_tokens=16, return_hidden=False):
    """
    使用CoLaR的latent_generate生成答案
    为了能够计算梯度，这个函数做了适配处理
    """
    # 构建输入 - 使用与test.py相同的方式
    if prefix_str:
        full_question = prefix_str + question_str
    else:
        full_question = question_str.strip()

    with torch.no_grad():
        # 使用模型的latent_generate方法（与test.py相同）
        pred_ids, n_latent_forward = model.latent_generate(questions=[full_question])

        output_string = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)[0]
        pred_answer = extract_answer_from_output(output_string)

    if return_hidden:
        return output_string, pred_answer, pred_ids
    return output_string, pred_answer


# ============ 4. 测试Baseline ============
print("\n[3] 测试Baseline...")

# 设置确定性模式（与test.py一致）
set_deterministic_mode(model, deterministic=True)

# 设置随机种子确保可复现
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

baseline_output, baseline_answer = generate_answer_with_latent(question, "")
baseline_correct = verify_answer(answer_str, baseline_answer)

print(f"Baseline输出: '{baseline_output}'")
print(f"Baseline提取答案: '{baseline_answer}'")
print(f"正确答案: '{answer_str}'")
print(f"Baseline是否正确: {baseline_correct}")


# ============ 5. 获取答案token位置 ============
print("\n[4] 分析答案token位置...")

# 策略：直接利用model.latent_generate()的结果
# 关键：我们知道正确的答案token是"260"，直接从生成的pred_ids中获取答案位置

# 设置随机种子确保与section[3]产生相同的latent采样
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

with torch.no_grad():
    full_question = question.strip()
    pred_ids, n_latent_forward = model.latent_generate(questions=[full_question])

    # 解码获取完整输出
    output_string = tokenizer.decode(pred_ids[0], skip_special_tokens=True)
    print(f"生成输出: '{output_string}'")

    # 分析输出结构
    answer_template = "Answer:"
    if answer_template in output_string:
        answer_part = output_string.split(answer_template)[-1].strip()
        print(f"答案部分: '{answer_part}'")

    # 获取完整的token序列
    full_tokens = pred_ids[0].tolist()
    print(f"完整token序列长度: {len(full_tokens)}")
    print(f"完整token序列: {full_tokens}")

    # 找到 "Answer:" token的位置
    answer_template_ids = tokenizer.encode(answer_template, add_special_tokens=False)
    print(f"Answer模板token ids: {answer_template_ids}")

    # 在token序列中查找 "Answer:" 的结束位置
    answer_start_pos = None
    for i in range(len(full_tokens) - len(answer_template_ids)):
        if full_tokens[i:i+len(answer_template_ids)] == answer_template_ids:
            answer_start_pos = i + len(answer_template_ids)
            break

    if answer_start_pos is not None:
        print(f"Answer模板结束位置（token索引）: {answer_start_pos}")
        ANSWER_TOKEN_POS = answer_start_pos
        print(f"答案token位置: {ANSWER_TOKEN_POS}")

        # 打印答案部分的tokens
        answer_tokens = full_tokens[answer_start_pos:answer_start_pos+10]
        answer_token_strs = [tokenizer.decode([t]) for t in answer_tokens]
        print(f"答案部分tokens: {list(zip(answer_tokens, answer_token_strs))}")
    else:
        ANSWER_TOKEN_POS = 5
        print(f"未找到Answer模板，使用默认答案位置: {ANSWER_TOKEN_POS}")

    # 正确的答案token
    answer_token_id = full_tokens[ANSWER_TOKEN_POS] if ANSWER_TOKEN_POS < len(full_tokens) else full_tokens[-1]
    print(f"答案token id: {answer_token_id}, token: '{tokenizer.decode([answer_token_id])}'")

    # 现在我们需要获取这个答案token对应的logits
    # 方法：使用llm.generate的score模式，或者逐个force forward

    # 由于我们无法直接获取latent_generate内部的past_key_values，
    # 我们使用一个技巧：直接调用model.llm.generate并获取每一步的logits

    # 首先，我们需要重建与latent_generate完全相同的输入序列
    # 然后用force decoding方式生成答案，获取logits

    # 重新运行latent forward获取past_key_values
    speed = COMPRESSION_FACTOR
    suffix = f"(Thinking speed: {speed})###"

    question_input_ids, question_attention_mask = model.prepare_inputs(
        [full_question],
        padding_side="left",
        part="question",
        suffix=suffix,
    )

    question_position_ids = get_position_ids_from_attention_mask(question_attention_mask)
    question_embeds = model.embedding(question_input_ids)

    outputs = model.llm.forward(
        inputs_embeds=question_embeds,
        attention_mask=question_attention_mask,
        position_ids=question_position_ids,
        output_hidden_states=True,
    )

    all_attention_mask = question_attention_mask
    current_position_ids = question_position_ids[:, -1:]
    past_key_values = outputs.past_key_values
    is_done = torch.zeros(size=(1, 1), device=DEVICE, dtype=torch.bool)

    # Latent forward
    for _ in range(MAX_N_LATENT_FORWARD):
        distributions = model.latent_policy.forward(
            outputs.hidden_states[-1][:, -1:, :],
            temperature=LATENT_TEMPERATURE
        )
        current_inputs_embeds = distributions.rsample() * model.embeds_std

        not_is_done_long = (~is_done).long()
        all_attention_mask = torch.cat([all_attention_mask, not_is_done_long], dim=1)
        current_position_ids = current_position_ids + not_is_done_long

        outputs = model.llm.forward(
            inputs_embeds=current_inputs_embeds,
            attention_mask=all_attention_mask,
            position_ids=current_position_ids,
            past_key_values=past_key_values,
            output_hidden_states=True,
        )
        past_key_values = outputs.past_key_values

        last_logits = outputs.logits[:, -1]
        probs = torch.softmax(last_logits, dim=-1)
        batch_next_token = torch.multinomial(probs, num_samples=1)

        is_eol = batch_next_token == model.thinking_separator_id
        is_done = is_done | is_eol
        if is_done.all():
            break

    # Add ###
    end_of_thinking_ids = torch.ones(size=(1, 1), device=DEVICE, dtype=torch.long) * model.thinking_separator_id
    end_of_thinking_embeds = model.embedding(end_of_thinking_ids)
    all_attention_mask = torch.cat([
        all_attention_mask,
        torch.ones(size=(1, 1), device=DEVICE, dtype=torch.long),
    ], dim=1)

    # 从这里开始，用FORCE DECODING方式：使用pred_ids中的答案token作为输入
    # 这样我们就用与第一次相同的方式获取了past_key_values，然后强制解码
    embed_layer = model.embedding

    # 答案token的offset：答案部分是从answer_start_pos开始的
    # 但在完整序列中，答案部分在 answer_start_pos 位置
    # 我们需要从Answer:之后的第一个token开始

    # 强制解码：使用ground truth的token
    current_emb = end_of_thinking_embeds
    current_mask = all_attention_mask
    current_past = past_key_values

    generated_tokens = []
    all_logits_list = []

    # 跳过Answer:模板，直接到答案部分
    # 我们需要跳过 answer_start_pos 个token（包括Answer:模板）
    # 但前向传播时我们从 ### 之后开始，所以要映射一下

    # 实际上，我们想要获取 logits[answer_start_pos] 位置对应的logits
    # 这对应的是在答案token生成之前的那个logits

    # 更简单的方法：直接生成答案部分，从答案token之前的那个token开始

    # 从头开始，逐个前向，但强制使用pred_ids中的token
    for step in range(len(full_tokens) + 10):
        outputs = model.llm.forward(
            inputs_embeds=current_emb,
            attention_mask=current_mask,
            past_key_values=current_past,
        )

        logits = outputs.logits[:, -1]
        all_logits_list.append(logits[0].detach().clone())

        # 使用ground truth的token（强制解码）
        if step < len(full_tokens):
            gt_token = full_tokens[step]
        else:
            gt_token = tokenizer.eos_token_id

        next_token_str = tokenizer.decode([gt_token])
        generated_tokens.append((step, gt_token, next_token_str))

        if gt_token == tokenizer.eos_token_id:
            break

        current_emb = embed_layer(torch.tensor([gt_token], device=DEVICE)).unsqueeze(0)
        current_mask = torch.cat([current_mask, torch.ones(1, 1, device=DEVICE, dtype=torch.long)], dim=1)
        current_past = outputs.past_key_values

    print(f"Force decoding生成的前{len(generated_tokens)}个token:")
    for idx, tok_id, tok_str in generated_tokens[:15]:
        print(f"  位置{idx}: '{tok_str}' (id={tok_id})")

    # 找到答案token在generated_tokens中的位置
    # 答案token应该是 Answer: 之后的第一个数字token
    answer_gen_pos = None
    for i, (step, tok_id, tok_str) in enumerate(generated_tokens):
        if i >= answer_start_pos and i < len(generated_tokens):
            answer_gen_pos = i
            break

    if answer_gen_pos is None:
        answer_gen_pos = answer_start_pos

    print(f"答案token在generated中的位置: {answer_gen_pos}, 对应token: '{generated_tokens[answer_gen_pos][2]}'")

    # 获取答案token位置的logits
    if answer_gen_pos < len(all_logits_list):
        answer_token_logits = all_logits_list[answer_gen_pos]
    else:
        answer_token_logits = all_logits_list[-1]

    sorted_logits, sorted_indices = torch.sort(answer_token_logits, descending=True)
    BASELINE_TOKEN_ID = sorted_indices[0].item()
    TARGET_TOKEN_ID = sorted_indices[1].item()

    baseline_token_str = tokenizer.decode([BASELINE_TOKEN_ID])
    target_token_str = tokenizer.decode([TARGET_TOKEN_ID])

    baseline_logit = answer_token_logits[BASELINE_TOKEN_ID].item()
    target_logit = answer_token_logits[TARGET_TOKEN_ID].item()
    baseline_prob = torch.softmax(answer_token_logits, dim=-1)[BASELINE_TOKEN_ID].item()
    target_prob = torch.softmax(answer_token_logits, dim=-1)[TARGET_TOKEN_ID].item()
    logit_diff = target_logit - baseline_logit

    top5_tokens = [tokenizer.decode([sorted_indices[i].item()]) for i in range(min(5, len(sorted_indices)))]
    print(f"\nBaseline答案token位置:")
    print(f"  最大token: '{baseline_token_str}' (id={BASELINE_TOKEN_ID}), logit={baseline_logit:.4f}, prob={baseline_prob:.4f}")
    print(f"  第二大token: '{target_token_str}' (id={TARGET_TOKEN_ID}), logit={target_logit:.4f}, prob={target_prob:.4f}")
    print(f"  Logit差值(次大-最大): {logit_diff:.4f}")
    print(f"  Top 5 tokens: {top5_tokens}")


# ============ 6. 初始化前缀 ============
print("\n[5] 初始化前缀...")

# 找出有效的token
char_tokens = []
for i in range(8000, 200000):
    try:
        decoded = tokenizer.decode([i])
        if decoded.strip() and len(decoded.strip()) > 0:
            char_tokens.append(i)
    except:
        pass

print(f"有效token数量: {len(char_tokens)}")

# 随机初始化前缀
prefix_ids = np.random.choice(char_tokens, size=PREFIX_LENGTH, replace=True)
prefix_str = tokenizer.decode(prefix_ids)
print(f"初始前缀: '{prefix_str}'")


# ============ 7. 梯度计算函数 ============
print("\n[6] 定义梯度计算函数...")

def compute_gradients(question_str, prefix_ids_arr, mode="both"):
    """
    计算损失相对于前缀的梯度

    mode 参数:
        - "both": 通过latent embedding传递梯度
        - "kv_only": 断开latent embedding的梯度

    损失函数: loss = logit[target] - logit[baseline]
    """
    prefix_len = len(prefix_ids_arr)

    # 构建输入 - 使用与latent_generate相同的方式
    prefix_str = tokenizer.decode(prefix_ids_arr)
    full_question = prefix_str + question_str

    # 使用model.prepare_inputs构建输入（与latent_generate一致）
    speed = COMPRESSION_FACTOR
    suffix = f"(Thinking speed: {speed})###"
    question_input_ids, question_attention_mask = model.prepare_inputs(
        [full_question],
        padding_side="left",
        part="question",
        suffix=suffix,
    )

    # 创建可训练的embedding
    input_embeds = model.embedding(question_input_ids).requires_grad_(True)
    attention_mask = question_attention_mask
    position_ids = get_position_ids_from_attention_mask(attention_mask)

    # Question forward
    outputs = model.llm.forward(
        inputs_embeds=input_embeds,
        attention_mask=attention_mask,
        position_ids=position_ids,
        output_hidden_states=True,
    )

    past_key_values = outputs.past_key_values
    all_attention_mask = attention_mask
    current_position_ids = position_ids[:, -1:]
    is_done = torch.zeros(size=(1, 1), device=DEVICE, dtype=torch.bool)

    # Latent forward
    latent_embeds_list = []
    for _ in range(MAX_N_LATENT_FORWARD):
        distributions = model.latent_policy.forward(
            outputs.hidden_states[-1][:, -1:, :],
            temperature=LATENT_TEMPERATURE
        )

        if mode == "kv_only":
            # 断开latent梯度
            latent_emb = distributions.rsample().detach()
        else:
            latent_emb = distributions.rsample()

        latent_emb = latent_emb * model.embeds_std

        not_is_done_long = (~is_done).long()
        all_attention_mask = torch.cat([all_attention_mask, not_is_done_long], dim=1)
        current_position_ids = current_position_ids + not_is_done_long

        outputs = model.llm.forward(
            inputs_embeds=latent_emb,
            attention_mask=all_attention_mask,
            position_ids=current_position_ids,
            past_key_values=past_key_values,
            output_hidden_states=True,
        )
        past_key_values = outputs.past_key_values

        latent_embeds_list.append(latent_emb)

        last_logits = outputs.logits[:, -1]
        probs = torch.softmax(last_logits, dim=-1)
        batch_next_token = torch.multinomial(probs, num_samples=1)

        is_eol = batch_next_token == model.thinking_separator_id
        is_done = is_done | is_eol
        if is_done.all():
            break

    # Add ###
    end_of_thinking_ids = torch.ones(size=(1, 1), device=DEVICE, dtype=torch.long) * model.thinking_separator_id
    end_of_thinking_embeds = model.embedding(end_of_thinking_ids)
    all_attention_mask = torch.cat([
        all_attention_mask,
        torch.ones(size=(1, 1), device=DEVICE, dtype=torch.long),
    ], dim=1)

    # 拼接所有embeds用于答案生成
    all_inputs_embeds = torch.cat([input_embeds] + latent_embeds_list + [end_of_thinking_embeds], dim=1)

    # Answer generation - 逐个生成直到答案token位置
    current_past = past_key_values
    current_emb = end_of_thinking_embeds

    for step in range(ANSWER_TOKEN_POS):
        outputs = model.llm.forward(
            inputs_embeds=current_emb,
            attention_mask=None,
            past_key_values=current_past
        )
        current_past = outputs.past_key_values

        # 贪婪解码
        next_token_id = outputs.logits[0, -1].argmax(dim=-1).item()

        if next_token_id == tokenizer.eos_token_id:
            break

        current_emb = model.embedding(torch.tensor([next_token_id], device=DEVICE)).unsqueeze(0)

    # 在答案token位置获取logits
    answer_outputs = model.llm.forward(
        inputs_embeds=current_emb,
        attention_mask=None,
        past_key_values=current_past
    )

    # 计算损失
    answer_token_logits = answer_outputs.logits[0, -1]
    logit_loss = answer_token_logits[TARGET_TOKEN_ID] - answer_token_logits[BASELINE_TOKEN_ID]
    total_loss = logit_loss

    print(f"[DEBUG] logit_loss: {logit_loss.item():.4f}")

    # 反向传播
    total_loss.backward()

    # 获取梯度
    input_grad = input_embeds.grad
    if input_grad is None:
        return np.zeros((prefix_len, model.embedding.weight.shape[0])), prefix_len

    # 前缀位置的梯度
    prefix_grad_emb = input_grad[0, :prefix_len]
    token_grads = torch.matmul(prefix_grad_emb, model.embedding.weight.T)
    return token_grads.cpu().detach().to(torch.float32).numpy(), prefix_len


def evaluate(question_str, prefix_ids_arr):
    """评估当前前缀"""
    prefix_str = tokenizer.decode(prefix_ids_arr)
    output, pred_answer = generate_answer_with_latent(question_str, prefix_str)
    is_correct = verify_answer(answer_str, pred_answer)
    return output, pred_answer, is_correct


def compute_logits_diff(question_str, prefix_ids_arr):
    """
    计算给定前缀的logits差值
    使用与latent_generate相同的方式
    """
    prefix_str = tokenizer.decode(prefix_ids_arr)
    full_question = prefix_str + question_str

    speed = COMPRESSION_FACTOR
    suffix = f"(Thinking speed: {speed})###"
    question_input_ids, attention_mask = model.prepare_inputs(
        [full_question],
        padding_side="left",
        part="question",
        suffix=suffix,
    )
    question_input_ids = question_input_ids.to(DEVICE)
    attention_mask = attention_mask.to(DEVICE)

    with torch.no_grad():
        question_embeds = model.embedding(question_input_ids)
        position_ids = get_position_ids_from_attention_mask(attention_mask)

        outputs = model.llm.forward(
            inputs_embeds=question_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_hidden_states=True,
        )

        past_key_values = outputs.past_key_values
        all_attention_mask = attention_mask
        current_position_ids = position_ids[:, -1:]
        is_done = torch.zeros(size=(1, 1), device=DEVICE, dtype=torch.bool)

        for _ in range(MAX_N_LATENT_FORWARD):
            distributions = model.latent_policy.forward(
                outputs.hidden_states[-1][:, -1:, :],
                temperature=LATENT_TEMPERATURE
            )
            current_inputs_embeds = distributions.rsample() * model.embeds_std

            not_is_done_long = (~is_done).long()
            all_attention_mask = torch.cat([all_attention_mask, not_is_done_long], dim=1)
            current_position_ids = current_position_ids + not_is_done_long

            outputs = model.llm.forward(
                inputs_embeds=current_inputs_embeds,
                attention_mask=all_attention_mask,
                position_ids=current_position_ids,
                past_key_values=past_key_values,
                output_hidden_states=True,
            )
            past_key_values = outputs.past_key_values

            last_logits = outputs.logits[:, -1]
            probs = torch.softmax(last_logits, dim=-1)
            batch_next_token = torch.multinomial(probs, num_samples=1)

            is_eol = batch_next_token == model.thinking_separator_id
            is_done = is_done | is_eol
            if is_done.all():
                break

        # Add ###
        end_of_thinking_ids = torch.ones(size=(1, 1), device=DEVICE, dtype=torch.long) * model.thinking_separator_id
        end_of_thinking_embeds = model.embedding(end_of_thinking_ids)

        # 生成答案token
        current_emb = end_of_thinking_embeds
        current_past = past_key_values

        answer_token_logits = None
        for step in range(ANSWER_TOKEN_POS + 5):
            outputs = model.llm.forward(
                inputs_embeds=current_emb,
                attention_mask=None,
                past_key_values=current_past
            )
            current_past = outputs.past_key_values
            logits = outputs.logits[0, -1].detach().clone()

            # 保存ANSWER_TOKEN_POS位置的logits
            if step == ANSWER_TOKEN_POS:
                answer_token_logits = logits.clone()

            next_token_id = logits.argmax(dim=-1).item()

            if next_token_id == tokenizer.eos_token_id:
                break

            current_emb = model.embedding(torch.tensor([next_token_id], device=DEVICE)).unsqueeze(0)

        # 如果循环提前结束，使用最后一个logits
        if answer_token_logits is None:
            answer_token_logits = logits

        # 获取答案位置的logits
        baseline_logit = answer_token_logits[BASELINE_TOKEN_ID].item()
        target_logit = answer_token_logits[TARGET_TOKEN_ID].item()
        baseline_prob = torch.softmax(answer_token_logits, dim=-1)[BASELINE_TOKEN_ID].item()
        target_prob = torch.softmax(answer_token_logits, dim=-1)[TARGET_TOKEN_ID].item()

        logit_diff = target_logit - baseline_logit

    return logit_diff, baseline_prob, target_prob


# ============ 8. GCG 迭代 ============
print("\n[7] 开始GCG迭代...")

current_text, current_answer, current_correct = evaluate(question, prefix_ids)
initial_logit_diff, init_baseline_prob, init_target_prob = compute_logits_diff(question, prefix_ids)

print(f"初始前缀 logits差值: {initial_logit_diff:.4f}")
print(f"初始生成: '{current_text[:100]}...'")
print(f"初始答案: '{current_answer}', 正确: {current_correct}")
print(f"固定token - BASELINE_TOKEN_ID={BASELINE_TOKEN_ID} ('{tokenizer.decode([BASELINE_TOKEN_ID])}'), TARGET_TOKEN_ID={TARGET_TOKEN_ID} ('{tokenizer.decode([TARGET_TOKEN_ID])}')")
print(f"Top1 token: '{tokenizer.decode([BASELINE_TOKEN_ID])}' (prob={init_baseline_prob:.6f}), Top2 token: '{tokenizer.decode([TARGET_TOKEN_ID])}' (prob={init_target_prob:.6f})")

best_prefix = prefix_ids.copy()
best_correct = current_correct
best_logit_diff = initial_logit_diff

scores_list = []

for iteration in range(N_ITERS):
    # 计算梯度
    mode = "both" if args.pass_gradient_through_latent else "kv_only"
    gradients, curr_len = compute_gradients(question, prefix_ids, mode=mode)

    # 计算梯度均值
    grad_mean = np.mean(np.abs(gradients))

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

        logit_diff, _, _ = compute_logits_diff(question, new_prefix)
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
    current_text, current_answer, current_correct = evaluate(question, prefix_ids)
    curr_logit_diff, curr_baseline_prob, curr_target_prob = compute_logits_diff(question, prefix_ids)

    print(f"Iter {iteration+1}: logit差值={curr_logit_diff:.4f} (top1='{tokenizer.decode([BASELINE_TOKEN_ID])}' prob={curr_baseline_prob:.6f}, "
          f"top2='{tokenizer.decode([TARGET_TOKEN_ID])}' prob={curr_target_prob:.6f}), 最佳差值={best_logit_diff:.4f}, "
          f"梯度均值={grad_mean:.6f}, 答案='{current_answer}', 正确={current_correct}")

    if best_logit_diff > 2 and current_correct == False:
        break


# ============ 9. 结果 ============
print("\n" + "=" * 60)
print("最终结果")
print("=" * 60)

final_prefix_str = tokenizer.decode(best_prefix)
print(f"对抗前缀: '{final_prefix_str}'")

# 无前缀
clean_text, clean_answer, clean_correct = evaluate(question, np.array([]))
clean_logit_diff, _, _ = compute_logits_diff(question, np.array([]))

# 有前缀
adv_text, adv_answer, adv_correct = evaluate(question, best_prefix)
adv_logit_diff, _, _ = compute_logits_diff(question, best_prefix)

print(f"\n无前缀:")
print(f"  生成: '{clean_text[:100]}...'")
print(f"  答案: '{clean_answer}', 正确: {clean_correct}")

print(f"\n有前缀 ('{final_prefix_str}'):")
print(f"  生成: '{adv_text[:100]}...'")
print(f"  答案: '{adv_answer}', 正确: {adv_correct}")

print(f"\n攻击效果: {'成功' if not adv_correct and clean_correct else '失败'}")

# 保存结果
result = {
    "question": question,
    "ground_truth": answer_str,
    "prefix": final_prefix_str,
    "prefix_token_ids": [int(x) for x in best_prefix],
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
