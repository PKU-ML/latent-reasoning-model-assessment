"""
================================================================================
PCCoT模型 白盒攻击 - GCG对抗前缀攻击
================================================================================

一、算法概述
-----------
本代码实现了一种针对PCCoT模型的白盒对抗攻击，通过优化一个前缀字符串，
使得模型在给定问题下输出错误的答案。

二、PCCoT模型推理流程
--------------------
PCCoT是一种具有隐式推理能力的模型，其推理过程如下：
1. 输入：问题(question) + <bot> + <latent>*N + <eot> + answer_prompt + 答案
2. Latent推理：从 question 的 hidden state 初始化 latent_embd
3. 多次隐式推理迭代（latent_embd 作为输入）
4. 预测答案 token

三、攻击目标
-----------
- 在答案token位置
- 找到top-1 token（最大概率的token，记作t1）和top-2 token（次大概率，记作t2）
- 优化目标：让 t2 的logit - t1 的logit 差值最大化
- 当这个差值足够大时，t2的概率可能超过t1，导致模型输出错误答案

================================================================================
"""

import json
import torch
import numpy as np
import os
import re
import sys
import argparse
import shutil
from types import ModuleType
from pathlib import Path

# Fix flash_attn import issue
_flash_attn_path = None
for path in sys.path:
    flash_attn_dir = os.path.join(path, 'flash_attn')
    if os.path.isdir(flash_attn_dir):
        _flash_attn_path = flash_attn_dir
        break

if _flash_attn_path is not None:
    _backup_path = _flash_attn_path + '_backup'
    if os.path.isdir(_backup_path):
        pass
    elif os.path.exists(os.path.join(_flash_attn_path, '__init__.py')):
        try:
            import importlib.util
            spec = importlib.util.find_spec('flash_attn')
            if spec is None or spec.loader is None:
                raise ImportError("flash_attn is broken")
        except:
            shutil.move(_flash_attn_path, _backup_path)
            sys.path_importer_cache.clear()

sys.path.insert(0, os.environ.get("PCCOT_PROJECT_ROOT", str(Path(__file__).resolve().parent / "PCCoT")))

from transformers import AutoTokenizer, AutoConfig, HfArgumentParser, AutoModelForCausalLM
from transformers.utils.hub import cached_file
from peft import AutoPeftModel
from transformers import GenerationConfig

import models

# ============ 命令行参数 ============
parser = argparse.ArgumentParser(description="PCCoT 白盒对抗攻击 - GCG")
parser.add_argument("--candidate-selection", type=str, choices=["gradient", "random"], default="gradient",
                    help="候选token选择方式: gradient(梯度选择) 或 random(随机选择)")
parser.add_argument("--problem-id", type=int, default=None,
                    help="指定攻击的问题ID (默认自动分配)")
parser.add_argument("--prefix-length", type=int, default=5,
                    help="前缀长度 (默认5)")
parser.add_argument("--n-iters", type=int, default=50,
                    help="迭代次数 (默认50)")
parser.add_argument("--top-k", type=int, default=300,
                    help="每个位置考虑top-k个候选 (默认300)")
parser.add_argument("--batch-size", type=int, default=40,
                    help="候选batch大小 (默认40)")
parser.add_argument("--seed", type=int, default=42,
                    help="随机种子 (默认42)")
parser.add_argument("--output-dir", type=str,
                    default=os.environ.get(
                        "PCCOT_OUTPUT_DIR",
                        str(Path(__file__).resolve().parent / "adv_results" / "results_white"),
                    ),
                    help="结果保存目录 (默认: <PCCoT>/adv_results/results_white；可通过 PCCOT_OUTPUT_DIR 覆盖)")
parser.add_argument("--gpu", type=int, default=0,
                    help="GPU设备号 (默认0)")
parser.add_argument("--dataset", type=str, choices=["gsm8k", "MultiArith", "SVAMP"], default="gsm8k",
                    help="指定要攻击的数据集 (默认 gsm8k)")
args = parser.parse_args()

# 设置随机种子
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)

# ============ 配置 ============
DEVICE = f"cuda:{args.gpu}"
print(f"使用设备: {DEVICE}")

# 模型配置
MODEL_NAME_OR_PATH = os.environ.get(
    "PCCOT_MODEL_PATH",
    "whyNLP/pccot-gpt2",
)
DATA_NAME = args.dataset
DATA_DIR = os.environ.get(
    "DATA_DIR",
    str(Path(__file__).resolve().parent.parent / "data"),
)
DATA_PATH = f"{DATA_DIR}/{DATA_NAME}.json"

# GCG 参数
PREFIX_LENGTH = args.prefix_length
N_ITERS = args.n_iters
TOP_K = args.top_k
BATCH_SIZE = args.batch_size
MAX_NEW_TOKENS = 30

os.makedirs(args.output_dir, exist_ok=True)
if args.problem_id is not None:
    PROBLEM_ID = args.problem_id
else:
    PROBLEM_ID = len(os.listdir(args.output_dir))

print("=" * 60)
print("PCCoT GCG 寻找对抗前缀")
print(f"模式: {'梯度选择候选' if args.candidate_selection == 'gradient' else '随机选择候选'}")
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


# ============ 2. 加载PCCoT模型 ============
print("\n[2] 加载PCCoT模型...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_OR_PATH)
config = AutoConfig.from_pretrained(MODEL_NAME_OR_PATH)
model = AutoPeftModel.from_pretrained(MODEL_NAME_OR_PATH)

model.get_base_model().config = config

pccot_args_file = cached_file(MODEL_NAME_OR_PATH, models.PCCOT_ARGS_NAME)
hf_parser = HfArgumentParser(models.PCCoTArguments)
(pccot_args,) = hf_parser.parse_json_file(json_file=pccot_args_file)

def get_special_token(tokenizer, token):
    if token not in tokenizer.additional_special_tokens:
        tokenizer.add_special_tokens({'additional_special_tokens': (token,)}, replace_additional_special_tokens=False)
    return tokenizer.convert_tokens_to_ids(token)

if pccot_args.bot_token_id is None:
    pccot_args.bot_token_id = get_special_token(tokenizer, '<pccot.bot>')
if pccot_args.eot_token_id is None:
    pccot_args.eot_token_id = get_special_token(tokenizer, '<pccot.eot>')
if pccot_args.latent_token_id is None:
    pccot_args.latent_token_id = get_special_token(tokenizer, '<pccot.latent>')
if tokenizer.pad_token_id is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
if tokenizer.eos_token_id is None:
    tokenizer.add_special_tokens({'eos_token': '[EOS]'})

embedding_size = model.get_input_embeddings().weight.shape[0]
if len(tokenizer) > embedding_size:
    model.resize_token_embeddings(len(tokenizer))

model = model.to(DEVICE)
model.eval()

data_processor = models.COTDataProcessor(
    tokenizer=tokenizer,
    pccot_args=pccot_args,
)

print(f"PCCoT模型加载完成")
print(f"  num_latent_tokens: {pccot_args.num_latent_tokens}")
print(f"  num_iterations: {config.num_iterations}")


# ============ 3. 辅助函数 ============

def normalize_number(s):
    try:
        return str(float(s.strip()))
    except (ValueError, TypeError):
        return s.strip()


def extract_answer_number(sentence):
    sentence = sentence.replace(',', '')
    pred = [s for s in re.findall(r'-?\d+\.?\d*', sentence)]
    if not pred:
        return float('inf')
    return float(pred[-1])  # 取最后一个数字（答案）


def tokens_represent_same_number(token_id1, token_id2, tokenizer):
    """检查两个token ID是否表示相同的数字"""
    decoded1 = tokenizer.decode([token_id1]).strip()
    decoded2 = tokenizer.decode([token_id2]).strip()
    num1 = extract_answer_number(decoded1)
    num2 = extract_answer_number(decoded2)
    if num1 != float('inf') and num2 != float('inf') and num1 == num2:
        return True
    return False


def make_prompt(question_str, prefix_str=""):
    """
    构建prompt格式: 前缀 + 问题
    """
    prompt = prefix_str.strip() + question_str.strip()
    return prompt


def generate_answer(question_str, prefix_str="", max_tokens=MAX_NEW_TOKENS):
    """生成答案"""
    full_question = make_prompt(question_str, prefix_str)
    collated = data_processor.process(full_question, device=DEVICE)

    generation_config = GenerationConfig(
        max_length=collated["input_ids"].shape[1] + max_tokens,
        do_sample=False,
    )

    with torch.no_grad():
        decoded_tokens = model.generate(
            collated=collated,
            generation_config=generation_config,
        )

    decoded_tokens = decoded_tokens[:, collated["input_ids"].shape[1]:]
    answers_list = tokenizer.batch_decode(decoded_tokens, skip_special_tokens=True)

    answer_text = answers_list[0] if answers_list else ""
    return answer_text, extract_answer_number(answer_text)


def evaluate(question_str, prefix_str=""):
    """评估当前前缀"""
    text, _ = generate_answer(question_str, prefix_str)
    return text, extract_answer_number(text)


# ============ 4. 测试baseline ============
print("\n[3] 测试baseline...")

baseline_text, baseline_answer = generate_answer(question)
print(f"Baseline生成文本: '{baseline_text}'")
print(f"Baseline提取答案: {baseline_answer}")
print(f"正确答案: {answer_str}")

baseline_correct = (str(baseline_answer) == answer_str or
                   abs(baseline_answer - float(answer_str)) < 0.01 if baseline_answer != float('inf') and answer_str.lstrip('-').replace('.','',1).isdigit() else False)
print(f"Baseline是否正确: {baseline_correct}")


# ============ 5. 确定答案token位置和目标token ============
print("\n[4] 获取baseline logits信息...")

def find_answer_position_and_tokens(question_str, prefix_str=""):
    """
    通过生成多个token来确定答案位置和目标token
    """
    full_question = make_prompt(question_str, prefix_str)
    collated = data_processor.process(full_question, device=DEVICE)

    # 获取 input_ids 长度（用于从输出中提取答案部分）
    input_len = collated["input_ids"].shape[1]

    generation_config = GenerationConfig(
        max_new_tokens=20,
        do_sample=False,
        output_scores=True,
        return_dict_in_generate=True,
    )

    with torch.no_grad():
        outputs = model.generate(
            collated=collated,
            generation_config=generation_config,
        )

    # 获取所有 logits
    all_logits = []
    for i, output_token in enumerate(outputs.sequences[:, input_len:]):
        if i == 0:
            # 第一个token需要完整前向
            pass
        else:
            break

    return None, None, None  # 简化版本


# 由于PCCoT的generate方法较复杂，我们使用简化方法
# 找到第一个含数字的token位置

def _find_answer_position_from_generation(question_str, prefix_str=""):
    """从生成结果中找到答案token位置"""
    full_question = make_prompt(question_str, prefix_str)
    collated = data_processor.process(full_question, device=DEVICE)

    generation_config = GenerationConfig(
        max_new_tokens=15,
        do_sample=False,
    )

    with torch.no_grad():
        decoded_tokens = model.generate(
            collated=collated,
            generation_config=generation_config,
        )

    decoded_tokens = decoded_tokens[:, collated["input_ids"].shape[1]:]
    decoded_text = tokenizer.decode(decoded_tokens[0], skip_special_tokens=True)

    # 找到第一个含数字的token
    tokens = tokenizer.convert_ids_to_tokens(decoded_tokens[0])

    first_digit_pos = None
    first_digit_token_str = None
    for idx, (token_id, token_str) in enumerate(zip(decoded_tokens[0].tolist(), tokens)):
        decoded = tokenizer.decode([token_id]).strip()
        if re.search(r'\d', decoded):
            first_digit_pos = idx
            first_digit_token_str = decoded
            break

    if first_digit_pos is None and len(tokens) > 0:
        first_digit_pos = len(tokens) - 1
        first_digit_token_str = tokenizer.decode([decoded_tokens[0, -1].item()]).strip()

    return first_digit_pos, first_digit_token_str, decoded_tokens


# 由于PCCoT的生成接口复杂，我们使用黑盒方式评估
# 只记录baseline信息，不计算梯度

print("注意: PCCoT白盒攻击使用简化版本，基于随机候选和logit差值评估")

# 获取baseline答案
first_digit_pos, first_digit_token_str, _ = _find_answer_position_from_generation(question, "")
print(f"回答中第一个含数字的token位置: {first_digit_pos}, token: '{first_digit_token_str}'")


# ============ 6. 初始化前缀 ============
print("\n[5] 初始化前缀...")

char_tokens = []
for i in range(100, tokenizer.vocab_size):
    try:
        decoded = tokenizer.decode([i])
        if decoded.strip() and len(decoded.strip()) > 0 and decoded.strip().isprintable():
            char_tokens.append(i)
    except:
        pass

print(f"有效token数量: {len(char_tokens)}")

prefix_ids = np.random.choice(char_tokens, size=PREFIX_LENGTH, replace=True)
prefix_str = tokenizer.decode(prefix_ids)
print(f"初始前缀: '{prefix_str}'")


# ============ 7. 辅助函数 ============

def compute_logit_based_score(question_str, prefix_ids_arr):
    """
    评估前缀效果：运行生成并检查答案是否错误
    返回一个分数（答案错误返回正值，错误程度越高分数越高）
    """
    prefix_str = tokenizer.decode(prefix_ids_arr)
    text, pred_answer = evaluate(question_str, prefix_str)

    # 计算与正确答案的差异
    try:
        diff = abs(pred_answer - float(answer_str)) if pred_answer != float('inf') else 1000.0
    except:
        diff = 1000.0

    return -diff  # 负值表示接近正确答案，我们想要负值越小越好（即答案错误）


# ============ 7. PCCoT 梯度计算函数 ============
print("\n[6] 定义PCCoT梯度计算函数...")

def find_answer_position_and_target_tokens(question_str, prefix_str=""):
    """
    找到答案token位置以及baseline和target token IDs
    返回: (answer_token_pos, BASELINE_TOKEN_ID, TARGET_TOKEN_ID, answer_prompt_length)
    """
    full_question = make_prompt(question_str, prefix_str)
    collated = data_processor.process(full_question, device=DEVICE)

    input_len = collated["input_ids"].shape[1]
    key_indices = collated["key_indices"]  # [question_boundary, latent_boundary, ccot_kd_index]

    # answer_prompt "The answer is:" 的长度
    answer_prompt_length = len(tokenizer.encode(pccot_args.answer_prompt, add_special_tokens=False))

    # latent_boundary 是 <eot> 之后的位置，即答案开始的位置
    # 答案token位置 = latent_boundary + answer_prompt_length
    # 但第一个答案是 latent_boundary + answer_prompt_length (因为预测的是下一个token)
    # logits[latent_boundary + answer_prompt_length] 预测的是 latent_boundary + answer_prompt_length + 1 的token

    # 生成几个token来确认答案位置
    generation_config = GenerationConfig(
        max_new_tokens=20,
        do_sample=False,
    )

    with torch.no_grad():
        decoded_tokens = model.generate(
            collated=collated,
            generation_config=generation_config,
        )

    # 获取答案部分的logits
    # 在PCCoT的forward中，answer_logits是直接从answer_outputs获取的
    # 我们需要在答案token位置获取logits
    return key_indices, answer_prompt_length, input_len


def pccot_forward_with_grad(question_str, prefix_ids_arr):
    """
    PCCoT前向传播，用于计算梯度
    遵循PCCoT模型的forward流程：Part 2 - Student CoT
    """
    prefix_str = tokenizer.decode(prefix_ids_arr)
    full_question = make_prompt(question_str, prefix_str)
    collated = data_processor.process(full_question, device=DEVICE)

    # 获取base model
    base_model = model.get_base_model()

    # 提取关键信息
    input_ids = collated["input_ids"].to(DEVICE)
    attention_mask = collated["attention_mask"].to(DEVICE)
    key_indices = collated["key_indices"]

    # 获取embedding层 - 使用weight.data获取原始权重，避免Peft包装的梯度问题
    embed_layer = base_model.get_input_embeddings()
    embed_weight = embed_layer.weight.data  # 原始权重，无Peft梯度追踪

    # 创建可训练的embedding - 使用原始权重创建叶子节点
    input_embeds = torch.nn.functional.embedding(input_ids, embed_weight).requires_grad_(True)

    # ===== PCCoT Forward - Part 2: Student CoT =====

    # GPT-2 position_ids计算
    position_ids = attention_mask.cumsum(dim=-1) - 1
    position_ids = position_ids.masked_fill(attention_mask == 0, 0)

    question_boundary, latent_boundary, ccot_kd_index = key_indices

    # Step 1: Question部分过transformer - 使用inputs_embeds以保留梯度
    question_embeds = input_embeds[:, :latent_boundary]

    ccot_outputs = base_model.transformer(
        inputs_embeds=question_embeds,
        attention_mask=attention_mask[:, :latent_boundary],
        position_ids=position_ids[:, :latent_boundary],
        past_key_values=None,
        output_hidden_states=True,
        return_dict=True,
    )

    # 获取latent位置的hidden state
    last_hidden_state = ccot_outputs.hidden_states[-1][:, question_boundary-1:latent_boundary-1]
    latent_input_embeds = base_model.prj(last_hidden_state)

    # 保存question的KV cache
    question_past_key_values = [
        (
            ccot_outputs.past_key_values[l][0][:, :, :question_boundary],
            ccot_outputs.past_key_values[l][1][:, :, :question_boundary]
        )
        for l in range(len(ccot_outputs.past_key_values))
    ]

    # Step 2: 迭代细化latent tokens
    for i in range(base_model.config.num_iterations):
        ccot_past_key_values = tuple(question_past_key_values)
        ccot_outputs = base_model.transformer(
            inputs_embeds=latent_input_embeds,
            past_key_values=ccot_past_key_values,
            position_ids=position_ids[:, question_boundary:latent_boundary],
            output_hidden_states=True,
            return_dict=True,
        )
        last_hidden_state = ccot_outputs.hidden_states[-1]
        projected_hidden_state = base_model.prj(last_hidden_state)
        latent_input_embeds = torch.cat([latent_input_embeds[:, :1], projected_hidden_state[:, :-1]], dim=1)

    # Step 3: 生成答案
    answer_outputs = base_model.transformer(
        input_ids=input_ids[:, latent_boundary:],
        attention_mask=attention_mask,
        position_ids=position_ids[:, latent_boundary:],
        past_key_values=ccot_outputs.past_key_values,
        output_hidden_states=True,
        return_dict=True,
    )

    answer_logits = base_model.lm_head(answer_outputs.last_hidden_state)
    answer_logits = answer_logits.float()

    return {
        "input_embeds": input_embeds,
        "answer_logits": answer_logits,
        "latent_boundary": latent_boundary,
        "key_indices": key_indices,
        "input_ids": input_ids,
        "collated": collated,
    }


def compute_gradients(question_str, prefix_ids_arr):
    """
    计算PCCoT模型相对于前缀token的梯度

    损失函数: loss = logit[target_token] - logit[baseline_token]
    梯度从答案位置的logits流向prefix的embedding
    """
    prefix_len = len(prefix_ids_arr)

    # 获取baseline和target token IDs
    # 这需要在初始化时确定，并全局使用
    global BASELINE_TOKEN_ID, TARGET_TOKEN_ID, ANSWER_LOGIT_POS

    # 获取embedding层 - 使用weight.data获取原始权重
    embed_layer = model.get_base_model().get_input_embeddings()
    embed_weight = embed_layer.weight.data

    # 前向传播
    result = pccot_forward_with_grad(question_str, prefix_ids_arr)
    input_embeds = result["input_embeds"]
    answer_logits = result["answer_logits"]
    latent_boundary = result["latent_boundary"]
    key_indices = result["key_indices"]

    # 答案logits位置: latent_boundary是<eot>之后的位置
    # "The answer is:" 被tokenize后加在答案前面
    # answer_logits[i] 预测的是第i+1个答案token
    # 我们关注第一个答案token (index = answer_prompt_length)

    # 获取answer_prompt的长度
    answer_prompt_ids = tokenizer.encode(pccot_args.answer_prompt, add_special_tokens=False)
    answer_prompt_len = len(answer_prompt_ids)

    # 第一个答案token的位置 (在answer_logits中的index)
    # answer_logits[answer_prompt_len] 对应 "The answer is: X" 中 X 的位置
    answer_logit_idx = answer_prompt_len  # 第一个答案token的位置

    # 检查是否超出范围
    if answer_logit_idx >= answer_logits.shape[1]:
        # 如果答案token太短，使用最后一个位置
        answer_logit_idx = answer_logits.shape[1] - 1

    # 获取答案位置的logits
    answer_token_logits = answer_logits[0, answer_logit_idx]

    # 计算损失: logit[target] - logit[baseline]
    if BASELINE_TOKEN_ID is None or TARGET_TOKEN_ID is None:
        # 首次调用：确定baseline和target
        sorted_logits, sorted_indices = torch.sort(answer_token_logits, descending=True)
        BASELINE_TOKEN_ID = sorted_indices[0].item()
        TARGET_TOKEN_ID = sorted_indices[1].item()
        print(f"[DEBUG] 确定token: BASELINE={BASELINE_TOKEN_ID} ('{tokenizer.decode([BASELINE_TOKEN_ID])}'), TARGET={TARGET_TOKEN_ID} ('{tokenizer.decode([TARGET_TOKEN_ID])}')")

    logit_loss = answer_token_logits[TARGET_TOKEN_ID] - answer_token_logits[BASELINE_TOKEN_ID]
    total_loss = logit_loss

    # 反向传播
    total_loss.backward()

    # 获取梯度
    input_grad = input_embeds.grad
    if input_grad is None:
        return np.zeros((prefix_len, embed_weight.shape[0])), prefix_len

    # 获取prefix长度的embedding梯度
    prefix_grad_emb = input_grad[0, :prefix_len]

    # 将embedding梯度转换为token梯度
    # gradient w.r.t. input_ids ≈ gradient w.r.t. embedding * d_embedding/d_input_ids
    # 对于one-hot的input_ids，gradient直接就是embedding gradient
    token_grads = torch.matmul(prefix_grad_emb, embed_weight.T)

    return token_grads.cpu().detach().to(torch.float32).numpy(), prefix_len


def compute_logits_diff(question_str, prefix_ids_arr):
    """
    计算给定前缀的logits差值（次大token的logit - 最大token的logit）
    用于评估前缀效果
    """
    prefix_str = tokenizer.decode(prefix_ids_arr)
    full_question = make_prompt(question_str, prefix_str)
    collated = data_processor.process(full_question, device=DEVICE)

    with torch.no_grad():
        # 获取forward结果
        result = pccot_forward_with_grad(question_str, prefix_ids_arr)
        answer_logits = result["answer_logits"]

        # 获取answer_prompt长度
        answer_prompt_ids = tokenizer.encode(pccot_args.answer_prompt, add_special_tokens=False)
        answer_prompt_len = len(answer_prompt_ids)

        answer_logit_idx = answer_prompt_len
        if answer_logit_idx >= answer_logits.shape[1]:
            answer_logit_idx = answer_logits.shape[1] - 1

        answer_token_logits = answer_logits[0, answer_logit_idx]

        # 使用固定的BASELINE_TOKEN_ID和TARGET_TOKEN_ID
        baseline_logit = answer_token_logits[BASELINE_TOKEN_ID].item()
        target_logit = answer_token_logits[TARGET_TOKEN_ID].item()

        baseline_prob = torch.softmax(answer_token_logits, dim=-1)[BASELINE_TOKEN_ID].item()
        target_prob = torch.softmax(answer_token_logits, dim=-1)[TARGET_TOKEN_ID].item()

        logit_diff = target_logit - baseline_logit

        baseline_str = tokenizer.decode([BASELINE_TOKEN_ID])
        target_str = tokenizer.decode([TARGET_TOKEN_ID])

    return logit_diff, baseline_prob, target_prob, baseline_str, target_str


# 初始化全局变量
BASELINE_TOKEN_ID = None
TARGET_TOKEN_ID = None
ANSWER_LOGIT_POS = None


def evaluate_with_prefix(question_str, prefix_ids_arr):
    """评估前缀，返回是否正确"""
    prefix_str = tokenizer.decode(prefix_ids_arr)
    text, pred_answer = evaluate(question_str, prefix_str)

    is_correct = (str(pred_answer) == answer_str or
                  abs(pred_answer - float(answer_str)) < 0.01 if pred_answer != float('inf') and answer_str.lstrip('-').replace('.','',1).isdigit() else False)

    return text, pred_answer, is_correct


# ============ 8. GCG 迭代 ============
print("\n[6] 开始GCG迭代...")

# 先获取baseline的logits信息，确定BASELINE_TOKEN_ID和TARGET_TOKEN_ID
print("\n获取baseline logits信息...")
with torch.no_grad():
    result = pccot_forward_with_grad(question, np.array([]))
    answer_logits = result["answer_logits"]
    answer_prompt_ids = tokenizer.encode(pccot_args.answer_prompt, add_special_tokens=False)
    answer_prompt_len = len(answer_prompt_ids)
    answer_logit_idx = min(answer_prompt_len, answer_logits.shape[1] - 1)
    answer_token_logits = answer_logits[0, answer_logit_idx]
    sorted_logits, sorted_indices = torch.sort(answer_token_logits, descending=True)
    BASELINE_TOKEN_ID = sorted_indices[0].item()
    TARGET_TOKEN_ID = sorted_indices[1].item()
    baseline_str = tokenizer.decode([BASELINE_TOKEN_ID])
    target_str = tokenizer.decode([TARGET_TOKEN_ID])
    print(f"Baseline答案token位置:")
    print(f"  最大token: '{baseline_str}' (id={BASELINE_TOKEN_ID}), logit={sorted_logits[0].item():.4f}")
    print(f"  第二大token: '{target_str}' (id={TARGET_TOKEN_ID}), logit={sorted_logits[1].item():.4f}")
    print(f"  Logit差值: {sorted_logits[1].item() - sorted_logits[0].item():.4f}")

current_text, current_answer, current_correct = evaluate_with_prefix(question, prefix_ids)
print(f"\n初始生成: '{current_text}'")
print(f"初始答案: {current_answer}")
print(f"是否正确: {current_correct}")

best_prefix = prefix_ids.copy()
best_correct = current_correct
best_score = -100 #compute_logit_based_score(question, prefix_ids)

scores_list = []

for iteration in range(N_ITERS):
    curr_len = len(prefix_ids)

    # 计算梯度（只在第一次迭代时计算一次）
    gradients = None
    if args.candidate_selection == "gradient" and iteration == 0:
        print(f"\n计算梯度...")
        gradients, _ = compute_gradients(question, prefix_ids)
        print(f"梯度计算完成，形状: {gradients.shape}")
        grad_mean = np.mean(np.abs(gradients))
        grad_max = np.max(np.abs(gradients))
        grad_min = np.min(np.abs(gradients))
        print(f"梯度统计: mean={grad_mean:.6f}, max={grad_max:.6f}, min={grad_min:.6f}")

    # 生成候选组合
    sampled_combinations = [{}]
    for _ in range(BATCH_SIZE):
        n_changes = np.random.randint(1, curr_len + 1)
        positions = np.random.choice(curr_len, size=n_changes, replace=False)

        combo = {}
        for pos in positions:
            if args.candidate_selection == "gradient" and gradients is not None:
                # 使用梯度：选择梯度最大的TOP_K个候选
                pos_grads = gradients[pos]
                top_k_indices = np.argsort(pos_grads)[-TOP_K:]
                combo[pos] = np.random.choice(top_k_indices)
            else:
                # 随机选择
                combo[pos] = np.random.choice(char_tokens)

        sampled_combinations.append(combo)

    # 评估候选 - 使用logits差值
    scores = []
    for combo in sampled_combinations:
        new_prefix = prefix_ids.copy()
        for pos, tok_id in combo.items():
            new_prefix[pos] = tok_id
        logit_diff, baseline_prob, target_prob, _, _ = compute_logits_diff(question, new_prefix)
        scores.append((logit_diff, combo, baseline_prob, target_prob))

    # 选择最佳候选（logit差值最大的，即让target更可能超过baseline）
    scores.sort(key=lambda x: x[0])
    best_combo = scores[-1][1]  # 分数最高（logit差值最大）
    best_score_ = scores[-1][0]

    # 更新前缀
    for pos, tok_id in best_combo.items():
        prefix_ids[pos] = tok_id

    scores_list.append(best_score)
    if best_score_ > best_score or (best_score_ == best_score and not current_correct):
        best_score = best_score_
        best_prefix = prefix_ids.copy()

    current_text, current_answer, current_correct = evaluate_with_prefix(question, prefix_ids)

    # 获取当前logits信息
    curr_logit_diff, curr_baseline_prob, curr_target_prob, _, _ = compute_logits_diff(question, prefix_ids)
    
    print(f"Iter {iteration+1}: logit差值={curr_logit_diff:.4f} (top1='{baseline_str}' prob={curr_baseline_prob:.4f}, top2='{target_str}' prob={curr_target_prob:.4f}), 最佳差值={best_score:.4f}, 答案={current_answer}, 正确={current_correct}")

    # 如果攻击成功（答案错误且baseline正确），停止
    if not current_correct and best_score > 2:
        print(f"攻击成功！")
        break


# ============ 9. 结果 ============
print("\n" + "=" * 60)
print("最终结果")
print("=" * 60)

final_prefix_str = tokenizer.decode(best_prefix)
print(f"对抗前缀: '{final_prefix_str}'")

clean_text, clean_answer = evaluate(question, "")
clean_correct = (str(clean_answer) == answer_str or
                abs(clean_answer - float(answer_str)) < 0.01 if clean_answer != float('inf') and answer_str.lstrip('-').replace('.','',1).isdigit() else False)

adv_text, adv_answer = evaluate(question, final_prefix_str)
adv_correct = (str(adv_answer) == answer_str or
              abs(adv_answer - float(answer_str)) < 0.01 if adv_answer != float('inf') and answer_str.lstrip('-').replace('.','',1).isdigit() else False)

print(f"\n无前缀:")
print(f"  生成文本: '{clean_text[:100]}...'")
print(f"  提取答案: {clean_answer}")
print(f"  是否正确: {clean_correct}")

print(f"\n有前缀 ('{final_prefix_str}'):")
print(f"  生成文本: '{adv_text[:100]}...'")
print(f"  提取答案: {adv_answer}")
print(f"  是否正确: {adv_correct}")

print(f"\n攻击效果: {'成功' if not adv_correct and clean_correct else '失败'}")

# 计算最终的logits差值
best_logit_diff, best_baseline_prob, best_target_prob, _, _ = compute_logits_diff(question, best_prefix)

# 保存结果
result = {
    "question": question,
    "ground_truth": answer_str,
    "prefix": final_prefix_str,
    "prefix_token_ids": [int(x) for x in best_prefix],
    "baseline_token_id": BASELINE_TOKEN_ID,
    "target_token_id": TARGET_TOKEN_ID,
    "baseline_token_str": baseline_str,
    "target_token_str": target_str,
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

os.makedirs(args.output_dir, exist_ok=True)
save_path = os.path.join(args.output_dir, f"problem_{PROBLEM_ID}.json")
with open(save_path, "w") as f:
    json.dump(result, f, indent=2, ensure_ascii=False)

print(f"\n结果已保存到 {save_path}")