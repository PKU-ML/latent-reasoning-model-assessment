"""
================================================================================
SIM-CoT (Coconut) 模型 GCG (Greedy Coordinate Gradient) 对抗前缀攻击
================================================================================

一、算法概述
------------
本代码实现了一种针对SIM-CoT(Coconut)模型的GCG对抗攻击，目标是通过优化一个
前缀字符串，使得模型在给定数学问题下输出错误的答案。

二、SIM-CoT模型推理流程
-----------------------
SIM-CoT是一种具有隐式推理能力的模型(Coconut)，其推理过程如下：
1. 输入：问题(question) + 前缀(prefix) + "\\n"
2. Encoder编码：prefix + question -> 获取hidden states
3. Latent推理：从最后一个token的hidden state初始化latent_embd
4. 多次隐式推理迭代（latent_embd作为输入）
5. 添加<|end-latent|> token
6. 自回归生成答案

三、攻击目标
------------
模型输出格式: "... ### 答案" (以###分隔，最后一部分是答案)

攻击目标：
- 在答案token位置（第一个含数字的token）
- 找到top-1 token（最大概率的token，记作t1）和top-2 token（次大概率，记作t2）
- 优化目标：让 t2 的logit - t1 的logit 差值最大化
- 当这个差值足够大时，t2的概率可能超过t1，导致模型输出错误答案

四、关键实现
------------
1. 使用CoconutGPT_Fixed的forward_embeds_for_gradient方法：
   - 支持梯度计算
   - 支持控制梯度是否流经latent向量

2. BASELINE_TOKEN_ID 和 TARGET_TOKEN_ID：
   - 在无前缀时，答案位置最大的token是baseline_token（正确答案是t1）
   - 次大的token是target_token（我们想让模型选择的t2）
   - 这两个token ID在整个攻击过程中保持固定

3. 损失函数：
   Loss = logit[target_token] - logit[baseline_token]

4. GCG优化：
   - 对前缀的每个位置，计算梯度
   - 选取梯度最大的TOP_K个候选token
   - 随机采样1到前缀长度个位置进行替换
   - 评估候选前缀的logit差值
   - 选择logit差值最大的前缀

================================================================================
"""

import json
import torch
import numpy as np
import os
import re
import argparse
from dataclasses import dataclass
from pathlib import Path

from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers

# ============ 命令行参数 ============
parser = argparse.ArgumentParser(description="SIM-CoT GCG 对抗前缀攻击")
parser.add_argument("--candidate-selection", type=str, choices=["gradient", "random"], default="gradient",
                    help="候选token选择方式: gradient(梯度选择) 或 random(随机选择)")
parser.add_argument("--problem-id", type=int, default=None,
                    help="指定攻击的问题ID (默认使用第0个)")
parser.add_argument("--prefix-length", type=int, default=5,
                    help="前缀长度 (默认5)")
parser.add_argument("--n-iters", type=int, default=50,
                    help="迭代次数 (默认50)")
parser.add_argument("--top-k", type=int, default=300,
                    help="每个位置考虑top-k个候选 (默认300)")
parser.add_argument("--batch-size", type=int, default=40,
                    help="候选batch大小 (默认40)")
parser.add_argument("--latent-gradient", type=str, choices=['true', 'false'], default='true',
                    help="梯度是否流经latent向量: true 或 false (默认true)")
parser.add_argument("--seed", type=int, default=42,
                    help="随机种子 (默认42)")
parser.add_argument("--dataset", type=str, choices=["gsm8k", "MultiArith", "SVAMP"], default="gsm8k",
                    help="指定要攻击的数据集 (默认 gsm8k)")
parser.add_argument("--output-dir", type=str,
                    default=os.environ.get(
                        "SIMCOT_OUTPUT_DIR",
                        str(Path(__file__).resolve().parent / "gcg_results" / "results_white"),
                    ),
                    help="结果保存目录 (默认: <SIM-CoT>/gcg_results/results_white；可通过 SIMCOT_OUTPUT_DIR 覆盖)")
args = parser.parse_args()
args.pass_gradient_through_latent = args.latent_gradient == 'true'

# 设置随机种子
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)

# ============ 配置 ============
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用设备: {DEVICE}")

# 模型配置
MODEL_ID = os.environ.get("SIMCOT_MODEL_ID", "gpt2")
CKPT_DIR = os.environ.get(
    "SIMCOT_CKPT_DIR",
    str(Path(__file__).resolve().parent / "SIM-CoT" / "Coconut" / "ckpts" / "SIM_COT-GPT2-Coconut" / "checkpoint_28"),
)
if not Path(CKPT_DIR).exists():
    print(f"[警告] SIMCOT_CKPT_DIR={CKPT_DIR} 不存在，请通过环境变量指向实际的 checkpoint 目录")
MAX_NEW_TOKENS = 64

# GCG 参数
PREFIX_LENGTH = args.prefix_length
N_ITERS = args.n_iters
TOP_K = args.top_k
BATCH_SIZE = args.batch_size

# 数据路径
DATA_DIR = os.environ.get(
    "DATA_DIR",
    str(Path(__file__).resolve().parent.parent / "data"),
)
DATA_NAME = args.dataset
DATA_PATH = f"{DATA_DIR}/{DATA_NAME}.json"

# latent tokens配置
N_LATENT_TOKENS = 10

os.makedirs(args.output_dir, exist_ok=True)
if args.problem_id is not None:
    PROBLEM_ID = args.problem_id
else:
    PROBLEM_ID = len(os.listdir(args.output_dir))

print("=" * 60)
print(f"SIM-CoT GCG 寻找对抗前缀 - 数据集: {DATA_NAME}")
print(f"模式: {'梯度选择候选' if args.candidate_selection == 'gradient' else '随机选择候选'}")
print(f"梯度流经latent: {args.pass_gradient_through_latent}")
print("=" * 60)


# ============ 1. 加载数据 ============
print(f"\n[1] 加载{DATA_NAME}数据...")

def load_local_dataset(data_path, dataset_name):
    """加载数据集，根据数据集名称使用不同的答案提取方式"""
    questions = []
    answers = []
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    for item in data:
        questions.append(item['question'])
        answer_text = item['answer'].replace(',', '')
        # 根据数据集使用不同的答案提取方式
        # gsm8k 格式: "... ### Answer" 或 "... #### Answer"
        # MultiArith/SVAMP: 直接是数字
        if dataset_name == "gsm8k":
            # GSM8K 答案通常在 "#### " 后面
            match = re.search(r'####\s*([\d\.\-]+)', answer_text)
            if match:
                answer_text = match.group(1)
            else:
                # 备选：取最后一个数字
                nums = re.findall(r'-?\d+\.?\d*', answer_text)
                if nums:
                    answer_text = nums[-1]
        else:
            # 其他数据集直接提取数字
            nums = re.findall(r'-?\d+\.?\d*', answer_text)
            if nums:
                answer_text = nums[-1]
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


# ============ 2. 配置和加载模型 ============
print("\n[2] 加载SIM-CoT模型...")

@dataclass
class Config:
    model_id: str = "gpt2"
    c_thought: int = 2
    max_latent_stage: int = 5
    training_method: str = "full"
    bf16: bool = False


class CoconutGPT_Fixed(torch.nn.Module):
    """修复版CoconutGPT，兼容transformers 4.x"""

    def __init__(self, base_causallm, expainable_llm, tokenizer, latent_token_id,
                 start_latent_id, end_latent_id, eos_token_id, step_start_id,
                 c_thought, configs):
        super().__init__()
        self.gen_forward_cnt = 0
        self.base_causallm = base_causallm
        self.expainable_llm = expainable_llm
        self.tokenizer = tokenizer
        self.latent_token_id = latent_token_id
        self.eos_token_id = eos_token_id
        self.start_latent_id = start_latent_id
        self.end_latent_id = end_latent_id
        self.step_start_id = step_start_id
        self.c_thought = c_thought
        self.config = configs

        if isinstance(self.base_causallm, transformers.models.gpt2.modeling_gpt2.GPT2LMHeadModel):
            self.embedding = self.base_causallm.transformer.get_input_embeddings()
        else:
            self.embedding = self.base_causallm.get_input_embeddings()

    def forward_embeds_for_gradient(self, input_ids, attention_mask, n_latent, target_position=-1,
                                     compute_gradient=False, target_token_id=None, baseline_token_id=None,
                                     pass_gradient_through_latent=True):
        """
        CODI 风格的 forward 方法，支持梯度计算。

        流程：
        1. 用 prefix + question 编码，获取初始 hidden states 和 KV cache
        2. 从最后一个 token 的 hidden state 获取 latent_embd
        3. n_latent 次 latent 推理迭代（latent_embd 作为输入，KV cache 传递）
        4. 添加 <|end-latent|> token
        5. 自回归生成，获取多个位置的 logits（用于确定答案位置）

        参数：
        - input_ids: [batch, seq_len] - 输入 token IDs（只包含 prefix + question）
        - attention_mask: [batch, seq_len]
        - n_latent: latent 推理迭代次数
        - target_position: 要获取 logits 的位置 (默认-1，即最后一个生成位置)
        - compute_gradient: 是否计算梯度
        - target_token_id: 目标 token id
        - baseline_token_id: 基线 token id
        - pass_gradient_through_latent: 控制梯度是否流经 latent 向量

        返回：
        - 如果compute_gradient=False: all_logits [batch, gen_len, vocab_size]
        - 如果compute_gradient=True: (all_logits, gradients) tuple
        """
        # 创建可训练的 input_embeds
        input_embeds = self.embedding(input_ids).clone().requires_grad_(True)

        # ===== 步骤1: 编码 prefix，获取初始 hidden states 和 KV =====
        outputs = self.base_causallm(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=True)
        kv_cache = outputs.past_key_values

        # 从 prefix 最后一个 token 的 hidden state 获取 initial latent_embd
        latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)  # [batch, 1, hidden]

        # kv_only 模式：在 latent 推理开始前 detach
        if not pass_gradient_through_latent:
            latent_embd = latent_embd.detach()

        # ===== 步骤2: n_latent 次 latent 推理迭代 =====
        for _ in range(n_latent):
            outputs = self.base_causallm(
                inputs_embeds=latent_embd,
                attention_mask=None,
                past_key_values=kv_cache,
                output_hidden_states=True,
                use_cache=True)
            kv_cache = outputs.past_key_values
            latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)

            if not pass_gradient_through_latent:
                latent_embd = latent_embd.detach()

        # ===== 步骤3: 添加 <|end-latent|> 并进行最终前向传播 =====
        end_latent_emb = self.embedding(torch.tensor([self.end_latent_id], device=input_ids.device))
        end_latent_emb = end_latent_emb.unsqueeze(1)

        '''outputs = self.base_causallm(
            inputs_embeds=end_latent_emb,
            attention_mask=None,
            past_key_values=kv_cache,
            output_hidden_states=True,
            use_cache=True)

        logits = outputs.logits'''

        # ===== 步骤4: 自回归生成多个 token，记录每个位置的 logits =====
        #all_logits = [logits]
        all_logits = []
        current_emb = end_latent_emb

        for _ in range(10):
            outputs = self.base_causallm(
                inputs_embeds=current_emb,
                attention_mask=None,
                past_key_values=kv_cache,
                output_hidden_states=True,
                use_cache=True)
            kv_cache = outputs.past_key_values
            logits = outputs.logits
            all_logits.append(logits)

            next_token_id = logits.argmax(dim=-1)
            current_emb = self.embedding(next_token_id.squeeze(1)).unsqueeze(1)

        all_logits = torch.cat(all_logits, dim=1)

        if compute_gradient:
            if target_position == -1:
                target_position = all_logits.shape[1] - 1
            loss = all_logits[0, target_position, target_token_id] - all_logits[0, target_position, baseline_token_id]

            grads = torch.autograd.grad(loss, input_embeds, create_graph=False, allow_unused=True)
            if grads is None or grads[0] is None:
                grad_wrt_tokens = torch.zeros(input_embeds.shape[1], self.embedding.weight.shape[0],
                                              device=input_embeds.device, dtype=input_embeds.dtype)
                return all_logits, grad_wrt_tokens
            grad_wrt_tokens = torch.matmul(grads[0], self.embedding.weight.T)
            return all_logits, grad_wrt_tokens

        return all_logits

    def forward(self, input_ids, attention_mask, labels, position_ids, **kwargs):
        inputs_embeds = self.embedding(input_ids)

        outputs = self.base_causallm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_hidden_states=True, use_cache=True)
        kv_cache = outputs.past_key_values
        latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)

        for _ in range(N_LATENT_TOKENS):
            outputs = self.base_causallm(
                inputs_embeds=latent_embd,
                attention_mask=None,
                past_key_values=kv_cache,
                output_hidden_states=True, use_cache=True)
            kv_cache = outputs.past_key_values
            latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)

        end_latent_emb = self.embedding(torch.tensor([self.end_latent_id], device=input_ids.device)).unsqueeze(1)

        outputs = self.base_causallm(
            inputs_embeds=end_latent_emb,
            attention_mask=None,
            past_key_values=kv_cache,
            output_hidden_states=True, use_cache=True)

        self.gen_forward_cnt = N_LATENT_TOKENS + 2
        return type('Outputs', (), {'loss': None, 'inputs_embeds': end_latent_emb, 'logits': outputs.logits})()

    def generate_clean(self, input_ids, attention_mask, max_new_tokens=16):
        """CODI 风格的生成，不保存中间状态"""
        self.gen_forward_cnt = 0
        assert input_ids.shape[0] == 1, "only support batch_size == 1"

        tokens = input_ids[0].detach().tolist()
        inputs_embeds = self.embedding(input_ids)

        outputs = self.base_causallm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True, use_cache=True)
        kv_cache = outputs.past_key_values
        latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)

        for _ in range(N_LATENT_TOKENS):
            outputs = self.base_causallm(
                inputs_embeds=latent_embd,
                attention_mask=None,
                past_key_values=kv_cache,
                output_hidden_states=True, use_cache=True)
            kv_cache = outputs.past_key_values
            latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)

        end_latent_emb = self.embedding(torch.tensor([self.end_latent_id], device=input_ids.device)).unsqueeze(1)

        outputs = self.base_causallm(
            inputs_embeds=end_latent_emb,
            attention_mask=None,
            past_key_values=kv_cache,
            output_hidden_states=True, use_cache=True)

        self.gen_forward_cnt = N_LATENT_TOKENS + 2
        logits = outputs.logits
        kv_cache = outputs.past_key_values

        next_token = torch.argmax(logits[0, -1]).item()
        tokens.append(next_token)
        new_token_embed = self.embedding(torch.tensor([next_token], device=input_ids.device)).unsqueeze(1)

        for _ in range(max_new_tokens - 1):
            outputs = self.base_causallm(
                inputs_embeds=new_token_embed,
                past_key_values=kv_cache,
                use_cache=True)
            kv_cache = outputs.past_key_values
            self.gen_forward_cnt += 1
            next_token = torch.argmax(outputs.logits[0, -1]).item()
            if next_token == self.eos_token_id:
                break
            tokens.append(next_token)
            new_token_embed = self.embedding(torch.tensor([next_token], device=input_ids.device)).unsqueeze(1)

        return torch.tensor(tokens).view(1, -1)


def load_coconut_model(checkpoint_path, model_id="gpt2", device="cuda"):
    print(f"Loading checkpoint from {checkpoint_path}...")
    saved_weights = torch.load(checkpoint_path, map_location="cpu")

    print(f"Loading tokenizer from {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    tokenizer.add_tokens("<|start-latent|>")
    tokenizer.add_tokens("<|end-latent|>")
    tokenizer.add_tokens("<|latent|>")

    latent_id = tokenizer.convert_tokens_to_ids("<|latent|>")
    start_id = tokenizer.convert_tokens_to_ids("<|start-latent|>")
    end_id = tokenizer.convert_tokens_to_ids("<|end-latent|>")

    print(f"Loading base_causallm from {model_id}...")
    base_causallm = AutoModelForCausalLM.from_pretrained(model_id)
    expainable_llm = AutoModelForCausalLM.from_pretrained(model_id)

    base_causallm.resize_token_embeddings(len(tokenizer))

    configs = Config()
    model = CoconutGPT_Fixed(
        base_causallm, expainable_llm, tokenizer, latent_id, start_id, end_id,
        tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<<"), configs.c_thought, configs)

    print("Loading state dict...")
    model.load_state_dict(saved_weights, strict=False)
    model = model.to(device)
    if configs.bf16:
        model = model.to(torch.bfloat16)
    model.eval()
    return model, tokenizer, latent_id, start_id, end_id


model, tokenizer, LATENT_ID, START_ID, END_ID = load_coconut_model(CKPT_DIR, MODEL_ID, DEVICE)
model = model.to(torch.bfloat16)
print(f"模型加载完成")


# ============ 3. 辅助函数 ============

def _find_answer_position(all_logits, tokenizer):
    """
    在生成的 logits 中找到第一个含数字的 token 位置。
    """
    probs = torch.softmax(all_logits[0], dim=-1)

    first_digit_pos = None
    first_digit_token_str = None

    for pos in range(probs.shape[0]):
        top_token_id = probs[pos].argmax().item()
        top_token_str = tokenizer.decode([top_token_id]).strip()
        if re.search(r'\d', top_token_str):
            first_digit_pos = pos
            first_digit_token_str = top_token_str
            break

    if first_digit_pos is None:
        first_digit_pos = probs.shape[0] - 1
        first_digit_token_str = tokenizer.decode([probs[first_digit_pos].argmax().item()]).strip()

    return first_digit_pos, first_digit_token_str


def extract_answer_number(sentence):
    sentence = sentence.replace(',', '')
    pred = [s for s in re.findall(r'-?\d+\.?\d*', sentence)]
    if not pred:
        return float('inf')
    return float(pred[-1])


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
    构建prompt格式: 前缀 + 问题 + "\n"
    """
    prompt = prefix_str.strip() + question_str.strip() + "\n"
    question_ids = tokenizer.encode(prompt, add_special_tokens=True)
    input_ids_tensor = torch.tensor([question_ids], dtype=torch.long).to(DEVICE)
    attention_mask = torch.ones_like(input_ids_tensor)
    return input_ids_tensor, attention_mask


def generate_answer(input_ids, attention_mask, max_tokens=MAX_NEW_TOKENS):
    """生成答案"""
    with torch.no_grad():
        output_ids = model.generate_clean(input_ids, attention_mask, max_tokens)
    return tokenizer.decode(output_ids[0], skip_special_tokens=True), output_ids[0].tolist()


def evaluate(question_str, prefix_ids_arr):
    """评估当前前缀"""
    prefix_str = tokenizer.decode(prefix_ids_arr)
    input_ids, attention_mask = make_prompt(question_str, prefix_str)
    text, _ = generate_answer(input_ids, attention_mask)
    return text, extract_answer_number(text)


def compute_logits_diff_at_answer(question_str, prefix_ids_arr):
    """
    计算前缀的logits差值
    """
    prefix_str = tokenizer.decode(prefix_ids_arr)
    input_ids, attention_mask = make_prompt(question_str, prefix_str)
    input_ids = input_ids.to(DEVICE)
    attention_mask = attention_mask.to(DEVICE)

    with torch.no_grad():
        all_logits = model.forward_embeds_for_gradient(
            input_ids=input_ids,
            attention_mask=attention_mask,
            n_latent=N_LATENT_TOKENS,
            target_position=-1
        )

        # 解码并打印每个位置的logits
        '''print(f"\n=== all_logits 解码 (共 {all_logits.shape[1]} 个位置) ===")
        for pos in range(all_logits.shape[1]):
            top_token_id = all_logits[0, pos].argmax().item()
            top_token_str = tokenizer.decode([top_token_id]).strip()
            top_logit = all_logits[0, pos, top_token_id].item()
            print(f"  位置 {pos}: token='{top_token_str}' (id={top_token_id}), logit={top_logit:.4f}")
        print(f"=== 解码结束 ===\n")'''

        first_digit_pos, _ = _find_answer_position(all_logits, tokenizer)
        answer_logits = all_logits[:, first_digit_pos:first_digit_pos+1, :].squeeze(1)

    baseline_id = BASELINE_TOKEN_ID
    target_id = TARGET_TOKEN_ID

    baseline_logit = answer_logits[0, baseline_id].item()
    target_logit = answer_logits[0, target_id].item()
    logit_diff = target_logit - baseline_logit
    baseline_prob = torch.softmax(answer_logits[0], dim=-1)[baseline_id].item()
    target_prob = torch.softmax(answer_logits[0], dim=-1)[target_id].item()

    return logit_diff, baseline_prob, target_prob


# ============ 4. 测试baseline ============
print("\n[3] 测试baseline...")

baseline_input_ids, baseline_attention_mask = make_prompt(question, "")
baseline_text, baseline_ids = generate_answer(baseline_input_ids, baseline_attention_mask)

baseline_answer = extract_answer_number(baseline_text)
print(f"Baseline生成文本: '{baseline_text}'")
print(f"Baseline提取答案: {baseline_answer}")
print(f"正确答案: {answer_str}")

baseline_correct = (str(baseline_answer) == answer_str or
                   abs(baseline_answer - float(answer_str)) < 0.01 if baseline_answer != float('inf') and answer_str.lstrip('-').replace('.','',1).isdigit() else False)
print(f"Baseline是否正确: {baseline_correct}")


# ============ 5. 确定答案token位置和目标token ============
print("\n[4] 获取baseline logits信息...")

with torch.no_grad():
    all_logits_output = model.forward_embeds_for_gradient(
        input_ids=baseline_input_ids,
        attention_mask=baseline_attention_mask,
        n_latent=N_LATENT_TOKENS,
        target_position=-1
    )

    first_digit_pos, first_digit_token_str = _find_answer_position(all_logits_output, tokenizer)

    print(f"回答中第一个含数字的token位置: {first_digit_pos}, token: '{first_digit_token_str}'")
    answer_logits = all_logits_output[:, first_digit_pos:first_digit_pos+1, :]
    ANSWER_TOKEN_POS = first_digit_pos
    answer_logits = answer_logits.squeeze(1)

    sorted_logits, sorted_indices = torch.sort(answer_logits[0], descending=True)

    BASELINE_TOKEN_ID = sorted_indices[0].item()

    TARGET_TOKEN_ID = None
    for i in range(1, sorted_indices.shape[0]):
        candidate_id = sorted_indices[i].item()
        if not tokens_represent_same_number(BASELINE_TOKEN_ID, candidate_id, tokenizer):
            TARGET_TOKEN_ID = candidate_id
            break

    if TARGET_TOKEN_ID is None:
        TARGET_TOKEN_ID = sorted_indices[1].item()
        print(f"  警告: 所有候选token都与top-1表示相同数字，使用top-2: {tokenizer.decode([TARGET_TOKEN_ID])}")

baseline_token_str = tokenizer.decode([BASELINE_TOKEN_ID])
target_token_str = tokenizer.decode([TARGET_TOKEN_ID])
baseline_logit = answer_logits[0, BASELINE_TOKEN_ID].item()
target_logit = answer_logits[0, TARGET_TOKEN_ID].item()
baseline_prob = torch.softmax(answer_logits[0], dim=-1)[BASELINE_TOKEN_ID].item()
target_prob = torch.softmax(answer_logits[0], dim=-1)[TARGET_TOKEN_ID].item()
logit_diff = target_logit - baseline_logit

top5_tokens = [tokenizer.decode([sorted_indices[i].item()]) for i in range(min(5, len(sorted_indices)))]
print(f"Baseline logits at position {ANSWER_TOKEN_POS} (first digit token):")
print(f"  最大token(正确答案): '{baseline_token_str}' (id={BASELINE_TOKEN_ID}), logit={baseline_logit:.4f}, prob={baseline_prob:.4f}")
print(f"  第二大token(目标): '{target_token_str}' (id={TARGET_TOKEN_ID}), logit={target_logit:.4f}, prob={target_prob:.4f}")
print(f"  Logit差值(次大-最大): {logit_diff:.4f}")
print(f"  Top 5 tokens: {top5_tokens}")


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


# ============ 7. 梯度计算函数 ============
print("\n[6] 定义梯度计算函数...")

def compute_gradients(question_str, prefix_ids_arr, pass_gradient_through_latent=True):
    """
    计算损失相对于前缀的梯度
    """
    prefix_len = len(prefix_ids_arr)

    prefix_str = tokenizer.decode(prefix_ids_arr)
    input_ids, attention_mask = make_prompt(question_str, prefix_str)
    input_ids = input_ids.to(DEVICE)
    attention_mask = attention_mask.to(DEVICE)

    embed_layer = model.embedding

    prefix_token_len = len(tokenizer.encode(prefix_str.strip(), add_special_tokens=False))

    with torch.no_grad():
        all_logits = model.forward_embeds_for_gradient(
            input_ids=input_ids,
            attention_mask=attention_mask,
            n_latent=N_LATENT_TOKENS,
            target_position=-1
        )
        first_digit_pos, _ = _find_answer_position(all_logits, tokenizer)

    answer_logits, token_grads = model.forward_embeds_for_gradient(
        input_ids=input_ids,
        attention_mask=attention_mask,
        n_latent=N_LATENT_TOKENS,
        target_position=first_digit_pos,
        compute_gradient=True,
        target_token_id=TARGET_TOKEN_ID,
        baseline_token_id=BASELINE_TOKEN_ID,
        pass_gradient_through_latent=pass_gradient_through_latent
    )

    token_grads_prefix = token_grads[0, :prefix_token_len, :]

    return token_grads_prefix.cpu().detach().to(torch.float32).numpy(), prefix_token_len


# ============ 8. GCG 迭代 ============
print("\n[7] 开始GCG迭代...")

current_text, current_answer = evaluate(question, prefix_ids)
current_correct = (str(current_answer) == answer_str or
                  abs(current_answer - float(answer_str)) < 0.01 if current_answer != float('inf') and answer_str.lstrip('-').replace('.','',1).isdigit() else False)

initial_logit_diff, init_baseline_prob, init_target_prob = compute_logits_diff_at_answer(question, prefix_ids)
print(f"初始前缀 logits差值: {initial_logit_diff:.4f}")
print(f"固定token - BASELINE_TOKEN_ID={BASELINE_TOKEN_ID}, TARGET_TOKEN_ID={TARGET_TOKEN_ID}")
print(f"初始生成: '{current_text}'")
print(f"初始答案: {current_answer}")
print(f"是否正确: {current_correct}")

best_prefix = prefix_ids.copy()
best_correct = current_correct
best_logit_diff = initial_logit_diff

scores_list = []
for iteration in range(N_ITERS):
    curr_len = len(prefix_ids)

    # 生成候选组合
    sampled_combinations = [{}]
    for _ in range(BATCH_SIZE):
        n_changes = np.random.randint(1, curr_len + 1)
        positions = np.random.choice(curr_len, size=n_changes, replace=False)

        combo = {}
        for pos in positions:
            combo[pos] = np.random.choice(char_tokens)

        sampled_combinations.append(combo)

    # 如果使用梯度模式，计算梯度并优先选择梯度最大的候选
    if args.candidate_selection == "gradient":
        gradients, actual_len = compute_gradients(question, prefix_ids, args.pass_gradient_through_latent)
        print(f"梯度均值: {np.mean(gradients):.6f}")
        for i, combo in enumerate(sampled_combinations[1:], 1):
            for pos in combo.keys():
                pos_grads = gradients[pos]
                top_k_indices = np.argsort(pos_grads)[-TOP_K:]
                combo[pos] = np.random.choice(top_k_indices)
        sampled_combinations[0] = {}

    # 评估候选
    scores = []
    for combo in sampled_combinations:
        new_prefix = prefix_ids.copy()
        for pos, tok_id in combo.items():
            new_prefix[pos] = tok_id
        logit_diff, _, _ = compute_logits_diff_at_answer(question, new_prefix)
        scores.append(logit_diff)

    scores_np = np.array(scores)
    best_idx = np.argmax(scores_np)
    best_combo = sampled_combinations[best_idx]
    best_score = scores_np[best_idx]

    for pos, tok_id in best_combo.items():
        prefix_ids[pos] = tok_id

    scores_list.append(best_score)
    if best_score > best_logit_diff:
        best_logit_diff = best_score
        best_prefix = prefix_ids.copy()

    current_text, current_answer = evaluate(question, prefix_ids)
    current_correct = (str(current_answer) == answer_str or
                      abs(current_answer - float(answer_str)) < 0.01 if current_answer != float('inf') and answer_str.lstrip('-').replace('.','',1).isdigit() else False)
    curr_logit_diff, curr_baseline_prob, curr_target_prob = compute_logits_diff_at_answer(question, prefix_ids)
    print(curr_baseline_prob, curr_target_prob)
    print(f"Iter {iteration+1}: logit差值={curr_logit_diff:.4f}, 最佳差值={best_logit_diff:.4f}, 答案={current_answer}, 正确={current_correct}")
    if best_logit_diff > 1.5 and current_correct == False:
        break


# ============ 9. 结果 ============
print("\n" + "=" * 60)
print("最终结果")
print("=" * 60)

final_prefix_str = tokenizer.decode(best_prefix)
print(f"对抗前缀: '{final_prefix_str}'")

clean_text, clean_answer = evaluate(question, np.array([]))
clean_correct = (str(clean_answer) == answer_str or
                abs(clean_answer - float(answer_str)) < 0.01 if clean_answer != float('inf') and answer_str.lstrip('-').replace('.','',1).isdigit() else False)

clean_logit_diff, _, _ = compute_logits_diff_at_answer(question, np.array([]))

adv_text, adv_answer = evaluate(question, best_prefix)
adv_correct = (str(adv_answer) == answer_str or
              abs(adv_answer - float(answer_str)) < 0.01 if adv_answer != float('inf') and answer_str.lstrip('-').replace('.','',1).isdigit() else False)

adv_logit_diff, _, _ = compute_logits_diff_at_answer(question, best_prefix)

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

os.makedirs(args.output_dir, exist_ok=True)
save_path = os.path.join(args.output_dir, f"problem_{PROBLEM_ID}.json")
with open(save_path, "w") as f:
    json.dump(result, f, indent=2, ensure_ascii=False)

print(f"\n结果已保存到 {save_path}")
