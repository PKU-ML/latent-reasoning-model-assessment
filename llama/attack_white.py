"""
================================================================================
Llama模型 白盒对抗攻击 - GCG优化对抗前缀
================================================================================

一、算法概述
------------
本代码实现了一种针对Llama模型的GCG对抗攻击，目标是通过优化一个前缀字符串，
使得模型在给定问题下输出错误的答案。

二、攻击目标
------------
模型输出格式为 "The answer is ..."

攻击目标：
- 在问题前添加3-5个token的对抗前缀
- 使得模型直接输出错误答案（错误答案格式为"The answer is ..."）
- 错误答案的确定：如果正确答案不是0，则错误答案为0；否则错误答案为10

三、关键实现
------------
1. 梯度计算：
   - 对前缀的每个token计算损失函数对embedding的梯度
   - 损失函数是模型输出错误答案序列（"The answer is 0"或"The answer is 10"）的logits概率
   - 使用梯度上升（不加负号），选择梯度最大的token

2. GCG优化：
   - 对前缀的每个位置，计算梯度
   - 选取梯度最大的TOP_K个候选token
   - 随机采样1到前缀长度个位置进行替换
   - 评估候选前缀（使用目标序列概率选择最佳候选）
   - 提前终止：使用vLLM两轮输出验证是否成功攻击

3. 评估方式：
   - 使用与llama/test.py完全一致的判定方式
   - 两轮vLLM输出，判定输出的答案是否正确（不是概率）

4. 问题ID加载：
   - 从结果文件加载correct_ids
   - 支持按数据集过滤
   - 支持跳过已存在的问题

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
from typing import List, Dict, Tuple
from pathlib import Path

from vllm import LLM, SamplingParams

# ============ 命令行参数 ============
parser = argparse.ArgumentParser(description="Llama GCG 对抗前缀攻击")
parser.add_argument("--results-file", type=str,
                    default=os.environ.get(
                        "LLAMA_RESULTS_FILE",
                        str(Path(__file__).resolve().parent / "results" / "org" / "test_org_results.json"),
                    ),
                    help="包含correct_ids的结果文件路径 (默认: <llama>/results/org/test_org_results.json)")
parser.add_argument("--dataset", type=str, default=None,
                    help="指定数据集 (gsm8k/MultiArith/SVAMP)，None表示所有")
parser.add_argument("--problem-ids", type=str, default=None,
                    help="指定要攻击的问题ID，逗号分隔，如 '11,16,18'")
parser.add_argument("--prefix-length", type=int, default=3,
                    help="前缀长度 (默认5)")
parser.add_argument("--n-iters", type=int, default=30,
                    help="迭代次数 (默认100)")
parser.add_argument("--top-k", type=int, default=300,
                    help="每个位置考虑top-k个候选 (默认300)")
parser.add_argument("--batch-size", type=int, default=40,
                    help="候选batch大小 (默认40)")
parser.add_argument("--seed", type=int, default=42,
                    help="随机种子 (默认42)")
parser.add_argument("--output-dir", type=str,
                    default=os.environ.get(
                        "LLAMA_OUTPUT_DIR",
                        str(Path(__file__).resolve().parent / "adv_results" / "results_white"),
                    ),
                    help="结果保存目录 (默认: <llama>/adv_results/results_white)")
parser.add_argument("--data-dir", type=str,
                    default=os.environ.get(
                        "DATA_DIR",
                        str(Path(__file__).resolve().parent.parent / "data"),
                    ),
                    help="数据目录 (默认: <repo>/data)")
parser.add_argument("--model-name-or-path", type=str,
                    default=os.environ.get(
                        "LLAMA_MODEL_PATH",
                        "meta-llama/Llama-3.2-1B-Instruct",
                    ),
                    help="模型名称或路径 (默认读取 LLAMA_MODEL_PATH 环境变量)")
parser.add_argument("--token", type=str, default=None,
                    help="HuggingFace token")
parser.add_argument("--max-model-len", type=int, default=4096,
                    help="模型最大长度")
parser.add_argument("--gpu-memory-utilization", type=float, default=0.5,
                    help="vLLM GPU内存利用率")
parser.add_argument("--temperature", type=float, default=0.0,
                    help="生成温度")
parser.add_argument("--skip-existing", action="store_true", default=False,
                    help="跳过已存在的问题结果")
args = parser.parse_args()

# 如果指定了dataset，则在output_dir中添加dataset子目录
if args.dataset:
    args.output_dir = os.path.join(args.output_dir, args.dataset)

# 设置随机种子
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)

# ============ 配置 ============
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = args.model_name_or_path
DATA_DIR = Path(args.data_dir)

# GCG 参数
PREFIX_LENGTH = args.prefix_length
N_ITERS = args.n_iters
TOP_K = args.top_k
BATCH_SIZE = args.batch_size
MAX_NEW_TOKENS = 256

# ============ 辅助函数 ============

def load_local_dataset(data_path):
    """加载本地数据集"""
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


def load_problem_ids(results_file):
    """从结果文件加载correct_ids"""
    with open(results_file, 'r') as f:
        data = json.load(f)
    all_ids = []
    for dataset_name, dataset_data in data.items():
        for pid in dataset_data.get("correct_ids", []):
            all_ids.append((dataset_name, pid))
    return all_ids


def extract_answer_number(sentence: str) -> float:
    """从\\boxed{}中提取答案数字"""
    matches = re.findall(r'\\boxed\{([^}]+)\}', sentence)
    if matches:
        try:
            return float(matches[-1])
        except ValueError:
            pass
    pred = re.findall(r'-?\d+\.?\d*', sentence)
    if not pred:
        return float('inf')
    return float(pred[-1])


def compute_accuracy(pred_answer, gold_answer):
    """判断预测是否正确"""
    if gold_answer == 0:
        return pred_answer == gold_answer
    else:
        return pred_answer == gold_answer or abs(pred_answer - gold_answer) < 0.01


# ============ 全局变量（初始化在main中） ============
tokenizer = None
model = None
llm = None
sampling_params = None
char_tokens = []


# ============ 初始化函数 ============

def initialize():
    """初始化tokenizer、模型和token列表"""
    global tokenizer, model, char_tokens

    print("\n[1] 加载tokenizer和模型...")

    # 使用 transformers 的 AutoModelForCausalLM 获取 tokenizer
    from transformers import AutoTokenizer, AutoModelForCausalLM

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=args.token, use_fast=False)
    if tokenizer.pad_token_id is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        token=args.token,
        torch_dtype=torch.bfloat16,
    ).to(DEVICE)
    model.eval()

    print(f"模型加载完成")

    # 找出有效的单token
    print("\n[2] 初始化token列表...")
    char_tokens = []
    for i in range(8000, 200000):
        try:
            decoded = tokenizer.decode([i])
            if decoded.strip() and len(decoded.strip()) > 0:
                char_tokens.append(i)
        except:
            pass

    print(f"有效token数量: {len(char_tokens)}")


def initialize_vllm():
    """初始化vLLM"""
    global llm, sampling_params

    print("\n[3] 加载vLLM...")

    llm = LLM(
        model=MODEL_NAME,
        hf_token=args.token,
        trust_remote_code=True,
        tensor_parallel_size=1,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
    )

    sampling_params = SamplingParams(
        max_tokens=MAX_NEW_TOKENS,
        temperature=args.temperature,
        top_p=1.0,
        stop=None,
    )

    print("vLLM加载完成")


# ============ 辅助函数（使用tokenizer） ============

def make_prompt(question_str, prefix_str=""):
    """构建prompt，使用chat template"""
    full_text = prefix_str.strip() + question_str.strip()
    messages = [{"role": "user", "content": full_text}]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    return prompt, messages


# ============ vLLM评估函数 ============

def vllm_two_round_eval(question_str, prefix_str="", correct_answer=0):
    """使用vLLM两轮输出评估，与llama/test.py一致"""
    prompt, messages = make_prompt(question_str, prefix_str)

    # 第一轮
    outputs = llm.generate([prompt], sampling_params)
    first_round_text = outputs[0].outputs[0].text

    # 构建第二轮prompt
    messages.append({"role": "assistant", "content": first_round_text + "\nSo the answer is \\boxed{"})
    round2_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # 第二轮
    round2_outputs = llm.generate([round2_prompt], sampling_params)
    second_round_text = round2_outputs[0].outputs[0].text

    # 组合完整输出
    full_text = first_round_text + "\nSo the answer is \\boxed{" + second_round_text

    # 提取答案
    pred_answer = extract_answer_number(full_text)
    is_correct = compute_accuracy(pred_answer, correct_answer)

    return full_text, pred_answer, is_correct


# ============ 梯度计算函数 ============

def compute_gradients(question_str, prefix_ids_arr, target_token_ids, num_target_tokens):
    """
    计算损失相对于前缀的梯度

    损失函数：计算模型输出的前几个token是目标序列的logits概率之和
    使用梯度上升（不加负号），选择梯度最大的token
    """
    prefix_len = len(prefix_ids_arr)

    # 构建输入
    prefix_str = tokenizer.decode(prefix_ids_arr)
    prompt, messages = make_prompt(question_str, prefix_str)

    # tokenize
    input_ids = tokenizer(prompt, return_tensors="pt", padding=True)
    input_ids = {k: v.to(DEVICE) for k, v in input_ids.items()}

    # 获取embedding层
    embed_layer = model.model.embed_tokens

    # 对输入embedding求和作为"锚点"，确保梯度连通
    with torch.no_grad():
        input_embeds_detach = embed_layer(input_ids['input_ids'])

    # 创建一个可训练的权重向量，用于"吸收"梯度
    grad_anchor = torch.zeros_like(input_embeds_detach).requires_grad_(True)

    # 前向传播
    outputs = model(
        inputs_embeds=input_embeds_detach + grad_anchor,
        attention_mask=input_ids['attention_mask'],
        use_cache=False,
        output_hidden_states=True,
    )

    # 自回归生成
    all_logits = []
    generated_ids = []
    past_key_values = outputs.past_key_values
    next_token_emb = outputs.hidden_states[-1][:, -1:, :]

    for step in range(num_target_tokens):
        outputs = model(
            inputs_embeds=next_token_emb,
            use_cache=True,
            past_key_values=past_key_values,
        )
        past_key_values = outputs.past_key_values
        logits = outputs.logits[:, -1, :]
        all_logits.append(logits[0])

        next_token_id = logits.argmax(dim=-1).item()
        generated_ids.append(next_token_id)

        if next_token_id == tokenizer.eos_token_id:
            break

        next_token_emb = embed_layer(torch.tensor([next_token_id], device=DEVICE)).unsqueeze(0)

    # 计算目标序列的logits之和
    seq_log_prob = torch.tensor(0.0, device=DEVICE, requires_grad=True)
    for i in range(min(len(generated_ids), num_target_tokens)):
        seq_log_prob = seq_log_prob + all_logits[i].gather(0, target_token_ids[i:i+1])

    # 反向传播
    seq_log_prob.backward()

    # 获取梯度
    input_grad = grad_anchor.grad
    if input_grad is None:
        return np.zeros((prefix_len, embed_layer.weight.shape[0])), prefix_len, 0.0

    # 前缀位置的梯度
    prefix_grad_emb = input_grad[0, :prefix_len]
    token_grads = torch.matmul(prefix_grad_emb, embed_layer.weight.T)

    # 计算目标序列的概率（用于记录）
    probs = torch.softmax(torch.stack(all_logits[:num_target_tokens]), dim=-1)
    target_prob = 1.0
    for i in range(min(len(generated_ids), num_target_tokens)):
        target_prob *= probs[i, target_token_ids[i]].item()

    return token_grads.cpu().detach().to(torch.float32).numpy(), prefix_len, target_prob


def compute_target_prob(question_str, prefix_ids_arr, target_token_ids, num_target_tokens):
    """
    计算给定前缀下，模型输出的前几个token是目标序列的概率
    """
    prefix_str = tokenizer.decode(prefix_ids_arr)
    prompt, messages = make_prompt(question_str, prefix_str)

    # tokenize
    input_ids = tokenizer(prompt, return_tensors="pt", padding=True)
    input_ids = {k: v.to(DEVICE) for k, v in input_ids.items()}

    embed_layer = model.model.embed_tokens

    with torch.no_grad():
        # 前向传播 - 获取初始hidden states和past_key_values
        outputs = model(
            input_ids=input_ids['input_ids'],
            attention_mask=input_ids['attention_mask'],
            use_cache=True,
            output_hidden_states=True,
        )
        past_key_values = outputs.past_key_values
        next_token_emb = outputs.hidden_states[-1][:, -1:, :]

        all_logits = []
        generated_ids = []

        for step in range(num_target_tokens):
            outputs = model(
                inputs_embeds=next_token_emb,
                use_cache=True,
                past_key_values=past_key_values,
            )
            past_key_values = outputs.past_key_values
            logits = outputs.logits[:, -1, :]
            all_logits.append(logits[0])

            next_token_id = logits.argmax(dim=-1).item()
            generated_ids.append(next_token_id)

            if next_token_id == tokenizer.eos_token_id:
                break

            next_token_emb = embed_layer(torch.tensor([next_token_id], device=DEVICE)).unsqueeze(0)

    # 计算目标序列的联合概率
    probs = torch.softmax(torch.stack(all_logits[:num_target_tokens]), dim=-1)
    target_prob = 1.0
    for i in range(min(len(generated_ids), num_target_tokens)):
        target_prob *= probs[i, target_token_ids[i]].item()

    return target_prob


# ============ 单问题攻击函数 ============

def run_attack_for_problem(dataset_name, problem_id):
    """对单个问题运行攻击"""
    # 加载该数据集
    data_path = DATA_DIR / f"{dataset_name}.json"
    if not data_path.exists():
        print(f"数据集文件不存在: {data_path}")
        return False

    questions, answers = load_local_dataset(str(data_path))

    if problem_id >= len(questions):
        print(f"问题ID {problem_id} 超出数据集范围")
        return False

    question = questions[problem_id]
    correct_answer = answers[problem_id]
    correct_answer_str = str(int(correct_answer)) if correct_answer != float("inf") else str(correct_answer)

    # 确定错误答案
    wrong_answer = 0 if correct_answer != 0 else 10
    wrong_answer_str = str(wrong_answer)

    # 确定目标序列
    if wrong_answer == 0:
        target_seq = "The answer is 0"
    else:
        target_seq = "The answer is 10"

    target_tokens = tokenizer.encode(target_seq, add_special_tokens=False)
    target_token_ids = torch.tensor(target_tokens, device=DEVICE)
    num_target_tokens = len(target_tokens)

    print(f"\n{'='*60}")
    print(f"攻击数据集={dataset_name}, 问题ID={problem_id}")
    print(f"问题: {question[:100]}...")
    print(f"正确答案: {correct_answer_str}, 错误答案: {wrong_answer_str}")
    print(f"目标序列: '{target_seq}'")
    print(f"{'='*60}")

    # 测试baseline
    print("\n测试baseline...")
    baseline_text, baseline_pred, baseline_correct = vllm_two_round_eval(question, "", correct_answer)
    print(f"Baseline预测: {baseline_pred}, 正确: {baseline_correct}")

    if not baseline_correct:
        print(f"Baseline就已经不正确，跳过")
        return False

    # 初始化前缀
    prefix_ids = np.random.choice(char_tokens, size=PREFIX_LENGTH, replace=True)
    best_prefix = prefix_ids.copy()
    best_wrong_prob = compute_target_prob(question, prefix_ids, target_token_ids, num_target_tokens)

    # GCG迭代
    for iteration in range(N_ITERS):
        gradients, actual_len, wrong_prob = compute_gradients(question, prefix_ids, target_token_ids, num_target_tokens)

        sampled_combinations = [{}]
        for _ in range(BATCH_SIZE):
            n_changes = np.random.randint(1, actual_len + 1)
            positions = np.random.choice(actual_len, size=n_changes, replace=False)

            combo = {}
            for pos in positions:
                pos_grads = gradients[pos]
                top_k_indices = np.argsort(pos_grads)[-TOP_K:]
                combo[pos] = np.random.choice(top_k_indices)

            sampled_combinations.append(combo)

        scores = []
        for combo in sampled_combinations:
            new_prefix = prefix_ids.copy()
            for pos, tok_id in combo.items():
                new_prefix[pos] = tok_id
            prob = compute_target_prob(question, new_prefix, target_token_ids, num_target_tokens)
            scores.append((prob, combo))

        scores = sorted(scores, key=lambda x: x[0])
        best_combo = scores[-1][1]
        best_prob = scores[-1][0]

        for pos, tok_id in best_combo.items():
            prefix_ids[pos] = tok_id

        if best_prob > best_wrong_prob:
            best_wrong_prob = best_prob
            best_prefix = prefix_ids.copy()

        current_prefix_str = tokenizer.decode(prefix_ids)
        _, _, is_correct = vllm_two_round_eval(question, current_prefix_str, correct_answer)

        if not is_correct:
            print(f"Iter {iteration+1}: 攻击成功!")
            break

        if iteration % 10 == 0:
            print(f"Iter {iteration+1}: 错误概率={best_prob:.4f}, 正确={is_correct}")

    # 评估
    final_prefix_str = tokenizer.decode(best_prefix)
    clean_text, clean_pred, clean_correct = vllm_two_round_eval(question, "", correct_answer)
    adv_text, adv_pred, adv_correct = vllm_two_round_eval(question, final_prefix_str, correct_answer)
    attack_success = not adv_correct and clean_correct

    print(f"\n无前缀: 预测={clean_pred}, 正确={clean_correct}")
    print(f"有前缀: 预测={adv_pred}, 正确={adv_correct}")
    print(f"攻击成功: {attack_success}")

    # 保存
    result = {
        "dataset": dataset_name,
        "problem_id": problem_id,
        "question": question,
        "ground_truth": correct_answer_str,
        "wrong_answer": wrong_answer_str,
        "target_seq": target_seq,
        "prefix": final_prefix_str,
        "prefix_token_ids": [int(x) for x in best_prefix],
        "best_wrong_prob": best_wrong_prob,
        "clean_text": clean_text[:200],
        "clean_pred": clean_pred,
        "clean_correct": clean_correct,
        "adv_text": adv_text[:200],
        "adv_pred": adv_pred,
        "adv_correct": adv_correct,
        "attack_success": attack_success,
    }

    # 保存路径
    dataset_output_dir = os.path.join(args.output_dir, dataset_name)
    os.makedirs(dataset_output_dir, exist_ok=True)
    save_path = os.path.join(dataset_output_dir, f"problem_{problem_id}.json")
    with open(save_path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"结果已保存到 {save_path}")
    return attack_success


# ============ 主函数 ============

def main():
    """主函数：批量攻击多个问题"""
    global DEVICE

    print(f"使用设备: {DEVICE}")
    print("=" * 60)
    print("Llama GCG 寻找对抗前缀")
    print("=" * 60)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    # ============ 初始化 ============
    initialize()
    initialize_vllm()

    # ============ 加载问题ID ============
    print("\n[4] 加载问题ID...")
    problem_ids = []
    if args.problem_ids:
        # 命令行指定
        for pid in args.problem_ids.split(","):
            problem_ids.append(("", int(pid.strip())))
    else:
        # 从结果文件加载
        problem_ids = load_problem_ids(args.results_file)
        if args.dataset:
            problem_ids = [(ds, pid) for ds, pid in problem_ids if ds == args.dataset]

    print(f"共有 {len(problem_ids)} 个问题需要攻击")

    # ============ 跳过已攻击的问题 ============
    if args.skip_existing:
        existing_ids = []
        for dataset_name, PROBLEM_ID in problem_ids:
            save_path = os.path.join(args.output_dir, dataset_name, f"problem_{PROBLEM_ID}.json")
            if os.path.exists(save_path):
                existing_ids.append((dataset_name, PROBLEM_ID))

        if existing_ids:
            print(f"跳过 {len(existing_ids)} 个已攻击的问题")
            for ds, pid in existing_ids:
                print(f"  - {ds}_{pid}")
            problem_ids = [p for p in problem_ids if p not in set(existing_ids)]
            print(f"剩余 {len(problem_ids)} 个问题需要攻击")

    # ============ 依次攻击每个问题 ============
    print("\n[5] 开始攻击...")
    print("=" * 60)

    all_results = {}

    for dataset_name, problem_id in problem_ids:
        try:
            attack_success = run_attack_for_problem(dataset_name, problem_id)
            all_results[f"{dataset_name}_{problem_id}"] = {
                "dataset": dataset_name,
                "problem_id": problem_id,
                "attack_success": attack_success,
            }
        except Exception as e:
            print(f"攻击失败: {e}")
            all_results[f"{dataset_name}_{problem_id}"] = {
                "dataset": dataset_name,
                "problem_id": problem_id,
                "attack_success": False,
                "error": str(e),
            }

    # ============ 汇总结果 ============
    print("\n" + "=" * 60)
    print("攻击完成 - 汇总结果")
    print("=" * 60)

    total = len(problem_ids)
    attack_ok = sum(1 for r in all_results.values() if r.get("attack_success"))

    print(f"总问题数: {total}")
    print(f"攻击成功: {attack_ok} ({100*attack_ok/total:.1f}% if total > 0 else 0)")

    # 保存汇总结果
    summary_path = os.path.join(args.output_dir, "attack_summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n汇总结果保存到: {summary_path}")


if __name__ == "__main__":
    main()