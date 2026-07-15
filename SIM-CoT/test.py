#!/usr/bin/env python3
"""
SIM-CoT Coconut 模型评估脚本
使用与attack_black.py相同的模型运行方式
"""

import logging
import sys
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

import math
import re
import os
import argparse
from dataclasses import dataclass
from typing import Dict, Optional, Sequence
from pathlib import Path

import torch
import torch.nn as nn
import transformers
from torch.nn import functional as F
import json

import numpy as np

from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.nn import CrossEntropyLoss

# ============ 模型配置（与attack_black.py一致）===========
MODEL_ID = os.environ.get("SIMCOT_MODEL_ID", "gpt2")
CKPT_DIR = os.environ.get(
    "SIMCOT_CKPT_DIR",
    str(Path(__file__).resolve().parent / "SIM-CoT" / "Coconut" / "ckpts" / "SIM_COT-GPT2-Coconut" / "checkpoint_28"),
)
MAX_NEW_TOKENS = 64
N_LATENT_TOKENS = 10

# 设备
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


# ============ CoconutGPT_Fixed（与attack_black.py完全一致）===========
class CoconutGPT_Fixed(torch.nn.Module):
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
        input_embeds = self.embedding(input_ids).clone().requires_grad_(True)

        outputs = self.base_causallm(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=True)
        kv_cache = outputs.past_key_values

        latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)

        if not pass_gradient_through_latent:
            latent_embd = latent_embd.detach()

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

        end_latent_emb = self.embedding(torch.tensor([self.end_latent_id], device=input_ids.device)).unsqueeze(1)

        outputs = self.base_causallm(
            inputs_embeds=end_latent_emb,
            attention_mask=None,
            past_key_values=kv_cache,
            output_hidden_states=True,
            use_cache=True)

        logits = outputs.logits

        all_logits = [logits]
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

    def generate_clean(self, input_ids, attention_mask, max_new_tokens=16, return_latents=False,
                    replace_latent_vectors=None, question_idx=None):
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

        # 用于存储latent向量
        all_latent_embds = []
        if return_latents:
            all_latent_embds.append(latent_embd.clone())

        # 替换初始推理向量 (如果启用)
        if replace_latent_vectors is not None and question_idx is not None:
            if question_idx in replace_latent_vectors:
                latent_embd[0, 0, :] = torch.from_numpy(replace_latent_vectors[question_idx][0]).to(
                    input_ids.device).to(self.config.bf16 and torch.bfloat16 or latent_embd.dtype)

        for i in range(N_LATENT_TOKENS):
            outputs = self.base_causallm(
                inputs_embeds=latent_embd,
                attention_mask=None,
                past_key_values=kv_cache,
                output_hidden_states=True, use_cache=True)
            kv_cache = outputs.past_key_values
            latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)
            if return_latents:
                all_latent_embds.append(latent_embd.clone())

            # 替换推理向量 (如果启用)
            if replace_latent_vectors is not None and question_idx is not None:
                if question_idx in replace_latent_vectors:
                    vec_idx = i + 1
                    if vec_idx < replace_latent_vectors[question_idx].shape[0]:
                        latent_embd[0, 0, :] = torch.from_numpy(replace_latent_vectors[question_idx][vec_idx]).to(
                            input_ids.device).to(self.config.bf16 and torch.bfloat16 or latent_embd.dtype)

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

        if return_latents:
            return torch.tensor(tokens).view(1, -1), all_latent_embds
        return torch.tensor(tokens).view(1, -1)


@dataclass
class Config:
    model_id: str = "gpt2"
    c_thought: int = 2
    max_latent_stage: int = 5
    training_method: str = "full"
    bf16: bool = False


def load_coconut_model(checkpoint_path, model_id="gpt2", device="cuda"):
    """加载Coconut模型（与attack_black.py一致）"""
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


def load_local_dataset(data_path):
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


def _find_answer_position(all_logits, tokenizer):
    """找到第一个含数字的token位置"""
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
    """从文本中提取答案数字"""
    sentence = sentence.replace(',', '')
    pred = [s for s in re.findall(r'-?\d+\.?\d*', sentence)]
    if not pred:
        return float('inf')
    return float(pred[-1])


def make_prompt(question_str, tokenizer, device):
    """构建prompt（与attack_black.py一致）"""
    prompt = question_str.strip() + "\n"
    question_ids = tokenizer.encode(prompt, add_special_tokens=True)
    input_ids_tensor = torch.tensor([question_ids], dtype=torch.long).to(device)
    attention_mask = torch.ones_like(input_ids_tensor)
    return input_ids_tensor, attention_mask


def generate_answer(model, input_ids, attention_mask, max_tokens=MAX_NEW_TOKENS, return_latents=False,
                   replace_latent_vectors=None, question_idx=None):
    """生成答案（与attack_black.py一致）"""
    with torch.no_grad():
        return model.generate_clean(input_ids, attention_mask, max_tokens, return_latents=return_latents,
                                   replace_latent_vectors=replace_latent_vectors, question_idx=question_idx)


def evaluation(checkpoint_path, model_id, data_args, max_new_tokens=64, question_ids=None, gcg_prefixes=None,
               decode_latent=False, compute_latent_length=False, save_latent_vectors=False,
               replace_latent_vectors=None):
    """
    Coconut模型评估函数（使用与attack_black.py一致的模型运行方式）

    参数：
    - decode_latent: 是否解码推理向量
    - compute_latent_length: 是否统计推理向量的长度
    - save_latent_vectors: 是否保存完整推理向量
    """
    # 加载模型
    model, tokenizer, latent_id, start_id, end_id = load_coconut_model(
        checkpoint_path, model_id, DEVICE
    )
    model = model.to(torch.bfloat16)

    # 加载数据集
    logging.warning(f"Loading local dataset from: {data_args.data_path}")
    question, answer = load_local_dataset(data_args.data_path)

    print(f"Loaded {len(question)} questions from local dataset")

    # 如果指定了question_ids，只评估这些id
    eval_indices = list(question_ids) if question_ids is not None else range(len(question))

    # Load replacement latent vectors if provided
    replacement_vectors = None
    if replace_latent_vectors is not None:
        try:
            replacement_data = np.load(replace_latent_vectors)
            replacement_vectors = {int(k.split('_')[-1]): v for k, v in replacement_data.items()}
            print(f"Loaded {len(replacement_vectors)} replacement latent vectors from {replace_latent_vectors}")
        except Exception as e:
            logging.warning(f"Failed to load replacement vectors: {e}")
            replacement_vectors = None

    results = []
    ans_pred_list = []

    # latent decode 配置
    probe_topk = 5

    # 存储latent向量相关数据
    question_latent_lengths = {idx: [] for idx in eval_indices} if compute_latent_length else None
    question_latent_vectors = {idx: [] for idx in eval_indices} if save_latent_vectors else None

    # 初始化batch级latent解码结果
    batch_latent_decoded = [] if decode_latent else None

    for idx in eval_indices:
        q = question[idx]
        gold_answer = answer[idx]
        original_question = q

        # 如果有 gcg_prefix，为问题添加前缀
        if gcg_prefixes and idx in gcg_prefixes:
            prefix = gcg_prefixes[idx]
            q = prefix.strip() + q.strip()
            if do_print:
                print(f"Added prefix for question {idx}: '{prefix}'")

        # 构建输入（与attack_black.py一致）
        input_ids, attention_mask = make_prompt(q, tokenizer, DEVICE)

        with torch.no_grad():
            # 使用generate_clean生成答案，同时可选返回latent向量
            if decode_latent or compute_latent_length or save_latent_vectors or replacement_vectors is not None:
                output_ids, all_latent_embds = generate_answer(model, input_ids, attention_mask, max_new_tokens, return_latents=True,
                                                                 replace_latent_vectors=replacement_vectors, question_idx=idx)
            else:
                output_ids = generate_answer(model, input_ids, attention_mask, max_new_tokens)
                all_latent_embds = None

            decoded_output = tokenizer.decode(output_ids[0], skip_special_tokens=True)

            # 处理latent向量
            if all_latent_embds is not None:
                # 计算latent向量长度
                if compute_latent_length:
                    for vec in all_latent_embds:
                        length = torch.sqrt(torch.dot(vec[0, 0, :], vec[0, 0, :])).item()
                        question_latent_lengths[idx].append(length)

                # 保存完整latent向量
                if save_latent_vectors:
                    for vec in all_latent_embds:
                        question_latent_vectors[idx].append(vec[0, 0, :].clone().float().cpu().numpy())

                # 解码latent token
                if decode_latent:
                    latent_decoded = []
                    for latent_embd in all_latent_embds:
                        probs = torch.nn.functional.softmax(model.base_causallm.lm_head(latent_embd), dim=-1)
                        top5_values, top5_indices = torch.topk(probs, k=probe_topk, dim=2)
                        latent_decoded.append(top5_indices[0, 0, 0].item())
                    batch_latent_decoded.append(latent_decoded)

        # 提取答案数字
        pred_answer = extract_answer_number(decoded_output)

        if do_print:
            print(f"Question {idx} Starts...")
            print(f"Q: {q}")
            print(f"Full output: {decoded_output}")
            print(f"Extracted: {pred_answer}")
            print(f"Prediction={pred_answer}; Groundtruth={gold_answer}")
            print("")

        sample_result = {
            'id': idx,
            'question': q,
            'original_question': original_question,
            'ground_truth': gold_answer,
            'prediction': pred_answer,
            'model_output': decoded_output,
            'correct': pred_answer == gold_answer,
            'has_prefix': gcg_prefixes is not None and idx in gcg_prefixes
        }

        # 添加latent相关结果
        if decode_latent and batch_latent_decoded is not None and len(batch_latent_decoded) > 0:
            latent_decoded_texts = [tokenizer.decode(tid) for tid in batch_latent_decoded[-1]]
            sample_result['latent_tokens_decoded'] = latent_decoded_texts

        if compute_latent_length and question_latent_lengths is not None:
            sample_result['latent_vector_lengths'] = question_latent_lengths[idx]

        results.append(sample_result)
        ans_pred_list.append(pred_answer)

    # 计算准确率
    eval_answers = [answer[i] for i in eval_indices]
    accuracy = compute_accuracy(eval_answers, ans_pred_list)

    print(f"Dataset: {data_args.data_name} | Accuracy: {100*accuracy:.2f}%")

    # 统计推理向量长度 (如果启用)
    avg_latent_length = None
    all_latent_lengths = []
    if compute_latent_length and question_latent_lengths:
        for qid in eval_indices:
            lengths = question_latent_lengths[qid]
            if lengths:
                avg_length = sum(lengths) / len(lengths)
                all_latent_lengths.append(avg_length)
        if all_latent_lengths:
            avg_latent_length = sum(all_latent_lengths) / len(all_latent_lengths)
            print(f"average latent vector length: {avg_latent_length}")

    return accuracy, results, avg_latent_length, question_latent_vectors


def compute_accuracy(gold: list, pred: list) -> float:
    """计算预测准确率"""
    acc = 0.0
    for p, g in zip(pred, gold):
        if isinstance(p, list):
            if g in p:
                acc += 1
        else:
            if p == g:
                acc += 1
    return acc / len(gold)


@dataclass
class DataArguments:
    """数据配置参数"""
    data_name: str = None
    data_path: str = None
    batch_size: int = 1


# 是否打印每个问题的详细信息
do_print = False


def load_correct_ids_from_results(correct_ids_file):
    """Load correct_ids from a previous test results JSON file."""
    correct_ids_set = {}
    correct_ids_path = Path(correct_ids_file)
    if correct_ids_path.exists():
        with open(correct_ids_path, 'r', encoding='utf-8') as f:
            correct_ids_data = json.load(f)
        if 'summary' in correct_ids_data:
            for dataset_name in correct_ids_data['summary']:
                correct_ids_set[dataset_name] = correct_ids_data['summary'][dataset_name].get("correct_ids", [])
        else:
            for dataset_name in correct_ids_data:
                correct_ids_set[dataset_name] = correct_ids_data[dataset_name].get("correct_ids", [])
        logger.info(f"Loaded correct_ids from {correct_ids_file}: { {k: len(v) for k, v in correct_ids_set.items()} }")
    else:
        logger.warning(f"correct_ids file not found: {correct_ids_path}")
    return correct_ids_set


def load_gcg_prefixes(gcg_dir, dataset_name):
    """Load gcg prefixes from directory containing problem_*.json files."""
    prefixes = {}
    dataset_dir = gcg_dir / dataset_name

    if dataset_dir.exists():
        for json_file in dataset_dir.glob("problem_*.json"):
            qid = int(json_file.stem.split("_")[1])
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                try:
                    prefixes[qid] = data["best_fact"] + " "
                except:
                    try:
                        prefixes[qid] = data['prefix']
                    except:
                        try:
                            prefixes[qid] = (data['all_results'][:1] + [i for i in data['all_results'][:30] if i['attack_success'] == True])[-1]['prefix']
                        except:
                            prefixes[qid] = ""
        logger.info(f"Loaded {len(prefixes)} gcg prefixes from {dataset_dir}")
    else:
        logger.warning(f"gcg_results subdir not found: {dataset_dir}")

    return prefixes


# =========================================================================
# 主程序入口
# =========================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SIM-CoT Coconut 模型评估脚本")
    parser.add_argument("--dataset", type=str, default=None,
                        help="指定要评估的数据集名称 (gsm8k, MultiArith, SVAMP)。如果为空，则评估所有数据集。")
    parser.add_argument("--question_id", type=int, default=None,
                        help="指定要评估的问题索引。如果为空，则评估所有问题。")
    parser.add_argument("--model_id", type=str, default="gpt2",
                        help="基础模型名称或路径 (默认: gpt2)")
    parser.add_argument("--checkpoint", type=str,
                        default=os.environ.get(
                            "SIMCOT_CKPT_DIR",
                            str(Path(__file__).resolve().parent / "SIM-CoT" / "Coconut" / "ckpts" / "SIM_COT-GPT2-Coconut" / "checkpoint_28"),
                        ),
                        help="模型checkpoint路径 (默认读取 SIMCOT_CKPT_DIR 环境变量)")
    parser.add_argument("--max_new_tokens", type=int, default=64,
                        help="最大生成token数量 (默认: 64)")
    parser.add_argument("--data_dir", type=str,
                        default=os.environ.get(
                            "DATA_DIR",
                            str(Path(__file__).resolve().parent.parent / "data"),
                        ),
                        help="数据集目录路径 (默认: <repo>/data)")
    parser.add_argument("--output_dir", type=str,
                        default=os.environ.get(
                            "SIMCOT_OUTPUT_DIR",
                            str(Path(__file__).resolve().parent / "results" / "adv"),
                        ),
                        help="输出结果目录 (默认: <SIM-CoT>/results/adv；可通过 SIMCOT_OUTPUT_DIR 覆盖)")
    parser.add_argument("--gcg_results_dir", type=str, default=None,
                        help="gcg_results 目录路径")
    parser.add_argument("--eval_correct_ids_from", type=str, default="results/org/test_org_results.json",
                        help="从指定的结果JSON文件读取correct_ids，只评估这些id的问题。")
    parser.add_argument("--eval_first_n", type=int, default=None,
                        help="评估前N个问题。")
    parser.add_argument("--decode_latent", action="store_true", default=False,
                        help="解码并保存推理向量 (latent tokens) 到结果文件中")
    parser.add_argument("--compute_latent_length", action="store_true", default=False,
                        help="统计推理向量的长度（欧几里得长度）的平均值")
    parser.add_argument("--save_latent_vectors", action="store_true", default=False,
                        help="保存完整推理向量到 .npz 文件")
    parser.add_argument("--replace_latent_vectors", type=str, default=None,
                        help="Path to .npz file containing replacement latent vectors. "
                             "When enabled, compute_latent_length and save_latent_vectors are automatically disabled.")

    args = parser.parse_args()

    # Auto-disable compute_latent_length and save_latent_vectors when replacement is enabled
    if args.replace_latent_vectors is not None:
        if args.compute_latent_length:
            print("Warning: compute_latent_length is disabled when replace_latent_vectors is set")
            args.compute_latent_length = False
        if args.save_latent_vectors:
            print("Warning: save_latent_vectors is disabled when replace_latent_vectors is set")
            args.save_latent_vectors = False

    #python test.py   --decode_latent --compute_latent_length --save_latent_vectors --gcg_results_dir ./adv_results/results_black
    #python test.py  --dataset gsm8k --replace_latent_vectors results/clean/latent_vectors_gsm8k.npz  --gcg_results_dir ./adv_results/results_black
    # 加载 gcg_results 前缀
    gcg_prefixes_dict = {}
    if args.gcg_results_dir is not None:
        for dataset_name in ["gsm8k", "MultiArith", "SVAMP"]:
            gcg_prefixes_dict[dataset_name] = load_gcg_prefixes(Path(args.gcg_results_dir), dataset_name)

    # 加载 correct_ids（模型能做对的问题ID）
    correct_ids_set = None
    if args.eval_correct_ids_from is not None:
        correct_ids_set = load_correct_ids_from_results(args.eval_correct_ids_from)

    # 数据目录
    data_dir = Path(args.data_dir)

    # 定义要评估的数据集
    datasets = {
        "gsm8k": data_dir / "gsm8k.json",
        "MultiArith": data_dir / "MultiArith.json",
        "SVAMP": data_dir / "SVAMP.json",
    }

    # 如果指定了dataset，过滤只评估该数据集
    if args.dataset is not None:
        if args.dataset not in datasets:
            logging.warning(f"Dataset '{args.dataset}' not found. Available: {list(datasets.keys())}")
            sys.exit(1)
        datasets = {args.dataset: datasets[args.dataset]}

    # 存储所有结果
    all_results = {}
    all_samples = {}

    # 输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 对每个数据集进行评估
    for dataset_name, dataset_path in datasets.items():
        if not dataset_path.exists():
            logging.warning(f"Dataset not found: {dataset_path}")
            continue

        logging.warning(f"\n{'='*50}")
        logging.warning(f"Evaluating on {dataset_name} dataset (using attack_black.py method)")

        # 确定要评估的问题ID
        eval_ids = None
        if args.question_id is not None:
            # 优先使用指定的单个 question_id
            eval_ids = [args.question_id]
            logging.warning(f"Evaluating question ID: {args.question_id}")
        elif args.eval_first_n is not None and correct_ids_set is not None and dataset_name in correct_ids_set:
            # 从 correct_ids 中取前 N 个
            all_correct_ids = list(correct_ids_set[dataset_name])
            eval_ids = all_correct_ids[:args.eval_first_n]
            logging.warning(f"Evaluating first {args.eval_first_n} of {len(all_correct_ids)} correct_ids")
        elif args.eval_first_n is not None:
            # 评估前 N 个问题（从数据集）
            eval_ids = list(range(args.eval_first_n))
            logging.warning(f"Evaluating first {args.eval_first_n} questions from dataset")
        elif correct_ids_set is not None and dataset_name in correct_ids_set:
            # 评估 correct_ids 中的所有问题
            eval_ids = list(correct_ids_set[dataset_name])
            logging.warning(f"Evaluating {len(eval_ids)} correct_ids from previous run")

        if eval_ids:
            logging.warning(f"Total questions to evaluate: {len(eval_ids)}")
        logging.warning(f"{'='*50}")

        data_args = DataArguments(
            data_name=dataset_name,
            data_path=str(dataset_path),
        )

        # Get gcg_prefixes for this dataset
        gcg_prefixes = gcg_prefixes_dict.get(dataset_name) if gcg_prefixes_dict else None

        # 运行评估
        accuracy, samples, avg_latent_len, latent_vectors = evaluation(
            checkpoint_path=args.checkpoint,
            model_id=args.model_id,
            data_args=data_args,
            max_new_tokens=args.max_new_tokens,
            question_ids=eval_ids,
            gcg_prefixes=gcg_prefixes,
            decode_latent=args.decode_latent,
            compute_latent_length=args.compute_latent_length,
            save_latent_vectors=args.save_latent_vectors,
            replace_latent_vectors=args.replace_latent_vectors
        )

        # 统计正确样本的ID
        correct_ids = [s['id'] for s in samples if s['correct']]

        # 保存完整推理向量到 .npz 文件
        if args.save_latent_vectors and latent_vectors is not None:
            npz_data = {f"{dataset_name}_{qid}": np.array(vecs) for qid, vecs in latent_vectors.items() if vecs}
            if npz_data:
                np.savez(output_dir / f"latent_vectors_{dataset_name}.npz", **npz_data)
                print(f"Saved latent vectors to {output_dir / f'latent_vectors_{dataset_name}.npz'}")

        # 保存结果
        all_results[dataset_name] = {
            "accuracy": accuracy,
            "num_samples": len(samples),
            "num_correct": len(correct_ids),
            "correct_ids": correct_ids,
        }
        if args.compute_latent_length and avg_latent_len is not None:
            all_results[dataset_name]["avg_latent_vector_length"] = avg_latent_len
        all_samples[dataset_name] = samples

    # 保存结果到文件
    final_output = {
        "summary": {},
        "samples": {}
    }
    for dataset_name, result in all_results.items():
        summary_entry = {
            "accuracy": result["accuracy"],
            "num_samples": result["num_samples"],
            "num_correct": result["num_correct"],
            "correct_ids": result["correct_ids"]
        }
        if "avg_latent_vector_length" in result:
            summary_entry["avg_latent_vector_length"] = result["avg_latent_vector_length"]
        final_output["summary"][dataset_name] = summary_entry
        final_output["samples"][dataset_name] = all_samples[dataset_name]

    output_file = output_dir / "test_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(final_output, f, indent=2, ensure_ascii=False)

    # 打印最终结果
    logging.warning(f"\n{'='*50}")
    logging.warning("Final Results:")
    for dataset_name, result in all_results.items():
        logging.warning(f"  {dataset_name}: {result['accuracy']:.4f} ({result['num_samples']} samples, "
                       f"{result['num_correct']} correct)")
    logging.warning(f"\nResults saved to {output_file}")
