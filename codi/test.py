#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

"""
CODI 模型在多个数学推理数据集上的测试脚本


功能：
1. 加载本地 JSON 格式的数据集（gsm8k, MultiArith, SVAMP）
2. 在多个数据集上进行评估
3. 支持从 gcg_results 目录加载前缀，添加到问题前
4. 支持评估模型之前做对的问题（correct_ids）
5. 支持评估前 N 个问题（可单独使用，也可与 --eval_correct_ids_from 组合）
6. 可以解码推理 latent 向量

使用方法：
    python test_org.py                                    # 评估所有数据集的所有问题
    python test_org.py --dataset gsm8k                   # 只评估 gsm8k 数据集
    python test_org.py --question_id 0                   # 评估所有数据集的第0个问题
    python test_org.py --dataset gsm8k --question_id 0  # 只评估 gsm8k 数据集的第0个问题
    python test_org.py --gcg_results_dir ./gcg_results   # 使用 gcg_results 中的前缀
    python test_org.py --gcg_results_dir ./gcg_results --dataset gsm8k --question_id 0
    python test_org.py --eval_correct_ids_from ./test_org_results.json  # 评估之前做对的问题
    python test_org.py --eval_first_n 10                # 评估前10个问题
    python test_org.py --eval_correct_ids_from ./test_org_results.json --eval_first_n 30  # 评估之前做对的前30个题
"""

import logging
import math
import re
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence
from pathlib import Path

import torch
import transformers
from torch.nn import functional as F
import json

from peft import PeftModel, LoraConfig, TaskType, get_peft_model
from peft import PeftModel
from datasets import load_dataset, concatenate_datasets
from accelerate.utils import set_seed
from safetensors.torch import load_file

import numpy as np

# 从model.py导入CODI模型和相关配置类
from codi.src.model import (
    CODI,
    ModelArguments,
    DataArguments,
    TrainingArguments,
)

# 是否打印每个问题的详细信息
do_print = False

# 设置设备：优先使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


def load_local_dataset(data_path):
    """
    从本地 JSON 文件加载数据集

    参数：
    - data_path: JSON 文件路径

    返回：
    - 问题列表和答案列表
    """
    questions = []
    answers = []

    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for item in data:
        questions.append(item['question'])
        # 处理答案：直接使用 answer 字段
        answer_text = item['answer']
        # 处理数字中的逗号
        answer_text = answer_text.replace(',', '')
        # 转换为浮点数
        try:
            ans = float(answer_text)
        except ValueError:
            ans = float("inf")

        answers.append(ans)

    return questions, answers


def evaluation(model_args, data_args, training_args, use_chat_template=False, cot_prompt=None,
               question_ids=None, gcg_prefixes=None, decode_latent=False, compute_latent_length=False,
               save_latent_vectors=False, replace_latent_vectors=None):
    """
    CODI模型评估函数 - 本地数据版本

    评估流程：
    1. 配置并初始化LoRA
    2. 加载预训练模型和LoRA权重
    3. 加载本地评估数据集
    4. 对每个问题进行推理
    5. 计算准确率

    参数：
    - use_chat_template: 是否使用模型的对话模板
    - cot_prompt: CoT提示词，会添加到问题前面引导模型进行推理
    - question_ids: 可选，指定评估的问题索引列表。如果为None，则评估所有问题。
    - gcg_prefixes: 可选，dict of question_id -> prefix string，用于在问题前添加前缀。
    - decode_latent: 是否解码推理向量
    """
    # =========================================================================
    # 步骤1: 配置LoRA参数
    # =========================================================================
    if model_args.lora_init:
        task_type = TaskType.CAUSAL_LM

        # 根据模型架构选择不同的目标模块
        if any(name in model_args.model_name_or_path.lower() for name in ["llama", "mistral", "falcon", "qwen"]):
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"]
        elif any(name in model_args.model_name_or_path.lower() for name in ["phi"]):
            target_modules = ["q_proj", "k_proj", "v_proj", "dense", "fc1", "fc2"]
        elif any(name in model_args.model_name_or_path.lower() for name in ["gpt2"]):
            target_modules = ["c_attn", "c_proj", 'c_fc']
        else:
            raise ValueError(f"Only support LLAMA, Mistral, Falcon, Phi-2, but got {model_args.model_name_or_path}.")

        lora_config = LoraConfig(
            task_type=task_type,
            inference_mode=False,
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            lora_dropout=0.1,
            target_modules=target_modules,
            init_lora_weights=True,
        )
    else:
        raise NotImplementedError

    # =========================================================================
    # 步骤2: 初始化CODI模型并加载权重
    # =========================================================================
    model = CODI(model_args, training_args, lora_config)

    # 尝试加载safetensors格式的权重，如果失败则加载pytorch格式
    try:
        state_dict = load_file(os.path.join(model_args.ckpt_dir, "model.safetensors"))
    except Exception:
        state_dict = torch.load(os.path.join(model_args.ckpt_dir, "pytorch_model.bin"))

    # 加载权重到模型
    model.load_state_dict(state_dict, strict=False)
    # 共享权重
    model.codi.tie_weights()

    # =========================================================================
    # 步骤3: 加载Tokenizer
    # =========================================================================
    tokenizer_path = model_args.model_name_or_path
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        tokenizer_path,
        token=model_args.token,
        model_max_length=training_args.model_max_length,
        padding_side="left",
        use_fast=False,
    )

    # 设置pad_token
    if tokenizer.pad_token_id is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        tokenizer.pad_token_id = model.pad_token_id
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids('[PAD]')

    # 将模型移到GPU并转换为bfloat16
    device = "cuda"
    model = model.to('cuda')
    model.to(torch.bfloat16)

    # =========================================================================
    # 步骤4: 加载本地数据集
    # =========================================================================
    logging.warning(f"Loading local dataset from: {data_args.data_path}")

    all_questions, answer = load_local_dataset(data_args.data_path)

    # 如果指定了 question_ids，只评估这些id
    eval_indices = list(question_ids) if question_ids is not None else range(len(all_questions))
    questions = [all_questions[i] for i in eval_indices]

    print(f"Loaded {len(all_questions)} questions from local dataset")
    print(f"Sample question: {all_questions[0][:100]}...")
    print(f"Will evaluate {len(questions)} questions")

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

    # =========================================================================
    # 步骤5: Tokenize输入数据
    # =========================================================================
    logging.warning("Tokenizing inputs...")

    # 检查是否使用对话模板
    if use_chat_template:
        logging.warning("Using chat template for model input")
        # 检查 tokenizer 是否有 chat_template
        if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template is not None:
            pass  # tokenizer 已配置好 chat_template
        else:
            logging.warning("Tokenizer does not have chat_template, using raw input")

    # 计算评估批次数
    eval_step = math.ceil(len(questions) / data_args.batch_size)
    logging.warning(f"Total example: {len(questions)} | eval batch size: {data_args.batch_size}"
                    f"eval steps: {eval_step}")

    # 对每个问题添加特殊token
    question_data = []
    for i in range(eval_step):
        # 获取当前批次的问题
        batch_questions = questions[i*data_args.batch_size: (i+1)*data_args.batch_size] if i < eval_step - 1 else questions[i*data_args.batch_size:]

        # 获取对应的原始索引
        batch_indices = eval_indices[i*data_args.batch_size: (i+1)*data_args.batch_size] if i < eval_step - 1 else eval_indices[i*data_args.batch_size:]

        # 处理问题：添加 cot_prompt 如果指定了
        if cot_prompt:
            batch_questions = [cot_prompt + q for q in batch_questions]

        # 如果有 gcg_prefix，为问题添加前缀
        if gcg_prefixes:
            for idx_pos, orig_idx in enumerate(batch_indices):
                if orig_idx in gcg_prefixes:
                    prefix = gcg_prefixes[orig_idx]
                    batch_questions[idx_pos] = prefix + batch_questions[idx_pos]
                    if do_print:
                        print(f"Added prefix for question {orig_idx}: '{prefix}'")

        # 使用对话模板或直接 tokenize
        if use_chat_template and hasattr(tokenizer, 'chat_template') and tokenizer.chat_template is not None:
            # 使用对话模板
            messages = [{"role": "user", "content": q} for q in batch_questions]
            # 应用 chat template，添加 generation prompt
            chat_result = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                padding="longest",
                padding_side="left",
            )
            # 确保格式正确
            if isinstance(chat_result, torch.Tensor):
                input_ids = chat_result
                attention_mask = torch.ones_like(input_ids)
                batch = {"input_ids": input_ids, "attention_mask": attention_mask}
            else:
                batch = {k: v for k, v in chat_result.items()}
            batch = {k: v.to(device) for k, v in batch.items()}
        else:
            # 直接 tokenize（原始模式）
            batch = tokenizer(
                batch_questions,
                return_tensors="pt",
                padding="longest",
            )
            batch = {k: v.to(device) for k, v in batch.items()}

        # 添加bot (beginning of thought) token
        if training_args.remove_eos:
            bot_tensor = torch.tensor([model.bot_id], dtype=torch.long).expand(batch["input_ids"].size(0), 1)
        else:
            bot_tensor = torch.tensor([tokenizer.eos_token_id, model.bot_id], dtype=torch.long).expand(batch["input_ids"].size(0), 2)

        # 确保 bot_tensor 在 GPU 上
        bot_tensor = bot_tensor.to(device)

        # 拼接input_ids和bot_tensor
        batch["input_ids"] = torch.cat((batch["input_ids"], bot_tensor), dim=1)
        # 更新attention mask
        batch["attention_mask"] = torch.cat((batch["attention_mask"], torch.ones_like(bot_tensor)), dim=1)
        # 记录输入长度
        batch['input_len'] = len(batch['input_ids'][0])
        # 移到GPU
        question_data.append(batch)

    # =========================================================================
    # 步骤6: 设置生成参数并开始推理
    # =========================================================================
    model.eval()

    # 生成配置
    gen_kwargs = {
        "max_new_tokens": 256,
        "temperature": 0.1,
        "top_k": 40,
        "top_p": 0.95,
        "do_sample": True,
    }

    # latent decode 配置
    probe_topk = 5

    # 初始化结果存储
    ans_pred_list = []
    len_cot = []
    results = []
    # 存储每个推理向量的长度
    all_latent_lengths = [] if compute_latent_length else None
    # 存储每个question id对应的所有向量长度
    question_latent_lengths = {idx: [] for idx in eval_indices} if compute_latent_length else None
    # 存储完整的推理向量
    question_latent_vectors = {idx: [] for idx in eval_indices} if save_latent_vectors else None

    # =========================================================================
    # 步骤7: 对每个批次进行推理
    # =========================================================================
    for step, batch in enumerate(question_data):
        batch_size = batch["input_ids"].size(0)
        batch_latent_decoded = [] if decode_latent else None

        with torch.no_grad():
            # 编码问题，获取初始latent embedding
            past_key_values = None

            outputs = model.codi(
                input_ids=batch["input_ids"],
                use_cache=True,
                output_hidden_states=True,
                past_key_values=past_key_values,
                attention_mask=batch["attention_mask"]
            )

            past_key_values = outputs.past_key_values
            latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)

            # 替换初始推理向量 (如果启用)
            if replacement_vectors is not None:
                for b in range(batch["input_ids"].size(0)):
                    qid = eval_indices[step * data_args.batch_size + b]
                    if qid in replacement_vectors:
                        latent_embd[b, 0, :] = torch.from_numpy(replacement_vectors[qid][0]).to(device).to(torch.bfloat16)

            # 统计推理向量长度 (如果启用)
            if compute_latent_length:
                for b in range(batch["input_ids"].size(0)):
                    qid = eval_indices[step * data_args.batch_size + b]
                    vec = latent_embd[b, 0, :]  # (hidden_dim,)
                    length = torch.sqrt(torch.dot(vec, vec)).item()
                    question_latent_lengths[qid].append(length)

            # 保存完整推理向量 (如果启用)
            if save_latent_vectors:
                for b in range(batch["input_ids"].size(0)):
                    qid = eval_indices[step * data_args.batch_size + b]
                    question_latent_vectors[qid].append(latent_embd[b, 0, :].clone().float().cpu().numpy())

            # 解码初始 latent token (如果启用)
            if decode_latent:
                probs = torch.nn.functional.softmax(model.codi.lm_head(latent_embd), dim=-1)
                top5_values, top5_indices = torch.topk(probs, k=probe_topk, dim=2)
                batch_latent_decoded.append(top5_indices[0, 0].tolist())

            if training_args.use_prj:
                latent_embd = model.prj(latent_embd)

            # 执行隐式推理步骤
            inf_latent_iterations = training_args.inf_latent_iterations
            for i in range(inf_latent_iterations):
                outputs = model.codi(
                    inputs_embeds=latent_embd,
                    use_cache=True,
                    output_hidden_states=True,
                    past_key_values=past_key_values
                )
                past_key_values = outputs.past_key_values
                latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)

                # 替换推理向量 (如果启用)
                if replacement_vectors is not None:
                    for b in range(batch["input_ids"].size(0)):
                        qid = eval_indices[step * data_args.batch_size + b]
                        if qid in replacement_vectors:
                            # 第 i 个推理步骤对应 replacement_vectors 中的第 i+1 个向量
                            vec_idx = i + 1
                            if vec_idx < replacement_vectors[qid].shape[0]:
                                latent_embd[b, 0, :] = torch.from_numpy(replacement_vectors[qid][vec_idx]).to(device).to(torch.bfloat16)

                # 统计推理向量长度 (如果启用)
                if compute_latent_length:
                    for b in range(batch["input_ids"].size(0)):
                        qid = eval_indices[step * data_args.batch_size + b]
                        vec = latent_embd[b, 0, :]  # (hidden_dim,)
                        length = torch.sqrt(torch.dot(vec, vec)).item()
                        question_latent_lengths[qid].append(length)

                # 保存完整推理向量 (如果启用)
                if save_latent_vectors:
                    for b in range(batch["input_ids"].size(0)):
                        qid = eval_indices[step * data_args.batch_size + b]
                        question_latent_vectors[qid].append(latent_embd[b, 0, :].clone().float().cpu().numpy())

                # 解码 latent token (如果启用)
                if decode_latent:
                    probs = torch.nn.functional.softmax(model.codi.lm_head(latent_embd), dim=-1)
                    top5_values, top5_indices = torch.topk(probs, k=probe_topk, dim=2)
                    batch_latent_decoded.append(top5_indices[0, 0].tolist())


                if training_args.use_prj:
                    latent_embd = model.prj(latent_embd)

            # 添加EOT token
            if training_args.remove_eos:
                eot_emb = model.get_embd(model.codi, model.model_name)(
                    torch.tensor([model.eot_id], dtype=torch.long, device='cuda')
                ).unsqueeze(0).to(device)
            else:
                eot_emb = model.get_embd(model.codi, model.model_name)(
                    torch.tensor([model.eot_id, tokenizer.eos_token_id], dtype=torch.long, device='cuda')
                ).unsqueeze(0).to(device)

            eot_emb = eot_emb.expand(batch["input_ids"].size(0), -1, -1)
            output = eot_emb

            # 自回归生成答案
            seq_len = 0
            finished = torch.zeros(batch_size, dtype=torch.bool, device="cuda")
            pred_tokens = [[] for _ in range(batch_size)]

            for i in range(gen_kwargs["max_new_tokens"]):
                seq_len += 1

                out = model.codi(
                    inputs_embeds=output,
                    output_hidden_states=False,
                    attention_mask=None,
                    use_cache=True,
                    output_attentions=False,
                    past_key_values=past_key_values
                )
                past_key_values = out.past_key_values
                logits = out.logits[:, -1, :model.codi.config.vocab_size-1]

                # 采样策略
                if training_args.greedy:
                    next_token_ids = torch.argmax(logits, dim=-1)
                    if next_token_ids.dim() == 0:
                        next_token_ids = next_token_ids.unsqueeze(0)
                else:
                    logits /= gen_kwargs["temperature"]

                    if gen_kwargs["top_k"] > 1:
                        top_k_values, _ = torch.topk(logits, gen_kwargs["top_k"], dim=-1)
                        min_top_k_value = top_k_values[:, -1].unsqueeze(-1)
                        logits[logits < min_top_k_value] = -float("inf")

                    if gen_kwargs["top_p"] < 1.0:
                        sorted_logit, sorted_indices = torch.sort(logits, descending=True, dim=-1)
                        cumulative_probs = torch.cumsum(F.softmax(sorted_logit, dim=-1), dim=-1)

                        sorted_indices_to_remove = cumulative_probs > gen_kwargs["top_p"]
                        if sorted_indices_to_remove.any():
                            sorted_indices_to_remove = sorted_indices_to_remove.roll(1, dims=-1)
                            sorted_indices_to_remove[:, 0] = False

                        for b in range(logits.size(0)):
                            logits[b, sorted_indices[b, sorted_indices_to_remove[b]]] = -float("inf")

                    probs = F.softmax(logits, dim=-1)
                    next_token_ids = torch.multinomial(probs, num_samples=1)
                    if next_token_ids.dim() > 1:
                        next_token_ids = next_token_ids.squeeze(-1)
                    if next_token_ids.dim() == 0:
                        next_token_ids = next_token_ids.unsqueeze(0)

                # 处理EOS token
                for b in range(batch_size):
                    if not finished[b]:
                        pred_tokens[b].append(next_token_ids[b].item())
                        if next_token_ids[b] == tokenizer.eos_token_id:
                            finished[b] = True

                if finished.all():
                    break

                output = model.get_embd(model.codi, model.model_name)(next_token_ids).unsqueeze(1).to(device)

            # 解码并提取答案
            for mini_step, pred_token in enumerate(pred_tokens):
                orig_idx = eval_indices[step * data_args.batch_size + mini_step]
                len_cot.append(len(pred_token))
                decoded_pred = tokenizer.decode(pred_token, skip_special_tokens=True)
                pred_answer = extract_answer_number(decoded_pred)
                gold_answer = answer[orig_idx]

                if do_print:
                    print(f"Question {orig_idx} Starts...")
                    print(f"Q: {all_questions[orig_idx]}")
                    print(decoded_pred)
                    print(f"Question {orig_idx} Ends")
                    print(f"Prediction={pred_answer}; Groundtruth={gold_answer}")
                    print("")

                # 保存结果
                # 构建完整问题（包含前缀）
                full_question = all_questions[orig_idx]
                if gcg_prefixes and orig_idx in gcg_prefixes:
                    full_question = gcg_prefixes[orig_idx] + full_question
                if cot_prompt:
                    full_question = cot_prompt + full_question

                sample_result = {
                    'id': orig_idx,
                    'question': full_question,
                    'ground_truth': gold_answer,
                    'prediction': pred_answer,
                    'model_output': decoded_pred,
                    'correct': pred_answer == gold_answer
                }
                if decode_latent and batch_latent_decoded is not None:
                    # 解码 latent tokens 为实际文本
                    latent_decoded_texts = []
                    for token_ids in batch_latent_decoded:
                        latent_decoded_texts.append([tokenizer.decode(tid) for tid in token_ids])
                    sample_result['latent_tokens_decoded'] = latent_decoded_texts
                if compute_latent_length and question_latent_lengths is not None:
                    sample_result['latent_vector_lengths'] = question_latent_lengths[orig_idx]

                results.append(sample_result)

                ans_pred_list.append(pred_answer)

    # =========================================================================
    # 步骤8: 计算准确率
    # =========================================================================
    eval_answers = [answer[i] for i in eval_indices]
    accuracy = compute_accuracy(eval_answers, ans_pred_list)

    # 打印结果
    print(f"adapter: {model_args.adapter_name_or_path} | Dataset: {data_args.data_name} | "
          f"Accuracy: {100*accuracy:.2f}% | ")
    print(f"average length of COT: {sum(len_cot)/len(len_cot)}")

    # 统计推理向量长度 (如果启用)
    avg_latent_length = None
    if compute_latent_length and question_latent_lengths:
        for qid in eval_indices:
            lengths = question_latent_lengths[qid]
            if lengths:
                avg_length = sum(lengths) / len(lengths)
                all_latent_lengths.append(avg_length)
        if all_latent_lengths:
            avg_latent_length = sum(all_latent_lengths)/len(all_latent_lengths)
            print(f"average latent vector length: {avg_latent_length}")

    return accuracy, results, avg_latent_length, question_latent_vectors


def extract_answer_number(sentence: str) -> float:
    """
    从生成的文本中提取答案数字
    """
    sentence = sentence.replace(',', '')
    pred = [s for s in re.findall(r'-?\d+\.?\d*', sentence)]

    if not pred:
        return float('inf')

    pred_answer = float(pred[-1])
    return pred_answer


def compute_accuracy(gold: list, pred: list) -> float:
    """
    计算预测准确率
    """
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
    """
    数据配置参数
    """
    data_name: str = None
    data_path: str = None
    batch_size: int = 1
    output_path: str = './test_results.jsonl'


def load_gcg_prefixes(gcg_dir, dataset_name):
    """
    从 gcg_results 目录加载指定数据集的前缀

    参数：
    - gcg_dir: gcg_results 根目录
    - dataset_name: 数据集名称 (gsm8k, MultiArith, SVAMP)

    返回：
    - dict of question_id -> prefix string
    """
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
                        prefixes[qid] = (data['all_results'][:1] + [i for i in data['all_results'] if i['attack_success'] == True])[-1]['prefix']
        print(f"Loaded {len(prefixes)} gcg prefixes from {dataset_dir}")
    else:
        logging.warning(f"gcg_results subdir not found: {dataset_dir}")

    return prefixes


# =========================================================================
# 主程序入口
# =========================================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='CODI Multi-Dataset Test')
    parser.add_argument('--dataset', type=str, default=None,
                        help='指定要评估的数据集名称 (gsm8k, MultiArith, SVAMP)。如果为空，则评估所有数据集。')
    parser.add_argument('--question_id', type=int, default=None,
                        help='指定要评估的问题索引。如果为空，则评估所有问题（可配合其他参数使用）。')
    parser.add_argument('--data_path', type=str, default=None,
                        help='Path to local JSON dataset (not used in multi-dataset mode)')
    parser.add_argument('--output_path', type=str, default='./test_results.jsonl',
                        help='Path to save test results')
    parser.add_argument('--model_name_or_path', type=str, default='meta-llama/Llama-3.2-1B-Instruct',
                        help='Path to pretrained model')
    parser.add_argument('--ckpt_dir', type=str, default='codi/codi_llama1b',
                        help='Path to checkpoint directory')
    parser.add_argument('--lora_r', type=int, default=128,
                        help='LoRA rank')
    parser.add_argument('--lora_alpha', type=int, default=32,
                        help='LoRA alpha')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size')
    parser.add_argument('--inf_latent_iterations', type=int, default=6,
                        help='Number of latent iterations')
    parser.add_argument('--model_max_length', type=int, default=512,
                        help='Model max length')
    parser.add_argument('--greedy', type=lambda x: x.lower() == 'true', default=True,
                        help='Use greedy decoding')
    parser.add_argument('--remove_eos', type=lambda x: x.lower() == 'true', default=True,
                        help='Remove EOS token')
    parser.add_argument('--use_prj', type=lambda x: x.lower() == 'true', default=True,
                        help='Use projection layer')
    parser.add_argument('--use_lora', type=lambda x: x.lower() == 'true', default=True,
                        help='Use LoRA')
    parser.add_argument('--num_latent', type=int, default=6,
                        help='Number of latent thoughts')
    parser.add_argument('--prj_dim', type=int, default=2048,
                        help='Projection layer dimension')
    parser.add_argument('--inf_num_iterations', type=int, default=1,
                        help='Number of evaluation iterations')
    parser.add_argument('--use_chat_template', type=lambda x: x.lower() == 'true', default=False,
                        help='Use chat template for model input')
    parser.add_argument('--cot_prompt', type=str, default=None,
                        help='CoT prompt to prepend to question (e.g., "Let\'s think step by step.")')
    parser.add_argument('--gcg_results_dir', type=str, default=None,
                        help='gcg_results 目录路径，包含三个子目录 (gsm8k, MultiArith, SVAMP)，分别存放各数据集的 prefix JSON 文件。')
    parser.add_argument('--eval_correct_ids_from', type=str, default="results/org/test_org_results.json",
                        help='从指定的结果JSON文件读取correct_ids，只评估这些id的问题。')
    parser.add_argument('--eval_first_n', type=int, default=None,
                        help='评估前N个问题。')
    parser.add_argument('--output_dir', type=str,
                        default=os.environ.get(
                            "CODI_OUTPUT_DIR",
                            str(Path(__file__).resolve().parent / "results" / "adv"),
                        ),
                        help='输出结果目录 (默认: <codi>/results/adv；可通过 CODI_OUTPUT_DIR 环境变量覆盖)')
    parser.add_argument('--decode_latent', action='store_true', default=False,
                        help='解码并保存推理向量 (latent tokens) 到结果文件中')
    parser.add_argument('--compute_latent_length', action='store_true', default=False,
                        help='统计推理向量的长度（欧几里得长度）的平均值')
    parser.add_argument('--save_latent_vectors', action='store_true', default=False,
                        help='保存完整推理向量到 .npz 文件')
    parser.add_argument('--replace_latent_vectors', type=str, default=None,
                        help='Path to .npz file containing replacement latent vectors. '
                             'When enabled, compute_latent_length and save_latent_vectors are automatically disabled.')

    args = parser.parse_args()

    # Auto-disable compute_latent_length and save_latent_vectors when replacement is enabled
    if args.replace_latent_vectors is not None:
        if args.compute_latent_length:
            print("Warning: compute_latent_length is disabled when replace_latent_vectors is set")
            args.compute_latent_length = False
        if args.save_latent_vectors:
            print("Warning: save_latent_vectors is disabled when replace_latent_vectors is set")
            args.save_latent_vectors = False
    #python test.py --decode_latent --compute_latent_length  --save_latent_vectors --gcg_results_dir adv_results/results_white_3
    #python test.py --dataset gsm8k --replace_latent_vectors results/clean/latent_vectors_gsm8k.npz  --gcg_results_dir adv_results/results_black
    # 创建配置对象
    model_args = ModelArguments(
        model_name_or_path=args.model_name_or_path,
        ckpt_dir=args.ckpt_dir,
        lora_init=args.use_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        adapter_name_or_path=args.ckpt_dir,
    )

    training_args = TrainingArguments(
        per_device_eval_batch_size=args.batch_size,
        model_max_length=args.model_max_length,
        num_latent=args.num_latent,
        inf_latent_iterations=args.inf_latent_iterations,
        use_prj=args.use_prj,
        prj_dim=args.prj_dim,
        greedy=args.greedy,
        remove_eos=args.remove_eos,
        inf_num_iterations=args.inf_num_iterations,
        output_dir="",
    )

    # 数据目录默认指向仓库根目录的 data/；可通过环境变量 DATA_DIR 覆盖
    data_dir = Path(os.environ.get("DATA_DIR", str(Path(__file__).resolve().parent.parent / "data")))

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

    # 加载 correct_ids（模型能做对的问题ID）
    correct_ids_set = None
    if args.eval_correct_ids_from is not None:
        correct_ids_file = Path(args.eval_correct_ids_from)
        if correct_ids_file.exists():
            with open(correct_ids_file, 'r', encoding='utf-8') as f:
                correct_ids_data = json.load(f)
            correct_ids_set = {}
            for dataset_name in correct_ids_data:
                correct_ids_set[dataset_name] = correct_ids_data[dataset_name].get("correct_ids", [])
            print(f"Loaded correct_ids from {correct_ids_file}: { {k: len(v) for k, v in correct_ids_set.items()} }")
        else:
            logging.warning(f"correct_ids file not found: {correct_ids_file}")

    # 存储所有结果
    all_results = {}
    all_samples = {}

    # 保存结果到文件
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 对每个数据集进行评估
    for dataset_name, dataset_path in datasets.items():
        if not dataset_path.exists():
            logging.warning(f"Dataset not found: {dataset_path}")
            continue

        logging.warning(f"\n{'='*50}")
        logging.warning(f"Evaluating on {dataset_name} dataset")

        # 为当前数据集加载对应的 gcg_prefixes
        gcg_prefixes = None
        if args.gcg_results_dir is not None:
            gcg_prefixes = load_gcg_prefixes(Path(args.gcg_results_dir), dataset_name)

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

        if eval_ids is not None:
            logging.warning(f"Total questions to evaluate: {len(eval_ids)}")
        logging.warning(f"{'='*50}")

        data_args = DataArguments(
            data_name=dataset_name,
            data_path=str(dataset_path),
            batch_size=args.batch_size,
            output_path=args.output_path,
        )

        # 运行评估
        accuracy, samples, avg_latent_len, latent_vectors = evaluation(
            model_args, data_args, training_args,
            use_chat_template=args.use_chat_template,
            cot_prompt=args.cot_prompt,
            question_ids=eval_ids,
            gcg_prefixes=gcg_prefixes if gcg_prefixes else None,
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
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    '''output_file = output_dir / "test_org_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)'''

    # 保存详细结果
    '''
    output_file_samples = output_dir / "test_org_results_samples.json"
    with open(output_file_samples, 'w', encoding='utf-8') as f:
        json.dump(all_samples, f, indent=2, ensure_ascii=False)'''

    # 保存完整结果（summary + all samples in one file）
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

    output_file_final = output_dir / "test_results.json"
    with open(output_file_final, 'w', encoding='utf-8') as f:
        json.dump(final_output, f, indent=2, ensure_ascii=False)

    # 打印最终结果
    logging.warning(f"\n{'='*50}")
    logging.warning("Final Results:")
    for dataset_name, result in all_results.items():
        logging.warning(f"  {dataset_name}: {result['accuracy']:.4f} ({result['num_samples']} samples, "
                       f"{result['num_correct']} correct)")
