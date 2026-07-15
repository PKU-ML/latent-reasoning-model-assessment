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
CODI 模型推理/测试脚本

功能：
1. 加载训练好的CODI模型（包括LoRA权重）
2. 在多个数学推理数据集上进行评估（GSM8K, GSM-Hard, MultiArith, SVAMP, CommonsenseQA等）
3. 支持多种解码策略（greedy, top-k, top-p）
4. 计算预测准确率和平均输出长度

推理流程：
1. 对问题进行编码，获取初始latent embedding
2. 迭代执行隐式推理步骤（num_latent次）
3. 添加EOT token作为答案开始标记
4. 自回归生成最终答案
5. 从生成文本中提取答案数字
6. 计算准确率
"""

import logging
import math
import re
import os
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

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
from src.model import (
    CODI,
    ModelArguments,
    DataArguments,
    TrainingArguments,
)

# 是否打印每个问题的详细信息
do_print = True

# 设置设备：优先使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


def evaluation(model_args, data_args, training_args):
    """
    CODI模型评估函数

    评估流程：
    1. 配置并初始化LoRA
    2. 加载预训练模型和LoRA权重
    3. 加载评估数据集
    4. 对每个问题进行推理
    5. 计算准确率

    参数：
    - model_args: 模型配置参数
    - data_args: 数据配置参数
    - training_args: 训练配置参数

    返回：
    - 准确率（百分比）
    """
    # =========================================================================
    # 步骤1: 配置LoRA参数
    # =========================================================================
    if model_args.lora_init:
        task_type = TaskType.CAUSAL_LM

        # 根据模型架构选择不同的目标模块
        if any(name in model_args.model_name_or_path.lower() for name in ["llama", "mistral", "falcon", "qwen"]):
            # LLaMA/Mistral/Qwen等架构的attention模块
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"]
        elif any(name in model_args.model_name_or_path.lower() for name in ["phi"]):
            # Phi-2模型架构
            target_modules = ["q_proj", "k_proj", "v_proj", "dense", "fc1", "fc2"]
        elif any(name in model_args.model_name_or_path.lower() for name in ["gpt2"]):
            # GPT-2架构
            target_modules = ["c_attn", "c_proj", 'c_fc']
        else:
            raise ValueError(f"Only support LLAMA, Mistral, Falcon, Phi-2, but got {model_args.model_name_or_path}.")

        # 创建LoRA配置
        lora_config = LoraConfig(
            task_type=task_type,
            inference_mode=False,
            r=model_args.lora_r,                   # LoRA rank
            lora_alpha=model_args.lora_alpha,      # LoRA alpha缩放参数
            lora_dropout=0.1,                      # Dropout比率
            target_modules=target_modules,         # 目标模块
            init_lora_weights=True,                # 初始化LoRA权重
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
    # 共享权重（优化推理效率）
    model.codi.tie_weights()

    # =========================================================================
    # 步骤3: 加载Tokenizer
    # =========================================================================
    tokenizer_path = model_args.model_name_or_path
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        tokenizer_path,
        token=model_args.token,
        model_max_length=training_args.model_max_length,
        padding_side="left",  # 生成时使用左padding
        use_fast=False,
    )

    # 设置pad_token
    if tokenizer.pad_token_id is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        tokenizer.pad_token_id = model.pad_token_id
        if tokenizer.pad_token_id is None:  # 错误处理
            tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids('[PAD]')

    # 将模型移到GPU并转换为bfloat16
    device = "cuda"
    model = model.to('cuda')
    model.to(torch.bfloat16)

    # =========================================================================
    # 步骤4: 加载评估数据集
    # =========================================================================
    logging.warning("Downloading Data")

    # 根据数据集名称确定列名
    question_name = "question"
    answer_name = "answer"

    # 支持多个评估数据集
    if "gsm-hard" == data_args.data_name:
        # GSM-Hard数据集
        dataset = load_dataset("juyoung-trl/gsm-hard")
        test_set = dataset['train']
        question_name = "instruction"
        answer_name = "response"
    elif "multi-arith" == data_args.data_name:
        # MultiArith数据集
        dataset = load_dataset("ChilleD/MultiArith")
        test_set = dataset['test']
        answer_name = "final_ans"
    elif "svamp" == data_args.data_name:
        # SVAMP数据集
        dataset = load_dataset("ChilleD/SVAMP")
        test_set = concatenate_datasets([dataset["train"], dataset["test"]])
        question_name = "question_concat"
        answer_name = "Answer"
    elif "commonsense" == data_args.data_name:
        # CommonsenseQA数据集
        dataset = load_dataset("zen-E/CommonsenseQA-GPT4omini")
        test_set = dataset['validation']
    elif "gsm8k" == data_args.data_name:
        # GSM8K数据集（最常用的数学推理基准）
        dataset = load_dataset("gsm8k", "main")
        test_set = dataset['test']
    else:
        raise NotImplementedError

    # =========================================================================
    # 步骤5: 格式化问题和答案
    # =========================================================================
    logging.warning("Formatting inputs...")

    # 提取问题文本
    question = [f"{example[question_name].strip().replace('  ', ' ')}" for example in test_set]

    # 提取并标准化答案（统一为数值格式）
    answer = []
    for example in test_set:
        example = example[answer_name]

        # 处理布尔类型答案
        if isinstance(example, bool):
            answer.append(example)
            continue

        # 处理"True"/"False"字符串
        if example in ["True", "False"]:
            if example == "True":
                ans = True
            else:
                ans = False
            answer.append(ans)
            continue

        # 处理选择题（A-E）
        if example in "ABCDE":
            answer.append(example)
            continue

        # 处理GSM8K格式的答案（####后面的才是最终答案）
        if "####" in example:
            ans = example.split('####')[-1]
        else:
            ans = example

        # 处理数字中的逗号（如2,000 -> 2000）
        ans = ans.replace(',', '')

        # 转换为浮点数
        try:
            ans = float(ans)
        except ValueError:
            ans = float("inf")
        answer.append(ans)

    # =========================================================================
    # 步骤6: Tokenize输入数据
    # =========================================================================
    logging.warning("Tokenizing inputs...")

    # 计算评估批次数
    eval_step = math.ceil(len(question) / data_args.batch_size)
    logging.warning(f"Total example: {len(question)} | eval batch size: {data_args.batch_size}"
                    f"eval steps: {eval_step}")

    # 对每个问题添加特殊token
    question_data = []
    for i in range(eval_step):
        # 分批tokenize
        if i < eval_step - 1:
            batch = tokenizer(
                question[i*data_args.batch_size: (i+1)*data_args.batch_size],
                return_tensors="pt",
                padding="longest",
            )
        else:
            batch = tokenizer(
                question[i*data_args.batch_size:],
                return_tensors="pt",
                padding="longest",
            )

        # 添加bot (beginning of thought) token
        # 表示latent推理空间的开始
        if training_args.remove_eos:
            # 不需要EOS作为分隔
            bot_tensor = torch.tensor([model.bot_id], dtype=torch.long).expand(batch["input_ids"].size(0), 1)
        else:
            # 在bot之前添加eos作为分隔
            bot_tensor = torch.tensor([tokenizer.eos_token_id, model.bot_id], dtype=torch.long).expand(batch["input_ids"].size(0), 2)

        # 拼接input_ids和bot_tensor
        batch["input_ids"] = torch.cat((batch["input_ids"], bot_tensor), dim=1)
        # 更新attention mask
        batch["attention_mask"] = torch.cat((batch["attention_mask"], torch.ones_like(bot_tensor)), dim=1)
        # 记录输入长度
        batch['input_len'] = len(batch['input_ids'][0])
        # 移到GPU
        question_data.append(batch.to(device))

    # =========================================================================
    # 步骤7: 设置生成参数并开始推理
    # =========================================================================
    model.eval()

    # 生成配置
    gen_kwargs = {
        "max_new_tokens": 256,    # 最大生成token数
        "temperature": 0.1,         # 温度参数（越小越确定性）
        "top_k": 40,               # top-k采样
        "top_p": 0.95,            # top-p（nucleus）采样
        "do_sample": True,         # 是否使用采样
    }

    # 初始化结果存储
    ans_pred_list = []
    len_cot = []  # 记录生成的CoT长度

    # =========================================================================
    # 步骤8: 对每个批次进行推理
    # =========================================================================
    for step, batch in enumerate(question_data):
        batch_size = batch["input_ids"].size(0)

        with torch.no_grad():
            # -------------------------------------------------------------------------
            # 8.1: 编码问题，获取初始latent embedding
            # -------------------------------------------------------------------------
            past_key_values = None

            # 对问题进行编码
            outputs = model.codi(
                input_ids=batch["input_ids"],
                use_cache=True,
                output_hidden_states=True,
                past_key_values=past_key_values,
                attention_mask=batch["attention_mask"]
            )

            past_key_values = outputs.past_key_values
            # 取最后一个隐藏状态作为latent embedding
            latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)

            # 如果使用投影层
            if training_args.use_prj:
                latent_embd = model.prj(latent_embd)

            # -------------------------------------------------------------------------
            # 8.2: 执行隐式推理步骤（latent space reasoning）
            # -------------------------------------------------------------------------
            inf_latent_iterations = training_args.inf_latent_iterations
            for i in range(inf_latent_iterations):
                # 使用latent embedding进行推理
                outputs = model.codi(
                    inputs_embeds=latent_embd,
                    use_cache=True,
                    output_hidden_states=True,
                    past_key_values=past_key_values
                )
                past_key_values = outputs.past_key_values
                # 更新latent embedding
                latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)

                if training_args.use_prj:
                    latent_embd = model.prj(latent_embd)

            # -------------------------------------------------------------------------
            # 8.3: 添加EOT (end of thought) token
            # 表示latent推理结束，开始生成答案
            # -------------------------------------------------------------------------
            if training_args.remove_eos:
                eot_emb = model.get_embd(model.codi, model.model_name)(
                    torch.tensor([model.eot_id], dtype=torch.long, device='cuda')
                ).unsqueeze(0).to(device)
            else:
                eot_emb = model.get_embd(model.codi, model.model_name)(
                    torch.tensor([model.eot_id, tokenizer.eos_token_id], dtype=torch.long, device='cuda')
                ).unsqueeze(0).to(device)

            # 扩展到batch size
            eot_emb = eot_emb.expand(batch["input_ids"].size(0), -1, -1)

            # 作为第一个输出token
            output = eot_emb

            # -------------------------------------------------------------------------
            # 8.4: 自回归生成答案
            # -------------------------------------------------------------------------
            seq_len = 0
            # 记录每个序列是否已经生成结束
            finished = torch.zeros(batch_size, dtype=torch.bool, device="cuda")
            # 存储预测的token列表
            pred_tokens = [[] for _ in range(batch_size)]

            # 最多生成max_new_tokens个token
            for i in range(gen_kwargs["max_new_tokens"]):
                seq_len += 1

                # 获取模型输出
                out = model.codi(
                    inputs_embeds=output,
                    output_hidden_states=False,
                    attention_mask=None,
                    use_cache=True,
                    output_attentions=False,
                    past_key_values=past_key_values
                )
                past_key_values = out.past_key_values
                # 取最后一个位置的logits
                logits = out.logits[:, -1, :model.codi.config.vocab_size-1]

                # -------------------------------------------------------------------------
                # 8.5: 采样策略
                # -------------------------------------------------------------------------
                if training_args.greedy:
                    # Greedy解码：总是选择概率最高的token
                    next_token_ids = torch.argmax(logits, dim=-1).squeeze(-1)
                else:
                    # 温度采样
                    logits /= gen_kwargs["temperature"]

                    # Top-k采样：只考虑概率最高的k个token
                    if gen_kwargs["top_k"] > 1:
                        top_k_values, _ = torch.topk(logits, gen_kwargs["top_k"], dim=-1)
                        min_top_k_value = top_k_values[:, -1].unsqueeze(-1)
                        logits[logits < min_top_k_value] = -float("inf")

                    # Top-p (nucleus)采样：只考虑累积概率达到p的token
                    if gen_kwargs["top_p"] < 1.0:
                        sorted_logit, sorted_indices = torch.sort(logits, descending=True, dim=-1)
                        cumulative_probs = torch.cumsum(F.softmax(sorted_logit, dim=-1), dim=-1)

                        # 移除累积概率超过top_p的token
                        sorted_indices_to_remove = cumulative_probs > gen_kwargs["top_p"]
                        if sorted_indices_to_remove.any():
                            sorted_indices_to_remove = sorted_indices_to_remove.roll(1, dims=-1)
                            sorted_indices_to_remove[:, 0] = False

                        for b in range(logits.size(0)):
                            logits[b, sorted_indices[b, sorted_indices_to_remove[b]]] = -float("inf")

                    # 从概率分布中采样
                    probs = F.softmax(logits, dim=-1)
                    next_token_ids = torch.multinomial(probs, num_samples=1).squeeze(-1)

                # -------------------------------------------------------------------------
                # 8.6: 处理EOS token
                # -------------------------------------------------------------------------
                for b in range(batch_size):
                    if not finished[b]:
                        pred_tokens[b].append(next_token_ids[b].item())
                        # 如果生成EOS，标记为完成
                        if next_token_ids[b] == tokenizer.eos_token_id:
                            finished[b] = True

                # 如果所有序列都已完成，提前退出
                if finished.all():
                    break

                # 将预测的token转换为embedding，作为下一步的输入
                output = model.get_embd(model.codi, model.model_name)(next_token_ids).unsqueeze(1).to(device)

            # -------------------------------------------------------------------------
            # 8.7: 解码并提取答案
            # -------------------------------------------------------------------------
            for mini_step, pred_token in enumerate(pred_tokens):
                len_cot.append(len(pred_token))
                # 解码token为文本
                decoded_pred = tokenizer.decode(pred_token, skip_special_tokens=True)

                # 打印详细信息
                if do_print:
                    print(f"Question {step*data_args.batch_size+mini_step} Starts...")
                    print(f"Q: {question[step*data_args.batch_size+mini_step]}")
                    print(decoded_pred)
                    print(f"Question {step*data_args.batch_size+mini_step} Ends")
                    print(f"Prediction={extract_answer_number(decoded_pred)}; Groundtruth={answer[step*data_args.batch_size+mini_step]}")
                    print("")

                # 提取答案数字并添加到列表
                ans_pred_list.append(extract_answer_number(decoded_pred))

    # =========================================================================
    # 步骤9: 计算准确率
    # =========================================================================
    accuracy = compute_accuracy(answer, ans_pred_list)

    # 打印结果
    print(f"adapter: {model_args.adapter_name_or_path} | GSM8K test accuracy: {100*accuracy:.2f}% | ")
    print(f"average length of COT: {sum(len_cot)/len(len_cot)}")

    return 100 * accuracy


def extract_answer_number(sentence: str) -> float:
    """
    从生成的文本中提取答案数字

    处理逻辑：
    1. 提取所有数字（包括负数和小数）
    2. 对于选择题，返回选项字母
    3. 对于判断题，返回True/False
    4. 返回最后一个数字作为答案

    参数：
    - sentence: 模型生成的文本

    返回：
    - 提取的答案（数值、字母或布尔值）
    """
    # 移除逗号
    sentence = sentence.replace(',', '')

    # 使用正则表达式提取数字（包括负数和小数）
    pred = [s for s in re.findall(r'-?\d+\.?\d*', sentence)]

    if not pred:
        # 没有找到数字，根据数据集类型处理
        if "commonsense" in data_args.data_name:
            # 提取选择题答案
            pred = sentence.split("The answer is:")[-1].strip()
            if pred[0] not in "ABCDE":
                return "C"  # 默认返回C
            return pred[0]
        elif "strategy" in data_args.data_name or "prontoqa" in data_args.data_name.lower():
            # 判断题
            if "True" in sentence:
                return True
            elif "False" in sentence:
                return False
            else:
                raise ValueError
        return float('inf')  # 未找到答案

    # 使用最后一个数字作为答案（常见的提取策略）
    pred_answer = float(pred[-1])

    return pred_answer


def compute_accuracy(gold: list, pred: list) -> float:
    """
    计算预测准确率

    比较预测答案和真实答案

    参数：
    - gold: 真实答案列表
    - pred: 预测答案列表

    返回：
    - 准确率（0-1之间）
    """
    acc = 0.0
    for p, g in zip(pred, gold):
        if isinstance(p, list):
            # 如果预测是列表，检查真实答案是否在其中
            if g in p:
                acc += 1
        else:
            # 直接比较
            if p == g:
                acc += 1

    return acc / len(gold)


# =========================================================================
# 主程序入口
# =========================================================================
if __name__ == "__main__":
    # 解析命令行参数
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # 多次运行取平均（用于评估采样稳定性）
    accu_list = []
    for i in range(training_args.inf_num_iterations):
        accu = evaluation(model_args, data_args, training_args)
        accu_list.append(accu)

    # 打印平均准确率
    print(f"Average accuracy over {training_args.inf_num_iterations} sampling: {sum(accu_list)/len(accu_list)}")
