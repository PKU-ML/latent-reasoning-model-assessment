"""
================================================================================
CODI模型 黑盒对抗攻击 - 基于问题改写
================================================================================

一、算法概述
-----------
本代码实现了一种针对CODI模型的黑盒对抗攻击，通过大模型(Qwen/Qwen3-4B)
对问题进行同义改写，使得模型在改写后的问题下输出错误的答案。

二、与白盒攻击的区别
--------------------
- 白盒攻击(attack_white.py): 在问题前添加对抗前缀，通过梯度优化前缀
- 黑盒攻击(attack_black.py): 改写问题本身，通过LLM生成同义表述

三、攻击目标
-----------
与白盒攻击相同:
- 在答案token位置，让 target_token 的logit - baseline_token 的logit 差值最大化
- 只要攻击成功即可，不需要考虑cosine相似度

================================================================================
"""

import json
import torch
import numpy as np
import os
import logging
import re
import argparse
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field

from transformers import AutoTokenizer
from peft import LoraConfig, TaskType

# 导入CODI模型
import sys
from pathlib import Path
# 子项目根目录（含 src/ 等包代码）通过环境变量 CODI_PROJECT_ROOT 注入；
# 默认使用本脚本同级目录下的 ./codi 子目录
_CODI_PROJECT_ROOT = Path(os.environ.get("CODI_PROJECT_ROOT", Path(__file__).resolve().parent / "codi"))
sys.path.insert(0, str(_CODI_PROJECT_ROOT))
from src.model import CODI, ModelArguments, DataArguments, TrainingArguments

# ============ 无用事实库 (必须包含数字，英文) ============
IRRELEVANT_FACTS = [
    "There are 24 hours in a day.",
    "April has 30 days.",
    "A bicycle has 2 wheels.",
    "Today is the 3rd of the month.",
    "The current temperature is 20 degrees.",
    "A year has 12 months.",
    "A week has 7 days.",
    "An hour has 60 minutes.",
    "A minute has 60 seconds.",
    "A year has 365 days.",
    "The Earth is about 12742 km in diameter.",
    "The speed of sound in air is about 343 meters per second.",
    "Water freezes at 0 degrees.",
    "The speed of light is about 300000 km/s.",
    "The human heart has 4 chambers.",
    "An adult has 32 teeth.",
    "China has 1.4 billion people.",
    "Japan has 125 million people.",
    "The US has 50 states.",
    "The EU has 27 member states.",
    "The Nile is about 6650 km long.",
    "Mount Everest is about 8849 meters high.",
    "The Pacific covers 165 million km².",
    "The human body has 206 bones.",
    "H2O contains 2 hydrogen atoms.",
    "Oxygen makes up about 21% of air.",
    "Standard atmospheric pressure is 101325 Pa.",
    "A plant has 46 chromosomes.",
    "DNA has 2 strands in its double helix.",
    "Bitcoin was created in 2009.",
]


def main():
    # ============ 命令行参数 ============
    parser = argparse.ArgumentParser(description="CODI 黑盒对抗攻击 - 问题改写")
    parser.add_argument("--problem-id", type=int, default=None,
                        help="指定攻击的问题ID (默认自动分配)")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子 (默认42)")
    parser.add_argument("--output-dir", type=str,
                        default=os.environ.get(
                            "CODI_OUTPUT_DIR",
                            str(Path(__file__).resolve().parent / "adv_results" / "results_black"),
                        ),
                        help="结果保存目录 (默认: <codi>/adv_results/results_black；可通过 CODI_OUTPUT_DIR 环境变量覆盖)")
    parser.add_argument("--codi-gpu", type=int, default=0,
                        help="CODI模型使用的GPU编号 (默认0)")
    parser.add_argument("--dataset", type=str, choices=["gsm8k", "MultiArith", "SVAMP"], default="gsm8k",
                        help="指定要攻击的数据集 (默认 gsm8k)")
    args = parser.parse_args()
    #python attack_black.py --problem-id 0

    # ============ 配置 ============
    CODI_DEVICE = f"cuda:{args.codi_gpu}"
    print(f"CODI模型使用设备: {CODI_DEVICE}")

    # 随机种子
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # CODI模型配置
    # 模型路径与 checkpoint 通过环境变量注入；找不到时给出明确报错而不是读取硬编码路径。
    MODEL_NAME = os.environ.get(
        "CODI_MODEL_NAME",
        "meta-llama/Llama-3.2-1B-Instruct",
    )
    CKPT_DIR = os.environ.get(
        "CODI_CKPT_DIR",
        str(Path(__file__).resolve().parent / "codi" / "codi_llama1b"),
    )
    if not Path(MODEL_NAME).exists() and not MODEL_NAME.startswith(("meta-llama/", "huggingface.co/", "http")):
        print(f"[警告] CODI_MODEL_NAME={MODEL_NAME} 在本地不存在，请检查环境变量或模型缓存路径")
    if not Path(CKPT_DIR).exists():
        print(f"[警告] CODI_CKPT_DIR={CKPT_DIR} 不存在，请设置环境变量指向实际的 CODI checkpoint 目录")
    NUM_LATENT = 6
    INF_LATENT_ITERATIONS = 6
    USE_PRJ = True
    REMOVE_EOS = True
    DATA_NAME = args.dataset
    # 数据目录默认指向本仓库上级目录的 data/，可通过环境变量 DATA_DIR 覆盖
    DATA_DIR = os.environ.get(
        "DATA_DIR",
        str(Path(__file__).resolve().parent.parent / "data"),
    )
    DATA_PATH = f"{DATA_DIR}/{DATA_NAME}.json"
    os.makedirs(args.output_dir, exist_ok=True)

    if args.problem_id is not None:
        PROBLEM_ID = args.problem_id
    else:
        PROBLEM_ID = len(os.listdir(args.output_dir))

    print("=" * 60)
    print("CODI 黑盒对抗攻击 - 问题改写")
    print(f"使用 {len(IRRELEVANT_FACTS)} 个无用事实进行测试")
    print("=" * 60)

    # ============ 1. 加载数据 ============
    print(f"\n[1] 加载{DATA_NAME}数据...")

    # 不同数据集的答案提取方式
    ANSWER_PATTERNS = {
        "gsm8k": r'The answer is (\d+)',
        "MultiArith": r'(\d+)',
        "SVAMP": r'(\d+)',
    }

    def load_local_dataset(data_path, dataset_name):
        questions = []
        answers = []
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for item in data:
            questions.append(item['question'])
            answer_text = item['answer'].replace(',', '')
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

    # ============ 2. 配置CODI模型 ============
    print("\n[2] 配置CODI模型...")

    model_args = ModelArguments(
        model_name_or_path=MODEL_NAME,
        lora_init=True,
        lora_r=128,
        lora_alpha=32,
        full_precision=True,
        train=False,
    )

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
        greedy=True,
        print_loss=False,
    )

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

    print("\n[3] 初始化CODI模型...")

    model = CODI(model_args, training_args, lora_config)

    try:
        state_dict = torch.load(os.path.join(CKPT_DIR, "pytorch_model.bin"), map_location="cpu")
    except:
        from safetensors.torch import load_file
        state_dict = load_file(os.path.join(CKPT_DIR, "model.safetensors"))

    model.load_state_dict(state_dict, strict=False)
    model.codi.tie_weights()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
    if tokenizer.pad_token_id is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        tokenizer.pad_token_id = model.pad_token_id

    model = model.to(CODI_DEVICE)
    model = model.to(torch.bfloat16)
    model.eval()

    print(f"CODI模型加载完成")

    # ============ 3. 辅助函数 ============

    def make_prompt(question_str, prefix_str=""):
        """构建prompt"""
        full_text = prefix_str.strip() + question_str.strip()
        inputs = tokenizer(full_text, return_tensors="pt", padding=True)

        if REMOVE_EOS:
            bot_tensor = torch.tensor([model.bot_id], dtype=torch.long).expand(inputs["input_ids"].size(0), 1)
        else:
            bot_tensor = torch.tensor([tokenizer.eos_token_id, model.bot_id], dtype=torch.long).expand(inputs["input_ids"].size(0), 2)

        input_ids = torch.cat((inputs["input_ids"], bot_tensor), dim=1)
        attention_mask = torch.cat((inputs["attention_mask"], torch.ones_like(bot_tensor)), dim=1)
        return input_ids, attention_mask

    def get_embedding_layer(m):
        """获取模型的embedding层"""
        return m.get_embd(m.codi, m.model_name)

    def generate_answer(input_ids, attention_mask, max_tokens=30):
        """完整生成答案"""
        input_ids = input_ids.to(CODI_DEVICE)
        attention_mask = attention_mask.to(CODI_DEVICE)

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

        eot_tensor = torch.tensor([model.eot_id], dtype=torch.long, device=CODI_DEVICE)
        eot_emb = get_embedding_layer(model)(eot_tensor).unsqueeze(0)
        eot_emb = eot_emb.expand(input_ids.size(0), -1, -1)

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
            next_token_id = logits.argmax(dim=-1).item()
            generated_ids.append(next_token_id)

            if next_token_id == tokenizer.eos_token_id:
                break

            output = get_embedding_layer(model)(
                torch.tensor([next_token_id], dtype=torch.long, device=CODI_DEVICE)
            ).unsqueeze(0)

        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        return generated_text, generated_ids

    def extract_answer_number(sentence):
        """从生成的文本中提取答案数字"""
        sentence = sentence.replace(',', '')
        pred = [s for s in re.findall(r'-?\d+\.?\d*', sentence)]
        if not pred:
            return float('inf')
        return float(pred[-1])

    # ============ 4. Baseline评估 ============
    print("\n[4] 测试baseline...")

    prefix_text = "The answer is:"
    prefix_ids = tokenizer.encode(prefix_text, add_special_tokens=False)
    NUM_PREFIX_TOKENS = len(prefix_ids)
    ANSWER_TOKEN_POS = NUM_PREFIX_TOKENS + 1

    baseline_input_ids, baseline_attn = make_prompt(question, "")
    with torch.no_grad():
        baseline_text, baseline_ids = generate_answer(baseline_input_ids, baseline_attn)

    baseline_answer = extract_answer_number(baseline_text)
    print(f"Baseline生成文本: '{baseline_text}'")
    print(f"Baseline提取答案: {baseline_answer}")
    print(f"正确答案: {answer_str}")

    baseline_correct = (str(baseline_answer) == answer_str or
                       abs(baseline_answer - float(answer_str)) < 0.01 if baseline_answer != float('inf') and answer_str.lstrip('-').replace('.','',1).isdigit() else False)
    print(f"Baseline是否正确: {baseline_correct}")

    # 获取baseline的hidden向量和token信息
    print("\n获取baseline信息...")
    with torch.no_grad():
        input_ids = baseline_input_ids.to(CODI_DEVICE)
        attention_mask = baseline_attn.to(CODI_DEVICE)

        outputs = model.codi(input_ids=input_ids, use_cache=True, output_hidden_states=True, attention_mask=attention_mask)
        past_key_values = outputs.past_key_values
        hidden_clean = outputs.hidden_states[-1][:, -2, :].unsqueeze(1).detach().clone()
        latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)

        if USE_PRJ:
            latent_embd = model.prj(latent_embd)

        for i in range(INF_LATENT_ITERATIONS):
            outputs = model.codi(inputs_embeds=latent_embd, use_cache=True, output_hidden_states=True, past_key_values=past_key_values)
            past_key_values = outputs.past_key_values
            latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)
            if USE_PRJ:
                latent_embd = model.prj(latent_embd)

        eot_emb = model.get_embd(model.codi, model.model_name)(torch.tensor([model.eot_id], dtype=torch.long, device=CODI_DEVICE)).unsqueeze(0)
        embed_layer = get_embedding_layer(model)
        all_logits = []
        current_emb = eot_emb
        current_past = past_key_values

        for step in range(NUM_PREFIX_TOKENS + 10):
            outputs = model.codi(inputs_embeds=current_emb, use_cache=True, past_key_values=current_past)
            current_past = outputs.past_key_values
            logits = outputs.logits[0, -1].detach().clone()
            all_logits.append(logits)
            next_token_id = logits.argmax(dim=-1).item()
            if next_token_id == tokenizer.eos_token_id:
                break
            current_emb = embed_layer(torch.tensor([next_token_id], dtype=torch.long, device=CODI_DEVICE)).unsqueeze(0)

        answer_token_logits = all_logits[ANSWER_TOKEN_POS] if len(all_logits) > ANSWER_TOKEN_POS else all_logits[-1]
        sorted_logits, sorted_indices = torch.sort(answer_token_logits, descending=True)
        BASELINE_TOKEN_ID = sorted_indices[0].item()
        TARGET_TOKEN_ID = sorted_indices[1].item()

        baseline_prob = torch.softmax(answer_token_logits, dim=-1)[BASELINE_TOKEN_ID].item()
        target_prob = torch.softmax(answer_token_logits, dim=-1)[TARGET_TOKEN_ID].item()
        logit_diff = answer_token_logits[TARGET_TOKEN_ID].item() - answer_token_logits[BASELINE_TOKEN_ID].item()
        baseline_answer_prob = baseline_prob  # 正确答案token的softmax概率

        baseline_token_str = tokenizer.decode([BASELINE_TOKEN_ID])
        target_token_str = tokenizer.decode([TARGET_TOKEN_ID])

        print(f"Baseline答案token位置:")
        print(f"  最大token: '{baseline_token_str}' (id={BASELINE_TOKEN_ID}), prob={baseline_prob:.4f}")
        print(f"  第二大token: '{target_token_str}' (id={TARGET_TOKEN_ID}), prob={target_prob:.4f}")
        print(f"  正确答案概率: {baseline_answer_prob:.4f}")

    # ============ 5. 评估函数 ============

    def evaluate_question(question_str):
        """评估一个问题，返回logit差值"""
        input_ids, attention_mask = make_prompt(question_str, "")
        input_ids = input_ids.to(CODI_DEVICE)
        attention_mask = attention_mask.to(CODI_DEVICE)

        with torch.no_grad():
            outputs = model.codi(input_ids=input_ids, use_cache=True, output_hidden_states=True, attention_mask=attention_mask)
            past_key_values = outputs.past_key_values
            latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)

            if USE_PRJ:
                latent_embd = model.prj(latent_embd)

            for _ in range(INF_LATENT_ITERATIONS):
                outputs = model.codi(inputs_embeds=latent_embd, use_cache=True, output_hidden_states=True, past_key_values=past_key_values)
                past_key_values = outputs.past_key_values
                latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)
                if USE_PRJ:
                    latent_embd = model.prj(latent_embd)

            embed_layer = get_embedding_layer(model)
            eot_emb = embed_layer(torch.tensor([model.eot_id], dtype=torch.long, device=CODI_DEVICE)).unsqueeze(0)

            current_emb = eot_emb
            current_past = past_key_values
            all_logits = []

            for step in range(ANSWER_TOKEN_POS + 5):
                outputs = model.codi(inputs_embeds=current_emb, use_cache=True, past_key_values=current_past)
                current_past = outputs.past_key_values
                logits = outputs.logits[0, -1].detach().clone()
                all_logits.append(logits)
                next_token_id = logits.argmax(dim=-1).item()
                if next_token_id == tokenizer.eos_token_id:
                    break
                current_emb = embed_layer(torch.tensor([next_token_id], dtype=torch.long, device=CODI_DEVICE)).unsqueeze(0)

            answer_logits = all_logits[ANSWER_TOKEN_POS] if len(all_logits) > ANSWER_TOKEN_POS else all_logits[-1]
            answer_prob = torch.softmax(answer_logits, dim=-1)[BASELINE_TOKEN_ID].item()

            text, _ = generate_answer(input_ids, attention_mask)
            pred_answer = extract_answer_number(text)
            is_correct = (str(pred_answer) == answer_str or
                         abs(pred_answer - float(answer_str)) < 0.01 if pred_answer != float('inf') and answer_str.lstrip('-').replace('.','',1).isdigit() else False)

        return {
            "answer_prob": answer_prob,
            "pred_answer": pred_answer,
            "is_correct": is_correct,
            "generated_text": text
        }

    # ============ 6. 使用无用事实生成改写问题 ============
    print("\n[6] 使用无用事实生成改写问题...")

    print(f"\n评估原始问题...")
    original_eval = evaluate_question(question)
    print(f"  answer_prob: {original_eval['answer_prob']:.4f}, correct: {original_eval['is_correct']}")

    # 生成所有改写问题（默认所有事实均为无关信息，跳过Qwen判断）
    print(f"\n生成 {len(IRRELEVANT_FACTS)} 个改写问题并调用CODI测试...")
    rewritten_questions = []
    for fact in IRRELEVANT_FACTS:
        rewritten_q = f"{fact} {question}"
        rewritten_questions.append((fact, rewritten_q))

    all_candidates = []
    for i, (fact, rw_q) in enumerate(rewritten_questions):
        print(f"  测试 [{i+1}/{len(IRRELEVANT_FACTS)}]: {fact[:40]}...")
        eval_result = evaluate_question(rw_q)
        all_candidates.append({
            "question": rw_q,
            "fact": fact,
            "answer_prob": eval_result["answer_prob"],
            "is_correct": eval_result["is_correct"],
        })
        print(f"    answer_prob={eval_result['answer_prob']:.4f}, correct={eval_result['is_correct']}")

    # ============ 8. 选择最佳结果 ============
    print("\n" + "=" * 60)
    print("选择最佳改写")
    print("=" * 60)

    if all_candidates:
        # 找正确答案概率最低的（最有效的攻击）
        best_candidate = min(all_candidates, key=lambda x: x["answer_prob"])
        best_question = best_candidate["question"]
        best_answer_prob = best_candidate["answer_prob"]
        best_fact = best_candidate["fact"]

        print(f"最佳改写:")
        print(f"  问题: {best_question[:100]}...")
        print(f"  无用信息: {best_fact}")
        print(f"  正确答案概率: {best_answer_prob:.4f} (baseline: {baseline_answer_prob:.4f})")
        print(f"  攻击成功: {not best_candidate['is_correct']}")

        final_eval = evaluate_question(best_question)
        print(f"\n最终验证:")
        print(f"  生成文本: '{final_eval['generated_text'][:100]}...'")
        print(f"  提取答案: {final_eval['pred_answer']}")
        print(f"  是否正确: {final_eval['is_correct']}")
        print(f"  正确答案概率: {final_eval['answer_prob']:.4f}")
    else:
        print("没有找到改写")
        best_candidate = None
        best_question = question
        best_answer_prob = original_eval["answer_prob"]
        best_fact = None
        final_eval = original_eval

    # ============ 9. 保存结果 ============
    print("\n[7] 保存结果...")

    result = {
        "question": question,
        "ground_truth": answer_str,
        "best_rewrite": best_question,
        "best_fact": best_fact,
        "best_answer_prob": best_answer_prob,
        "baseline_answer_prob": baseline_answer_prob,
        "baseline_correct": baseline_correct,
        "final_eval": {
            "is_correct": final_eval["is_correct"],
            "answer_prob": final_eval["answer_prob"],
            "generated_text": final_eval["generated_text"]
        },
        "attack_success": baseline_correct and not final_eval["is_correct"],
        "num_facts": len(IRRELEVANT_FACTS),
        "all_candidates": all_candidates
    }

    os.makedirs(args.output_dir, exist_ok=True)
    save_path = os.path.join(args.output_dir, f"problem_{PROBLEM_ID}.json")
    with open(save_path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"\n结果已保存到 {save_path}")
    print(f"\n攻击效果: {'成功' if result['attack_success'] else '失败'}")


if __name__ == "__main__":
    main()
