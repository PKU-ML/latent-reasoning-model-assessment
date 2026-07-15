"""
================================================================================
CODI模型 随机前缀攻击
================================================================================

一、算法概述
-----------
本代码实现了一种针对CODI模型的随机前缀攻击，通过随机选取token作为前缀，
添加到问题前面，测试是否能导致模型输出错误答案。

二、攻击方式
-----------
- 随机选取5个token作为前缀
- 前缀直接添加在问题前面，中间不加空格（参考attack_white.py）
- 使用CODI模型评估，答案错误则攻击成功
- 重复多次，每次独立随机选取前缀

================================================================================
"""

import json
import torch
import numpy as np
import os
import re
import argparse

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


def main():
    # ============ 命令行参数 ============
    parser = argparse.ArgumentParser(description="CODI 随机前缀攻击")
    parser.add_argument("--problem-id", type=int, default=None,
                        help="指定攻击的问题ID (默认自动分配)")
    parser.add_argument("--num-trials", type=int, default=100,
                        help="随机前缀尝试次数 (默认100)")
    parser.add_argument("--prefix-length", type=int, default=3,
                        help="前缀token数量 (默认5)")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子 (默认42)")
    parser.add_argument("--output-dir", type=str,
                        default=os.environ.get(
                            "CODI_OUTPUT_DIR",
                            str(Path(__file__).resolve().parent / "adv_results" / "results_random"),
                        ),
                        help="结果保存目录 (默认: <codi>/adv_results/results_random；可通过 CODI_OUTPUT_DIR 环境变量覆盖)")
    parser.add_argument("--codi-gpu", type=int, default=0,
                        help="CODI模型使用的GPU编号 (默认0)")
    parser.add_argument("--dataset", type=str, choices=["gsm8k", "MultiArith", "SVAMP"], default="gsm8k",
                        help="指定要攻击的数据集 (默认 gsm8k)")
    args = parser.parse_args()

    # ============ 配置 ============
    CODI_DEVICE = f"cuda:{args.codi_gpu}"
    print(f"CODI模型使用设备: {CODI_DEVICE}")

    # 随机种子
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # 配置参数
    PREFIX_LENGTH = args.prefix_length
    NUM_TRIALS = args.num_trials

    # CODI模型配置
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
    print("CODI 随机前缀攻击")
    print(f"前缀长度: {PREFIX_LENGTH}")
    print(f"尝试次数: {NUM_TRIALS}")
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
        output_dir="/tmp/codi_train",
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
        """构建prompt - 前缀直接加在问题前，中间不加空格（参考attack_white.py）"""
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

    def is_correct(pred_answer, ground_truth):
        """判断答案是否正确"""
        if pred_answer == float('inf') or not ground_truth.lstrip('-').replace('.','',1).isdigit():
            return False
        return str(pred_answer) == ground_truth or abs(pred_answer - float(ground_truth)) < 0.01

    # ============ 4. Baseline评估 ============
    print("\n[4] 测试baseline...")

    baseline_input_ids, baseline_attn = make_prompt(question, "")
    with torch.no_grad():
        baseline_text, baseline_ids = generate_answer(baseline_input_ids, baseline_attn)

    baseline_answer = extract_answer_number(baseline_text)
    baseline_correct = is_correct(baseline_answer, answer_str)

    print(f"Baseline生成文本: '{baseline_text}'")
    print(f"Baseline提取答案: {baseline_answer}")
    print(f"正确答案: {answer_str}")
    print(f"Baseline是否正确: {baseline_correct}")

    # ============ 5. 构建有效token池 ============
    print("\n[5] 构建有效token池...")

    char_tokens = []
    for i in range(8000, 200000):
        try:
            decoded = tokenizer.decode([i])
            if decoded.strip() and len(decoded.strip()) > 0:
                char_tokens.append(i)
        except:
            pass

    print(f"有效token数量: {len(char_tokens)}")

    # ============ 6. 随机前缀攻击 ============
    print(f"\n[6] 开始随机前缀攻击 ({NUM_TRIALS} 次)...")

    all_results = []
    success_count = 0

    for trial in range(NUM_TRIALS):
        # 随机选取PREFIX_LENGTH个token作为前缀
        prefix_ids = np.random.choice(char_tokens, size=PREFIX_LENGTH, replace=True)
        prefix_str = tokenizer.decode(prefix_ids)

        # 构建带前缀的问题
        input_ids, attention_mask = make_prompt(question, prefix_str)

        # 评估
        with torch.no_grad():
            generated_text, _ = generate_answer(input_ids, attention_mask)

        pred_answer = extract_answer_number(generated_text)
        correct = is_correct(pred_answer, answer_str)
        attack_success = not correct  # 答案错误则认为攻击成功

        if attack_success:
            success_count += 1

        result = {
            "trial": trial + 1,
            "prefix": prefix_str,
            "prefix_token_ids": [int(x) for x in prefix_ids],
            "generated_text": generated_text,
            "pred_answer": pred_answer,
            "correct": correct,
            "attack_success": attack_success,
        }
        all_results.append(result)

        if (trial + 1) % 10 == 0 or attack_success:
            status = "✓ 攻击成功" if attack_success else ""
            print(f"  Trial {trial+1}/{NUM_TRIALS}: prefix='{prefix_str[:30]}...' correct={correct} pred={pred_answer} {status}")


    # ============ 7. 结果统计 ============
    print("\n" + "=" * 60)
    print("攻击结果统计")
    print("=" * 60)

    print(f"总尝试次数: {NUM_TRIALS}")
    print(f"攻击成功次数: {success_count}")
    print(f"攻击成功率: {success_count/NUM_TRIALS*100:.2f}%")
    print(f"Baseline正确: {baseline_correct}")

    # 统计攻击成功的前缀
    successful_prefixes = [r for r in all_results if r["attack_success"]]
    if successful_prefixes:
        print(f"\n攻击成功的前缀示例:")
        for r in successful_prefixes[:5]:
            print(f"  '{r['prefix']}' -> pred={r['pred_answer']}, correct={r['correct']}")

    # ============ 8. 保存结果 ============
    print("\n[8] 保存结果...")

    result = {
        "question": question,
        "ground_truth": answer_str,
        "baseline": {
            "generated_text": baseline_text,
            "pred_answer": baseline_answer,
            "correct": baseline_correct,
        },
        "config": {
            "prefix_length": PREFIX_LENGTH,
            "num_trials": NUM_TRIALS,
            "seed": args.seed,
        },
        "success_count": success_count,
        "success_rate": success_count / NUM_TRIALS,
        "all_results": all_results,
        "attack_success": success_count > 0,
    }

    os.makedirs(args.output_dir, exist_ok=True)
    save_path = os.path.join(args.output_dir, f"problem_{PROBLEM_ID}.json")
    with open(save_path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"\n结果已保存到 {save_path}")
    print(f"攻击效果: {'成功' if success_count > 0 else '失败'}")


if __name__ == "__main__":
    main()
