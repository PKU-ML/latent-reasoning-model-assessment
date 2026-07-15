"""
================================================================================
RoT模型 随机前缀攻击
================================================================================

一、算法概述
-----------
本代码实现了一种针对RoT模型的随机前缀攻击，通过随机选取token作为前缀，
添加到问题前面，测试是否能导致模型输出错误答案。

二、攻击方式
-----------
- 随机选取5个token作为前缀
- 前缀直接添加在问题前面，中间不加空格
- 使用RoT模型评估，答案错误则攻击成功
- 重复多次，每次独立随机选取前缀

================================================================================
"""

import os
import sys
import json
import yaml
import re
import argparse
from pathlib import Path

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent) + "/RoT")

from models.cot_compressor import CoTCompressor
from scripts.evaluate import load_model


def main():
    # ============ 命令行参数 ============
    parser = argparse.ArgumentParser(description="RoT 随机前缀攻击")
    parser.add_argument("--problem-id", type=int, default=None,
                        help="指定攻击的问题ID (默认自动分配)")
    parser.add_argument("--num-trials", type=int, default=100,
                        help="随机前缀尝试次数 (默认200)")
    parser.add_argument("--prefix-length", type=int, default=5,
                        help="前缀token数量 (默认5)")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子 (默认42)")
    parser.add_argument("--output-dir", type=str,
                        default=os.environ.get(
                            "ROT_OUTPUT_DIR",
                            str(Path(__file__).resolve().parent / "adv_results" / "results_random"),
                        ),
                        help="结果保存目录 (默认: <RoT>/adv_results/results_random；可通过 ROT_OUTPUT_DIR 覆盖)")
    parser.add_argument("--rot-gpu", type=int, default=0,
                        help="RoT模型使用的GPU编号 (默认0)")
    parser.add_argument("--dataset", type=str, choices=["gsm8k", "MultiArith", "SVAMP"], default="gsm8k",
                        help="指定要攻击的数据集 (默认 gsm8k)")
    args = parser.parse_args()

    # ============ 配置 ============
    DEVICE = f"cuda:{args.rot_gpu}"
    print(f"RoT模型使用设备: {DEVICE}")

    # 随机种子
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    PREFIX_LENGTH = args.prefix_length
    NUM_TRIALS = args.num_trials

    # RoT模型配置
    CHECKPOINT = os.environ.get(
        "ROT_CHECKPOINT",
        str(Path(__file__).resolve().parent / "RoT" / "rot_model" / "converted"),
    )
    STAGE1_CHECKPOINT = os.environ.get(
        "ROT_STAGE1_CHECKPOINT",
        str(Path(__file__).resolve().parent / "RoT" / "rot_model"),
    )
    CONFIG = os.environ.get(
        "ROT_CONFIG",
        str(Path(__file__).resolve().parent / "RoT" / "configs" / "stage2_config_qwen3vl_2b.yaml"),
    )
    if not Path(CHECKPOINT).exists():
        print(f"[警告] ROT_CHECKPOINT={CHECKPOINT} 不存在，请通过环境变量指向实际目录")
    if not Path(CONFIG).exists():
        print(f"[警告] ROT_CONFIG={CONFIG} 不存在，请通过环境变量指向实际 yaml")
    DATA_DIR = os.environ.get(
        "DATA_DIR",
        str(Path(__file__).resolve().parent.parent / "data"),
    )
    DATA_PATH = f"{DATA_DIR}/{args.dataset}.json"

    MAX_NEW_TOKENS = 256
    TEMPERATURE = 0.0
    NUM_VISION_TOKENS = 32
    STOP_THRESHOLD = 0.02

    os.makedirs(args.output_dir, exist_ok=True)
    if args.problem_id is not None:
        PROBLEM_ID = args.problem_id
    else:
        existing_files = list(Path(args.output_dir).glob("problem_*.json"))
        if existing_files:
            max_id = max([int(f.stem.split('_')[1]) for f in existing_files])
            PROBLEM_ID = max_id + 1
        else:
            PROBLEM_ID = 0

    print("=" * 60)
    print("RoT 随机前缀攻击")
    print(f"前缀长度: {PREFIX_LENGTH}")
    print(f"尝试次数: {NUM_TRIALS}")
    print(f"数据集: {args.dataset}")
    print("=" * 60)

    # ============ 1. 加载数据 ============
    print(f"\n[1] 加载{args.dataset}数据...")

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

    # ============ 2. 加载RoT模型 ============
    print("\n[2] 加载RoT模型...")

    with open(CONFIG, "r") as f:
        config = yaml.safe_load(f)

    model = load_model(
        checkpoint_path=CHECKPOINT,
        config=config,
        model_type="v2",
        verbose=True,
        stage1_checkpoint=STAGE1_CHECKPOINT
    )

    # 获取tokenizer用于构建有效token池
    tokenizer = model.language_model.get_input_embeddings()
    # 使用一个简单的tokenizer路径来获取decode功能
    # RoT模型实际上不需要直接tokenize问题，因为它使用OCR处理

    print(f"RoT模型加载完成")

    # ============ 3. 辅助函数 ============

    def extract_answer_number(sentence):
        """从生成的文本中提取答案数字"""
        sentence = sentence.replace(',', '')
        pred = re.findall(r'-?\d+\.?\d*', sentence)
        if not pred:
            return float('inf')
        return float(pred[-1])

    def generate_answer(question_str, cot_str=""):
        """使用RoT模型生成答案"""
        with torch.no_grad():
            generated = model.generate(
                question_text=question_str,
                cot_text=cot_str if cot_str else None,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
                max_vision_tokens=NUM_VISION_TOKENS,
                stop_threshold=STOP_THRESHOLD,
                verbose=False
            )
            return generated

    def is_correct(pred_answer, ground_truth):
        """判断答案是否正确"""
        if pred_answer == float('inf') or ground_truth == float('inf'):
            return False
        return abs(pred_answer - float(ground_truth)) < 1e-9

    # ============ 4. Baseline评估 ============
    print("\n[3] 测试baseline...")

    baseline_text = generate_answer(question, "")
    baseline_answer = extract_answer_number(baseline_text)
    baseline_correct = is_correct(baseline_answer, answers[PROBLEM_ID])

    print(f"Baseline生成文本: '{baseline_text[:100]}...'")
    print(f"Baseline提取答案: {baseline_answer}")
    print(f"正确答案: {answer_str}")
    print(f"Baseline是否正确: {baseline_correct}")

    # ============ 5. 构建有效token池 ============
    # 使用Llama-3.2-1B-Instruct的tokenizer来构建token池
    print("\n[4] 构建有效token池...")
    try:
        from transformers import AutoTokenizer
        llama_tokenizer = AutoTokenizer.from_pretrained(
            os.environ.get(
                "ROT_LLAMA_TOKENIZER",
                "meta-llama/Llama-3.2-1B-Instruct",
            ),
            trust_remote_code=True
        )
        char_tokens = []
        for i in range(8000, 200000):
            try:
                decoded = llama_tokenizer.decode([i])
                if decoded.strip() and len(decoded.strip()) > 0:
                    char_tokens.append(i)
            except:
                pass
        print(f"有效token数量: {len(char_tokens)}")
        use_tokenizer = True
    except Exception as e:
        print(f"无法加载tokenizer，跳过token池构建: {e}")
        char_tokens = []
        use_tokenizer = False

    # ============ 6. 随机前缀攻击 ============
    print(f"\n[5] 开始随机前缀攻击 ({NUM_TRIALS} 次)...")

    all_results = []
    success_count = 0

    for trial in range(NUM_TRIALS):
        # 随机选取PREFIX_LENGTH个token作为前缀
        prefix_ids = np.random.choice(char_tokens, size=PREFIX_LENGTH, replace=True)
        prefix_str = llama_tokenizer.decode(prefix_ids)

        # 构建带前缀的问题 - 前缀直接加在问题前，中间不加空格
        prefixed_question = prefix_str + question

        # 评估
        generated_text = generate_answer(prefixed_question, "")
        pred_answer = extract_answer_number(generated_text)
        correct = is_correct(pred_answer, answers[PROBLEM_ID])
        attack_success = not correct

        if attack_success:
            success_count += 1

        result = {
            "trial": trial + 1,
            "prefix": prefix_str,
            "prefix_token_ids": [int(x) for x in prefix_ids] if char_tokens else [],
            "generated_text": generated_text,
            "pred_answer": pred_answer,
            "correct": correct,
            "attack_success": attack_success,
        }
        all_results.append(result)

        if (trial + 1) % 20 == 0 or attack_success:
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

    successful_prefixes = [r for r in all_results if r["attack_success"]]
    if successful_prefixes:
        print(f"\n攻击成功的前缀示例:")
        for r in successful_prefixes[:5]:
            print(f"  '{r['prefix']}' -> pred={r['pred_answer']}, correct={r['correct']}")

    # ============ 8. 保存结果 ============
    print("\n[6] 保存结果...")

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
            "dataset": args.dataset,
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