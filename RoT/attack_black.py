#!/usr/bin/env python3
"""
================================================================================
RoT模型 黑盒对抗攻击 - 基于问题改写
================================================================================

一、算法概述
-----------
本代码实现了一种针对RoT模型的黑盒对抗攻击，通过在问题前添加无关事实，
使得模型在改写后的问题下输出错误的答案。

二、攻击目标
-----------
- 让模型在改写后的问题下输出错误的答案
- 只要攻击成功即可，不需要考虑cosine相似度

================================================================================
"""

import os
import sys
import json
import yaml
import re
from pathlib import Path
from typing import List, Dict, Optional, Any
import argparse
import torch

# 设置路径
sys.path.insert(0, str(Path(__file__).parent) + "/RoT")

from models.cot_compressor import CoTCompressor
from scripts.evaluate import load_model

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


def load_local_dataset(data_path: str) -> tuple:
    """
    从本地 JSON 文件加载数据集
    """
    questions = []
    cots = []
    answers = []

    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for item in data:
        questions.append(item['question'])
        if 'steps' in item and isinstance(item['steps'], list):
            cot = "\n".join(item['steps'])
        else:
            cot = ""
        cots.append(cot)
        answer_text = item['answer']
        answer_text = answer_text.replace(',', '')
        try:
            ans = float(answer_text)
        except ValueError:
            ans = float("inf")
        answers.append(ans)

    return questions, cots, answers


def extract_answer_number(sentence: str) -> float:
    """从生成的文本中提取答案数字"""
    sentence = sentence.replace(',', '')
    pred = re.findall(r'-?\d+\.?\d*', sentence)
    if not pred:
        return float('inf')
    return float(pred[-1])


def main():
    # ============ 命令行参数 ============
    parser = argparse.ArgumentParser(description="RoT 黑盒对抗攻击 - 问题改写")
    parser.add_argument("--problem-id", type=int, default=None,
                        help="指定攻击的问题ID (默认自动分配)")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子 (默认42)")
    parser.add_argument("--output-dir", type=str,
                        default=os.environ.get(
                            "ROT_OUTPUT_DIR",
                            str(Path(__file__).resolve().parent / "adv_results" / "results_black"),
                        ),
                        help="结果保存目录 (默认: <RoT>/adv_results/results_black；可通过 ROT_OUTPUT_DIR 覆盖)")
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU设备号 (默认0)")
    parser.add_argument("--dataset", type=str, choices=["gsm8k", "MultiArith", "SVAMP"], default="gsm8k",
                        help="指定要攻击的数据集 (默认 gsm8k)")
    args = parser.parse_args()
    # python attack_black.py --problem-id 0

    # ============ 配置 ============
    DEVICE = f"cuda:{args.gpu}"
    print(f"RoT模型使用设备: {DEVICE}")

    # 随机种子
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

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
    DATA_NAME = args.dataset
    DATA_DIR = os.environ.get(
        "DATA_DIR",
        str(Path(__file__).resolve().parent.parent / "data"),
    )
    DATA_PATH = f"{DATA_DIR}/{DATA_NAME}.json"

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
    print("RoT 黑盒对抗攻击 - 问题改写")
    print(f"使用 {len(IRRELEVANT_FACTS)} 个无用事实进行测试")
    print("=" * 60)

    # ============ 1. 加载数据 ============
    print(f"\n[1] 加载{DATA_NAME}数据...")

    questions, cots, answers = load_local_dataset(DATA_PATH)

    question = questions[PROBLEM_ID]
    answer_str = str(answers[PROBLEM_ID])
    print(f"问题: {question[:100]}...")
    print(f"正确答案: {answer_str}")

    # ============ 2. 加载RoT模型 ============
    print("\n[2] 加载RoT模型...")

    # 加载配置
    with open(CONFIG, "r") as f:
        config = yaml.safe_load(f)

    # 加载模型
    model = load_model(
        checkpoint_path=CHECKPOINT,
        config=config,
        model_type="v2",
        verbose=True,
        stage1_checkpoint=STAGE1_CHECKPOINT
    )

    model.eval()
    print(f"RoT模型加载完成")

    # ============ 3. 辅助函数 ============

    def generate_answer(question_str: str, cot_str: str = ""):
        """生成答案"""
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

    def evaluate_question(question_str: str, ground_truth_str: float, cot_str: str = ""):
        """评估一个问题"""
        with torch.no_grad():
            text = generate_answer(question_str, cot_str)
            pred_answer = extract_answer_number(text)

            is_correct = abs(pred_answer - ground_truth_str) < 1e-9 if ground_truth_str != float('inf') else pred_answer == ground_truth_str

        return {
            "pred_answer": pred_answer,
            "is_correct": is_correct,
            "generated_text": text
        }

    # ============ 4. Baseline评估 ============
    print("\n[3] 测试baseline...")

    baseline_text = generate_answer(question, cots[PROBLEM_ID] if PROBLEM_ID < len(cots) else "")
    baseline_answer = extract_answer_number(baseline_text)

    print(f"Baseline生成文本: '{baseline_text[:100]}...'")
    print(f"Baseline提取答案: {baseline_answer}")
    print(f"正确答案: {answer_str}")

    baseline_correct = abs(baseline_answer - float(answer_str)) < 1e-9 if float(answer_str) != float('inf') else baseline_answer == float(answer_str)
    print(f"Baseline是否正确: {baseline_correct}")

    # ============ 5. 使用无用事实生成改写问题 ============
    print("\n[4] 使用无用事实生成改写问题...")

    print(f"\n评估原始问题...")
    original_eval = evaluate_question(question, float(answer_str), cots[PROBLEM_ID] if PROBLEM_ID < len(cots) else "")
    print(f"  correct: {original_eval['is_correct']}")

    # 生成所有改写问题
    print(f"\n测试 {len(IRRELEVANT_FACTS)} 个改写问题...")
    all_candidates = []
    for i, fact in enumerate(IRRELEVANT_FACTS):
        rw_q = f"{fact} {question}"
        print(f"  测试 [{i+1}/{len(IRRELEVANT_FACTS)}]: {fact[:40]}...")
        eval_result = evaluate_question(rw_q, float(answer_str), cots[PROBLEM_ID] if PROBLEM_ID < len(cots) else "")
        all_candidates.append({
            "question": rw_q,
            "fact": fact,
            "is_correct": eval_result["is_correct"],
            "generated_text": eval_result["generated_text"],
            "pred_answer": eval_result["pred_answer"],
        })
        print(f"    correct={eval_result['is_correct']}")

    # ============ 6. 选择最佳结果 ============
    print("\n" + "=" * 60)
    print("选择最佳改写")
    print("=" * 60)

    # 找攻击成功的
    success_candidates = [c for c in all_candidates if not c["is_correct"]]

    if success_candidates:
        # 有攻击成功的
        best_candidate = success_candidates[0]
        attack_success = True
    else:
        # 没有攻击成功的
        best_candidate = all_candidates[0] if all_candidates else None
        attack_success = False

    if best_candidate:
        best_question = best_candidate["question"]
        best_fact = best_candidate["fact"]

        print(f"最佳改写:")
        print(f"  问题: {best_question[:100]}...")
        print(f"  无用信息: {best_fact}")
        print(f"  攻击成功: {not best_candidate['is_correct']}")

        # 最终验证
        final_eval = evaluate_question(best_question, float(answer_str), cots[PROBLEM_ID] if PROBLEM_ID < len(cots) else "")
        print(f"\n最终验证:")
        print(f"  生成文本: '{final_eval['generated_text'][:100]}...'")
        print(f"  提取答案: {final_eval['pred_answer']}")
        print(f"  是否正确: {final_eval['is_correct']}")
    else:
        print("没有找到改写")
        best_candidate = None
        best_question = question
        best_fact = None
        final_eval = original_eval

    # ============ 7. 保存结果 ============
    print("\n[5] 保存结果...")

    result = {
        "question": question,
        "ground_truth": answer_str,
        "best_rewrite": best_question,
        "best_fact": best_fact,
        "baseline_correct": baseline_correct,
        "final_eval": {
            "is_correct": final_eval["is_correct"],
            "generated_text": final_eval["generated_text"],
            "pred_answer": final_eval["pred_answer"]
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