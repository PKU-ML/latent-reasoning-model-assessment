"""
================================================================================
COLAR模型 黑盒对抗攻击 - 基于问题改写
================================================================================

一、算法概述
-----------
本代码实现了一种针对COLAR模型的黑盒对抗攻击，通过在问题前添加无关事实，
使得模型在改写后的问题下输出错误的答案。

二、攻击目标
-----------
- 让模型在改写后的问题下输出错误的答案
- 只要攻击成功即可，不需要考虑cosine相似度

================================================================================
"""

import json
import os
import sys
import re
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import torch
import numpy as np

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

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
    parser = argparse.ArgumentParser(description="COLAR 黑盒对抗攻击 - 问题改写")
    parser.add_argument("--problem-id", type=int, default=None,
                        help="指定攻击的问题ID (默认自动分配)")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子 (默认42)")
    parser.add_argument("--output-dir", type=str,
                        default=os.environ.get(
                            "COLAR_OUTPUT_DIR",
                            str(Path(__file__).resolve().parent / "adv_results" / "results_black"),
                        ),
                        help="结果保存目录 (默认: <colar>/adv_results/results_black；可通过 COLAR_OUTPUT_DIR 环境变量覆盖)")
    parser.add_argument("--colar-gpu", type=int, default=0,
                        help="COLAR模型使用的GPU编号 (默认0)")
    parser.add_argument("--dataset", type=str, choices=["gsm8k", "MultiArith", "SVAMP"], default="gsm8k",
                        help="指定要攻击的数据集 (默认 gsm8k)")
    args = parser.parse_args()
    # python attack_black.py --problem-id 0

    # ============ 配置 ============
    DEVICE = f"cuda:{args.colar_gpu}"
    print(f"COLAR模型使用设备: {DEVICE}")

    # 随机种子
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # COLAR模型配置
    # 工作目录与 checkpoint 通过环境变量注入；未设置时使用仓库内同级 colar/ 子目录
    WORKSPACE_PATH = os.environ.get(
        "COLAR_WORKSPACE",
        str(Path(__file__).resolve().parent / "colar"),
    )
    CHECKPOINT_PATH = os.environ.get(
        "COLAR_CHECKPOINT",
        str(
            Path(__file__).resolve().parent
            / "colar"
            / "models"
            / "CoLaR"
            / "logs"
            / "colar"
            / "qsa-gsm"
            / "colar-final"
            / "checkpoints"
            / "colar_best.ckpt"
        ),
    )
    if not Path(WORKSPACE_PATH).exists():
        print(f"[警告] COLAR_WORKSPACE={WORKSPACE_PATH} 不存在，请通过环境变量指向 colar 仓库根目录")
    if not Path(CHECKPOINT_PATH).exists():
        print(f"[警告] COLAR_CHECKPOINT={CHECKPOINT_PATH} 不存在，请通过环境变量指向实际的 .ckpt 文件")
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
        existing_files = list(Path(args.output_dir).glob("problem_*.json"))
        if existing_files:
            max_id = max([int(f.stem.split('_')[1]) for f in existing_files])
            PROBLEM_ID = max_id + 1
        else:
            PROBLEM_ID = 0

    print("=" * 60)
    print("COLAR 黑盒对抗攻击 - 问题改写")
    print(f"使用 {len(IRRELEVANT_FACTS)} 个无用事实进行测试")
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

    # ============ 2. 加载COLAR模型 ============
    print("\n[2] 加载COLAR模型...")

    os.chdir(WORKSPACE_PATH)

    import yaml
    from omegaconf import OmegaConf
    from colar.src.models.colar import LitCoLaR
    from colar.src.utils.utils import get_position_ids_from_attention_mask

    # Load hparams.yaml to get config
    hparams_path = os.path.dirname(CHECKPOINT_PATH).replace('/checkpoints', '') + '/hparams.yaml'
    logger.info(f"Loading hparams from {hparams_path}")

    with open(hparams_path, 'r') as f:
        hparams_data = yaml.safe_load(f)

    all_config = OmegaConf.create(hparams_data['all_config'])

    # Add args to config
    all_config.args = OmegaConf.create({
        "workspace_path": WORKSPACE_PATH,
        "no_log": True,
    })

    # Create model
    model = LitCoLaR(
        model_kwargs=all_config.model.model_kwargs,
        training_kwargs=all_config.model.training_kwargs,
        all_config=all_config,
    )

    # Load checkpoint weights
    logger.info(f"Loading checkpoint from {CHECKPOINT_PATH}")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    state_dict = checkpoint["state_dict"]
    model.load_state_dict(state_dict=state_dict, strict=False)

    tokenizer = model.tokenizer

    model.eval()
    model.to(DEVICE)

    print(f"COLAR模型加载完成")

    # ============ 3. 辅助函数 ============

    def get_pure_string(s: str):
        """标准化答案字符串"""
        return s.strip("#\n ").rstrip(".").replace(",", "").lower()

    def extract_answer_number(sentence: str) -> float:
        """从生成的文本中提取答案数字"""
        sentence = sentence.replace(',', '')
        pred = re.findall(r'-?\d+\.?\d*', sentence)
        if not pred:
            return float('inf')
        return float(pred[-1])

    def extract_answer_from_output(output_string: str) -> str:
        """从模型输出中提取答案"""
        try:
            return output_string.strip('#').split(model.answer_template.format(""))[-1]
        except (ValueError, IndexError):
            return output_string

    def verify_answer(gt_answer: str, pred_answer: str) -> bool:
        """验证答案是否正确"""
        gt_str = get_pure_string(gt_answer)
        pred_str = get_pure_string(pred_answer)

        try:
            gt_num = float(gt_str)
            pred_num = float(pred_str)
            return abs(gt_num - pred_num) < 1e-6
        except ValueError:
            pass

        return gt_str == pred_str

    def generate_answer(question_str: str, deterministic=False):
        """使用COLAR模型生成答案

        Args:
            question_str: 输入问题
            deterministic: 是否使用确定性模式（不随机采样）
        """
        with torch.no_grad():
            if deterministic:
                # 保存原始配置
                orig_answer_config = model.model_kwargs.answer_generation_config.copy()
                orig_latent_temp = model.model_kwargs.latent_generation_config.get("latent_temperature", 1.0)

                # 设置确定性模式：do_sample=False + 极小temperature
                model.model_kwargs.answer_generation_config.do_sample = False
                # 使用极小值而非0，避免 Normal分布 scale=0 的错误
                model.model_kwargs.latent_generation_config.latent_temperature = 1e-9

                try:
                    pred_ids, n_latent_forward = model.latent_generate(questions=[question_str])
                finally:
                    # 恢复原始配置
                    model.model_kwargs.answer_generation_config.do_sample = orig_answer_config.get("do_sample", True)
                    model.model_kwargs.latent_generation_config.latent_temperature = orig_latent_temp
            else:
                pred_ids, n_latent_forward = model.latent_generate(questions=[question_str])

            output_string = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)[0]
            return output_string

    # ============ 4. Baseline评估 ============
    print("\n[3] 测试baseline...")

    baseline_text = generate_answer(question, deterministic=True)
    baseline_answer = extract_answer_number(baseline_text)
    baseline_pred = extract_answer_from_output(baseline_text)

    print(f"Baseline生成文本: '{baseline_text[:100]}...'")
    print(f"Baseline提取答案(数字): {baseline_answer}")
    print(f"Baseline提取答案(原文): '{baseline_pred}'")
    print(f"正确答案: {answer_str}")

    baseline_correct = verify_answer(answer_str, baseline_pred)
    print(f"Baseline是否正确: {baseline_correct}")

    # ============ 5. 评估函数 ============

    def evaluate_question(question_str: str):
        """评估一个问题，返回评估结果（确定性模式）"""
        with torch.no_grad():
            # 生成答案（使用确定性模式，避免随机性）
            text = generate_answer(question_str, deterministic=True)
            pred_answer = extract_answer_from_output(text)
            pred_number = extract_answer_number(text)
            is_correct = verify_answer(answer_str, pred_answer)

        return {
            "pred_answer": pred_answer,
            "pred_number": pred_number,
            "is_correct": is_correct,
            "generated_text": text
        }

    # ============ 6. 使用无用事实生成改写问题 ============
    print("\n[4] 使用无用事实生成改写问题...")

    print(f"\n评估原始问题...")
    original_eval = evaluate_question(question)
    print(f"  correct: {original_eval['is_correct']}")

    # 生成所有改写问题
    print(f"\n测试 {len(IRRELEVANT_FACTS)} 个改写问题...")
    all_candidates = []
    for i, fact in enumerate(IRRELEVANT_FACTS):
        rw_q = f"{fact} {question}"
        print(f"  测试 [{i+1}/{len(IRRELEVANT_FACTS)}]: {fact[:40]}...")
        eval_result = evaluate_question(rw_q)
        all_candidates.append({
            "question": rw_q,
            "fact": fact,
            "is_correct": eval_result["is_correct"],
            "generated_text": eval_result["generated_text"],
            "pred_answer": eval_result["pred_answer"],
        })
        print(f"    correct={eval_result['is_correct']}")

    # ============ 7. 选择最佳结果 ============
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
        final_eval = evaluate_question(best_question)
        print(f"\n最终验证:")
        print(f"  生成文本: '{final_eval['generated_text'][:100]}...'")
        print(f"  提取答案: '{final_eval['pred_answer']}'")
        print(f"  是否正确: {final_eval['is_correct']}")
    else:
        print("没有找到改写")
        best_candidate = None
        best_question = question
        best_fact = None
        final_eval = original_eval

    # ============ 8. 保存结果 ============
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