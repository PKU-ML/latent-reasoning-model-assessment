"""
================================================================================
COLAR模型 随机前缀攻击
================================================================================

一、算法概述
-----------
本代码实现了一种针对COLAR模型的随机前缀攻击，通过随机选取token作为前缀，
添加到问题前面，测试是否能导致模型输出错误答案。

二、攻击目标数据集
-----------
- GSM8K
- MultiArith
- SVAMP

三、攻击方式
-----------
- 随机选取5个token作为前缀
- 前缀直接添加在问题前面，中间不加空格（参考attack_white.py）
- 使用COLAR模型评估，答案错误则攻击成功
- 重复多次，每次独立随机选取前缀

使用方法：
    python attack_random.py --dataset gsm8k --problem-id 0
    python attack_random.py --dataset MultiArith --problem-id 0
    python attack_random.py --dataset SVAMP --problem-id 0

================================================================================
"""

import json
import os
import sys
import re
import logging
import argparse
from pathlib import Path

import torch
import numpy as np

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def main():
    # ============ 命令行参数 ============
    parser = argparse.ArgumentParser(description="COLAR 随机前缀攻击")
    parser.add_argument("--problem-id", type=int, default=None,
                        help="指定攻击的问题ID (默认自动分配)")
    parser.add_argument("--num-trials", type=int, default=100,
                        help="随机前缀尝试次数 (默认100)")
    parser.add_argument("--prefix-length", type=int, default=5,
                        help="前缀token数量 (默认5)")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子 (默认42)")
    parser.add_argument("--output-dir", type=str,
                        default=os.environ.get(
                            "COLAR_OUTPUT_DIR",
                            str(Path(__file__).resolve().parent / "adv_results" / "results_random"),
                        ),
                        help="结果保存目录 (默认: <colar>/adv_results/results_random；可通过 COLAR_OUTPUT_DIR 环境变量覆盖)")
    parser.add_argument("--colar-gpu", type=int, default=0,
                        help="COLAR模型使用的GPU编号 (默认0)")
    parser.add_argument("--dataset", type=str, choices=["gsm8k", "MultiArith", "SVAMP"], default="gsm8k",
                        help="指定要攻击的数据集 (默认 gsm8k)")
    args = parser.parse_args()

    # ============ 配置 ============
    DEVICE = f"cuda:{args.colar_gpu}"
    print(f"COLAR模型使用设备: {DEVICE}")

    # 随机种子
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # 配置参数
    PREFIX_LENGTH = args.prefix_length
    NUM_TRIALS = args.num_trials

    # COLAR模型配置
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
    OUTPUT_DIR = args.output_dir
    os.makedirs(OUTPUT_DIR, exist_ok=True)

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
    print("COLAR 随机前缀攻击")
    print(f"数据集: {DATA_NAME}")
    print(f"前缀长度: {PREFIX_LENGTH}")
    print(f"尝试次数: {NUM_TRIALS}")
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
    print("\n[4] 测试baseline...")

    baseline_text = generate_answer(question, deterministic=True)
    baseline_answer = extract_answer_number(baseline_text)
    baseline_pred = extract_answer_from_output(baseline_text)

    print(f"Baseline生成文本: '{baseline_text[:100]}...'")
    print(f"Baseline提取答案(数字): {baseline_answer}")
    print(f"Baseline提取答案(原文): '{baseline_pred}'")
    print(f"正确答案: {answer_str}")

    baseline_correct = verify_answer(answer_str, baseline_pred)
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

        # 构建带前缀的问题 - 前缀直接加在问题前，中间不加空格（参考attack_white.py）
        prefixed_question = prefix_str.strip() + question.strip()

        # 评估（使用确定性模式避免随机性）
        generated_text = generate_answer(prefixed_question, deterministic=True)
        pred_answer = extract_answer_number(generated_text)
        pred_answer_str = extract_answer_from_output(generated_text)
        correct = verify_answer(answer_str, pred_answer_str)
        attack_success = not correct  # 答案错误则认为攻击成功

        if attack_success:
            success_count += 1

        result = {
            "trial": trial + 1,
            "prefix": prefix_str,
            "prefix_token_ids": [int(x) for x in prefix_ids],
            "generated_text": generated_text,
            "pred_answer": pred_answer,
            "pred_answer_str": pred_answer_str,
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
            "pred_answer_str": baseline_pred,
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

    save_path = os.path.join(args.output_dir, f"problem_{PROBLEM_ID}.json")
    with open(save_path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"\n结果已保存到 {save_path}")
    print(f"攻击效果: {'成功' if success_count > 0 else '失败'}")


if __name__ == "__main__":
    main()
