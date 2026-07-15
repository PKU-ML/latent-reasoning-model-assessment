"""
================================================================================
PCCoT模型 黑盒对抗攻击 - 基于问题改写
================================================================================

一、算法概述
-----------
本代码实现了一种针对PCCoT模型的黑盒对抗攻击，通过在问题前添加无关事实，
使得模型在改写后的问题下输出错误的答案。

二、攻击目标
-----------
- 让模型在改写后的问题下输出错误的答案
- 只要攻击成功即可，不需要考虑cosine相似度

================================================================================
"""

import json
import torch
import numpy as np
import os
import logging
import re
import sys
import argparse
import shutil
from types import ModuleType
from typing import List, Dict, Tuple, Optional

# Fix flash_attn import issue - relocate broken flash_attn package
_flash_attn_path = None
for path in sys.path:
    flash_attn_dir = os.path.join(path, 'flash_attn')
    if os.path.isdir(flash_attn_dir):
        _flash_attn_path = flash_attn_dir
        break

if _flash_attn_path is not None:
    _backup_path = _flash_attn_path + '_backup'
    if os.path.isdir(_backup_path):
        pass  # Already relocated
    elif os.path.exists(os.path.join(_flash_attn_path, '__init__.py')):
        try:
            import importlib.util
            spec = importlib.util.find_spec('flash_attn')
            if spec is None or spec.loader is None:
                raise ImportError("flash_attn is broken")
        except:
            shutil.move(_flash_attn_path, _backup_path)
            sys.path_importer_cache.clear()

# Add PCCoT directory to path
# 默认指向仓库根目录下的 PCCoT 子项目；可通过 PCCOT_PROJECT_ROOT 环境变量覆盖
_pccot_project_root = os.environ.get(
    "PCCOT_PROJECT_ROOT",
    str(Path(__file__).resolve().parent / "PCCoT"),
)
sys.path.insert(0, _pccot_project_root)

from transformers import AutoTokenizer, AutoConfig, HfArgumentParser
from transformers.utils.hub import cached_file
from peft import AutoPeftModel
from transformers import GenerationConfig

import models

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


def normalize_number(s):
    """Normalize number string for comparison. Handles cases like '1' == '1.0'"""
    try:
        return str(float(s.strip()))
    except (ValueError, TypeError):
        return s.strip()


def extract_answer_number(sentence):
    """从生成的文本中提取答案数字"""
    sentence = sentence.replace(',', '')
    pred = [s for s in re.findall(r'-?\d+\.?\d*', sentence)]
    if not pred:
        return float('inf')
    return float(pred[-1])


def main():
    # ============ 命令行参数 ============
    parser = argparse.ArgumentParser(description="PCCoT 黑盒对抗攻击 - 问题改写")
    parser.add_argument("--problem-id", type=int, default=None,
                        help="指定攻击的问题ID (默认自动分配)")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子 (默认42)")
    parser.add_argument("--output-dir", type=str,
                        default=os.environ.get(
                            "PCCOT_OUTPUT_DIR",
                            str(Path(__file__).resolve().parent / "adv_results" / "results_black"),
                        ),
                        help="结果保存目录 (默认: <PCCoT>/adv_results/results_black；可通过 PCCOT_OUTPUT_DIR 覆盖)")
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU设备号 (默认0)")
    parser.add_argument("--model-path", type=str,
                        default=os.environ.get(
                            "PCCOT_MODEL_PATH",
                            "whyNLP/pccot-gpt2",
                        ),
                        help="PCCoT模型路径 (默认读取 PCCOT_MODEL_PATH 环境变量)")
    parser.add_argument("--dataset", type=str, choices=["gsm8k", "MultiArith", "SVAMP"], default="gsm8k",
                        help="指定要攻击的数据集 (默认 gsm8k)")
    args = parser.parse_args()
    # python attack_black.py --problem-id 0

    # ============ 配置 ============
    DEVICE = f"cuda:{args.gpu}"
    print(f"使用设备: {DEVICE}")

    # 随机种子
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # PCCoT模型配置
    MODEL_NAME_OR_PATH = args.model_path
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
    print("PCCoT 黑盒对抗攻击 - 问题改写")
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

    # ============ 2. 加载PCCoT模型 ============
    print("\n[2] 加载PCCoT模型...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_OR_PATH)
    config = AutoConfig.from_pretrained(MODEL_NAME_OR_PATH)
    model = AutoPeftModel.from_pretrained(MODEL_NAME_OR_PATH)

    # Override the model config after loading the model
    model.get_base_model().config = config

    # Load the PCCoT arguments
    pccot_args_file = cached_file(MODEL_NAME_OR_PATH, models.PCCOT_ARGS_NAME)
    parser = HfArgumentParser(models.PCCoTArguments)
    (pccot_args,) = parser.parse_json_file(json_file=pccot_args_file)

    # Add special tokens to tokenizer
    def get_special_token(tokenizer, token):
        if token not in tokenizer.additional_special_tokens:
            tokenizer.add_special_tokens({'additional_special_tokens': (token,)}, replace_additional_special_tokens=False)
        return tokenizer.convert_tokens_to_ids(token)

    if pccot_args.bot_token_id is None:
        pccot_args.bot_token_id = get_special_token(tokenizer, '<pccot.bot>')
    if pccot_args.eot_token_id is None:
        pccot_args.eot_token_id = get_special_token(tokenizer, '<pccot.eot>')
    if pccot_args.latent_token_id is None:
        pccot_args.latent_token_id = get_special_token(tokenizer, '<pccot.latent>')
    if tokenizer.pad_token_id is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    if tokenizer.eos_token_id is None:
        tokenizer.add_special_tokens({'eos_token': '[EOS]'})

    # Resize model embeddings if needed
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    model = model.to(DEVICE)
    model.eval()

    # Load the data processor
    data_processor = models.COTDataProcessor(
        tokenizer=tokenizer,
        pccot_args=pccot_args,
    )

    print(f"PCCoT模型加载完成")

    # ============ 3. 辅助函数 ============

    def generate_answer(question_str, max_new_tokens=30):
        """完整生成答案"""
        collated = data_processor.process(question_str, device=DEVICE)

        generation_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

        with torch.no_grad():
            decoded_tokens = model.generate(
                collated=collated,
                generation_config=generation_config,
            )

        # Remove input_ids part and decode
        decoded_tokens = decoded_tokens[:, collated["input_ids"].shape[1]:]
        answers_list = tokenizer.batch_decode(decoded_tokens, skip_special_tokens=True)

        return answers_list[0] if answers_list else "", answers_list

    # ============ 4. Baseline评估 ============
    print("\n[3] 测试baseline...")

    baseline_text, _ = generate_answer(question)
    baseline_answer = extract_answer_number(baseline_text)
    print(f"Baseline生成文本: '{baseline_text}'")
    print(f"Baseline提取答案: {baseline_answer}")
    print(f"正确答案: {answer_str}")

    baseline_correct = (str(baseline_answer) == answer_str or
                       abs(baseline_answer - float(answer_str)) < 0.01 if baseline_answer != float('inf') and answer_str.lstrip('-').replace('.','',1).isdigit() else False)
    print(f"Baseline是否正确: {baseline_correct}")

    # ============ 5. 评估函数 ============

    def evaluate_question(question_str):
        """评估一个问题，返回答案正确性"""
        text, _ = generate_answer(question_str)
        pred_answer = extract_answer_number(text)
        is_correct = (str(pred_answer) == answer_str or
                     abs(pred_answer - float(answer_str)) < 0.01 if pred_answer != float('inf') and answer_str.lstrip('-').replace('.','',1).isdigit() else False)

        return {
            "pred_answer": pred_answer,
            "is_correct": is_correct,
            "generated_text": text,
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
        })
        print(f"    correct={eval_result['is_correct']}")

    # ============ 7. 选择最佳结果 ============
    print("\n" + "=" * 60)
    print("选择最佳改写")
    print("=" * 60)

    if all_candidates:
        # 找攻击成功的（正确答案概率最低的）
        best_candidate = None
        for c in all_candidates:
            if not c["is_correct"]:
                if best_candidate is None:
                    best_candidate = c
                else:
                    # 随机选择一个成功的
                    best_candidate = c
                    break

        # 如果没有成功的，选择任意一个
        if best_candidate is None:
            best_candidate = all_candidates[0]

        best_question = best_candidate["question"]
        best_fact = best_candidate["fact"]

        print(f"最佳改写:")
        print(f"  问题: {best_question[:100]}...")
        print(f"  无用信息: {best_fact}")
        print(f"  攻击成功: {not best_candidate['is_correct']}")

        final_eval = evaluate_question(best_question)
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