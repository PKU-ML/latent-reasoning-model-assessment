"""
================================================================================
PCCoT模型 随机前缀攻击
================================================================================

一、算法概述
-----------
本代码实现了一种针对PCCoT模型的随机前缀攻击，通过随机选取token作为前缀，
添加到问题前面，测试是否能导致模型输出错误答案。

二、攻击方式
-----------
- 随机选取5个token作为前缀
- 前缀直接添加在问题前面，中间不加空格
- 使用PCCoT模型评估，答案错误则攻击成功
- 重复多次，每次独立随机选取前缀

================================================================================
"""

import json
import torch
import numpy as np
import os
import re
import sys
import argparse
import shutil
from types import ModuleType
from pathlib import Path

# Fix flash_attn import issue
_flash_attn_path = None
for path in sys.path:
    flash_attn_dir = os.path.join(path, 'flash_attn')
    if os.path.isdir(flash_attn_dir):
        _flash_attn_path = flash_attn_dir
        break

if _flash_attn_path is not None:
    _backup_path = _flash_attn_path + '_backup'
    if os.path.isdir(_backup_path):
        pass
    elif os.path.exists(os.path.join(_flash_attn_path, '__init__.py')):
        try:
            import importlib.util
            spec = importlib.util.find_spec('flash_attn')
            if spec is None or spec.loader is None:
                raise ImportError("flash_attn is broken")
        except:
            shutil.move(_flash_attn_path, _backup_path)
            sys.path_importer_cache.clear()

sys.path.insert(0, os.environ.get("PCCOT_PROJECT_ROOT", str(Path(__file__).resolve().parent / "PCCoT")))

from transformers import AutoTokenizer, AutoConfig, HfArgumentParser
from transformers.utils.hub import cached_file
from peft import AutoPeftModel
from transformers import GenerationConfig

import models


def normalize_number(s):
    """Normalize number string for comparison."""
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
    parser = argparse.ArgumentParser(description="PCCoT 随机前缀攻击")
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
                            "PCCOT_OUTPUT_DIR",
                            str(Path(__file__).resolve().parent / "adv_results" / "results_random"),
                        ),
                        help="结果保存目录 (默认: <PCCoT>/adv_results/results_random；可通过 PCCOT_OUTPUT_DIR 覆盖)")
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
    parser.add_argument("--data-path", type=str, default=None,
                        help="数据集路径 (默认根据dataset参数自动确定)")
    args = parser.parse_args()

    # ============ 配置 ============
    DEVICE = f"cuda:{args.gpu}"
    print(f"使用设备: {DEVICE}")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    PREFIX_LENGTH = args.prefix_length
    NUM_TRIALS = args.num_trials

    MODEL_NAME_OR_PATH = args.model_path
    DATA_DIR = os.environ.get(
        "DATA_DIR",
        str(Path(__file__).resolve().parent.parent / "data"),
    )
    DATA_NAME = args.dataset
    if args.data_path:
        DATA_PATH = args.data_path
    else:
        DATA_PATH = f"{DATA_DIR}/{DATA_NAME}.json"
    os.makedirs(args.output_dir, exist_ok=True)

    if args.problem_id is not None:
        PROBLEM_ID = args.problem_id
    else:
        PROBLEM_ID = len(os.listdir(args.output_dir))

    print("=" * 60)
    print(f"PCCoT 随机前缀攻击 - 数据集: {DATA_NAME}")
    print(f"前缀长度: {PREFIX_LENGTH}")
    print(f"尝试次数: {NUM_TRIALS}")
    print("=" * 60)

    # ============ 1. 加载数据 ============
    print(f"\n[1] 加载{DATA_NAME}数据...")

    def load_local_dataset(data_path, dataset_name):
        questions = []
        answers = []
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for item in data:
            questions.append(item['question'])
            answer_text = item['answer'].replace(',', '')
            # 根据数据集使用不同的答案提取方式
            # gsm8k 格式: "... ### Answer" 或 "... #### Answer"
            # MultiArith/SVAMP: 直接是数字
            if dataset_name == "gsm8k":
                # GSM8K 答案通常在 "#### " 后面
                match = re.search(r'####\s*([\d\.\-]+)', answer_text)
                if match:
                    answer_text = match.group(1)
                else:
                    # 备选：取最后一个数字
                    nums = re.findall(r'-?\d+\.?\d*', answer_text)
                    if nums:
                        answer_text = nums[-1]
            else:
                # 其他数据集直接提取数字
                nums = re.findall(r'-?\d+\.?\d*', answer_text)
                if nums:
                    answer_text = nums[-1]
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

    # ============ 2. 加载PCCoT模型 ============
    print("\n[2] 加载PCCoT模型...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_OR_PATH)
    config = AutoConfig.from_pretrained(MODEL_NAME_OR_PATH)
    model = AutoPeftModel.from_pretrained(MODEL_NAME_OR_PATH)
    model.get_base_model().config = config

    pccot_args_file = cached_file(MODEL_NAME_OR_PATH, models.PCCOT_ARGS_NAME)
    parser = HfArgumentParser(models.PCCoTArguments)
    (pccot_args,) = parser.parse_json_file(json_file=pccot_args_file)

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

    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    model = model.to(DEVICE)
    model.eval()

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

        decoded_tokens = decoded_tokens[:, collated["input_ids"].shape[1]:]
        answers_list = tokenizer.batch_decode(decoded_tokens, skip_special_tokens=True)

        return answers_list[0] if answers_list else ""

    def is_correct(pred_answer, ground_truth):
        """判断答案是否正确"""
        if pred_answer == float('inf') or not ground_truth.lstrip('-').replace('.','',1).isdigit():
            return False
        return str(pred_answer) == ground_truth or abs(pred_answer - float(ground_truth)) < 0.01

    # ============ 4. Baseline评估 ============
    print("\n[3] 测试baseline...")

    baseline_text = generate_answer(question)
    baseline_answer = extract_answer_number(baseline_text)
    baseline_correct = is_correct(baseline_answer, answer_str)

    print(f"Baseline生成文本: '{baseline_text}'")
    print(f"Baseline提取答案: {baseline_answer}")
    print(f"正确答案: {answer_str}")
    print(f"Baseline是否正确: {baseline_correct}")

    # ============ 5. 构建有效token池 ============
    print("\n[4] 构建有效token池...")

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
    print(f"\n[5] 开始随机前缀攻击 ({NUM_TRIALS} 次)...")

    all_results = []
    success_count = 0

    for trial in range(NUM_TRIALS):
        # 随机选取PREFIX_LENGTH个token作为前缀
        prefix_ids = np.random.choice(char_tokens, size=PREFIX_LENGTH, replace=True)
        prefix_str = tokenizer.decode(prefix_ids)

        # 构建带前缀的问题 - 前缀直接加在问题前，中间不加空格
        prefixed_question = prefix_str.strip() + question.strip()

        # 评估
        generated_text = generate_answer(prefixed_question)
        pred_answer = extract_answer_number(generated_text)
        correct = is_correct(pred_answer, answer_str)
        attack_success = not correct

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
