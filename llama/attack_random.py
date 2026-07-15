"""
================================================================================
Llama模型 随机前缀攻击
================================================================================

一、算法概述
-----------
本代码实现了一种针对Llama模型的随机前缀攻击，通过随机选取token作为前缀，
添加到问题前面，测试是否能导致模型输出错误答案。

二、攻击方式
-----------
- 随机选取5个token作为前缀
- 前缀直接添加在问题前面，中间不加空格
- 使用Llama模型评估，答案错误则攻击成功
- 重复多次，每次独立随机选取前缀

三、攻击流程
-----------
- 从test_org_results.json加载baseline正确的问题ID
- 初始化一次vLLM，依次攻击每个问题
- 每个问题独立随机前缀，独立保存结果

================================================================================
"""

import json
import numpy as np
import os
import re
import argparse
from typing import List, Dict
from pathlib import Path

from vllm import LLM, SamplingParams


# 不同数据集的答案提取方式
ANSWER_PATTERNS = {
    "gsm8k": r'The answer is (\d+)',
    "MultiArith": r'(\d+)',
    "SVAMP": r'(\d+)',
}


def load_local_dataset(data_path, dataset_name):
    """加载本地数据集"""
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


def extract_answer_number(sentence: str) -> float:
    """从生成的文本中提取答案数字"""
    matches = re.findall(r'\\boxed\{([^}]+)\}', sentence)
    if matches:
        try:
            return float(matches[-1])
        except ValueError:
            pass
    sentence = sentence.replace(',', '')
    pred = re.findall(r'-?\d+\.?\d*', sentence)
    if not pred:
        return float('inf')
    return float(pred[-1])


def load_problem_ids(results_file):
    """从结果文件加载correct_ids"""
    with open(results_file, 'r') as f:
        data = json.load(f)
    all_ids = []
    for dataset_name, dataset_data in data.items():
        for pid in dataset_data.get("correct_ids", []):
            all_ids.append((dataset_name, pid))
    return all_ids


def main():
    # ============ 命令行参数 ============
    parser = argparse.ArgumentParser(description="Llama 随机前缀攻击")
    parser.add_argument("--results-file", type=str,
                        default=os.environ.get(
                            "LLAMA_RESULTS_FILE",
                            str(Path(__file__).resolve().parent / "results" / "org" / "test_org_results.json"),
                        ),
                        help="包含correct_ids的结果文件 (默认: <llama>/results/org/test_org_results.json)")
    parser.add_argument("--num-trials", type=int, default=30,
                        help="随机前缀尝试次数 (默认100)")
    parser.add_argument("--prefix-length", type=int, default=5,
                        help="前缀token数量 (默认5)")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子 (默认42)")
    parser.add_argument("--output-dir", type=str,
                        default=os.environ.get(
                            "LLAMA_OUTPUT_DIR",
                            str(Path(__file__).resolve().parent / "adv_results" / "results_random"),
                        ),
                        help="结果保存目录 (默认: <llama>/adv_results/results_random)")
    parser.add_argument("--data-dir", type=str,
                        default=os.environ.get(
                            "DATA_DIR",
                            str(Path(__file__).resolve().parent.parent / "data"),
                        ),
                        help="数据目录 (默认: <repo>/data)")
    parser.add_argument("--dataset", type=str, default=None,
                        help="指定数据集 (gsm8k/MultiArith/SVAMP)，None表示所有")
    parser.add_argument("--problem-ids", type=str, default=None,
                        help="指定要攻击的问题ID，逗号分隔，如 '11,16,18'")
    parser.add_argument("--model-name-or-path", type=str,
                        default=os.environ.get(
                            "LLAMA_MODEL_PATH",
                            "meta-llama/Llama-3.2-1B-Instruct",
                        ),
                        help="模型名称或路径 (默认读取 LLAMA_MODEL_PATH 环境变量)")
    parser.add_argument("--tensor-parallel-size", type=int, default=1,
                        help="vLLM tensor parallel size")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.5,
                        help="vLLM GPU memory utilization")
    parser.add_argument("--max-model-len", type=int, default=4096,
                        help="vLLM max model length")
    parser.add_argument("--max-new-tokens", type=int, default=512,
                        help="生成的最大token数")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="生成温度")
    parser.add_argument("--use-chat-template", type=lambda x: x.lower() == 'true', default=True,
                        help="使用chat template")
    parser.add_argument("--token", type=str, default=None,
                        help="HuggingFace token")
    parser.add_argument("--skip-existing", action="store_true", default=False,
                        help="跳过输出目录中已存在的问题结果")
    args = parser.parse_args()

    # ============ 配置 ============
    print("=" * 60)
    print("Llama 随机前缀攻击")
    print("=" * 60)

    # 随机种子
    np.random.seed(args.seed)

    # 配置参数
    PREFIX_LENGTH = args.prefix_length
    NUM_TRIALS = args.num_trials

    # 如果指定了dataset，则在output_dir中添加dataset子目录
    if args.dataset:
        args.output_dir = os.path.join(args.output_dir, args.dataset)

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"前缀长度: {PREFIX_LENGTH}")
    print(f"尝试次数: {NUM_TRIALS}")

    # ============ 1. 加载问题ID ============
    print("\n[1] 加载问题ID...")
    problem_ids = []
    if args.problem_ids:
        # 命令行指定
        for pid in args.problem_ids.split(","):
            problem_ids.append(("", int(pid.strip())))
    else:
        # 从结果文件加载
        problem_ids = load_problem_ids(args.results_file)
        if args.dataset:
            problem_ids = [(ds, pid) for ds, pid in problem_ids if ds == args.dataset]

    print(f"共有 {len(problem_ids)} 个问题需要攻击")

    # ============ 跳过已攻击的问题 ============
    if args.skip_existing:
        existing_ids = []
        for dataset_name, PROBLEM_ID in problem_ids:
            save_path = os.path.join(args.output_dir, f"problem_{PROBLEM_ID}.json")
            if os.path.exists(save_path):
                existing_ids.append((dataset_name, PROBLEM_ID))

        if existing_ids:
            print(f"跳过 {len(existing_ids)} 个已攻击的问题")
            for ds, pid in existing_ids:
                print(f"  - {ds}_{pid}")
            problem_ids = [p for p in problem_ids if p not in set(existing_ids)]
            print(f"剩余 {len(problem_ids)} 个问题需要攻击")

    # ============ 2. 加载数据 ============
    print("\n[2] 加载数据...")
    data_dir = Path(args.data_dir)
    datasets = {
        "gsm8k": data_dir / "gsm8k.json",
        "MultiArith": data_dir / "MultiArith.json",
        "SVAMP": data_dir / "SVAMP.json",
    }
    all_data = {}
    for name, path in datasets.items():
        if path.exists():
            all_data[name] = load_local_dataset(str(path), name)

    # ============ 3. 加载vLLM模型 ============
    print("\n[3] 加载vLLM模型...")

    llm = LLM(
        model=args.model_name_or_path,
        hf_token=args.token,
        trust_remote_code=True,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
    )
    tokenizer = llm.get_tokenizer()

    print("vLLM模型加载完成")

    # ============ 4. 辅助函数 ============

    def make_prompt(question_str, prefix_str=""):
        """构建prompt - 前缀直接加在问题前，中间不加空格"""
        full_text = prefix_str.strip() + question_str.strip()
        if args.use_chat_template:
            messages = [{"role": "user", "content": full_text}]
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            messages = None
            prompt = full_text
        return prompt, messages

    def round2_prompt_from_messages(messages, first_round_text):
        """在messages中追加第一轮回复，然后追加第二轮提示，返回第二轮prompt"""
        messages.append({"role": "assistant", "content": first_round_text + "\nSo the answer is \\boxed{"})
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def round2_prompt_from_text(prompt, first_round_text):
        """在文本prompt后追加第一轮回复和第二轮提示，返回第二轮prompt"""
        return prompt + "\n" + first_round_text + "\nSo the answer is \\boxed{"

    def is_correct(pred_answer, ground_truth):
        """判断答案是否正确"""
        if pred_answer == float('inf') or not ground_truth.lstrip('-').replace('.','',1).isdigit():
            return False
        return str(pred_answer) == ground_truth or abs(pred_answer - float(ground_truth)) < 0.01

    sampling_params = SamplingParams(
        max_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=1.0,
        stop=None,
    )

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

    # ============ 6. 依次攻击每个问题 ============
    print("\n[5] 开始攻击...")
    print("=" * 60)

    all_results = {}

    for dataset_name, PROBLEM_ID in problem_ids:
        print(f"\n>>> 攻击数据集={dataset_name}, 问题ID={PROBLEM_ID}")
        print("-" * 40)

        questions = all_data[dataset_name][0]
        answers = all_data[dataset_name][1]

        question = questions[PROBLEM_ID]
        answer_str = str(answers[PROBLEM_ID])
        print(f"问题: {question[:80]}...")
        print(f"正确答案: {answer_str}")

        # ============ 6.1 Baseline评估（两轮生成） ============
        baseline_prompt, baseline_messages = make_prompt(question, "")
        baseline_output = llm.generate([baseline_prompt], sampling_params)[0]
        baseline_text = baseline_output.outputs[0].text

        # 第二轮
        if args.use_chat_template:
            baseline_round2_prompt = round2_prompt_from_messages(baseline_messages, baseline_text)
        else:
            baseline_round2_prompt = round2_prompt_from_text(baseline_prompt, baseline_text)
        baseline_round2_output = llm.generate([baseline_round2_prompt], sampling_params)[0]
        baseline_round2_text = baseline_round2_output.outputs[0].text
        baseline_full_text = baseline_text + "\nSo the answer is \\boxed{" + baseline_round2_text
        baseline_answer = extract_answer_number(baseline_full_text)
        baseline_correct = is_correct(baseline_answer, answer_str)

        print(f"Baseline: 提取答案={baseline_answer}, 正确={baseline_correct}")

        if not baseline_correct:
            print("  [跳过] baseline本身就答错了")
            all_results[f"{dataset_name}_{PROBLEM_ID}"] = {
                "dataset": dataset_name,
                "problem_id": PROBLEM_ID,
                "question": question,
                "ground_truth": answer_str,
                "skipped": True,
                "reason": "baseline incorrect"
            }
            continue

        # ============ 6.2 随机前缀攻击 (vLLM批处理 + 两轮生成) ============
        print(f"开始随机前缀攻击 ({NUM_TRIALS} 次, 每20个batch)...")

        trial_all_results = []
        success_count = 0
        BATCH_SIZE = 20

        for batch_start in range(0, NUM_TRIALS, BATCH_SIZE):
            batch_end = min(batch_start + BATCH_SIZE, NUM_TRIALS)

            # 批量生成prompts
            batch_prompts = []
            batch_messages = []
            batch_prefix_info = []
            for trial in range(batch_start, batch_end):
                # 随机选取PREFIX_LENGTH个token作为前缀
                prefix_ids = np.random.choice(char_tokens, size=PREFIX_LENGTH, replace=True)
                prefix_str = tokenizer.decode(prefix_ids)

                # 构建带前缀的问题
                prompt, messages = make_prompt(question, prefix_str)
                batch_prompts.append(prompt)
                batch_messages.append(messages)
                batch_prefix_info.append({
                    "trial": trial + 1,
                    "prefix": prefix_str,
                    "prefix_token_ids": [int(x) for x in prefix_ids],
                })

            # 第一轮批量评估
            outputs = llm.generate(batch_prompts, sampling_params)

            # 构建第二轮 prompts
            batch_round2_prompts = []
            for i, output in enumerate(outputs):
                first_round_text = output.outputs[0].text
                if args.use_chat_template:
                    round2_prompt = round2_prompt_from_messages(batch_messages[i], first_round_text)
                else:
                    round2_prompt = round2_prompt_from_text(batch_prompts[i], first_round_text)
                batch_round2_prompts.append(round2_prompt)

            # 第二轮批量评估
            round2_outputs = llm.generate(batch_round2_prompts, sampling_params)

            # 解析结果
            for i, (output, round2_output) in enumerate(zip(outputs, round2_outputs)):
                first_round_text = output.outputs[0].text
                second_round_text = round2_output.outputs[0].text
                generated_text = first_round_text + "\nSo the answer is \\boxed{" + second_round_text
                pred_answer = extract_answer_number(generated_text)
                correct = is_correct(pred_answer, answer_str)
                attack_success = not correct

                if attack_success:
                    success_count += 1

                result = {
                    **batch_prefix_info[i],
                    "generated_text": generated_text,
                    "pred_answer": pred_answer,
                    "correct": correct,
                    "attack_success": attack_success,
                }
                trial_all_results.append(result)

                if result["trial"] % 20 == 0 or attack_success:
                    status = "✓ 攻击成功" if attack_success else ""
                    print(f"  Trial {result['trial']}/{NUM_TRIALS}: correct={correct} pred={pred_answer} {status}")

        # ============ 6.3 结果统计 ============
        print(f"攻击成功次数: {success_count}/{NUM_TRIALS}")
        print(f"攻击成功率: {success_count/NUM_TRIALS*100:.2f}%")

        # 统计攻击成功的前缀
        successful_prefixes = [r for r in trial_all_results if r["attack_success"]]
        if successful_prefixes:
            print(f"攻击成功的前缀示例:")
            for r in successful_prefixes[:3]:
                print(f"  '{r['prefix']}' -> pred={r['pred_answer']}, correct={r['correct']}")

        # ============ 6.4 保存单个问题结果 ============
        result = {
            "dataset": dataset_name,
            "problem_id": PROBLEM_ID,
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
            "all_results": trial_all_results,
            "attack_success": success_count > 0,
        }

        save_path = os.path.join(args.output_dir, f"problem_{PROBLEM_ID}.json")
        with open(save_path, "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        print(f"结果已保存到 {save_path}")

        all_results[f"{dataset_name}_{PROBLEM_ID}"] = {
            "dataset": dataset_name,
            "problem_id": PROBLEM_ID,
            "question": question,
            "ground_truth": answer_str,
            "baseline_correct": baseline_correct,
            "attack_success": success_count > 0,
            "success_count": success_count,
        }

    # ============ 7. 保存汇总结果 ============
    print("\n" + "=" * 60)
    print("攻击完成 - 汇总结果")
    print("=" * 60)

    total = len(problem_ids)
    baseline_ok = sum(1 for r in all_results.values() if not r.get("skipped") and r.get("baseline_correct"))
    attack_ok = sum(1 for r in all_results.values() if r.get("attack_success"))

    print(f"总问题数: {total}")
    print(f"Baseline正确: {baseline_ok}")
    print(f"攻击成功: {attack_ok} ({100*attack_ok/baseline_ok:.1f}% if baseline_ok > 0 else 0)")

    summary_path = os.path.join(args.output_dir, "attack_summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n汇总结果保存到: {summary_path}")


if __name__ == "__main__":
    main()
