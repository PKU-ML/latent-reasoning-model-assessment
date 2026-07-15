"""
================================================================================
Llama模型 黑盒对抗攻击 - 基于问题改写
================================================================================

使用无用事实改写问题，通过vLLM并行评估，使模型输出错误答案。
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


# ============ 无用事实库 ============
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
    """从\\boxed{}中提取答案数字"""
    matches = re.findall(r'\\boxed\{([^}]+)\}', sentence)
    if matches:
        try:
            return float(matches[-1])
        except ValueError:
            pass
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
    parser = argparse.ArgumentParser(description="Llama 黑盒对抗攻击 - 问题改写")
    parser.add_argument("--results-file", type=str,
                        default=os.environ.get(
                            "LLAMA_RESULTS_FILE",
                            str(Path(__file__).resolve().parent / "results" / "org" / "test_org_results.json"),
                        ),
                        help="包含correct_ids的结果文件 (默认: <llama>/results/org/test_org_results.json)")
    parser.add_argument("--dataset", type=str, default="gsm8k",
                        help="指定数据集 (gsm8k/MultiArith/SVAMP)，None表示所有")
    parser.add_argument("--problem-ids", type=str, default=None,
                        help="指定要攻击的问题ID，逗号分隔，如 '11,16,18'")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子 (默认42)")
    parser.add_argument("--output-dir", type=str,
                        default=os.environ.get(
                            "LLAMA_OUTPUT_DIR",
                            str(Path(__file__).resolve().parent / "adv_results" / "results_black"),
                        ),
                        help="结果保存目录 (默认: <llama>/adv_results/results_black)")
    parser.add_argument("--data-dir", type=str,
                        default=os.environ.get(
                            "DATA_DIR",
                            str(Path(__file__).resolve().parent.parent / "data"),
                        ),
                        help="数据目录 (默认: <repo>/data)")
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
    parser.add_argument("--batch-size", type=int, default=16,
                        help="vLLM批处理大小")
    parser.add_argument("--skip-existing", action="store_true", default=False,
                        help="跳过输出目录中已存在的问题结果")
    args = parser.parse_args()

    # 如果指定了dataset，则在output_dir中添加dataset子目录
    if args.dataset:
        args.output_dir = os.path.join(args.output_dir, args.dataset)

    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # ============ 加载问题ID ============
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

    # ============ 加载数据 ============
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

    def make_prompt(question_str, prefix_str=""):
        """构建prompt"""
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

    sampling_params = SamplingParams(
        max_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=1.0,
        stop=None,
    )

    # ============ 4. 依次攻击每个问题 ============
    print("\n[4] 开始攻击...")
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

        # ============ 4.1 Baseline评估（两轮生成） ============
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

        baseline_correct = (str(baseline_answer) == answer_str or
                           abs(baseline_answer - float(answer_str)) < 0.01 if baseline_answer != float('inf') and answer_str.lstrip('-').replace('.','',1).isdigit() else False)
        print(f"Baseline: 提取答案={baseline_answer}, 正确={baseline_correct}")

        # ============ 4.2 并行评估所有改写问题（两轮生成） ============
        all_prompts = []
        all_messages = []
        fact_list = []
        for fact in IRRELEVANT_FACTS:
            rewritten_q = f"{fact} {question}"
            prompt, messages = make_prompt(rewritten_q, "")
            all_prompts.append(prompt)
            all_messages.append(messages)
            fact_list.append(fact)

        # 第一轮批量并行生成
        all_outputs = []
        for i in range(0, len(all_prompts), args.batch_size):
            batch_prompts = all_prompts[i:i+args.batch_size]
            outputs = llm.generate(batch_prompts, sampling_params)
            all_outputs.extend(outputs)

        # 构建第二轮 prompts
        all_round2_prompts = []
        for i, output in enumerate(all_outputs):
            first_round_text = output.outputs[0].text
            if args.use_chat_template:
                round2_prompt = round2_prompt_from_messages(all_messages[i], first_round_text)
            else:
                round2_prompt = round2_prompt_from_text(all_prompts[i], first_round_text)
            all_round2_prompts.append(round2_prompt)

        # 第二轮批量并行生成
        all_round2_outputs = []
        for i in range(0, len(all_round2_prompts), args.batch_size):
            batch_prompts = all_round2_prompts[i:i+args.batch_size]
            outputs = llm.generate(batch_prompts, sampling_params)
            all_round2_outputs.extend(outputs)

        # 解析结果
        all_candidates = []
        for fact, output, round2_output in zip(fact_list, all_outputs, all_round2_outputs):
            first_round_text = output.outputs[0].text
            second_round_text = round2_output.outputs[0].text
            generated_text = first_round_text + "\nSo the answer is \\boxed{" + second_round_text
            pred_answer = extract_answer_number(generated_text)
            is_correct = (str(pred_answer) == answer_str or
                         abs(pred_answer - float(answer_str)) < 0.01 if pred_answer != float('inf') and answer_str.lstrip('-').replace('.','',1).isdigit() else False)

            all_candidates.append({
                "question": f"{fact} {question}",
                "fact": fact,
                "is_correct": is_correct,
                "generated_text": generated_text,
                "pred_answer": pred_answer,
            })

        # ============ 4.3 选择最佳结果 ============
        failed_candidates = [c for c in all_candidates if not c["is_correct"]]
        num_failed = len(failed_candidates)

        if failed_candidates:
            best_candidate = failed_candidates[np.random.randint(0, len(failed_candidates))]
            attack_success = True
        else:
            best_candidate = all_candidates[np.random.randint(0, len(all_candidates))]
            attack_success = False

        print(f"改写后失败数: {num_failed}/{len(IRRELEVANT_FACTS)}")
        print(f"攻击成功: {attack_success}")
        print(f"  最佳无用信息: {best_candidate['fact']}")
        print(f"  预测答案: {best_candidate['pred_answer']}")

        # 保存单个结果
        result = {
            "dataset": dataset_name,
            "problem_id": PROBLEM_ID,
            "question": question,
            "ground_truth": answer_str,
            "baseline_correct": baseline_correct,
            "baseline_answer": baseline_answer,
            "best_rewrite": best_candidate["question"],
            "best_fact": best_candidate["fact"],
            "best_pred_answer": best_candidate["pred_answer"],
            "best_is_correct": best_candidate["is_correct"],
            "attack_success": attack_success,
            "num_facts": len(IRRELEVANT_FACTS),
            "num_failed": num_failed,
            "all_candidates": all_candidates
        }

        save_path = os.path.join(args.output_dir, f"problem_{PROBLEM_ID}.json")
        with open(save_path, "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        all_results[f"{dataset_name}_{PROBLEM_ID}"] = {
            "dataset": dataset_name,
            "problem_id": PROBLEM_ID,
            "question": question,
            "ground_truth": answer_str,
            "baseline_correct": baseline_correct,
            "attack_success": attack_success,
            "num_failed": num_failed,
        }

    # ============ 5. 保存汇总结果 ============
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
