#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

"""
Llama 模型在多个数学推理数据集上的测试脚本

默认配置：
- 不使用 CoT 提示词
- 使用贪心解码 (temperature=0, do_sample=False) 确保结果可复现

功能：
1. 加载本地 JSON 格式的数据集（gsm8k, MultiArith, SVAMP）
2. 在多个数据集上进行评估
3. 支持从 gcg_results 目录加载前缀，添加到问题前
4. 支持评估模型之前做对的问题（correct_ids）
5. 支持评估前 N 个问题（可单独使用，也可与 --eval_correct_ids_from 组合）

使用方法：
    python test.py                                    # 评估所有数据集的所有问题
    python test.py --dataset gsm8k                   # 只评估 gsm8k 数据集
    python test.py --question_id 0                   # 评估所有数据集的第0个问题
    python test.py --dataset gsm8k --question_id 0  # 只评估 gsm8k 数据集的第0个问题
    python test.py --gcg_results_dir ./gcg_results   # 使用 gcg_results 中的前缀
    python test.py --gcg_results_dir ./gcg_results --dataset gsm8k --question_id 0
    python test.py --eval_correct_ids_from ./test_results.json  # 评估之前做对的问题
    python test.py --eval_first_n 10                # 评估前10个问题
    python test.py --eval_correct_ids_from ./test_results.json --eval_first_n 30  # 评估之前做对的前30个题
    python test.py --temperature 0.5 --do_sample True  # 使用随机采样
"""

import logging
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from tqdm import tqdm

import json

from vllm import LLM, SamplingParams

# 是否打印每个问题的详细信息
do_print = False


def load_local_dataset(data_path):
    """
    从本地 JSON 文件加载数据集

    参数：
    - data_path: JSON 文件路径

    返回：
    - 问题列表和答案列表
    """
    questions = []
    answers = []

    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for item in data:
        questions.append(item['question'])
        # 处理答案：直接使用 answer 字段
        answer_text = item['answer']
        # 处理数字中的逗号
        answer_text = answer_text.replace(',', '')
        # 转换为浮点数
        try:
            ans = float(answer_text)
        except ValueError:
            ans = float("inf")

        answers.append(ans)

    return questions, answers


def evaluation(model_name_or_path, data_args, use_chat_template=False,
               question_ids=None, gcg_prefixes=None, gen_kwargs=None, token=None,
               dataset_name=None):
    # =========================================================================
    # 步骤1: 加载 vLLM 模型
    # =========================================================================
    logging.warning(f"Loading vLLM model from {model_name_or_path}")

    llm = LLM(
        model=model_name_or_path,
        hf_token=token,
        trust_remote_code=True,
        tensor_parallel_size=gen_kwargs.get("tensor_parallel_size", 1),
        gpu_memory_utilization=gen_kwargs.get("gpu_memory_utilization", 0.9),
        max_model_len=gen_kwargs.get("max_model_len", 4096),
    )

    # =========================================================================
    # 步骤2: 加载本地数据集
    # =========================================================================
    logging.warning(f"Loading local dataset from: {data_args.data_path}")

    all_questions, answer = load_local_dataset(data_args.data_path)

    # 如果指定了 question_ids，只评估这些id
    eval_indices = list(question_ids) if question_ids is not None else range(len(all_questions))
    questions = [all_questions[i] for i in eval_indices]

    print(f"Loaded {len(all_questions)} questions from local dataset")
    print(f"Sample question: {all_questions[0][:100]}...")
    print(f"Will evaluate {len(questions)} questions")

    # =========================================================================
    # 步骤3: 准备 prompts
    # =========================================================================
    prompts = []
    messages_list = []
    raw_prompts = []
    input_lens = []

    for orig_idx, q in zip(eval_indices, questions):
        # 添加 gcg_prefix
        if gcg_prefixes and (dataset_name, orig_idx) in gcg_prefixes:
            q = gcg_prefixes[(dataset_name, orig_idx)] + q

        prompts.append(q)
        messages_list.append([{"role": "user", "content": q}])
        raw_prompts.append(q)

    # =========================================================================
    # 步骤4: vLLM 第一轮批量推理
    # =========================================================================
    sampling_params = SamplingParams(
        max_tokens=gen_kwargs.get("max_new_tokens", 256),
        temperature=gen_kwargs.get("temperature", 0.0),
        top_p=gen_kwargs.get("top_p", 0.95),
        stop=gen_kwargs.get("stop", None),
    )

    # 构建 prompt token ids 以获取 input lengths
    tokenizer = llm.get_tokenizer()
    for p in raw_prompts:
        input_ids = tokenizer.encode(p, add_special_tokens=True)
        input_lens.append(len(input_ids))

    print(f"Starting vLLM generation for {len(prompts)} questions...")

    if use_chat_template:
        chat_prompts = [
            llm.get_tokenizer().apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            for messages in messages_list
        ]
        outputs = llm.generate(chat_prompts, sampling_params)
    else:
        outputs = llm.generate(prompts, sampling_params)

    # =========================================================================
    # 步骤5: vLLM 第二轮推理 - 在回复后追加 "So the answer is \boxed{"
    # =========================================================================
    round2_prompts = []
    for i, output in enumerate(outputs):
        first_round_text = output.outputs[0].text
        if use_chat_template:
            messages = messages_list[i]
            messages.append({"role": "assistant", "content": first_round_text + "\nSo the answer is \\boxed{"})
            round2_prompt = llm.get_tokenizer().apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            round2_prompt = prompts[i] + "\n" + first_round_text + "\nSo the answer is \\boxed{"
        round2_prompts.append(round2_prompt)

    print(f"Starting vLLM round-2 generation for {len(round2_prompts)} questions...")
    round2_outputs = llm.generate(round2_prompts, sampling_params)

    # =========================================================================
    # 步骤6: 解码并提取答案
    # =========================================================================
    results = []
    ans_pred_list = []
    len_cot = []

    for i, (output, round2_output) in enumerate(zip(tqdm(outputs, desc="Evaluating"), round2_outputs)):
        orig_idx = eval_indices[i]
        first_round_text = output.outputs[0].text
        second_round_text = round2_output.outputs[0].text
        generated_text = first_round_text + "So the answer is \\boxed{" + second_round_text

        # 计算 COT 长度（估算）
        gen_tokens = tokenizer.encode(generated_text)
        len_cot.append(len(gen_tokens))

        pred_answer = extract_answer_number(generated_text)
        gold_answer = answer[orig_idx]

        if do_print:
            print(f"Question {orig_idx} Starts...")
            print(f"Q: {all_questions[orig_idx]}")
            print(generated_text)
            print(f"Question {orig_idx} Ends")
            print(f"Prediction={pred_answer}; Groundtruth={gold_answer}")
            print("")

        results.append({
            'id': orig_idx,
            'question': all_questions[orig_idx],
            'ground_truth': gold_answer,
            'prediction': pred_answer,
            'model_output': generated_text,
            'correct': pred_answer == gold_answer
        })

        ans_pred_list.append(pred_answer)

    # =========================================================================
    # 步骤7: 计算准确率
    # =========================================================================
    eval_answers = [answer[i] for i in eval_indices]
    accuracy = compute_accuracy(eval_answers, ans_pred_list)

    # 打印结果
    print(f"Dataset: {data_args.data_name} | "
          f"Accuracy: {100*accuracy:.2f}% | ")
    print(f"average length of COT: {sum(len_cot)/len(len_cot) if len_cot else 0}")

    return accuracy, results


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


def compute_accuracy(gold: list, pred: list) -> float:
    """
    计算预测准确率
    """
    acc = 0.0
    for p, g in zip(pred, gold):
        if isinstance(p, list):
            if g in p:
                acc += 1
        else:
            if p == g:
                acc += 1

    return acc / len(gold)


@dataclass
class DataArguments:
    """
    数据配置参数
    """
    data_name: str = None
    data_path: str = None
    batch_size: int = 1
    output_path: str = './test_results.jsonl'


# =========================================================================
# 主程序入口
# =========================================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Llama Multi-Dataset Test')
    parser.add_argument('--dataset', type=str, default=None,
                        help='指定要评估的数据集名称 (gsm8k, MultiArith, SVAMP)。如果为空，则评估所有数据集。')
    parser.add_argument('--question_id', type=int, default=None,
                        help='指定要评估的问题索引。如果为空，则评估所有问题（可配合其他参数使用）。')
    parser.add_argument('--data_path', type=str, default=None,
                        help='Path to local JSON dataset (not used in multi-dataset mode)')
    parser.add_argument('--output_path', type=str, default='./test_results.jsonl',
                        help='Path to save test results')
    parser.add_argument('--model_name_or_path', type=str,
                        default=os.environ.get(
                            "LLAMA_MODEL_PATH",
                            "meta-llama/Llama-3.2-1B-Instruct",
                        ),
                        help='Path to pretrained model (默认读取 LLAMA_MODEL_PATH 环境变量)')
    parser.add_argument('--token', type=str, default=None,
                        help='HuggingFace token')
    parser.add_argument('--max_new_tokens', type=int, default=1024,
                        help='Max new tokens for generation')
    parser.add_argument('--temperature', type=float, default=0,
                        help='Temperature for generation (default 0 for no randomness)')
    parser.add_argument('--top_p', type=float, default=1.0,
                        help='Top p for generation')
    parser.add_argument('--use_chat_template', type=lambda x: x.lower() == 'true', default=True,
                        help='Use chat template for model input')
    parser.add_argument('--tensor_parallel_size', type=int, default=1,
                        help='vLLM tensor parallel size (default 1)')
    parser.add_argument('--gpu_memory_utilization', type=float, default=0.5,
                        help='vLLM GPU memory utilization (default 0.5)')
    parser.add_argument('--max_model_len', type=int, default=4096,
                        help='vLLM max model length (default 4096)')
    parser.add_argument('--gcg_results_dir', type=str, default=None,
                        help='gcg_results 目录路径，包含 prefix JSON 文件。如果为空，则不使用前缀。')
    parser.add_argument('--eval_correct_ids_from', type=str, default=None,
                        help='从指定的结果JSON文件读取correct_ids，只评估这些id的问题。')
    parser.add_argument('--eval_first_n', type=int, default=None,
                        help='评估前N个问题。')
    parser.add_argument('--select_ids', type=str,
                        default=os.environ.get(
                            "LLAMA_SELECT_IDS",
                            str(Path(__file__).resolve().parent.parent / "data" / "select.json"),
                        ),
                        help='指定包含各数据集问题ID的JSON文件路径 (默认: <repo>/data/select.json)。')
    parser.add_argument('--output_dir', type=str,
                        default=os.environ.get(
                            "LLAMA_OUTPUT_DIR",
                            str(Path(__file__).resolve().parent / "results"),
                        ),
                        help='输出结果目录 (默认: <llama>/results；可通过 LLAMA_OUTPUT_DIR 环境变量覆盖)')

    args = parser.parse_args()
    #python test.py --eval_correct_ids_from results/org/test_org_results.json --gcg_results_dir adv_results/results_black 


    # 创建生成参数
    gen_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "tensor_parallel_size": args.tensor_parallel_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_model_len": args.max_model_len,
    }

    # 数据目录（默认指向仓库根目录的 data/；可通过 DATA_DIR 环境变量覆盖）
    data_dir = Path(os.environ.get("DATA_DIR", str(Path(__file__).resolve().parent.parent / "data")))

    # 定义要评估的数据集
    datasets = {
        "gsm8k": data_dir / "gsm8k.json",
        "MultiArith": data_dir / "MultiArith.json",
        "SVAMP": data_dir / "SVAMP.json",
    }

    # 如果指定了dataset，过滤只评估该数据集
    if args.dataset is not None:
        if args.dataset not in datasets:
            logging.warning(f"Dataset '{args.dataset}' not found. Available: {list(datasets.keys())}")
            sys.exit(1)
        datasets = {args.dataset: datasets[args.dataset]}

    # 加载 gcg_results 前缀（按数据集子目录组织）
    gcg_prefixes = {}
    if args.gcg_results_dir is not None:
        gcg_base_dir = Path(args.gcg_results_dir)
        if gcg_base_dir.exists():
            for dataset_name in ["gsm8k", "MultiArith", "SVAMP"]:
                gcg_subdir = gcg_base_dir / dataset_name
                if not gcg_subdir.exists():
                    continue
                for json_file in gcg_subdir.glob("problem_*.json"):
                    qid = int(json_file.stem.split("_")[1])
                    with open(json_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        if "random" in args.gcg_results_dir:
                            try:
                                gcg_prefixes[(dataset_name, qid)] = [re for re in data["all_results"] if re['correct']==False][0]['prefix']
                            except:
                                gcg_prefixes[(dataset_name, qid)] = data["all_results"][0]['prefix']
                        elif "black" in args.gcg_results_dir:
                            gcg_prefixes[(dataset_name, qid)] = data["best_fact"]
                        else:
                            gcg_prefixes[(dataset_name, qid)] = data.get("prefix", "")
            print(f"Loaded {len(gcg_prefixes)} gcg prefixes from {gcg_base_dir}")
        else:
            logging.warning(f"gcg_results_dir not found: {gcg_base_dir}")

    # 加载 correct_ids（模型能做对的问题ID）
    correct_ids_set = None
    if args.eval_correct_ids_from is not None:
        correct_ids_file = Path(args.eval_correct_ids_from)
        if correct_ids_file.exists():
            with open(correct_ids_file, 'r', encoding='utf-8') as f:
                correct_ids_data = json.load(f)
            # 读取每个数据集的 correct_ids
            correct_ids_set = {}
            for dataset_name in correct_ids_data:
                correct_ids_set[dataset_name] = set(correct_ids_data[dataset_name].get("correct_ids", []))
            print(f"Loaded correct_ids from {correct_ids_file}: { {k: len(v) for k, v in correct_ids_set.items()} }")
        else:
            logging.warning(f"correct_ids file not found: {correct_ids_file}")

    # 加载 select_ids
    select_ids = {}
    if args.select_ids is not None:
        select_ids_file = Path(args.select_ids)
        if select_ids_file.exists():
            with open(select_ids_file, 'r', encoding='utf-8') as f:
                select_ids = json.load(f)
            print(f"Loaded select_ids from {select_ids_file}: { {k: len(v) for k, v in select_ids.items()} }")
        else:
            logging.warning(f"select_ids file not found: {select_ids_file}")

    # 存储所有结果
    all_results = {}
    all_samples = {}

    # 对每个数据集进行评估
    for dataset_name, dataset_path in datasets.items():
        if not dataset_path.exists():
            logging.warning(f"Dataset not found: {dataset_path}")
            continue

        logging.warning(f"\n{'='*50}")
        logging.warning(f"Evaluating on {dataset_name} dataset")

        # 确定要评估的问题ID
        eval_ids = None
        if args.question_id is not None:
            # 优先使用指定的单个 question_id
            eval_ids = [args.question_id]
            logging.warning(f"Evaluating question ID: {args.question_id}")
        elif args.eval_first_n is not None and correct_ids_set is not None and dataset_name in correct_ids_set:
            # 从 correct_ids 中取前 N 个（同时过滤 select_ids）
            all_correct_ids = list(correct_ids_set[dataset_name])
            if dataset_name in select_ids:
                all_correct_ids = [cid for cid in all_correct_ids if cid in select_ids[dataset_name]]
            eval_ids = all_correct_ids[:args.eval_first_n]
            logging.warning(f"Evaluating first {args.eval_first_n} of {len(all_correct_ids)} correct_ids (filtered by select_ids)")
        elif args.eval_first_n is not None:
            # 评估前 N 个问题（从 select_ids 中）
            if dataset_name in select_ids:
                eval_ids = select_ids[dataset_name][:args.eval_first_n]
            else:
                eval_ids = list(range(args.eval_first_n))
            logging.warning(f"Evaluating first {args.eval_first_n} questions from select_ids")
        elif correct_ids_set is not None and dataset_name in correct_ids_set:
            # 评估 correct_ids 中的所有问题（同时过滤 select_ids）
            all_correct_ids = list(correct_ids_set[dataset_name])
            if dataset_name in select_ids:
                all_correct_ids = [cid for cid in all_correct_ids if cid in select_ids[dataset_name]]
            eval_ids = all_correct_ids
            logging.warning(f"Evaluating {len(eval_ids)} correct_ids from previous run (filtered by select_ids)")
        elif dataset_name in select_ids:
            # 默认使用 select_ids
            eval_ids = select_ids[dataset_name]
            logging.warning(f"Using select_ids: {len(eval_ids)} questions")

        if eval_ids is not None:
            logging.warning(f"Total questions to evaluate: {len(eval_ids)}")
        logging.warning(f"{'='*50}")

        data_args = DataArguments(
            data_name=dataset_name,
            data_path=str(dataset_path),
            batch_size=1,
            output_path=args.output_path,
        )

        # 运行评估
        accuracy, samples = evaluation(
            args.model_name_or_path, data_args,
            use_chat_template=args.use_chat_template,
            question_ids=eval_ids,
            gcg_prefixes=gcg_prefixes if gcg_prefixes else None,
            gen_kwargs=gen_kwargs,
            token=args.token,
            dataset_name=dataset_name,
        )

        # 统计正确样本的ID
        correct_ids = [s['id'] for s in samples if s['correct']]

        # 保存结果
        all_results[dataset_name] = {
            "accuracy": accuracy,
            "num_samples": len(samples),
            "num_correct": len(correct_ids),
            "correct_ids": correct_ids,
        }
        all_samples[dataset_name] = samples

    # 保存结果到文件
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "test_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    # 保存详细结果
    output_file_samples = output_dir / "test_results_samples.json"
    with open(output_file_samples, 'w', encoding='utf-8') as f:
        json.dump(all_samples, f, indent=2, ensure_ascii=False)

    # 打印最终结果
    logging.warning(f"\n{'='*50}")
    logging.warning("Final Results:")
    for dataset_name, result in all_results.items():
        logging.warning(f"  {dataset_name}: {result['accuracy']:.4f} ({result['num_samples']} samples, "
                       f"{result['num_correct']} correct)")
    logging.warning(f"\nResults saved to {output_file}")
    logging.warning(f"Samples saved to {output_file_samples}")
