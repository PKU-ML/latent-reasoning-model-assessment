#!/usr/bin/env python3
"""
RoT 模型在多个数学推理数据集上的测试脚本

功能：
1. 加载本地 JSON 格式的数据集（gsm8k, MultiArith, SVAMP）
2. 在多个数据集上进行评估
3. 支持多种解码策略
4. 支持从 gcg_results 目录加载前缀，添加到问题前
5. 支持评估模型之前做对的问题（correct_ids）
6. 支持评估前 N 个问题
7. 每次运行只保存一个 test_results.json 文件

使用方法：
    python test_org.py
    python test_org.py --dataset gsm8k
    python test_org.py --gcg_results_dir ./gcg_results   # 使用 gcg_results 中的前缀
    python test_org.py --eval_correct_ids_from ./test_results.json  # 评估之前做对的问题
    python test_org.py --eval_first_n 10                # 评估前10个问题
"""

import os
import sys
import json
import yaml
import re
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Any
from tqdm import tqdm
import torch
import numpy as np

# 设置路径
sys.path.insert(0, str(Path(__file__).parent) + "/RoT")

from models.cot_compressor import CoTCompressor
from scripts.evaluate import load_model


def load_local_dataset(data_path: str) -> tuple:
    """
    从本地 JSON 文件加载数据集

    Args:
        data_path: JSON 文件路径

    Returns:
        问题列表, CoT列表, 答案列表
    """
    questions = []
    cots = []
    answers = []

    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for item in data:
        questions.append(item['question'])
        # 处理 CoT: 将 steps 数组连接成字符串
        if 'steps' in item and isinstance(item['steps'], list):
            cot = "\n".join(item['steps'])
        else:
            cot = ""
        cots.append(cot)
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

    return questions, cots, answers


def extract_answer_number(sentence: str) -> float:
    """
    从生成的文本中提取答案数字
    """
    sentence = sentence.replace(',', '')
    pred = re.findall(r'-?\d+\.?\d*', sentence)

    if not pred:
        return float('inf')

    pred_answer = float(pred[-1])
    return pred_answer


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

    return acc / len(gold) if len(gold) > 0 else 0.0


def load_gcg_prefixes(gcg_dir, dataset_name):
    """Load gcg prefixes from directory containing problem_*.json files."""
    prefixes = {}
    dataset_dir = gcg_dir / dataset_name

    if dataset_dir.exists():
        for json_file in dataset_dir.glob("problem_*.json"):
            qid = int(json_file.stem.split("_")[1])
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                try:
                    prefixes[qid] = data["best_fact"] + " "
                except:
                    try:
                        prefixes[qid] = data['prefix']
                    except:
                        try:
                            prefixes[qid] = (data['all_results'][:1] + [i for i in data['all_results'] if i['attack_success'] == True])[-1]['prefix']
                        except:
                            prefixes[qid] = ""
        print(f"Loaded {len(prefixes)} gcg prefixes from {dataset_dir}")
    else:
        print(f"gcg_results subdir not found: {dataset_dir}")

    return prefixes


def load_correct_ids_from_results(correct_ids_file):
    """Load correct_ids from a previous test results JSON file."""
    correct_ids_set = {}
    correct_ids_path = Path(correct_ids_file)
    if correct_ids_path.exists():
        with open(correct_ids_path, 'r', encoding='utf-8') as f:
            correct_ids_data = json.load(f)
        # Handle the new format with summary key
        if 'summary' in correct_ids_data:
            for dataset_name in correct_ids_data['summary']:
                correct_ids_set[dataset_name] = correct_ids_data['summary'][dataset_name].get("correct_ids", [])
        else:
            for dataset_name in correct_ids_data:
                correct_ids_set[dataset_name] = correct_ids_data[dataset_name].get("correct_ids", [])
        print(f"Loaded correct_ids from {correct_ids_file}: { {k: len(v) for k, v in correct_ids_set.items()} }")
    else:
        print(f"correct_ids file not found: {correct_ids_path}")
    return correct_ids_set


def evaluate_model(
    model,
    questions: List[str],
    cots: List[str],
    answers: List[float],
    max_new_tokens: int = 32,
    temperature: float = 0.0,
    max_vision_tokens: int = 32,
    stop_threshold: float = 0.02,
    verbose: bool = False,
    gcg_prefixes: dict = None,
    eval_indices: list = None,
    decode_latent: bool = False,
    compute_latent_length: bool = False,
    save_latent_vectors: bool = False,
    replace_latent_vectors: str = None
) -> tuple:
    """
    评估模型

    参数：
    - decode_latent: 是否解码推理向量
    - compute_latent_length: 是否统计推理向量的长度
    - save_latent_vectors: 是否保存完整推理向量

    Returns:
        (accuracy, results, avg_latent_length, question_latent_vectors)
    """
    model.eval()
    pred_answers = []
    results = []

    # latent decode 配置
    probe_topk = 5

    # 存储latent向量相关数据
    question_latent_lengths = {idx: [] for idx in eval_indices} if compute_latent_length else None
    question_latent_vectors = {idx: [] for idx in eval_indices} if save_latent_vectors else None

    # 初始化batch级latent解码结果
    batch_latent_decoded = [] if decode_latent else None

    # If eval_indices is provided, filter questions/cots/answers accordingly
    if eval_indices is not None:
        questions = [questions[i] for i in eval_indices]
        cots = [cots[i] for i in eval_indices]
        answers = [answers[i] for i in eval_indices]

    # Load replacement latent vectors if provided
    replacement_vectors = None
    if replace_latent_vectors is not None:
        try:
            replacement_data = np.load(replace_latent_vectors)
            replacement_vectors = {int(k.split('_')[-1]): v for k, v in replacement_data.items()}
            print(f"Loaded {len(replacement_vectors)} replacement latent vectors from {replace_latent_vectors}")
        except Exception as e:
            print(f"Failed to load replacement vectors: {e}")
            replacement_vectors = None

    with torch.no_grad():
        for idx, (question, cot, answer) in tqdm(enumerate(zip(questions, cots, answers))):
            orig_idx = eval_indices[idx] if eval_indices is not None else idx
            original_question = question

            # Add gcg prefix if available
            if gcg_prefixes and orig_idx in gcg_prefixes:
                question = gcg_prefixes[orig_idx] + question

            try:
                # 生成答案
                result = model.generate(
                    question_text=question,
                    cot_text=cot if verbose else None,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    max_vision_tokens=max_vision_tokens,
                    stop_threshold=stop_threshold,
                    verbose=False,
                    return_latents=True,
                    replace_latent_vectors=replacement_vectors,
                    question_idx=orig_idx
                )

                # 处理model.generate返回值（可能是 str 或 tuple）
                if isinstance(result, tuple):
                    generated, latent_vectors = result
                else:
                    generated = result
                    latent_vectors = None

                # 处理latent向量
                if latent_vectors is not None and len(latent_vectors) > 0:
                    # 计算latent向量长度
                    if compute_latent_length:
                        for vec in latent_vectors:
                            if isinstance(vec, torch.Tensor):
                                length = torch.sqrt(torch.dot(vec.flatten(), vec.flatten())).item()
                            else:
                                length = np.linalg.norm(vec.flatten())
                            question_latent_lengths[orig_idx].append(length)

                    # 保存完整latent向量
                    if save_latent_vectors:
                        for vec in latent_vectors:
                            if isinstance(vec, torch.Tensor):
                                question_latent_vectors[orig_idx].append(vec.clone().float().cpu().numpy())
                            else:
                                question_latent_vectors[orig_idx].append(np.array(vec))

                    # 解码latent token
                    if decode_latent and latent_vectors:
                        latent_decoded = []
                        for vec in latent_vectors:
                            if isinstance(vec, torch.Tensor):
                                # 使用llm_lm_head将latent向量投影到词汇空间
                                logits = model.llm_lm_head(vec.unsqueeze(0).unsqueeze(0))  # [1, 1, vocab_size]
                                probs = torch.nn.functional.softmax(logits, dim=-1)
                                top5_values, top5_indices = torch.topk(probs, k=probe_topk, dim=-1)  # [1, 1, vocab_size, 5]
                                # 取概率最高的5个token的文本
                                token_ids = top5_indices[0, 0, 0].tolist()  # 单个向量
                                token_texts = tokenizer.convert_ids_to_tokens(token_ids)
                                latent_decoded.append(token_texts)
                        batch_latent_decoded.append(latent_decoded)

                # 提取答案
                predicted_answer = extract_answer_number(generated)
                pred_answers.append(predicted_answer)

                # 判断正确性
                is_correct = abs(predicted_answer - answer) < 1e-9 if answer != float('inf') else predicted_answer == answer

                if verbose:
                    print(f"[{idx}] Q: {question[:80]}...")
                    print(f"    Generated: {generated[:200]}...")
                    print(f"    Pred: {predicted_answer}, GT: {answer}, Correct: {is_correct}")

            except Exception as e:
                print(f"[{idx}] Error: {e}")
                pred_answers.append(float('inf'))
                is_correct = False

            sample_result = {
                'id': orig_idx,
                'question': question,
                'original_question': original_question,
                'cot': cot,
                'ground_truth': answer,
                'prediction': pred_answers[-1],
                'correct': is_correct,
                'has_prefix': gcg_prefixes is not None and orig_idx in gcg_prefixes
            }

            # 添加latent相关结果
            if compute_latent_length and question_latent_lengths is not None:
                sample_result['latent_vector_lengths'] = question_latent_lengths[orig_idx]

            if decode_latent and batch_latent_decoded is not None and len(batch_latent_decoded) > 0:
                sample_result['latent_tokens_decoded'] = batch_latent_decoded[-1]

            results.append(sample_result)

    accuracy = compute_accuracy(answers, pred_answers)

    # 统计推理向量长度 (如果启用)
    avg_latent_length = None
    all_latent_lengths = []
    if compute_latent_length and question_latent_lengths:
        for qid in eval_indices:
            lengths = question_latent_lengths[qid]
            if lengths:
                avg_length = sum(lengths) / len(lengths)
                all_latent_lengths.append(avg_length)
        if all_latent_lengths:
            avg_latent_length = sum(all_latent_lengths) / len(all_latent_lengths)
            print(f"average latent vector length: {avg_latent_length}")

    return accuracy, results, avg_latent_length, question_latent_vectors


# =========================================================================
# 默认配置参数 — 所有路径可通过环境变量覆盖，详见仓库根 README.md
# =========================================================================
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
DATA_DIR = os.environ.get(
    "DATA_DIR",
    str(Path(__file__).resolve().parent.parent / "data"),
)

MAX_SAMPLES = None #表示全部样本, 设为数字则限制样本数
MAX_NEW_TOKENS = 256
TEMPERATURE = 0.0
NUM_VISION_TOKENS = 32
STOP_THRESHOLD = 0.02

OUTPUT_DIR = os.environ.get(
    "ROT_OUTPUT_DIR",
    str(Path(__file__).resolve().parent / "results" / "adv"),
)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="RoT Model Evaluation")
    parser.add_argument("--dataset", type=str, default=None,
                        help="指定要评估的数据集名称 (gsm8k, MultiArith, SVAMP)。如果为空，则评估所有数据集。")
    parser.add_argument("--question_id", type=int, default=None,
                        help="指定要评估的问题索引。")
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR,
                        help="输出结果目录")
    parser.add_argument("--checkpoint", type=str, default=CHECKPOINT,
                        help="模型checkpoint路径")
    parser.add_argument("--stage1_checkpoint", type=str, default=STAGE1_CHECKPOINT,
                        help="Stage1 checkpoint路径")
    parser.add_argument("--config", type=str, default=CONFIG,
                        help="配置文件路径")
    parser.add_argument("--gcg_results_dir", type=str, default=None,
                        help="gcg_results 目录路径，包含三个子目录 (gsm8k, MultiArith, SVAMP)，分别存放各数据集的 prefix JSON 文件。")
    parser.add_argument("--eval_correct_ids_from", type=str, default=None,
                        help="从指定的结果JSON文件读取correct_ids，只评估这些id的问题。")
    parser.add_argument("--eval_first_n", type=int, default=None,
                        help="评估前N个问题。")
    parser.add_argument("--decode_latent", action="store_true", default=False,
                        help="解码并保存推理向量 (latent tokens) 到结果文件中")
    parser.add_argument("--compute_latent_length", action="store_true", default=False,
                        help="统计推理向量的长度（欧几里得长度）的平均值")
    parser.add_argument("--save_latent_vectors", action="store_true", default=False,
                        help="保存完整推理向量到 .npz 文件")
    parser.add_argument("--replace_latent_vectors", type=str, default=None,
                        help="Path to .npz file containing replacement latent vectors. "
                             "When enabled, compute_latent_length and save_latent_vectors are automatically disabled.")

    args = parser.parse_args()

    # Auto-disable compute_latent_length and save_latent_vectors when replacement is enabled
    if args.replace_latent_vectors is not None:
        if args.compute_latent_length:
            print("Warning: compute_latent_length is disabled when replace_latent_vectors is set")
            args.compute_latent_length = False
        if args.save_latent_vectors:
            print("Warning: save_latent_vectors is disabled when replace_latent_vectors is set")
            args.save_latent_vectors = False

    #python test.py --eval_correct_ids_from results/org/test_org_results.json --compute_latent_length --save_latent_vectors  --gcg_results_dir adv_results/results_white_3
    #python test.py --dataset gsm8k  --eval_correct_ids_from results/org/test_org_results.json --replace_latent_vectors results/clean/latent_vectors_gsm8k.npz --gcg_results_dir adv_results/results_black

    print("=" * 80)
    print("RoT Model Evaluation")
    print("=" * 80)

    # 加载配置
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # 加载模型
    print("\nLoading model...")
    model = load_model(
        checkpoint_path=args.checkpoint,
        config=config,
        model_type="v2",
        verbose=True,
        stage1_checkpoint=args.stage1_checkpoint
    )

    # 定义要评估的数据集
    datasets = {
        "gsm8k": os.path.join(DATA_DIR, "gsm8k.json"),
        "MultiArith": os.path.join(DATA_DIR, "MultiArith.json"),
        "SVAMP": os.path.join(DATA_DIR, "SVAMP.json"),
    }

    # Filter by dataset if specified
    if args.dataset is not None:
        if args.dataset not in datasets:
            print(f"Dataset '{args.dataset}' not found. Available: {list(datasets.keys())}")
            sys.exit(1)
        datasets = {args.dataset: datasets[args.dataset]}

    # Load gcg prefixes
    gcg_prefixes_dict = {}
    if args.gcg_results_dir is not None:
        for dataset_name in datasets.keys():
            gcg_prefixes_dict[dataset_name] = load_gcg_prefixes(Path(args.gcg_results_dir), dataset_name)

    # Load correct_ids
    correct_ids_set = None
    if args.eval_correct_ids_from is not None:
        correct_ids_set = load_correct_ids_from_results(args.eval_correct_ids_from)

    # 存储所有结果
    all_results = {}
    all_samples = {}

    # 输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 对每个数据集进行评估
    for dataset_name, dataset_path in datasets.items():
        if not os.path.exists(dataset_path):
            print(f"\nDataset not found: {dataset_path}")
            continue

        print(f"\n{'='*60}")
        print(f"Evaluating on {dataset_name} dataset")
        print(f"{'='*60}")

        # 加载数据
        questions, cots, answers = load_local_dataset(dataset_path)
        print(f"Loaded {len(questions)} questions from {dataset_name}")

        if MAX_SAMPLES:
            questions = questions[:MAX_SAMPLES]
            cots = cots[:MAX_SAMPLES]
            answers = answers[:MAX_SAMPLES]
            print(f"Limited to {MAX_SAMPLES} samples")

        # 确定要评估的问题ID
        eval_indices = None
        if args.question_id is not None:
            eval_indices = [args.question_id]
            print(f"Evaluating question ID: {args.question_id}")
        elif args.eval_first_n is not None and correct_ids_set is not None and dataset_name in correct_ids_set:
            all_correct_ids = list(correct_ids_set[dataset_name])
            eval_indices = all_correct_ids[:args.eval_first_n]
            print(f"Evaluating first {args.eval_first_n} of {len(all_correct_ids)} correct_ids")
        elif args.eval_first_n is not None:
            eval_indices = list(range(args.eval_first_n))
            print(f"Evaluating first {args.eval_first_n} questions from dataset")
        elif correct_ids_set is not None and dataset_name in correct_ids_set:
            eval_indices = list(correct_ids_set[dataset_name])
            print(f"Evaluating {len(eval_indices)} correct_ids from previous run")

        # Get gcg_prefixes for this dataset
        gcg_prefixes = gcg_prefixes_dict.get(dataset_name) if gcg_prefixes_dict else None

        # 评估
        accuracy, samples, avg_latent_len, latent_vectors = evaluate_model(
            model=model,
            questions=questions,
            cots=cots,
            answers=answers,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            max_vision_tokens=NUM_VISION_TOKENS,
            stop_threshold=STOP_THRESHOLD,
            verbose=False,
            gcg_prefixes=gcg_prefixes,
            eval_indices=eval_indices,
            decode_latent=args.decode_latent,
            compute_latent_length=args.compute_latent_length,
            save_latent_vectors=args.save_latent_vectors,
            replace_latent_vectors=args.replace_latent_vectors
        )

        # 统计正确样本的ID
        correct_ids = [i for i, s in enumerate(samples) if s['correct']]

        # 保存完整推理向量到 .npz 文件
        if args.save_latent_vectors and latent_vectors is not None:
            npz_data = {f"{dataset_name}_{qid}": np.array(vecs) for qid, vecs in latent_vectors.items() if vecs}
            if npz_data:
                np.savez(output_dir / f"latent_vectors_{dataset_name}.npz", **npz_data)
                print(f"Saved latent vectors to {output_dir / f'latent_vectors_{dataset_name}.npz'}")

        print(f"\n{dataset_name} Results:")
        print(f"  Accuracy: {100*accuracy:.2f}% ({len(correct_ids)}/{len(samples)})")

        # 保存结果
        all_results[dataset_name] = {
            "accuracy": accuracy,
            "num_samples": len(samples),
            "num_correct": len(correct_ids),
            "correct_ids": correct_ids,
        }
        if args.compute_latent_length and avg_latent_len is not None:
            all_results[dataset_name]["avg_latent_vector_length"] = avg_latent_len
        all_samples[dataset_name] = samples

    # 保存结果到文件 (single file format like codi/test.py)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    final_output = {
        "summary": {},
        "samples": {}
    }
    for dataset_name, result in all_results.items():
        summary_entry = {
            "accuracy": result["accuracy"],
            "num_samples": result["num_samples"],
            "num_correct": result["num_correct"],
            "correct_ids": result["correct_ids"]
        }
        if "avg_latent_vector_length" in result:
            summary_entry["avg_latent_vector_length"] = result["avg_latent_vector_length"]
        final_output["summary"][dataset_name] = summary_entry
        final_output["samples"][dataset_name] = all_samples[dataset_name]

    output_file = output_dir / "test_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(final_output, f, indent=2, ensure_ascii=False)

    # 打印最终结果
    print(f"\n{'='*60}")
    print("Final Results:")
    for dataset_name, result in all_results.items():
        print(f"  {dataset_name}: {result['accuracy']:.4f} ({result['num_samples']} samples, "
               f"{result['num_correct']} correct)")
    print(f"{'='*60}")
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()
