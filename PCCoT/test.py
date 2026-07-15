#!/usr/bin/env python
# coding=utf-8
"""
Test script for PCCoT model on various math reasoning datasets.
Tests gsm8k, gsm-hard, MultiArith, and SVAMP datasets.

功能：
1. 加载本地 JSON 格式的数据集（gsm8k, MultiArith, SVAMP）
2. 在多个数据集上进行评估
3. 支持从 gcg_results 目录加载前缀，添加到问题前
4. 支持评估模型之前做对的问题（correct_ids）
5. 支持评估前 N 个问题（可单独使用，也可与 --eval_correct_ids_from 组合）
6. 每次运行只保存一个 test_results.json 文件

使用方法：
    python test.py                                    # 评估所有数据集的所有问题
    python test.py --dataset gsm8k                   # 只评估 gsm8k 数据集
    python test.py --question_id 0                   # 评估所有数据集的第0个问题
    python test.py --dataset gsm8k --question_id 0    # 只评估 gsm8k 数据集的第0个问题
    python test.py --gcg_results_dir ./gcg_results   # 使用 gcg_results 中的前缀
    python test.py --gcg_results_dir ./gcg_results --dataset gsm8k --question_id 0
    python test.py --eval_correct_ids_from ./test_org_results.json  # 评估之前做对的问题
    python test.py --eval_first_n 10                 # 评估前10个问题
    python test.py --eval_correct_ids_from ./test_org_results.json --eval_first_n 30 --gcg_results_dir ./gcg_results --dataset gsm8k
"""

import json
import logging
import os
import sys
import re
from pathlib import Path

import torch
import numpy as np
from transformers import AutoTokenizer, AutoConfig, HfArgumentParser
from transformers.utils.hub import cached_file
from peft import AutoPeftModel
import evaluate
from tqdm import tqdm
sys.path.insert(0, os.environ.get("PCCOT_PROJECT_ROOT", str(Path(__file__).resolve().parent / "PCCoT")))
import models

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Model name — 通过环境变量 PCCOT_MODEL_PATH 注入；默认为 whyNLP/pccot-gpt2 (HuggingFace 仓库名)
MODEL_NAME_OR_PATH = os.environ.get(
    "PCCOT_MODEL_PATH",
    "whyNLP/pccot-gpt2",
)

# Data directory (默认指向仓库根目录的 data/)
DATA_DIR = Path(os.environ.get("DATA_DIR", str(Path(__file__).resolve().parent.parent / "data")))

# Output directory (默认指向 <PCCoT>/results/adv)
OUTPUT_DIR = Path(os.environ.get("PCCOT_OUTPUT_DIR", str(Path(__file__).resolve().parent / "results" / "adv")))


def load_model_and_tokenizer():
    """Load model, tokenizer and PCCoT arguments."""
    logger.info(f"Loading model from {MODEL_NAME_OR_PATH}")

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

    logger.info("Model loaded successfully")
    return model, tokenizer, pccot_args


def load_dataset(dataset_path: str):
    """Load dataset from JSON file."""
    with open(dataset_path, 'r') as f:
        data = json.load(f)
    return data


def normalize_number(s):
    """Normalize number string for comparison. Handles cases like '1' == '1.0'"""
    try:
        return str(float(s.strip()))
    except (ValueError, TypeError):
        return s.strip()


def extract_answer_number(sentence: str) -> float:
    """Extract answer number from generated text."""
    sentence = sentence.replace(',', '')
    pred = re.findall(r'-?\d+\.?\d*', sentence)
    if not pred:
        return float('inf')
    return float(pred[-1])


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
        logger.info(f"Loaded {len(prefixes)} gcg prefixes from {dataset_dir}")
    else:
        logger.warning(f"gcg_results subdir not found: {dataset_dir}")

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
                correct_ids_set[dataset_name] = set(correct_ids_data['summary'][dataset_name].get("correct_ids", []))
        else:
            for dataset_name in correct_ids_data:
                correct_ids_set[dataset_name] = set(correct_ids_data[dataset_name].get("correct_ids", []))
        logger.info(f"Loaded correct_ids from {correct_ids_file}: { {k: len(v) for k, v in correct_ids_set.items()} }")
    else:
        logger.warning(f"correct_ids file not found: {correct_ids_path}")
    return correct_ids_set


def evaluate_model(model, tokenizer, pccot_args, dataset, dataset_name, batch_size=1,
                   question_ids=None, gcg_prefixes=None,
                   decode_latent=False, compute_latent_length=False, save_latent_vectors=False,
                   replace_latent_vectors=None):
    """Evaluate model on a dataset."""
    # Determine which questions to evaluate
    if question_ids is not None:
        eval_indices = list(question_ids)
        questions = [dataset[i]['question'] for i in eval_indices]
        labels = [dataset[i]['answer'] for i in eval_indices]
        eval_dataset = [dataset[i] for i in eval_indices]
    else:
        eval_indices = list(range(len(dataset)))
        questions = [item['question'] for item in dataset]
        labels = [item['answer'] for item in dataset]
        eval_dataset = dataset

    logger.info(f"Evaluating on {dataset_name} dataset with {len(eval_indices)} samples (total: {len(dataset)})")

    # Load replacement latent vectors if provided
    replacement_vectors = None
    if replace_latent_vectors is not None:
        try:
            replacement_data = np.load(replace_latent_vectors)
            replacement_vectors = {int(k.split('_')[-1]): v for k, v in replacement_data.items()}
            logger.info(f"Loaded {len(replacement_vectors)} replacement latent vectors from {replace_latent_vectors}")
        except Exception as e:
            logger.warning(f"Failed to load replacement vectors: {e}")
            replacement_vectors = None

    # Create data processor
    data_processor = models.COTDataProcessor(
        tokenizer=tokenizer,
        pccot_args=pccot_args,
    )

    # Move model to device
    device = model.device
    model.eval()

    # Store all samples with their details
    correct_ids = []
    all_results = []

    # latent decode 配置
    probe_topk = 5

    # 存储latent向量相关数据
    question_latent_lengths = {idx: [] for idx in eval_indices} if compute_latent_length else None
    question_latent_vectors = {idx: [] for idx in eval_indices} if save_latent_vectors else None

    # Process in batches
    num_batches = (len(eval_indices) + batch_size - 1) // batch_size
    for batch_idx in tqdm(range(num_batches), desc=f"Evaluating {dataset_name}"):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(eval_indices))
            batch_indices = eval_indices[start_idx:end_idx]
            batch_questions = [dataset[i]['question'] for i in batch_indices]

            # Add gcg prefixes if provided
            if gcg_prefixes:
                for idx_pos, orig_idx in enumerate(batch_indices):
                    if orig_idx in gcg_prefixes:
                        batch_questions[idx_pos] = gcg_prefixes[orig_idx] + batch_questions[idx_pos]

            batch_labels = [dataset[i]['answer'] for i in batch_indices]

            # Process using data_processor
            collated = data_processor.process(batch_questions, device=device)

            # Generate using the same method as example.py
            from transformers import GenerationConfig

            generation_config = GenerationConfig(
                max_length=collated["input_ids"].shape[1] + 10,
                do_sample=False,
            )

            # 生成时可选返回latent向量
            if decode_latent or compute_latent_length or save_latent_vectors or replacement_vectors is not None:
                try:
                    decoded_tokens, latent_to_save, second_to_last_hidden_state = model.generate(
                        collated=collated,
                        generation_config=generation_config,
                        return_latents=True,
                        replace_latent_vectors=replacement_vectors,
                        question_indices=batch_indices
                    )
                except TypeError as err:
                    print(str(err))
                    decoded_tokens = model.generate(
                        collated=collated,
                        generation_config=generation_config,
                    )
                    latent_to_save = None
                    second_to_last_hidden_state = None
            else:
                decoded_tokens = model.generate(
                    collated=collated,
                    generation_config=generation_config,
                )
                latent_embds = None

            # Remove input_ids part and decode
            decoded_tokens = decoded_tokens[:, collated["input_ids"].shape[1]:]
            answers = tokenizer.batch_decode(decoded_tokens, skip_special_tokens=True)

            # 处理latent向量（保存latent_to_save拼接second_to_last_hidden_state）
            if save_latent_vectors and latent_to_save is not None and second_to_last_hidden_state is not None:
                for b_idx, qid in enumerate(batch_indices):
                    latent_vec = torch.cat([latent_to_save[b_idx], second_to_last_hidden_state[b_idx]], dim=0)
                    question_latent_vectors[qid].append(latent_vec.clone().float().cpu().numpy())

            # Store each sample with its details
            for i, (pred, label, orig_idx) in enumerate(zip(answers, batch_labels, batch_indices)):
                is_correct = pred.strip() == label.strip()
                if not is_correct:
                    pred_normalized = normalize_number(pred)
                    label_normalized = normalize_number(label)
                    is_correct = pred_normalized == label_normalized

                if is_correct:
                    correct_ids.append(orig_idx)

                original_question = dataset[orig_idx]['question']
                full_question = original_question
                if gcg_prefixes and orig_idx in gcg_prefixes:
                    full_question = gcg_prefixes[orig_idx] + original_question

                sample_result = {
                    "id": orig_idx,
                    "question": full_question,
                    "original_question": original_question,
                    "prediction": pred,
                    "label": label,
                    "correct": is_correct,
                    "has_prefix": gcg_prefixes is not None and orig_idx in gcg_prefixes
                }

                if compute_latent_length and question_latent_lengths is not None:
                    sample_result['latent_vector_lengths'] = question_latent_lengths[orig_idx]

                all_results.append(sample_result)

    accuracy = len(correct_ids) / len(eval_indices) if len(eval_indices) > 0 else 0
    logger.info(f"{dataset_name} Accuracy: {accuracy:.4f} ({len(correct_ids)}/{len(eval_indices)})")

    # 统计推理向量长度
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
            logger.info(f"average latent vector length: {avg_latent_length}")

    return {
        "dataset": dataset_name,
        "accuracy": accuracy,
        "num_samples": len(eval_indices),
        "num_correct": len(correct_ids),
        "correct_ids": correct_ids,
    }, all_results, avg_latent_length, question_latent_vectors


def main():
    """Main function to run evaluations on all datasets."""
    import argparse

    parser = argparse.ArgumentParser(description='PCCoT Multi-Dataset Test')
    parser.add_argument('--dataset', type=str, default=None,
                        help='Specify dataset to evaluate (gsm8k, MultiArith, SVAMP). If empty, evaluate all.')
    parser.add_argument('--question_id', type=int, default=None,
                        help='Specify question index to evaluate.')
    parser.add_argument('--output_dir', type=str, default=str(OUTPUT_DIR),
                        help='Output results directory')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size')
    parser.add_argument('--gcg_results_dir', type=str, default=None,
                        help='gcg_results 目录路径，包含三个子目录 (gsm8k, MultiArith, SVAMP)，分别存放各数据集的 prefix JSON 文件。')
    parser.add_argument('--eval_correct_ids_from', type=str, default="results/org/test_org_results.json",
                        help='Load correct_ids from previous results JSON and only evaluate those.')
    parser.add_argument('--eval_first_n', type=int, default=None,
                        help='Evaluate first N questions.')
    parser.add_argument('--rephrase_results', type=str, default=None,
                        help='Result file path containing rephrased questions')
    parser.add_argument('--decode_latent', action='store_true', default=False,
                        help='Decode and save latent vectors to result file')
    parser.add_argument('--compute_latent_length', action='store_true', default=False,
                        help='Compute Euclidean length of latent vectors')
    parser.add_argument('--save_latent_vectors', action='store_true', default=False,
                        help='Save complete latent vectors to .npz file')
    parser.add_argument('--replace_latent_vectors', type=str, default=None,
                        help='Path to .npz file containing replacement latent vectors. '
                             'When enabled, compute_latent_length and save_latent_vectors are automatically disabled.')

    args = parser.parse_args()

    # Auto-disable compute_latent_length and save_latent_vectors when replacement is enabled
    if args.replace_latent_vectors is not None:
        if args.compute_latent_length:
            print("Warning: compute_latent_length is disabled when replace_latent_vectors is set")
            args.compute_latent_length = False
        if args.save_latent_vectors:
            print("Warning: save_latent_vectors is disabled when replace_latent_vectors is set")
            args.save_latent_vectors = False

    #python test.py --eval_correct_ids_from ./results/org/test_org_results.json  --decode_latent --compute_latent_length --save_latent_vectors --gcg_results_dir adv_results/results_white_3
    #python test.py --dataset gsm8k --replace_latent_vectors results/black/latent_vectors_gsm8k.npz  --gcg_results_dir adv_results/results_black

    # Load model and tokenizer
    model, tokenizer, pccot_args = load_model_and_tokenizer()

    # Define datasets to evaluate
    datasets = {
        "gsm8k": DATA_DIR / "gsm8k.json",
        "MultiArith": DATA_DIR / "MultiArith.json",
        "SVAMP": DATA_DIR / "SVAMP.json",
    }

    # Filter by dataset if specified
    if args.dataset is not None:
        if args.dataset not in datasets:
            logger.warning(f"Dataset {args.dataset} not found. Available: {list(datasets.keys())}")
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

    # Load rephrase results if provided
    rephrase_dict = {}
    if args.rephrase_results is not None:
        rephrase_path = Path(args.rephrase_results)
        if rephrase_path.exists():
            with open(rephrase_path, 'r', encoding='utf-8') as f:
                rephrase_dict = json.load(f)
            logger.info(f"Loaded {len(rephrase_dict)} rephrased questions from {rephrase_path}")
        else:
            logger.warning(f"rephrase_results file not found: {rephrase_path}")

    results = {}
    all_samples = {}

    # 输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Evaluate on each dataset
    for dataset_name, dataset_path in datasets.items():
        if not dataset_path.exists():
            logger.warning(f"Dataset not found: {dataset_path}")
            continue

        logger.info(f"\n{'='*50}")
        logger.info(f"Loading {dataset_name} dataset from {dataset_path}")
        dataset = load_dataset(dataset_path)
        logger.info(f"Loaded {len(dataset)} samples")

        # Determine which questions to evaluate
        question_ids = None
        if args.question_id is not None:
            # Evaluate specific question_id
            question_ids = [args.question_id]
            logger.info(f"Evaluating question ID: {args.question_id}")
        elif args.eval_first_n is not None and correct_ids_set is not None and dataset_name in correct_ids_set:
            # From correct_ids, take first N
            all_correct_ids = list(correct_ids_set[dataset_name])
            question_ids = all_correct_ids[:args.eval_first_n]
            logger.info(f"Evaluating first {args.eval_first_n} of {len(all_correct_ids)} correct_ids")
        elif args.eval_first_n is not None:
            # Evaluate first N questions from dataset
            question_ids = list(range(args.eval_first_n))
            logger.info(f"Evaluating first {args.eval_first_n} questions from dataset")
        elif correct_ids_set is not None and dataset_name in correct_ids_set:
            # Evaluate all correct_ids
            question_ids = list(correct_ids_set[dataset_name])
            logger.info(f"Evaluating {len(question_ids)} correct_ids from previous run")

        # Apply rephrase if provided
        dataset_for_eval = dataset
        if rephrase_dict and question_ids is not None:
            # Create modified dataset with rephrased questions
            dataset_for_eval = dataset.copy()
            for qid in question_ids:
                if str(qid) in rephrase_dict:
                    dataset_for_eval[qid] = dataset[qid].copy()
                    dataset_for_eval[qid]['question'] = rephrase_dict[str(qid)]

        # Get gcg_prefixes for this dataset
        gcg_prefixes = gcg_prefixes_dict.get(dataset_name) if gcg_prefixes_dict else None

        result, samples, avg_latent_len, latent_vectors = evaluate_model(
            model, tokenizer, pccot_args,
            dataset_for_eval if rephrase_dict else dataset,
            dataset_name,
            batch_size=args.batch_size,
            question_ids=question_ids,
            gcg_prefixes=gcg_prefixes,
            decode_latent=args.decode_latent,
            compute_latent_length=args.compute_latent_length,
            save_latent_vectors=args.save_latent_vectors,
            replace_latent_vectors=args.replace_latent_vectors
        )
        results[dataset_name] = result
        all_samples[dataset_name] = samples

        # 保存完整推理向量到 .npz 文件
        if args.save_latent_vectors and latent_vectors is not None:
            npz_data = {f"{dataset_name}_{qid}": np.array(vecs) for qid, vecs in latent_vectors.items() if vecs}
            if npz_data:
                np.savez(output_dir / f"latent_vectors_{dataset_name}.npz", **npz_data)
                logger.info(f"Saved latent vectors to {output_dir / f'latent_vectors_{dataset_name}.npz'}")

        if args.compute_latent_length and avg_latent_len is not None:
            results[dataset_name]["avg_latent_vector_length"] = avg_latent_len

    # Save results in single file format (like codi/test.py)
    final_output = {
        "summary": {},
        "samples": {}
    }
    for dataset_name, result in results.items():
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
    with open(output_file, 'w') as f:
        json.dump(final_output, f, indent=2, ensure_ascii=False)

    logger.info(f"\n{'='*50}")
    logger.info("Final Results:")
    for dataset_name, result in results.items():
        logger.info(f"  {dataset_name}: {result['accuracy']:.4f} ({result['num_samples']} samples, {result['num_correct']} correct)")

    logger.info(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()
