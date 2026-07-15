#!/usr/bin/env python
# coding=utf-8
"""
Test script for CoLaR model on GSM8K dataset.

功能：
1. 加载本地 JSON 格式的数据集（gsm8k, MultiArith, SVAMP）
2. 在多个数据集上进行评估
3. 支持评估模型之前做对的问题（correct_ids）
4. 支持评估前 N 个问题
5. 支持评估指定的问题ID
6. 支持确定性采样模式
7. 支持使用攻击结果中的改写问题（best_rewrite）替换原问题
8. 支持从 gcg_results 目录加载前缀，添加到问题前
9. 每次运行只保存一个 test_results.json 文件

使用方法：
    python test.py                                    # 评估所有数据集的所有问题
    python test.py --dataset gsm8k                   # 只评估 gsm8k 数据集
    python test.py --question_id 0                   # 评估所有数据集的第0个问题
    python test.py --dataset gsm8k --question_id 0  # 只评估 gsm8k 数据集的第0个问题
    python test.py --eval_correct_ids                # 评估之前做对的问题
    python test.py --eval_first_n 10                # 评估前10个问题
    python test.py --eval_first_n 30 --dataset gsm8k  # 评估 gsm8k 前30个问题
    python test.py --correct_ids_file ./test_results.json  # 从指定文件读取correct_ids
    python test.py --output_dir ./results            # 指定输出目录
    python test.py --non-deterministic              # 使用随机采样（默认确定性）
    python test.py --adv_results_dir ./adv_results/results_black  # 使用改写后的问题进行评估
    python test.py --gcg_results_dir ./gcg_results   # 使用 gcg_results 中的前缀
"""

import json
import logging
import sys
import os
from pathlib import Path
import argparse

import torch
import numpy as np

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Configuration — 所有路径都支持通过环境变量覆盖，详见 README.md
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
WORKSPACE_PATH = os.environ.get(
    "COLAR_WORKSPACE",
    str(Path(__file__).resolve().parent / "colar"),
)

# Data directory（默认指向仓库根目录的 data/）
DATA_DIR = Path(os.environ.get("DATA_DIR", str(Path(__file__).resolve().parent.parent / "data")))

# Output directory
OUTPUT_DIR = Path(os.environ.get("COLAR_OUTPUT_DIR", str(Path(__file__).resolve().parent / "results" / "adv")))


def extract_answer_from_output(output_string: str, answer_template="Answer:"):
    """Extract answer from model output."""
    try:
        return output_string.strip('#').split(answer_template)[-1]
    except (ValueError, IndexError):
        return output_string


def verify_answer(gt_answer: str, pred_answer: str) -> bool:
    """Verify if predicted answer matches ground truth."""
    def get_pure_string(s: str):
        return s.strip("#\n ").rstrip(".").replace(",", "").lower()

    gt_answer = get_pure_string(gt_answer)
    pred_answer = get_pure_string(pred_answer)

    try:
        gt_num = float(gt_answer)
        pred_num = float(pred_answer)
        return abs(gt_num - pred_num) < 1e-6
    except ValueError:
        pass

    return gt_answer == pred_answer


def load_model_and_tokenizer(checkpoint_path: str, workspace_path: str):
    """Load model and tokenizer from Lightning checkpoint."""
    import yaml
    # chdir 到 workspace_path（默认是 colar/colar 子目录），不再硬编码绝对路径
    os.chdir(workspace_path)

    from omegaconf import OmegaConf
    from colar.src.models.colar import LitCoLaR

    # Load hparams.yaml to get config
    hparams_path = os.path.dirname(checkpoint_path).replace('/checkpoints', '') + '/hparams.yaml'
    logger.info(f"Loading hparams from {hparams_path}")

    with open(hparams_path, 'r') as f:
        hparams_data = yaml.safe_load(f)

    all_config = OmegaConf.create(hparams_data['all_config'])

    # Add args to config
    all_config.args = OmegaConf.create({
        "workspace_path": workspace_path,
        "no_log": True,
    })

    # Create model (loads from workspace_path + "models/llms/" + model_id)
    model = LitCoLaR(
        model_kwargs=all_config.model.model_kwargs,
        training_kwargs=all_config.model.training_kwargs,
        all_config=all_config,
    )

    # Load checkpoint weights
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint["state_dict"]
    model.load_state_dict(state_dict=state_dict, strict=False)

    tokenizer = model.tokenizer

    model.eval()
    model.to("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(f"Model loaded successfully from checkpoint")
    return model, tokenizer


def load_dataset(dataset_path: str):
    """Load dataset from JSON file."""
    with open(dataset_path, 'r') as f:
        data = json.load(f)
    return data


def set_deterministic_mode(model, deterministic=True):
    """Set or unset deterministic mode for reproducible results.

    Args:
        model: The CoLaR model
        deterministic: If True, set deterministic mode; if False, restore original
    """
    if deterministic:
        # Save original config
        model._orig_answer_config = model.model_kwargs.answer_generation_config.copy()
        model._orig_latent_temp = model.model_kwargs.latent_generation_config.get("latent_temperature", 1.0)

        # Set deterministic mode
        model.model_kwargs.answer_generation_config.do_sample = False
        # Use 1e-9 instead of 0.0 to avoid Normal distribution scale=0 error
        model.model_kwargs.latent_generation_config.latent_temperature = 1e-9

        logger.debug("Deterministic mode enabled (do_sample=False, temperature=1e-9)")
    else:
        # Restore original config
        if hasattr(model, '_orig_answer_config'):
            model.model_kwargs.answer_generation_config.do_sample = model._orig_answer_config.get("do_sample", True)
        if hasattr(model, '_orig_latent_temp'):
            model.model_kwargs.latent_generation_config.latent_temperature = model._orig_latent_temp

        logger.debug("Deterministic mode disabled, restored original config")


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


def evaluate_model(model, tokenizer, dataset, dataset_name, eval_ids=None, deterministic=True,
                   question_rewrites=None, gcg_prefixes=None,
                   decode_latent=False, compute_latent_length=False, save_latent_vectors=False,
                   replace_latent_vectors=None):
    """Evaluate model on a dataset using CoLaR's latent_generate method.

    Args:
        model: The CoLaR model
        tokenizer: The tokenizer
        dataset: Dataset to evaluate on
        dataset_name: Name of the dataset for logging
        eval_ids: List of indices to evaluate. If None, evaluate all.
        deterministic: If True, use deterministic sampling (no random sampling)
        question_rewrites: Dict of question_id -> rewritten_question. If provided, use rewritten questions.
        gcg_prefixes: Dict of question_id -> prefix string to prepend to question.
        decode_latent: Whether to decode latent vectors
        compute_latent_length: Whether to compute latent vector lengths
        save_latent_vectors: Whether to save complete latent vectors
    """
    logger.info(f"Evaluating on {dataset_name} dataset with {len(dataset)} samples")

    device = next(model.parameters()).device

    # Set deterministic mode if requested
    if deterministic:
        set_deterministic_mode(model, deterministic=True)

    correct_ids = []
    results = []

    # latent decode 配置
    probe_topk = 5

    # 存储latent向量相关数据
    # 注意：eval_ids可能为None，需要先确定
    if eval_ids is None:
        eval_ids = list(range(len(dataset)))
    question_latent_lengths = {idx: [] for idx in eval_ids} if compute_latent_length else None
    question_latent_vectors = {idx: [] for idx in eval_ids} if save_latent_vectors else None

    # 初始化batch级latent解码结果
    sample_latent_decoded = {} if decode_latent else None

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

    try:
        # Determine which questions to evaluate
        if eval_ids is not None:
            eval_questions = [dataset[i] for i in eval_ids]
        else:
            eval_questions = dataset

        logger.info(f"Will evaluate {len(eval_questions)} questions")

        if question_rewrites:
            logger.info(f"Using {len(question_rewrites)} rewritten questions from adv_results")
        if gcg_prefixes:
            logger.info(f"Using {len(gcg_prefixes)} gcg prefixes")

        from tqdm import tqdm
        for idx, (item, orig_idx) in enumerate(zip(eval_questions, eval_ids)):
            question = item['question']
            label = item['answer']

            # Use rewritten question if available
            if question_rewrites and orig_idx in question_rewrites:
                question = question_rewrites[orig_idx]
                if (idx + 1) % 10 == 0:
                    logger.debug(f"  Question {orig_idx} replaced with rewritten version")

            # Add gcg prefix if available
            original_question = question
            if gcg_prefixes and orig_idx in gcg_prefixes:
                question = gcg_prefixes[orig_idx] + question
                if (idx + 1) % 10 == 0:
                    logger.debug(f"  Question {orig_idx} added prefix")

            with torch.no_grad():
                # 尝试获取latent向量
                try:
                    pred_ids, n_latent_forward, latent_vectors = model.latent_generate(
                        questions=[question],
                        return_latent_hidden_states=True,
                        replace_latent_vectors=replacement_vectors,
                        question_idx=orig_idx
                    )
                except (TypeError, ValueError):
                    pred_ids, n_latent_forward = model.latent_generate(questions=[question])
                    latent_vectors = None

            # 处理latent向量
            if latent_vectors is not None and len(latent_vectors) > 0:
                # latent_vectors 是 all_latent_hidden_states，形状为 [batch, num_layers, 1, hidden_dim]
                # 每个元素对应一次latent forward的最后一层hidden state

                # 提取每个时间步的最后一层hidden state (形状: [batch, 1, hidden])
                for step_hidden in latent_vectors:
                    # step_hidden shape: [batch_size, num_layers, 1, hidden]
                    # 取最后一层的hidden state
                    last_layer_hidden = step_hidden[:, -1, 0, :]  # [batch_size, hidden]
                    if last_layer_hidden.shape[0] == 1:
                        vec = last_layer_hidden[0]  # [hidden]
                    else:
                        vec = last_layer_hidden

                    # 计算latent向量长度
                    if compute_latent_length:
                        if isinstance(vec, torch.Tensor):
                            length = torch.sqrt(torch.dot(vec.flatten(), vec.flatten())).item()
                        else:
                            length = np.linalg.norm(vec.flatten())
                        question_latent_lengths[orig_idx].append(length)

                    # 保存完整latent向量
                    if save_latent_vectors:
                        if isinstance(vec, torch.Tensor):
                            question_latent_vectors[orig_idx].append(vec.clone().float().cpu().numpy())
                        else:
                            question_latent_vectors[orig_idx].append(np.array(vec))

                    # 解码latent token
                    if decode_latent and isinstance(vec, torch.Tensor):
                        latent_decoded = []
                        # 使用llm的lm_head将latent向量投影到词汇空间
                        logits = model.llm.lm_head(vec.unsqueeze(0))  # [1, vocab_size]
                        probs = torch.nn.functional.softmax(logits, dim=-1)
                        top5_values, top5_indices = torch.topk(probs, k=probe_topk, dim=-1)  # [1, vocab_size, 5]
                        # 取概率最高的5个token的文本
                        token_ids = top5_indices[0].tolist()  # [5] -> [id1, id2, id3, id4, id5]
                        token_texts = tokenizer.convert_ids_to_tokens(token_ids)
                        latent_decoded.append(token_texts)
                        sample_latent_decoded[idx] = latent_decoded

            output_string = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)[0]
            pred_answer = extract_answer_from_output(output_string)
            is_correct = verify_answer(label, pred_answer)

            if is_correct:
                correct_ids.append(orig_idx)

            sample_result = {
                'id': orig_idx,
                'question': question,
                'original_question': item['question'],
                'ground_truth': label,
                'prediction': pred_answer,
                'model_output': output_string,
                'correct': is_correct,
                'is_rewritten': question_rewrites is not None and orig_idx in question_rewrites,
                'has_prefix': gcg_prefixes is not None and orig_idx in gcg_prefixes
            }

            # 添加latent相关结果
            if compute_latent_length and question_latent_lengths is not None:
                sample_result['latent_vector_lengths'] = question_latent_lengths[orig_idx]

            if decode_latent and sample_latent_decoded is not None:
                sample_result['latent_tokens_decoded'] = sample_latent_decoded.get(idx, [])

            results.append(sample_result)

            # Log each question and model's complete response
            if (idx + 1) % 10 == 0:
                logger.info(f"\n{'='*80}")
                logger.info(f"[{idx + 1}/{len(eval_questions)}] Question ID: {orig_idx}")
                logger.info(f"Question: {question[:200]}{'...' if len(question) > 200 else ''}")
                logger.info(f"Ground Truth: {label}")
                logger.info(f"Prediction: {pred_answer}")
                logger.info(f"Model Output: {output_string[:500]}{'...' if len(output_string) > 500 else ''}")
                logger.info(f"Correct: {is_correct}")
    finally:
        # Restore original config if deterministic mode was enabled
        if deterministic:
            set_deterministic_mode(model, deterministic=False)

    accuracy = len(correct_ids) / len(dataset) if len(dataset) > 0 else 0
    logger.info(f"{dataset_name} Accuracy: {accuracy:.4f} ({len(correct_ids)}/{len(dataset)})")

    # 统计推理向量长度
    avg_latent_length = None
    all_latent_lengths = []
    if compute_latent_length and question_latent_lengths:
        for qid in eval_ids:
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
        "num_samples": len(dataset),
        "num_correct": len(correct_ids),
        "correct_ids": correct_ids,
    }, results, avg_latent_length, question_latent_vectors


def load_correct_ids_from_file(file_path: str):
    """Load correct_ids from a previous test results file.

    Args:
        file_path: Path to the results JSON file

    Returns:
        Dict of dataset_name -> list of correct_ids
    """
    correct_ids_dict = {}
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            data = json.load(f)
        # Handle the new format with summary key
        if 'summary' in data:
            for dataset_name, result in data['summary'].items():
                if isinstance(result, dict) and 'correct_ids' in result:
                    correct_ids_dict[dataset_name] = result['correct_ids']
        else:
            for dataset_name, result in data.items():
                if isinstance(result, dict) and 'correct_ids' in result:
                    correct_ids_dict[dataset_name] = result['correct_ids']
        logger.info(f"Loaded correct_ids from {file_path}: { {k: len(v) for k, v in correct_ids_dict.items()} }")
    else:
        logger.warning(f"Correct_ids file not found: {file_path}")
    return correct_ids_dict


def load_question_rewrites_from_adv_results(adv_results_dir: str):
    """Load rewritten questions from attack results directory.

    Args:
        adv_results_dir: Path to directory containing problem_*.json attack result files

    Returns:
        Dict of question_id -> rewritten_question (best_rewrite)
    """
    rewrites = {}
    adv_dir = Path(adv_results_dir)

    if not adv_dir.exists():
        logger.warning(f"Adv results directory not found: {adv_dir}")
        return rewrites

    # Load all problem_*.json files
    for json_file in adv_dir.glob("problem_*.json"):
        try:
            qid = int(json_file.stem.split("_")[1])
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if "best_rewrite" in data:
                    rewrites[qid] = data["best_rewrite"]
        except (ValueError, KeyError, json.JSONDecodeError) as e:
            logger.warning(f"Failed to load {json_file}: {e}")
            continue

    logger.info(f"Loaded {len(rewrites)} rewritten questions from {adv_dir}")
    return rewrites


def main():
    parser = argparse.ArgumentParser(description="Test CoLaR model on GSM8K dataset")
    parser.add_argument("--dataset", type=str, default=None,
                       help="指定要评估的数据集名称 (gsm8k, MultiArith, SVAMP)。如果为空，则评估所有数据集。")
    parser.add_argument("--question_id", type=int, default=None,
                       help="指定要评估的问题索引（单个）。")
    parser.add_argument("--correct_ids_file", type=str, default=None,
                       help="从指定的结果JSON文件读取correct_ids。")
    parser.add_argument("--eval_correct_ids", action="store_true",
                       help="评估之前做对的问题（使用默认结果文件 results/org/test_org_results.json）。")
    parser.add_argument("--eval_first_n", type=int, default=None,
                       help="评估前N个问题（从数据集中）。")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="输出结果目录 (默认: <colar>/results/adv；可通过 COLAR_OUTPUT_DIR 环境变量或 --output_dir 覆盖)")
    parser.add_argument("--checkpoint_path", type=str, default=None,
                       help="模型checkpoint路径")
    parser.add_argument("--workspace_path", type=str, default=None,
                       help="工作目录路径")
    parser.add_argument("--non-deterministic", action="store_true",
                       help="使用随机采样模式（默认使用确定性模式）")
    parser.add_argument("--adv_results_dir", type=str, default=None,
                       help="攻击结果目录，包含problem_*.json文件，会用best_rewrite替换原问题")
    parser.add_argument("--gcg_results_dir", type=str, default=None,
                       help="gcg_results 目录路径，包含三个子目录 (gsm8k, MultiArith, SVAMP)，分别存放各数据集的 prefix JSON 文件。")
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

    #python test.py --gcg_results_dir ./adv_results/results_white_3 --eval_correct_ids --decode_latent --compute_latent_length --save_latent_vectors
    #python test.py --dataset gsm8k --eval_first_n 82 --adv_results_dir ./adv_results/results_random --output_dir ./results/random --eval_correct_ids
    #python test.py --dataset gsm8k --eval_correct_ids --replace_latent_vectors results/clean/latent_vectors_gsm8k.npz  --adv_results_dir ./adv_results/results_black/gsm8k
    # Update paths if provided
    global CHECKPOINT_PATH, WORKSPACE_PATH, OUTPUT_DIR
    if args.checkpoint_path:
        CHECKPOINT_PATH = args.checkpoint_path
    if args.workspace_path:
        WORKSPACE_PATH = args.workspace_path
    if args.output_dir:
        OUTPUT_DIR = Path(args.output_dir)

    # Load model
    model, tokenizer = load_model_and_tokenizer(CHECKPOINT_PATH, WORKSPACE_PATH)

    # Define datasets
    datasets = {
        "gsm8k": DATA_DIR / "gsm8k.json",
        "MultiArith": DATA_DIR / "MultiArith.json",
        "SVAMP": DATA_DIR / "SVAMP.json",
    }

    # Filter by dataset if specified
    if args.dataset is not None:
        if args.dataset not in datasets:
            logger.warning(f"Dataset '{args.dataset}' not found. Available: {list(datasets.keys())}")
            sys.exit(1)
        datasets = {args.dataset: datasets[args.dataset]}

    # Load correct_ids if requested
    correct_ids_dict = None
    if args.eval_correct_ids or args.correct_ids_file:
        file_path = args.correct_ids_file if args.correct_ids_file else str("results/org/test_org_results.json")
        correct_ids_dict = load_correct_ids_from_file(file_path)

    # Use deterministic mode unless explicitly disabled
    deterministic = not args.non_deterministic
    if deterministic:
        logger.info("Using deterministic mode (do_sample=False, temperature=1e-9)")
    else:
        logger.info("Using non-deterministic/sampling mode")

    # Load question rewrites if adv_results_dir is provided
    question_rewrites = None
    if args.adv_results_dir:
        question_rewrites = load_question_rewrites_from_adv_results(args.adv_results_dir)

    # Load gcg prefixes if gcg_results_dir is provided
    gcg_prefixes_dict = None
    if args.gcg_results_dir:
        gcg_prefixes_dict = {}
        for dataset_name in datasets.keys():
            gcg_prefixes_dict[dataset_name] = load_gcg_prefixes(Path(args.gcg_results_dir), dataset_name)

    results = {}
    all_samples = {}

    for dataset_name, dataset_path in datasets.items():
        if not dataset_path.exists():
            logger.warning(f"Dataset not found: {dataset_path}")
            continue

        logger.info(f"\n{'='*50}")
        logger.info(f"Loading {dataset_name} dataset from {dataset_path}")
        dataset = load_dataset(str(dataset_path))
        logger.info(f"Loaded {len(dataset)} samples")

        # Determine which indices to evaluate
        eval_ids = None

        if args.question_id is not None:
            # Single question ID
            eval_ids = [args.question_id]
            logger.info(f"Evaluating single question ID: {args.question_id}")
        elif args.eval_first_n is not None and correct_ids_dict is not None and dataset_name in correct_ids_dict:
            # From correct_ids, take first N
            all_correct_ids = list(correct_ids_dict[dataset_name])
            eval_ids = all_correct_ids[:args.eval_first_n]
            logger.info(f"Evaluating first {args.eval_first_n} of {len(all_correct_ids)} correct_ids")
        elif args.eval_first_n is not None:
            # First N questions from dataset
            eval_ids = list(range(min(args.eval_first_n, len(dataset))))
            logger.info(f"Evaluating first {len(eval_ids)} questions from dataset")
        elif correct_ids_dict and dataset_name in correct_ids_dict:
            # Questions that were previously correct
            eval_ids = correct_ids_dict[dataset_name]
            logger.info(f"Evaluating {len(eval_ids)} previously correct questions")
        else:
            # All questions
            logger.info(f"Evaluating all {len(dataset)} questions")

        # Get gcg_prefixes for this dataset
        gcg_prefixes = gcg_prefixes_dict.get(dataset_name) if gcg_prefixes_dict else None

        # Run evaluation
        result, samples, avg_latent_len, latent_vectors = evaluate_model(
            model, tokenizer, dataset, dataset_name,
            eval_ids=eval_ids,
            deterministic=deterministic,
            question_rewrites=question_rewrites,
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
                np.savez(OUTPUT_DIR / f"latent_vectors_{dataset_name}.npz", **npz_data)
                logger.info(f"Saved latent vectors to {OUTPUT_DIR / f'latent_vectors_{dataset_name}.npz'}")

        if args.compute_latent_length and avg_latent_len is not None:
            results[dataset_name]["avg_latent_vector_length"] = avg_latent_len

    # Save results in single file format (like codi/test.py)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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

    output_file = OUTPUT_DIR / "test_results.json"
    with open(output_file, 'w') as f:
        json.dump(final_output, f, indent=2, ensure_ascii=False)

    logger.info(f"\n{'='*50}")
    logger.info("Final Results:")
    for dataset_name, result in results.items():
        logger.info(f"  {dataset_name}: {result['accuracy']:.4f} ({result['num_correct']}/{result['num_samples']})")

    logger.info(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()
