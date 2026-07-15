# CODI

This is the official implementation of the paper: [CODI: Compressing Chain-of-Thought into Continuous Space via Self-Distillation](https://arxiv.org/abs/2502.21074).
Accepted by EMNLP 2025 🎉


![codi](imgs/codi_method_v4.png)

## Setup

**Clone the Repository**:
```
git clone git@github.com:zhenyi4/CODI.git
cd CODI
```

**Environment Setup**:
```
conda create --name codi python=3.12
conda activate codi
pip install -r requirements.txt
```

## Running the Results

Pretrained model weights are available at https://huggingface.co/zen-E, including:
* zen-E/CODI-gpt2
* zen-E/CODI-llama3.2-1b-Instruct
 
To evaluate accuracy on GSM8K, run:
```
bash script/test_gpt2.sh # or script/test_llama1b.sh
```
You can change the data_name argument to "svamp", "gsm-hard", or "multi-arith" to evaluate on out-of-distribution (OOD) mathematical benchmarks. 

## Interpret Latent Thoughts (for Section 5 in the Paper)

To probe and visualize latent thoughts on GSM8K, run:
```
bash scripts/probe_latent_token.sh
```
The output file will be saved in outputs/.

## Training
**GSM8k-Aug**
```
bash scripts/train_gpt2_gsm8k-aug.sh # or train_llama1b_gsm8k-aug.sh
```

**GSM8k-Aug-NL**
```
bash scripts/train_gpt2_gsm8k-aug-nl.sh
```

**Commonsense**
```
bash train_gpt2_commonsense.sh # or train_llama_commonsense.sh
```
You can change the testing script's data_name to "commonsense" for evaluation.

## Key Arguments
* `use_prj`: Whether use a projection layer for the last layer hidden state.

* `prj_dim`: The dimension of the hidden state of the projection layer.

* `prj_no_ln`: Whether the projection layer is not followed by a LayerNorm layer.

* `distill_loss_div_std`: Whether divide the distillation loss via the standard deviation of the teacher's hidden states.

* `distill_loss_type`: The type of loss used for distillation (e.g. l1, l2, smoothl1).

* `distill_loss_factor`: The multiplier that scales the distillation loss in the total loss calculation.

* `ref_loss_factor`: The multiplier that scales the teacher's cross entropy loss in the total loss calculation.

* `num_latent`: The number of latent thoughts used for training.

* `inf_latent_iterations`: The number of latent thoughts used for inference.

* `include_last_cot`: Include the last CoT step in the training data.

* `fix_attn_mask`: Flag to fix a known attention mask bug (leave as False by default).

* `max_token_num`: Discards training samples that exceed this token length threshold.

## Local Dataset Testing

This section allows you to test CODI on a local dataset (e.g., a modified version of GSM8K).

### Files

| File | Description |
|------|-------------|
| `gsm8k_test.jsonl` | Original GSM8K test set (1319 samples) |
| `gsm8k_test_modified.jsonl` | Copy for modification |
| `test_local_gsm8k.py` | Test script for implicit CoT (local data) |
| `test_explicit_cot.py` | Test script for explicit CoT (for comparison) |

### Data Format

Each line in the JSONL file is a JSON object:
```json
{"id": 0, "question": "Question content", "answer": "Reasoning process\n#### Final answer"}
```

### Usage

#### 1. Test on original GSM8K

```bash
python test_local_gsm8k.py \
    --data_path ./gsm8k_test.jsonl \
    --model_name_or_path meta-llama/Llama-3.2-1B-Instruct \
    --ckpt_dir codi/codi_llama1b
```

#### 2. Test on modified dataset

First, edit the `question` field in `gsm8k_test_modified.jsonl`, then run:

```bash
python test_local_gsm8k.py \
    --data_path ./gsm8k_test_modified.jsonl \
    --model_name_or_path meta-llama/Llama-3.2-1B-Instruct \
    --ckpt_dir codi/codi_llama1b
```

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--data_path` | `./gsm8k_test.jsonl` | Path to local dataset |
| `--model_name_or_path` | **Required** | Pretrained model path |
| `--ckpt_dir` | **Required** | Checkpoint directory |
| `--batch_size` | 1 | Batch size |
| `--lora_r` | 128 | LoRA rank |
| `--lora_alpha` | 32 | LoRA alpha |
| `--inf_latent_iterations` | 6 | Latent reasoning iterations |
| `--model_max_length` | 512 | Model max length |
| `--greedy` | False | Use greedy decoding |
| `--remove_eos` | False | Remove EOS token |
| `--use_prj` | False | Use projection layer |
| `--num_latent` | 6 | Number of latent thoughts |
| `--prj_dim` | 2048 | Projection layer dimension |
| `--inf_num_iterations` | 1 | Evaluation iterations (for averaging) |
| `--use_chat_template` | False | Use model's chat template for input |
| `--cot_prompt` | None | CoT prompt to prepend (e.g., "Let's think step by step.") |

### Examples

```bash
# Basic test
python test_local_gsm8k.py \
    --data_path ./gsm8k_test.jsonl \
    --model_name_or_path meta-llama/Llama-3.2-1B-Instruct \
    --ckpt_dir codi_llama1b \
    --lora_r 128 \
    --lora_alpha 32 \
    --model_max_length 512 \
    --batch_size 1 \
    --greedy True \
    --num_latent 6 \
    --use_prj True \
    --prj_dim 2048 \
    --inf_latent_iterations 6 \
    --remove_eos True \
    --use_lora True \
    --use_chat_template True

# Test on modified dataset
python test_local_gsm8k.py \
    --data_path ./gsm8k_test_modified.jsonl \
    --model_name_or_path meta-llama/Llama-3.2-1B-Instruct \
    --ckpt_dir codi_llama1b \
    --lora_r 128 \
    --lora_alpha 32 \
    --model_max_length 512 \
    --batch_size 128 \
    --greedy True \
    --num_latent 6 \
    --use_prj True \
    --prj_dim 2048 \
    --inf_latent_iterations 6 \
    --remove_eos True \
    --use_lora True
```

### Explicit CoT Testing (for Comparison)

To compare with the implicit CoT reasoning, you can also test the model's explicit CoT reasoning capability using `test_explicit_cot.py`. This script performs standard autoregressive generation to produce the full reasoning process.

#### Usage

```bash
python test_explicit_cot.py \
    --data_path ./gsm8k_test.jsonl \
    --model_name_or_path meta-llama/Llama-3.2-1B-Instruct \
    --ckpt_dir codi_llama1b \
    --lora_r 128 \
    --lora_alpha 32 \
    --batch_size 1 \
    --greedy True \
    --use_chat_template True
```

#### Example

```bash
# Basic explicit CoT test with greedy decoding
python test_explicit_cot.py \
    --data_path ./gsm8k_test.jsonl \
    --model_name_or_path meta-llama/Llama-3.2-1B-Instruct \
    --ckpt_dir codi_llama1b \
    --lora_r 128 \
    --lora_alpha 32 \
    --model_max_length 512 \
    --batch_size 1 \
    --greedy True

# Explicit CoT test with sampling
python test_explicit_cot.py \
    --data_path ./gsm8k_test.jsonl \
    --model_name_or_path meta-llama/Llama-3.2-1B-Instruct \
    --ckpt_dir codi_llama1b \
    --lora_r 128 \
    --lora_alpha 32 \
    --model_max_length 512 \
    --batch_size 1 \
    --greedy False
```

#### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--data_path` | `./gsm8k_test.jsonl` | Path to local dataset |
| `--model_name_or_path` | **Required** | Pretrained model path |
| `--ckpt_dir` | **Required** | Checkpoint directory |
| `--batch_size` | 1 | Batch size |
| `--lora_r` | 128 | LoRA rank |
| `--lora_alpha` | 32 | LoRA alpha |
| `--model_max_length` | 2048 | Model max length |
| `--greedy` | False | Use greedy decoding |
| `--remove_eos` | False | Remove EOS token |
| `--use_prj` | False | Use projection layer |
| `--use_lora` | True | Use LoRA |
| `--num_latent` | 6 | Number of latent thoughts (for model loading) |
| `--prj_dim` | 2048 | Projection layer dimension |
| `--use_chat_template` | False | Use model's chat template for input |
| `--cot_prompt` | None | CoT prompt to prepend (e.g., "Let's think step by step.") |

#### Output

The results will be saved to `./test_explicit_cot_results.jsonl` by default, containing:
- `cot_length`: Length of generated CoT tokens
- `model_output`: Full generated reasoning and answer
- `prediction` / `ground_truth`: Predicted and ground truth answers

## Citation
If you use this code base in your research, please cite our paper with the following BibTex entry:
```bibtex
@article{shen2025codi,
      title={CODI: Compressing Chain-of-Thought into Continuous Space via Self-Distillation},
      author={Zhenyi Shen and Hanqi Yan and Linhai Zhang and Zhanghao Hu and Yali Du and Yulan He},
      year={2025},
      journal={arXiv preprint arxiv:2502.21074},
}
```
