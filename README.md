# CoT Latent Reasoning — Evaluation Suite

This repository contains a collection of evaluation pipelines for CoT /
latent-reasoning models, together with white-box, black-box, and random
perturbation studies for each one.

## Repository layout

```
.
├── codi/      # CODI (Llama-based latent CoT)
├── colar/     # CoLaR (Lightning-based latent CoT)
├── llama/     # Vanilla Llama-3.2-1B-Instruct (vLLM evaluation)
├── PCCoT/     # PCCoT (PECoT + GPT-2)
├── RoT/       # Rest-of-Thought (Qwen2.5-VL based)
├── SIM-CoT/   # SIM-CoT / Coconut (GPT-2 latent CoT)
├── lvr/       # LVR-7B (Qwen2.5-VL based)
├── Monet/     # Monet-7B (Qwen2.5-VL based)
├── SkiLa/     # SkiLa-7B (Qwen2.5-VL based)
├── qwen/      # Vanilla Qwen2.5-VL-7B-Instruct (multimodal baseline)
└── data/      # Default dataset root (override via DATA_DIR)
    ├── gsm8k.json
    ├── MultiArith.json
    ├── SVAMP.json
    ├── select.json
    ├── MMVP/                  # contains MMVP Images/ and Questions.csv
    ├── vstar/                 # contains test_questions.jsonl
    └── MMStar/                # contains metadata.json
```

Each subdirectory is a self-contained project whose `test.py`,
`attack_white.py`, `attack_black.py`, `attack_random.py`, and
`run_attack.py` all resolve paths relative to the script itself unless
overridden by an environment variable.

## Quick start

```bash
# 1. Install Python deps for the subproject you want to run
#    (each subproject usually has its own requirements.txt)
pip install -r requirements.txt

# 2. Point to your model and data locations
export CODI_MODEL_PATH=/path/to/CODI-llama3.2-1b-Instruct
export CODI_CKPT_DIR=/path/to/codi_llama1b
export DATA_DIR=/path/to/datasets

# 3. Run an evaluation
python codi/test.py --dataset gsm8k --question_id 0
```

For multimodal models the pattern is the same — set `LVR_MODEL_PATH`,
`QWEN_MODEL_PATH`, etc., and the dataset is auto-resolved under
`$DATA_DIR`.

## Models

All models are hosted on HuggingFace. The eight custom models that ship
with this repo are:

| Subproject | Model | Download |
| --- | --- | --- |
| PCCoT | whyNLP/pccot-gpt2 | https://huggingface.co/whynlp/pccot-gpt2 |
| CODI | zen-E/CODI-llama3.2-1b-Instruct | https://huggingface.co/zen-E/CODI-llama3.2-1b-Instruct |
| SIM-CoT | internlm/SIM_COT-GPT2-Coconut | https://huggingface.co/internlm/SIM_COT-GPT2-Coconut |
| CoLaR | AlbertTan/CoLaR | https://huggingface.co/AlbertTan/CoLaR |
| RoT | TencentBAC/RoT-Qwen3-VL-2B | https://huggingface.co/TencentBAC/RoT-Qwen3-VL-2B |
| LVR | vincentleebang/LVR-7B | https://huggingface.co/vincentleebang/LVR-7B |
| Monet | NOVAglow646/Monet-7B | https://huggingface.co/NOVAglow646/Monet-7B |
| SkiLa | JosephTong/SkiLa-7B | https://huggingface.co/JosephTong/SkiLa-7B |

The two vanilla baselines use stock HuggingFace checkpoints:

| Subproject | Model |
| --- | --- |
| llama | `meta-llama/Llama-3.2-1B-Instruct` |
| qwen | `Qwen/Qwen2.5-VL-7B-Instruct` |

Download examples:

```bash
# Using huggingface-cli
huggingface-cli download zen-E/CODI-llama3.2-1b-Instruct --local-dir /opt/models/CODI-llama3.2-1b-Instruct
huggingface-cli download vincentleebang/LVR-7B --local-dir /opt/models/LVR-7B

# Or let transformers download on first run
export CODI_MODEL_NAME=zen-E/CODI-llama3.2-1b-Instruct
python codi/test.py     # auto-downloads into the HF cache
```

## Environment variables

Every script reads paths from environment variables. If a variable is not
set, the script falls back to a relative default. If a critical path does
not exist, the script prints a `[警告] ...` warning so you know which
variable to fix.

### Common

| Variable | Default | Description |
| --- | --- | --- |
| `DATA_DIR` | `<repo>/data` | Root directory containing `gsm8k.json`, MMVP, vstar, MMStar, … |

### Per-subproject — model and checkpoint paths

| Subproject | Variable | Default | Description |
| --- | --- | --- | --- |
| codi | `CODI_MODEL_NAME` | `meta-llama/Llama-3.2-1B-Instruct` | Base LLM |
| codi | `CODI_CKPT_DIR` | `<codi>/codi/codi_llama1b` | LoRA weights |
| codi | `CODI_PROJECT_ROOT` | `<codi>/codi` | CODI source package |
| colar | `COLAR_CHECKPOINT` | `<colar>/colar/.../colar_best.ckpt` | Lightning checkpoint |
| colar | `COLAR_WORKSPACE` | `<colar>/colar` | CoLaR source root |
| llama | `LLAMA_MODEL_PATH` | `meta-llama/Llama-3.2-1B-Instruct` | vLLM model |
| llama | `LLAMA_RESULTS_FILE` | `<llama>/results/org/test_org_results.json` | correct_ids file |
| llama | `LLAMA_SELECT_IDS` | `<repo>/data/select.json` | question filter |
| PCCoT | `PCCOT_MODEL_PATH` | `whyNLP/pccot-gpt2` | PCCoT repo name |
| PCCoT | `PCCOT_PROJECT_ROOT` | `<PCCoT>/PCCoT` | PCCoT source package |
| SIM-CoT | `SIMCOT_MODEL_ID` | `gpt2` | Base LM |
| SIM-CoT | `SIMCOT_CKPT_DIR` | `<SIM-CoT>/SIM-CoT/Coconut/ckpts/SIM_COT-GPT2-Coconut/checkpoint_28` | Coconut checkpoint |
| RoT | `ROT_CHECKPOINT` | `<RoT>/RoT/rot_model/converted` | Stage-2 weights |
| RoT | `ROT_STAGE1_CHECKPOINT` | `<RoT>/RoT/rot_model` | Stage-1 weights |
| RoT | `ROT_CONFIG` | `<RoT>/RoT/configs/stage2_config_qwen3vl_2b.yaml` | Training config |
| RoT | `ROT_LLAMA_TOKENIZER` | `meta-llama/Llama-3.2-1B-Instruct` | Tokenizer used during evaluation |
| LVR | `LVR_MODEL_PATH` | `vincentleebang/LVR-7B` | LVR-7B |
| Monet | `MONET_MODEL_PATH` | `NOVAglow646/Monet-7B` | Monet-7B |
| Qwen | `QWEN_MODEL_PATH` | `Qwen/Qwen2.5-VL-7B-Instruct` | Qwen2.5-VL-7B |
| SkiLa | `SKILA_MODEL_PATH` | `./SkiLa-7B` | SkiLa-7B |
| SkiLa | `SKILA_SKETCH_ENCODER` | `./models--google--siglip2-so400m-patch14-384/...` | SigLIP sketch encoder |
| SkiLa | `SKILA_HF_HOME` | `~/.cache/huggingface` | HF cache root |
| SkiLa | `SKILA_HF_HUB_CACHE` | `~/.cache/huggingface/hub` | HF hub cache |