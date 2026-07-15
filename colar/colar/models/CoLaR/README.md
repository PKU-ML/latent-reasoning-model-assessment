---
license: llama3.2
license_link: https://github.com/meta-llama/llama-models/blob/main/models/llama3_2/LICENSE
datasets:
- openai/gsm8k
language:
- en
metrics:
- accuracy
base_model:
- meta-llama/Llama-3.2-1B-Instruct
- deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
pipeline_tag: text-generation
library_name: transformers
tags:
- LoRA
---