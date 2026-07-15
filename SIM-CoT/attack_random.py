"""
================================================================================
SIM-CoT模型 随机前缀攻击
================================================================================

一、算法概述
-----------
本代码实现了一种针对SIM-CoT(Coconut)模型的随机前缀攻击，通过随机选取token作为前缀，
添加到问题前面，测试是否能导致模型输出错误答案。

二、攻击方式
-----------
- 随机选取5个token作为前缀
- 前缀直接添加在问题前面，中间不加空格
- 使用SIM-CoT模型评估，答案错误则攻击成功
- 重复多次，每次独立随机选取前缀

================================================================================
"""

import json
import torch
import numpy as np
import os
import re
import argparse
from dataclasses import dataclass
from pathlib import Path

from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers


def main():
    # ============ 命令行参数 ============
    parser = argparse.ArgumentParser(description="SIM-CoT 随机前缀攻击")
    parser.add_argument("--problem-id", type=int, default=None,
                        help="指定攻击的问题ID (默认自动分配)")
    parser.add_argument("--num-trials", type=int, default=100,
                        help="随机前缀尝试次数 (默认200)")
    parser.add_argument("--prefix-length", type=int, default=5,
                        help="前缀token数量 (默认5)")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子 (默认42)")
    parser.add_argument("--dataset", type=str, choices=["gsm8k", "MultiArith", "SVAMP"], default="gsm8k",
                        help="指定要攻击的数据集 (默认 gsm8k)")
    parser.add_argument("--output-dir", type=str,
                        default=os.environ.get(
                            "SIMCOT_OUTPUT_DIR",
                            str(Path(__file__).resolve().parent / "adv_results" / "results_random"),
                        ),
                        help="结果保存目录 (默认: <SIM-CoT>/adv_results/results_random；可通过 SIMCOT_OUTPUT_DIR 覆盖)")
    parser.add_argument("--simcot-gpu", type=int, default=0,
                        help="SIM-CoT模型使用的GPU编号 (默认0)")
    args = parser.parse_args()

    # ============ 配置 ============
    DEVICE = f"cuda:{args.simcot_gpu}"
    print(f"SIM-CoT模型使用设备: {DEVICE}")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    PREFIX_LENGTH = args.prefix_length
    NUM_TRIALS = args.num_trials

    # SIM-CoT模型配置
    MODEL_ID = "gpt2"
    CKPT_DIR = os.environ.get(
        "SIMCOT_CKPT_DIR",
        str(Path(__file__).resolve().parent / "SIM-CoT" / "Coconut" / "ckpts" / "SIM_COT-GPT2-Coconut" / "checkpoint_28"),
    )
    if not Path(CKPT_DIR).exists():
        print(f"[警告] SIMCOT_CKPT_DIR={CKPT_DIR} 不存在，请通过环境变量指向实际的 checkpoint 目录")
    MAX_NEW_TOKENS = 64
    N_LATENT_TOKENS = 10
    DATA_DIR = os.environ.get(
        "DATA_DIR",
        str(Path(__file__).resolve().parent.parent / "data"),
    )
    DATA_NAME = args.dataset
    DATA_PATH = f"{DATA_DIR}/{DATA_NAME}.json"

    os.makedirs(args.output_dir, exist_ok=True)
    if args.problem_id is not None:
        PROBLEM_ID = args.problem_id
    else:
        PROBLEM_ID = len(os.listdir(args.output_dir))

    print("=" * 60)
    print(f"SIM-CoT 随机前缀攻击 - 数据集: {DATA_NAME}")
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

    # ============ 2. 配置和加载SIM-CoT模型 ============
    print("\n[2] 加载SIM-CoT模型...")

    @dataclass
    class Config:
        model_id: str = "gpt2"
        c_thought: int = 2
        max_latent_stage: int = 5
        training_method: str = "full"
        bf16: bool = False

    class CoconutGPT_Fixed(torch.nn.Module):
        def __init__(self, base_causallm, expainable_llm, tokenizer, latent_token_id,
                     start_latent_id, end_latent_id, eos_token_id, step_start_id,
                     c_thought, configs):
            super().__init__()
            self.gen_forward_cnt = 0
            self.base_causallm = base_causallm
            self.expainable_llm = expainable_llm
            self.tokenizer = tokenizer
            self.latent_token_id = latent_token_id
            self.eos_token_id = eos_token_id
            self.start_latent_id = start_latent_id
            self.end_latent_id = end_latent_id
            self.step_start_id = step_start_id
            self.c_thought = c_thought
            self.config = configs

            if isinstance(self.base_causallm, transformers.models.gpt2.modeling_gpt2.GPT2LMHeadModel):
                self.embedding = self.base_causallm.transformer.get_input_embeddings()
            else:
                self.embedding = self.base_causallm.get_input_embeddings()

        def forward_embeds_for_gradient(self, input_ids, attention_mask, n_latent, target_position=-1,
                                        compute_gradient=False, target_token_id=None, baseline_token_id=None,
                                        pass_gradient_through_latent=True):
            input_embeds = self.embedding(input_ids).clone().requires_grad_(True)

            outputs = self.base_causallm(
                inputs_embeds=input_embeds,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=True)
            kv_cache = outputs.past_key_values

            latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)

            if not pass_gradient_through_latent:
                latent_embd = latent_embd.detach()

            for _ in range(n_latent):
                outputs = self.base_causallm(
                    inputs_embeds=latent_embd,
                    attention_mask=None,
                    past_key_values=kv_cache,
                    output_hidden_states=True,
                    use_cache=True)
                kv_cache = outputs.past_key_values
                latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)

                if not pass_gradient_through_latent:
                    latent_embd = latent_embd.detach()

            end_latent_emb = self.embedding(torch.tensor([self.end_latent_id], device=input_ids.device)).unsqueeze(1)

            outputs = self.base_causallm(
                inputs_embeds=end_latent_emb,
                attention_mask=None,
                past_key_values=kv_cache,
                output_hidden_states=True,
                use_cache=True)

            logits = outputs.logits

            all_logits = [logits]
            current_emb = end_latent_emb

            for _ in range(10):
                outputs = self.base_causallm(
                    inputs_embeds=current_emb,
                    attention_mask=None,
                    past_key_values=kv_cache,
                    output_hidden_states=True,
                    use_cache=True)
                kv_cache = outputs.past_key_values
                logits = outputs.logits
                all_logits.append(logits)

                next_token_id = logits.argmax(dim=-1)
                current_emb = self.embedding(next_token_id.squeeze(1)).unsqueeze(1)

            all_logits = torch.cat(all_logits, dim=1)

            if compute_gradient:
                if target_position == -1:
                    target_position = all_logits.shape[1] - 1
                loss = all_logits[0, target_position, target_token_id] - all_logits[0, target_position, baseline_token_id]

                grads = torch.autograd.grad(loss, input_embeds, create_graph=False, allow_unused=True)
                if grads is None or grads[0] is None:
                    grad_wrt_tokens = torch.zeros(input_embeds.shape[1], self.embedding.weight.shape[0],
                                                  device=input_ids.device, dtype=input_embeds.dtype)
                    return all_logits, grad_wrt_tokens
                grad_wrt_tokens = torch.matmul(grads[0], self.embedding.weight.T)
                return all_logits, grad_wrt_tokens

            return all_logits

        def generate_clean(self, input_ids, attention_mask, max_new_tokens=16):
            self.gen_forward_cnt = 0
            assert input_ids.shape[0] == 1, "only support batch_size == 1"

            tokens = input_ids[0].detach().tolist()
            inputs_embeds = self.embedding(input_ids)

            outputs = self.base_causallm(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                output_hidden_states=True, use_cache=True)
            kv_cache = outputs.past_key_values
            latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)

            for _ in range(N_LATENT_TOKENS):
                outputs = self.base_causallm(
                    inputs_embeds=latent_embd,
                    attention_mask=None,
                    past_key_values=kv_cache,
                    output_hidden_states=True, use_cache=True)
                kv_cache = outputs.past_key_values
                latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)

            end_latent_emb = self.embedding(torch.tensor([self.end_latent_id], device=input_ids.device)).unsqueeze(1)

            outputs = self.base_causallm(
                inputs_embeds=end_latent_emb,
                attention_mask=None,
                past_key_values=kv_cache,
                output_hidden_states=True, use_cache=True)

            self.gen_forward_cnt = N_LATENT_TOKENS + 2
            logits = outputs.logits
            kv_cache = outputs.past_key_values

            next_token = torch.argmax(logits[0, -1]).item()
            tokens.append(next_token)
            new_token_embed = self.embedding(torch.tensor([next_token], device=input_ids.device)).unsqueeze(1)

            for _ in range(max_new_tokens - 1):
                outputs = self.base_causallm(
                    inputs_embeds=new_token_embed,
                    past_key_values=kv_cache,
                    use_cache=True)
                kv_cache = outputs.past_key_values
                self.gen_forward_cnt += 1
                next_token = torch.argmax(outputs.logits[0, -1]).item()
                if next_token == self.eos_token_id:
                    break
                tokens.append(next_token)
                new_token_embed = self.embedding(torch.tensor([next_token], device=input_ids.device)).unsqueeze(1)

            return torch.tensor(tokens).view(1, -1)


    def load_coconut_model(checkpoint_path, model_id="gpt2", device="cuda"):
        print(f"Loading checkpoint from {checkpoint_path}...")
        saved_weights = torch.load(checkpoint_path, map_location="cpu")

        print(f"Loading tokenizer from {model_id}...")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizer.pad_token = tokenizer.eos_token

        tokenizer.add_tokens("<|start-latent|>")
        tokenizer.add_tokens("<|end-latent|>")
        tokenizer.add_tokens("<|latent|>")

        latent_id = tokenizer.convert_tokens_to_ids("<|latent|>")
        start_id = tokenizer.convert_tokens_to_ids("<|start-latent|>")
        end_id = tokenizer.convert_tokens_to_ids("<|end-latent|>")

        print(f"Loading base_causallm from {model_id}...")
        base_causallm = AutoModelForCausalLM.from_pretrained(model_id)
        expainable_llm = AutoModelForCausalLM.from_pretrained(model_id)

        base_causallm.resize_token_embeddings(len(tokenizer))

        configs = Config()
        model = CoconutGPT_Fixed(
            base_causallm, expainable_llm, tokenizer, latent_id, start_id, end_id,
            tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<<"), configs.c_thought, configs)

        print("Loading state dict...")
        model.load_state_dict(saved_weights, strict=False)
        model = model.to(device)
        if configs.bf16:
            model = model.to(torch.bfloat16)
        model.eval()
        return model, tokenizer, latent_id, start_id, end_id

    model, simcot_tokenizer, LATENT_ID, START_ID, END_ID = load_coconut_model(CKPT_DIR, MODEL_ID, DEVICE)
    model = model.to(torch.bfloat16)
    print(f"SIM-CoT模型加载完成")

    # ============ 3. 辅助函数 ============

    def extract_answer_number(sentence):
        sentence = sentence.replace(',', '')
        pred = [s for s in re.findall(r'-?\d+\.?\d*', sentence)]
        if not pred:
            return float('inf')
        return float(pred[-1])

    def make_prompt(question_str, prefix_str=""):
        """构建prompt - 前缀直接加在问题前，中间不加空格"""
        prompt = prefix_str.strip() + question_str.strip() + "\n"
        question_ids = simcot_tokenizer.encode(prompt, add_special_tokens=True)
        input_ids_tensor = torch.tensor([question_ids], dtype=torch.long).to(DEVICE)
        attention_mask = torch.ones_like(input_ids_tensor)
        return input_ids_tensor, attention_mask

    def generate_answer(input_ids, attention_mask, max_tokens=MAX_NEW_TOKENS):
        """生成答案"""
        with torch.no_grad():
            output_ids = model.generate_clean(input_ids, attention_mask, max_tokens)
        return simcot_tokenizer.decode(output_ids[0], skip_special_tokens=True), output_ids[0].tolist()

    def is_correct(pred_answer, ground_truth):
        """判断答案是否正确"""
        if pred_answer == float('inf') or not ground_truth.lstrip('-').replace('.','',1).isdigit():
            return False
        return str(pred_answer) == ground_truth or abs(pred_answer - float(ground_truth)) < 0.01

    # ============ 4. Baseline评估 ============
    print("\n[3] 测试baseline...")

    baseline_input_ids, baseline_attention_mask = make_prompt(question, "")
    baseline_text, baseline_ids = generate_answer(baseline_input_ids, baseline_attention_mask)

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
            decoded = simcot_tokenizer.decode([i])
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
        prefix_str = simcot_tokenizer.decode(prefix_ids)

        # 构建带前缀的问题 - 前缀直接加在问题前，中间不加空格
        prefixed_question = prefix_str.strip() + question.strip()

        # 评估
        input_ids, attention_mask = make_prompt(prefixed_question, "")
        input_ids = input_ids.to(DEVICE)
        attention_mask = attention_mask.to(DEVICE)

        generated_text, _ = generate_answer(input_ids, attention_mask)
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
