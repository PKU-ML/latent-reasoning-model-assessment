"""
================================================================================
SIM-CoT模型 黑盒对抗攻击 - 基于问题改写
================================================================================

一、算法概述
-----------
本代码实现了一种针对SIM-CoT(Coconut)模型的黑盒对抗攻击，通过向问题添加无关事实
进行改写，使得模型在改写后的问题下输出错误的答案。

二、与白盒攻击的区别
--------------------
- 白盒攻击(attack_white.py): 在问题前添加对抗前缀，通过梯度优化前缀
- 黑盒攻击(attack_black.py): 改写问题本身，通过添加无关事实

三、攻击目标
-----------
- 在答案token位置，让 target_token 的logit - baseline_token 的logit 差值最大化
- 只要攻击成功即可，不需要考虑cosine相似度

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

# ============ 无用事实库 (必须包含数字，英文) ============
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


def main():
    # ============ 命令行参数 ============
    parser = argparse.ArgumentParser(description="SIM-CoT 黑盒对抗攻击 - 问题改写")
    parser.add_argument("--problem-id", type=int, default=None,
                        help="指定攻击的问题ID (默认自动分配)")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子 (默认42)")
    parser.add_argument("--output-dir", type=str,
                        default=os.environ.get(
                            "SIMCOT_OUTPUT_DIR",
                            str(Path(__file__).resolve().parent / "adv_results" / "results_black"),
                        ),
                        help="结果保存目录 (默认: <SIM-CoT>/adv_results/results_black；可通过 SIMCOT_OUTPUT_DIR 覆盖)")
    parser.add_argument("--simcot-gpu", type=int, default=0,
                        help="SIM-CoT模型使用的GPU编号 (默认0)")
    parser.add_argument("--dataset", type=str, choices=["gsm8k", "MultiArith", "SVAMP"], default="gsm8k",
                        help="指定要攻击的数据集 (默认 gsm8k)")
    args = parser.parse_args()

    # ============ 配置 ============
    DEVICE = f"cuda:{args.simcot_gpu}"
    print(f"SIM-CoT模型使用设备: {DEVICE}")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

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
    DATA_NAME = args.dataset
    DATA_DIR = os.environ.get(
        "DATA_DIR",
        str(Path(__file__).resolve().parent.parent / "data"),
    )
    DATA_PATH = f"{DATA_DIR}/{DATA_NAME}.json"

    os.makedirs(args.output_dir, exist_ok=True)
    if args.problem_id is not None:
        PROBLEM_ID = args.problem_id
    else:
        PROBLEM_ID = len(os.listdir(args.output_dir))

    print("=" * 60)
    print("SIM-CoT 黑盒对抗攻击 - 问题改写")
    print(f"使用 {len(IRRELEVANT_FACTS)} 个无用事实进行测试")
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
                                                  device=input_embeds.device, dtype=input_embeds.dtype)
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

    def _find_answer_position(all_logits, tokenizer):
        """找到第一个含数字的token位置"""
        probs = torch.softmax(all_logits[0], dim=-1)
        first_digit_pos = None
        first_digit_token_str = None

        for pos in range(probs.shape[0]):
            top_token_id = probs[pos].argmax().item()
            top_token_str = tokenizer.decode([top_token_id]).strip()
            if re.search(r'\d', top_token_str):
                first_digit_pos = pos
                first_digit_token_str = top_token_str
                break

        if first_digit_pos is None:
            first_digit_pos = probs.shape[0] - 1
            first_digit_token_str = tokenizer.decode([probs[first_digit_pos].argmax().item()]).strip()

        return first_digit_pos, first_digit_token_str


    def extract_answer_number(sentence):
        sentence = sentence.replace(',', '')
        pred = [s for s in re.findall(r'-?\d+\.?\d*', sentence)]
        if not pred:
            return float('inf')
        return float(pred[-1])


    def make_prompt(question_str, prefix_str=""):
        """构建prompt"""
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


    # ============ 4. Baseline评估 ============
    print("\n[3] 测试baseline...")

    baseline_input_ids, baseline_attention_mask = make_prompt(question, "")
    baseline_text, baseline_ids = generate_answer(baseline_input_ids, baseline_attention_mask)

    baseline_answer = extract_answer_number(baseline_text)
    print(f"Baseline生成文本: '{baseline_text}'")
    print(f"Baseline提取答案: {baseline_answer}")
    print(f"正确答案: {answer_str}")

    baseline_correct = (str(baseline_answer) == answer_str or
                       abs(baseline_answer - float(answer_str)) < 0.01 if baseline_answer != float('inf') and answer_str.lstrip('-').replace('.','',1).isdigit() else False)
    print(f"Baseline是否正确: {baseline_correct}")

    # ============ 5. 获取baseline信息 ============
    print("\n获取baseline信息...")

    with torch.no_grad():
        all_logits_output = model.forward_embeds_for_gradient(
            input_ids=baseline_input_ids,
            attention_mask=baseline_attention_mask,
            n_latent=N_LATENT_TOKENS,
            target_position=-1
        )

        first_digit_pos, _ = _find_answer_position(all_logits_output, simcot_tokenizer)
        answer_logits = all_logits_output[:, first_digit_pos:first_digit_pos+1, :].squeeze(1)

        sorted_logits, sorted_indices = torch.sort(answer_logits[0], descending=True)

        BASELINE_TOKEN_ID = sorted_indices[0].item()
        TARGET_TOKEN_ID = sorted_indices[1].item()

        baseline_prob = torch.softmax(answer_logits[0], dim=-1)[BASELINE_TOKEN_ID].item()
        target_prob = torch.softmax(answer_logits[0], dim=-1)[TARGET_TOKEN_ID].item()
        baseline_answer_prob = baseline_prob

        baseline_token_str = simcot_tokenizer.decode([BASELINE_TOKEN_ID])
        target_token_str = simcot_tokenizer.decode([TARGET_TOKEN_ID])

        print(f"Baseline答案token位置:")
        print(f"  最大token: '{baseline_token_str}' (id={BASELINE_TOKEN_ID}), prob={baseline_prob:.4f}")
        print(f"  第二大token: '{target_token_str}' (id={TARGET_TOKEN_ID}), prob={target_prob:.4f}")
        print(f"  正确答案概率: {baseline_answer_prob:.4f}")

    # ============ 6. 评估函数 ============

    def evaluate_question(question_str):
        """评估一个问题，返回正确答案概率"""
        input_ids, attention_mask = make_prompt(question_str, "")
        input_ids = input_ids.to(DEVICE)
        attention_mask = attention_mask.to(DEVICE)

        with torch.no_grad():
            # 获取答案token的logits
            all_logits = model.forward_embeds_for_gradient(
                input_ids=input_ids,
                attention_mask=attention_mask,
                n_latent=N_LATENT_TOKENS,
                target_position=-1
            )
            first_digit_pos, _ = _find_answer_position(all_logits, simcot_tokenizer)
            answer_logits = all_logits[:, first_digit_pos:first_digit_pos+1, :].squeeze(1)
            answer_prob = torch.softmax(answer_logits[0], dim=-1)[BASELINE_TOKEN_ID].item()

            text, _ = generate_answer(input_ids, attention_mask)
            pred_answer = extract_answer_number(text)
            is_correct = (str(pred_answer) == answer_str or
                         abs(pred_answer - float(answer_str)) < 0.01 if pred_answer != float('inf') and answer_str.lstrip('-').replace('.','',1).isdigit() else False)

        return {
            "answer_prob": answer_prob,
            "pred_answer": pred_answer,
            "is_correct": is_correct,
            "generated_text": text
        }

    # ============ 7. 使用无用事实生成改写问题 ============
    print("\n[5] 使用无用事实生成改写问题...")

    print(f"\n评估原始问题...")
    original_eval = evaluate_question(question)
    print(f"  answer_prob: {original_eval['answer_prob']:.4f}, correct: {original_eval['is_correct']}")

    # ============ 8. 测试改写问题 ============
    print(f"\n测试 {len(IRRELEVANT_FACTS)} 个改写问题...")
    all_candidates = []
    for i, fact in enumerate(IRRELEVANT_FACTS):
        rw_q = f"{fact} {question}"
        print(f"  测试 [{i+1}/{len(IRRELEVANT_FACTS)}]: {fact[:40]}...")
        eval_result = evaluate_question(rw_q)
        all_candidates.append({
            "question": rw_q,
            "fact": fact,
            "answer_prob": eval_result["answer_prob"],
            "is_correct": eval_result["is_correct"],
        })
        print(f"    answer_prob={eval_result['answer_prob']:.4f}, correct={eval_result['is_correct']}")

    # ============ 9. 选择最佳结果 ============
    print("\n" + "=" * 60)
    print("选择最佳改写")
    print("=" * 60)

    if all_candidates:
        # 找正确答案概率最低的（最有效的攻击）
        best_candidate = min(all_candidates, key=lambda x: x["answer_prob"])
        best_question = best_candidate["question"]
        best_answer_prob = best_candidate["answer_prob"]
        best_fact = best_candidate["fact"]

        print(f"最佳改写:")
        print(f"  问题: {best_question[:100]}...")
        print(f"  无用信息: {best_fact}")
        print(f"  正确答案概率: {best_answer_prob:.4f} (baseline: {baseline_answer_prob:.4f})")
        print(f"  攻击成功: {not best_candidate['is_correct']}")

        final_eval = evaluate_question(best_question)
        print(f"\n最终验证:")
        print(f"  生成文本: '{final_eval['generated_text'][:100]}...'")
        print(f"  提取答案: {final_eval['pred_answer']}")
        print(f"  是否正确: {final_eval['is_correct']}")
        print(f"  正确答案概率: {final_eval['answer_prob']:.4f}")
    else:
        print("没有找到改写")
        best_candidate = None
        best_question = question
        best_answer_prob = original_eval["answer_prob"]
        best_fact = None
        final_eval = original_eval

    # ============ 10. 保存结果 ============
    print("\n[6] 保存结果...")

    result = {
        "question": question,
        "ground_truth": answer_str,
        "best_rewrite": best_question,
        "best_fact": best_fact,
        "best_answer_prob": best_answer_prob,
        "baseline_answer_prob": baseline_answer_prob,
        "baseline_correct": baseline_correct,
        "final_eval": {
            "is_correct": final_eval["is_correct"],
            "answer_prob": final_eval["answer_prob"],
            "generated_text": final_eval["generated_text"]
        },
        "attack_success": baseline_correct and not final_eval["is_correct"],
        "num_facts": len(IRRELEVANT_FACTS),
        "all_candidates": all_candidates
    }

    os.makedirs(args.output_dir, exist_ok=True)
    save_path = os.path.join(args.output_dir, f"problem_{PROBLEM_ID}.json")
    with open(save_path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"\n结果已保存到 {save_path}")
    print(f"\n攻击效果: {'成功' if result['attack_success'] else '失败'}")


if __name__ == "__main__":
    main()