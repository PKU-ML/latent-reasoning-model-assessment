import tqdm
import random
from typing import List, Tuple
import torch.nn.functional as F
import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from .model_base import LitCoTModelBase
from ..modules.projector import LatentPolicy
from ..modules import grpo
from ..utils.utils import get_position_ids_from_attention_mask, sample_indices_from_attention_mask_3d
from ..utils.constants import MODEL_EMB_STD


class LitCoLaR(LitCoTModelBase):
    def __init__(
        self,
        model_kwargs,
        training_kwargs,
        all_config=None,
    ):
        super().__init__(model_kwargs=model_kwargs, training_kwargs=training_kwargs, all_config=all_config)

        latent_policy_config = model_kwargs.latent_policy_config
        self.latent_policy = LatentPolicy(
            feature_size=self.llm.config.hidden_size,
            intermediate_size=latent_policy_config.get("lp_intermediate_size", self.llm.config.hidden_size),
            deterministic=latent_policy_config.get("lp_determinisitc", False),
        )
        self.embeds_std = MODEL_EMB_STD[model_kwargs.model_id]

        if model_kwargs.do_rl:
            self.init_rl()

    # ++ basic methods implemenration begins ++#
    def limit_rl_train_epoch_length(self):
        n_indices = self.model_kwargs.rl_config.n_train_samples_per_epoch
        all_indices = self.trainer.datamodule.get_all_train_indices()
        indices = random.choices(all_indices, k=n_indices)
        self.trainer.datamodule.set_train_indices(indices)

    def on_fit_start(self):
        if self.model_kwargs.do_rl:
            self.limit_rl_train_epoch_length()
        return super().on_fit_start()

    def on_train_epoch_start(self):
        if self.model_kwargs.do_rl:
            self.limit_rl_train_epoch_length()
        return super().on_train_epoch_start()

    def training_step(self, batch, batch_idx=None, dataloader_idx=0):
        if self.model_kwargs.do_rl:
            return self.rl_training_step(batch=batch, batch_idx=batch_idx, dataloader_idx=dataloader_idx)
        else:
            return self.sft_training_step(batch=batch, batch_idx=batch_idx, dataloader_idx=dataloader_idx)
    # -- basic methods implemenration ends --#

    # ++ sft implementation begins ++#
    def sft_training_step(self, batch, batch_idx, dataloader_idx=0):
        log_dict = self.forward(batch=batch)
        log_dict = {f'train/{k}': v for k, v in log_dict.items()}
        self.log_dict(log_dict, sync_dist=True, prog_bar=True, batch_size=len(batch['idx']))
        return log_dict['train/total_loss']

    def forward(self, batch):
        latent_cot_config = self.model_kwargs.latent_cot_config
        max_compression_factor = latent_cot_config.max_compression_factor
        if isinstance(max_compression_factor, int):
            r = random.randint(1, max_compression_factor)
        elif isinstance(max_compression_factor, str):
            max_compression_factor = max_compression_factor.strip(",").split(",")
            r = int(random.choice(max_compression_factor))
        else:
            raise ValueError("max_compression_factor should be int or str")
        # 0: prepare inputs
        question = batch["question"]
        steps = batch["steps"]
        answer = batch["answer"]
        batch_size = len(question)

        # question: [pad, question, speed]
        auto_prob = latent_cot_config.get("replace_r_with_auto_prob", 0)
        if random.random() < auto_prob:
            speed = "auto"
        else:
            speed = r
        question_input_ids, question_attention_mask = self.prepare_inputs(
            question, padding_side="left", part="question", suffix=self.speed_template.format(speed)
        )
        question_inputs_embeds = self.embedding(question_input_ids)

        # steps: [pad, ###, steps]
        steps_input_ids, steps_attention_mask = self.prepare_inputs(
            steps, padding_side="left", part="steps", prefix=self.thinking_separator
        )
        if r == 1:
            steps_inputs_embeds = self.embedding(steps_input_ids)
            steps_labels = steps_input_ids
        else:
            # better skip this else branch as it is really complex
            steps_pad_lengths = -(steps_attention_mask - 1).sum(dim=-1)
            # make sure there are $k*r - 1$ pad tokens before first token ('###'), so that the first token will be at position $k*r$
            # left pad the steps_input_ids and steps_attention_mask
            # suppose r=3, and there are already 4 pad tokens, to make '###' the 6th token, we need to pad 1 more
            n_extra_left_pad_length = r - 1 - steps_pad_lengths % r
            steps_length_left_padded = steps_attention_mask.shape[1] + n_extra_left_pad_length.max()
            # make the whole seq divisible to r
            min_right_pad_length = r - steps_length_left_padded % r
            all_steps_input_ids = []
            all_steps_attention_mask = []
            for b, l_length in enumerate(n_extra_left_pad_length):
                r_length = min_right_pad_length + (r - 1 - l_length)
                if r_length == r:  # if we should pad r extra tokens to the right
                    l_length += r  # we pad it left instead
                    r_length = 0  # to keep the last compressed token not a pad token
                s_ids = steps_input_ids[b]
                s_attn_mask = steps_attention_mask[b]
                if l_length > 0:
                    s_ids = torch.cat(
                        [
                            torch.ones(l_length, device=s_ids.device, dtype=s_ids.dtype) * self.tokenizer.pad_token_id,
                            s_ids,
                        ]
                    )
                    s_attn_mask = torch.cat(
                        [torch.zeros(l_length, device=s_attn_mask.device, dtype=s_attn_mask.dtype), s_attn_mask]
                    )
                if r_length > 0:
                    s_ids = torch.cat(
                        [
                            s_ids,
                            torch.ones(r_length, device=s_ids.device, dtype=s_ids.dtype) * self.tokenizer.pad_token_id,
                        ]
                    )
                    s_attn_mask = torch.cat(
                        [s_attn_mask, torch.zeros(r_length, device=s_attn_mask.device, dtype=s_attn_mask.dtype)]
                    )
                all_steps_input_ids.append(s_ids)
                all_steps_attention_mask.append(s_attn_mask)
            padded_steps_input_ids = torch.stack(all_steps_input_ids, dim=0)
            padded_steps_attention_mask = torch.stack(all_steps_attention_mask, dim=0)
            padded_steps_inputs_embeds = self.embedding(padded_steps_input_ids)
            padded_steps_inputs_embeds *= padded_steps_attention_mask.unsqueeze(-1)

            padded_steps_length = padded_steps_inputs_embeds.shape[1]
            compressed_steps_length = padded_steps_length // r
            compressed_steps_inputs_embeds = padded_steps_inputs_embeds.reshape(
                batch_size, compressed_steps_length, r, padded_steps_inputs_embeds.shape[-1]
            ).sum(dim=2)
            compressed_steps_attention_mask = padded_steps_attention_mask.reshape(
                batch_size, compressed_steps_length, r
            ).sum(dim=2)
            if latent_cot_config.get("sqrt_mean", False):
                compressed_steps_attention_mask = compressed_steps_attention_mask.sqrt()
            compressed_steps_inputs_embeds /= compressed_steps_attention_mask.unsqueeze(-1) + 1e-5
            compressed_steps_attention_mask = (compressed_steps_attention_mask != 0).long()
            compressed_steps_labels = padded_steps_input_ids.reshape(batch_size, compressed_steps_length, r)
            # rand_steps_indices = torch.randint(0, r, (batch_size, compressed_steps_length, 1), device=steps_input_ids.device)
            rand_steps_indices = sample_indices_from_attention_mask_3d(
                padded_steps_attention_mask.view(batch_size, compressed_steps_length, r)
            )
            compressed_steps_labels = compressed_steps_labels.gather(dim=2, index=rand_steps_indices).squeeze(dim=2)

            # finally we are here:
            steps_inputs_embeds = compressed_steps_inputs_embeds
            steps_attention_mask = compressed_steps_attention_mask
            steps_labels = compressed_steps_labels

        # answer: [###, answer, eos, pad]
        answer_input_ids, answer_attention_mask = self.prepare_inputs(
            answer,
            padding_side="right",
            part="answer",
            prefix=self.thinking_separator,
            suffix=self.tokenizer.eos_token,
        )
        answer_inputs_embeds = self.embedding(answer_input_ids)

        question_length = question_inputs_embeds.shape[1]
        steps_length = steps_inputs_embeds.shape[1]

        inputs_embeds = torch.cat([question_inputs_embeds, steps_inputs_embeds, answer_inputs_embeds], dim=1)
        attention_mask = torch.cat([question_attention_mask, steps_attention_mask, answer_attention_mask], dim=1)
        position_ids = get_position_ids_from_attention_mask(attention_mask)
        labels = torch.cat([question_input_ids, steps_labels, answer_input_ids], dim=1)
        labels[labels == self.tokenizer.pad_token_id] = -100
        labels[:, :question_length] = -100

        outputs = self.llm.forward(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            labels=labels,
            output_hidden_states=True,
        )
        ce_loss = outputs.loss

        # latent loss
        steps_outputs = outputs.hidden_states[-1][:, question_length : question_length + steps_length, :]
        distributions = self.latent_policy.forward(steps_outputs)
        gold_embeds = inputs_embeds[:, question_length + 1 : question_length + steps_length + 1, :]
        pred_embeds = distributions.rsample()
        if latent_cot_config.get("embed_modeling_loss", "nll") == "nll":
            embed_modeling_loss = -distributions.log_prob(gold_embeds.detach() / self.embeds_std).mean(dim=-1)
        else:
            embed_modeling_loss = F.mse_loss(
                pred_embeds, gold_embeds.detach() / self.embeds_std, reduction="none"
            ).mean(dim=-1)
        embed_modeling_loss = (embed_modeling_loss * steps_attention_mask).sum() / steps_attention_mask.sum()

        entropy = distributions.entropy().mean(dim=-1)
        entropy = (entropy * steps_attention_mask).sum() / steps_attention_mask.sum()

        # pred_embed_forward  # only used in NLL loss for faster convergence
        if latent_cot_config.pred_embed_forward_weight != 0:
            second_input_embeds = torch.cat(
                [
                    question_inputs_embeds,  # question: [pad, question]
                    answer_inputs_embeds[:, 0:1, :],  #: ['###']
                    pred_embeds[:, 1:, :],  # [pad, steps]
                    answer_inputs_embeds,  # [###, answer, eos, pad]
                ],
                dim=1,
            )
            second_attention_mask = torch.cat(
                [
                    question_attention_mask,
                    torch.ones_like(answer_attention_mask[:, 0:1]),
                    steps_attention_mask[:, 1:],
                    answer_attention_mask,
                ],
                dim=1,
            )
            second_position_ids = get_position_ids_from_attention_mask(second_attention_mask)
            # only supervise the answer
            second_outputs = self.llm.forward(
                inputs_embeds=second_input_embeds,
                attention_mask=second_attention_mask,
                position_ids=second_position_ids,
                labels=labels,
            )
            pred_embed_forward_loss = second_outputs.loss
        else:
            pred_embed_forward_loss = 0.0

        # total loss
        total_loss = 0
        if latent_cot_config.get("ce_weight", 1) != 0:
            total_loss += ce_loss * latent_cot_config.ce_weight
        if latent_cot_config.get("embed_modeling_weight", 0) != 0:
            total_loss += embed_modeling_loss * latent_cot_config.embed_modeling_weight
        if latent_cot_config.get("entropy_weight", 0) != 0:
            total_loss += entropy * latent_cot_config.entropy_weight
        if latent_cot_config.get("pred_embed_forward_weight", 0) != 0:
            total_loss += pred_embed_forward_loss * latent_cot_config.pred_embed_forward_weight

        return {
            "total_loss": total_loss,
            "ce_loss": ce_loss,
            "pred_embed_forward_loss": pred_embed_forward_loss,
            "embed_modeling_loss": embed_modeling_loss,
            "entropy": entropy,
        }

    # -- sft training ends --#

    # ++ rl implementation begins ++#
    def init_rl(self):
        self.grpo_loss = grpo.GRPOLoss(rl_config=self.model_kwargs.rl_config)
        self.replay_buffer = grpo.ReplayBuffer()
        self.automatic_optimization = False

    @torch.no_grad()
    def filter_train_indices(self, dataloader_to_filter_indices):
        """
        this function is not used in our paper, but might be helpful for future work
        """
        train_indices = []
        for batch in tqdm.tqdm(dataloader_to_filter_indices, desc="filtering train indices"):
            q = batch["question"]
            a = batch["answer"]
            idx = batch["idx"]
            batch_size = idx.shape[0]  # (batch_size, 1)
            exp = self.batch_rollout(questions=q, gt_answers=a)
            mean_acc = exp.accuracies.reshape(batch_size, -1).mean(-1)  # (batch_size, 1)
            # remove too easy questions
            train_indices.extend(idx[mean_acc.cpu() < 1.0].tolist())
        self.text_logger.log(f"filtered {len(train_indices)} train indices")
        return train_indices

    def rl_training_step(self, batch, batch_idx, dataloader_idx=0):
        rl_config = self.model_kwargs.rl_config
        questions = batch["question"]
        answers = batch["answer"]
        self.replay_buffer.clear()
        optimizer = self.optimizers()

        experience = self.rollout(questions=questions, gt_answers=answers)
        self.replay_buffer.append(experience.to("cpu"))

        self.log_dict(
            {
                "train/rewards": experience.rewards.mean(),
                "train/accuracies": experience.accuracies.mean(),
                "train/n_latent_forward": experience.n_latent_forward.float().mean(),
            }
        )
        torch.cuda.empty_cache()
        experience_dataloader = DataLoader(
            dataset=self.replay_buffer,
            batch_size=rl_config.exp_batch_size,
            shuffle=True,
            collate_fn=grpo.join_experience_batch,
        )

        for experience in experience_dataloader:
            experience: grpo.Experience = experience.to(self.device)
            latent_logprobs, answer_logprobs = self.get_logprobs(e=experience)
            loss_dict = self.grpo_loss(
                latent_logprobs=latent_logprobs,
                answer_logprobs=answer_logprobs,
                experience=experience,
            )
            optimizer.zero_grad()
            self.manual_backward(loss_dict["total_loss"])
            grad_norm = clip_grad_norm_(self.parameters(), max_norm=1.0)
            optimizer.step()

            log_dict = {f"train/{k}": v for k, v in loss_dict.items()}
            log_dict["train/grad_norm"] = grad_norm
            self.log_dict(log_dict)

    @torch.no_grad()
    def rollout(self, questions: List[str], gt_answers) -> grpo.Experience:
        # 0: prepare variables
        rl_config = self.model_kwargs.rl_config
        batch_size = len(questions)
        group_size = rl_config.group_size

        group_questions = []
        for q in questions:
            group_questions.extend([q] * group_size)

        # 1: sample
        (question_input_ids, question_attention_mask, latent_inputs_embeds, latent_attention_mask, pred_ids) = (
            self.latent_generate(
                questions=group_questions,
                rl_mode=True,
            )
        )
        pred_answer_strings = self.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)

        # 2: calculate rewards
        n_latent_forward = latent_attention_mask.sum(dim=1)
        all_rewards = []
        all_accuracies = []
        all_advantages = []
        for sample_idx in range(batch_size):
            group_answers = pred_answer_strings[sample_idx * group_size : (sample_idx + 1) * group_size]
            group_n_latent_forward = n_latent_forward[sample_idx * group_size : (sample_idx + 1) * group_size]
            gt_answer = gt_answers[sample_idx]
            rewards, accuracies = self.get_group_rewards_and_acc(
                pred_answers=group_answers, gt_answer=gt_answer, n_latent_forward=group_n_latent_forward
            )
            advantages = grpo.group_advantages(rewards)
            all_rewards.append(rewards)
            all_accuracies.append(accuracies)
            all_advantages.append(advantages)

        rewards = torch.cat(all_rewards, dim=0)
        accuracies = torch.cat(all_accuracies, dim=0)
        advantages = torch.cat(all_advantages, dim=0)

        # 3: calculate logprobs
        experience = grpo.Experience(
            question_input_ids=question_input_ids,
            question_attention_mask=question_attention_mask,
            latent_inputs_embeds=latent_inputs_embeds,
            latent_attention_mask=latent_attention_mask,
            answer_input_ids=pred_ids,
            answer_attention_mask=pred_ids.ne(self.tokenizer.pad_token_id).long(),
            n_latent_forward=n_latent_forward.unsqueeze(1),
            rewards=rewards,
            accuracies=accuracies,
            advantages=advantages,
        )

        latent_logprobs, answer_logprobs = self.get_logprobs(experience)
        experience.latent_logprobs = latent_logprobs
        experience.answer_logprobs = answer_logprobs

        return experience

    def get_group_rewards_and_acc(
        self, pred_answers: List[str], gt_answer: str, n_latent_forward: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        rl_config = self.model_kwargs.rl_config
        group_size = len(pred_answers)

        accuracies = torch.zeros(size=(group_size, 1), device=self.device, dtype=torch.float32)
        for i, pred_answer in enumerate(pred_answers):
            pred_a = self.extract_answer_from_output(pred_answer)
            accuracies[i] = self.verify_answer(gt_answer=gt_answer, pred_answer=pred_a)

        rewards = accuracies.detach().clone()
        if rl_config.punish_latent_length:
            rewards /= n_latent_forward.unsqueeze(1)

        return rewards, accuracies

    def get_logprobs(self, e: grpo.Experience):
        question_length = e.question_input_ids.shape[1]
        latent_length = e.latent_inputs_embeds.shape[1]
        answer_length = e.answer_input_ids.shape[1]

        question_inputs_embeds = self.embedding(e.question_input_ids)
        answer_inputs_embeds = self.embedding(e.answer_input_ids)

        all_inputs_embeds = torch.cat([question_inputs_embeds, e.latent_inputs_embeds, answer_inputs_embeds], dim=1)
        all_attention_mask = torch.cat(
            [e.question_attention_mask, e.latent_attention_mask, e.answer_attention_mask], dim=1
        )

        all_position_ids = get_position_ids_from_attention_mask(all_attention_mask)

        all_outputs = self.llm.forward(
            inputs_embeds=all_inputs_embeds,
            attention_mask=all_attention_mask,
            position_ids=all_position_ids,
            output_hidden_states=True,
        )
        last_hidden_states_for_latents = all_outputs.hidden_states[-1][
            :, question_length - 1 : question_length + latent_length - 1
        ]
        distributions = self.latent_policy.forward(last_hidden_states_for_latents)
        latent_logprobs = distributions.log_prob(e.latent_inputs_embeds / self.embeds_std).mean(dim=-1)

        # logits for end of think
        logits_for_eol = []
        for b, latent_length in enumerate(e.n_latent_forward):
            logits_for_eol.append(all_outputs.logits[b, question_length + latent_length - 1])
        logits_for_eol = torch.stack(logits_for_eol, dim=0)
        # answer_logprobs
        answer_logits = torch.cat([logits_for_eol, all_outputs.logits[:, -answer_length:-1, :]], dim=1)
        answer_logprobs = F.log_softmax(answer_logits, dim=-1)
        answer_logprobs = answer_logprobs.gather(dim=-1, index=e.answer_input_ids.unsqueeze(-1)).squeeze(-1)

        return latent_logprobs, answer_logprobs

    # -- rl ends --#
