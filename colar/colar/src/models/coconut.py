"""
Modified from https://github.com/facebookresearch/coconut
"""

import math
import torch

from .model_base import LitCoTModelBase


class LitCoconut(LitCoTModelBase):
    def __init__(
        self,
        model_kwargs,
        training_kwargs,
        all_config=None,
    ):
        super().__init__(model_kwargs=model_kwargs, training_kwargs=training_kwargs, all_config=all_config)

    def forward(self, batch):
        coconut_config = self.model_kwargs.coconut_config
        current_epoch = self.current_epoch
        current_stage = math.ceil(
            current_epoch + 1 / coconut_config.n_epochs_per_stage
        )  # load pretrained ckpt, start from stage 1
        current_stage = min(coconut_config.max_n_stage, current_stage)
        n_latents = coconut_config.n_latents_per_step * current_stage

        # 0: prepare inputs
        question = batch["question"]
        steps = batch["steps"]
        answer = batch["answer"]
        batch_size = len(question)

        # 1: question
        # question: [pad, question, speed, ###]
        question_input_ids, all_attention_mask = self.prepare_inputs(
            question,
            padding_side="left",
            part="question",
            suffix=self.speed_template.format("auto") + self.thinking_separator,
        )
        outputs = self.llm.forward(
            input_ids=question_input_ids,
            attention_mask=all_attention_mask,
            output_hidden_states=True,
        )
        last_hidden_state = outputs.hidden_states[-1][:, -1:, :]

        # 2: latent forward:
        for i in range(n_latents):
            all_attention_mask = torch.cat(
                [all_attention_mask, torch.ones(size=(batch_size, 1), device=all_attention_mask.device)], dim=1
            )
            if coconut_config.get("coconut_proj"):
                inputs_embeds = self.latent_proj(last_hidden_state)
            else:
                inputs_embeds = last_hidden_state
            outputs = self.llm.forward(
                inputs_embeds=inputs_embeds,
                attention_mask=all_attention_mask,
                output_hidden_states=True,
                past_key_values=outputs.past_key_values,
            )
            last_hidden_state = outputs.hidden_states[-1][:, -1:, :]

        # 3: steps and answer
        if current_stage < coconut_config.max_n_stage:
            steps = [
                "\n".join(s.split("\n")[current_stage:]) + "\n" for s in steps
            ]  # remove first $current_stage$ steps, make sure at least one token
        else:
            steps = ["\n"] * batch_size  # remove all steps
        # steps: [pad, steps, ###]
        steps_input_ids, steps_attention_mask = self.prepare_inputs(
            text_list=steps, padding_side="left", part="steps", suffix=self.thinking_separator
        )
        # answer: [answer, eos, pad]
        answer_input_ids, answer_attention_mask = self.prepare_inputs(
            answer,
            padding_side="right",
            part="answer",
            suffix=self.tokenizer.eos_token,
        )
        final_input_ids = torch.cat([steps_input_ids, answer_input_ids], dim=1)
        all_attention_mask = torch.cat([all_attention_mask, steps_attention_mask, answer_attention_mask], dim=1)

        outputs = self.llm.forward(
            input_ids=final_input_ids,
            attention_mask=all_attention_mask,
            output_hidden_states=False,
            past_key_values=outputs.past_key_values,
            labels=final_input_ids,
        )

        return {
            "total_loss": outputs.loss,
        }
