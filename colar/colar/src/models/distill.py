import torch.amp
import torch.nn.functional as F
import torch


from .model_base import LitCoTModelBase
from ..modules.projector import MLPProjector


class LitCoLaR(LitCoTModelBase):
    def __init__(
        self,
        model_kwargs,
        training_kwargs,
        all_config=None,
    ):
        super().__init__(model_kwargs=model_kwargs, training_kwargs=training_kwargs, all_config=all_config)

        self.latent_proj = MLPProjector(
            feature_size=self.llm.config.hidden_size,
        )

    def forward(self, batch):
        distill_config = self.model_kwargs.distill_config
        n_latents = distill_config.n_latents

        # 0: prepare inputs
        question = batch["question"]
        steps = batch["steps"]
        answer = batch["answer"]
        batch_size = len(question)

        # question: [pad, question, speed, ###]
        question_input_ids, question_attention_mask = self.prepare_inputs(
            question,
            padding_side="left",
            part="question",
            suffix=self.speed_template.format("auto") + self.thinking_separator,
        )
        # steps: [pad, steps, ###]
        steps_input_ids, steps_attention_mask = self.prepare_inputs(
            text_list=steps,
            padding_side="left",
            part="steps",
            suffix=self.thinking_separator,
        )
        # answer: [answer, eos, pad]
        answer_input_ids, answer_attention_mask = self.prepare_inputs(
            answer,
            padding_side="right",
            part="answer",
            suffix=self.tokenizer.eos_token,
        )

        # 1: question forward
        outputs = self.llm.forward(
            input_ids=question_input_ids,
            attention_mask=question_attention_mask,
            output_hidden_states=True,
        )
        last_hidden_state = outputs.hidden_states[-1][:, -1:, :]

        # 2: teacher forward:
        teacher_input_ids = torch.cat([steps_input_ids, answer_input_ids], dim=1)
        teacher_attention_mask = torch.cat(
            [question_attention_mask, steps_attention_mask, answer_attention_mask], dim=1
        )
        teacher_outputs = self.llm.forward(
            input_ids=teacher_input_ids,
            attention_mask=teacher_attention_mask,
            output_hidden_states=True,
            past_key_values=outputs.past_key_values,
            labels=teacher_input_ids,
        )
        teacher_loss = teacher_outputs.loss
        steps_length = steps_input_ids.shape[1]
        # distill the ':' token
        # <...steps...> <Answer> <:>, so +1 here
        teacher_hidden_states_to_distill = torch.stack(teacher_outputs.hidden_states, dim=-1)[
            :, steps_length + 1, :, 1:
        ]  # embedding layer is frozen

        # 3: student forward:
        all_attention_mask = question_attention_mask
        for i in range(n_latents):
            all_attention_mask = torch.cat(
                [all_attention_mask, torch.ones(size=(batch_size, 1), device=all_attention_mask.device)], dim=1
            )
            inputs_embeds = last_hidden_state
            inputs_embeds = self.latent_proj(last_hidden_state)
            outputs = self.llm.forward(
                inputs_embeds=inputs_embeds,
                attention_mask=all_attention_mask,
                output_hidden_states=True,
                past_key_values=outputs.past_key_values,
            )
            last_hidden_state = outputs.hidden_states[-1][:, -1:, :]
        # add ### token and answer
        student_answer_input_ids = torch.cat(
            [
                torch.ones(size=(batch_size, 1), device=last_hidden_state.device, dtype=torch.long)
                * self.thinking_separator_id,
                answer_input_ids,
            ],
            dim=1,
        )
        student_answer_attention_mask = torch.cat(
            [
                all_attention_mask,
                torch.ones(size=(batch_size, 1), device=all_attention_mask.device, dtype=torch.long),
                answer_attention_mask,
            ],
            dim=1,
        )
        outputs = self.llm.forward(
            input_ids=student_answer_input_ids,
            attention_mask=student_answer_attention_mask,
            output_hidden_states=True,
            past_key_values=outputs.past_key_values,
            labels=student_answer_input_ids,
        )
        student_loss = outputs.loss
        # student inputs: [<###>, <answer>, <:>]
        # so ':' is at index 2
        student_hidden_states_to_distill = torch.stack(outputs.hidden_states, dim=-1)[:, 2, :, 1:]

        # loss
        distill_loss = F.smooth_l1_loss(student_hidden_states_to_distill, teacher_hidden_states_to_distill.detach())
        total_loss = (
            teacher_loss * distill_config.alpha
            + student_loss * distill_config.beta
            + distill_loss * distill_config.gamma
        )

        return {
            "total_loss": total_loss,
            "teacher_loss": teacher_loss,
            "student_loss": student_loss,
            "distill_loss": distill_loss,
        }
