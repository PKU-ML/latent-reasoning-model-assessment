import torch

from .model_base import LitCoTModelBase
from ..utils.utils import get_position_ids_from_attention_mask


class LitCot(LitCoTModelBase):
    def __init__(
        self,
        model_kwargs,
        training_kwargs,
        all_config=None,
    ):
        super().__init__(model_kwargs=model_kwargs, training_kwargs=training_kwargs, all_config=all_config)

    def forward(self, batch):
        # 0: prepare inputs
        question = batch["question"]
        steps = batch["steps"]
        answer = batch["answer"]

        # 1: question forward
        # question: [pad, question, speed]
        question_input_ids, question_attention_mask = self.prepare_inputs(
            question, padding_side="left", part="question", suffix=self.speed_template.format(1)
        )
        # steps: [pad, ###, steps, ###]
        steps_input_ids, steps_attention_mask = self.prepare_inputs(
            steps,
            padding_side="left",
            part="steps",
            prefix=self.thinking_separator,
            suffix=self.thinking_separator,
        )
        # answer: [answer, eos, pad]
        answer_input_ids, answer_attention_mask = self.prepare_inputs(
            answer,
            padding_side="right",
            part="answer",
            suffix=self.tokenizer.eos_token,
        )
        question_length = question_input_ids.shape[1]

        input_ids = torch.cat([question_input_ids, steps_input_ids, answer_input_ids], dim=1)
        attention_mask = torch.cat([question_attention_mask, steps_attention_mask, answer_attention_mask], dim=1)
        labels = input_ids.detach().clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        labels[:, :question_length] = -100

        position_ids = get_position_ids_from_attention_mask(attention_mask)
        outputs = self.llm.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            labels=labels,
            output_hidden_states=False,
        )

        return {
            "total_loss": outputs.loss,
        }
