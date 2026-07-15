"""
Monkey patch for Qwen2.5-VL's forward function.

This patch enables gradient flow through the visual module for adversarial attacks.
"""
import torch
from typing import Optional, List, Union, Tuple
from torch.nn import CrossEntropyLoss

import transformers.models.qwen2_5_vl.modeling_qwen2_5_vl
from transformers.modeling_outputs import ModelOutput
from dataclasses import dataclass


@dataclass
class Qwen2_5_VLCausalLMOutputWithPast(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    past_key_values: Optional[List[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    rope_deltas: Optional[torch.LongTensor] = None
    inputs_embeds: Optional[torch.FloatTensor] = None


def is_torchdynamo_compiling():
    """Check if torch.compile is being used."""
    try:
        import torch._dynamo
        return torch._dynamo.is_compiling()
    except ImportError:
        return False


def replace_qwen2_5_with_vision_forward():
    """
    Replace Qwen2.5-VL's forward with a version that supports gradient flow through visual module.
    """
    transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLForConditionalGeneration.forward = qwen2_5_vision_forward


def qwen2_5_vision_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    pixel_values: Optional[torch.Tensor] = None,
    pixel_values_videos: Optional[torch.FloatTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    rope_deltas: Optional[torch.LongTensor] = None,
    cache_position: Optional[torch.LongTensor] = None,
    second_per_grid_ts: Optional[torch.Tensor] = None,
    **kwargs,
) -> Union[Tuple, Qwen2_5_VLCausalLMOutputWithPast]:
    """
    Qwen2.5-VL forward with gradient support for visual module.

    This forward function ensures that gradients can flow back through the visual
    encoder to the pixel_values input, enabling adversarial attacks on the image space.
    """
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if inputs_embeds is None:
        inputs_embeds = self.model.get_input_embeddings()(input_ids)

    # ========== 处理图像嵌入 ==========
    if pixel_values is not None:
        # 确保 pixel_values 类型与 visual 模块一致
        pixel_values = pixel_values.type(self.visual.dtype)

        # 获取图像嵌入 - 这是关键，确保梯度流经此处
        image_embeds = self.model.get_image_features(pixel_values, image_grid_thw)
        image_embeds = torch.cat(image_embeds, dim=0)

        # 获取 image token 的数量
        image_token_id = self.config.image_token_id
        if input_ids is None:
            image_mask = inputs_embeds == self.get_input_embeddings()(
                torch.tensor(image_token_id, dtype=torch.long, device=inputs_embeds.device)
            )
            image_mask = image_mask.all(-1)
        else:
            image_mask = input_ids == image_token_id

        n_image_tokens = image_mask.sum().item()
        n_image_features = image_embeds.shape[0]
        if n_image_tokens != n_image_features:
            raise ValueError(
                f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
            )

        # 将 image embeddings 散布到 input embeddings 中
        image_mask_unsqueeze = image_mask.unsqueeze(-1).expand_as(inputs_embeds)
        image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.masked_scatter(image_mask_unsqueeze, image_embeds)

    # ========== 处理视频嵌入 ==========
    if pixel_values_videos is not None:
        pixel_values_videos = pixel_values_videos.type(self.visual.dtype)
        video_embeds = self.model.get_image_features(pixel_values_videos, video_grid_thw)
        video_embeds = torch.cat(video_embeds, dim=0)

        video_token_id = self.config.video_token_id
        video_mask = input_ids == video_token_id
        n_video_tokens = video_mask.sum().item()
        n_video_features = video_embeds.shape[0]
        if n_video_tokens != n_video_features:
            raise ValueError(
                f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}"
            )

        video_mask_unsqueeze = video_mask.unsqueeze(-1).expand_as(inputs_embeds)
        video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.masked_scatter(video_mask_unsqueeze, video_embeds)

    if attention_mask is not None:
        attention_mask = attention_mask.to(inputs_embeds.device)

    # ========== 处理位置编码 ==========
    if position_ids is None:
        cached_rope_deltas = getattr(self.model, '_rope_deltas', None)

        prefill_compiled_stage = is_torchdynamo_compiling() and (
            (input_ids is not None and input_ids.shape[1] != 1)
            or (inputs_embeds is not None and inputs_embeds.shape[1] != 1)
        )
        prefill_noncompiled_stage = not is_torchdynamo_compiling() and (
            (cache_position is not None and cache_position[0] == 0)
            or (past_key_values is None or past_key_values.get_seq_length() == 0)
        )

        if (prefill_compiled_stage or prefill_noncompiled_stage) or cached_rope_deltas is None:
            position_ids, rope_deltas = self.model.get_rope_index(
                input_ids,
                image_grid_thw,
                video_grid_thw,
                second_per_grid_ts=second_per_grid_ts,
                attention_mask=attention_mask,
            )
            self.model._rope_deltas = rope_deltas
        else:
            batch_size, seq_length, _ = inputs_embeds.shape
            position_ids = torch.arange(seq_length, device=inputs_embeds.device)
            position_ids = position_ids.view(1, 1, -1).expand(3, batch_size, -1).clone()
            if cache_position is not None:
                delta = (cache_position[0] + cached_rope_deltas).to(inputs_embeds.device)
            else:
                delta = torch.zeros((batch_size, seq_length), device=inputs_embeds.device)
            delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=1)
            position_ids = position_ids + delta.to(position_ids.device).long()

    # ========== 语言模型前向传播 ==========
    outputs = self.model.language_model(
        input_ids=None,
        position_ids=position_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        cache_position=cache_position,
        **kwargs
    )

    hidden_states = outputs[0]
    logits = self.lm_head(hidden_states)

    loss = None
    if labels is not None:
        logits = logits.float()
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_fct = CrossEntropyLoss()
        shift_logits = shift_logits.view(-1, self.config.vocab_size)
        shift_labels = shift_labels.view(-1)
        shift_labels = shift_labels.to(shift_logits.device)
        loss = loss_fct(shift_logits, shift_labels)

    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    return Qwen2_5_VLCausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        rope_deltas=getattr(self.model, '_rope_deltas', None),
        inputs_embeds=inputs_embeds,
    )