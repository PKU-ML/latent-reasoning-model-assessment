"""
Monkey patch for Monet's forward function - TEST VERSION with latent vector storage.

This patch adds support for latent_poss and latents parameters to inject
latent vectors at specific positions during forward pass, enabling
implicit latent visual reasoning without vLLM.
Also stores latent vectors during inference for analysis.
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
    logits: torch.FloatTensor = None
    past_key_values: Optional[List[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    rope_deltas: Optional[torch.LongTensor] = None
    inputs_embeds: Optional[torch.FloatTensor] = None
    latent_hidden_state: Optional[torch.FloatTensor] = None


def replace_qwen2_5_with_monet_forward():
    """
    Replace Qwen2.5-VL's forward with Monet's forward that supports latent injection and storage.
    """
    transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLForConditionalGeneration.forward = qwen2_5_monet_forward


def qwen2_5_monet_forward(
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
    # Monet specific parameters
    latent_poss: Optional[torch.LongTensor] = None,
    latents: Optional[torch.Tensor] = None,
) -> Union[Tuple, Qwen2_5_VLCausalLMOutputWithPast]:
    """
    Monet's forward function with latent vector injection support and storage for testing.

    Parameters:
        latent_poss: Positions of latent tokens in the input sequence
        latents: Latent vectors to inject at the latent token positions

    The latent injection mechanism:
    1. At each decoding step, if latent_poss and latents are provided,
       we inject the latent vectors into the input embeddings at the specified positions
    2. This allows the model to perform implicit visual reasoning by
       feeding its own hidden states back as "visual" representations
    """
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    #print(self._latent_hidden_state_list)
    if inputs_embeds is None:
        inputs_embeds = self.model.get_input_embeddings()(input_ids)

        # Handle latent vector injection
        if latent_poss is not None and latents is not None:
            #print('shape1: ',latents.shape)
            #print('poss: ', latent_poss)
            latent_poss = latent_poss.to(inputs_embeds.device)
            latents = latents.to(inputs_embeds.device, inputs_embeds.dtype)

            # Expand latents if needed
            if latents.dim() == 2:
                latents = latents.unsqueeze(1)  # Add sequence dimension

            #print('shape2: ',latents.shape)
            # Inject latents at specified positions
        
        if getattr(self, '_latent_hidden_state_list', None) is not None and len(self._latent_hidden_state_list) > 0:
            # Replace from pos onwards until we hit a position where input_ids differs from previous
            for l, j in enumerate(range(self.first_latent_poss[0], inputs_embeds.shape[1])):
                if j > 0 and input_ids[0, j] != input_ids[0, j - 1]:
                    break  # Stop before the position with different input_ids
                inputs_embeds[0, j] = self._latent_hidden_state_list[l].to(inputs_embeds.device).to(inputs_embeds.dtype)

        if pixel_values is not None:
            pixel_values = pixel_values.type(self.visual.dtype)
            image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)

            # Only count actual image tokens, NOT latent tokens
            image_token_id = self.config.image_token_id
            latent_token_id = getattr(self.config, 'latent_token_id', None)

            # Create mask for image tokens only (exclude latent tokens)
            token_mask = (input_ids == image_token_id)

            n_image_tokens = token_mask.sum().item()
            n_image_features = image_embeds.shape[0]

            if n_image_tokens != n_image_features:
                raise ValueError(
                    f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                )

            mask_unsqueezed = token_mask.unsqueeze(-1)
            mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
            image_mask = mask_expanded.to(inputs_embeds.device)

            image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        if pixel_values_videos is not None:
            pixel_values_videos = pixel_values_videos.type(self.visual.dtype)
            video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
            n_video_tokens = (input_ids == self.config.video_token_id).sum().item()
            n_video_features = video_embeds.shape[0]
            if n_video_tokens != n_video_features:
                raise ValueError(
                    f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}"
                )

            mask = input_ids == self.config.video_token_id
            mask_unsqueezed = mask.unsqueeze(-1)
            mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
            video_mask = mask_expanded.to(inputs_embeds.device)

            video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        if attention_mask is not None:
            attention_mask = attention_mask.to(inputs_embeds.device)

    # if we get 4D attention mask we cannot calculate rope deltas anymore.
    cached_rope_deltas = getattr(self, '_rope_deltas', None)

    if position_ids is None and (attention_mask is None or attention_mask.ndim == 2):
        # calculate RoPE index once per generation in the pre-fill stage only
        if (
            (cache_position is not None and cache_position[0] == 0)
            or cached_rope_deltas is None
            or (past_key_values is None or past_key_values.get_seq_length() == 0)
        ):

            position_ids, rope_deltas = self.model.get_rope_index(
                input_ids,
                image_grid_thw,
                video_grid_thw,
                second_per_grid_ts,
                attention_mask,
            )
            self._rope_deltas = rope_deltas
        # then use the prev pre-calculated rope-deltas to get the correct position ids
        else:
            batch_size, seq_length, _ = inputs_embeds.shape
            delta = (
                (cache_position[0] + cached_rope_deltas).to(inputs_embeds.device)
                if cache_position is not None
                else 0
            )
            position_ids = torch.arange(seq_length, device=inputs_embeds.device)
            position_ids = position_ids.view(1, -1).expand(batch_size, -1)
            if cache_position is not None:  # otherwise `deltas` is an int `0`
                delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
            position_ids = position_ids.add(delta)
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)
    #print(inputs_embeds[-15:])

    outputs = self.model(
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
    )

    hidden_states = outputs[0]
    # Return hidden states for latent reasoning
    latent_hidden_state = hidden_states[:, -1, :] if hidden_states is not None else None

    logits = self.lm_head(hidden_states)

    loss = None
    if labels is not None:
        # Upcast to float if we need to compute the loss to avoid potential precision issues
        logits = logits.float()
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = CrossEntropyLoss()
        shift_logits = shift_logits.view(-1, self.config.vocab_size)
        shift_labels = shift_labels.view(-1)

        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        loss = loss_fct(shift_logits, shift_labels)

    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    return Qwen2_5_VLCausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=hidden_states,
        attentions=outputs.attentions,
        rope_deltas=outputs.rope_deltas if hasattr(outputs, 'rope_deltas') else None,
        inputs_embeds=inputs_embeds,
        latent_hidden_state=latent_hidden_state,
    )