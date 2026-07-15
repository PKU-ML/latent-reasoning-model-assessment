"""
    Implementation of Monet based on Qwen-2.5-VL series
    Similar architecture to SkiLa but uses implicit latent visual reasoning instead of sketch mode.

    Key differences from SkiLa:
    - Monet uses <abs_vis_token> as the latent token for implicit visual reasoning
    - No separate sketch extractor needed - uses the LLM's own hidden states as latent representations
    - Latent vectors are injected at the positions of latent tokens each step
"""
import torch
import torch.nn as nn
from transformers import Qwen2_5_VLForConditionalGeneration

import os
from typing import Optional, Union, Tuple

from transformers.generation.configuration_utils import GenerationConfig
from transformers.generation.logits_process import LogitsProcessorList


from transformers.generation.stopping_criteria import (
    StoppingCriteriaList,
)
from transformers.generation.utils import (
    GenerateNonBeamOutput,
    GenerateEncoderDecoderOutput,
    GenerateDecoderOnlyOutput,
)

from transformers.generation.streamers import BaseStreamer
from transformers.cache_utils import Cache

from transformers.modeling_outputs import ModelOutput
from dataclasses import dataclass


class MonetModel(Qwen2_5_VLForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)

    def _sample(
        self,
        input_ids: torch.LongTensor,
        logits_processor: LogitsProcessorList,
        stopping_criteria: StoppingCriteriaList,
        generation_config: GenerationConfig,
        synced_gpus: bool,
        streamer: Optional["BaseStreamer"],
        **model_kwargs,
    ) -> Union[GenerateNonBeamOutput, torch.LongTensor]:
        """
        Custom sampling method for Monet's latent reasoning.

        The latent reasoning works as follows:
        1. At each step, we detect if we're at a latent token position
        2. If so, we take the previous step's hidden state as the latent vector
        3. We inject this latent vector back into the current forward pass
        4. This creates an implicit feedback loop for visual reasoning

        Unlike SkiLa which uses explicit sketch tokens and a separate sketch encoder,
        Monet relies on the LLM's hidden states to serve as latent visual representations.
        """
        # init values
        pad_token_id = generation_config._pad_token_tensor
        output_attentions = generation_config.output_attentions
        output_hidden_states = generation_config.output_hidden_states
        output_scores = generation_config.output_scores
        output_logits = generation_config.output_logits
        return_dict_in_generate = generation_config.return_dict_in_generate
        max_length = generation_config.max_length
        has_eos_stopping_criteria = any(hasattr(criteria, "eos_token_id") for criteria in stopping_criteria)
        do_sample = generation_config.do_sample

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        raw_logits = () if (return_dict_in_generate and output_logits) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
            )

        # keep track of which sequences are already finished
        batch_size, cur_len = input_ids.shape
        this_peer_finished = False
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
        model_kwargs = self._get_initial_cache_position(input_ids.shape[1], input_ids.device, model_kwargs)

        model_forward = self.__call__
        if isinstance(model_kwargs.get("past_key_values"), Cache):
            is_compileable = model_kwargs["past_key_values"].is_compileable and self._supports_static_cache
            if getattr(self, "hf_quantizer", None) is not None:
                is_compileable &= self.hf_quantizer.is_compileable
            is_compileable = is_compileable and not generation_config.disable_compile
            if is_compileable and (
                self.device.type == "cuda" or generation_config.compile_config._compile_all_devices
            ):
                os.environ["TOKENIZERS_PARALLELISM"] = "0"
                model_forward = self.get_compiled_call(generation_config.compile_config)

        # Latent mode init
        in_latent_mode = torch.zeros(batch_size, dtype=torch.bool, device=input_ids.device)
        latent_hidden_state = None  # Store hidden state from previous step for injection

        # Get latent token ID from config
        latent_token_id = getattr(self.config, 'latent_token_id', None)
        if latent_token_id is None:
            # Fallback: try to find it from tokenizer
            latent_token_id = getattr(self.config, 'latent_start_id', 151666)

        # Latent reasoning parameters
        max_latent_steps = getattr(self.config, 'max_latent_steps', 20)
        latent_steps_orig = torch.tensor([max_latent_steps] * batch_size, dtype=torch.long, device=input_ids.device)
        latent_remaining_steps = latent_steps_orig.clone()
        is_prefill = True

        # Store original pixel values for re-processing at latent steps
        pixel_values = model_kwargs.get("pixel_values")
        image_grid_thw = model_kwargs.get("image_grid_thw")

        while self._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device):
            # prepare model inputs
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

            # prepare variable output controls (note: some models won't accept all output controls)
            model_inputs.update({"output_attentions": output_attentions} if output_attentions else {})
            model_inputs.update({"output_hidden_states": output_hidden_states} if output_hidden_states else {})

            # Handle latent mode - inject previous hidden state if in latent mode
            if in_latent_mode.any():
                # Find latent token positions in the current input
                latent_mask = (input_ids == latent_token_id)
                if latent_mask.any():
                    # Get latent positions for each sequence in batch
                    for b in range(batch_size):
                        if in_latent_mode[b] and latent_hidden_state is not None:
                            latent_poss = torch.where(latent_mask[b])[0]
                            if len(latent_poss) > 0:
                                # Inject latent hidden state at latent token positions
                                if "latent_poss" not in model_inputs:
                                    model_inputs["latent_poss"] = []
                                    model_inputs["latents"] = []
                                model_inputs["latent_poss"].append(latent_poss)
                                model_inputs["latents"].append(latent_hidden_state[b].unsqueeze(0))

                # Convert lists to tensors if we have latent injections
                if "latent_poss" in model_inputs and len(model_inputs["latent_poss"]) > 0:
                    # For simplicity, take the first latent position and state
                    # In a more sophisticated implementation, we could handle multiple
                    model_inputs["latent_poss"] = torch.stack([lp[0] for lp in model_inputs["latent_poss"]])
                    model_inputs["latents"] = torch.cat(model_inputs["latents"], dim=0)
                    # Remove batch dimension if singleton
                    if model_inputs["latents"].dim() == 3 and model_inputs["latents"].shape[0] == 1:
                        model_inputs["latents"] = model_inputs["latents"].squeeze(0)

            if is_prefill:
                outputs = self(**model_inputs, return_dict=True)
                is_prefill = False
            else:
                outputs = model_forward(**model_inputs, return_dict=True)

            # synced_gpus: don't waste resources running the code we don't need; kwargs must be updated before skipping
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs,
                model_kwargs,
                is_encoder_decoder=self.config.is_encoder_decoder,
            )
            if synced_gpus and this_peer_finished:
                continue

            next_token_logits = outputs.logits[:, -1, :].clone().float()
            next_token_logits = next_token_logits.to(input_ids.device)

            next_token_scores = logits_processor(input_ids, next_token_logits)

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_token_scores,)
                if output_logits:
                    raw_logits += (next_token_logits,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                    )
                    if self.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)

                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )

            # token selection
            if do_sample:
                probs = nn.functional.softmax(next_token_scores, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                next_tokens = torch.argmax(next_token_scores, dim=-1)

            # finished sentences should have their next token be a padding token
            if has_eos_stopping_criteria:
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

            # Check if we just entered latent mode (current token is latent token)
            last_tokens = input_ids[:, -1]
            entered_latent = (last_tokens == latent_token_id) & (~in_latent_mode)

            # Check if we should exit latent mode
            # Exit when we generate a non-latent token after entering
            exited_latent = in_latent_mode & (last_tokens != latent_token_id)

            # Update remaining latent steps
            just_entered = (~in_latent_mode) & (last_tokens == latent_token_id)
            latent_remaining_steps = torch.where(just_entered, latent_steps_orig, latent_remaining_steps)
            latent_remaining_steps = latent_remaining_steps - in_latent_mode.long()

            # Force exit if out of latent steps
            force_end = in_latent_mode & (latent_remaining_steps <= 0)
            exited_latent = exited_latent | force_end

            # Update latent mode state
            in_latent_mode = (in_latent_mode | entered_latent) & (~exited_latent)

            # Get hidden state for next latent injection
            # Use latent_hidden_state from output (set in monkey patch forward)
            if hasattr(outputs, 'latent_hidden_state') and outputs.latent_hidden_state is not None:
                latent_hidden_state = outputs.latent_hidden_state.clone()

            # Force next token to be latent token when in latent mode
            next_tokens[in_latent_mode] = latent_token_id

            # Append token
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)

            if streamer is not None:
                streamer.put(next_tokens.cpu())

            unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, scores)
            this_peer_finished = unfinished_sequences.max() == 0
            cur_len += 1

            # Clear pixel values after first step to save memory
            if cur_len == 2:
                model_kwargs["pixel_values"] = None
                model_kwargs["pixel_values_videos"] = None

            del outputs

        if streamer is not None:
            streamer.end()

        if return_dict_in_generate:
            if self.config.is_encoder_decoder:
                return GenerateEncoderDecoderOutput(
                    sequences=input_ids,
                    scores=scores,
                    logits=raw_logits,
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                )
            else:
                return GenerateDecoderOnlyOutput(
                    sequences=input_ids,
                    scores=scores,
                    logits=raw_logits,
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                )
        else:
            return input_ids
