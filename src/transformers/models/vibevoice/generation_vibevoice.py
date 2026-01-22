# coding=utf-8
# Copyright 2025 The Microsoft Team and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import importlib
from dataclasses import dataclass
from typing import Optional

import torch

from ...generation import (
    GenerateDecoderOnlyOutput,
    GenerationConfig,
    GenerationMixin,
    LogitsProcessor,
    LogitsProcessorList,
)
from ...generation.stopping_criteria import StoppingCriteriaList
from ...generation.utils import GenerateNonBeamOutput
from ...utils import logging


logger = logging.get_logger(__name__)


@dataclass
class VibeVoiceGenerateOutput(GenerateDecoderOnlyOutput):
    """
    Outputs of VibeVoiceForConditionalGeneration.generate.

    Args:
        audio (`list(torch.FloatTensor)` of length `batch_size`):
            The generated audio.
        reach_max_step_sample (`torch.BoolTensor`, *optional*):
            Boolean tensor indicating which samples reached maximum generation steps.
    """

    audio: Optional[list[torch.FloatTensor]] = None
    reach_max_step_sample: Optional[torch.BoolTensor] = None


class VibeVoiceTokenConstraintProcessor(LogitsProcessor):
    """Constrains token generation to only valid tokens during speech generation."""

    def __init__(self, valid_token_ids: list[int], device: torch.device = None):
        self.valid_token_ids = torch.tensor(valid_token_ids, dtype=torch.long, device=device)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # Create a mask for valid tokens
        mask = torch.full_like(scores, float("-inf"))
        mask[:, self.valid_token_ids] = 0

        # Apply mask to scores
        scores = scores + mask
        return scores


class VibeVoiceGenerationMixin(GenerationMixin):

    def _prepare_generation_config(self, generation_config: Optional[GenerationConfig], **kwargs):
        """
        This method overrides [~generation.utils.GenerationMixin._prepare_generation_config].

        It extracts VibeVoice-specific parameters for the generation config and sets up the default noise scheduler (if
        not provided).

        VibeVoice-specific parameters include:
        - `noise_scheduler`: An optional noise scheduler instance to use instead of the default.
        - `monitor_progress`: A callable to monitor generation progress. If provided, this function can be called to
            report the progress of the audio generation. The function takes a tensor argument `p` of shape `(n, 2)`,
            where `n` is the batch size. `p[i, 0]` contains the current generation step for batch item `i`, and `p[i, 1]`
            contains the maximum generation steps for batch item `i`. No return value is expected.
        - `cfg_scale`: A classifier-free guidance scale to use during generation.
        - `n_diffusion_steps`: Number of diffusion steps to use during generation of each audio chunk.
        """
        # Call the base class method to load from default generation_config.json
        generation_config, model_kwargs = super()._prepare_generation_config(generation_config, **kwargs)

        # try creating VibeVoice noise scheduler if not provided
        noise_scheduler = model_kwargs.pop("noise_scheduler", kwargs.pop("noise_scheduler", None))
        # TODO (ebezzam) ok with this? so user doesn't need to define noise scheduler each time?
        # Alternatively, require user to create noise scheduler outside
        if noise_scheduler is None:
            try:
                scheduler_class = getattr(
                    importlib.import_module("diffusers"), generation_config.noise_scheduler_class
                )
                noise_scheduler = scheduler_class(**generation_config.noise_scheduler_config)
            except ImportError:
                raise ImportError(
                    "The default VibeVoice noise scheduler could not be created because `diffusers` is not installed or "
                    "the specified noise scheduler class is not available. "
                    f"Please install with `pip install diffusers` and verify that {generation_config.noise_scheduler_class} exists."
                )
        generation_config.noise_scheduler = noise_scheduler
        if not (
            hasattr(generation_config.noise_scheduler, "set_timesteps")
            and hasattr(generation_config.noise_scheduler, "step")
            and hasattr(generation_config.noise_scheduler, "timesteps")
        ):
            raise ValueError(
                "The provided noise scheduler is not compatible with VibeVoice generation. "
                "It must implement `set_timesteps` and `step` methods, and have a `timesteps` attribute."
            )
        if "monitor_progress" in model_kwargs:
            generation_config.monitor_progress = model_kwargs.pop("monitor_progress")
        if "cfg_scale" in model_kwargs:
            generation_config.cfg_scale = model_kwargs.pop("cfg_scale")
        if "n_diffusion_steps" in model_kwargs:
            generation_config.n_diffusion_steps = model_kwargs.pop("n_diffusion_steps")
        return generation_config, model_kwargs

    def _sample(
        self,
        input_ids: torch.LongTensor,
        logits_processor: LogitsProcessorList,
        stopping_criteria: StoppingCriteriaList,
        generation_config: GenerationConfig,
        synced_gpus: bool = False,
        streamer: Optional["BaseStreamer"] = None,
        **model_kwargs,
    ) -> GenerateNonBeamOutput | torch.LongTensor:
        """
        This method overrides [~generation.utils.GenerationMixin._sample].
        To ease maintenance, modifications are marked with the comment "VibeVoice specific".

        Indeed, VibeVoice model requires a custom generation sampling step:
        1. Extract VibeVoice-specific parameters and setup diffusion components
        2. Setup negative generation for classifier-free guidance
        3. Generate tokens with diffusion-based speech synthesis for speech tokens
        4. Apply stopping criteria (EOS token and max length)

        Expected streamer object can take the form of the following from the original code base:
        https://github.com/vibevoice-community/VibeVoice/blob/b9d561240ada3ee5d8fb5812bebb32f7ecfd97ae/vibevoice/modular/streamer.py#L13
        """
        # init values
        pad_token_id = generation_config._pad_token_tensor
        output_attentions = generation_config.output_attentions
        output_hidden_states = generation_config.output_hidden_states
        output_scores = generation_config.output_scores
        output_logits = generation_config.output_logits
        return_dict_in_generate = generation_config.return_dict_in_generate
        has_eos_stopping_criteria = any(hasattr(criteria, "eos_token_id") for criteria in stopping_criteria)
        do_sample = generation_config.do_sample

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        raw_logits = () if (return_dict_in_generate and output_logits) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # keep track of which sequences are already finished
        batch_size, cur_len = input_ids.shape[:2]
        this_peer_finished = False
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
        model_kwargs = self._get_initial_cache_position(cur_len, input_ids.device, model_kwargs)

        # *************** VibeVoice specific ***************
        input_values = model_kwargs.pop("input_values", None)
        padding_mask = model_kwargs.pop("padding_mask", None)
        noise_scheduler = generation_config.noise_scheduler
        monitor_progress = getattr(generation_config, "monitor_progress", None)
        cfg_scale = generation_config.cfg_scale
        n_diffusion_steps = generation_config.n_diffusion_steps
        diffusion_head_device = next(self.model.diffusion_head.parameters()).device

        # State tracking
        acoustic_cache = None
        semantic_cache = None
        is_prefill = True
        inputs_embeds = None

        # Output audio
        audio_chunks = [[] for _ in range(batch_size)]

        # Token constraints for VibeVoice - only allow speech tokens
        valid_tokens = [
            self.config.audio_bos_token_id,
            self.config.audio_eos_token_id,
            self.config.audio_diffusion_token_id,
            self.config.eos_token_id,
        ]
        if hasattr(self.config, "bos_token_id") and self.config.bos_token_id is not None:
            valid_tokens.append(self.config.bos_token_id)
        token_constraint_processor = VibeVoiceTokenConstraintProcessor(valid_tokens, device=input_ids.device)
        logits_processor.append(token_constraint_processor)

        # Setup negative generation for classifier-free guidance
        negative_kwargs = {
            "input_ids": torch.full(
                (batch_size, 1), self.config.audio_bos_token_id, dtype=torch.long, device=input_ids.device
            ),
            "attention_mask": torch.ones((batch_size, 1), dtype=torch.long, device=input_ids.device),
            "max_new_tokens": generation_config.max_new_tokens,
            "noise_scheduler": noise_scheduler,
        }
        negative_generation_config, negative_model_kwargs = self._prepare_generation_config(
            generation_config, **negative_kwargs
        )
        _, negative_model_input_name, negative_model_kwargs = self._prepare_model_inputs(
            None, negative_generation_config.bos_token_id, negative_model_kwargs
        )
        self._prepare_special_tokens(negative_generation_config, True, device=input_ids.device)
        negative_generation_config.use_cache = self.config.use_cache
        negative_model_kwargs["use_cache"] = self.config.use_cache
        negative_input_ids = negative_kwargs["input_ids"].to(input_ids.device)

        negative_input_ids_length = negative_input_ids.shape[1]
        negative_has_default_max_length = (
            negative_kwargs.get("max_length") is None and negative_generation_config.max_length is not None
        )
        negative_has_default_min_length = (
            negative_kwargs.get("min_length") is None and negative_generation_config.min_length is not None
        )
        negative_generation_config = self._prepare_generated_length(
            generation_config=negative_generation_config,
            has_default_max_length=negative_has_default_max_length,
            has_default_min_length=negative_has_default_min_length,
            model_input_name=negative_model_input_name,
            inputs_tensor=negative_kwargs["input_ids"],
            input_ids_length=negative_input_ids_length,
        )

        negative_max_cache_length = negative_generation_config.max_length - 1
        negative_batch_size = negative_kwargs["input_ids"].shape[0]
        self._prepare_cache_for_generation(
            negative_generation_config, negative_model_kwargs, None, negative_batch_size, negative_max_cache_length
        )
        negative_model_kwargs["cache_position"] = torch.arange(
            negative_input_ids_length, device=input_ids.device, dtype=torch.long
        )
        for k, v in negative_model_kwargs.items():
            if isinstance(v, torch.Tensor):
                negative_model_kwargs[k] = v.to(device=input_ids.device)

        # Calculate generation limits for progress tracking
        initial_length = input_ids.shape[-1]
        initial_length_per_sample = model_kwargs["attention_mask"].sum(dim=-1)
        max_steps = generation_config.max_length - initial_length
        max_step_per_sample = torch.min(
            generation_config.max_length - initial_length_per_sample,
            torch.full_like(initial_length_per_sample, max_steps),
        )
        step = 0
        # ============================================

        while self._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device):
            # *************** VibeVoice specific ***************
            if monitor_progress is not None:
                current_steps = torch.full((batch_size,), step, dtype=torch.long, device=input_ids.device)
                progress_tensor = torch.stack((current_steps, max_step_per_sample), dim=1)
                monitor_progress(progress_tensor)
            # ============================================

            # prepare model inputs
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

            # *************** VibeVoice specific ***************
            # Handle prefill vs normal generation
            if is_prefill:
                # First step: process speech inputs for conditioning
                if input_values is not None and padding_mask is not None:
                    model_inputs.update(
                        {
                            "input_values": input_values.to(device=input_ids.device),
                            "padding_mask": padding_mask.to(input_ids.device),
                        }
                    )
                is_prefill = False
            else:
                # Subsequent steps: use embeddings from previous step
                model_inputs.pop("inputs_embeds", None)
                model_inputs["inputs_embeds"] = inputs_embeds

            # Set logits_to_keep for positive pass
            model_inputs["logits_to_keep"] = 1
            outputs = self(**model_inputs, return_dict=True)
            # ============================================

            # synced_gpus: don't waste resources running the code we don't need; kwargs must be updated before skipping
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs,
                model_kwargs,
                is_encoder_decoder=self.config.is_encoder_decoder,
            )
            if synced_gpus and this_peer_finished:
                continue

            # Copy is needed to avoid keeping a hanging ref to outputs.logits which may be very large for first iteration
            # (the clone itself is always small)
            next_token_logits = outputs.logits[:, -1, :].to(copy=True, dtype=torch.float32, device=input_ids.device)

            # pre-process distribution
            next_token_scores = logits_processor(input_ids, next_token_logits)

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_token_scores,)
                if output_logits:
                    raw_logits += (next_token_logits,)
                if output_attentions:
                    decoder_attentions += (outputs.attentions,)
                if output_hidden_states:
                    decoder_hidden_states += (outputs.hidden_states,)

            # *************** VibeVoice specific ***************
            # token selection
            # NOTE (ebezzam): For VibeVoice, we always use deterministic token selection
            # regardless of do_sample setting. The real sampling happens in the diffusion
            # process for audio generation, not in token selection.
            if do_sample:
                logger.warning(
                    "VibeVoice generation does not support sampling-based token selection. "
                    "Tokens will be selected using argmax regardless of do_sample=True."
                )
            next_tokens = torch.argmax(next_token_scores, dim=-1)

            # ============================================

            # finished sentences should have their next token be a padding token
            if has_eos_stopping_criteria:
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

            # update generated ids, model inputs, and length for next step
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            if streamer is not None:
                streamer.put(next_tokens.cpu())

            # *************** VibeVoice specific ***************
            # Handle speech start tokens
            diffusion_start_mask = unfinished_sequences.bool() & (next_tokens == self.config.audio_bos_token_id)
            if diffusion_start_mask.any():
                diffusion_start_indices = diffusion_start_mask.nonzero(as_tuple=False).squeeze(1)
                # Update negative generation state
                for sample_idx in diffusion_start_indices.tolist():
                    negative_model_kwargs["attention_mask"][sample_idx, :] = 0
                    negative_model_kwargs["attention_mask"][sample_idx, -1] = 1
                if "past_key_values" in negative_model_kwargs and negative_model_kwargs["past_key_values"] is not None:
                    for layer_idx in range(len(negative_model_kwargs["past_key_values"])):
                        k_cache = negative_model_kwargs["past_key_values"].layers[layer_idx].keys
                        v_cache = negative_model_kwargs["past_key_values"].layers[layer_idx].values
                        for sample_idx in diffusion_start_indices.tolist():
                            k_cache[sample_idx, :, -1, :] = k_cache[sample_idx, :, 0, :].clone()
                            v_cache[sample_idx, :, -1, :] = v_cache[sample_idx, :, 0, :].clone()
                for sample_idx in diffusion_start_indices.tolist():
                    negative_input_ids[sample_idx, -1] = self.config.audio_bos_token_id

            # Prepare embeddings for next iteration
            next_inputs_embeds = self.get_input_embeddings()(next_tokens).unsqueeze(1)

            # Handle diffusion tokens
            diffusion_mask = unfinished_sequences.bool() & (next_tokens == self.config.audio_diffusion_token_id)
            negative_outputs = None
            if diffusion_mask.any():
                diffusion_indices = diffusion_mask.nonzero(as_tuple=False).squeeze(1)

                # Negative pass for classifier-free guidance
                negative_model_inputs = self.prepare_inputs_for_generation(negative_input_ids, **negative_model_kwargs)
                if negative_model_inputs["inputs_embeds"] is None and inputs_embeds is not None:
                    negative_model_inputs["inputs_embeds"] = inputs_embeds
                    negative_model_inputs["input_ids"] = None
                negative_model_inputs["logits_to_keep"] = 0

                negative_outputs = self(**negative_model_inputs, return_dict=True)
                negative_model_kwargs = self._update_model_kwargs_for_generation(
                    negative_outputs,
                    negative_model_kwargs,
                    is_encoder_decoder=False,
                )
                negative_input_ids = torch.cat([negative_input_ids, next_tokens[:, None]], dim=-1)

                # Diffusion process with classifier-free guidance
                positive_condition = outputs.last_hidden_state[diffusion_indices, -1, :]
                negative_condition = negative_outputs.last_hidden_state[diffusion_indices, -1, :]

                noise_scheduler.set_timesteps(num_inference_steps=n_diffusion_steps)
                condition = torch.cat([positive_condition, negative_condition], dim=0).to(diffusion_head_device)
                speech = torch.randn(condition.shape[0], self.config.acoustic_tokenizer_config.hidden_size).to(condition)

                for timestep in noise_scheduler.timesteps:
                    half = speech[: len(speech) // 2]
                    combined = torch.cat([half, half], dim=0)
                    eps = self.model.diffusion_head(
                        combined, timestep.repeat(combined.shape[0]).to(combined), condition=condition
                    )
                    cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
                    half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
                    eps = torch.cat([half_eps, half_eps], dim=0)
                    speech = noise_scheduler.step(eps, timestep, speech).prev_sample
                speech_latent = speech[: len(speech) // 2].unsqueeze(1)

                # Decode to audio
                scaled_latent = speech_latent / self.latent_scaling_factor.to(
                    speech_latent.device
                ) - self.latent_bias_factor.to(speech_latent.device)
                if len(diffusion_indices) != batch_size:
                    # pad non-diffusion samples with zeros
                    padded_latent = torch.zeros(batch_size, scaled_latent.shape[1], scaled_latent.shape[2]).to(
                        scaled_latent.device
                    )
                    padded_latent[diffusion_indices] = scaled_latent
                else:
                    padded_latent = scaled_latent
                audio_output = self.model.acoustic_tokenizer.decode(
                    padded_latent.to(self.model.acoustic_tokenizer.device),
                    padding_cache=acoustic_cache,
                    use_cache=self.config.use_cache,
                )
                audio_chunk = audio_output.audio
                acoustic_cache = audio_output.padding_cache

                # Store and stream audio
                for i, sample_idx in enumerate(diffusion_indices):
                    idx = sample_idx.item()
                    audio_chunks[idx].append(audio_chunk[i])
                if streamer is not None:
                    streamer.put(audio_chunk, diffusion_indices)

                # Get semantic features for next step
                semantic_outputs = self.model.semantic_tokenizer.encode(
                    audio_chunk,
                    padding_cache=semantic_cache,
                    use_cache=self.config.use_cache,
                )
                semantic_features = semantic_outputs.latents[diffusion_indices]
                semantic_cache = semantic_outputs.padding_cache

                # Combine features for next input
                acoustic_embed = self.model.acoustic_connector(speech_latent)
                semantic_embed = self.model.semantic_connector(semantic_features)
                diffusion_embeds = acoustic_embed + semantic_embed
                next_inputs_embeds[diffusion_indices] = diffusion_embeds

            inputs_embeds = next_inputs_embeds
            step += 1
            # ============================================

            unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, scores)
            this_peer_finished = unfinished_sequences.max() == 0
            cur_len += 1

            # This is needed to properly delete outputs.logits which may be very large for first iteration
            # Otherwise a reference to outputs is kept which keeps the logits alive in the next iteration
            del outputs

            # *************** VibeVoice specific ***************
            del negative_outputs
            # ============================================

        if streamer is not None:
            streamer.end()

        # *************** VibeVoice specific ***************
        # Prepare final audio outputs
        final_audio_outputs = []
        for sample_chunks in audio_chunks:
            if sample_chunks:
                final_audio_outputs.append(torch.cat(sample_chunks, dim=-1))
            else:
                final_audio_outputs.append(None)

        if return_dict_in_generate:
            return VibeVoiceGenerateOutput(
                sequences=input_ids,
                scores=scores,
                logits=raw_logits,
                attentions=decoder_attentions,
                hidden_states=decoder_hidden_states,
                past_key_values=model_kwargs.get("past_key_values"),
                audio=final_audio_outputs,
                reach_max_step_sample=(step >= max_step_per_sample).any(),
            )
        else:
            # NOTE (ebezzam): new tokens in input_ids are simply speech tokens (mainly `audio_diffusion_token_id`)
            # so returning `input_ids` is insufficient for generated audio
            return final_audio_outputs
        # ============================================
