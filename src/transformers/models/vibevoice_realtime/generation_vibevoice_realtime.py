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
from typing import TYPE_CHECKING, Optional

import torch

from transformers.cache_utils import DynamicCache
from transformers.generation.logits_process import LogitsProcessorList
from transformers.modeling_outputs import BaseModelOutputWithPast

from ...generation import (
    BaseStreamer,
    GenerateDecoderOnlyOutput,
    GenerationConfig,
    GenerationMixin,
    LogitsProcessor,
)
from ...generation.stopping_criteria import EosTokenCriteria, MaxLengthCriteria, StoppingCriteriaList
from ...utils import logging


if TYPE_CHECKING:
    from ...generation.streamers import BaseStreamer

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

    audio: list[torch.FloatTensor] | None = None
    reach_max_step_sample: torch.BoolTensor | None = None


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


class VibeVoiceRealTimeGenerationMixin(GenerationMixin):
    def _get_stopping_criteria(
        self,
        *args,
        **kwargs,
    ) -> StoppingCriteriaList:
        criteria = super()._get_stopping_criteria(*args, **kwargs)

        kept_criteria = StoppingCriteriaList()
        for criterion in criteria:
            if not isinstance(criterion, (MaxLengthCriteria, EosTokenCriteria)):
                logger.warning(
                    f"VibeVoiceRealTime does not support {criterion.__class__.__name__} stopping criteria, it will be ignored. Generation uses a EOS classifier and a max_length criteria."
                )
            else:
                kept_criteria.append(criterion)
        return kept_criteria

    def _prepare_generation_config(self, generation_config: GenerationConfig | None, **kwargs):
        """
        This method overrides [~generation.utils.GenerationMixin._prepare_generation_config].

        It extracts VibeVoice-specific parameters from kwargs and sets up the default noise scheduler (if not provided).

        VibeVoice-specific parameters include:
        - `noise_scheduler`: An optional noise scheduler instance to use instead of the default.
        - `cfg_scale`: A classifier-free guidance scale to use during generation.
        - `n_diffusion_steps`: Number of diffusion steps to use during generation of each audio chunk.
        - `monitor_progress`: A callable to monitor generation progress. If provided, this function can be called to
            report the progress of the audio generation. The function takes a tensor argument `p` of shape `(n, 3)`,
            where `n` is the batch size. `p[i, 0]` contains the current generation step for batch item `i`, `p[i, 1]`
            contains the maximum generation steps for batch item `i` (which may vary based on input length), and
            `p[i, 2]` contains the actual completion step for finished samples. No return value is expected.
        - `voice_preset`: A dictionary of past key values for the language and TTS decoders for a specific voice.

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
        if "voice_preset" in model_kwargs:
            generation_config.voice_preset = model_kwargs.pop("voice_preset")
        if "text_window_size" in model_kwargs:
            generation_config.tts_text_window_size = model_kwargs.pop("text_window_size")
        if "speech_window_size" in model_kwargs:
            generation_config.tts_speech_window_size = model_kwargs.pop("speech_window_size")
        return generation_config, model_kwargs

    def _rebuild_history_prompt(self, entry, device):
        """Main part is rebuilding DynamicCache (for past_key_values) from raw tensors."""

        past_key_value_dict = entry["past_key_values"]
        key_cache = past_key_value_dict["key_cache"]
        value_cache = past_key_value_dict["value_cache"]
        if len(key_cache) != len(value_cache):
            raise ValueError("key_cache and value_cache must have the same length")
        cache = DynamicCache()
        for layer_idx, (k, v) in enumerate(zip(key_cache, value_cache)):
            cache.update(
                key_states=k.to(device),
                value_states=v.to(device),
                layer_idx=layer_idx,
            )
        # TODO (ebezzam) ok to use model output here?
        return BaseModelOutputWithPast(
            last_hidden_state=entry["last_hidden_state"].to(device),
            past_key_values=cache,
        )

    def _sample(
        self,
        input_ids: torch.LongTensor,
        logits_processor: LogitsProcessorList,
        stopping_criteria: StoppingCriteriaList,
        generation_config: GenerationConfig,
        synced_gpus: bool = False,
        streamer: Optional["BaseStreamer"] = None,
        **model_kwargs,
    ):
        """
        This method overrides [~generation.utils.GenerationMixin._sample].
        To ease maintenance, modifications are marked with the comment "VibeVoice specific".

        VibeVoiceRealTime differs from standard VibeVoice generation:
        1. **Windowed text processing**: Text is fed in small windows enabling streaming input
        2. **Interleaved generation**: After each text window, generate multiple speech tokens
        3. **No semantic tokenizer**: Only uses acoustic tokenizer (no semantic feedback loop)
        4. **TTS LM separate from main LM**: TTS LM tracks interleaved text+speech sequence for conditioning
        5. **TTS EOS classifier**: Uses binary classifier instead of standard EOS tokens to detect completion
        6. **Separate negative TTS LM**: For classifier-free guidance in speech generation

        The windowed approach enables true streaming: you don't need the full text upfront, and audio is generated
        incrementally as text arrives.
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

        # *************** VibeVoice specific ***************
        noise_scheduler = generation_config.noise_scheduler
        monitor_progress = getattr(generation_config, "monitor_progress", None)
        cfg_scale = generation_config.cfg_scale
        n_diffusion_steps = generation_config.n_diffusion_steps
        text_window_size = generation_config.text_window_size
        speech_window_size = generation_config.speech_window_size
        diffusion_head_device = next(self.diffusion_head.parameters()).device
        if do_sample:
            logger.warning(
                "VibeVoiceRealTime generation does not support sampling-based token selection. An internal diffusion process naturally samples latent representations, and a seed can be set for a deterministic output."
            )

        # previous outputs for a specific voice preset
        voice_preset = generation_config.voice_preset
        lm_outputs = self._rebuild_history_prompt(voice_preset["lm"], device=input_ids.device)
        tts_lm_outputs = self._rebuild_history_prompt(voice_preset["tts_lm"], device=input_ids.device)
        tts_lm_negative_outputs = self._rebuild_history_prompt(voice_preset["neg_tts_lm"], device=input_ids.device)

        # State tracking
        batch_size = input_ids.shape[0]
        this_peer_finished = False
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
        acoustic_cache = None
        finished_tags = torch.zeros(batch_size, dtype=torch.bool, device=input_ids.device)
        text_window_index = 0
        audio_chunks = [[] for _ in range(batch_size)]

        # Prepare initial LM model inputs
        lm_input_ids = torch.full(
            (batch_size, lm_outputs.last_hidden_state.shape[1]),
            pad_token_id,
            dtype=torch.long,
            device=input_ids.device,
        )
        lm_kwargs = {
            "attention_mask": torch.ones_like(lm_input_ids, dtype=torch.long),
            "max_new_tokens": generation_config.max_new_tokens,
        }
        _, lm_model_kwargs = self._prepare_generation_config(generation_config, **lm_kwargs)
        lm_model_kwargs["use_cache"] = self.config.use_cache

        # Prepare TTS LM inputs and negative inputs for CFG
        tts_lm_input_ids = torch.full(
            (batch_size, tts_lm_outputs.last_hidden_state.shape[1]),
            pad_token_id,
            dtype=torch.long,
            device=input_ids.device,
        )
        tts_lm_attention_mask = torch.ones_like(tts_lm_input_ids, dtype=torch.long)
        tts_lm_kwargs = {
            "attention_mask": tts_lm_attention_mask,
            "max_new_tokens": generation_config.max_new_tokens,
        }
        _, tts_lm_model_kwargs = self._prepare_generation_config(generation_config, **tts_lm_kwargs)
        tts_lm_model_kwargs["use_cache"] = self.config.use_cache
        tts_lm_negative_kwargs = {
            "attention_mask": torch.ones((batch_size, 1), dtype=torch.long, device=input_ids.device),
            "max_new_tokens": generation_config.max_new_tokens,
        }
        _, tts_lm_negative_model_kwargs = self._prepare_generation_config(generation_config, **tts_lm_negative_kwargs)
        tts_lm_negative_input_ids = torch.full(
            (batch_size, 1), pad_token_id, dtype=torch.long, device=input_ids.device
        )
        tts_lm_negative_model_kwargs["use_cache"] = self.config.use_cache

        # Initialize kwargs from prefilled outputs
        first_text_window_size = min(text_window_size, input_ids.shape[1])
        lm_model_kwargs = self._update_model_kwargs_for_generation(
            lm_outputs, lm_model_kwargs, num_new_tokens=first_text_window_size
        )
        tts_lm_model_kwargs = self._update_model_kwargs_for_generation(
            tts_lm_outputs, tts_lm_model_kwargs, num_new_tokens=first_text_window_size
        )
        tts_lm_negative_model_kwargs = self._update_model_kwargs_for_generation(
            tts_lm_negative_outputs, tts_lm_negative_model_kwargs
        )

        # Calculate generation limits for progress tracking
        initial_length = tts_lm_input_ids.shape[-1]
        initial_length_per_sample = model_kwargs["attention_mask"].sum(dim=-1)
        max_steps = generation_config.max_new_tokens
        max_step_per_sample = torch.min(
            generation_config.max_new_tokens - (initial_length_per_sample - initial_length),
            torch.full_like(initial_length_per_sample, max_steps),
        )
        generation_config.max_length = generation_config.max_new_tokens + initial_length
        completion_steps = torch.zeros(batch_size, dtype=torch.long, device=input_ids.device)
        step = 0

        # ============================================

        while self._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device):
            # *************** VibeVoice specific ***************
            # Check for external streaming termination
            if streamer is not None and hasattr(streamer, "finished_flags"):
                if any(streamer.finished_flags):
                    break

            if monitor_progress is not None:
                current_steps = torch.full((batch_size,), step, dtype=torch.long, device=input_ids.device)
                current_steps[finished_tags] = completion_steps[finished_tags]
                progress_tensor = torch.stack((current_steps, max_step_per_sample, completion_steps), dim=1)
                monitor_progress(progress_tensor)

            if finished_tags.all():
                break

            # Get current text window
            cur_input_tts_text_ids = input_ids[
                :, text_window_index * text_window_size : (text_window_index + 1) * text_window_size
            ]
            next_text_window_size = input_ids[
                :, (text_window_index + 1) * text_window_size : (text_window_index + 2) * text_window_size
            ].shape[1]
            text_window_index += 1

            if cur_input_tts_text_ids.shape[1] > 0:
                lm_input_ids = torch.cat([lm_input_ids, cur_input_tts_text_ids], dim=-1)
                tts_lm_input_ids = torch.cat([tts_lm_input_ids, cur_input_tts_text_ids], dim=-1)

                if tts_lm_input_ids.shape[1] >= generation_config.max_length:
                    finished_tags[:] = True
                    completion_steps[:] = step + 1
                    break

                step += cur_input_tts_text_ids.shape[1]

                # Forward pass through LM
                lm_model_inputs = self.prepare_inputs_for_generation(
                    lm_input_ids, next_sequence_length=cur_input_tts_text_ids.shape[1], **lm_model_kwargs
                )
                lm_outputs = self.forward_lm(**lm_model_inputs, return_dict=True)
                lm_model_kwargs = self._update_model_kwargs_for_generation(
                    lm_outputs, lm_model_kwargs, num_new_tokens=next_text_window_size
                )

                # Forward pass through TTS LM with text
                tts_lm_model_inputs = self.prepare_inputs_for_generation(
                    input_ids=tts_lm_input_ids,
                    next_sequence_length=cur_input_tts_text_ids.shape[1],
                    **tts_lm_model_kwargs,
                )
                tts_lm_model_inputs["inputs_embeds"] = lm_outputs.last_hidden_state
                tts_lm_additional_inputs = {
                    "tts_text_masks": torch.ones_like(tts_lm_input_ids[:, -1:], dtype=torch.long)
                }
                tts_lm_outputs = self(**tts_lm_model_inputs, **tts_lm_additional_inputs, return_dict=True)
                tts_lm_model_kwargs = self._update_model_kwargs_for_generation(tts_lm_outputs, tts_lm_model_kwargs)

            # Generate speech tokens
            diffusion_indices = torch.arange(batch_size, device=input_ids.device)[~finished_tags]

            if diffusion_indices.numel() > 0:
                for cur_speech_index in range(speech_window_size):
                    positive_condition = tts_lm_outputs.last_hidden_state[diffusion_indices, -1, :]
                    negative_condition = tts_lm_negative_outputs.last_hidden_state[diffusion_indices, -1, :]
                    noise_scheduler.set_timesteps(num_inference_steps=n_diffusion_steps)
                    condition = torch.cat([positive_condition, negative_condition], dim=0).to(diffusion_head_device)
                    speech = torch.randn(condition.shape[0], self.config.audio_config.hidden_size).to(condition)
                    # NOTE (ebezzam) exact match at this point

                    for timestep in noise_scheduler.timesteps:
                        half = speech[: len(speech) // 2]
                        combined = torch.cat([half, half], dim=0)
                        eps = self.diffusion_head(
                            combined, timestep.repeat(combined.shape[0]).to(combined), condition=condition
                        )
                        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
                        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
                        eps = torch.cat([half_eps, half_eps], dim=0)
                        speech = noise_scheduler.step(eps, timestep, speech).prev_sample
                    # NOTE (ebezzam) slight deviation for bfloat16 at this point but seems numerical

                    speech_latent = speech[: len(speech) // 2].unsqueeze(1)

                    # Decode to audio
                    scaled_latent = speech_latent / self.model.latent_scaling_factor.to(
                        speech_latent.device
                    ) - self.model.latent_bias_factor.to(speech_latent.device)

                    if len(diffusion_indices) != batch_size:
                        padded_latent = torch.zeros(batch_size, scaled_latent.shape[1], scaled_latent.shape[2]).to(
                            scaled_latent
                        )
                        padded_latent[diffusion_indices] = scaled_latent
                    else:
                        padded_latent = scaled_latent
                    audio_output = self.model.audio_tower.decode(
                        padded_latent.to(self.model.audio_tower.device),
                        padding_cache=acoustic_cache,
                        use_cache=self.config.use_cache,
                    )
                    audio_chunk = audio_output.audio
                    acoustic_cache = audio_output.padding_cache

                    # Store and stream audio
                    for sample_idx in diffusion_indices:
                        idx = sample_idx.item()
                        audio_chunks[idx].append(audio_chunk[idx])
                    if streamer is not None:
                        streamer.put(audio_chunk, diffusion_indices)

                    # Append speech token to TTS LM sequence
                    tts_lm_input_ids = torch.cat([tts_lm_input_ids, torch.ones_like(tts_lm_input_ids[:, -1:])], dim=-1)
                    if tts_lm_input_ids.shape[1] >= generation_config.max_length:
                        break
                    step += 1

                    # Forward pass through TTS LM with speech
                    acoustic_embed = self.model.multi_modal_projector(speech_latent)
                    tts_lm_model_inputs = self.prepare_inputs_for_generation(
                        tts_lm_input_ids, next_sequence_length=1, **tts_lm_model_kwargs
                    )
                    tts_lm_model_inputs["inputs_embeds"] = acoustic_embed
                    tts_lm_additional_inputs = {
                        "tts_text_masks": torch.zeros_like(tts_lm_input_ids[:, -1:], dtype=torch.long)
                    }
                    tts_lm_outputs = self(**tts_lm_model_inputs, **tts_lm_additional_inputs, return_dict=True)
                    if cur_speech_index == speech_window_size - 1 and next_text_window_size > 0:
                        tts_lm_model_kwargs = self._update_model_kwargs_for_generation(
                            tts_lm_outputs, tts_lm_model_kwargs, num_new_tokens=next_text_window_size
                        )
                    else:
                        tts_lm_model_kwargs = self._update_model_kwargs_for_generation(
                            tts_lm_outputs, tts_lm_model_kwargs
                        )

                    # Update negative TTS LM
                    tts_lm_negative_input_ids = torch.cat(
                        [tts_lm_negative_input_ids, torch.ones_like(tts_lm_input_ids[:, -1:])], dim=-1
                    )
                    tts_lm_negative_model_inputs = self.prepare_inputs_for_generation(
                        tts_lm_negative_input_ids, next_sequence_length=1, **tts_lm_negative_model_kwargs
                    )
                    tts_lm_negative_model_inputs["inputs_embeds"] = acoustic_embed
                    tts_lm_negative_additional_inputs = {
                        "tts_text_masks": torch.zeros_like(tts_lm_negative_input_ids[:, -1:], dtype=torch.long)
                    }
                    tts_lm_negative_outputs = self(
                        **tts_lm_negative_model_inputs, **tts_lm_negative_additional_inputs, return_dict=True
                    )
                    tts_lm_negative_model_kwargs = self._update_model_kwargs_for_generation(
                        tts_lm_negative_outputs, tts_lm_negative_model_kwargs
                    )

                    # Check EOS classifier
                    tts_eos_logits = torch.sigmoid(tts_lm_outputs.logits)[diffusion_indices]
                    eos_mask = tts_eos_logits[:, 0] > 0.5
                    if eos_mask.any():
                        eos_indices = diffusion_indices[eos_mask]
                        finished_tags[eos_indices] = True
                        completion_steps[eos_indices] = step + 1
                        if streamer is not None:
                            streamer.end(eos_indices)

                        # Report completion progress
                        if monitor_progress is not None:
                            current_steps = torch.full(
                                (batch_size,), step + 1, dtype=torch.long, device=input_ids.device
                            )
                            current_steps[finished_tags] = max_step_per_sample[finished_tags]
                            progress_tensor = torch.stack(
                                (current_steps, max_step_per_sample, completion_steps), dim=1
                            )
                            monitor_progress(progress_tensor)

            if tts_lm_input_ids.shape[1] >= generation_config.max_length:
                unfinished_indices = ~finished_tags
                if unfinished_indices.any():
                    finished_tags[unfinished_indices] = True
                    completion_steps[unfinished_indices] = step + 1
                    unfinished_idx_list = unfinished_indices.nonzero(as_tuple=False).squeeze(1)
                    if streamer is not None:
                        streamer.end(unfinished_idx_list)
                break

            # Handle per-sample max length
            max_length_reached = step >= max_step_per_sample
            new_max_length_mask = max_length_reached & ~finished_tags
            if new_max_length_mask.any():
                finished_tags[new_max_length_mask] = True
                completion_steps[new_max_length_mask] = step + 1
                new_max_length_indices = new_max_length_mask.nonzero(as_tuple=False).squeeze(1)
                if streamer is not None:
                    streamer.end(new_max_length_indices)

            unfinished_sequences = unfinished_sequences & ~finished_tags.long()
            this_peer_finished = unfinished_sequences.max() == 0
            # ============================================

        if streamer is not None:
            streamer.end()

        # *************** VibeVoice specific ***************
        final_audio_outputs = []
        for sample_chunks in audio_chunks:
            if sample_chunks:
                final_audio_outputs.append(torch.cat(sample_chunks, dim=-1))
            else:
                final_audio_outputs.append(None)

        if return_dict_in_generate:
            return VibeVoiceGenerateOutput(
                sequences=tts_lm_input_ids,
                scores=scores,
                logits=raw_logits,
                attentions=decoder_attentions,
                hidden_states=decoder_hidden_states,
                past_key_values=tts_lm_model_kwargs.get("past_key_values"),
                audio=final_audio_outputs,
                reach_max_step_sample=completion_steps >= generation_config.max_length,
            )
        else:
            return final_audio_outputs
        # ============================================
