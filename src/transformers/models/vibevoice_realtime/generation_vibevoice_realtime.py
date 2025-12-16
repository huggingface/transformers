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
import queue
import time
from dataclasses import dataclass
from queue import Queue
from typing import Optional, Union
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from transformers.modeling_outputs import BaseModelOutputWithPast, ModelOutput
import torch
from transformers.cache_utils import DynamicCache

from ...generation import (
    BaseStreamer,
    GenerateDecoderOnlyOutput,
    GenerationConfig,
    GenerationMixin,
    LogitsProcessor,
)
from ...generation.stopping_criteria import MaxLengthCriteria, StoppingCriteriaList
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


# TODO (ebezzam) used in Gradio demo: https://github.com/vibevoice-community/VibeVoice/blob/main/demo/gradio_demo.py
class AudioStreamer(BaseStreamer):
    """
    Audio streamer that stores audio chunks in queues for each sample in the batch.
    This allows streaming audio generation for multiple samples simultaneously.

    Parameters:
        batch_size (`int`):
            The batch size for generation
        stop_signal (`any`, *optional*):
            The signal to put in the queue when generation ends. Defaults to None.
        timeout (`float`, *optional*):
            The timeout for the audio queue. If `None`, the queue will block indefinitely.
    """

    def __init__(
        self,
        batch_size: int,
        stop_signal: Optional[any] = None,
        timeout: Optional[float] = None,
    ):
        self.batch_size = batch_size
        self.stop_signal = stop_signal
        self.timeout = timeout

        self.audio_queues = [Queue() for _ in range(batch_size)]
        self.finished_flags = [False for _ in range(batch_size)]
        self.sample_indices_map = {}  # Map from sample index to queue index

    def put(self, audio_chunks: torch.Tensor, sample_indices: torch.Tensor):
        """
        Receives audio chunks and puts them in the appropriate queues.

        Args:
            audio_chunks: Tensor of shape (num_samples, ...) containing audio chunks
            sample_indices: Tensor indicating which samples these chunks belong to
        """
        for i, sample_idx in enumerate(sample_indices):
            idx = sample_idx.item()
            if idx < self.batch_size and not self.finished_flags[idx]:
                # Convert to numpy or keep as tensor based on preference
                audio_chunk = audio_chunks[i].detach().cpu()
                self.audio_queues[idx].put(audio_chunk, timeout=self.timeout)

    def end(self, sample_indices: Optional[torch.Tensor] = None):
        """
        Signals the end of generation for specified samples or all samples.

        Args:
            sample_indices: Optional tensor of sample indices to end. If None, ends all.
        """
        if sample_indices is None:
            for idx in range(self.batch_size):
                if not self.finished_flags[idx]:
                    self.audio_queues[idx].put(self.stop_signal, timeout=self.timeout)
                    self.finished_flags[idx] = True
        else:
            for sample_idx in sample_indices:
                idx = sample_idx.item() if torch.is_tensor(sample_idx) else sample_idx
                if idx < self.batch_size and not self.finished_flags[idx]:
                    self.audio_queues[idx].put(self.stop_signal, timeout=self.timeout)
                    self.finished_flags[idx] = True

    def __iter__(self):
        """Returns an iterator over the batch of audio streams."""
        return AudioBatchIterator(self)

    def get_stream(self, sample_idx: int):
        """Get the audio stream for a specific sample."""
        if sample_idx >= self.batch_size:
            raise ValueError(f"Sample index {sample_idx} exceeds batch size {self.batch_size}")
        return AudioSampleIterator(self, sample_idx)


class AudioSampleIterator:
    """Iterator for a single audio stream from the batch."""

    def __init__(self, streamer: AudioStreamer, sample_idx: int):
        self.streamer = streamer
        self.sample_idx = sample_idx

    def __iter__(self):
        return self

    def __next__(self):
        value = self.streamer.audio_queues[self.sample_idx].get(timeout=self.streamer.timeout)
        if value == self.streamer.stop_signal:
            raise StopIteration()
        return value


class AudioBatchIterator:
    """Iterator that yields audio chunks for all samples in the batch."""

    def __init__(self, streamer: AudioStreamer):
        self.streamer = streamer
        self.active_samples = set(range(streamer.batch_size))

    def __iter__(self):
        return self

    def __next__(self):
        if not self.active_samples:
            raise StopIteration()

        batch_chunks = {}
        samples_to_remove = set()
        for idx in self.active_samples:
            try:
                value = self.streamer.audio_queues[idx].get(block=False)
                if value == self.streamer.stop_signal:
                    samples_to_remove.add(idx)
                else:
                    batch_chunks[idx] = value
            except queue.Empty:
                pass

        self.active_samples -= samples_to_remove

        if batch_chunks:
            return batch_chunks
        elif self.active_samples:
            # If no chunks were ready but we still have active samples, wait a bit and try again
            time.sleep(0.01)
            return self.__next__()
        else:
            raise StopIteration()


# TODO (ebezzam) can we avoid overwritting this method?
def _update_model_kwargs_for_generation(
    outputs: ModelOutput,
    model_kwargs: Dict[str, Any],
    num_new_tokens: int = 1,
) -> Dict[str, Any]:
    """
    Update model_kwargs after adding new tokens.

    Mainly for the case num_new_tokens > 1 (e.g. a whole text window):
    - past_key_values: take from current outputs
    - attention_mask: append num_new_tokens ones
    - cache_position: advance by creating a range for all new positions
    """

    # update past_key_values keeping its naming used in model code
    model_kwargs["past_key_values"] = getattr(outputs, "past_key_values")

    attention_mask = model_kwargs["attention_mask"]
    model_kwargs["attention_mask"] = torch.cat(
        [attention_mask, attention_mask.new_ones((attention_mask.shape[0], num_new_tokens))], dim=-1
    )

    model_kwargs["cache_position"] = torch.arange(model_kwargs["cache_position"][-1] + 1, model_kwargs["cache_position"][-1] + num_new_tokens + 1).to(model_kwargs["cache_position"].device)
    
    return model_kwargs



class VibeVoiceRealTimeGenerationMixin(GenerationMixin):
    def _get_stopping_criteria(
        self,
        *args,
        **kwargs,
    ) -> StoppingCriteriaList:
        criteria = super()._get_stopping_criteria(*args, **kwargs)

        kept_criteria = StoppingCriteriaList()
        for criterion in criteria:
            if isinstance(criterion, MaxLengthCriteria):
                kept_criteria.append(criterion)
            else:
                logger.warning(
                    f"VibeVoiceRealTime does not support {criterion.__class__.__name__} stopping criteria. "
                    f"Generation uses TTS EOS classifier and max_length only."
                )
        return kept_criteria

    def _prepare_generation_config(
        self, generation_config: Optional[GenerationConfig], use_model_defaults: Optional[bool] = None, **kwargs
    ):
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

        """
        noise_scheduler = kwargs.pop("noise_scheduler", None)
        if "max_new_tokens" in kwargs and kwargs["max_new_tokens"] is None:
            # pop to default to generation_config max_length behavior instead of internal default of 20
            kwargs.pop("max_new_tokens")
        cfg_scale = kwargs.pop("cfg_scale", None)
        n_diffusion_steps = kwargs.pop("n_diffusion_steps", None)
        monitor_progress = kwargs.pop("monitor_progress", None)

        # TODO (ebezzam) remove verbose
        verbose = kwargs.pop("verbose", None)
        history_prompt = kwargs.pop("history_prompt", None)

        # Call the base class method to load from default generation_config.json
        generation_config, model_kwargs = super()._prepare_generation_config(
            generation_config, use_model_defaults, **kwargs
        )

        # try creating VibeVoice noise scheduler
        # TODO (ebezzam) ok with this? so user doesn't need to defined noise scheduler each time?
        # Alternatively, require user to create noise scheduler outside
        if (
            noise_scheduler is None
            and hasattr(generation_config, "noise_scheduler_class")
            and generation_config.noise_scheduler_class
        ):
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
        if noise_scheduler is not None:
            if not (
                hasattr(noise_scheduler, "set_timesteps")
                and hasattr(noise_scheduler, "step")
                and hasattr(noise_scheduler, "timesteps")
            ):
                raise ValueError(
                    "The provided noise scheduler is not compatible with VibeVoice generation. "
                    "It must implement `set_timesteps` and `step` methods, and have a `timesteps` attribute."
                )
            generation_config.noise_scheduler = noise_scheduler
        if not hasattr(generation_config, "noise_scheduler"):
            raise ValueError(
                "A noise scheduler must be provided for VibeVoice generation, either through the `noise_scheduler` "
                "argument or by defining `noise_scheduler_class` and `noise_scheduler_config` in the generation config."
            )
        if cfg_scale is not None:
            generation_config.cfg_scale = cfg_scale
        if not hasattr(generation_config, "cfg_scale"):
            raise ValueError("cfg_scale must be provided for VibeVoice generation.")
        if n_diffusion_steps is not None:
            generation_config.n_diffusion_steps = n_diffusion_steps
        if not hasattr(generation_config, "n_diffusion_steps"):
            raise ValueError("n_diffusion_steps must be provided for VibeVoice generation.")
        generation_config.monitor_progress = monitor_progress

        if verbose is not None:
            generation_config.verbose = verbose
        if history_prompt is not None:
            generation_config.history_prompt = history_prompt

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
        # TODO (ebezzam) don't use model output here?
        return BaseModelOutputWithPast(
            last_hidden_state=entry["last_hidden_state"].to(device),
            past_key_values=cache,
        )

    def _sample(
        self,
        input_ids: torch.LongTensor,    # used in different way... for first language model
        tts_input_ids: torch.LongTensor,  # normally called input_ids, called "tts_text_ids" in original
        stopping_criteria: StoppingCriteriaList,
        generation_config: GenerationConfig,
        synced_gpus: bool = False,
        streamer: Optional[Union[AudioStreamer]] = None,
        tts_lm_input_ids: Optional[torch.LongTensor] = None,
        tts_lm_attention_mask: Optional[torch.LongTensor] = None,
        **model_kwargs,
    ):
        """
        This method overrides [~generation.utils.GenerationMixin._sample].
        To ease maintenance, modifications are marked with the comment "VibeVoice specific".

        VibeVoiceRealTime differs from standard VibeVoice generation:
        1. **Windowed text processing**: Text is fed in small windows (TTS_TEXT_WINDOW_SIZE) enabling streaming input
        2. **Interleaved generation**: After each text window, generate multiple speech tokens (TTS_SPEECH_WINDOW_SIZE)
        3. **No semantic tokenizer**: Only uses acoustic tokenizer (no semantic feedback loop)
        4. **TTS LM separate from main LM**: TTS LM tracks interleaved text+speech sequence for conditioning
        5. **TTS EOS classifier**: Uses binary classifier instead of standard EOS tokens to detect completion
        6. **Separate negative TTS LM**: For classifier-free guidance in speech generation
        
        The windowed approach enables true streaming: you don't need the full text upfront, and audio is generated
        incrementally as text arrives.

        VibeVoice supports stopping criteria through internal finished_tags logic combined with
        the standard stopping criteria framework.
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
        noise_scheduler = generation_config.noise_scheduler
        monitor_progress = getattr(generation_config, "monitor_progress", None)
        cfg_scale = generation_config.cfg_scale
        n_diffusion_steps = generation_config.n_diffusion_steps
        diffusion_head_device = next(self.diffusion_head.parameters()).device

        # Extract prefilled outputs from history_prompt
        history_prompt = getattr(generation_config, "history_prompt", None)
        if history_prompt is None:
            raise ValueError("history_prompt must be provided in generation_config for VibeVoiceRealTime")
        outputs = self._rebuild_history_prompt(history_prompt[0]["lm"], device=input_ids.device)
        tts_lm_outputs = self._rebuild_history_prompt(history_prompt[0]["tts_lm"], device=input_ids.device)
        tts_lm_negative_outputs = self._rebuild_history_prompt(history_prompt[0]["neg_tts_lm"], device=input_ids.device)
        
        tokenizer = model_kwargs.pop("tokenizer", None)
        if tokenizer is None:
            raise ValueError("tokenizer must be provided in model_kwargs for VibeVoiceRealTime")
        neg_text_input_id = tokenizer.convert_tokens_to_ids("<|image_pad|>")

        verbose = getattr(generation_config, "verbose", False)

        # State tracking
        acoustic_cache = None
        finished_tags = torch.zeros(batch_size, dtype=torch.bool, device=input_ids.device)
        tts_text_window_index = 0
        
        # Constants for windowed processing
        TTS_TEXT_WINDOW_SIZE = getattr(self.config, "tts_text_window_size", 5)
        TTS_SPEECH_WINDOW_SIZE = getattr(self.config, "tts_speech_window_size", 6)
        
        # Output audio
        audio_chunks = [[] for _ in range(batch_size)]
        
        # Track token counts
        total_generated_speech_tokens = 0
        total_prefilled_text_tokens = 0

        # Prepare TTS inputs and that of negative for CFG
        tts_lm_kwargs = {
            # "input_ids": tts_lm_input_ids,
            "attention_mask": tts_lm_attention_mask,
            "max_new_tokens": generation_config.max_new_tokens,
            # "use_cache": self.config.use_cache,
        }
        _, tts_lm_model_kwargs = self._prepare_generation_config(generation_config, True, **tts_lm_kwargs)
        tts_lm_model_kwargs['cache_position'] = torch.arange(tts_lm_input_ids.shape[-1], device=tts_lm_input_ids.device, dtype=torch.long)
        tts_lm_model_kwargs["use_cache"] = self.config.use_cache
        tts_lm_negative_input_ids = torch.full((batch_size, 1), neg_text_input_id, dtype=torch.long, device=input_ids.device)
        tts_lm_negative_kwargs = {
            # "input_ids": torch.full((batch_size, 1), neg_text_input_id, dtype=torch.long, device=input_ids.device),
            "attention_mask": torch.ones((batch_size, 1), dtype=torch.long, device=input_ids.device),
            "max_new_tokens": generation_config.max_new_tokens,
            # "use_cache": self.config.use_cache,
        }
        _, tts_lm_negative_model_kwargs = self._prepare_generation_config(generation_config, True, **tts_lm_negative_kwargs)
        tts_lm_negative_model_kwargs['cache_position'] = torch.arange(tts_lm_negative_input_ids.shape[-1], device=tts_lm_negative_input_ids.device, dtype=torch.long)
        tts_lm_negative_model_kwargs["use_cache"] = self.config.use_cache
        
        # Update kwargs from prefilled outputs
        first_text_window_size = TTS_TEXT_WINDOW_SIZE if tts_input_ids.shape[1] >= TTS_TEXT_WINDOW_SIZE else tts_input_ids.shape[1]

        # Initialize kwargs from prefilled outputs
        # Use standalone function to pre-allocate for first text window
        model_kwargs = _update_model_kwargs_for_generation(outputs, model_kwargs, num_new_tokens=first_text_window_size)
        tts_lm_model_kwargs = _update_model_kwargs_for_generation(tts_lm_outputs, tts_lm_model_kwargs, num_new_tokens=first_text_window_size)
        # Negative uses base class method (only 1 token processed)
        tts_lm_negative_model_kwargs = self._update_model_kwargs_for_generation(tts_lm_negative_outputs, tts_lm_negative_model_kwargs, is_encoder_decoder=False)

        # Calculate generation limits for progress tracking
        initial_length = input_ids.shape[-1]
        step = tts_lm_input_ids.shape[1]
        max_steps = generation_config.max_length - initial_length
        completion_steps = torch.zeros(batch_size, dtype=torch.long, device=input_ids.device)
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
                progress_tensor = torch.stack((current_steps, torch.full_like(current_steps, max_steps), completion_steps), dim=1)
                monitor_progress(progress_tensor)
                
            if finished_tags.all():
                break

            # Get current text window
            cur_input_tts_text_ids = tts_input_ids[:, tts_text_window_index*TTS_TEXT_WINDOW_SIZE:(tts_text_window_index+1)*TTS_TEXT_WINDOW_SIZE]
            next_text_window_size = tts_input_ids[:, (tts_text_window_index+1)*TTS_TEXT_WINDOW_SIZE:(tts_text_window_index+2)*TTS_TEXT_WINDOW_SIZE].shape[1]
            tts_text_window_index += 1

            # Process text window if available
            if cur_input_tts_text_ids.shape[1] > 0:
                input_ids = torch.cat([input_ids, cur_input_tts_text_ids], dim=-1)
                tts_lm_input_ids = torch.cat([tts_lm_input_ids, cur_input_tts_text_ids], dim=-1)

                if tts_lm_input_ids.shape[1] > generation_config.max_length:
                    if verbose:
                        logger.info(f"Reached maximum generation length {generation_config.max_length}")
                    finished_tags[:] = True
                    completion_steps[:] = step + 1
                    break
                
                step += cur_input_tts_text_ids.shape[1]
                total_prefilled_text_tokens += cur_input_tts_text_ids.shape[1]

                # Forward pass through LM
                model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
                outputs = self.forward_lm(
                    **model_inputs, return_dict=True, output_attentions=False, output_hidden_states=False,
                )
                # Pre-allocate attention mask for next text window
                model_kwargs = _update_model_kwargs_for_generation(outputs, model_kwargs, num_new_tokens=next_text_window_size)

                # Forward pass through TTS LM with text
                tts_lm_model_inputs = self.prepare_inputs_for_generation(tts_lm_input_ids, **tts_lm_model_kwargs)
                # Need to pass the same number of input_ids as lm_last_hidden_state tokens
                window_size = outputs.last_hidden_state.shape[1]
                tts_lm_model_inputs["input_ids"] = tts_lm_input_ids[:, -window_size:]
                tts_lm_additional_inputs = {
                    "tts_text_masks": torch.ones_like(tts_lm_input_ids[:, -1:]),
                    "lm_last_hidden_state": outputs.last_hidden_state,
                }
                tts_lm_outputs = self.forward_tts_lm(
                    **tts_lm_model_inputs, **tts_lm_additional_inputs, return_dict=True, output_attentions=False, output_hidden_states=False,
                )
                # Update for the tokens just processed (base class method)
                tts_lm_model_kwargs = self._update_model_kwargs_for_generation(
                    tts_lm_outputs, tts_lm_model_kwargs, is_encoder_decoder=False,
                )

            # Generate speech tokens
            diffusion_indices = torch.arange(batch_size, device=input_ids.device)[~finished_tags]
            
            if diffusion_indices.numel() > 0:
                for cur_speech_index in range(TTS_SPEECH_WINDOW_SIZE):
                    # Diffusion process with classifier-free guidance
                    positive_condition = tts_lm_outputs.last_hidden_state[diffusion_indices, -1, :]
                    negative_condition = tts_lm_negative_outputs.last_hidden_state[diffusion_indices, -1, :]

                    noise_scheduler.set_timesteps(num_inference_steps=n_diffusion_steps)
                    condition = torch.cat([positive_condition, negative_condition], dim=0).to(diffusion_head_device)
                    speech = torch.randn(condition.shape[0], self.config.acoustic_hidden_size).to(condition)

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
                    scaled_latent = speech_latent / self.latent_scaling_factor.to(
                        speech_latent.device
                    ) - self.latent_bias_factor.to(speech_latent.device)
                    
                    if len(diffusion_indices) != batch_size:
                        padded_latent = torch.zeros(batch_size, scaled_latent.shape[1], scaled_latent.shape[2]).to(scaled_latent.device)
                        padded_latent[diffusion_indices] = scaled_latent
                    else:
                        padded_latent = scaled_latent
                    audio_output = self.acoustic_tokenizer.decode(
                        padded_latent.to(self.acoustic_tokenizer.device),
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

                    # Get acoustic embedding for next step
                    acoustic_embed = self.acoustic_connector(speech_latent)
                    
                    # Append speech token to TTS LM sequence
                    tts_lm_input_ids = torch.cat([tts_lm_input_ids, torch.ones_like(tts_lm_input_ids[:, -1:])], dim=-1)
                    
                    if tts_lm_input_ids.shape[1] > generation_config.max_length:
                        break
                    
                    step += 1
                    total_generated_speech_tokens += 1

                    # Forward pass through TTS LM with speech
                    tts_lm_model_inputs = self.prepare_inputs_for_generation(tts_lm_input_ids, **tts_lm_model_kwargs)
                    tts_lm_additional_inputs = {
                        "tts_text_masks": torch.zeros_like(tts_lm_input_ids[:, -1:]),
                        "lm_last_hidden_state": acoustic_embed,
                    }
                    tts_lm_outputs = self.forward_tts_lm(
                        **tts_lm_model_inputs, **tts_lm_additional_inputs, return_dict=True, output_attentions=False, output_hidden_states=False)
                    
                    # Update model kwargs: on last speech token + more text coming, pre-allocate for next window
                    if cur_speech_index == TTS_SPEECH_WINDOW_SIZE - 1 and next_text_window_size > 0:
                        tts_lm_model_kwargs = _update_model_kwargs_for_generation(
                            tts_lm_outputs, tts_lm_model_kwargs, num_new_tokens=next_text_window_size
                        )
                    else:
                        tts_lm_model_kwargs = self._update_model_kwargs_for_generation(
                            tts_lm_outputs, tts_lm_model_kwargs, is_encoder_decoder=False,
                        )

                    # Update negative TTS LM
                    tts_lm_negative_input_ids = torch.cat([tts_lm_negative_input_ids, torch.ones_like(tts_lm_input_ids[:, -1:])], dim=-1)
                    tts_lm_negative_model_inputs = self.prepare_inputs_for_generation(tts_lm_negative_input_ids, **tts_lm_negative_model_kwargs)
                    tts_lm_negative_additional_inputs = {
                        "tts_text_masks": torch.zeros_like(tts_lm_negative_input_ids[:, -1:]),
                        "lm_last_hidden_state": acoustic_embed,
                    }
                    tts_lm_negative_outputs = self.forward_tts_lm(
                        **tts_lm_negative_model_inputs, **tts_lm_negative_additional_inputs, return_dict=True, output_attentions=False, output_hidden_states=False
                    )
                    tts_lm_negative_model_kwargs = self._update_model_kwargs_for_generation(
                        tts_lm_negative_outputs, tts_lm_negative_model_kwargs, is_encoder_decoder=False,
                    )

                    # Check TTS EOS
                    # TODO (ebezzam) move sigmoid to classifer?
                    tts_eos_logits = torch.sigmoid(self.tts_eos_classifier(tts_lm_outputs.last_hidden_state[diffusion_indices, -1, :]))
                    eos_mask = tts_eos_logits[:, 0] > 0.5
                    if eos_mask.any():
                        eos_indices = diffusion_indices[eos_mask]
                        finished_tags[eos_indices] = True
                        completion_steps[eos_indices] = step + 1
                        if streamer is not None:
                            streamer.end(eos_indices)
                        if verbose:
                            logger.info(f"Samples {eos_indices.tolist()} reached TTS EOS at step {step + 1}")

            # Check max length
            if tts_lm_input_ids.shape[1] >= generation_config.max_length:
                if verbose:
                    logger.info(f"Reached maximum generation length {generation_config.max_length}")
                unfinished_indices = ~finished_tags
                if unfinished_indices.any():
                    finished_tags[unfinished_indices] = True
                    completion_steps[unfinished_indices] = step + 1
                break

            unfinished_sequences = unfinished_sequences & ~finished_tags.long()
            this_peer_finished = unfinished_sequences.max() == 0
            cur_len += 1
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
                sequences=tts_lm_input_ids,
                scores=scores,
                logits=raw_logits,
                attentions=decoder_attentions,
                hidden_states=decoder_hidden_states,
                past_key_values=model_kwargs.get("past_key_values"),
                audio=final_audio_outputs,
                reach_max_step_sample=tts_lm_input_ids.shape[1] >= generation_config.max_length,
            )
        else:
            # NOTE (ebezzam): new tokens in input_ids are simply speech tokens (mainly `speech_diffusion_id`)
            # so returning `input_ids` is insufficient for generated audio
            return final_audio_outputs
        # ============================================
