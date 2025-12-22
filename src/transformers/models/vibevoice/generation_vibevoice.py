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

import torch

from ...generation import (
    BaseStreamer,
    GenerateDecoderOnlyOutput,
    GenerationConfig,
    GenerationMixin,
    LogitsProcessor,
    LogitsProcessorList,
)
from ...generation.stopping_criteria import EosTokenCriteria, MaxLengthCriteria, StoppingCriteriaList
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


class VibeVoiceGenerationMixin(GenerationMixin):
    def _get_stopping_criteria(
        self,
        *args,
        **kwargs,
    ) -> StoppingCriteriaList:
        criteria = super()._get_stopping_criteria(*args, **kwargs)

        kept_criteria = StoppingCriteriaList()
        for criterion in criteria:
            if not isinstance(criterion, MaxLengthCriteria):
                if isinstance(criterion, EosTokenCriteria):
                    logger.debug(
                        f"VibeVoice handles EOS tokens internally, ignoring {criterion.__class__.__name__} stopping criteria."
                    )
                else:
                    logger.warning(
                        f"VibeVoice does not support {criterion.__class__.__name__} stopping criteria, it will be ignored."
                    )
            else:
                kept_criteria.append(criterion)
        return kept_criteria

    def _prepare_generation_config(self, generation_config: Optional[GenerationConfig], **kwargs):
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
        input_features = kwargs.pop("input_features", None)
        input_features_mask = kwargs.pop("input_features_mask", None)

        # Call the base class method to load from default generation_config.json
        generation_config, model_kwargs = super()._prepare_generation_config(generation_config, **kwargs)

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
        if input_features is not None:
            model_kwargs["input_features"] = input_features
        if input_features_mask is not None:
            model_kwargs["input_features_mask"] = input_features_mask

        return generation_config, model_kwargs

    def _sample(
        self,
        input_ids: torch.LongTensor,
        logits_processor: LogitsProcessorList,
        stopping_criteria: StoppingCriteriaList,
        generation_config: GenerationConfig,
        synced_gpus: bool = False,
        streamer: Optional[Union[AudioStreamer]] = None,
        **model_kwargs,
    ):
        """
        This method overrides [~generation.utils.GenerationMixin._sample].
        To ease maintenance, modifications are marked with the comment "VibeVoice specific".

        Indeed, VibeVoice model requires a custom generation sampling step:
        1. Extract VibeVoice-specific parameters and setup diffusion components
        2. Setup negative generation for classifier-free guidance
        3. Generate tokens with diffusion-based speech synthesis for speech tokens
        4. Apply VibeVoice-specific stopping conditions alongside standard criteria

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
        input_features = model_kwargs.pop("input_features", None)
        input_features_mask = model_kwargs.pop("input_features_mask", None)
        noise_scheduler = generation_config.noise_scheduler
        monitor_progress = getattr(generation_config, "monitor_progress", None)
        cfg_scale = generation_config.cfg_scale
        n_diffusion_steps = generation_config.n_diffusion_steps
        diffusion_head_device = next(self.diffusion_head.parameters()).device

        # State tracking
        acoustic_cache = None
        semantic_cache = None
        finished_tags = torch.zeros(batch_size, dtype=torch.bool, device=input_ids.device)
        correct_cnt = torch.zeros(batch_size, dtype=torch.long, device=input_ids.device)
        is_prefill = True
        inputs_embeds = None

        # Output audio
        audio_chunks = [[] for _ in range(batch_size)]

        # Token constraints for VibeVoice - only allow speech tokens
        valid_tokens = [
            self.config.speech_start_id,
            self.config.speech_end_id,
            self.config.speech_diffusion_id,
            self.config.eos_token_id,
        ]
        if hasattr(self.config, "bos_token_id") and self.config.bos_token_id is not None:
            valid_tokens.append(self.config.bos_token_id)
        token_constraint_processor = VibeVoiceTokenConstraintProcessor(valid_tokens, device=input_ids.device)
        logits_processor.append(token_constraint_processor)

        # Setup negative generation for classifier-free guidance
        negative_kwargs = {
            "input_ids": torch.full(
                (batch_size, 1), self.config.speech_start_id, dtype=torch.long, device=input_ids.device
            ),
            "attention_mask": torch.ones((batch_size, 1), dtype=torch.long, device=input_ids.device),
            "max_new_tokens": generation_config.max_new_tokens,
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
            # ============================================

            # prepare model inputs
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

            # *************** VibeVoice specific ***************
            # Handle prefill vs normal generation
            if is_prefill:
                # First step: process speech inputs for conditioning
                if input_features is not None and input_features_mask is not None:
                    model_inputs.update(
                        {
                            "input_features": input_features.to(device=input_ids.device),
                            "input_features_mask": input_features_mask.to(input_ids.device),
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

            # Force finished samples to generate EOS tokens
            next_tokens[finished_tags] = self.config.eos_token_id
            # ============================================

            # finished sentences should have their next token be a padding token
            if has_eos_stopping_criteria:
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

            # update generated ids, model inputs, and length for next step
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)

            # *************** VibeVoice specific ***************
            # Handle EOS detection and completion tracking
            eos_mask = next_tokens == self.config.eos_token_id
            if eos_mask.any():
                new_eos_indices = eos_mask & ~finished_tags
                if new_eos_indices.any():
                    finished_tags[new_eos_indices] = True
                    completion_steps[new_eos_indices] = step + 1
                    new_eos_idx_list = new_eos_indices.nonzero(as_tuple=False).squeeze(1)
                    if streamer is not None:
                        streamer.end(new_eos_idx_list)

                    if monitor_progress is not None:
                        current_steps = torch.full((batch_size,), step + 1, dtype=torch.long, device=input_ids.device)
                        current_steps[finished_tags] = max_step_per_sample[finished_tags]
                        progress_tensor = torch.stack((current_steps, max_step_per_sample, completion_steps), dim=1)
                        monitor_progress(progress_tensor)

            # Handle max length termination
            max_length_reached = step >= max_step_per_sample
            new_max_length_mask = max_length_reached & ~finished_tags
            if new_max_length_mask.any():
                finished_tags[new_max_length_mask] = True
                completion_steps[new_max_length_mask] = step + 1
                new_max_length_indices = new_max_length_mask.nonzero(as_tuple=False).squeeze(1)
                if streamer is not None:
                    streamer.end(new_max_length_indices)

            # Handle speech start tokens
            diffusion_start_mask = ~finished_tags & (next_tokens == self.config.speech_start_id)
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
                    negative_input_ids[sample_idx, -1] = self.config.speech_start_id

            # Prepare embeddings for next iteration
            next_inputs_embeds = self.get_input_embeddings()(next_tokens).unsqueeze(1)

            # Handle diffusion tokens
            diffusion_mask = ~finished_tags & (next_tokens == self.config.speech_diffusion_id)
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

                # Correct non-diffusion indices in negative generation
                non_diffusion_mask = ~finished_tags & (next_tokens != self.config.speech_diffusion_id)
                if non_diffusion_mask.any():
                    non_diffusion_indices = non_diffusion_mask.nonzero(as_tuple=False).squeeze(1)
                    start_indices = correct_cnt[non_diffusion_indices]

                    seq_len = negative_model_kwargs["attention_mask"].shape[1]
                    for sample_idx, start_idx in zip(non_diffusion_indices.tolist(), start_indices.tolist()):
                        if start_idx + 1 < seq_len - 1:
                            negative_model_kwargs["attention_mask"][sample_idx, start_idx + 1 :] = (
                                negative_model_kwargs["attention_mask"][sample_idx, start_idx:-1].clone()
                            )
                        negative_model_kwargs["attention_mask"][sample_idx, start_idx] = 0

                    if (
                        "past_key_values" in negative_model_kwargs
                        and negative_model_kwargs["past_key_values"] is not None
                    ):
                        for layer_idx in range(len(negative_model_kwargs["past_key_values"])):
                            k_cache = negative_model_kwargs["past_key_values"].layers[layer_idx].keys
                            v_cache = negative_model_kwargs["past_key_values"].layers[layer_idx].values
                            for sample_idx, start_idx in zip(non_diffusion_indices.tolist(), start_indices.tolist()):
                                if start_idx + 1 < k_cache.shape[2] - 1:
                                    k_cache[sample_idx, :, start_idx + 1 :, :] = k_cache[
                                        sample_idx, :, start_idx:-1, :
                                    ].clone()
                                    v_cache[sample_idx, :, start_idx + 1 :, :] = v_cache[
                                        sample_idx, :, start_idx:-1, :
                                    ].clone()

                    for sample_idx, start_idx in zip(non_diffusion_indices.tolist(), start_indices.tolist()):
                        if start_idx + 1 < negative_input_ids.shape[1] - 1:
                            negative_input_ids[sample_idx, start_idx + 1 :] = negative_input_ids[
                                sample_idx, start_idx:-1
                            ].clone()

                    correct_cnt[non_diffusion_indices] += 1

                # Diffusion process with classifier-free guidance
                positive_condition = outputs.last_hidden_state[diffusion_indices, -1, :]
                negative_condition = negative_outputs.last_hidden_state[diffusion_indices, -1, :]

                noise_scheduler.set_timesteps(num_inference_steps=n_diffusion_steps)
                condition = torch.cat([positive_condition, negative_condition], dim=0).to(diffusion_head_device)
                speech = torch.randn(condition.shape[0], self.config.acoustic_hidden_size).to(condition)

                # # TODO (ebezzam) something like below to use `do_sample`? only problem is original would differ and would break integration tests
                # if do_sample:
                #     # Stochastic generation: use random noise
                #     speech = torch.randn(condition.shape[0], self.config.acoustic_hidden_size).to(condition)
                # else:
                #     # Deterministic generation: use reproducible noise based on conditioning content
                #     # This ensures same input gives same output, but different inputs get different noise patterns
                #     seed = int(torch.sum(condition).item() * 1000) % (2**31)  # Simple hash of conditioning
                #     generator = torch.Generator(device=condition.device).manual_seed(seed)
                #     speech = torch.randn(
                #         condition.shape[0], self.config.acoustic_hidden_size,
                #         generator=generator, device=condition.device
                #     )

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

                # Get semantic features for next step
                semantic_outputs = self.semantic_tokenizer.encode(
                    audio_chunk,
                    padding_cache=semantic_cache,
                    use_cache=self.config.use_cache,
                )
                semantic_features = semantic_outputs.latents[diffusion_indices]
                semantic_cache = semantic_outputs.padding_cache

                # Combine features for next input
                acoustic_embed = self.acoustic_connector(speech_latent)
                semantic_embed = self.semantic_connector(semantic_features)
                diffusion_embeds = acoustic_embed + semantic_embed
                next_inputs_embeds[diffusion_indices] = diffusion_embeds

            inputs_embeds = next_inputs_embeds
            unfinished_sequences = unfinished_sequences & ~finished_tags.long()
            step += 1
            # ============================================

            if streamer is not None:
                streamer.put(next_tokens.cpu())

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
                reach_max_step_sample=completion_steps >= max_step_per_sample,
            )
        else:
            # NOTE (ebezzam): new tokens in input_ids are simply speech tokens (mainly `speech_diffusion_id`)
            # so returning `input_ids` is insufficient for generated audio
            return final_audio_outputs
        # ============================================
