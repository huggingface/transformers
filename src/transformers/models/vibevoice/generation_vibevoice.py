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

from collections.abc import Callable
from dataclasses import dataclass
from queue import Queue
from typing import Optional, Union

import torch
import torch.nn as nn

from ...generation import (
    BaseStreamer,
    GenerateDecoderOnlyOutput,
    GenerationConfig,
    GenerationMixin,
    LogitsProcessor,
    LogitsProcessorList,
)
from ...utils import logging


logger = logging.get_logger(__name__)


@dataclass
class VibeVoiceGenerateOutput(GenerateDecoderOnlyOutput):
    """
    Outputs of VibeVoiceForConditionalGeneration.generate.

    Args:
        sequences (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            The generated sequences.
        speech_outputs (`List[torch.FloatTensor]`, *optional*):
            List of generated speech waveforms for each speech segment.
        reach_max_step_sample (`torch.BoolTensor`, *optional*):
            Boolean tensor indicating which samples reached maximum generation steps.
    """
    # TODO (ebezzam) necessary to output sequences?
    sequences: Optional[torch.LongTensor] = None
    speech_outputs: Optional[list[torch.FloatTensor]] = None
    reach_max_step_sample: Optional[torch.BoolTensor] = None


class VibeVoiceTokenConstraintProcessor(LogitsProcessor):
    """Constrains token generation to only valid tokens during speech generation."""

    def __init__(self, valid_token_ids: list[int], device: torch.device = None):
        self.valid_token_ids = torch.tensor(valid_token_ids, dtype=torch.long, device=device)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # Create a mask for valid tokens
        mask = torch.full_like(scores, float('-inf'))
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

        # Create a queue for each sample in the batch
        self.audio_queues = [Queue() for _ in range(batch_size)]
        self.finished_flags = [False for _ in range(batch_size)]
        self.sample_indices_map = {}  # Maps from sample index to queue index

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
            # End all samples
            for idx in range(self.batch_size):
                if not self.finished_flags[idx]:
                    self.audio_queues[idx].put(self.stop_signal, timeout=self.timeout)
                    self.finished_flags[idx] = True
        else:
            # End specific samples
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

        # Try to get chunks from all active samples
        for idx in self.active_samples:
            try:
                value = self.streamer.audio_queues[idx].get(block=False)
                if value == self.streamer.stop_signal:
                    samples_to_remove.add(idx)
                else:
                    batch_chunks[idx] = value
            except:
                # Queue is empty for this sample, skip it this iteration
                pass

        # Remove finished samples
        self.active_samples -= samples_to_remove

        if batch_chunks:
            return batch_chunks
        elif self.active_samples:
            # If no chunks were ready but we still have active samples,
            # wait a bit and try again
            import time
            time.sleep(0.01)
            return self.__next__()
        else:
            raise StopIteration()


class VibeVoiceGenerationMixin(GenerationMixin):

    def _prepare_generation_config(
        self, generation_config: Optional[GenerationConfig], use_model_defaults: Optional[bool] = None, **kwargs
    ):
        """
        Override to filter out VibeVoice-specific parameters before passing to the base class.
        This follows the same pattern as CSM's approach with depth_decoder_* parameters.
        """
        # Extract VibeVoice-specific parameters and remove them from kwargs
        vibevoice_kwargs = {
            k[len("vibevoice_"):]: v for k, v in kwargs.items() if k.startswith("vibevoice_")
        }
        
        # Remove the VibeVoice-specific keys from the original kwargs
        kwargs = {k: v for k, v in kwargs.items() if not k.startswith("vibevoice_")}
        
        # Ensure generation_config is a proper GenerationConfig object
        if generation_config is None:
            generation_config = GenerationConfig()
        elif isinstance(generation_config, dict):
            generation_config = GenerationConfig(**generation_config)
        
        # Call the base class method with filtered kwargs
        generation_config, model_kwargs = super()._prepare_generation_config(generation_config, use_model_defaults, **kwargs)
        
        # Store generation-specific parameters as instance attributes
        for param in ['cfg_scale', 'n_diffusion_steps', 'noise_scheduler', 'monitor_progress']:
            if param in vibevoice_kwargs:
                setattr(self, f'_vibevoice_{param}', vibevoice_kwargs[param])
        
        # Add actual model parameters to model_kwargs
        for param in ['input_features', 'input_features_mask']:
            if param in vibevoice_kwargs:
                model_kwargs[param] = vibevoice_kwargs[param]
        
        return generation_config, model_kwargs

    def _sample(
        self,
        input_ids: torch.LongTensor,
        logits_processor: LogitsProcessorList,
        stopping_criteria,
        generation_config: GenerationConfig,
        synced_gpus: bool = False,
        streamer: Optional[Union[AudioStreamer]] = None,
        **model_kwargs,
    ):
        """
        VibeVoice-specific sampling method that implements diffusion-based speech synthesis.
        This method overrides the base _sample method to add VibeVoice-specific functionality.
        """

        # Initialize cache position and other model_kwargs properly
        batch_size, cur_len = input_ids.shape[:2]
        device = input_ids.device
        model_kwargs = self._get_initial_cache_position(cur_len, input_ids.device, model_kwargs)
        
        # Get VibeVoice model parameters from model_kwargs and generation parameters from instance attributes
        input_features = model_kwargs.pop('input_features', None)
        input_features_mask = model_kwargs.pop('input_features_mask', None)
        cfg_scale = getattr(self, '_vibevoice_cfg_scale', 1.3)
        n_diffusion_steps = getattr(self, '_vibevoice_n_diffusion_steps', 10)
        noise_scheduler = getattr(self, '_vibevoice_noise_scheduler', None)
        monitor_progress = getattr(self, '_vibevoice_monitor_progress', None)

        if noise_scheduler is None:
            raise ValueError("`noise_scheduler` from `diffusers` must be provided for VibeVoice generation.")

        # VibeVoice-specific setup
        acoustic_cache = None
        semantic_cache = None
        finished_tags = torch.zeros(batch_size, dtype=torch.bool, device=device)
        correct_cnt = torch.zeros(batch_size, dtype=torch.long, device=device)
        is_prefill = True
        inputs_embeds = None
        
        # Initialize audio chunks storage for each sample
        audio_chunks = [[] for _ in range(batch_size)]
        
        # Define all valid tokens that can be generated
        valid_tokens = [
            self.config.speech_start_id,
            self.config.speech_end_id,
            self.config.speech_diffusion_id,
            self.config.eos_token_id
        ]
        # Add bos_token_id if it exists
        if hasattr(self.config, 'bos_token_id') and self.config.bos_token_id is not None:
            valid_tokens.append(self.config.bos_token_id)
            
        # Add custom processor to constrain token generation
        token_constraint_processor = VibeVoiceTokenConstraintProcessor(valid_tokens, device=device)
        logits_processor.append(token_constraint_processor)
        
        # === Negative generation config setup (for classifier-free guidance) ===
        negative_kwargs = {
            'input_ids': torch.full((batch_size, 1), self.config.speech_start_id, dtype=torch.long, device=device),
            'attention_mask': torch.ones((batch_size, 1), dtype=torch.long, device=device),
            'max_new_tokens': generation_config.max_new_tokens or 100
        }
        negative_generation_config = GenerationConfig()
        negative_generation_config, negative_model_kwargs = self._prepare_generation_config(
            negative_generation_config, True, **negative_kwargs
        )
        _, negative_model_input_name, negative_model_kwargs = self._prepare_model_inputs(
            None, negative_generation_config.bos_token_id, negative_model_kwargs
        )
        self._prepare_special_tokens(negative_generation_config, True, device=device)
        negative_generation_config.use_cache = self.config.use_cache
        negative_model_kwargs["use_cache"] = self.config.use_cache
        negative_input_ids = negative_kwargs['input_ids'].to(device)

        negative_input_ids_length = negative_input_ids.shape[1]
        negative_has_default_max_length = negative_kwargs.get("max_length") is None and negative_generation_config.max_length is not None
        negative_has_default_min_length = negative_kwargs.get("min_length") is None and negative_generation_config.min_length is not None
        negative_generation_config = self._prepare_generated_length(
            generation_config=negative_generation_config,
            has_default_max_length=negative_has_default_max_length,
            has_default_min_length=negative_has_default_min_length,
            model_input_name=negative_model_input_name,
            inputs_tensor=negative_kwargs['input_ids'],
            input_ids_length=negative_input_ids_length,
        )

        negative_max_cache_length = negative_generation_config.max_length - 1
        negative_batch_size = negative_kwargs['input_ids'].shape[0]
        self._prepare_cache_for_generation(negative_generation_config, negative_model_kwargs, None, negative_batch_size, negative_max_cache_length)
        negative_model_kwargs['cache_position'] = torch.arange(negative_input_ids_length, device=device, dtype=torch.long)
        for k, v in negative_model_kwargs.items():
            if isinstance(v, torch.Tensor):
                negative_model_kwargs[k] = v.to(device=device)
                
        # Calculate maximum generation steps
        initial_length = input_ids.shape[-1]
        initial_length_per_sample = model_kwargs['attention_mask'].sum(dim=-1)
        max_steps = generation_config.max_length - initial_length
        max_step_per_sample = torch.min(generation_config.max_length - initial_length_per_sample, torch.full_like(initial_length_per_sample, max_steps))
        completion_steps = torch.zeros(batch_size, dtype=torch.long, device=device)  # Track when each sample completed

        # VibeVoice generation loop
        for step in range(max_steps):
            # Check if audio_streamer has been ended (stopped externally)
            if streamer is not None and hasattr(streamer, 'finished_flags'):
                if any(streamer.finished_flags):
                    break

            if finished_tags.all():
                break

            # Report progress if monitor_progress callback is provided
            if monitor_progress is not None:
                current_steps = torch.full((batch_size,), step, dtype=torch.long, device=device)
                current_steps[finished_tags] = completion_steps[finished_tags]
                progress_tensor = torch.stack((current_steps, max_step_per_sample, completion_steps), dim=1)
                monitor_progress(progress_tensor)

            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
            
            # Handle prefill vs normal generation inputs
            if is_prefill:
                # First step: process speech inputs for conditioning
                model_inputs.update({
                    "input_features": input_features.to(device=device),
                    "input_features_mask": input_features_mask.to(device),
                })
                is_prefill = False
            else:
                # Subsequent steps: use embeddings from previous step
                model_inputs.pop('inputs_embeds', None)  # Remove any existing inputs_embeds
                model_inputs['inputs_embeds'] = inputs_embeds

            # Set logits_to_keep for positive pass
            model_inputs['logits_to_keep'] = 1

            # Forward pass through the model
            outputs = self(
                **model_inputs, return_dict=True
            )
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=False,
            )

            # Get logits and apply logits processor
            next_token_logits = outputs.logits[:, -1, :].to(copy=True, dtype=torch.float32, device=input_ids.device)
            next_token_scores = logits_processor(input_ids, next_token_logits)

            # token selection
            if generation_config.do_sample:
                probs = nn.functional.softmax(next_token_scores, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                next_tokens = torch.argmax(next_token_scores, dim=-1)

            next_tokens[finished_tags] = self.config.eos_token_id
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)

            # reached end of generation
            eos_mask = (next_tokens == self.config.eos_token_id)
            if eos_mask.any():
                new_eos_indices = eos_mask & ~finished_tags
                if new_eos_indices.any():
                    finished_tags[new_eos_indices] = True
                    completion_steps[new_eos_indices] = step + 1  # Record actual completion step
                    new_eos_idx_list = new_eos_indices.nonzero(as_tuple=False).squeeze(1)
                    if streamer is not None:
                        streamer.end(new_eos_idx_list)

                    # Report progress immediately after samples complete
                    if monitor_progress is not None:
                        current_steps = torch.full((batch_size,), step + 1, dtype=torch.long, device=device)
                        # For finished samples, set current = max so they're detected as completed
                        current_steps[finished_tags] = max_step_per_sample[finished_tags]
                        progress_tensor = torch.stack((current_steps, max_step_per_sample, completion_steps), dim=1)
                        monitor_progress(progress_tensor)

            # Check if any sample reached its maximum generation length
            max_length_reached = step >= max_step_per_sample
            new_max_length_mask = max_length_reached & ~finished_tags
            if new_max_length_mask.any():
                finished_tags[new_max_length_mask] = True
                completion_steps[new_max_length_mask] = step + 1  # Record actual completion step
                new_max_length_indices = new_max_length_mask.nonzero(as_tuple=False).squeeze(1)
                if streamer is not None:
                    streamer.end(new_max_length_indices)

            # speech_begin
            diffusion_start_mask = ~finished_tags & (next_tokens == self.config.speech_start_id)
            if diffusion_start_mask.any():
                diffusion_start_indices = diffusion_start_mask.nonzero(as_tuple=False).squeeze(1)
                # update attention mask
                for sample_idx in diffusion_start_indices.tolist():
                    negative_model_kwargs['attention_mask'][sample_idx, :] = 0
                    negative_model_kwargs['attention_mask'][sample_idx, -1] = 1
                # update past key values
                for layer_idx in range(len(negative_model_kwargs['past_key_values'])):
                    k_cache = negative_model_kwargs['past_key_values'].layers[layer_idx].keys
                    v_cache = negative_model_kwargs['past_key_values'].layers[layer_idx].values
                    # Process each non-diffusion sample
                    for sample_idx in diffusion_start_indices.tolist():
                        # Shift cache for this sample
                        k_cache[sample_idx, :, -1, :] = k_cache[sample_idx, :, 0, :].clone()
                        v_cache[sample_idx, :, -1, :] = v_cache[sample_idx, :, 0, :].clone()
                # update negative_input_ids
                for sample_idx in diffusion_start_indices.tolist():
                    negative_input_ids[sample_idx, -1] = self.config.speech_start_id

            # Prepare inputs_embeds for next iteration
            # Initialize with default embeddings for all tokens
            next_inputs_embeds = self.get_input_embeddings()(next_tokens).unsqueeze(1)  # [batch_size, 1, hidden_size]

            # forward diffusion
            # Diffusion indices are those that are not finished and not special tokens
            diffusion_mask = ~finished_tags & (next_tokens == self.config.speech_diffusion_id)
            if diffusion_mask.any():
                diffusion_indices = diffusion_mask.nonzero(as_tuple=False).squeeze(1)
                negative_model_inputs = self.prepare_inputs_for_generation(negative_input_ids, **negative_model_kwargs)
                # Forward negative pass through the model
                if negative_model_inputs['inputs_embeds'] is None and inputs_embeds is not None:
                    negative_model_inputs['inputs_embeds'] = inputs_embeds
                    negative_model_inputs['input_ids'] = None

                # Set logits_to_keep for negative pass
                negative_model_inputs['logits_to_keep'] = 0

                negative_outputs = self(
                    **negative_model_inputs, return_dict=True
                )
                negative_model_kwargs = self._update_model_kwargs_for_generation(
                    negative_outputs, negative_model_kwargs, is_encoder_decoder=False,
                )
                negative_input_ids = torch.cat([negative_input_ids, next_tokens[:, None]], dim=-1)

                # correct the non-diffusion indices
                non_diffusion_mask = ~finished_tags & (next_tokens != self.config.speech_diffusion_id)
                if non_diffusion_mask.any():
                    non_diffusion_indices = non_diffusion_mask.nonzero(as_tuple=False).squeeze(1)
                    start_indices = correct_cnt[non_diffusion_indices]

                    # 1. Update attention_mask - need to handle each sample separately
                    seq_len = negative_model_kwargs['attention_mask'].shape[1]
                    for sample_idx, start_idx in zip(non_diffusion_indices.tolist(), start_indices.tolist()):
                        # Shift the attention mask for this sample
                        if start_idx + 1 < seq_len - 1:
                            negative_model_kwargs['attention_mask'][sample_idx, start_idx+1:] = \
                                negative_model_kwargs['attention_mask'][sample_idx, start_idx:-1].clone()
                        negative_model_kwargs['attention_mask'][sample_idx, start_idx] = 0

                    # 2. Update past_key_values
                    for layer_idx in range(len(negative_model_kwargs['past_key_values'])):
                        k_cache = negative_model_kwargs['past_key_values'].layers[layer_idx].keys
                        v_cache = negative_model_kwargs['past_key_values'].layers[layer_idx].values
                        # Process each non-diffusion sample
                        for sample_idx, start_idx in zip(non_diffusion_indices.tolist(), start_indices.tolist()):
                            if start_idx + 1 < k_cache.shape[2] - 1:
                                # Shift cache for this sample
                                k_cache[sample_idx, :, start_idx+1:, :] = k_cache[sample_idx, :, start_idx:-1, :].clone()
                                v_cache[sample_idx, :, start_idx+1:, :] = v_cache[sample_idx, :, start_idx:-1, :].clone()

                    # 3. Update negative_input_ids
                    for sample_idx, start_idx in zip(non_diffusion_indices.tolist(), start_indices.tolist()):
                        if start_idx + 1 < negative_input_ids.shape[1] - 1:
                            negative_input_ids[sample_idx, start_idx+1:] = \
                                negative_input_ids[sample_idx, start_idx:-1].clone()

                    correct_cnt[non_diffusion_indices] += 1

                positive_condition = outputs.last_hidden_state[diffusion_indices, -1, :]
                negative_condition = negative_outputs.last_hidden_state[diffusion_indices, -1, :]

                # Diffusion process to sample speech tokens, with classifier-free guidance (CFG)
                noise_scheduler.set_timesteps(num_inference_steps=n_diffusion_steps)
                condition = torch.cat([positive_condition, negative_condition], dim=0).to(self.diffusion_head.device)
                speech = torch.randn(condition.shape[0], self.config.acoustic_hidden_size).to(condition)
                for t in noise_scheduler.timesteps:
                    half = speech[: len(speech) // 2]
                    combined = torch.cat([half, half], dim=0)
                    eps = self.diffusion_head(combined, t.repeat(combined.shape[0]).to(combined), condition=condition)
                    cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
                    half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
                    eps = torch.cat([half_eps, half_eps], dim=0)
                    speech = noise_scheduler.step(eps, t, speech).prev_sample
                speech_latent = speech[: len(speech) // 2].unsqueeze(1)

                # Decode acoustic latent to audio using acoustic streaming cache
                scaled_latent = speech_latent / self.speech_scaling_factor.to(speech_latent.device) - self.speech_bias_factor.to(speech_latent.device)
                audio_output = self.acoustic_tokenizer.decode(
                    scaled_latent.to(self.acoustic_tokenizer.device),
                    padding_cache=acoustic_cache,
                    batch_mask=diffusion_indices.to(self.acoustic_tokenizer.device),
                    use_cache=self.config.use_cache
                )
                audio_chunk = audio_output.audio
                acoustic_cache = audio_output.padding_cache

                # Store audio chunks for each sample
                for i, sample_idx in enumerate(diffusion_indices):
                    idx = sample_idx.item()
                    # Only append audio chunk if the sample is not finished
                    if not finished_tags[idx]:
                        audio_chunks[idx].append(audio_chunk[i])

                # Add streaming support here
                if streamer is not None:
                    # Stream the audio chunks immediately
                    streamer.put(audio_chunk, diffusion_indices)

                # Encode audio to semantic features using semantic streaming cache
                semantic_outputs = self.semantic_tokenizer.encode(
                    audio_chunk,
                    padding_cache=semantic_cache,
                    batch_mask=diffusion_indices,
                    use_cache=self.config.use_cache
                )
                semantic_features = semantic_outputs.latents
                semantic_cache = semantic_outputs.padding_cache

                # Combine acoustic and semantic features for next input
                acoustic_embed = self.acoustic_connector(speech_latent)
                semantic_embed = self.semantic_connector(semantic_features)
                diffusion_embeds = acoustic_embed + semantic_embed

                # Update embeddings for diffusion indices
                next_inputs_embeds[diffusion_indices] = diffusion_embeds

            # Set inputs_embeds for next iteration
            inputs_embeds = next_inputs_embeds

        if streamer is not None:
            streamer.end()

        # Concatenate audio chunks for each sample
        final_audio_outputs = []
        for sample_chunks in audio_chunks:
            if sample_chunks:
                # Concatenate all chunks along the time dimension (assumed to be the last dimension)
                final_audio_outputs.append(torch.cat(sample_chunks, dim=-1))
            else:
                # If no audio was generated for this sample, append None
                final_audio_outputs.append(None)

        if generation_config.return_dict_in_generate:
            return VibeVoiceGenerateOutput(
                sequences=input_ids,
                speech_outputs=final_audio_outputs,
                reach_max_step_sample=completion_steps >= max_step_per_sample,
            )
        else:
            # Return the generated audio directly
            return final_audio_outputs

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        generation_config: Optional[GenerationConfig] = None,
        streamer: Optional[Union[AudioStreamer]] = None,
        input_features: Optional[torch.FloatTensor] = None,
        input_features_mask: Optional[torch.BoolTensor] = None,
        cfg_scale: float = 1.3,
        n_diffusion_steps: int = 10,
        noise_scheduler = None,
        monitor_progress: Optional[Callable[[torch.Tensor], None]] = None,
        **kwargs,
    ) -> Union[torch.LongTensor, VibeVoiceGenerateOutput]:
        """
        Generates sequences of token ids and speech outputs for VibeVoice models.
        
        This method implements VibeVoice-specific generation that combines text generation with diffusion-based
        speech synthesis. It supports streaming audio generation and classifier-free guidance for high-quality
        speech output.

        Args:
            inputs (`torch.Tensor`, *optional*):
                The sequence used as a prompt for the model.
            generation_config (`GenerationConfig`, *optional*):
                The generation configuration to be used as base parametrization for the generation call.
            streamer (`AudioStreamer`, *optional*):
                Streamer object for real-time audio generation streaming.
            input_features (`torch.FloatTensor`, *optional*):
                Input speech tensors for voice cloning or conditioning.
            input_features_mask (`torch.BoolTensor`, *optional*):
                Masks for speech tensors to ignore padded parts.
            cfg_scale (`float`, *optional*, defaults to 1.3):
                Classifier-free guidance scale for speech generation quality control.
            n_diffusion_steps (`int`, *optional*, defaults to 10):
                Number of diffusion denoising steps for speech generation.
            noise_scheduler (`SchedulerMixin` from `diffusers, *required*):
                The noise scheduler to use for diffusion sampling.
            monitor_progress (`Callable[[torch.Tensor], None]`, *optional*):
                If provided, this function can be called to report the progress of the audio generation. The function
                takes a tensor argument `p` of shape `(n, 3)`, where `n` is the batch size. `p[i, 0]` contains the
                current generation step for batch item `i`, `p[i, 1]` contains the maximum generation steps 
                for batch item `i` (which may vary based on input length), and `p[i, 2]` contains the actual
                completion step for finished samples. No return value is expected.
            **kwargs:
                Additional model-specific kwargs that will be forwarded to the model.

        Example:
            Using `monitor_progress` to track generation progress:

            ```python
            >>> import torch
            >>> from transformers import VibeVoiceForConditionalGeneration
            >>> from tqdm import tqdm

            >>> model = VibeVoiceForConditionalGeneration.from_pretrained("vibevoice-model")

            >>> # Track which samples have completed to avoid duplicate logging
            >>> completed_samples = set()

            >>> # Define a callback to monitor the progress of the generation
            >>> with tqdm(desc="Generating") as pbar:
            >>>     def monitor_progress(p_batch):
            >>>         # Check for newly completed samples
            >>>         finished_samples = (p_batch[:, 0] == p_batch[:, 1]).nonzero(as_tuple=False).squeeze(1)
            >>>         if finished_samples.numel() > 0:
            >>>             for sample_idx in finished_samples.tolist():
            >>>                 if sample_idx not in completed_samples:
            >>>                     completed_samples.add(sample_idx)
            >>>                     # Use the actual completion step from column 2
            >>>                     completion_step = int(p_batch[sample_idx, 2])
            >>>                     print(f"Sample {sample_idx} completed at step {completion_step}", flush=True)
            >>>         
            >>>         # Find the sample with the maximum progress (most advanced)
            >>>         # This ensures the progress bar continues updating as long as any sample is generating
            >>>         active_samples = p_batch[:, 0] < p_batch[:, 1]  # Samples that haven't finished
            >>>         if active_samples.any():
            >>>             # Use the most advanced active sample
            >>>             active_progress = p_batch[active_samples]
            >>>             max_active_idx = torch.argmax(active_progress[:, 0])
            >>>             p = active_progress[max_active_idx].detach().cpu()
            >>>         else:
            >>>             # All samples finished, use any sample (they should all be at max)
            >>>             p = p_batch[0].detach().cpu()
            >>>         
            >>>         pbar.total = int(p[1])
            >>>         pbar.n = int(p[0])
            >>>         pbar.update()

            >>>     # Generate audio with progress monitoring
            >>>     output = model.generate(
            >>>         input_ids=input_ids,
            >>>         input_features=input_features,
            >>>         noise_scheduler=noise_scheduler,
            >>>         monitor_progress=monitor_progress
            >>>     )
            ```

        Returns:
            [`VibeVoiceGenerateOutput`] or `torch.LongTensor`: A [`VibeVoiceGenerateOutput`] (if 
            `return_dict_in_generate=True` or when `config.return_dict_in_generate=True`) or a 
            `torch.LongTensor` containing the generated token sequences. When speech synthesis
            is performed, also includes the generated audio waveforms.
        """
        if noise_scheduler is None:
            raise ValueError("`noise_scheduler` from `diffusers` must be provided for VibeVoice generation.")

        # Pass VibeVoice-specific parameters through kwargs to _sample method
        kwargs.update({
            'vibevoice_input_features': input_features,
            'vibevoice_input_features_mask': input_features_mask,
            'vibevoice_cfg_scale': cfg_scale,
            'vibevoice_n_diffusion_steps': n_diffusion_steps,
            'vibevoice_noise_scheduler': noise_scheduler,
            'vibevoice_monitor_progress': monitor_progress,
        })
        
        # Call the base generate method, which will use our custom _sample method
        return super().generate(
            inputs=inputs,
            generation_config=generation_config,
            streamer=streamer,
            use_model_defaults=False,  # Prevent warning about model-specific defaults
            **kwargs
        )
