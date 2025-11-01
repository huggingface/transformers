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

from dataclasses import dataclass
from queue import Queue
from typing import Optional, Union

import torch
import torch.nn as nn
from tqdm import tqdm

from ...generation import BaseStreamer, GenerationConfig, GenerationMixin, LogitsProcessor, LogitsProcessorList
from ...modeling_outputs import ModelOutput
from ...utils import logging


logger = logging.get_logger(__name__)


@dataclass
class VibeVoiceGenerationOutput(ModelOutput):
    """
    Output type for VibeVoice generation.
    
    Args:
        sequences (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            The generated sequences. 
        speech_outputs (`List[torch.FloatTensor]`, *optional*):
            List of generated speech waveforms or latents for each speech segment.
    """
    sequences: torch.LongTensor = None
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

    def _build_generate_config_model_kwargs(self, generation_config, inputs, tokenizer, return_processors=False, **kwargs):
        if generation_config is None:
            generation_config = GenerationConfig(
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id = tokenizer.pad_token_id
            )
        else:
            generation_config = GenerationConfig(
                **generation_config,
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id = tokenizer.pad_token_id
            )

        generation_config, model_kwargs = self._prepare_generation_config(
            generation_config,
            True,
            speech_start_id=tokenizer.speech_start_id,
            speech_end_id=tokenizer.speech_end_id,
            speech_diffusion_id=tokenizer.speech_diffusion_id,
            **kwargs
        )
        generation_config.speech_start_id = tokenizer.speech_start_id
        generation_config.speech_end_id = tokenizer.speech_end_id
        generation_config.speech_diffusion_id = tokenizer.speech_diffusion_id

        inputs_tensor, model_input_name, model_kwargs = self._prepare_model_inputs(inputs, generation_config.bos_token_id, model_kwargs)
        batch_size = inputs_tensor.shape[0]
        device = self.device

        self._prepare_special_tokens(generation_config, True, device=device)
        generation_config.use_cache = True
        model_kwargs["use_cache"] = True
        input_ids = inputs_tensor.to(self.device)

        input_ids_length = input_ids.shape[1]
        has_default_max_length = kwargs.get("max_length") is None and generation_config.max_length is not None
        has_default_min_length = kwargs.get("min_length") is None and generation_config.min_length is not None
        generation_config = self._prepare_generated_length(
            generation_config=generation_config,
            has_default_max_length=has_default_max_length,
            has_default_min_length=has_default_min_length,
            model_input_name=model_input_name,
            inputs_tensor=inputs_tensor,
            input_ids_length=input_ids_length,
        )

        max_cache_length = generation_config.max_length - 1
        self._prepare_cache_for_generation(generation_config, model_kwargs, None, batch_size, max_cache_length)
        model_kwargs['cache_position'] = torch.arange(input_ids_length, device=device, dtype=torch.long)
        for k, v in model_kwargs.items():
            if isinstance(v, torch.Tensor):
                model_kwargs[k] = v.to(device=device)

        if return_processors:
            logits_processor = self._get_logits_processor(
                generation_config=generation_config,
                input_ids_seq_length=input_ids_length,
                encoder_input_ids=inputs_tensor,
                prefix_allowed_tokens_fn=None,
                logits_processor=LogitsProcessorList(),
                device=inputs_tensor.device,
                model_kwargs=model_kwargs,
            )

            return generation_config, model_kwargs, input_ids, logits_processor
        else:
            return generation_config, model_kwargs, input_ids

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        generation_config: Optional[GenerationConfig] = None,
        audio_streamer: Optional[Union[AudioStreamer]] = None,
        speech_tensors: Optional[torch.FloatTensor] = None,
        speech_masks: Optional[torch.BoolTensor] = None,        # TODO rename, this is to ignore padded parts
        speech_input_mask: Optional[torch.BoolTensor] = None,   # TODO rename, this is to know where is speech in script
        cfg_scale: float = 1.0,
        **kwargs,
    ) -> Union[torch.LongTensor, VibeVoiceGenerationOutput]:
        """
        Generates sequences of token ids and optionally speech outputs.
        
        Args:
            All standard generation arguments from GenerationMixin
            negative_prompt_ids: Negative prompt for CFG in speech generation
            negative_prompt_attention_mask: Attention mask for negative prompt
            speech_tensors: Input speech for voice cloning
            speech_masks: Masks for speech tensors  
            speech_input_mask: Positions to insert speech embeddings
            cfg_scale: CFG scale for speech generation
 
        Returns:
            Generated token sequences and optionally speech outputs
        """

        # 1. Handle `generation_config` and kwargs that might update it, and validate the `.generate()` call
        tokenizer = kwargs.pop("tokenizer", None)  # Pull this out first, we only use it for stopping criteria
        max_length_times = kwargs.pop("max_length_times", 2)

        if kwargs.get('max_new_tokens') is None:
            kwargs['max_new_tokens'] = self.config.decoder_config.max_position_embeddings - kwargs['input_ids'].shape[-1]

        generation_config, model_kwargs, input_ids, logits_processor = self._build_generate_config_model_kwargs(
            generation_config, inputs, tokenizer, return_processors=True, **kwargs
        )

        negative_kwargs = {
            'input_ids': torch.full((kwargs['input_ids'].shape[0], 1), tokenizer.speech_start_id, dtype=torch.long, device=kwargs['input_ids'].device),
            'attention_mask':  torch.ones((kwargs['input_ids'].shape[0], 1), dtype=torch.long, device=kwargs['input_ids'].device),
            'max_new_tokens': kwargs.get('max_new_tokens', 100)
        }
        _, negative_model_kwargs, negative_input_ids = self._build_generate_config_model_kwargs(
            None, None, tokenizer, return_processors=False, **negative_kwargs
        )

        acoustic_cache = None
        semantic_cache = None

        batch_size = input_ids.shape[0]
        device = input_ids.device
        finished_tags = torch.zeros(batch_size, dtype=torch.bool, device=device)
        correct_cnt = torch.zeros(batch_size, dtype=torch.long, device=device)
        is_prefill = True
        inputs_embeds = None
        verbose = kwargs.get("verbose", False)

        # Initialize audio chunks storage for each sample
        audio_chunks = [[] for _ in range(batch_size)]

        initial_length = input_ids.shape[-1]
        initial_length_per_sample = model_kwargs['attention_mask'].sum(dim=-1)

       # Define all valid tokens that can be generated
        valid_tokens = [
            generation_config.speech_start_id,
            generation_config.speech_end_id,
            generation_config.speech_diffusion_id,
            generation_config.eos_token_id
        ]
        # Add bos_token_id if it exists
        if hasattr(generation_config, 'bos_token_id') and generation_config.bos_token_id is not None:
            valid_tokens.append(generation_config.bos_token_id)

        # Add custom processor to constrain token generation
        token_constraint_processor = VibeVoiceTokenConstraintProcessor(valid_tokens, device=device)
        logits_processor.append(token_constraint_processor)

        max_steps = min(generation_config.max_length - initial_length, int(max_length_times * initial_length))
        max_step_per_sample = torch.min(generation_config.max_length - initial_length_per_sample, (max_length_times * initial_length_per_sample).long())
        reach_max_step_sample = torch.zeros(batch_size, dtype=torch.bool, device=device)

        # Create progress iterator if verbose
        # TODO (ebezzam) remove from final?
        if kwargs.get("show_progress_bar", True):
            progress_bar = tqdm(range(max_steps), desc="Generating", leave=False)
        else:
            progress_bar = range(max_steps)

        for step in progress_bar:

            # Check if audio_streamer has been ended (stopped externally)
            if audio_streamer is not None and hasattr(audio_streamer, 'finished_flags'):
                if any(audio_streamer.finished_flags):
                    if verbose:
                        print(f"Audio generation stopped externally at step {step + 1}")
                    break

            if finished_tags.all():
                if hasattr(progress_bar, 'set_description'):
                    progress_bar.set_description("Generation complete")
                break

            if input_ids.shape[-1] >= generation_config.max_length:
                print(f"Reached maximum generation length {generation_config.max_length}, stopped it.")
                reached_samples = torch.arange(batch_size, device=device)[~finished_tags]
                if reached_samples.numel() > 0:
                    reach_max_step_sample[reached_samples] = True
                break

            # Update progress bar description with active samples
            if hasattr(progress_bar, 'set_description'):
                active_samples = (~finished_tags).sum().item()
                progress_bar.set_description(f"Generating (active: {active_samples}/{batch_size})")

            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
            if is_prefill:
                # we process the speech inputs only during the first generation step
                prefill_inputs = {
                    "speech_tensors": speech_tensors.to(device=device),
                    "speech_masks": speech_masks.to(device),
                    "speech_input_mask": speech_input_mask.to(device),
                }
                is_prefill = False
            else:
                _ = model_inputs.pop('inputs_embeds', None)
                prefill_inputs = {'inputs_embeds': inputs_embeds}

            # Forward pass through the model
            outputs = self(
                **model_inputs, **prefill_inputs, logits_to_keep=1, return_dict=True, output_attentions=False, output_hidden_states=False,
            )
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=False,
            )

            # Get logits and apply logits processor
            next_token_logits = outputs.logits[:, -1, :].to(copy=True, dtype=torch.float32, device=input_ids.device)
            # next_token_logits = outputs.logits[:, -1, :].to(copy=True, device=input_ids.device)
            next_token_scores = logits_processor(input_ids, next_token_logits)

            # token selection
            if generation_config.do_sample:
                probs = nn.functional.softmax(next_token_scores, dim=-1)
                # TODO (joao): this OP throws "skipping cudagraphs due to ['incompatible ops']", find solution
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                next_tokens = torch.argmax(next_token_scores, dim=-1)

            next_tokens[finished_tags] = generation_config.eos_token_id
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)

            # reached end of generation
            if (next_tokens == generation_config.eos_token_id).any():
                eos_indices = (next_tokens == generation_config.eos_token_id).nonzero(as_tuple=False).squeeze(1)
                # Only print for samples that are newly finished (not already marked as finished)
                new_eos_indices = eos_indices[~finished_tags[eos_indices]]
                if new_eos_indices.numel() > 0:
                    finished_tags[new_eos_indices] = True
                    if verbose:
                        print(f"Samples {new_eos_indices.tolist()} reached EOS token at step {step + 1}.", flush=True)
                    if audio_streamer is not None:
                        audio_streamer.end(new_eos_indices)

            # Check if any sample reached its maximum generation length
            max_length_reached = step >= max_step_per_sample
            new_max_length_indices = torch.nonzero(max_length_reached & ~finished_tags, as_tuple=False).squeeze(1)
            if new_max_length_indices.numel() > 0:
                finished_tags[new_max_length_indices] = True
                reach_max_step_sample[new_max_length_indices] = True
                if verbose:
                    print(f"Samples {new_max_length_indices.tolist()} reached max generation length at step {step + 1}.", flush=True)
                if audio_streamer is not None:
                    audio_streamer.end(new_max_length_indices)

            # speech_end
            diffusion_end_indices = (next_tokens == generation_config.speech_end_id).nonzero(as_tuple=False).squeeze(1)

            # speech_begin
            diffusion_start_indices = torch.arange(batch_size, device=device)[~finished_tags & (next_tokens == generation_config.speech_start_id)]
            if diffusion_start_indices.numel() > 0 and kwargs.get('refresh_negative', True):
                # update attention mask
                for i, sample_idx in enumerate(diffusion_start_indices.tolist()):
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
                    negative_input_ids[sample_idx, -1] = generation_config.speech_start_id

            # Prepare inputs_embeds for next iteration
            # Initialize with default embeddings for all tokens
            next_inputs_embeds = self.model.get_input_embeddings()(next_tokens).unsqueeze(1)  # [batch_size, 1, hidden_size]

            # forward diffusion
            # Diffusion indices are those that are not finished and not special tokens
            diffusion_indices = torch.arange(batch_size, device=device)[~finished_tags & (next_tokens == generation_config.speech_diffusion_id)]

            if diffusion_indices.numel() > 0:
                negative_model_inputs = self.prepare_inputs_for_generation(negative_input_ids, **negative_model_kwargs)
                # Forward negative pass through the model
                if negative_model_inputs['inputs_embeds'] is None and inputs_embeds is not None:
                    negative_model_inputs['inputs_embeds'] = inputs_embeds
                    negative_model_inputs['input_ids'] = None

                negative_outputs = self(
                    **negative_model_inputs, logits_to_keep=0, return_dict=True, output_attentions=False, output_hidden_states=False,
                )
                negative_model_kwargs = self._update_model_kwargs_for_generation(
                    negative_outputs, negative_model_kwargs, is_encoder_decoder=False,
                )
                negative_input_ids = torch.cat([negative_input_ids, next_tokens[:, None]], dim=-1)

                # correct the non-diffusion indices
                # we forward all samples' negative outputs even if
                #   they are not in diffusion mode to keep the cache consistent
                # So we need to correct the kv cache of non-diffusion samples
                non_diffusion_mask = ~finished_tags & (next_tokens != generation_config.speech_diffusion_id)
                if non_diffusion_mask.any():
                    non_diffusion_indices = torch.arange(batch_size, device=device)[non_diffusion_mask]
                    start_indices = correct_cnt[non_diffusion_indices]

                    # 1. Update attention_mask - need to handle each sample separately
                    seq_len = negative_model_kwargs['attention_mask'].shape[1]
                    for i, (sample_idx, start_idx) in enumerate(zip(non_diffusion_indices.tolist(), start_indices.tolist())):
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

                # Sample speech tokens from Diffusion model with classifier-free guidance (CFG)
                self.model.noise_scheduler.set_timesteps(self.ddpm_inference_steps)
                condition = torch.cat([positive_condition, negative_condition], dim=0).to(self.model.diffusion_head.device)
                speech = torch.randn(condition.shape[0], self.config.acoustic_hidden_size).to(condition)
                with torch.no_grad():
                    for t in self.model.noise_scheduler.timesteps:
                        half = speech[: len(speech) // 2]
                        combined = torch.cat([half, half], dim=0)
                        eps = self.model.diffusion_head(combined, t.repeat(combined.shape[0]).to(combined), condition=condition)
                        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
                        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
                        eps = torch.cat([half_eps, half_eps], dim=0)
                        speech = self.model.noise_scheduler.step(eps, t, speech).prev_sample
                speech_latent = speech[: len(speech) // 2].unsqueeze(1)

                # Decode acoustic latent to audio using acoustic streaming cache
                scaled_latent = speech_latent / self.model.speech_scaling_factor.to(speech_latent.device) - self.model.speech_bias_factor.to(speech_latent.device)
                with torch.no_grad():
                    audio_output = self.model.acoustic_tokenizer.decode(
                        scaled_latent.to(self.model.acoustic_tokenizer.device),
                        padding_cache=acoustic_cache,
                        batch_mask=diffusion_indices.to(self.model.acoustic_tokenizer.device),
                        use_cache=True
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
                if audio_streamer is not None:
                    # Stream the audio chunks immediately
                    audio_streamer.put(audio_chunk, diffusion_indices)

                # Encode audio to semantic features using semantic streaming cache
                with torch.no_grad():
                    semantic_outputs = self.model.semantic_tokenizer.encode(
                        audio_chunk,
                        padding_cache=semantic_cache,
                        batch_mask=diffusion_indices,
                        use_cache=True
                    )
                semantic_features = semantic_outputs.latents
                semantic_cache = semantic_outputs.padding_cache

                # Combine acoustic and semantic features for next input
                acoustic_embed = self.model.acoustic_connector(speech_latent)
                semantic_embed = self.model.semantic_connector(semantic_features)
                diffusion_embeds = acoustic_embed + semantic_embed

                # Update embeddings for diffusion indices
                next_inputs_embeds[diffusion_indices] = diffusion_embeds

            # Set inputs_embeds for next iteration
            inputs_embeds = next_inputs_embeds

        if audio_streamer is not None:
            audio_streamer.end()

        # Concatenate audio chunks for each sample
        final_audio_outputs = []
        for sample_chunks in audio_chunks:
            if sample_chunks:
                # Concatenate all chunks along the time dimension (assumed to be the last dimension)
                concatenated_audio = torch.cat(sample_chunks, dim=-1)
                final_audio_outputs.append(concatenated_audio)
            else:
                # If no audio was generated for this sample, append None
                final_audio_outputs.append(None)

        return VibeVoiceGenerationOutput(
            sequences=input_ids,
            speech_outputs=final_audio_outputs,
            reach_max_step_sample=reach_max_step_sample,
        )
