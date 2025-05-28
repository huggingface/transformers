# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team.
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

import time

import numpy as np
import torch

from ...generation import GenerationMixin
from ...utils import logging
from .configuration_dia import DiaConfig
from .processing_dia import apply_audio_delay, build_delay_indices, build_revert_indices, revert_audio_delay


logger = logging.get_logger(__name__)

DEFAULT_AUDIO_SAMPLE_RATE = 44100
AUDIO_CODE_TO_SAMPLE_RATIO = 512  # Each audio token corresponds to this many audio samples


@torch.no_grad()
@torch.inference_mode()
def decode(
    model,  # This is expected to be the DAC model
    audio_codes: torch.Tensor,  # Expects [B, C, T] from codebook.transpose(1, 2)
):
    """
    Decodes the given frames into an output audio waveform. Handles batches.
    """
    try:
        audio_latents, _, _ = model.quantizer.from_codes(audio_codes)  # DAC model's quantizer
        audio_values = model.decode(
            audio_latents
        )  # DAC model's decoder, handles batch [B, 1, T_audio] or [B, T_audio]

        if audio_values.ndim == 3 and audio_values.shape[1] == 1:
            audio_values = audio_values.squeeze(1)

        return audio_values  # Return [B, T_audio]
    except Exception as e:
        print(f"Error in decode method: {str(e)}")
        raise


def _sample_next_token(
    logits_BCxV: torch.Tensor,
    temperature: float,
    top_p: float,
    top_k: int | None,
    audio_eos_value: int,
) -> torch.Tensor:
    if temperature == 0.0:
        return torch.argmax(logits_BCxV, dim=-1)

    logits_BCxV = logits_BCxV / temperature

    if audio_eos_value is not None and audio_eos_value >= 0:
        top_logit_indices_BC = torch.argmax(logits_BCxV, dim=-1)
        eos_not_highest_mask_BC = top_logit_indices_BC != audio_eos_value
        mask_eos_unless_highest_BCxV = torch.zeros_like(logits_BCxV, dtype=torch.bool)
        mask_eos_unless_highest_BCxV[eos_not_highest_mask_BC, audio_eos_value] = True
        logits_BCxV = logits_BCxV.masked_fill(mask_eos_unless_highest_BCxV, -torch.inf)
        eos_highest_mask_BC = top_logit_indices_BC == audio_eos_value
        mask_eos_highest_BCxV = torch.zeros_like(logits_BCxV, dtype=torch.bool)
        mask_eos_highest_BCxV[eos_highest_mask_BC, :audio_eos_value] = True
        logits_BCxV = logits_BCxV.masked_fill(mask_eos_highest_BCxV, -torch.inf)

    if top_k is not None:
        _, top_k_indices_BCxV = torch.topk(logits_BCxV, k=top_k, dim=-1)
        mask = torch.ones_like(logits_BCxV, dtype=torch.bool)
        mask = mask.scatter(dim=-1, index=top_k_indices_BCxV, value=False)
        logits_BCxV = logits_BCxV.masked_fill(mask, -torch.inf)

    if top_p < 1.0:
        probs_BCxV = torch.softmax(logits_BCxV, dim=-1)
        sorted_probs_BCxV, sorted_indices_BCxV = torch.sort(probs_BCxV, dim=-1, descending=True)
        cumulative_probs_BCxV = torch.cumsum(sorted_probs_BCxV, dim=-1)

        sorted_indices_to_remove_BCxV = cumulative_probs_BCxV > top_p
        sorted_indices_to_remove_BCxV = torch.roll(sorted_indices_to_remove_BCxV, shifts=1, dims=-1)
        sorted_indices_to_remove_BCxV[..., 0] = torch.zeros_like(sorted_indices_to_remove_BCxV[..., 0])

        indices_to_remove_BCxV = torch.zeros_like(sorted_indices_to_remove_BCxV)
        indices_to_remove_BCxV = indices_to_remove_BCxV.scatter(
            dim=-1, index=sorted_indices_BCxV, src=sorted_indices_to_remove_BCxV
        )
        logits_BCxV = logits_BCxV.masked_fill(indices_to_remove_BCxV, -torch.inf)

    final_probs_BCxV = torch.softmax(logits_BCxV, dim=-1)

    sampled_indices_BC = torch.multinomial(final_probs_BCxV, num_samples=1)
    sampled_indices_C = sampled_indices_BC.squeeze(-1)
    return sampled_indices_C


class DiaGenerationMixin(GenerationMixin):
    def __init__(self, config: DiaConfig):
        super().__init__(config)
        self.dac_model = None
        self.config = config

    def _load_dac_model(self):
        try:
            import dac

            if not hasattr(self, "dac_model") or self.dac_model is None:
                logger.info("Loading DAC model...")
                dac_model_path = dac.utils.download()
                # type: ignore
                self.dac_model = dac.DAC.load(dac_model_path).to(self.device)
                self.dac_model.eval()
        except ImportError:
            raise ImportError(
                "DAC model not found. Please install it with `pip install descript-audio-codec`"
            ) from None
        except Exception as e:
            raise RuntimeError("Failed to load DAC model") from e

    def _load_audio_file_to_dac_codes(self, audio_path: str) -> torch.Tensor:
        """Loads an audio file, resamples, and encodes it to DAC codes."""
        import torchaudio

        if not hasattr(self, "dac_model") or self.dac_model is None:
            self._load_dac_model()

        try:
            audio_waveform, sr = torchaudio.load(audio_path)
            audio_waveform = audio_waveform.to(self.device)
            if audio_waveform.ndim == 1:
                audio_waveform = audio_waveform.unsqueeze(0)
            if audio_waveform.shape[0] > 1:
                audio_waveform = audio_waveform[0:1, :]

        except Exception as e:
            raise RuntimeError(f"Error loading audio file {audio_path}") from e

        if sr != DEFAULT_AUDIO_SAMPLE_RATE:
            audio_waveform = torchaudio.functional.resample(audio_waveform, sr, DEFAULT_AUDIO_SAMPLE_RATE)

        try:
            processed_audio = self.dac_model.preprocess(audio_waveform, DEFAULT_AUDIO_SAMPLE_RATE)
            _, codes, _, _, _ = self.dac_model.encode(processed_audio)
            return codes.squeeze(0).transpose(0, 1)
        except Exception as e:
            logger.error(f"Error encoding audio prompt {audio_path} with DAC: {str(e)}")
            raise

    def _prepare_audio_prompts_for_decoder(
        self,
        audio_prompts_dac_list: list[torch.Tensor | None],
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_channels = self.config.decoder_config.num_channels
        audio_bos_value = self.generation_config.bos_token_id
        if audio_bos_value is None:
            logger.warning("`bos_token_id` not found in generation_config. Using a default value for audio BOS.")
            audio_bos_value = 1026

        delay_pattern = self.config.delay_pattern
        max_delay_pattern = max(delay_pattern)
        batch_size = len(audio_prompts_dac_list)

        max_prompt_len = 0
        for p in audio_prompts_dac_list:
            if p is not None:
                max_prompt_len = max(max_prompt_len, p.shape[0])

        max_len_dim_for_prefill_tensor = max_prompt_len + 1 + max_delay_pattern

        prefill_data_tensor = torch.full(
            (batch_size, max_len_dim_for_prefill_tensor, num_channels),
            fill_value=-1,
            dtype=torch.long,
            device=device,
        )

        prefill_actual_lengths_list = []

        prefill_data_tensor[:, 0, :] = audio_bos_value

        for i in range(batch_size):
            prompt_tensor = audio_prompts_dac_list[i]
            current_actual_length = 1
            if prompt_tensor is not None:
                prompt_tensor = prompt_tensor.to(device=device, dtype=torch.long)
                p_len = prompt_tensor.shape[0]
                if p_len > 0:
                    copy_len = min(p_len, max_len_dim_for_prefill_tensor - 1)
                    prefill_data_tensor[i, 1 : copy_len + 1, :] = prompt_tensor[:copy_len, :]
                    current_actual_length += copy_len
            prefill_actual_lengths_list.append(current_actual_length)

        prefill_actual_lengths_tensor = torch.tensor(prefill_actual_lengths_list, device=device, dtype=torch.long)

        delay_indices_precomputation = build_delay_indices(
            B=batch_size,
            T=max_len_dim_for_prefill_tensor,
            C=num_channels,
            delay_pattern=delay_pattern,
        )

        delayed_audio_prompt_codes = apply_audio_delay(
            audio_BxTxC=prefill_data_tensor,
            pad_value=-1,
            bos_value=audio_bos_value,
            precomp=delay_indices_precomputation,
        )
        return delayed_audio_prompt_codes, prefill_actual_lengths_tensor

    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.Tensor,
        eos_token_id: int | None = None,
        attention_mask: torch.Tensor | None = None,
        max_new_tokens: int | None = None,
        temperature: float = 1.3,
        top_p: float = 0.95,
        top_k: int | None = None,
        cfg_scale: float = 3.0,
        use_torch_compile: bool = False,
        audio_prompt: list[str | torch.Tensor | None] | str | torch.Tensor | None = None,
        verbose: bool = False,
        **kwargs,
    ) -> np.ndarray | list[np.ndarray]:
        # self._load_dac_model()
        device = self.device
        batch_size = input_ids.shape[0]

        audio_eos_value = eos_token_id or self.generation_config.eos_token_id
        audio_pad_value = self.generation_config.pad_token_id
        audio_bos_value = self.generation_config.bos_token_id
        delay_pattern = self.config.delay_pattern
        max_delay_pattern = max(delay_pattern)
        num_audio_channels = self.config.decoder_config.num_channels

        if max_new_tokens is None:
            max_new_tokens = self.generation_config.max_new_tokens or self.config.max_length_for_generation or 256

        if audio_eos_value is None or audio_pad_value is None or audio_bos_value is None:
            raise ValueError("eos_token_id, pad_token_id, and bos_token_id must be set in generation_config.")

        if verbose:
            total_start_time = time.time()
            logger.info("Starting Dia generation...")

        uncond_input_ids = torch.zeros_like(input_ids)
        stacked_input_ids = torch.cat([input_ids, uncond_input_ids], dim=0)
        if attention_mask is not None:
            uncond_attention_mask = torch.ones_like(attention_mask)
            stacked_attention_mask = torch.cat([attention_mask, uncond_attention_mask], dim=0)
        else:
            stacked_attention_mask = None

        audio_prompts_dac_list: list[torch.Tensor | None] = []
        if audio_prompt is None:
            audio_prompts_dac_list = [None] * batch_size
        elif isinstance(audio_prompt, list):
            if len(audio_prompt) != batch_size:
                raise ValueError(
                    f"Length of audio_prompt list ({len(audio_prompt)}) must match batch_size ({batch_size})."
                )
            for item in audio_prompt:
                if item is None:
                    audio_prompts_dac_list.append(None)
                elif isinstance(item, str):
                    audio_prompts_dac_list.append(self._load_audio_file_to_dac_codes(item))
                elif isinstance(item, torch.Tensor):
                    audio_prompts_dac_list.append(item.to(device=device))
                else:
                    raise ValueError(f"Invalid item type in audio_prompt list: {type(item)}")
        elif isinstance(audio_prompt, str):
            audio_prompts_dac_list = [self._load_audio_file_to_dac_codes(audio_prompt)] * batch_size
        elif isinstance(audio_prompt, torch.Tensor):
            audio_prompts_dac_list = [audio_prompt.to(device=device)] * batch_size
        else:
            raise ValueError(f"Invalid audio_prompt type: {type(audio_prompt)}")

        initial_decoder_prompt_codes_BxTxC, prefill_lengths_Bx = self._prepare_audio_prompts_for_decoder(
            audio_prompts_dac_list, device
        )

        encoder = self.get_encoder()
        encoder_outputs = encoder(
            input_ids=stacked_input_ids,
            attention_mask=stacked_attention_mask,
            return_dict=True,
        )

        past_key_values = None

        max_len_prompt_in_generated_tensor = initial_decoder_prompt_codes_BxTxC.shape[1]
        total_max_audio_len = max_len_prompt_in_generated_tensor + max_new_tokens

        generated_audio_tokens_BxTxC = torch.full(
            (batch_size, total_max_audio_len, num_audio_channels),
            fill_value=audio_pad_value,
            dtype=torch.long,
            device=device,
        )

        if max_len_prompt_in_generated_tensor > 0:
            generated_audio_tokens_BxTxC[:, :max_len_prompt_in_generated_tensor, :] = (
                initial_decoder_prompt_codes_BxTxC
            )

        cache_position = torch.zeros((batch_size, 1), dtype=torch.long, device=device)
        for i in range(batch_size):
            cache_position[i, 0] = prefill_lengths_Bx[i].item()

        eos_detected_Bx = torch.zeros((batch_size,), dtype=torch.bool, device=device)
        eos_countdown_Bx = torch.full((batch_size,), -1, dtype=torch.long, device=device)
        finished_generation_step_idx_Bx = torch.full((batch_size,), -1, dtype=torch.long, device=device)
        delay_pattern_tensor_Cx = torch.tensor(delay_pattern, device=device, dtype=torch.long)

        for gen_idx in range(max_new_tokens):
            if (eos_countdown_Bx == 0).all():
                if verbose:
                    logger.info("All sequences finished generation.")
                break

            indices_for_input_token_Bx = (cache_position.squeeze(-1) - 1).clamp(min=0)
            decoder_input_ids_step = torch.stack(
                [generated_audio_tokens_BxTxC[b, indices_for_input_token_Bx[b], :] for b in range(batch_size)]
            ).unsqueeze(1)

            decoder_input_ids_step_cfg = decoder_input_ids_step.repeat_interleave(2, dim=0)
            current_cache_pos_cfg = cache_position.repeat_interleave(2, dim=0)

            outputs = self(
                input_ids=None,
                attention_mask=None,
                encoder_outputs=encoder_outputs,
                audio_codes=decoder_input_ids_step_cfg,
                past_key_values=past_key_values,
                cache_position=current_cache_pos_cfg,
                use_cache=True,
                return_dict=True,
            )

            past_key_values = outputs.past_key_values
            logits_2BxCxV = outputs.logits.squeeze(1)

            cond_logits_BxCxV, uncond_logits_BxCxV = torch.chunk(logits_2BxCxV, 2, dim=0)
            logits_cfg_BxCxV = uncond_logits_BxCxV + cfg_scale * (cond_logits_BxCxV - uncond_logits_BxCxV)

            logits_cfg_BxCxV[:, :, audio_eos_value + 1 :] = -torch.inf
            if num_audio_channels > 1:
                logits_cfg_BxCxV[:, 1:, audio_eos_value] = -torch.inf

            flat_logits_BCxV = logits_cfg_BxCxV.reshape(-1, logits_cfg_BxCxV.shape[-1])

            pred_flat_BC = _sample_next_token(
                flat_logits_BCxV,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                audio_eos_value=audio_eos_value,
            )
            pred_BxC = pred_flat_BC.reshape(batch_size, num_audio_channels)

            active_mask_Bx = eos_countdown_Bx != 0
            eos_trigger_this_step_Bx = torch.zeros_like(active_mask_Bx)

            if active_mask_Bx.any():
                is_eos_token_on_main_channel_Bx = (~eos_detected_Bx[active_mask_Bx]) & (
                    pred_BxC[active_mask_Bx, 0] == audio_eos_value
                )
                is_max_len_for_eos_trigger_Bx = gen_idx >= max_new_tokens - max_delay_pattern
                eos_trigger_this_step_Bx[active_mask_Bx] = (
                    is_eos_token_on_main_channel_Bx | is_max_len_for_eos_trigger_Bx
                )

            eos_detected_Bx |= eos_trigger_this_step_Bx
            start_countdown_mask_Bx = eos_trigger_this_step_Bx & (eos_countdown_Bx < 0)
            if start_countdown_mask_Bx.any():
                eos_countdown_Bx[start_countdown_mask_Bx] = max_delay_pattern
                finished_generation_step_idx_Bx[start_countdown_mask_Bx] = gen_idx

            padding_during_eos_mask_Bx = eos_countdown_Bx > 0
            if padding_during_eos_mask_Bx.any():
                pred_for_padding_NxC = pred_BxC[padding_during_eos_mask_Bx].clone()
                countdown_active_Nx = eos_countdown_Bx[padding_during_eos_mask_Bx]
                step_after_eos_Nx = max_delay_pattern - countdown_active_Nx
                step_after_eos_Nx_ = step_after_eos_Nx.unsqueeze(1)
                delay_pattern_Cx_ = delay_pattern_tensor_Cx.unsqueeze(0)
                eos_fill_mask_NxC = step_after_eos_Nx_ == delay_pattern_Cx_
                pad_fill_mask_NxC = step_after_eos_Nx_ > delay_pattern_Cx_
                pred_for_padding_NxC[eos_fill_mask_NxC] = audio_eos_value
                pred_for_padding_NxC[pad_fill_mask_NxC] = audio_pad_value
                pred_BxC[padding_during_eos_mask_Bx] = pred_for_padding_NxC
                eos_countdown_Bx[padding_during_eos_mask_Bx] -= 1

            current_token_storage_idx_Bx = cache_position.squeeze(-1)
            for b_idx in range(batch_size):
                if active_mask_Bx[b_idx] or eos_countdown_Bx[b_idx] == 0:
                    if current_token_storage_idx_Bx[b_idx] < total_max_audio_len:
                        generated_audio_tokens_BxTxC[b_idx, current_token_storage_idx_Bx[b_idx], :] = pred_BxC[
                            b_idx, :
                        ]
                    else:
                        if verbose:
                            logger.warning(
                                f"Index out of bounds for generated_audio_tokens_BxTxC at step {gen_idx} for batch {b_idx}."
                            )

            cache_position[active_mask_Bx, 0] += 1

            if verbose and (gen_idx + 1) % 50 == 0:
                logger.info(f"Generated {gen_idx + 1}/{max_new_tokens} new audio tokens.")

        final_newly_generated_lengths_Bx = torch.where(
            finished_generation_step_idx_Bx == -1,
            max_new_tokens,
            (finished_generation_step_idx_Bx + max_delay_pattern).clamp(max=max_new_tokens),
        )

        total_valid_token_lengths_Bx = prefill_lengths_Bx + final_newly_generated_lengths_Bx
        total_valid_token_lengths_Bx = total_valid_token_lengths_Bx.clamp(max=total_max_audio_len)

        max_final_len_in_batch = (
            total_valid_token_lengths_Bx.max().item()
            if batch_size > 0 and total_valid_token_lengths_Bx.numel() > 0
            else 0
        )
        final_codes_for_dac_BxTxC = generated_audio_tokens_BxTxC[:, :max_final_len_in_batch, :]

        if verbose:
            avg_steps = total_valid_token_lengths_Bx.float().mean().item()
            total_duration = time.time() - total_start_time
            logger.info(
                f"Generation finished. Avg total tokens: {avg_steps:.1f}, Total duration: {total_duration:.3f}s"
            )

        raw_decoded_audio_output = self._generate_output(final_codes_for_dac_BxTxC)
        final_audio_outputs = []
        is_batched_output = isinstance(raw_decoded_audio_output, list)

        for b_idx in range(batch_size):
            current_raw_audio = raw_decoded_audio_output[b_idx] if is_batched_output else raw_decoded_audio_output
            num_codes_for_dac = total_valid_token_lengths_Bx[b_idx].item()
            if num_codes_for_dac <= max_delay_pattern:
                final_audio_outputs.append(np.array([], dtype=np.float32))
                continue

            effective_code_frames = num_codes_for_dac - max_delay_pattern
            expected_audio_samples = effective_code_frames * AUDIO_CODE_TO_SAMPLE_RATIO
            trimmed_audio = current_raw_audio[:expected_audio_samples]
            final_audio_outputs.append(trimmed_audio)

        return final_audio_outputs if batch_size > 1 else final_audio_outputs[0]

    def _generate_output(self, generated_codes_BxTxC: torch.Tensor) -> list[np.ndarray] | np.ndarray:
        # `generated_codes_BxTxC` is [B, T_max_valid_across_batch, C]
        # This method should ideally take `lengths_Bx` to properly handle varying lengths BEFORE DAC.
        # The current version processes the full `generated_codes_BxTxC` through revert_audio_delay & DAC.
        # Trimming is done in `generate` method after this call. This is kept for now.

        num_channels = self.config.decoder_config.num_channels
        batch_size = generated_codes_BxTxC.shape[0]
        seq_length = generated_codes_BxTxC.shape[1]  # Max length in the batch provided
        delay_pattern = self.config.delay_pattern
        audio_pad_value = self.generation_config.pad_token_id
        max_delay_pattern = max(delay_pattern)

        if seq_length == 0:  # Handle empty input
            empty_audio = np.array([], dtype=np.float32)
            return [empty_audio] * batch_size if batch_size > 1 else empty_audio

        revert_precomp = build_revert_indices(
            B=batch_size,
            T=seq_length,  # Use actual seq_length of the input tensor
            C=num_channels,
            delay_pattern=delay_pattern,
        )

        codebook = revert_audio_delay(
            audio_BxTxC=generated_codes_BxTxC,
            pad_value=audio_pad_value,  # Ensure pad tokens are handled if present
            precomp=revert_precomp,
            T=seq_length,  # Max T of the input
        )
        if codebook.shape[1] <= 0:
            empty_audio = np.array([], dtype=np.float32)
            return [empty_audio] * batch_size if batch_size > 1 else empty_audio

        min_valid_index = 0
        max_valid_index = 1023

        invalid_mask = (codebook < min_valid_index) | (codebook > max_valid_index)
        codebook[invalid_mask] = 0

        audio_output_tensor_batch = decode(self.dac_model, codebook.transpose(1, 2))

        if batch_size > 1:
            return [audio_output_tensor_batch[i].cpu().numpy() for i in range(batch_size)]
        else:
            return audio_output_tensor_batch[0].cpu().numpy()
