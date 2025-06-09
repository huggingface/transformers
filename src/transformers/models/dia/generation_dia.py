# coding=utf-8
# Copyright 2025 The Nari Labs and HuggingFace Inc. team. All rights reserved.
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

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch

from ...generation.logits_process import (
    DiaClassifierFreeGuidanceLogitsProcessor,
    DiaEOSChannelFilterLogitsProcessor,
    DiaEOSDelayPatternLogitsProcessor,
    LogitsProcessorList,
)
from ...generation.stopping_criteria import StoppingCriteriaList
from ...generation.streamers import BaseStreamer
from ...generation.utils import GenerateOutput, GenerationConfig, GenerationMixin
from ...modeling_utils import PreTrainedModel


class DiaGenerationMixin(GenerationMixin):
    # Indicates CFG which needs preparation to be properly handled by repeats
    _uses_cfg = None

    def _get_logits_processor(
        self,
        generation_config: GenerationConfig,
        input_ids_seq_length: Optional[int] = None,
        encoder_input_ids: torch.LongTensor = None,
        prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        device: Optional[str] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
        negative_prompt_ids: Optional[torch.Tensor] = None,
        negative_prompt_attention_mask: Optional[torch.Tensor] = None,
    ) -> LogitsProcessorList:
        custom_processors = LogitsProcessorList()

        cfg_processor = None
        if generation_config.guidance_scale is not None and generation_config.guidance_scale != 1:
            cfg_processor = DiaClassifierFreeGuidanceLogitsProcessor(
                guidance_scale=generation_config.guidance_scale,
                guidance_top_k=generation_config.top_k,
            )
            # Avoid adding CFG again
            generation_config.guidance_scale = None

        custom_processors.append(
            DiaEOSChannelFilterLogitsProcessor(
                num_channels=len(self.config.delay_pattern),
                eos_token_id=generation_config.eos_token_id,
            )
        )

        custom_processors.append(
            DiaEOSDelayPatternLogitsProcessor(
                delay_pattern=self.config.delay_pattern,
                eos_token_id=generation_config.eos_token_id,
                max_generation_len=generation_config.max_length,
                device=device,
            )
        )

        merged_processors = super()._get_logits_processor(
            generation_config=generation_config,
            input_ids_seq_length=input_ids_seq_length,
            encoder_input_ids=encoder_input_ids,
            prefix_allowed_tokens_fn=None,
            logits_processor=custom_processors,
            device=device,
            model_kwargs=model_kwargs,
            negative_prompt_ids=negative_prompt_ids,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
        )

        # We need to guarantee CFG to be at the first position
        if cfg_processor is not None:
            merged_processors.insert(0, cfg_processor)

        return merged_processors

    def _prepare_generation_config(
        self, generation_config: Optional[GenerationConfig], use_model_defaults: Optional[bool] = None, **kwargs: Dict
    ) -> Tuple[GenerationConfig, Dict]:
        generation_config, model_kwargs = super()._prepare_generation_config(
            generation_config, use_model_defaults, **kwargs
        )

        # We allow generation up to max length + max delay pattern
        # (will revert back to max length after generation)
        generation_config.max_length += max(self.config.delay_pattern)

        # Internal flag to indicate CFG that needs to prepare unconditioned input
        self._uses_cfg = generation_config.guidance_scale is not None and generation_config.guidance_scale != 1

        return generation_config, model_kwargs

    def _prepare_model_inputs(
        self,
        inputs: Optional[torch.Tensor] = None,
        bos_token_id: Optional[torch.Tensor] = None,
        model_kwargs: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[str], Dict[str, torch.Tensor]]:
        inputs, input_name, model_kwargs = super()._prepare_model_inputs(
            inputs=inputs,
            bos_token_id=bos_token_id,
            model_kwargs=model_kwargs,
        )

        # If CFG is requested we fill in the unconditioned parts
        if self._uses_cfg:
            unconditioned_inputs = torch.zeros_like(inputs)
            inputs = torch.cat([inputs, unconditioned_inputs], dim=0)

            if model_kwargs.get("attention_mask", None) is not None:
                model_kwargs["attention_mask"] = model_kwargs["attention_mask"].repeat(2, 1)

        return inputs, input_name, model_kwargs

    def _prepare_decoder_input_ids_for_generation(
        self,
        batch_size: int,
        model_input_name: str,
        model_kwargs: Dict[str, torch.Tensor],
        decoder_start_token_id: torch.Tensor,
        device: Optional[torch.device] = None,
    ) -> Tuple[torch.LongTensor, Dict[str, torch.Tensor]]:
        """Prepares `decoder_input_ids` for generation with encoder-decoder models"""
        # TODO: check for correctness - in the case audio is provided

        # 1. Check whether the user has defined `decoder_input_ids` and `decoder_attention_mask` manually.
        decoder_input_ids = decoder_attention_mask = None
        if model_kwargs is not None and "decoder_input_ids" in model_kwargs:
            decoder_input_ids = model_kwargs.pop("decoder_input_ids")
        if model_kwargs is not None and "decoder_attention_mask" in model_kwargs:
            decoder_attention_mask = model_kwargs.pop("decoder_attention_mask")

        # 2. Prepare audio into being shifted according to the delay patterns
        # Some default values
        num_channels = self.config.decoder_config.num_channels
        delay_pattern = self.config.delay_pattern
        real_batch_size = batch_size // 2 if self._uses_cfg else batch_size

        # Default to all bos tokens if nothing is provided
        if decoder_input_ids is None:
            decoder_input_ids = torch.full(
                (real_batch_size, 1, num_channels), decoder_start_token_id, dtype=torch.long, device=device
            )

        # Get all audio and padding lengths
        if decoder_attention_mask is not None:
            # DAC in hf only works with right padding, we already know that this needs to be flipped
            decoder_attention_mask = decoder_attention_mask.flip(dims=[1])
            padding_lens = decoder_attention_mask.shape[-1] - decoder_attention_mask.sum(dim=-1)
            audio_lens = decoder_attention_mask.shape[-1] - padding_lens
        else:
            decoder_attention_mask = torch.ones(
                size=(real_batch_size, decoder_input_ids.shape[1]), dtype=torch.long, device=device
            )
            padding_lens = [0] * real_batch_size
            audio_lens = [decoder_input_ids.shape[1]] * real_batch_size
        # +1 for bos
        max_seq_len = max(audio_lens) + max(delay_pattern) + 1

        # 3. Create delayed batch (and hence also the mask)
        prefill = torch.full(
            (real_batch_size, max_seq_len, num_channels),
            fill_value=-1,
            dtype=torch.int,
            device=device,
        )

        max_audio_len = 0
        for i in range(real_batch_size):
            padding_size = padding_lens[i]
            prefill[i, : padding_size + 1, :] = decoder_start_token_id

            # Right padded due to DAC
            prompt = decoder_input_ids[i, : audio_lens[i], ...]

            # Second condition in case no audio has been given
            if prompt is not None and not (prompt == decoder_start_token_id).any():
                prompt = prompt.to(dtype=torch.int)
                prefill[i, padding_size + 1 : prompt.shape[0] + 1, :] = prompt
                max_audio_len = max(max_audio_len, prompt.shape[0])

        delay_precomp = self.build_delay_indices(
            B=real_batch_size,
            T=max_seq_len,
            C=num_channels,
            delay_pattern=delay_pattern,
        )

        delayed_batch = self.apply_audio_delay(
            audio_BxTxC=prefill,
            pad_value=-1,
            bos_value=decoder_start_token_id,
            precomp=delay_precomp,
        )

        # 4. Overwrite and convert to 2D
        decoder_input_ids = delayed_batch[:, : max_audio_len + 1, :].reshape(real_batch_size * num_channels, -1)
        model_kwargs["decoder_attention_mask"] = decoder_attention_mask[:, : max_audio_len + 1]
        model_kwargs["decoder_delay_mask"] = delayed_batch

        return decoder_input_ids, model_kwargs

    def prepare_inputs_for_generation(
        self,
        input_ids,
        encoder_outputs=None,  # Using this to easily get the batch size
        decoder_delay_mask=None,
        **kwargs,
    ):
        # Base method handles most things except CFG and the delay pattern mask
        model_inputs = super().prepare_inputs_for_generation(input_ids, encoder_outputs=encoder_outputs, **kwargs)

        # Post processing for CFG and overwriting via delay pattern mask
        # 1. Reshape (bsz * channels, seq_len) to (bsz, seq_len, channels)
        batch_size = encoder_outputs[0].shape[0] // 2 if self._uses_cfg else encoder_outputs[0].shape[0]
        model_inputs["decoder_input_ids"] = model_inputs["decoder_input_ids"].reshape(
            batch_size, -1, self.config.decoder_config.num_channels
        )

        # 2. Delay pattern mask -- force tokens if not allowed to predict (!= -1 in mask)
        model_inputs["decoder_input_ids"] = self.apply_delay_mask(
            model_inputs["decoder_input_ids"], decoder_delay_mask
        )

        # 3. Apply CFG duplication if needed
        if self._uses_cfg:
            for key in ["decoder_input_ids", "decoder_attention_mask", "decoder_position_ids"]:
                if model_inputs.get(key, None) is not None:
                    # double first dimension and keep everything else the same
                    repeat_pattern = tuple([2] + [1] * (model_inputs[key].ndim - 1))
                    model_inputs[key] = model_inputs[key].repeat(*repeat_pattern)

        return model_inputs

    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        generation_config: Optional[GenerationConfig] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]] = None,
        synced_gpus: Optional[bool] = None,
        assistant_model: Optional["PreTrainedModel"] = None,
        streamer: Optional["BaseStreamer"] = None,
        negative_prompt_ids: Optional[torch.Tensor] = None,
        negative_prompt_attention_mask: Optional[torch.Tensor] = None,
        use_model_defaults: Optional[bool] = None,
        custom_generate: Optional[str] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        decoder_input_ids = kwargs.get("decoder_input_ids", None)
        decoder_input_length = decoder_input_ids.shape[1] if decoder_input_ids is not None else 0

        # TODO: find a way for generation mode to be forced to greedy / sample

        output = super().generate(
            inputs=inputs,
            generation_config=generation_config,
            logits_processor=logits_processor,
            stopping_criteria=stopping_criteria,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            synced_gpus=synced_gpus,
            assistant_model=assistant_model,
            streamer=streamer,
            negative_prompt_ids=negative_prompt_ids,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
            use_model_defaults=use_model_defaults,
            custom_generate=custom_generate,
            **kwargs,
        )

        return_dict_in_generate = not isinstance(output, torch.Tensor)

        if return_dict_in_generate:
            output_sequences = output.sequences
        else:
            output_sequences = output

        # TODO: postprocessing needs to be rewritten as well

        # 1 for bos token
        output_sequences = output_sequences[:, 1 + decoder_input_length :]
        delay_pattern = torch.tensor(self.config.delay_pattern, dtype=torch.long, device=output_sequences.device)
        max_delay_pattern = delay_pattern.max().item()

        delay_precomp = self.build_revert_indices(
            B=output_sequences.shape[0],
            T=output_sequences.shape[1],
            C=self.config.decoder_config.num_channels,
            delay_pattern=delay_pattern,
        )
        output_sequences = self.revert_audio_delay(
            output_sequences,
            pad_value=self.config.pad_token_id,
            precomp=delay_precomp,
            T=output_sequences.shape[1],
        )

        # see `DiaEosTokenCriteria` why we need to +1
        output_sequences = output_sequences[:, -max_delay_pattern + 1 :]

        if return_dict_in_generate:
            output.sequences = output_sequences
        else:
            output = output_sequences

        return output

    @staticmethod
    def apply_delay_mask(input_ids: torch.Tensor, delay_mask: Optional[torch.Tensor]) -> torch.Tensor:
        if delay_mask is None:
            return input_ids

        mask_len = min(input_ids.shape[1], delay_mask.shape[1])
        valid_mask = delay_mask[:, :mask_len, :]
        valid_input = input_ids[:, :mask_len, :]

        # Overwrite the respective parts of the input
        input_ids[:, :mask_len, :] = torch.where(valid_mask == -1, valid_input, valid_mask)

        return input_ids

    @staticmethod
    def build_delay_indices(B: int, T: int, C: int, delay_pattern: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Precompute (t_idx_BxTxC, indices_BTCx3) so that out[t, c] = in[t - delay[c], c].
        Negative t_idx => BOS; t_idx >= T => PAD.
        """
        delay_arr = torch.tensor(delay_pattern, dtype=torch.int32)

        t_idx_BxT = torch.broadcast_to(
            torch.arange(T, dtype=torch.int32)[None, :],
            [B, T],
        )
        t_idx_BxTx1 = t_idx_BxT[..., None]
        t_idx_BxTxC = t_idx_BxTx1 - delay_arr.view(1, 1, C)

        b_idx_BxTxC = torch.broadcast_to(
            torch.arange(B, dtype=torch.int32).view(B, 1, 1),
            [B, T, C],
        )
        c_idx_BxTxC = torch.broadcast_to(
            torch.arange(C, dtype=torch.int32).view(1, 1, C),
            [B, T, C],
        )

        # We must clamp time indices to [0..T-1] so gather_nd equivalent won't fail
        t_clamped_BxTxC = torch.clamp(t_idx_BxTxC, 0, T - 1)

        indices_BTCx3 = torch.stack(
            [
                b_idx_BxTxC.reshape(-1),
                t_clamped_BxTxC.reshape(-1),
                c_idx_BxTxC.reshape(-1),
            ],
            dim=1,
        ).long()  # Ensure indices are long type for indexing

        return t_idx_BxTxC, indices_BTCx3

    @staticmethod
    def apply_audio_delay(
        audio_BxTxC: torch.Tensor,
        pad_value: int,
        bos_value: int,
        precomp: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """
        Applies the delay pattern to batched audio tokens using precomputed indices,
        inserting BOS where t_idx < 0 and PAD where t_idx >= T.

        Args:
            audio_BxTxC: [B, T, C] int16 audio tokens (or int32/float)
            pad_value: the padding token
            bos_value: the BOS token
            precomp:  (t_idx_BxTxC, indices_BTCx3) from build_delay_indices

        Returns:
            result_BxTxC: [B, T, C] delayed audio tokens
        """
        device = audio_BxTxC.device  # Get device from input tensor
        t_idx_BxTxC, indices_BTCx3 = precomp
        t_idx_BxTxC = t_idx_BxTxC.to(device)  # Move precomputed indices to device
        indices_BTCx3 = indices_BTCx3.to(device)

        # Equivalent of tf.gather_nd using advanced indexing
        # Ensure indices are long type if not already (build_delay_indices should handle this)
        gathered_flat = audio_BxTxC[indices_BTCx3[:, 0], indices_BTCx3[:, 1], indices_BTCx3[:, 2]]
        gathered_BxTxC = gathered_flat.view(audio_BxTxC.shape)

        # Create masks on the correct device
        mask_bos = t_idx_BxTxC < 0  # => place bos_value
        mask_pad = t_idx_BxTxC >= audio_BxTxC.shape[1]  # => place pad_value

        # Create scalar tensors on the correct device
        bos_tensor = torch.tensor(bos_value, dtype=audio_BxTxC.dtype, device=device)
        pad_tensor = torch.tensor(pad_value, dtype=audio_BxTxC.dtype, device=device)

        # If mask_bos, BOS; else if mask_pad, PAD; else original gather
        # All tensors should now be on the same device
        result_BxTxC = torch.where(mask_bos, bos_tensor, torch.where(mask_pad, pad_tensor, gathered_BxTxC))

        return result_BxTxC

    @staticmethod
    def build_revert_indices(
        B: int, T: int, C: int, delay_pattern: List[int], padding_sizes: List[int]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Precompute indices for the revert operation using PyTorch.

        Returns:
            A tuple (t_idx_BxTxC, indices_BTCx3) where:
                - t_idx_BxTxC is a tensor of shape [B, T, C] computed as time indices plus the delay.
                - indices_BTCx3 is a tensor of shape [B*T*C, 3] used for gathering, computed from:
                    batch indices, clamped time indices, and channel indices.
        """
        # Use default device unless specified otherwise; assumes inputs might define device later
        device = None  # Or determine dynamically if needed, e.g., from a model parameter

        # TODO: remove this when everything is properly adjusted
        if padding_sizes is None:
            padding_sizes = [0] * len(delay_pattern)

        # We shift the delays in order to account for left padding (if needed)
        delay_arr = torch.tensor(delay_pattern, dtype=torch.int32, device=device)
        delay_arr = (delay_arr[None, :] + torch.tensor(padding_sizes, dtype=torch.int32)[:, None])[:, None, :]

        t_idx_BT1 = torch.broadcast_to(torch.arange(T, device=device).unsqueeze(0), [B, T])
        t_idx_BT1 = t_idx_BT1.unsqueeze(-1)

        t_idx_BxTxC = torch.minimum(
            t_idx_BT1 + delay_arr,
            torch.tensor(T - 1, device=device),
        )
        b_idx_BxTxC = torch.broadcast_to(torch.arange(B, device=device).view(B, 1, 1), [B, T, C])
        c_idx_BxTxC = torch.broadcast_to(torch.arange(C, device=device).view(1, 1, C), [B, T, C])

        indices_BTCx3 = torch.stack(
            [
                b_idx_BxTxC.reshape(-1),
                t_idx_BxTxC.reshape(-1),
                c_idx_BxTxC.reshape(-1),
            ],
            axis=1,
        ).long()  # Ensure indices are long type

        return t_idx_BxTxC, indices_BTCx3

    @staticmethod
    def revert_audio_delay(
        audio_BxTxC: torch.Tensor,
        pad_value: int,
        precomp: Tuple[torch.Tensor, torch.Tensor],
        T: int,
    ) -> torch.Tensor:
        """
        Reverts a delay pattern from batched audio tokens using precomputed indices (PyTorch version).

        Args:
            audio_BxTxC: Input delayed audio tensor
            pad_value: Padding value for out-of-bounds indices
            precomp: Precomputed revert indices tuple containing:
                - t_idx_BxTxC: Time offset indices tensor
                - indices_BTCx3: Gather indices tensor for original audio
            T: Original sequence length before padding

        Returns:
            Reverted audio tensor with same shape as input
        """
        t_idx_BxTxC, indices_BTCx3 = precomp
        device = audio_BxTxC.device  # Get device from input tensor

        # Move precomputed indices to the same device as audio_BxTxC if they aren't already
        t_idx_BxTxC = t_idx_BxTxC.to(device)
        indices_BTCx3 = indices_BTCx3.to(device)

        # Using PyTorch advanced indexing (equivalent to tf.gather_nd or np equivalent)
        gathered_flat = audio_BxTxC[indices_BTCx3[:, 0], indices_BTCx3[:, 1], indices_BTCx3[:, 2]]
        gathered_BxTxC = gathered_flat.view(audio_BxTxC.size())  # Use .size() for robust reshaping

        # Create pad_tensor on the correct device
        pad_tensor = torch.tensor(pad_value, dtype=audio_BxTxC.dtype, device=device)
        # Create T tensor on the correct device for comparison
        T_tensor = torch.tensor(T, device=device)

        result_BxTxC = torch.where(
            t_idx_BxTxC >= T_tensor, pad_tensor, gathered_BxTxC
        )  # Changed np.where to torch.where

        return result_BxTxC
