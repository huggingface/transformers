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

from collections.abc import Callable
from typing import Any

import torch

from ...generation import GenerationConfig, GenerationMixin, GenerationMode, GenerationState
from ...generation.logits_process import (
    DiaClassifierFreeGuidanceLogitsProcessor,
    DiaEOSChannelFilterLogitsProcessor,
    DiaEOSDelayPatternLogitsProcessor,
    LogitsProcessor,
    LogitsProcessorList,
)
from ...generation.utils import GenerateOutput
from ...utils import logging


logger = logging.get_logger(__name__)


class DiaGenerationMixin(GenerationMixin):
    """
    Dia generates `num_channels` codebook tokens per step. The decoding loop runs on the channels flattened into the
    batch, `(batch_size * num_channels, seq_len)`, so that each channel finishes (and is padded) on its own as the
    EOS delay pattern reaches it; `prepare_inputs_for_generation` gives the model `(batch_size, seq_len,
    num_channels)` and `_build_generate_output` returns that layout, with the prompt's delay mask re-applied.
    Classifier-free guidance batches the conditional and unconditional branches: the encoder sees both prompts, the
    decoder inputs are duplicated per step, and [`DiaClassifierFreeGuidanceLogitsProcessor`] merges the two halves.
    """

    _supported_generation_modes = [GenerationMode.GREEDY_SEARCH, GenerationMode.SAMPLE]

    @staticmethod
    def _uses_classifier_free_guidance(generation_config: GenerationConfig) -> bool:
        return generation_config.guidance_scale is not None and generation_config.guidance_scale != 1

    def _get_classifier_free_guidance_processor(
        self,
        generation_config: GenerationConfig,
        model_kwargs: dict[str, Any] | None,
        negative_prompt_ids: torch.Tensor | None,
        negative_prompt_attention_mask: torch.Tensor | None,
    ) -> LogitsProcessor | None:
        # The unconditional branch runs in the same forward as the conditional one (see
        # `_prepare_encoder_decoder_kwargs_for_generation`), so the batched Dia processor replaces the default one.
        if not self._uses_classifier_free_guidance(generation_config):
            return None
        return DiaClassifierFreeGuidanceLogitsProcessor(
            guidance_scale=generation_config.guidance_scale, guidance_top_k=generation_config.top_k
        )

    def _get_logits_processor(
        self,
        generation_config: GenerationConfig,
        input_ids_seq_length: int | None = None,
        encoder_input_ids: torch.LongTensor | None = None,
        prefix_allowed_tokens_fn: Callable[[int, torch.Tensor], list[int]] | None = None,
        logits_processor: LogitsProcessorList | None = None,
        device: str | None = None,
        model_kwargs: dict[str, Any] | None = None,
        negative_prompt_ids: torch.Tensor | None = None,
        negative_prompt_attention_mask: torch.Tensor | None = None,
    ) -> LogitsProcessorList:
        # Only channel 0 may emit EOS, and only when it is the top logit: runs before the sampling warpers
        custom_processors = LogitsProcessorList(
            [
                DiaEOSChannelFilterLogitsProcessor(
                    num_channels=len(self.config.delay_pattern),
                    eos_token_id=self.config.decoder_config.eos_token_id,
                )
            ]
        )
        if logits_processor is not None:
            custom_processors.extend(logits_processor)

        processors = super()._get_logits_processor(
            generation_config=generation_config,
            input_ids_seq_length=input_ids_seq_length,
            encoder_input_ids=encoder_input_ids,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            logits_processor=custom_processors,
            device=device,
            model_kwargs=model_kwargs,
            negative_prompt_ids=negative_prompt_ids,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
        )

        # Must run last: forces EOS in the delayed channels once channel 0 emitted it
        processors.append(
            DiaEOSDelayPatternLogitsProcessor(
                delay_pattern=self.config.delay_pattern,
                eos_token_id=self.config.decoder_config.eos_token_id,
                max_generation_len=generation_config.max_length,
                device=device,
            )
        )
        return processors

    def _prepare_generation_config(
        self, generation_config: GenerationConfig | None, **kwargs: Any
    ) -> tuple[GenerationConfig, dict]:
        generation_config, model_kwargs = super()._prepare_generation_config(generation_config, **kwargs)

        if generation_config.temperature is not None and generation_config.temperature < 1.0:
            logger.warning_once(
                f"temperature < 1.0 is not supported for Dia; clamping to 1.0 (got {generation_config.temperature})"
            )
            generation_config.temperature = 1.0
        if generation_config.num_return_sequences > 1:
            raise ValueError("`num_return_sequences>1` is incompatible with Dia.")
        # We allow generation up to max length + max delay pattern
        # (will revert back to max length after generation)
        generation_config.max_length += max(self.config.delay_pattern)

        return generation_config, model_kwargs

    def _prepare_encoder_decoder_kwargs_for_generation(
        self,
        inputs_tensor: torch.Tensor,
        model_kwargs: dict[str, Any],
        model_input_name: str | None,
        generation_config: GenerationConfig,
    ) -> dict[str, Any]:
        # CFG: the unconditional branch is an all-zero prompt, batched after the conditional one
        if self._uses_classifier_free_guidance(generation_config):
            inputs_tensor = torch.cat([inputs_tensor, torch.zeros_like(inputs_tensor)], dim=0)
            if model_kwargs.get("attention_mask") is not None:
                model_kwargs["attention_mask"] = model_kwargs["attention_mask"].repeat(2, 1)
        return super()._prepare_encoder_decoder_kwargs_for_generation(
            inputs_tensor, model_kwargs, model_input_name, generation_config
        )

    def _prepare_cache_for_generation(
        self,
        generation_config: GenerationConfig,
        model_kwargs: dict,
        generation_mode: GenerationMode,
        batch_size: int,
        max_cache_length: int,
        max_cache_length_attr: str = "_previous_max_cache_length",
    ) -> bool:
        # The decoder runs both CFG branches in one batch
        if self._uses_classifier_free_guidance(generation_config):
            batch_size *= 2
        return super()._prepare_cache_for_generation(
            generation_config, model_kwargs, generation_mode, batch_size, max_cache_length, max_cache_length_attr
        )

    def _prepare_decoder_input_ids_for_generation(
        self,
        batch_size: int,
        model_input_name: str,
        model_kwargs: dict[str, torch.Tensor],
        decoder_start_token_id: torch.Tensor,
        device: torch.device | None = None,
    ) -> tuple[torch.LongTensor, dict[str, torch.Tensor]]:
        """Prepares `decoder_input_ids` for generation with encoder-decoder models"""
        # 1. Check whether the user has defined `decoder_input_ids` and `decoder_attention_mask`; if not error out
        decoder_input_ids = decoder_attention_mask = None
        if model_kwargs is not None and "decoder_input_ids" in model_kwargs:
            decoder_input_ids = model_kwargs.pop("decoder_input_ids")
        if model_kwargs is not None and "decoder_attention_mask" in model_kwargs:
            decoder_attention_mask = model_kwargs.pop("decoder_attention_mask")

        # We allow generating without preparation (no proper delay) but discourage it
        if decoder_input_ids is None or decoder_attention_mask is None:
            logger.warning_once(
                "In order to generate with Dia, we need the processed audio input: Got `decoder_input_ids`:"
                f" {decoder_input_ids is not None} and got `decoder_attention_mask`={decoder_attention_mask is not None}."
                f" This can be achieved via the [`DiaProcessor`] but now defaulting to non-delayed generation."
            )

            # `batch_size` is the conditional batch size: the CFG doubling only happens on the encoder side
            num_channels = self.config.decoder_config.num_channels
            if decoder_input_ids is None:
                decoder_input_ids = torch.full(
                    (batch_size, 1, num_channels), decoder_start_token_id, dtype=torch.long, device=device
                )

            decoder_attention_mask = torch.ones(
                size=(batch_size, decoder_input_ids.shape[1]), dtype=torch.long, device=device
            )

        # 2. Determine the valid input and what works as mask within the input
        delay_mask = decoder_input_ids.long()
        valid_input_size = (
            decoder_input_ids.shape[1]
            - (decoder_input_ids[:, :, 0] == self.config.decoder_config.pad_token_id).sum(dim=-1).max()
        )
        # The decoding loop runs on the channels flattened into the batch: (batch_size * num_channels, seq_len)
        decoder_input_ids = delay_mask[:, :valid_input_size].transpose(1, 2).reshape(-1, valid_input_size)
        decoder_attention_mask = decoder_attention_mask[:, :valid_input_size].long()

        # 3. Overwrite into model kwargs
        model_kwargs["decoder_attention_mask"] = decoder_attention_mask
        model_kwargs["decoder_delay_mask"] = delay_mask

        return decoder_input_ids, model_kwargs

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        encoder_outputs: Any = None,  # Using this to easily get the batch size
        decoder_delay_mask: torch.Tensor | None = None,
        is_first_iteration: bool | None = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        # Reshape decoder input_ids to 3D to be compile friendly and to fit the expected model input shape
        num_channels = self.config.decoder_config.num_channels
        # CFG batched the unconditional branch after the conditional one on the encoder side, so the encoder batch is
        # twice the decoder batch (`input_ids.shape[0] // num_channels`)
        uses_cfg = encoder_outputs[0].shape[0] * num_channels == 2 * input_ids.shape[0]
        batch_size = encoder_outputs[0].shape[0] // 2 if uses_cfg else encoder_outputs[0].shape[0]
        # (batch_size * num_channels, seq_len) -> (batch_size, seq_len, num_channels)
        input_ids = input_ids.reshape(batch_size, num_channels, -1).transpose(1, 2)

        # Base method handles most things except CFG and the delay pattern mask
        model_inputs = super().prepare_inputs_for_generation(input_ids, encoder_outputs=encoder_outputs, **kwargs)

        # Post processing for CFG and overwriting via delay pattern mask
        # 1. Delay pattern mask -- force tokens if not allowed to predict (!= pad_token in mask)
        model_inputs["decoder_input_ids"] = self.apply_delay_mask(
            input_ids, self.config.decoder_config.pad_token_id, decoder_delay_mask
        )

        # Depending on cache usage we need to pass all or just one
        if model_inputs.get("use_cache", False) and not is_first_iteration:
            model_inputs["decoder_input_ids"] = model_inputs["decoder_input_ids"][:, -1, :][:, None, :]

        # Be compile friendly
        model_inputs["decoder_input_ids"] = model_inputs["decoder_input_ids"].contiguous()

        # 2. Apply CFG duplication if needed
        if uses_cfg:
            for key in ["decoder_input_ids", "decoder_attention_mask", "decoder_position_ids"]:
                if model_inputs.get(key, None) is not None:
                    # double first dimension and keep everything else the same
                    repeat_pattern = tuple([2] + [1] * (model_inputs[key].ndim - 1))
                    model_inputs[key] = model_inputs[key].repeat(*repeat_pattern)

        return model_inputs

    @staticmethod
    def apply_delay_mask(input_ids: torch.Tensor, pad_id: int, delay_mask: torch.Tensor | None) -> torch.Tensor:
        if delay_mask is None:
            return input_ids

        mask_len = min(input_ids.shape[1], delay_mask.shape[1])
        valid_mask = delay_mask[:, :mask_len, :]
        valid_input = input_ids[:, :mask_len, :]

        # Overwrite the respective parts of the input
        input_ids[:, :mask_len, :] = torch.where(valid_mask == pad_id, valid_input, valid_mask)

        return input_ids

    def _build_generate_output(
        self,
        sequences: torch.LongTensor,
        state: GenerationState,
        generation_config: GenerationConfig,
        model_kwargs: dict[str, Any],
        **kwargs: Any,
    ) -> GenerateOutput | torch.LongTensor:
        output = super()._build_generate_output(sequences, state, generation_config, model_kwargs, **kwargs)

        # (batch_size * num_channels, seq_len) -> (batch_size, seq_len, num_channels), delay mask re-applied
        num_channels = self.config.decoder_config.num_channels
        sequences = sequences.reshape(-1, num_channels, sequences.shape[-1]).transpose(1, 2)
        sequences = self.apply_delay_mask(
            sequences, self.config.decoder_config.pad_token_id, model_kwargs.get("decoder_delay_mask")
        )

        if isinstance(output, torch.Tensor):
            return sequences
        output.sequences = sequences
        return output
