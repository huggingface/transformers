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

from typing import Any, Callable, Optional, Union

import torch
import torch.distributed as dist

from ...generation.logits_process import (
    DiaClassifierFreeGuidanceLogitsProcessor,
    DiaEOSChannelFilterLogitsProcessor,
    DiaEOSDelayPatternLogitsProcessor,
    LogitsProcessorList,
    TemperatureLogitsWarper,
)
from ...generation.stopping_criteria import StoppingCriteriaList
from ...generation.streamers import BaseStreamer
from ...generation.utils import GenerateOutput, GenerationConfig, GenerationMixin, GenerationMode
from ...integrations.deepspeed import is_deepspeed_zero3_enabled
from ...integrations.fsdp import is_fsdp_managed_module
from ...modeling_utils import PreTrainedModel
from ...utils import logging


logger = logging.get_logger(__name__)


class DiaGenerationMixin(GenerationMixin):
    # Indicates CFG which needs preparation to be properly handled by repeats
    _uses_cfg = None

    def _get_logits_processor(
        self,
        generation_config: GenerationConfig,
        input_ids_seq_length: Optional[int] = None,
        encoder_input_ids: Optional[torch.LongTensor] = None,
        prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], list[int]]] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        device: Optional[str] = None,
        model_kwargs: Optional[dict[str, Any]] = None,
        negative_prompt_ids: Optional[torch.Tensor] = None,
        negative_prompt_attention_mask: Optional[torch.Tensor] = None,
    ) -> LogitsProcessorList:
        # Need either custom order or custom processor instead
        # (Temporarily disabling those for the super function)
        original_guidance_scale = generation_config.guidance_scale
        original_temperature = generation_config.temperature
        generation_config.guidance_scale = None
        generation_config.temperature = None

        # Get base processors and those we can integrate easily
        custom_processors = LogitsProcessorList()

        if original_temperature is not None and original_temperature != 1.0:
            custom_processors.append(TemperatureLogitsWarper(original_temperature))

        custom_processors.append(
            DiaEOSChannelFilterLogitsProcessor(
                num_channels=len(self.config.delay_pattern),
                eos_token_id=self.config.eos_token_id,
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

        # Custom processors we need at specific positions
        if original_guidance_scale is not None and original_guidance_scale != 1:
            cfg_processor = DiaClassifierFreeGuidanceLogitsProcessor(
                guidance_scale=original_guidance_scale,
                guidance_top_k=generation_config.top_k,
            )
            merged_processors.insert(0, cfg_processor)

        merged_processors.append(
            DiaEOSDelayPatternLogitsProcessor(
                delay_pattern=self.config.delay_pattern,
                eos_token_id=self.config.eos_token_id,
                max_generation_len=generation_config.max_length,
                device=device,
            )
        )

        # Enable temporarily disabled values back
        generation_config.guidance_scale = original_guidance_scale
        generation_config.temperature = original_temperature

        return merged_processors

    def _prepare_generation_config(
        self, generation_config: Optional[GenerationConfig], use_model_defaults: Optional[bool] = None, **kwargs: Any
    ) -> tuple[GenerationConfig, dict]:
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
        model_kwargs: Optional[dict[str, torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, Optional[str], dict[str, torch.Tensor]]:
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
        model_kwargs: dict[str, torch.Tensor],
        decoder_start_token_id: torch.Tensor,
        device: Optional[torch.device] = None,
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

            num_channels = self.config.decoder_config.num_channels
            real_batch_size = batch_size // 2 if self._uses_cfg else batch_size

            if decoder_input_ids is None:
                decoder_input_ids = torch.full(
                    (real_batch_size, 1, num_channels), decoder_start_token_id, dtype=torch.long, device=device
                )

            decoder_attention_mask = torch.ones(
                size=(real_batch_size, decoder_input_ids.shape[1]), dtype=torch.long, device=device
            )

        # 2. Determine the valid input and what works as mask within the input
        delay_mask = decoder_input_ids.long()
        valid_input_size = (
            decoder_input_ids.shape[1] - (decoder_input_ids[:, :, 0] == self.config.pad_token_id).sum(dim=-1).max()
        )
        decoder_input_ids = delay_mask[:, :valid_input_size].transpose(1, 2).long()
        decoder_attention_mask = decoder_attention_mask[:, :valid_input_size].long()

        # 3. Overwrite into model kwargs
        model_kwargs["decoder_attention_mask"] = decoder_attention_mask
        model_kwargs["decoder_delay_mask"] = delay_mask

        return decoder_input_ids, model_kwargs

    def prepare_inputs_for_generation(
        self,
        input_ids,
        encoder_outputs=None,  # Using this to easily get the batch size
        decoder_delay_mask=None,
        **kwargs,
    ):
        # Reshape decoder input_ids to 3D to be compile friendly and to fit the expected model input shape
        batch_size = encoder_outputs[0].shape[0] // 2 if self._uses_cfg else encoder_outputs[0].shape[0]
        input_ids = input_ids.reshape(batch_size, self.config.decoder_config.num_channels, -1).transpose(1, 2)

        # Base method handles most things except CFG and the delay pattern mask
        model_inputs = super().prepare_inputs_for_generation(input_ids, encoder_outputs=encoder_outputs, **kwargs)

        # Post processing for CFG and overwriting via delay pattern mask
        # 1. Delay pattern mask -- force tokens if not allowed to predict (!= pad_token in mask)
        model_inputs["decoder_input_ids"] = self.apply_delay_mask(
            input_ids, self.config.pad_token_id, decoder_delay_mask
        )

        # Depending on cache usage we need to pass all or just one
        if model_inputs.get("use_cache", False) and model_inputs["cache_position"][0] > 0:
            model_inputs["decoder_input_ids"] = model_inputs["decoder_input_ids"][:, -1, :][:, None, :]

        # Be compile friendly
        model_inputs["decoder_input_ids"] = model_inputs["decoder_input_ids"].contiguous()

        # 2. Apply CFG duplication if needed
        if self._uses_cfg:
            for key in ["decoder_input_ids", "decoder_attention_mask", "decoder_position_ids"]:
                if model_inputs.get(key, None) is not None:
                    # double first dimension and keep everything else the same
                    repeat_pattern = tuple([2] + [1] * (model_inputs[key].ndim - 1))
                    model_inputs[key] = model_inputs[key].repeat(*repeat_pattern)

        return model_inputs

    @staticmethod
    def apply_delay_mask(input_ids: torch.Tensor, pad_id: int, delay_mask: Optional[torch.Tensor]) -> torch.Tensor:
        if delay_mask is None:
            return input_ids

        mask_len = min(input_ids.shape[1], delay_mask.shape[1])
        valid_mask = delay_mask[:, :mask_len, :]
        valid_input = input_ids[:, :mask_len, :]

        # Overwrite the respective parts of the input
        input_ids[:, :mask_len, :] = torch.where(valid_mask == pad_id, valid_input, valid_mask)

        return input_ids

    def _main_generate_loop(
        self,
        inputs: Optional[torch.Tensor] = None,
        generation_config: Optional[GenerationConfig] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], list[int]]] = None,
        synced_gpus: Optional[bool] = None,
        assistant_model: Optional["PreTrainedModel"] = None,
        streamer: Optional["BaseStreamer"] = None,
        negative_prompt_ids: Optional[torch.Tensor] = None,
        negative_prompt_attention_mask: Optional[torch.Tensor] = None,
        use_model_defaults: Optional[bool] = None,
        custom_generate: Optional[str] = None,
        **kwargs,
    ):
        # ********** mostly taken from main generate function up to calling the different methods (see NOTE) **********
        # 1. Handle `generation_config` and kwargs that might update it, and validate the `.generate()` call
        generation_mode_kwargs = self._extract_generation_mode_kwargs(
            custom_generate,
            kwargs,
            synced_gpus,
            assistant_model,
            streamer,
        )
        generation_config, model_kwargs = self._prepare_generation_config(
            generation_config, use_model_defaults, **kwargs
        )
        generation_mode = generation_config.get_generation_mode(assistant_model)

        self._validate_model_kwargs(model_kwargs.copy())
        self._validate_generation_mode(generation_mode, generation_config, generation_mode_kwargs)

        # 2. Set generation parameters if not already defined
        if synced_gpus is None:
            synced_gpus = (is_deepspeed_zero3_enabled() or is_fsdp_managed_module(self)) and dist.get_world_size() > 1

        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()

        # 3. Define model inputs
        kwargs_has_attention_mask = model_kwargs.get("attention_mask", None) is not None
        inputs_tensor, model_input_name, model_kwargs = self._prepare_model_inputs(
            inputs, generation_config.bos_token_id, model_kwargs
        )
        batch_size = inputs_tensor.shape[0]

        device = inputs_tensor.device
        self._prepare_special_tokens(generation_config, kwargs_has_attention_mask, device=device)

        # 4. Define other model kwargs
        if "encoder_outputs" not in model_kwargs:
            # if model is encoder decoder encoder_outputs are created and added to `model_kwargs`
            model_kwargs = self._prepare_encoder_decoder_kwargs_for_generation(
                inputs_tensor, model_kwargs, model_input_name, generation_config
            )

        # 5. Prepare `input_ids` which will be used for auto-regressive generation
        input_ids, model_kwargs = self._prepare_decoder_input_ids_for_generation(
            batch_size=batch_size,
            model_input_name=model_input_name,
            model_kwargs=model_kwargs,
            decoder_start_token_id=generation_config._decoder_start_token_tensor,
            device=inputs_tensor.device,
        )

        if generation_config.token_healing:
            input_ids = self.heal_tokens(input_ids, generation_mode_kwargs.get("tokenizer"))

        if streamer is not None:
            streamer.put(input_ids.cpu())

        # 6. Prepare `max_length` depending on other stopping criteria.
        # NOTE: incorrect `input_ids.shape[1]` previously
        input_ids_length = input_ids.shape[-1]
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

        # If the model supports `logits_to_keep` in forward(), set it to 1 to avoid computing the whole
        # logit matrix. This can save a lot of memory during the first forward pass. Note that assisted decoding
        # dynamically overrides this value as it can need more than the last token logits
        if self._supports_logits_to_keep() and "logits_to_keep" not in model_kwargs:
            model_kwargs["logits_to_keep"] = 1

        self._validate_generated_length(generation_config, input_ids_length, has_default_max_length)

        # 7. Prepare the cache.
        # - `model_kwargs` may be updated in place with a cache as defined by the parameters in `generation_config`.
        # - different models have a different cache name expected by the model (default = "past_key_values")
        # - `max_length`, prepared above, is used to determine the maximum cache length
        max_cache_length = generation_config.max_length - 1
        if (
            inputs_tensor.shape[1] != input_ids_length
            and model_input_name == "inputs_embeds"
            and not self.config.is_encoder_decoder
        ):
            max_cache_length += inputs_tensor.shape[1]
        self._prepare_cache_for_generation(
            generation_config, model_kwargs, generation_mode, batch_size, max_cache_length
        )

        # 8. prepare logits processors and stopping criteria
        prepared_logits_processor = self._get_logits_processor(
            generation_config=generation_config,
            input_ids_seq_length=input_ids_length,
            encoder_input_ids=inputs_tensor,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            logits_processor=logits_processor,
            device=inputs_tensor.device,
            model_kwargs=model_kwargs,
            negative_prompt_ids=negative_prompt_ids,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
        )
        prepared_stopping_criteria = self._get_stopping_criteria(
            generation_config=generation_config,
            stopping_criteria=stopping_criteria,
            tokenizer=generation_mode_kwargs.get("tokenizer"),
        )

        # Set model_kwargs `use_cache` so we can use it later in forward runs
        model_kwargs["use_cache"] = generation_config.use_cache
        # ******************* taken from main generate function up to calling the different methods *******************

        # Prepare inner 2D logic in generation loop
        input_ids = input_ids.reshape(-1, input_ids.shape[-1])

        # 10. go into different generation modes
        if generation_mode in (GenerationMode.SAMPLE, GenerationMode.GREEDY_SEARCH):
            # 11. expand input_ids with `num_return_sequences` additional sequences per batch
            if generation_config.num_return_sequences > 1:
                raise ValueError("`num_return_sequences>1` is incompatible with Dia.")

            # 12. run sample (it degenerates to greedy search when `generation_config.do_sample=False`)
            return self._sample(
                input_ids,
                logits_processor=prepared_logits_processor,
                stopping_criteria=prepared_stopping_criteria,
                generation_config=generation_config,
                **generation_mode_kwargs,
                **model_kwargs,
            )
        else:
            raise ValueError(
                "Got incompatible mode for generation, should be one of greedy or sampling. "
                "Ensure that beam search is de-activated by setting `num_beams=1`."
            )

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        generation_config: Optional[GenerationConfig] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], list[int]]] = None,
        synced_gpus: Optional[bool] = None,
        assistant_model: Optional["PreTrainedModel"] = None,
        streamer: Optional["BaseStreamer"] = None,
        negative_prompt_ids: Optional[torch.Tensor] = None,
        negative_prompt_attention_mask: Optional[torch.Tensor] = None,
        use_model_defaults: Optional[bool] = None,
        custom_generate: Optional[str] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        # We expect the initial input ids to be the complete mask (delayed input)
        delay_mask = kwargs.get("decoder_input_ids")
        if delay_mask is not None:
            delay_mask = delay_mask.clone()

        output = self._main_generate_loop(
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

        # Reshape from 2D (bsz * channels, seq_len) to 3D (bsz, seq_len, channels)
        num_channels = self.config.decoder_config.num_channels
        bsz = output_sequences.shape[0] // num_channels
        output_sequences = output_sequences.reshape(bsz, num_channels, -1).transpose(1, 2)

        # Apply delay mask
        output_sequences = self.apply_delay_mask(output_sequences, self.config.pad_token_id, delay_mask)

        if return_dict_in_generate:
            output.sequences = output_sequences
        else:
            output = output_sequences

        return output
