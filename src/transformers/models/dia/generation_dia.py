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
"""PyTorch Dia model."""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch

from ...generation.logits_process import LogitsProcessor, LogitsProcessorList
from ...generation.stopping_criteria import StoppingCriteria, StoppingCriteriaList
from ...generation.streamers import BaseStreamer
from ...generation.utils import GenerateOutput, GenerationConfig, GenerationMixin
from ...modeling_utils import PreTrainedModel
from ...tokenization_utils_base import PreTrainedTokenizerBase


class DiaClassifierFreeGuidanceFilterLogitsProcessor(LogitsProcessor):
    def __init__(self, cfg_scale: float = 3.0, cfg_filter_top_k: int = 50, device: str = "cpu"):
        self.cfg_scale = torch.tensor(cfg_scale, device=device)
        self.cfg_filter_top_k = cfg_filter_top_k

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if scores.shape[0] != 2 * input_ids.shape[0]:
            raise ValueError(
                f"Logits should have twice the batch size of the input ids, the first half of batches corresponding to "
                f"the conditional inputs, and the second half of batches corresponding to the unconditional inputs. Got "
                f"batch size {scores.shape[0]} for the logits and {input_ids.shape[0]} for the input ids."
            )
        # TODO: reshape from (B * C, V) to (B, C, V)

        # cfg
        scores_last = scores.view(scores.shape[0] // 2, 2, *scores.shape[1:])
        uncond_scores = scores_last[:, 0, :]
        cond_scores = scores_last[:, 1, :]
        scores = cond_scores + self.cfg_scale * (cond_scores - uncond_scores)  # Shape [B_orig, C, V]

        # cfg filter top k
        _, top_k_indices = torch.topk(scores, k=self.cfg_filter_top_k, dim=-1)
        mask = torch.ones_like(scores, dtype=torch.bool)
        mask = mask.scatter(dim=-1, index=top_k_indices, value=False)
        scores = cond_scores.masked_fill(mask, -torch.inf)

        return scores


class DiaEOSFilterAndScaleLogitsProcessor(LogitsProcessor):
    def __init__(self, eos_value: int, eos_scale: float, device: str = "cpu"):
        self.eos_value = eos_value
        self.eos_scale = torch.tensor(eos_scale, device=device) if eos_scale != 1.0 else None

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """
        EOS filter, this ensures:
        # 1. only channel 0 can generate EOS
        # 2. if channel 0 has EOS with highest logit, it will be the only candidate
        # 3. if channel 0 has EOS not with highest logit, it will be suppressed
        """
        # TODO: reshape from (B * C, V) to (B, C, V)

        scores[:, 1:, self.eos_value :] = torch.full_like(
            scores[:, 1:, self.eos_value :],
            fill_value=-torch.inf,
        )
        if self.eos_scale is not None:
            scores[:, 0, self.eos_value] *= self.eos_scale

        scores_flat = scores.view(-1, scores.shape[-1])

        top_logit_indices = torch.argmax(scores_flat, dim=-1)
        eos_not_highest_mask = top_logit_indices != self.eos_value
        mask_eos_unless_highest = torch.zeros_like(scores_flat, dtype=torch.bool)
        mask_eos_unless_highest[eos_not_highest_mask, self.eos_value] = True
        scores_flat = scores_flat.masked_fill(mask_eos_unless_highest, -torch.inf)
        eos_highest_mask = top_logit_indices == self.eos_value
        mask_eos_highest = torch.zeros_like(scores_flat, dtype=torch.bool)
        mask_eos_highest[eos_highest_mask, : self.eos_value] = True
        scores_flat = scores_flat.masked_fill(mask_eos_highest, -torch.inf)

        scores = scores_flat.view(scores.shape)

        return scores


class DiaEOSDelayPatternLogitsProcessor(LogitsProcessor):
    def __init__(
        self,
        delay_pattern: torch.Tensor,
        eos_value: int,
        pad_value: int,
        max_step: int,
        device: str = "cpu",
    ):
        self.delay_pattern = delay_pattern
        self.max_delay_pattern = delay_pattern.max().item()
        self.num_channels = delay_pattern.shape[0]
        self.eos_value = eos_value
        self.pad_value = pad_value
        self.max_step = torch.tensor(max_step, device=device)
        self.eos_countup: Optional[torch.Tensor] = None
        self.device = device

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """
        This logits processor is used to apply the delay pattern to the generated tokens when EOS is generated.

        If delay pattern is [0, 2, 3, 4] then:

            s   s+1 s+2 s+3 s+4 s+5 ...
            |   |   |   |   |   |
        C0: EOS PAD PAD PAD PAD PAD ...
        C1: x   x   EOS PAD PAD PAD ...
        C2: x   x   x   EOS PAD PAD ...
        C3: x   x   x   x   EOS PAD ...

        The PAD & EOS are forced from step s+1.
        """
        # TODO: reshape from (B * C, V) to (B, C, V)
        # and reshape from (B * C, S) to (B, S, C)

        # EOS Countup
        # Due to delay pattern, we do not stop generation at the first EOS token.
        # Instead, we force EOS, PAD at delay pattern steps.
        batch_size = scores.shape[0]
        if self.eos_countup is None:
            self.eos_countup = torch.full((batch_size,), -1, dtype=torch.long, device=self.device)

        step = input_ids.shape[1]

        # EOS countdown and delay pattern application
        active_mask = self.eos_countup > 0

        if active_mask.any():
            # Logits for active items: [num_active, C, V]
            scores_active = scores[active_mask]
            # Countdown values for active items: [num_active]
            eos_countup_active = self.eos_countup[active_mask]

            # Expand for comparison with delay_pattern: [num_active, C]
            eos_countup_active = eos_countup_active.unsqueeze(1).expand(-1, self.num_channels)
            delay_pattern = self.delay_pattern.unsqueeze(0).expand(scores_active.shape[0], -1)  # [num_active, C]

            # Mask for forcing EOS: [num_active, C]
            force_eos_mask_BxC = eos_countup_active == delay_pattern
            # Mask for forcing PAD: [num_active, C]
            force_pad_mask_BxC = eos_countup_active > delay_pattern

            # Efficiently apply forced EOS and PAD logits
            vocab_size = scores_active.shape[-1]

            # Create template rows for forced EOS and PAD
            eos_row = torch.full((vocab_size,), -torch.inf, device=self.device, dtype=scores_active.dtype)
            eos_row[self.eos_value] = 0.0
            pad_row = torch.full((vocab_size,), -torch.inf, device=self.device, dtype=scores_active.dtype)
            pad_row[self.pad_value] = 0.0

            # Clone the active slice to modify it
            final_modified_slice = scores_active.clone()
            final_modified_slice[force_eos_mask_BxC] = eos_row
            final_modified_slice[force_pad_mask_BxC] = pad_row

            # Update the original logits tensor with the modified slice
            scores[active_mask] = final_modified_slice

        # This is possible because we applied `DiaEOSFilterAndScaleLogitsProcessor`
        last_generated_tokens = torch.argmax(scores, dim=-1)[:, 0]  # Shape [B_orig]
        eos_start_mask = last_generated_tokens == self.eos_value
        eos_start_mask |= step + self.max_delay_pattern >= self.max_step
        eos_start_mask &= self.eos_countup < 0

        # Make sure that the EOS token is the only candidate for the first token
        scores[eos_start_mask, 0, :] = -torch.inf
        scores[eos_start_mask, 0, self.eos_value] = 0.0

        self.eos_countup[eos_start_mask] = 0
        self.eos_countup[self.eos_countup >= 0] += 1

        return scores


class DiaEosTokenCriteria(StoppingCriteria):
    """
    This class can be used to stop generation whenever the "end-of-sequence" token is generated.
    By default, it uses the `model.generation_config.eos_token_id`.

    Args:
        eos_token_id (`Union[int, List[int], torch.Tensor]`):
            The id(s) of the *end-of-sequence* token.
    """

    def __init__(self, eos_value: int, delay_pattern: torch.Tensor, device: str = "cpu"):
        self.eos_token_id = torch.tensor(eos_value, device=device)
        self.max_delay_pattern = delay_pattern.max().item()

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> torch.BoolTensor:
        """
        This stopping criteria is used to stop generation when EOS is generated.

        If delay pattern is [0, 2, 3, 4] then:

            s   s+1 s+2 s+3 s+4 s+5 ...
            |   |   |   |   |   |
        C0: EOS PAD PAD PAD PAD PAD ...
        C1: x   x   EOS PAD PAD PAD ...
        C2: x   x   x   EOS PAD PAD ...
        C3: x   x   x   x   EOS PAD ...

        We need to stop generation in step s+3, where all of the information is generated.
        We check by if the first channel has EOS in the `step - max_delay_pattern + 1` step.
        """
        # TODO: reshape from (B * C, V) to (B, C, V)
        # and reshape from (B * C, S) to (B, S, C)

        if input_ids.shape[1] < self.max_delay_pattern:
            return torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)

        is_done = input_ids[:, -self.max_delay_pattern + 1, 0] == self.eos_token_id
        return is_done


class DiaGenerationMixin(GenerationMixin):
    # A few special cases in Dia that we need for custom preprocessing:
    #   1. "uses_cfg": Indicates CFG which needs preparation to be properly handled by repeats
    #   2. "cfg_guidance_top_k": Unique to Dia used for logits processing
    #   3. "eos_scale": Unique to Dia used for logits processing
    _valid_external_model_kwargs = ["uses_cfg", "cfg_guidance_top_k", "eos_scale"]

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
        # TODO: save dia generation config with proper values and disallow non-cfg?
        # if generation_config.guidance_scale is not None and generation_config.guidance_scale != 1:
        cfg_processor = DiaClassifierFreeGuidanceFilterLogitsProcessor(
            cfg_scale=generation_config.guidance_scale if generation_config.guidance_scale is not None else 3.0,
            cfg_filter_top_k=model_kwargs.get("cfg_guidance_top_k", 50),
            device=device,
        )
        # Avoid adding cfg again
        generation_config.guidance_scale = None

        custom_processors.append(
            DiaEOSFilterAndScaleLogitsProcessor(
                eos_value=self.config.eos_token_id,
                eos_scale=model_kwargs.get("eos_scale", 0.8),
                device=device,
            )
        )

        custom_processors.append(
            DiaEOSDelayPatternLogitsProcessor(
                delay_pattern=torch.tensor(self.config.delay_pattern, device=device, dtype=torch.long),
                eos_value=self.config.eos_token_id,
                pad_value=self.config.pad_token_id,
                max_step=generation_config.max_length,
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

    def _get_stopping_criteria(
        self,
        generation_config: GenerationConfig,
        stopping_criteria: Optional[StoppingCriteriaList],
        tokenizer: Optional["PreTrainedTokenizerBase"] = None,
        **kwargs,
    ) -> StoppingCriteriaList:
        custom_stopping_criteria = StoppingCriteriaList()

        # We end generation after `max delays` if every sample generated eos on their first channel
        custom_stopping_criteria.append(
            DiaEosTokenCriteria(
                eos_value=self.config.eos_token_id,
                delay_pattern=torch.tensor(self.config.delay_pattern, device=self.device, dtype=torch.long),
                device=self.device,
            )
        )

        return super()._get_stopping_criteria(
            generation_config=generation_config,
            stopping_criteria=custom_stopping_criteria,
            tokenizer=tokenizer,
            **kwargs,
        )

    def _prepare_generation_config(
        self, generation_config: Optional[GenerationConfig], use_model_defaults: Optional[bool] = None, **kwargs: Dict
    ) -> Tuple[GenerationConfig, Dict]:
        generation_config, model_kwargs = super()._prepare_generation_config(
            generation_config, use_model_defaults, **kwargs
        )

        # We allow generation up to max length + max delay pattern
        # (will revert back to max length after generation)
        # TODO: check where max delay wasn't considered but added afterwards
        generation_config.max_length += max(self.config.delay_pattern)

        # TODO: move default value in saved generation config and make it dependent on this - disallow non-cfg?
        # TODO: decoder prep in prepare_for_gen...
        # Indicating cfg to prepare unconditioned input
        model_kwargs["uses_cfg"] = True

        # We need this for our custom logit processors and stopping criteria
        self.config.eos_token_id = self.config.eos_token_id or generation_config.eos_token_id
        self.config.pad_token_id = self.config.pad_token_id or generation_config.pad_token_id
        self.config.bos_token_id = self.config.bos_token_id or generation_config.bos_token_id

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
        if model_kwargs["uses_cfg"]:
            inputs = inputs[:, None, :]
            unconditioned_inputs = torch.zeros_like(inputs)
            inputs = torch.stack([unconditioned_inputs, inputs], dim=1).view(-1, inputs.shape[-1])

            if model_kwargs.get("attention_mask", None) is not None:
                model_kwargs["attention_mask"] = model_kwargs["attention_mask"].repeat_interleave(2, dim=0)

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
        real_batch_size = batch_size // 2 if model_kwargs["uses_cfg"] else batch_size

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
        uses_cfg=None,
        **kwargs,
    ):
        # Base method handles most things except CFG and the delay pattern mask
        model_inputs = super().prepare_inputs_for_generation(input_ids, encoder_outputs=encoder_outputs, **kwargs)

        # Post processing for CFG and overwriting via delay pattern mask
        # 1. Reshape (bsz * channels, seq_len) to (bsz, seq_len, channels)
        batch_size = encoder_outputs[0].shape[0] // 2 if uses_cfg else encoder_outputs[0].shape[0]
        model_inputs["decoder_input_ids"] = model_inputs["decoder_input_ids"].reshape(
            batch_size, -1, self.config.decoder_config.num_channels
        )

        # 2. Delay pattern mask -- force tokens if not allowed to predict (!= -1 in mask)
        model_inputs["decoder_input_ids"] = self.apply_delay_mask(
            model_inputs["decoder_input_ids"], decoder_delay_mask
        )

        # 3. Apply CFG duplication if needed
        if uses_cfg:
            model_inputs["decoder_input_ids"] = model_inputs["decoder_input_ids"].repeat_interleave(2, dim=0)

            if model_inputs["decoder_attention_mask"] is not None:
                model_inputs["decoder_attention_mask"] = model_inputs["decoder_attention_mask"].repeat_interleave(
                    2, dim=0
                )

            if model_inputs["decoder_position_ids"] is not None:
                model_inputs["decoder_position_ids"] = model_inputs["decoder_position_ids"].repeat_interleave(2, dim=0)

        # Avoid specific kwarg clashes
        for key in ["uses_cfg", "cfg_guidance_top_k", "eos_scale"]:
            model_inputs.pop(key, None)

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

        # A few special cases in Dia that we need for custom preprocessing
        # Check `self._valid_external_model_kwargs` for more details
        kwargs["valid_external_model_kwargs"] = self._valid_external_model_kwargs

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
        # TODO: check this, prototype
        if delay_mask is None:
            return input_ids

        mask_len = min(input_ids.shape[1], delay_mask.shape[1])
        valid_mask = delay_mask[:, :mask_len, :]
        valid_input = input_ids[:, :mask_len, :]

        return torch.where(valid_mask == -1, valid_input, valid_mask)

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
