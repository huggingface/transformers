# coding=utf-8
# Copyright 2025 Kyutai and The HuggingFace Inc. team. All rights reserved.
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

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from ...generation import GenerationMixin
from ...modeling_flash_attention_utils import (
    is_flash_attn_available,
)
from ...utils import (
    is_torch_flex_attn_available,
)
from ..auto import AutoModel
from ..llama.modeling_llama import LlamaForCausalLM
from ..moshi.modeling_moshi import MoshiModel


if is_torch_flex_attn_available():
    pass


if is_flash_attn_available():
    pass


class MoshiAsrEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        # TODO: should it be splitted to audio and text embeddings?
        self.embed_tokens = nn.Embedding(
            config.vocab_size + (config.num_codebooks * config.codebook_vocab_size) + 1,
            config.hidden_size,
            padding_idx=config.audio_pad_token_id,
        )
        audio_tokens_offsets = torch.arange(config.num_codebooks) * config.codebook_vocab_size
        audio_tokens_offsets += config.vocab_size
        audio_tokens_offsets = nn.functional.pad(
            audio_tokens_offsets, (1, 0)
        )  # pad one 0 to the left for the text token
        self.register_buffer("audio_tokens_offsets", audio_tokens_offsets, persistent=False)

    def forward(self, input_ids):
        input_ids = torch.where(
            input_ids == self.embed_tokens.padding_idx, input_ids, input_ids + self.audio_tokens_offsets
        )
        inputs_embeds = self.embed_tokens(input_ids)
        inputs_embeds = inputs_embeds.sum(dim=2)
        return inputs_embeds


class MoshiAsrModel(MoshiModel):
    def __init__(self, config):
        super().__init__(config)
        self.embed_tokens = MoshiAsrEmbeddings(config)


class MoshiAsrForConditionalGeneration(LlamaForCausalLM, GenerationMixin):
    def __init__(self, config):
        super().__init__(config)
        self.codec_model = AutoModel.from_config(config.codec_config)

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

        if "input_values" in model_kwargs:  # TODO: better handling of edge case here
            audio_window_size = model_kwargs.get("audio_window_size", None)
            if audio_window_size is None:
                audio_window_size = int(model_kwargs["input_values"].shape[-1] / self.config.frame_size)
                model_kwargs["audio_window_size"] = audio_window_size

            batch_size = inputs.shape[0]
            device = inputs.device

            # initialize audio tokens
            model_kwargs["audio_tokens"] = torch.zeros(
                (batch_size, audio_window_size, self.config.num_codebooks),
                device=device,
                dtype=torch.long,
            )
            model_kwargs["current_window"] = (
                torch.tensor([0, audio_window_size], device=device, dtype=torch.long)
                .expand(batch_size, -1)
                .contiguous()
            )

        return inputs, input_name, model_kwargs

    def prepare_inputs_for_generation(
        self,
        *args,
        audio_tokens: Optional[torch.LongTensor] = None,
        input_values: Optional[torch.FloatTensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        audio_window_size: Optional[int] = None,
        current_window: Optional[Tuple[int, int]] = None,
        **kwargs,
    ):
        model_inputs = super().prepare_inputs_for_generation(*args, **kwargs)

        if input_values is not None:
            cache_position = model_inputs["cache_position"]

            audio_positions = cache_position - self.config.delay_in_tokens
            start, end = current_window[0]  # closed interval
            if audio_positions[-1] >= end:
                # we need to encode the new audio tokens
                with torch.no_grad():
                    input_values_start_idx = start * self.config.frame_size
                    input_values_end_idx = (start + audio_window_size) * self.config.frame_size
                    current_input_values = input_values[..., input_values_start_idx:input_values_end_idx]
                    new_audio_tokens = self.codec_model.encode(current_input_values).audio_codes.transpose(1, 2)

                audio_tokens.copy_(new_audio_tokens)

                start = end.clone()
                end = end + audio_window_size
                current_window.copy_(
                    torch.tensor([start, end], device=current_window.device).expand(current_window.shape[0], -1)
                )

            current_audio_tokens_idxs = (audio_positions - start).clamp(min=0)
            current_audio_tokens = audio_tokens[:, current_audio_tokens_idxs, :]

            current_audio_tokens[:, cache_position == 0, :] = self.config.audio_bos_token_id
            current_audio_tokens[:, audio_positions < 0, :] = self.config.audio_pad_token_id

            input_ids = model_inputs.pop("input_ids")
            input_ids = torch.cat(
                [input_ids.unsqueeze(2), current_audio_tokens],
                dim=2,
            )
            model_inputs["input_ids"] = input_ids

        input_ids = model_inputs.pop("input_ids")
        model_inputs["inputs_embeds"] = self.model.embed_tokens(input_ids)

        return model_inputs

    def generate(self, *args, audio_window_size: Optional[int] = None, **kwargs):
        # TODO: clean
        padding_mask = kwargs.get("padding_mask")
        max_new_tokens = kwargs.pop("max_new_tokens", None)
        kwargs["audio_window_size"] = audio_window_size

        if padding_mask is not None:
            audio_tokens_mask = self.codec_model.get_audio_codes_mask(padding_mask)

            # TODO: @eustlb, we way padded sequences should be handled from generate would be with max_new_tokens
            # batch idx specific parameters to that the stopping_criteria handles it
            max_new_tokens = audio_tokens_mask.sum(dim=-1).max()
        # TODO: handle when max_new_tokens is in kwargs
        # TODO: cache_implementation = sliding_window should be in default generation_config

        return GenerationMixin.generate(
            *args,
            max_new_tokens=max_new_tokens,
            **kwargs,
        )


__all__ = ["MoshiAsrModel", "MoshiAsrForConditionalGeneration"]
