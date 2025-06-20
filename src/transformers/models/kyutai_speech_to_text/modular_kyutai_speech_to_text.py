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

import types
from typing import Optional

import torch
import torch.nn as nn

from ...cache_utils import Cache
from ...generation import GenerationConfig, GenerationMixin
from ...modeling_utils import PreTrainedModel
from ...utils import logging
from ..auto import AutoModel
from ..llama.modeling_llama import LlamaForCausalLM
from ..mimi.modeling_mimi import MimiConv1dPaddingCache
from ..moshi.modeling_moshi import MoshiModel
from ..encodec.feature_extraction_encodec import EncodecFeatureExtractor


logger = logging.get_logger(__name__)


class KyutaiSpeechToTextFeatureExtractor(EncodecFeatureExtractor):
    def __init__(
        self,
        audio_delay_seconds: Optional[float] = 0.0,
        audio_silence_prefix_seconds: Optional[float] = 0.0,
        **super_kwargs,
    ):
        super().__init__(**super_kwargs)
        self.audio_delay_seconds = audio_delay_seconds
        self.audio_silence_prefix_seconds = audio_silence_prefix_seconds


class KyutaiSpeechToTextConv1dPaddingCache(MimiConv1dPaddingCache):
    pass


class KyutaiSpeechToTextEmbeddings(nn.Module):
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


class KyutaiSpeechToTextModel(MoshiModel):
    def __init__(self, config):
        super().__init__(config)
        self.embed_tokens = KyutaiSpeechToTextEmbeddings(config)


class KyutaiSpeechToTextForConditionalGeneration(LlamaForCausalLM, GenerationMixin, PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.codec_model = AutoModel.from_config(config.codec_config)

        # we are in an edge case where for the codec_model self.can_generate is False, setting self.codec_model.generation_config to None
        # yet the codec_model needs a generation config to initalize it's cache for streaming inference
        # we therefore initialize a generation config for the codec model
        self.codec_model.generation_config = GenerationConfig.from_model_config(config.codec_config)

    def _prepare_generation_config(self, *args, **kwargs):
        generation_config, model_kwargs = GenerationMixin._prepare_generation_config(*args, **kwargs)
        # this should be passed to the model kwargs for the input preparation
        model_kwargs["audio_window_size"] = (
            generation_config.audio_window_size if hasattr(generation_config, "audio_window_size") else None
        )
        return generation_config, model_kwargs

    def _prepare_model_inputs(
        self,
        inputs: Optional[torch.Tensor] = None,
        bos_token_id: Optional[torch.Tensor] = None,
        model_kwargs: Optional[dict[str, torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, Optional[str], dict[str, torch.Tensor]]:
        inputs, input_name, model_kwargs = GenerationMixin._prepare_model_inputs(
            inputs=inputs,
            bos_token_id=bos_token_id,
            model_kwargs=model_kwargs,
        )

        if "input_values" in model_kwargs:  # TODO: @eustlb, better handling of edge case here
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
                torch.tensor([0, 0], device=device, dtype=torch.long).expand(batch_size, -1).contiguous()
            )

            # let's use generate's cache preparation to prepare the cache for the codec model
            temporary_model_kwargs = {}

            # monkey patching the codec model with cache preparation methods since we don't want it to inherit fully from GenerationMixin
            # Add cache-related methods from GenerationMixin to codec model
            cache_methods = [
                "_prepare_cache_for_generation",
                "_get_cache",
                "_supports_default_dynamic_cache",
                "_get_layer_device_map_for_cache_init",
            ]
            for method in cache_methods:
                setattr(self.codec_model, method, types.MethodType(getattr(self, method).__func__, self.codec_model))

            self.codec_model._prepare_cache_for_generation(
                generation_config=self.codec_model.generation_config,
                model_kwargs=temporary_model_kwargs,
                assistant_model=None,
                batch_size=batch_size,
                max_cache_length=self.config.codec_config.sliding_window,
                device=device,
            )

            if "past_key_values" in temporary_model_kwargs:
                model_kwargs["encoder_past_key_values"] = temporary_model_kwargs["past_key_values"]

            # initialize the padding cache for the codec model
            model_kwargs["padding_cache"] = KyutaiSpeechToTextConv1dPaddingCache()

        return inputs, input_name, model_kwargs

    def prepare_inputs_for_generation(
        self,
        *args,
        audio_tokens: Optional[torch.LongTensor] = None,
        input_values: Optional[torch.FloatTensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        audio_window_size: Optional[int] = None,
        current_window: Optional[tuple[int, int]] = None,
        encoder_past_key_values: Optional[Cache] = None,
        padding_cache: Optional[KyutaiSpeechToTextConv1dPaddingCache] = None,
        **kwargs,
    ):
        model_inputs = GenerationMixin.prepare_inputs_for_generation(*args, **kwargs)

        if input_values is not None:
            cache_position = model_inputs["cache_position"]
            start, end = current_window[0]

            # first cache position is for bos token, so we need to offset by -1
            if cache_position[-1] - 1 >= end:
                # we need to encode the new audio tokens
                with torch.no_grad():
                    input_values_start_idx = start * self.config.frame_size
                    input_values_end_idx = (start + audio_window_size) * self.config.frame_size
                    current_input_values = input_values[..., input_values_start_idx:input_values_end_idx]
                    codec_model_output = self.codec_model.encode(
                        current_input_values,
                        encoder_past_key_values=encoder_past_key_values,
                        padding_cache=padding_cache,
                    )
                    new_audio_tokens = codec_model_output.audio_codes.transpose(1, 2)

                audio_tokens.copy_(new_audio_tokens)

                start = end.clone()
                end = end + audio_window_size
                current_window.copy_(
                    torch.tensor([start, end], device=current_window.device).expand(current_window.shape[0], -1)
                )

            # first cache position is for bos token, so we need to offset by -1
            current_audio_tokens_idxs = (cache_position - start - 1).clamp(min=0)
            current_audio_tokens = audio_tokens[:, current_audio_tokens_idxs, :]

            current_audio_tokens[:, cache_position == 0, :] = self.config.audio_bos_token_id

            input_ids = model_inputs.pop("input_ids")
            input_ids = torch.cat(
                [input_ids.unsqueeze(2), current_audio_tokens],
                dim=2,
            )
            model_inputs["input_ids"] = input_ids

        return model_inputs

    # TODO: @eustlb, this should be standardized
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        if kwargs.get("output_loading_info", False):
            model, loading_info = PreTrainedModel.from_pretrained(*args, **kwargs)
        else:
            model = PreTrainedModel.from_pretrained(*args, **kwargs)

        # copy depth decoder generation conf attr to the depth decoder generation config
        prefix = "codec_"
        prefix_len = len(prefix)
        codec_model_attrs = {
            attr[prefix_len:]: value
            for attr, value in vars(model.generation_config).items()
            if attr.startswith(prefix)
        }

        vars(model.codec_model.generation_config).update({"_from_model_config": False, **codec_model_attrs})

        # remove the depth decoder generation conf attr from the model generation config
        for attr in codec_model_attrs:
            delattr(model.generation_config, prefix + attr)

        if "output_loading_info" in kwargs:
            return model, loading_info
        else:
            return model

    # TODO: @eustlb, this should be standardized
    def save_pretrained(self, *args, **kwargs):
        prefix = "codec_"
        codec_model_attrs = self.codec_model.generation_config.to_diff_dict()
        codec_model_attrs.pop("transformers_version", None)
        for attr, value in codec_model_attrs.items():
            setattr(self.generation_config, prefix + attr, value)

        PreTrainedModel.save_pretrained(self, *args, **kwargs)

    def generate(self, *args, **kwargs):
        padding_mask = kwargs.get("padding_mask")
        max_new_tokens = kwargs.pop("max_new_tokens", None)

        if padding_mask is not None:
            audio_tokens_mask = self.codec_model.get_audio_codes_mask(padding_mask)

            # TODO: @eustlb, we should have per-batch-idx values
            max_audio_frames = audio_tokens_mask.sum(dim=-1).max()

            if max_new_tokens > max_audio_frames:
                logger.warning(
                    f"`max_new_tokens` ({max_new_tokens}) is greater than the maximum number of audio frames ({max_audio_frames})."
                    f"Setting `max_new_tokens` to {max_audio_frames}."
                )
                max_new_tokens = max_audio_frames

        return GenerationMixin.generate(
            self,
            *args,
            max_new_tokens=max_new_tokens,
            **kwargs,
        )


__all__ = ["KyutaiSpeechToTextModel", "KyutaiSpeechToTextForConditionalGeneration", "KyutaiSpeechToTextFeatureExtractor"]
