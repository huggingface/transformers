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

import numpy as np
import torch
import torch.nn as nn

from ... import initialization as init
from ...cache_utils import Cache
from ...feature_extraction_utils import BatchFeature
from ...generation import GenerationConfig, GenerationMixin
from ...modeling_utils import PreTrainedModel
from ...utils import PaddingStrategy, TensorType, logging
from ..auto import AutoModel
from ..encodec.audio_processing_encodec import EncodecAudioProcessor
from ..llama.modeling_llama import LlamaForCausalLM
from ..mimi.modeling_mimi import MimiConv1dPaddingCache
from ..moshi.modeling_moshi import MoshiModel, MoshiPreTrainedModel


logger = logging.get_logger(__name__)


class KyutaiSpeechToTextPreTrainedModel(MoshiPreTrainedModel):
    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, KyutaiSpeechToTextEmbeddings):
            audio_tokens_offsets = torch.arange(module.config.num_codebooks) * module.config.codebook_vocab_size
            audio_tokens_offsets += module.config.vocab_size
            audio_tokens_offsets = nn.functional.pad(audio_tokens_offsets, (1, 0))
            init.copy_(module.audio_tokens_offsets, audio_tokens_offsets)


class KyutaiSpeechToTextConv1dPaddingCache(MimiConv1dPaddingCache):
    pass


class KyutaiSpeechToTextEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
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
        self.audio_tokens_offsets = nn.Buffer(audio_tokens_offsets, persistent=False)

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


class KyutaiSpeechToTextForConditionalGeneration(LlamaForCausalLM, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.embed_tokens.weight"}
    _keep_in_fp32_modules_strict = ["codec_model"]
    output_modalities = ("audio", "text")

    def __init__(self, config):
        super().__init__(config)
        self.codec_model = AutoModel.from_config(config.codec_config)

        # we are in an edge case where for the codec_model self.can_generate is False, setting self.codec_model.generation_config to None
        # yet the codec_model needs a generation config to initialize it's cache for streaming inference
        # we therefore initialize a generation config for the codec model
        self.codec_model.generation_config = GenerationConfig.from_model_config(config.codec_config)

    def forward(self, **super_kwargs):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Example:

        ```python
        >>> import torch
        >>> from datasets import load_dataset, Audio
        >>> from transformers import KyutaiSpeechToTextProcessor, KyutaiSpeechToTextForConditionalGeneration

        >>> torch_device = "cuda" if torch.cuda.is_available() else "cpu"
        >>> model_id = "kyutai/stt-2.6b-en-trfs"

        >>> processor = KyutaiSpeechToTextProcessor.from_pretrained(model_id)
        >>> model = KyutaiSpeechToTextForConditionalGeneration.from_pretrained(model_id, device_map=torch_device)

        >>> ds = load_dataset(
        ...     "hf-internal-testing/librispeech_asr_dummy", "clean", split="validation"
        ... )

        >>> ds = ds.cast_column("audio", Audio(sampling_rate=24000))
        >>> inputs = processor(
        ...     ds[0]["audio"]["array"],
        ... )
        >>> inputs.to(torch_device)

        >>> output_tokens = model.generate(**inputs)
        >>> print(processor.batch_decode(output_tokens, skip_special_tokens=True))
        ```"""
        super().forward(**super_kwargs)

    def _prepare_generation_config(self, *args, **kwargs):
        generation_config, model_kwargs = GenerationMixin._prepare_generation_config(self, *args, **kwargs)
        # this should be passed to the model kwargs for the input preparation
        model_kwargs["audio_window_size"] = (
            generation_config.audio_window_size if hasattr(generation_config, "audio_window_size") else None
        )
        return generation_config, model_kwargs

    def _prepare_model_inputs(
        self,
        inputs: torch.Tensor | None = None,
        bos_token_id: torch.Tensor | None = None,
        model_kwargs: dict[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, str | None, dict[str, torch.Tensor]]:
        inputs, input_name, model_kwargs = GenerationMixin._prepare_model_inputs(
            self,
            inputs=inputs,
            bos_token_id=bos_token_id,
            model_kwargs=model_kwargs,
        )

        audio_window_size = model_kwargs.get("audio_window_size", None)
        if audio_window_size is None:
            audio_window_size = self.codec_model.get_encoded_length(model_kwargs["input_values"].shape[-1]).item()
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
            "_prepare_static_cache",
        ]
        for method in cache_methods:
            setattr(self.codec_model, method, types.MethodType(getattr(self, method).__func__, self.codec_model))

        setattr(
            self.codec_model, "_supports_default_dynamic_cache", types.MethodType(lambda x: True, self.codec_model)
        )

        self.codec_model.generation_config.cache_implementation = "dynamic"
        self.codec_model._prepare_cache_for_generation(
            generation_config=self.codec_model.generation_config,
            model_kwargs=temporary_model_kwargs,
            generation_mode=None,
            batch_size=batch_size,
            max_cache_length=self.config.codec_config.sliding_window,
        )

        if "past_key_values" in temporary_model_kwargs:
            model_kwargs["encoder_past_key_values"] = temporary_model_kwargs["past_key_values"]

        # initialize the padding cache for the codec model
        per_layer_padding, per_layer_padding_mode, per_layer_in_channels = [], [], []
        for layer_name in self.codec_model.encoder._mimiconv1d_layer_names:
            per_layer_padding.append(self.codec_model.encoder.get_submodule(layer_name).padding_total)
            per_layer_padding_mode.append(self.codec_model.encoder.get_submodule(layer_name).pad_mode)
            per_layer_in_channels.append(self.codec_model.encoder.get_submodule(layer_name).in_channels)

        # downsample layer
        per_layer_padding.append(self.codec_model.downsample.padding_total)
        per_layer_padding_mode.append(self.codec_model.downsample.pad_mode)
        per_layer_in_channels.append(self.codec_model.downsample.in_channels)

        model_kwargs["padding_cache"] = KyutaiSpeechToTextConv1dPaddingCache(
            num_layers=len(self.codec_model.encoder._mimiconv1d_layer_names) + 1,
            per_layer_padding=per_layer_padding,
            per_layer_padding_mode=per_layer_padding_mode,
            per_layer_in_channels=per_layer_in_channels,
        )

        return inputs, input_name, model_kwargs

    def prepare_inputs_for_generation(
        self,
        *args,
        audio_tokens: torch.LongTensor | None = None,
        input_values: torch.FloatTensor | None = None,
        padding_mask: torch.Tensor | None = None,
        audio_window_size: int | None = None,
        current_window: tuple[int, int] | None = None,
        encoder_past_key_values: Cache | None = None,
        padding_cache: KyutaiSpeechToTextConv1dPaddingCache | None = None,
        **kwargs,
    ):
        model_inputs = GenerationMixin.prepare_inputs_for_generation(self, *args, **kwargs)

        if input_values is not None:
            seqlen, device = model_inputs["position_ids"].shape[-1], model_inputs["position_ids"].device
            cache = model_inputs.get("past_key_values")
            past_seen_tokens = cache.get_seq_length() if cache is not None else 0
            positions = torch.arange(seqlen, device=device) + past_seen_tokens
            start, end = current_window[0]

            # first cache position is for bos token, so we need to offset by -1
            if positions[-1] - 1 >= end:
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

                # last window can be shorter than audio_window_size, copy only the overlap
                n = min(audio_tokens.shape[1], new_audio_tokens.shape[1])
                audio_tokens[:, :n].copy_(new_audio_tokens[:, :n])

                start = end.clone()
                end = end + audio_window_size
                current_window.copy_(
                    torch.tensor([start, end], device=current_window.device).expand(current_window.shape[0], -1)
                )

            # first cache position is for bos token, so we need to offset by -1
            current_audio_tokens_idxs = (positions - start - 1).clamp(min=0)
            current_audio_tokens = audio_tokens[:, current_audio_tokens_idxs, :]

            current_audio_tokens[:, positions == 0, :] = self.config.audio_bos_token_id

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
        r"""
        This method forwards all its arguments to GenerationMixin's [`~GenerationMixin.generate`]. Please refer to the docstring of this method for more information.
        """
        max_new_tokens = kwargs.pop("max_new_tokens", None)
        input_values = kwargs.get("input_values")

        # TODO: @eustlb, we should have per-batch-idx values
        # here we do not use padding_mask to be aligned to what's done in the original codebase
        max_audio_frames = input_values.shape[-1] // self.config.codec_config.frame_size

        if max_new_tokens is None or max_new_tokens > max_audio_frames:
            if max_new_tokens is not None:
                logger.warning(
                    f"`max_new_tokens` ({max_new_tokens}) is greater than the maximum number of audio frames ({max_audio_frames})."
                    f"Setting `max_new_tokens` to {max_audio_frames}."
                )
            max_new_tokens = max_audio_frames

        return GenerationMixin.generate(
            *args,
            max_new_tokens=max_new_tokens,
            **kwargs,
        )


__all__ = [
    "KyutaiSpeechToTextPreTrainedModel",
    "KyutaiSpeechToTextModel",
    "KyutaiSpeechToTextForConditionalGeneration",
]
