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
from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn

from ...cache_utils import Cache
from ...feature_extraction_utils import BatchFeature
from ...generation import GenerationConfig, GenerationMixin
from ...modeling_utils import PreTrainedModel
from ...utils import PaddingStrategy, TensorType, logging
from ..auto import AutoModel
from ..encodec.feature_extraction_encodec import EncodecFeatureExtractor
from ..llama.modeling_llama import LlamaForCausalLM
from ..mimi.modeling_mimi import MimiConv1dPaddingCache
from ..moshi.modeling_moshi import MoshiModel, MoshiPreTrainedModel


logger = logging.get_logger(__name__)


class KyutaiSpeechToTextFeatureExtractor(EncodecFeatureExtractor):
    r"""
    Constructs an KyutaiSpeechToText feature extractor.

    This feature extractor inherits from [`~feature_extraction_sequence_utils.SequenceFeatureExtractor`] which contains
    most of the main methods. Users should refer to this superclass for more information regarding those methods.

    Args:
        feature_size (`int`, *optional*, defaults to 1):
            The feature dimension of the extracted features. Use 1 for mono, 2 for stereo.
        sampling_rate (`int`, *optional*, defaults to 24000):
            The sampling rate at which the audio waveform should be digitalized expressed in hertz (Hz).
        padding_value (`float`, *optional*, defaults to 0.0):
            The value that is used to fill the padding values.
        chunk_length_s (`float`, *optional*):
            If defined the audio is pre-processed into chunks of lengths `chunk_length_s` and then encoded.
        overlap (`float`, *optional*):
            Defines the overlap between each chunk. It is used to compute the `chunk_stride` using the following
            formulae : `int((1.0 - self.overlap) * self.chunk_length)`.
        audio_delay_seconds (`float`, *optional*, defaults to 0.0):
            The delay in seconds to add after the audio (right padding).
        audio_silence_prefix_seconds (`float`, *optional*, defaults to 0.0):
            The silence prefix in seconds to add before the audio (left padding).
    """

    def __init__(
        self,
        audio_delay_seconds: Optional[float] = 0.0,
        audio_silence_prefix_seconds: Optional[float] = 0.0,
        **super_kwargs,
    ):
        super().__init__(**super_kwargs)
        self.audio_delay_seconds = audio_delay_seconds
        self.audio_silence_prefix_seconds = audio_silence_prefix_seconds

    def __call__(
        self,
        raw_audio: Union[np.ndarray, list[float], list[np.ndarray], list[list[float]]],
        padding: Optional[Union[bool, str, PaddingStrategy]] = None,
        truncation: Optional[bool] = False,
        max_length: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        sampling_rate: Optional[int] = None,
    ) -> BatchFeature:
        """
        Main method to featurize and prepare for the model one or several sequence(s).

        Args:
            raw_audio (`np.ndarray`, `list[float]`, `list[np.ndarray]`, `list[list[float]]`):
                The sequence or batch of sequences to be processed. Each sequence can be a numpy array, a list of float
                values, a list of numpy arrays or a list of list of float values. The numpy array must be of shape
                `(num_samples,)` for mono audio (`feature_size = 1`), or `(2, num_samples)` for stereo audio
                (`feature_size = 2`).
            padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `True`):
                Select a strategy to pad the returned sequences (according to the model's padding side and padding
                index) among:

                - `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
                  sequence if provided).
                - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
                  acceptable input length for the model if that argument is not provided.
                - `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different
                  lengths).
            truncation (`bool`, *optional*, defaults to `False`):
                Activates truncation to cut input sequences longer than `max_length` to `max_length`.
            max_length (`int`, *optional*):
                Maximum length of the returned list and optionally padding length (see above).
            return_tensors (`str` or [`~utils.TensorType`], *optional*):
                If set, will return tensors instead of list of python integers. Acceptable values are:

                - `'tf'`: Return TensorFlow `tf.constant` objects.
                - `'pt'`: Return PyTorch `torch.Tensor` objects.
                - `'np'`: Return Numpy `np.ndarray` objects.
            sampling_rate (`int`, *optional*):
                The sampling rate at which the `audio` input was sampled. It is strongly recommended to pass
                `sampling_rate` at the forward call to prevent silent errors.
        """
        if sampling_rate is not None:
            if sampling_rate != self.sampling_rate:
                raise ValueError(
                    f"The model corresponding to this feature extractor: {self} was trained using a sampling rate of"
                    f" {self.sampling_rate}. Please make sure that the provided audio input was sampled with"
                    f" {self.sampling_rate} and not {sampling_rate}."
                )
        else:
            logger.warning(
                f"It is strongly recommended to pass the `sampling_rate` argument to `{self.__class__.__name__}()`. "
                "Failing to do so can result in silent errors that might be hard to debug."
            )

        if padding and truncation:
            raise ValueError("Both padding and truncation were set. Make sure you only set one.")
        elif padding is None:
            # by default let's pad the inputs
            padding = True

        is_batched = bool(
            isinstance(raw_audio, (list, tuple)) and (isinstance(raw_audio[0], (np.ndarray, tuple, list)))
        )

        if is_batched:
            raw_audio = [np.asarray(audio, dtype=np.float32).T for audio in raw_audio]
        elif not is_batched and not isinstance(raw_audio, np.ndarray):
            raw_audio = np.asarray(raw_audio, dtype=np.float32)
        elif isinstance(raw_audio, np.ndarray) and raw_audio.dtype is np.dtype(np.float64):
            raw_audio = raw_audio.astype(np.float32)

        # always return batch
        if not is_batched:
            raw_audio = [np.asarray(raw_audio).T]

        # verify inputs are valid
        for idx, example in enumerate(raw_audio):
            if example.ndim > 2:
                raise ValueError(f"Expected input shape (channels, length) but got shape {example.shape}")
            if self.feature_size == 1 and example.ndim != 1:
                raise ValueError(f"Expected mono audio but example has {example.shape[-1]} channels")
            if self.feature_size == 2 and example.shape[-1] != 2:
                raise ValueError(f"Expected stereo audio but example has {example.shape[-1]} channels")

        padded_inputs = None
        input_values = BatchFeature({"input_values": raw_audio})
        if self.chunk_stride is not None and self.chunk_length is not None and max_length is None:
            if truncation:
                max_length = min(array.shape[0] for array in raw_audio)
                nb_step = int(np.floor(max_length / self.chunk_stride))
                max_length = (nb_step - 1) * self.chunk_stride + self.chunk_length
            elif padding:
                max_length = max(array.shape[0] for array in raw_audio)
                nb_step = int(np.ceil(max_length / self.chunk_stride))
                max_length = (nb_step - 1) * self.chunk_stride + self.chunk_length
                padding = "max_length"
            else:
                padded_inputs = input_values

        # normal padding on batch
        if padded_inputs is None:
            padded_inputs = self.pad(
                input_values,
                max_length=max_length,
                truncation=truncation,
                padding=padding,
                return_attention_mask=padding,
            )

            if padding:
                padded_inputs["padding_mask"] = padded_inputs.pop("attention_mask")

        # now let's padd left and right
        pad_left = int(self.audio_silence_prefix_seconds * self.sampling_rate)
        pad_right = int((self.audio_delay_seconds + 1.0) * self.sampling_rate)
        padded_inputs["input_values"] = np.pad(
            padded_inputs["input_values"],
            ((0, 0), (pad_left, pad_right)),
            mode="constant",
            constant_values=0.0,
        )
        if padding:
            padded_inputs["padding_mask"] = np.pad(
                padded_inputs["padding_mask"],
                ((0, 0), (pad_left, pad_right)),
                mode="constant",
                constant_values=0,
            )

        input_values = []
        for example in padded_inputs.pop("input_values"):
            if self.feature_size == 1:
                example = example[..., None]
            input_values.append(example.T)

        padded_inputs["input_values"] = input_values
        if return_tensors is not None:
            padded_inputs = padded_inputs.convert_to_tensors(return_tensors)

        return padded_inputs


class KyutaiSpeechToTextPreTrainedModel(MoshiPreTrainedModel):
    pass


class KyutaiSpeechToTextConv1dPaddingCache(MimiConv1dPaddingCache):
    pass


class KyutaiSpeechToTextEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
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


class KyutaiSpeechToTextForConditionalGeneration(LlamaForCausalLM, GenerationMixin):
    _keep_in_fp32_modules_strict = ["codec_model"]

    def __init__(self, config):
        super().__init__(config)
        self.codec_model = AutoModel.from_config(config.codec_config)

        # we are in an edge case where for the codec_model self.can_generate is False, setting self.codec_model.generation_config to None
        # yet the codec_model needs a generation config to initalize it's cache for streaming inference
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
        inputs: Optional[torch.Tensor] = None,
        bos_token_id: Optional[torch.Tensor] = None,
        model_kwargs: Optional[dict[str, torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, Optional[str], dict[str, torch.Tensor]]:
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
            "_get_cache",
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
            assistant_model=None,
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
        audio_tokens: Optional[torch.LongTensor] = None,
        input_values: Optional[torch.FloatTensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        audio_window_size: Optional[int] = None,
        current_window: Optional[tuple[int, int]] = None,
        encoder_past_key_values: Optional[Cache] = None,
        padding_cache: Optional[KyutaiSpeechToTextConv1dPaddingCache] = None,
        **kwargs,
    ):
        model_inputs = GenerationMixin.prepare_inputs_for_generation(self, *args, **kwargs)

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
    "KyutaiSpeechToTextFeatureExtractor",
]
