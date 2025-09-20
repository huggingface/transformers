# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
"""Processor class for Vocos"""

from collections.abc import Sequence
from typing import Optional

import numpy as np

from ...processing_utils import AudioKwargs, BatchFeature, ProcessingKwargs, ProcessorMixin, Unpack
from ...utils import is_torch_available


if is_torch_available():
    import torch
    import torch.nn.functional as F


class VocosAudioKwargs(AudioKwargs, total=False):
    bandwidths: Sequence[float]


class VocosProcessorKwargs(ProcessingKwargs, total=False):
    audio_kwargs: VocosAudioKwargs
    _defaults = {
        "audio_kwargs": {
            "bandwidths": [1.5, 3.0, 6.0, 12.0],
            "sampling_rate": 24000,
        },
        "common_kwargs": {"return_tensors": "pt"},
    }


class VocosProcessor(ProcessorMixin):
    r"""
    Constructs a Vocos processor which wraps a [`VocosFeatureExtractor`] and an audio tokenizer [`EncodecModel`]  into
    a single processor that can handle both mel-spectrogram feature extraction and EnCodec neural codec based
    feature extraction.

    Args:
        feature_extractor (`VocosFeatureExtractor`):
            Feature extractor that computes mel-spectrogram features.
        audio_tokenizer (`EncodecModel`):
            Audio tokenizer used to encode raw audio into discrete codebooks
            (when `bandwidth` is given) via the EnCodec model, then turns those codes into token embeddings.
    """

    feature_extractor_class = "VocosFeatureExtractor"
    audio_tokenizer_class = "EncodecModel"
    attributes = ["feature_extractor"]

    def __init__(self, feature_extractor, audio_tokenizer):
        super().__init__(feature_extractor=feature_extractor, audio_tokenizer=audio_tokenizer)
        self.audio_tokenizer.eval()

    def __call__(
        self,
        audio=None,
        codes=None,
        bandwidth: Optional[float] = None,
        return_tensors: str = "pt",
        **kwargs: Unpack[VocosProcessorKwargs],
    ) -> BatchFeature:
        """
        Main method to prepare inputs for the Vocos model, it supports two processing workflows:
        - Mel-spectrogram variant (only `audio` provided): extracts mel-spectrogram features from raw audio, by passing
        it to VocosFeatureExtractor [`~VocosFeatureExtractor.__call__`].
        - EnCodec variant (`codes` or `bandwidth` provided): if `audio` provided and `codes` not provided, embeddings
        are computed with the neural audio codec EnCodec [`EncodecModel`] with the given `bandwidth`; if `codes` are
        provided, the corresponding embeddings are computed for the target bandwidth.

        Args:
            audio (`np.ndarray`, `torch.Tensor`, *optional*):
                Audio input to be processed of shape `(sequence_length,)` or  `(batch_size, sequence_length)`.
            codes (`torch.LongTensor`, *optional*):
                Pre-computed EnCodec quantized codes of shape `(num_codebooks, sequence_length)` or `(num_codebooks, batch_size, sequence_length)`
            bandwidth (`float`, *optional*):
                EnCodec bandwidth [1.5, 3.0, 6.0, 12.0] kbps, this triggers EnCodec pathway when provided.
            return_tensors (`str`, defaults to `"pt"`):
                Only `"pt"` (PyTorch tensors) is supported.

        Returns:
            [`BatchFeature`]: Contains `features` tensor and optional `bandwidth` value.
        """

        output_kwargs = self._merge_kwargs(VocosProcessorKwargs, **kwargs)
        audio_kwargs = output_kwargs["audio_kwargs"]
        common_kwargs = output_kwargs["common_kwargs"]

        return_tensors = common_kwargs.pop("return_tensors", None)
        if return_tensors != "pt":
            raise ValueError(f"{self.__class__.__name__} only supports `return_tensors='pt'`.")

        self.sampling_rate = audio_kwargs["sampling_rate"]
        self.bandwidths = audio_kwargs["bandwidths"]
        if bandwidth is not None and bandwidth not in self.bandwidths:
            raise ValueError(f"bandwidth {bandwidth} is not supported, supported bandwidths are {self.bandwidths}")

        features = None
        if audio is not None:
            if bandwidth is not None:
                # encode audio as in:
                # https://github.com/gemelo-ai/vocos/blob/c859e3b7b534f3776a357983029d34170ddd6fc3/vocos/feature_extractors.py#L79
                if isinstance(audio, np.ndarray):
                    audio = torch.from_numpy(audio)
                if audio.dim() == 1:
                    audio = audio.unsqueeze(0).unsqueeze(1)
                elif audio.dim() == 2:
                    audio = audio.unsqueeze(1)
                with torch.no_grad():
                    encoded_frames = self.audio_tokenizer.encoder(audio)
                    codes = self.audio_tokenizer.quantizer.encode(encoded_frames, bandwidth=bandwidth)
            else:
                # mel spectrogram path, TODO allow tensor input for feature extractor
                if isinstance(audio, torch.Tensor):
                    audio = audio.numpy()
                features = self.feature_extractor(audio, **audio_kwargs).input_features

        if codes is not None:
            if codes.ndim not in (2, 3):
                raise ValueError(
                    f"`codes` must have shape (num_codebooks, sequence_length) or (num_codebooks, batch_size, sequence_length), but got {codes.shape}."
                )
            if codes.dim() == 2:
                # add batch dimension
                codes = codes.unsqueeze(1)

            # Extract codebook weights: https://github.com/gemelo-ai/vocos/blob/c859e3b7b534f3776a357983029d34170ddd6fc3/vocos/feature_extractors.py#L71
            num_quantizers = self.audio_tokenizer.quantizer.get_num_quantizers_for_bandwidth(max(self.bandwidths))
            codebook_weights = torch.cat(
                [layer.codebook.embed for layer in self.audio_tokenizer.quantizer.layers[:num_quantizers]], dim=0
            )
            num_bins = self.audio_tokenizer.quantizer.codebook_size
            # Embed with position https://github.com/gemelo-ai/vocos/blob/c859e3b7b534f3776a357983029d34170ddd6fc3/vocos/pretrained.py#L117
            offsets = torch.arange(0, num_bins * len(codes), num_bins, device=codes.device).reshape(-1, 1, 1)
            embeddings_idxs = codes + offsets
            features = F.embedding(embeddings_idxs, codebook_weights).sum(dim=0).transpose(1, 2)

        if features is None:
            raise ValueError("Either 'codes' or 'audio' must be provided to compute features.")

        data = {"features": features}
        if bandwidth is not None:
            data["bandwidth"] = float(bandwidth)
        return BatchFeature(data, tensor_type=return_tensors)


__all__ = ["VocosProcessor"]
