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

from ...audio_utils import AudioInput, make_list_of_audio
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
            "padding": True,
        },
        "common_kwargs": {"return_tensors": "pt"},
    }


class VocosEncodecProcessor(ProcessorMixin):
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

    def __call__(
        self,
        audio: Optional[AudioInput] = None,
        codes=None,
        bandwidth: Optional[float] = None,
        return_tensors: str = "pt",
        device: Optional[str] = None,
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
                Audio input to be processed of shape `(sequence_length,)` or `(batch_size, sequence_length)`.
            codes (`torch.LongTensor`, *optional*):
                Pre-computed EnCodec quantized codes of shape `(num_codebooks, sequence_length)` or `(num_codebooks, batch_size, sequence_length)`
            bandwidth (`float`, *optional*):
                EnCodec bandwidth [1.5, 3.0, 6.0, 12.0] kbps, this triggers EnCodec pathway when provided.
            return_tensors (`str`, defaults to `"pt"`):
                Only `"pt"` (PyTorch tensors) is supported.
            device (`str`, *optional*):
                Device on to compute mel spectrogram. If left to `None`, uses the device of the first input
                `audio` element, or CPU if the input is a numpy array.

        Returns:
            [`BatchFeature`]: Contains `audio_spectrogram` or `input_features` tensor for the model input.
        """

        output_kwargs = self._merge_kwargs(VocosProcessorKwargs, **kwargs)
        audio_kwargs = output_kwargs["audio_kwargs"]
        return_tensors = audio_kwargs.get("return_tensors", None)
        if return_tensors != "pt":
            raise ValueError(f"{self.__class__.__name__} only supports `return_tensors='pt'`.")

        self.sampling_rate = audio_kwargs["sampling_rate"]
        self.bandwidths = audio_kwargs["bandwidths"]
        if bandwidth is not None and bandwidth not in self.bandwidths:
            raise ValueError(f"bandwidth {bandwidth} is not supported, supported bandwidths are {self.bandwidths}")

        # Prepare model inputs
        audio_spectrogram = None
        input_features = None
        padding_mask = None
        if audio is not None:
            if bandwidth is not None:
                # pad audio into batch
                pad_to_multiple_of = (
                    None if len(make_list_of_audio(audio)) == 1 else self.audio_tokenizer.config.hop_length
                )
                fe_outputs = self.feature_extractor(
                    audio,
                    return_audio_only=True,
                    pad_to_multiple_of=pad_to_multiple_of,
                    device=device,
                    **audio_kwargs,
                )
                audio = fe_outputs["audio"]

                # encode audio as in original:
                # https://github.com/gemelo-ai/vocos/blob/c859e3b7b534f3776a357983029d34170ddd6fc3/vocos/feature_extractors.py#L79
                audio = audio.unsqueeze(1)
                with torch.no_grad():
                    encoded_frames = self.audio_tokenizer.encoder(audio.to(self.audio_tokenizer.device))
                    codes = self.audio_tokenizer.quantizer.encode(encoded_frames, bandwidth=bandwidth)
            else:
                fe_outputs = self.feature_extractor(audio, device=device, **audio_kwargs)
                audio_spectrogram = fe_outputs.audio_spectrogram
            padding_mask = fe_outputs["padding_mask"]

        if codes is not None:
            if codes.ndim not in (2, 3):
                raise ValueError(
                    f"`codes` must have shape (num_codebooks, sequence_length) or (num_codebooks, batch_size, sequence_length), but got {codes.shape}."
                )
            if codes.dim() == 2:
                # add batch dimension
                codes = codes.unsqueeze(1)
            if bandwidth is None:
                raise ValueError("When passing `codes`, `bandwidth` must be also be provided.")

            # Extract codebook weights: https://github.com/gemelo-ai/vocos/blob/c859e3b7b534f3776a357983029d34170ddd6fc3/vocos/feature_extractors.py#L71
            num_quantizers = self.audio_tokenizer.quantizer.get_num_quantizers_for_bandwidth(max(self.bandwidths))
            codebook_weights = torch.cat(
                [layer.codebook.embed for layer in self.audio_tokenizer.quantizer.layers[:num_quantizers]], dim=0
            ).to(codes.device)
            num_bins = self.audio_tokenizer.quantizer.codebook_size
            # Embed with position https://github.com/gemelo-ai/vocos/blob/c859e3b7b534f3776a357983029d34170ddd6fc3/vocos/pretrained.py#L117
            offsets = torch.arange(0, num_bins * len(codes), num_bins, device=codes.device).reshape(-1, 1, 1)
            embeddings_idxs = codes + offsets
            input_features = F.embedding(embeddings_idxs, codebook_weights).sum(dim=0).transpose(1, 2)

        if input_features is not None:
            data = {"input_features": input_features, "bandwidth": float(bandwidth)}
        elif audio_spectrogram is not None:
            data = {"audio_spectrogram": audio_spectrogram}
        else:
            raise ValueError("Either 'codes' or 'audio' must be provided to compute features.")
        if padding_mask is not None:
            data["padding_mask"] = padding_mask

        return BatchFeature(data, tensor_type=return_tensors)


__all__ = ["VocosEncodecProcessor"]
