# coding=utf-8
# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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
"""Feature extractor class for Qwen3-TTS single-codebook tokenizer."""

import math

import numpy as np

from ...feature_extraction_utils import BatchFeature
from ...utils import TensorType
from ..whisper.feature_extraction_whisper import WhisperFeatureExtractor


class Qwen3TTSTokenizerSingleCodebookFeatureExtractor(WhisperFeatureExtractor):
    r"""
    Constructs a Qwen3-TTS single-codebook feature extractor.

    This feature extractor reuses Whisper's log-mel extraction defaults while preserving the Qwen3-TTS VQ encoder's
    waveform padding rule. It returns both `input_features` for the VQ encoder and raw `input_values`/`padding_mask`
    for the x-vector/reference-mel extractor.
    """

    model_input_names = ["input_features", "feature_attention_mask", "input_values", "padding_mask"]

    def __init__(
        self,
        feature_size=128,
        sampling_rate=16000,
        hop_length=160,
        chunk_length=30,
        n_fft=400,
        padding_value=0.0,
        dither=0.0,
        return_attention_mask=True,
        audio_vq_ds_rate=2,
        **kwargs,
    ):
        super().__init__(
            feature_size=feature_size,
            sampling_rate=sampling_rate,
            hop_length=hop_length,
            chunk_length=chunk_length,
            n_fft=n_fft,
            padding_value=padding_value,
            dither=dither,
            return_attention_mask=return_attention_mask,
            **kwargs,
        )
        self.audio_vq_ds_rate = audio_vq_ds_rate

    def _pad_to_vq_multiple(self, speech: np.ndarray) -> np.ndarray:
        reduction = self.hop_length * 2 * self.audio_vq_ds_rate
        pad_length = math.ceil(len(speech) / reduction) * reduction - len(speech)
        if pad_length == 0:
            return speech
        return np.pad(speech, (0, pad_length), mode="constant", constant_values=self.padding_value)

    def __call__(
        self,
        raw_speech: np.ndarray | list[float] | list[np.ndarray] | list[list[float]],
        return_tensors: str | TensorType | None = None,
        sampling_rate: int | None = None,
        device: str | None = "cpu",
        **kwargs,
    ) -> BatchFeature:
        is_batched = isinstance(raw_speech, (list, tuple)) and len(raw_speech) > 0 and isinstance(
            raw_speech[0], (np.ndarray, list, tuple)
        )
        if is_batched:
            input_values = [np.asarray(speech, dtype=np.float32) for speech in raw_speech]
        else:
            input_values = [np.asarray(raw_speech, dtype=np.float32)]

        padded_for_vq = [self._pad_to_vq_multiple(speech) for speech in input_values]
        max_raw_length = max(speech.shape[0] for speech in input_values)
        padded_input_values = np.stack(
            [
                np.pad(speech, (0, max_raw_length - speech.shape[0]), mode="constant", constant_values=self.padding_value)
                for speech in input_values
            ]
        )
        padding_mask = np.stack(
            [
                np.pad(np.ones(speech.shape[0], dtype=np.int64), (0, max_raw_length - speech.shape[0]))
                for speech in input_values
            ]
        )

        features = super().__call__(
            padded_for_vq,
            truncation=False,
            padding="longest",
            return_tensors=None,
            return_attention_mask=True,
            sampling_rate=sampling_rate,
            device=device,
            **kwargs,
        )
        features["feature_attention_mask"] = features.pop("attention_mask")
        features["input_values"] = padded_input_values
        features["padding_mask"] = padding_mask

        if return_tensors is not None:
            features = features.convert_to_tensors(return_tensors)

        return features


__all__ = ["Qwen3TTSTokenizerSingleCodebookFeatureExtractor"]
