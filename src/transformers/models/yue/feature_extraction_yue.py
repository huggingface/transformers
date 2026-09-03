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
"""Feature extractor class for YuE."""

import numpy as np

from ...feature_extraction_sequence_utils import SequenceFeatureExtractor
from ...feature_extraction_utils import BatchFeature
from ...utils import logging


logger = logging.get_logger(__name__)


class YuEFeatureExtractor(SequenceFeatureExtractor):
    model_input_names = ["input_values", "padding_mask"]

    def __init__(
        self,
        feature_size=1,
        sampling_rate=16000,
        padding_value=0.0,
        hop_length=320,
        **kwargs,
    ):
        super().__init__(feature_size=feature_size, sampling_rate=sampling_rate, padding_value=padding_value, **kwargs)
        self.hop_length = hop_length

    def __call__(
        self,
        raw_audio,
        padding=None,
        truncation=False,
        max_length=None,
        return_tensors="pt",
        sampling_rate=None,
        pad_to_multiple_of=None,
    ):
        if sampling_rate is not None:
            if sampling_rate != self.sampling_rate:
                raise ValueError(
                    f"Expected {self.sampling_rate} Hz audio but got {sampling_rate} Hz,"
                    f"please make sure that the provided audio input was sampled with {self.sampling_rate}."
                )

        else:
            logger.warning(
                f"It is strongly recommended to pass the `sampling_rate` argument to `{self.__class__.__name__}()`. "
                "Failing to do so can result in silent errors that might be hard to debug."
            )

        if padding and truncation:
            raise ValueError("Both padding and truncation were set. Set only one.")

        elif padding is None:
            padding = True

        is_batched = (
            isinstance(raw_audio, (list, tuple))
            and len(raw_audio) > 0
            and isinstance(raw_audio[0], (np.ndarray, list, tuple))
        )

        if is_batched:
            raw_audio = [np.asarray(_audio, dtype=np.float32) for _audio in raw_audio]

        elif not isinstance(raw_audio, np.ndarray):
            raw_audio = np.asarray(raw_audio, dtype=np.float32)

        if not is_batched:
            raw_audio = [raw_audio]

        for i, audio in enumerate(raw_audio):
            if audio.ndim > 2:
                raise ValueError(f"Expected input shape (channels, length) but got shape {audio.shape}")

            if self.feature_size == 1 and audio.ndim == 2:
                logger.warning(
                    "The model corresponding to this feature extractor expects a mono channel audio."
                    "We're averaging the audio signals into mono."
                )

                audio = np.mean(audio, -1)

            raw_audio[i] = audio

        batch = BatchFeature({"input_values": raw_audio})

        padded = self.pad(
            batch,
            max_length=max_length,
            truncation=truncation,
            padding=padding,
            return_attention_mask=True,
            pad_to_multiple_of=pad_to_multiple_of,
        )

        padded["padding_mask"] = padded.pop("attention_mask")

        values = []

        for example in padded.pop("input_values"):
            example = np.asarray(example, dtype=np.float32)
            values.append(example[None, :])
        padded["input_values"] = values

        if return_tensors is not None:
            padded = padded.convert_to_tensors(return_tensors)

        return padded
