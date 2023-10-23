# coding=utf-8
# Copyright 2023 Xiaomi Corporation and The HuggingFace Inc. team.
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
"""
Feature extractor class for CED.
"""

from typing import Optional

import torch
import torchaudio.transforms as audio_transforms

from ...feature_extraction_sequence_utils import SequenceFeatureExtractor
from ...feature_extraction_utils import BatchFeature
from ...utils import logging


logger = logging.get_logger(__name__)


class CedFeatureExtractor(SequenceFeatureExtractor):
    r"""
    TODO: Add docstring
    """

    def __init__(
        self,
        f_min: int = 0,
        sample_rate: int = 16000,
        win_size: int = 512,
        center: bool = True,
        n_fft: int = 512,
        f_max: Optional[int] = None,
        hop_size: int = 160,
        n_mels: int = 64,
        **kwargs,
    ):
        r"""
        TODO: Add docstring
        """
        super().__init__(feature_size=n_mels, sampling_rate=sample_rate, padding_value=0.0, **kwargs)
        self.f_min = f_min
        self.sample_rate = sample_rate
        self.win_size = win_size
        self.center = center
        self.n_fft = n_fft
        self.f_max = f_max
        self.hop_size = hop_size
        self.n_mels = n_mels

    def __call__(self, x: torch.Tensor) -> BatchFeature:
        r"""
        TODO: Add docstring
        """
        mel_spectrogram = audio_transforms.MelSpectrogram(
            f_min=self.f_min,
            sample_rate=self.sample_rate,
            win_length=self.win_size,
            center=self.center,
            n_fft=self.n_fft,
            f_max=self.f_max,
            hop_length=self.hop_size,
            n_mels=self.n_mels,
        )
        amplitude_to_db = audio_transforms.AmplitudeToDB(top_db=120)

        x = mel_spectrogram(x)
        x = amplitude_to_db(x)
        return BatchFeature({"input_values": x})
