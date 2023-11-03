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

from typing import Optional, Union

import numpy as np
import torch
import torchaudio.transforms as audio_transforms

from ...feature_extraction_sequence_utils import SequenceFeatureExtractor
from ...feature_extraction_utils import BatchFeature
from ...utils import logging


logger = logging.get_logger(__name__)


class CedFeatureExtractor(SequenceFeatureExtractor):
    r"""
    CedFeatureExtractor extracts Mel spectrogram features from audio signals.

    Args:
        f_min (int, *optional*, defaults to 0): Minimum frequency for the Mel filterbank.
        sampling_rate (int, *optional*, defaults to 16000):
            Sampling rate of the input audio signal.
        win_size (int, *optional*, defaults to 512): Window size for the STFT.
        center (bool, *optional*, defaults to `True`):
            Whether to pad the signal on both sides to center it.
        n_fft (int, *optional*, defaults to 512): Number of FFT points for the STFT.
        f_max (int, optional, *optional*): Maximum frequency for the Mel filterbank.
        hop_size (int, *optional*, defaults to 160): Hop size for the STFT.
        feature_size (int, *optional*, defaults to 64): Number of Mel bands to generate.
        padding_value (float, *optional*, defaults to 0.0): Value for padding.

    Returns:
        BatchFeature: A BatchFeature object containing the extracted features.
    """

    def __init__(
        self,
        f_min: int = 0,
        sampling_rate: int = 16000,
        win_size: int = 512,
        center: bool = True,
        n_fft: int = 512,
        f_max: Optional[int] = None,
        hop_size: int = 160,
        feature_size: int = 64,
        padding_value: float = 0.0,
        **kwargs,
    ):
        super().__init__(feature_size=feature_size, sampling_rate=sampling_rate, padding_value=padding_value, **kwargs)
        self.f_min = f_min
        self.win_size = win_size
        self.center = center
        self.n_fft = n_fft
        self.f_max = f_max
        self.hop_size = hop_size

    def __call__(
        self, x: Union[np.ndarray, torch.Tensor], sampling_rate: Optional[int] = None, return_tensors="pt"
    ) -> BatchFeature:
        r"""
        Extracts Mel spectrogram features from an audio signal tensor.

        Args:
            x: Input audio signal tensor.

        Returns:
            BatchFeature: A dictionary containing the extracted features.
        """
        if sampling_rate is None:
            sampling_rate = self.sampling_rate

        if return_tensors != "pt":
            raise NotImplementedError("Only return_tensors='pt' is currently supported.")

        mel_spectrogram = audio_transforms.MelSpectrogram(
            f_min=self.f_min,
            sample_rate=sampling_rate,
            win_length=self.win_size,
            center=self.center,
            n_fft=self.n_fft,
            f_max=self.f_max,
            hop_length=self.hop_size,
            n_mels=self.feature_size,
        )
        amplitude_to_db = audio_transforms.AmplitudeToDB(top_db=120)

        x = torch.from_numpy(x).float() if isinstance(x, np.ndarray) else x.float()
        if x.dim() == 1:
            x = x.unsqueeze(0)

        x = mel_spectrogram(x)
        x = amplitude_to_db(x)
        return BatchFeature({"input_values": x})
