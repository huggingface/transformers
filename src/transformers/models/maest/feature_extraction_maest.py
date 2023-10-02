# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team.
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
Feature extractor class for Music Audio Efficient Spectrogram Transformer.
"""


import numpy as np
import torch
from librosa.feature import melspectrogram

from ...utils import logging
from ..audio_spectrogram_transformer import ASTFeatureExtractor


logger = logging.get_logger(__name__)


class MAESTFeatureExtractor(ASTFeatureExtractor):
    r"""
    Constructs a Music Audio Efficient Spectrogram Transformer (MAEST) feature extractor.

    This feature extractor inherits from [`~feature_extraction_sequence_utils.SequenceFeatureExtractor`] which contains
    most of the main methods. Users should refer to this superclass for more information regarding those methods.

    This class extracts mel-filter bank features from raw audio, pads/truncates them to a fixed length and normalizes
    them using a mean and standard deviation.

    Args:
        feature_size (`int`, *optional*, defaults to 1):
            The feature dimension of the extracted features.
        sampling_rate (`int`, *optional*, defaults to 16000):
            The sampling rate at which the audio files should be digitalized expressed in hertz (Hz).
        num_mel_bins (`int`, *optional*, defaults to 96):
            Number of Mel-frequency bins.
        max_length (`int`, *optional*, defaults to 1876):
            Maximum length to which to pad/truncate the extracted features. Set to -1 to deactivate the functionallity.
        do_normalize (`bool`, *optional*, defaults to `True`):
            Whether or not to normalize the log-Mel features using `mean` and `std`.
        mean (`float`, *optional*, defaults to 2.06755686098554):
            The mean value used to normalize the log-Mel features. Uses the Discogs20 mean by default.
        std (`float`, *optional*, defaults to 1.268292820667291):
            The standard deviation value used to normalize the log-Mel features. Uses the Discogs20 standard deviation
            by default.
        return_attention_mask (`bool`, *optional*, defaults to `False`):
            Whether or not [`~ASTFeatureExtractor.__call__`] should return `attention_mask`.
        n_fft (`int`, *optional*, defaults to 512):
            Length of the FFT window.
        hop_length (`int`, *optional*, defaults to 256):
            Number of samples between successive frames.
        log_compression (`str`, *optional*, defaults to `logC`):
            Type of log compression to apply to the mel-spectrogram. Can be one of [`None`, `log`, `logC`].
    """

    model_input_names = ["input_values", "attention_mask"]

    def __init__(
        self,
        feature_size=1,
        sampling_rate=16000,
        num_mel_bins=96,
        max_length=1876,
        padding_value=0.0,
        do_normalize=True,
        mean=2.06755686098554,
        std=1.268292820667291,
        return_attention_mask=False,
        n_fft=512,
        hop_length=256,
        log_compression="logC",
        **kwargs,
    ):
        super().__init__(
            feature_size=feature_size,
            sampling_rate=sampling_rate,
            num_mel_bins=num_mel_bins,
            max_length=max_length,
            padding_value=padding_value,
            do_normalize=True,
            mean=mean,
            std=std,
            **kwargs,
        )

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.log_compression = log_compression
        self.num_mel_bins = num_mel_bins
        self.max_length = max_length
        self.do_normalize = do_normalize
        self.mean = mean
        self.std = std
        self.return_attention_mask = return_attention_mask

    def _extract_fbank_features(
        self,
        waveform: np.ndarray,
        max_length: int,
    ) -> np.ndarray:
        """
        Get mel-spectrogram features using Librosa.
        """

        melspec = melspectrogram(
            y=waveform,
            sr=self.sampling_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.num_mel_bins,
        ).T

        if not self.log_compression:
            pass
        elif self.log_compression == "log":
            melspec = np.log(melspec + np.finfo(float).eps)
        elif self.log_compression == "logC":
            melspec = np.log10(1 + melspec * 10000)
        else:
            raise ValueError(
                f"`log_compression` can only be one of [None, 'log', 'logC'], but got: {self.log_compression}"
            )

        melspec = torch.Tensor(melspec)
        n_frames = melspec.shape[0]

        if max_length > 0:
            difference = max_length - n_frames

            # pad or truncate, depending on difference
            if difference > 0:
                pad_module = torch.nn.ZeroPad2d((0, 0, 0, difference))
                melspec = pad_module(melspec)
            elif difference < 0:
                melspec = melspec[0:max_length, :]

        return melspec.numpy()
