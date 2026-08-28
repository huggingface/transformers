# Copyright 2026 The HuggingFace Inc. team.
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

import numpy as np

from ...audio_processing_backends import NumpyAudioBackend
from .audio_processing_univnet import UnivNetAudioProcessorMixin


class UnivNetAudioProcessorNumpy(UnivNetAudioProcessorMixin, NumpyAudioBackend):
    def _reflect_pad(self, audio, pad_amount):
        pad_width = [(0, 0)] * (audio.ndim - 1) + [(pad_amount, pad_amount)]
        return np.pad(audio, pad_width, mode="reflect")

    def _compute_magnitudes(self, stft_out, power, spectrogram_config=None):
        return np.sqrt(np.real(stft_out) ** 2 + np.imag(stft_out) ** 2 + self.magnitude_floor)

    def _apply_mel_scale(self, features, *, spectrogram_config, **kwargs):
        # No mel-scale clamp: UnivNet's floor is already inside the magnitude sqrt
        return np.matmul(self.mel_filters.T, features)


__all__ = ["UnivNetAudioProcessorNumpy"]
