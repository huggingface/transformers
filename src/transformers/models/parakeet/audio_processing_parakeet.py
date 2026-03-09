# Copyright 2025 The HuggingFace Inc. team.
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

import librosa
import torch

from ...audio_processing_backends import TorchAudioBackend
from ...audio_utils import MelScaleConfig, SpectrogramConfig, StftConfig
from ...feature_extraction_utils import BatchFeature

LOG_ZERO_GUARD_VALUE = 2**-24
EPSILON = 1e-5


class ParakeetAudioProcessor(TorchAudioBackend):
    sample_rate = 16000
    force_mono = True
    preemphasis = 0.97
    n_fft = 512
    hop_length = 160
    win_length = 400
    n_mels = 80

    spectrogram_config = SpectrogramConfig(
        stft_config=StftConfig(
            n_fft=512,
            hop_length=160,
            win_length=400,
            window_fn="hann_window",
            periodic=False,
            pad_mode="constant",
            power=2.0,
        ),
        mel_scale_config=MelScaleConfig(
            n_mels=80,
            f_min=0.0,
            norm="slaney",
        ),
        preemphasis=0.97,
    )

    def _mel_filter_bank(self, spectrogram_config):
        """Use librosa for mel filters to match the FeatureExtractor exactly
        (mel_filter_bank uses float64 internally, causing numerical differences)."""
        msc = spectrogram_config.mel_scale_config
        return librosa.filters.mel(
            sr=self.sample_rate,
            n_fft=spectrogram_config.stft_config.n_fft,
            n_mels=msc.n_mels,
            fmin=msc.f_min,
            fmax=msc.f_max if msc.f_max is not None else self.sample_rate / 2,
            norm=msc.norm,
        )

__all__ = ["ParakeetAudioProcessor"]
