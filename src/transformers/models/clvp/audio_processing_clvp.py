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

import numpy as np

from ...audio_processing_backends import NumpyAudioBackend
from ...audio_utils import MelScaleConfig, SpectrogramConfig, StftConfig


class ClvpAudioProcessor(NumpyAudioBackend):
    sample_rate = 22050
    force_mono = True
    max_length = 132300  # 6 seconds at 22050 Hz
    truncation = True
    mask_level = "audio"

    spectrogram_config = SpectrogramConfig(
        stft_config=StftConfig(
            n_fft=1024,
            hop_length=256,
            window_fn="hann_window",
            power=2.0,
        ),
        mel_scale_config=MelScaleConfig(
            n_mels=80,
            f_min=0.0,
            f_max=8000.0,
            norm="slaney",
            mel_scale="htk",
            frequency_bin_mode="linspace",
        ),
        log_mode="log",
        mel_floor=1e-5,
    )

    def __init__(self, mel_norms=None, **kwargs):
        super().__init__(**kwargs)
        self.mel_norms = mel_norms

    def _normalize_magnitude(self, features, *, spectrogram_config, **kwargs):
        # Compute log and mel_norms division in float64 before casting to float32
        # to match the legacy feature extractor's precision
        mel_floor = spectrogram_config.mel_floor
        features = np.log(np.maximum(mel_floor, features))
        if self.mel_norms is not None:
            features = features / np.array(self.mel_norms)[:, None]
        return features.astype(np.float32)

__all__ = ["ClvpAudioProcessor"]
