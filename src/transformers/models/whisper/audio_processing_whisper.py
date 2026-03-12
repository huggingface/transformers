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

import torch

from ...audio_processing_backends import TorchAudioBackend
from ...audio_utils import MelScaleConfig, SpectrogramConfig, StftConfig


class WhisperAudioProcessor(TorchAudioBackend):
    sample_rate = 16000
    force_mono = True
    return_attention_mask = False
    truncation = True
    max_length = 480000  # 30 seconds at 16000 Hz
    spectrogram_config = SpectrogramConfig(
        stft_config=StftConfig(
            n_fft=400,
            hop_length=160,
            power=2.0,
        ),
        mel_scale_config=MelScaleConfig(
            n_mels=80,
            mel_scale="slaney",
            norm="slaney",
            computation_dtype="float64",
        ),
        log_mode="log10",
    )

    def _normalize_magnitude(self, features, *, spectrogram_config, **kwargs):
        features = super()._normalize_magnitude(features, spectrogram_config=spectrogram_config, **kwargs)
        features = features[..., :-1]  # whisper skips last frame

        max_vals = features.amax(dim=(-2, -1), keepdim=True)
        features = torch.maximum(features, max_vals - 8.0)
        features = (features + 4.0) / 4.0

        return features

    def _apply_mel_scale(self, features, *, spectrogram_config, **kwargs):
        # Override to match WhisperFeatureExtractor's mel transformation order for numerical compatibility.
        stacked = torch.stack(features) if isinstance(features, list) else features
        mel_spec = torch.matmul(self.mel_filters.T, stacked)
        return torch.clamp(mel_spec, min=spectrogram_config.mel_floor)


__all__ = ["WhisperAudioProcessor"]
