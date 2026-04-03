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


class VoxtralRealtimeAudioProcessor(TorchAudioBackend):
    sample_rate = 16000
    force_mono = True
    spectrogram_config = SpectrogramConfig(
        stft_config=StftConfig(
            n_fft=400,
            hop_length=160,
            power=2.0,
        ),
        mel_scale_config=MelScaleConfig(
            n_mels=128,
            mel_scale="slaney",
            norm="slaney",
            computation_dtype="float64",
        ),
        log_mode="log10",
        skip_last_frame=True,
    )
    global_log_mel_max = 1.5

    def _apply_mel_scale(self, features, *, spectrogram_config, **kwargs):
        mel_filters = self.mel_filters.to(device=features.device)
        return torch.clamp(torch.matmul(mel_filters.T, features), min=spectrogram_config.mel_floor)

    def _normalize_magnitude(self, features, *, spectrogram_config, **kwargs):
        features = super()._normalize_magnitude(features, spectrogram_config=spectrogram_config, **kwargs)

        if self.global_log_mel_max is not None:
            spec_max = torch.tensor(self.global_log_mel_max, device=features.device, dtype=features.dtype)
        else:
            spec_max = features.amax(dim=(-2, -1), keepdim=True)
        features = torch.maximum(features, spec_max - 8.0)
        features = (features + 4.0) / 4.0
        return features

    def _get_features_lengths(self, audio_lengths, spectrogram_config, include_center_frame=False):
        stft_cfg = spectrogram_config.stft_config
        win_length = stft_cfg.win_length or stft_cfg.n_fft
        return (audio_lengths - win_length) // stft_cfg.hop_length + 1


__all__ = ["VoxtralRealtimeAudioProcessor"]
