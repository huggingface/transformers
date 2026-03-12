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

from spectrograms import numpy_mel_spectrogram as _np_spec

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

    def _mel_filter_bank(self, spectrogram_config):
        stft_cfg = spectrogram_config.stft_config
        mel_cfg = spectrogram_config.mel_scale_config
        mel_filters_np = _np_spec.mel_filter_bank(
            num_frequency_bins=1 + stft_cfg.n_fft // 2,
            num_mel_filters=mel_cfg.n_mels,
            min_frequency=mel_cfg.f_min,
            max_frequency=mel_cfg.f_max if mel_cfg.f_max is not None else self.sample_rate / 2,
            sampling_rate=self.sample_rate,
            norm=mel_cfg.norm,
            mel_scale=mel_cfg.mel_scale,
            triangularize_in_mel_space=mel_cfg.triangularize_in_mel_space,
        )
        return torch.from_numpy(mel_filters_np).to(torch.float32)

    def _apply_mel_scale(self, features, *, spectrogram_config, **kwargs):
        """
        Override to use the same matrix multiplication order as WhisperFeatureExtractor
        for exact numerical compatibility. FeatureExtractor uses (n_mels, n_freq) @ (n_freq, time),
        while the generic spectrograms module uses (time, n_freq) @ (n_freq, n_mels) then transpose.
        The different summation order produces slightly different rounding (1 ULP).
        """
        stacked = torch.stack(features) if isinstance(features, list) else features
        mel_spec = torch.matmul(self.mel_filters.T, stacked)
        return torch.clamp(mel_spec, min=spectrogram_config.mel_floor)


__all__ = ["WhisperAudioProcessor"]
