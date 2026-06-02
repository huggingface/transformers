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
from ...audio_processing_base import make_legacy_audio_processor_alias
from ...audio_utils import MelScaleConfig, SpectrogramConfig, StftConfig


class UnivNetAudioProcessor(TorchAudioBackend):
    """Torch sibling of [`UnivNetAudioProcessorNumpy`]. Reflect-padded STFT with mel_floor
    added inside the magnitude sqrt, no mel-floor clamp, and a `(frames, n_mels)` output layout."""

    sample_rate = 24000
    force_mono = True
    mask_level = "audio"
    mel_floor = 1e-9
    compression_clip_val = 1e-5
    compression_factor = 1.0
    do_normalize = False
    normalize_min = -11.512925148010254
    normalize_max = 2.3143386840820312
    max_length_s = 10
    spectrogram_config = SpectrogramConfig(
        stft_config=StftConfig(
            n_fft=1024,
            hop_length=256,
            center=False,
            window_fn="hann",
            periodic=True,
            power=1.0,
        ),
        mel_scale_config=MelScaleConfig(
            n_mels=100,
            f_min=0.0,
            f_max=12000.0,
            mel_scale="slaney",
            norm="slaney",
        ),
        log_mode="log",
        mel_floor=1e-5,
        computation_dtype="float64",
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.num_max_samples = self.max_length_s * self.sample_rate

    def _stft(self, audio, *, spectrogram_config, **kwargs):
        # UnivNet uses reflect padding with (n_fft - hop_length) / 2 instead of center padding
        stft_cfg = spectrogram_config.stft_config
        pad_amount = int((stft_cfg.n_fft - stft_cfg.hop_length) / 2)
        # `torch.nn.functional.pad` reflects on the last dim by default; works for 1D or 2D.
        audio = torch.nn.functional.pad(audio, (pad_amount, pad_amount), mode="reflect")
        return super()._stft(audio, spectrogram_config=spectrogram_config, **kwargs)

    def _compute_magnitudes(self, stft_out, power, spectrogram_config=None):
        # UnivNet adds mel_floor inside the sqrt: sqrt(real² + imag² + mel_floor)
        return torch.sqrt(stft_out.real ** 2 + stft_out.imag ** 2 + self.mel_floor)

    def _apply_mel_scale(self, features, *, spectrogram_config, **kwargs):
        # UnivNet applies mel filterbank without a floor.
        # `mel_filters` is shape `(n_freq, n_mels)`; transposing gives `(n_mels, n_freq)`.
        mel_filters = self.mel_filters.to(device=features.device, dtype=features.dtype)
        return torch.matmul(mel_filters.T, features)

    def _normalize_magnitude(self, features, *, spectrogram_config, **kwargs):
        features = super()._normalize_magnitude(features, spectrogram_config=spectrogram_config, **kwargs)
        if self.do_normalize:
            features = 2 * ((features - self.normalize_min) / (self.normalize_max - self.normalize_min)) - 1
        return features

    def extract_spectrogram(self, audio, *, spectrogram_config, **kwargs):
        features = super().extract_spectrogram(audio, spectrogram_config=spectrogram_config, **kwargs)
        # Transpose from (..., n_mels, frames) to (..., frames, n_mels)
        if isinstance(features, list):
            return [f.transpose(-2, -1) for f in features]
        return features.transpose(-2, -1)


UnivNetFeatureExtractor = make_legacy_audio_processor_alias(UnivNetAudioProcessor, "UnivNetFeatureExtractor")


__all__ = ["UnivNetAudioProcessor", "UnivNetFeatureExtractor"]
