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
from ...audio_utils import mel_filter_bank
from .audio_processing_numpy_univnet import UnivNetAudioProcessorNumpy


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
    # Single source of truth for the config lives on the numpy sibling (importable without torch).
    spectrogram_config = UnivNetAudioProcessorNumpy.spectrogram_config

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

    def _mel_filter_bank(self, spectrogram_config):
        # The legacy FE builds its filters with the numpy `mel_filter_bank` in float64; the torch
        # backend's default builds them in float32. Reuse the numpy path (same as the numpy
        # sibling) so the float64 filter values are bit-identical to the legacy extractor's.
        stft_cfg = spectrogram_config.stft_config
        mel_cfg = spectrogram_config.mel_scale_config
        filters = mel_filter_bank(
            num_frequency_bins=1 + stft_cfg.n_fft // 2,
            num_mel_filters=mel_cfg.n_mels,
            min_frequency=mel_cfg.f_min,
            max_frequency=mel_cfg.f_max if mel_cfg.f_max is not None else self.sample_rate / 2,
            sampling_rate=self.sample_rate,
            norm=mel_cfg.norm,
            mel_scale=mel_cfg.mel_scale,
        )
        return torch.from_numpy(filters)

    def _compute_magnitudes(self, stft_out, power, spectrogram_config=None):
        # UnivNet adds mel_floor inside the sqrt: sqrt(real² + imag² + mel_floor).
        # The legacy FE stores the STFT in a complex64 buffer and takes the sqrt in float32
        # (mirrors the numpy sibling's complex64 cast). torch's float32 sqrt is not correctly
        # rounded on all platforms, so round via a float64 sqrt (bit-identical to np.sqrt on
        # float32 inputs), then promote back to float64 for the mel matmul the way numpy's
        # float32 x float64 matmul promotion does.
        stft_out = stft_out.to(torch.complex64)
        presqrt = stft_out.real ** 2 + stft_out.imag ** 2 + self.mel_floor
        return presqrt.double().sqrt().float().double()

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


__all__ = ["UnivNetAudioProcessor"]
