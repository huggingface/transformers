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
from .audio_processing_numpy_clvp import ClvpAudioProcessorNumpy


class ClvpAudioProcessor(TorchAudioBackend):
    """Torch sibling of [`ClvpAudioProcessorNumpy`]. Applies log compression and an optional
    per-mel-bin normalization (``mel_norms`` ~ per-bin stddev with implicit zero mean)."""

    sample_rate = 22050
    force_mono = True
    max_length = 132300  # 6 seconds at 22050 Hz
    truncation = True
    mask_level = "audio"

    # Single source of truth for the config lives on the numpy sibling (importable without torch).
    spectrogram_config = ClvpAudioProcessorNumpy.spectrogram_config

    def __init__(self, mel_norms=None, **kwargs):
        super().__init__(**kwargs)
        self.mel_norms = mel_norms

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
        # The legacy FE stores the STFT in a complex64 buffer before taking float64 magnitudes
        # (`np.abs(spectrogram, dtype=np.float64) ** power`). Replicate that rounding step so the
        # float64 power spectrum is bit-identical (mirrors the numpy sibling's complex64 cast).
        return stft_out.to(torch.complex64).to(torch.complex128).abs() ** power

    def _apply_mel_scale(self, features, *, spectrogram_config, **kwargs):
        # Cast mel_filters to the features' dtype so the float64 spectrogram path matches the
        # numpy sibling, which casts via `mel_filters.astype(features.dtype, copy=False)`.
        mel_filters = self.mel_filters.to(device=features.device, dtype=features.dtype)
        mel_spec = torch.nn.functional.linear(features.transpose(-2, -1), mel_filters.T).transpose(-2, -1)
        return torch.clamp(mel_spec, min=spectrogram_config.mel_floor)

    def _normalize_magnitude(self, features, *, spectrogram_config, **kwargs):
        # Compute log and mel_norms division in float64 before casting to float32
        # to match the legacy feature extractor's precision (same recipe as the numpy sibling).
        mel_floor = spectrogram_config.mel_floor
        features = torch.log(torch.maximum(torch.tensor(mel_floor, dtype=features.dtype, device=features.device), features))
        if self.mel_norms is not None:
            mel_norms = torch.as_tensor(self.mel_norms, dtype=features.dtype, device=features.device)[:, None]
            features = features / mel_norms
        return features.to(torch.float32)


__all__ = ["ClvpAudioProcessor"]
