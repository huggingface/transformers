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
from .audio_processing_numpy_clvp import ClvpAudioProcessorNumpy


class ClvpAudioProcessor(TorchAudioBackend):
    """Torch sibling of [`ClvpAudioProcessorNumpy`]. Applies log compression and an optional
    per-mel-bin normalization (``mel_norms`` ~ per-bin stddev with implicit zero mean)."""

    sample_rate = 22050
    force_mono = True
    max_length = 132300  # 6 seconds at 22050 Hz
    truncation = True
    mask_level = "audio"


    spectrogram_config = ClvpAudioProcessorNumpy.spectrogram_config

    def __init__(self, mel_norms=None, **kwargs):
        super().__init__(**kwargs)
        self.mel_norms = mel_norms

    # Mel filters: the base dispatcher resolves the top-level `computation_dtype="float64"`
    # into float64 torch-native filters (matching the legacy FE's float64 numpy build
    # within ~1e-16), kept float64 for the mel matmul below.

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
