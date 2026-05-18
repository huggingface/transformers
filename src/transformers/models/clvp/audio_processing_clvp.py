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


class ClvpAudioProcessor(TorchAudioBackend):
    """Torch sibling of [`ClvpAudioProcessorNumpy`]. Applies log compression and an optional
    per-mel-bin normalization (``mel_norms`` ~ per-bin stddev with implicit zero mean)."""

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
        computation_dtype="float64",
    )

    def __init__(self, mel_norms=None, **kwargs):
        super().__init__(**kwargs)
        self.mel_norms = mel_norms

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


ClvpFeatureExtractor = make_legacy_audio_processor_alias(ClvpAudioProcessor, "ClvpFeatureExtractor")


__all__ = ["ClvpAudioProcessor", "ClvpFeatureExtractor"]
