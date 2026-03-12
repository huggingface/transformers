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


class LasrAudioProcessor(TorchAudioBackend):
    sample_rate = 16000
    force_mono = True
    add_channel_dim = True
    spectrogram_config = SpectrogramConfig(
        stft_config=StftConfig(
            n_fft=512,
            hop_length=160,
            win_length=400,
            power=2.0,
            center=False,
            periodic=False,
            left_align_fft=True,
        ),
        mel_scale_config=MelScaleConfig(
            n_mels=128,
            f_min=125.0,
            f_max=7500.0,
            mel_scale="kaldi",
            triangularize_in_mel_space=True,
            bands_to_zero=1,
            computation_dtype="float64",
        ),
        log_mode="log",
        mel_floor=1e-5,
        computation_dtype="float64",
    )

    def _apply_mel_scale(self, features, *, spectrogram_config, **kwargs):
        # LASR uses (time, freq) @ (freq, mels) -> (time, mels) ordering,
        # matching the upstream FE's unfold-based output layout.
        mel_spec = torch.matmul(features.transpose(-2, -1), self.mel_filters.to(device=features.device, dtype=features.dtype))
        return torch.clamp(mel_spec, min=spectrogram_config.mel_floor)

    def _get_features_lengths(self, audio_lengths, spectrogram_config, include_center_frame=False):
        stft_cfg = spectrogram_config.stft_config
        win_length = stft_cfg.win_length or stft_cfg.n_fft
        return (audio_lengths - win_length) // stft_cfg.hop_length + 1


__all__ = ["LasrAudioProcessor"]
