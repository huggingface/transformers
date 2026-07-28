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
from .audio_processing_numpy_clap import ClapAudioProcessorMixin


class ClapAudioProcessor(ClapAudioProcessorMixin, TorchAudioBackend):
    """Torch sibling of [`ClapAudioProcessorNumpy`]. See the mixin for the pipeline."""

    def _native_stft(self, audio, window, frame_length, hop_length, n_fft, stft_cfg):
        stft_out = super()._native_stft(audio, window, frame_length, hop_length, n_fft, stft_cfg)
        # round-trip through complex64 like the legacy FE, so float64 magnitudes match bit-exactly
        return stft_out.to(torch.complex64).to(torch.complex128)

    def _apply_mel_scale(self, features, *, spectrogram_config, **kwargs):
        # cast mel_filters to the features' dtype, matching the numpy sibling's float64 path
        mel_filters = self.mel_filters.to(device=features.device, dtype=features.dtype)
        mel_spec = torch.nn.functional.linear(features.transpose(-2, -1), mel_filters.T).transpose(-2, -1)
        return torch.clamp(mel_spec, min=spectrogram_config.mel_floor)

    def _bilinear_shrink(self, mel, chunk_frames):
        # legacy torch dtype path: round-trip through float32 (numpy sibling stays float64)
        mel_tensor = mel.unsqueeze(0).unsqueeze(0).to(torch.float32)
        mel_shrink = torch.nn.functional.interpolate(
            mel_tensor, size=[chunk_frames, 64], mode="bilinear", align_corners=False
        )
        return mel_shrink[0][0].to(mel.dtype)


__all__ = ["ClapAudioProcessor"]
