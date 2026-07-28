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
from .audio_processing_numpy_gemma3n import Gemma3nAudioProcessorMixin


class Gemma3nAudioProcessor(Gemma3nAudioProcessorMixin, TorchAudioBackend):
    """Torch sibling of [`Gemma3nAudioProcessorNumpy`]. USM-style unfold-based STFT framed
    at `win_length + 1` samples with HTK-flavor preemphasis, driven by `spectrogram_config`."""

    def _normalize_magnitude(self, features, *, spectrogram_config, **kwargs):
        result = super()._normalize_magnitude(features, spectrogram_config=spectrogram_config, **kwargs)
        # stats cast to float32 BEFORE subtracting (legacy rounding, unlike the numpy sibling)
        if self.per_bin_mean is not None:
            result = result - self.per_bin_mean.to(device=result.device, dtype=result.dtype)
        if self.per_bin_stddev is not None:
            result = result / self.per_bin_stddev.to(device=result.device, dtype=result.dtype)
        return result.to(torch.float32)


__all__ = ["Gemma3nAudioProcessor"]
