# Copyright 2026 The HuggingFace Inc. team.
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
from .audio_processing_numpy_gemma4 import Gemma4AudioProcessorMixin


class Gemma4AudioProcessor(Gemma4AudioProcessorMixin, TorchAudioBackend):
    """Torch sibling of [`Gemma4AudioProcessorNumpy`]. See the mixin for the pipeline."""

    def _apply_mel_scale(self, features, *, spectrogram_config, **kwargs):
        # cast the float64 filters to the float32 features dtype (legacy torch behavior)
        mel_filters = self.mel_filters.to(device=features.device, dtype=features.dtype)
        return torch.matmul(features.transpose(-2, -1), mel_filters)

    def _postprocess_output(self, output, audio_ranges=None, **kwargs):
        if audio_ranges is None or "audio_features" not in output:
            return output
        features = output["audio_features"]
        # same arithmetic as the numpy sibling; per-backend only for the device move
        if self.per_bin_mean is not None:
            features = features - self.per_bin_mean.to(device=features.device, dtype=features.dtype)
        if self.per_bin_stddev is not None:
            features = features / self.per_bin_stddev.to(device=features.device, dtype=features.dtype)
        mask = output.get("audio_features_mask")
        if mask is not None:
            features = features * mask.to(features.dtype).unsqueeze(-1)
        output["audio_features"] = features
        return output


__all__ = ["Gemma4AudioProcessor"]
