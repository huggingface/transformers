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

import numpy as np

from ...audio_processing_backends import NumpyAudioBackend
from .audio_processing_clvp import ClvpAudioProcessorMixin


class ClvpAudioProcessorNumpy(ClvpAudioProcessorMixin, NumpyAudioBackend):
    def _normalize_magnitude(self, features, *, spectrogram_config, **kwargs):
        # Compute log and mel_norms division in float64 before casting to float32
        # to match the legacy feature extractor's precision
        mel_floor = spectrogram_config.mel_floor
        features = np.log(np.maximum(mel_floor, features))
        if self.mel_norms is not None:
            features = features / np.array(self.mel_norms)[:, None]
        return features.astype(np.float32)


__all__ = ["ClvpAudioProcessorNumpy"]
