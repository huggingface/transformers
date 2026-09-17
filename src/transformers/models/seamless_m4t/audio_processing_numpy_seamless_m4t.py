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
from .audio_processing_seamless_m4t import SeamlessM4tAudioProcessorMixin


class SeamlessM4tAudioProcessorNumpy(SeamlessM4tAudioProcessorMixin, NumpyAudioBackend):
    def spectrogram(self, audio, *, spectrogram_config, **kwargs):
        audio = np.squeeze(audio)
        features = super().spectrogram(audio, spectrogram_config=spectrogram_config, **kwargs)
        return features.T

    def _finalize_features(self, features, feature_lengths, **kwargs):
        normalized = []
        for f in features:
            mean = np.expand_dims(f.mean(axis=0), 0)
            var = np.expand_dims(f.var(axis=0, ddof=1), 0)
            normalized.append((f - mean) / np.sqrt(var + 1e-7))
        return normalized


__all__ = ["SeamlessM4tAudioProcessorNumpy"]
