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

import numpy as np

from ...audio_processing_backends import NumpyAudioBackend
from ...audio_utils import _clamp_min
from .audio_processing_voxtral_realtime import VoxtralRealtimeAudioProcessorMixin


class VoxtralRealtimeAudioProcessorNumpy(VoxtralRealtimeAudioProcessorMixin, NumpyAudioBackend):
    def _apply_mel_scale(self, features, *, spectrogram_config, **kwargs):
        mel_filters = self.mel_filters.astype(features.dtype, copy=False)
        return _clamp_min(np.matmul(mel_filters.T, features), spectrogram_config.mel_floor)


__all__ = ["VoxtralRealtimeAudioProcessorNumpy"]
