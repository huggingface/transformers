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
from .audio_processing_xcodec2 import Xcodec2AudioProcessorMixin


class Xcodec2AudioProcessorNumpy(Xcodec2AudioProcessorMixin, NumpyAudioBackend):
    def _standardize_frames(self, features):
        return (features - features.mean(0)) / np.sqrt(features.var(0, ddof=1) + 1e-7)


__all__ = ["Xcodec2AudioProcessorNumpy"]
