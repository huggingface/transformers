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
from .audio_processing_gemma4_unified import Gemma4UnifiedAudioProcessorMixin


class Gemma4UnifiedAudioProcessorNumpy(Gemma4UnifiedAudioProcessorMixin, NumpyAudioBackend):
    def _chunk_waveform(self, waveform):
        """Chunk a 1-D waveform into fixed-length frames of `audio_samples_per_token`
        samples, zero-padding the tail so the last (partial) frame is kept."""
        pad_len = (-waveform.shape[-1]) % self.audio_samples_per_token
        if pad_len:
            waveform = np.pad(waveform, (0, pad_len))
        num_tokens = waveform.shape[-1] // self.audio_samples_per_token
        return waveform.reshape(num_tokens, self.audio_samples_per_token).astype(np.float32)


__all__ = ["Gemma4UnifiedAudioProcessorNumpy"]
