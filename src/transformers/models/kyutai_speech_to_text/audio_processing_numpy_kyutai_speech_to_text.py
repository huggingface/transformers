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
from .audio_processing_kyutai_speech_to_text import KyutaiSpeechToTextAudioProcessorMixin


class KyutaiSpeechToTextAudioProcessorNumpy(KyutaiSpeechToTextAudioProcessorMixin, NumpyAudioBackend):
    def _postprocess_output(self, output, **kwargs):
        pad_left = int(self.audio_silence_prefix_seconds * self.sampling_rate)
        pad_right = int((self.audio_delay_seconds + 1.0) * self.sampling_rate)

        if pad_left > 0 or pad_right > 0:
            output["audio_values"] = np.pad(
                output["audio_values"],
                [(0, 0), (0, 0), (pad_left, pad_right)],
                mode="constant",
                constant_values=0.0,
            )
            output["audio_values_mask"] = np.pad(
                output["audio_values_mask"],
                [(0, 0), (pad_left, pad_right)],
                mode="constant",
                constant_values=0,
            )

        return output


__all__ = ["KyutaiSpeechToTextAudioProcessorNumpy"]
