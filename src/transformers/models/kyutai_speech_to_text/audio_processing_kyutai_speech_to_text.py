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


class KyutaiSpeechToTextAudioProcessor(NumpyAudioBackend):
    sample_rate = 24000
    force_mono = True
    add_channel_dim = True
    audio_delay_seconds = 2.5
    audio_silence_prefix_seconds = 1.0

    def _postprocess_output(self, output, **kwargs):
        # Add silence prefix (left) and delay (right) padding
        pad_left = int(self.audio_silence_prefix_seconds * self.sample_rate)
        pad_right = int((self.audio_delay_seconds + 1.0) * self.sample_rate)

        if pad_left > 0 or pad_right > 0:
            output["audio_values"] = np.pad(
                output["audio_values"], [(0, 0), (0, 0), (pad_left, pad_right)], mode="constant", constant_values=0.0,
            )
            output["audio_values_mask"] = np.pad(
                output["audio_values_mask"], [(0, 0), (pad_left, pad_right)], mode="constant", constant_values=0,
            )

        return output


__all__ = ["KyutaiSpeechToTextAudioProcessor"]
