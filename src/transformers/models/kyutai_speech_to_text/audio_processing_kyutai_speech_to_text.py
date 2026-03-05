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

    def __init__(self, audio_delay_seconds=2.5, audio_silence_prefix_seconds=1.0, **kwargs):
        self.audio_delay_seconds = audio_delay_seconds
        self.audio_silence_prefix_seconds = audio_silence_prefix_seconds
        super().__init__(**kwargs)

    def _preprocess(self, audio, padding, max_length, truncation, pad_to_multiple_of, return_tensors, **kwargs):
        result = super()._preprocess(
            audio, padding, max_length, truncation, pad_to_multiple_of, return_tensors, **kwargs
        )

        pad_left = int(self.audio_silence_prefix_seconds * self.sample_rate)
        pad_right = int((self.audio_delay_seconds + 1.0) * self.sample_rate)

        if pad_left > 0 or pad_right > 0:
            output_key = self.model_input_names[0]
            data = result[output_key]

            if isinstance(data, np.ndarray):
                pad_width = [(0, 0)] * (data.ndim - 1) + [(pad_left, pad_right)]
                result[output_key] = np.pad(data, pad_width, mode="constant", constant_values=0.0)
            else:
                import torch.nn.functional as F

                result[output_key] = F.pad(data, (pad_left, pad_right), mode="constant", value=0.0)

        return result


__all__ = ["KyutaiSpeechToTextAudioProcessor"]
