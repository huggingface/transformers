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
from ...feature_extraction_utils import BatchFeature


class KyutaiSpeechToTextAudioProcessor(NumpyAudioBackend):
    sample_rate = 24000
    force_mono = True
    add_channel_dim = True

    def __init__(self, audio_delay_seconds=2.5, audio_silence_prefix_seconds=1.0, **kwargs):
        self.audio_delay_seconds = audio_delay_seconds
        self.audio_silence_prefix_seconds = audio_silence_prefix_seconds
        super().__init__(**kwargs)

    def _preprocess(self, audio, padding, max_length, truncation, pad_to_multiple_of, return_tensors, **kwargs):
        # Track lengths for padding_mask
        lengths = [a.shape[-1] for a in audio]

        # Pad audio to batch longest
        audio = self.pad(audio, padding, max_length, truncation, pad_to_multiple_of)
        padded_length = max(a.shape[-1] for a in audio)

        # Create padding_mask (1 for real audio, 0 for padding)
        padding_mask = np.array([[1] * l + [0] * (padded_length - l) for l in lengths])

        # Stack audio with channel dim
        stacked = np.stack(audio)[:, np.newaxis, :]  # (batch, 1, length)

        # Add silence prefix (left) and delay (right) padding
        pad_left = int(self.audio_silence_prefix_seconds * self.sample_rate)
        pad_right = int((self.audio_delay_seconds + 1.0) * self.sample_rate)

        if pad_left > 0 or pad_right > 0:
            # Pad audio
            audio_pad_width = [(0, 0), (0, 0), (pad_left, pad_right)]
            stacked = np.pad(stacked, audio_pad_width, mode="constant", constant_values=0.0)

            # Pad padding_mask
            mask_pad_width = [(0, 0), (pad_left, pad_right)]
            padding_mask = np.pad(padding_mask, mask_pad_width, mode="constant", constant_values=0)

        output = BatchFeature({"audio_values": stacked, "padding_mask": padding_mask}, tensor_type=return_tensors)
        return output


__all__ = ["KyutaiSpeechToTextAudioProcessor"]
