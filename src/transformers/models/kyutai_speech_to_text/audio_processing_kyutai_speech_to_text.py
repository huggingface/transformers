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

    def _to_batch(self, audio):
        return np.stack(audio)[:, np.newaxis, :]  # (batch, 1, length)

    def _get_mask(self, audio_ranges, padded_length, do_extract_spectrogram, spectrogram_config):
        mask = np.zeros((len(audio_ranges), padded_length), dtype=np.int32)
        for i, (start, end) in enumerate(audio_ranges):
            mask[i, start:end] = 1
        return {"audio_values_mask": mask}

    def _preprocess(self, audio, padding, max_length, truncation, pad_to_multiple_of, return_tensors, **kwargs):
        # Pad audio to batch longest
        audio, audio_ranges = self.pad(audio, padding, max_length, truncation, pad_to_multiple_of)
        padded_length = audio[0].shape[-1]

        stacked = self._to_batch(audio)
        mask_dict = self._get_mask(audio_ranges, padded_length, do_extract_spectrogram=False, spectrogram_config=None)
        audio_values_mask = mask_dict["audio_values_mask"]

        # Add silence prefix (left) and delay (right) padding
        pad_left = int(self.audio_silence_prefix_seconds * self.sample_rate)
        pad_right = int((self.audio_delay_seconds + 1.0) * self.sample_rate)

        if pad_left > 0 or pad_right > 0:
            stacked = np.pad(stacked, [(0, 0), (0, 0), (pad_left, pad_right)], mode="constant", constant_values=0.0)
            audio_values_mask = np.pad(audio_values_mask, [(0, 0), (pad_left, pad_right)], mode="constant", constant_values=0)

        return BatchFeature({"audio_values": stacked, "audio_values_mask": audio_values_mask}, tensor_type=return_tensors)


__all__ = ["KyutaiSpeechToTextAudioProcessor"]
