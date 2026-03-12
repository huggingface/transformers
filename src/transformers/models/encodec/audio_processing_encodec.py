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


class EncodecAudioProcessor(NumpyAudioBackend):
    sample_rate = 24000
    force_mono = True
    add_channel_dim = True

    def _to_batch(self, audio):
        return np.stack(audio)[:, np.newaxis, :]  # (batch, 1, length)

    def _get_mask(self, audio_ranges, padded_length, do_extract_spectrogram, spectrogram_config):
        mask = np.zeros((len(audio_ranges), padded_length), dtype=np.int32)
        for i, (start, end) in enumerate(audio_ranges):
            mask[i, start:end] = 1
        return {"audio_values_mask": mask}


__all__ = ["EncodecAudioProcessor"]
