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


class EncodecAudioProcessor(NumpyAudioBackend):
    sample_rate = 24000
    force_mono = True
    add_channel_dim = True

    def _preprocess(self, audio, padding, max_length, truncation, pad_to_multiple_of, return_tensors, **kwargs):
        lengths = [a.shape[-1] for a in audio]
        audio = self.pad(audio, padding, max_length, truncation, pad_to_multiple_of)
        padded_length = max(a.shape[-1] for a in audio)
        padding_mask = np.array([[1] * l + [0] * (padded_length - l) for l in lengths])
        stacked = np.stack(audio)[:, np.newaxis, :]  # (batch, 1, length)
        output = BatchFeature({"audio_values": stacked, "padding_mask": padding_mask}, tensor_type=return_tensors)
        return output


__all__ = ["EncodecAudioProcessor"]
