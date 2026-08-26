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
from .audio_processing_speech_to_text import SpeechToTextAudioProcessorMixin


class SpeechToTextAudioProcessorNumpy(SpeechToTextAudioProcessorMixin, NumpyAudioBackend):
    @staticmethod
    def utterance_cmvn(x, input_length, normalize_means=True, normalize_vars=True, padding_value=0.0):
        if normalize_means:
            mean = x[:input_length].mean(axis=0)
            x = np.subtract(x, mean)
        if normalize_vars:
            std = x[:input_length].std(axis=0)
            x = np.divide(x, std)
        if input_length < x.shape[0]:
            x[input_length:] = padding_value
        return x.astype(np.float32)

    def _postprocess_output(self, output, feature_ranges=None, **kwargs):
        features = output["audio_features"]
        normalized = []
        for i, (start, end) in enumerate(feature_ranges):
            length = end - start
            normalized.append(
                self.utterance_cmvn(features[i], length, self.normalize_means, self.normalize_vars, self.padding_value)
            )
        output["audio_features"] = np.stack(normalized)
        return output


__all__ = ["SpeechToTextAudioProcessorNumpy"]
