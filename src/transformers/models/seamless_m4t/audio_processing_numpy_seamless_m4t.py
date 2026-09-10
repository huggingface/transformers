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
from .audio_processing_seamless_m4t import SeamlessM4tAudioProcessorMixin


class SeamlessM4tAudioProcessorNumpy(SeamlessM4tAudioProcessorMixin, NumpyAudioBackend):
    def extract_spectrogram(self, audio, **kwargs):
        features = []
        for waveform in audio:
            waveform = np.squeeze(waveform)
            f = super().extract_spectrogram([waveform], spectrogram_config=self.spectrogram_config)
            features.append(f[0].T)
        return features

    def _postprocess_features(self, features, feature_lengths):
        normalized = []
        for f in features:
            mean = np.expand_dims(f.mean(axis=0), 0)
            var = np.expand_dims(f.var(axis=0, ddof=1), 0)
            normalized.append((f - mean) / np.sqrt(var + 1e-7))
        return normalized

    def _postprocess_output(self, output, feature_ranges=None, **kwargs):
        features = output["audio_features"]
        batch_size, num_frames, num_channels = features.shape

        remainder = num_frames % self.stride
        if remainder != 0:
            features = features[:, : num_frames - remainder, :]
            num_frames = num_frames - remainder

        output["audio_features"] = features.reshape(batch_size, num_frames // self.stride, num_channels * self.stride)

        if "audio_features_mask" in output:
            mask = output["audio_features_mask"]
            if remainder != 0:
                mask = mask[:, :num_frames]
            indices = np.arange(0, num_frames)
            output["audio_features_mask"] = mask[:, indices % self.stride == 1]

        return output


__all__ = ["SeamlessM4tAudioProcessorNumpy"]
