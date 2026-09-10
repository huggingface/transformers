# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

from ..xcodec2.audio_processing_numpy_xcodec2 import Xcodec2AudioProcessorNumpy
from .audio_processing_neucodec import NeuCodecAudioProcessorMixin


class NeuCodecAudioProcessorNumpy(NeuCodecAudioProcessorMixin, Xcodec2AudioProcessorNumpy):
    def _postprocess_output(self, output, audio_ranges=None, **kwargs):
        # See the torch sibling for why NeuCodec skips XCodec2's valid-length trim and half-hop
        # padding, and why the padding is per-utterance rather than batch-collated.
        audio_values = output["audio_values"]
        padded_length = audio_values.shape[-1]

        features = []
        for i, (start, end) in enumerate(audio_ranges):
            valid_length = min(-(-(end - start) // self.hop_length) * self.hop_length, padded_length)
            waveform = audio_values[i, 0, :valid_length]
            f = self.extract_spectrogram([waveform], spectrogram_config=self.spectrogram_config)[0].T
            f = (f - f.mean(axis=0)) / np.sqrt(f.var(axis=0, ddof=1) + 1e-7)
            features.append(f)

        frame_lengths = [f.shape[0] for f in features]
        max_frames = max(frame_lengths)
        if max_frames % self.stride:
            max_frames += self.stride - max_frames % self.stride
        batch = np.stack(
            [
                np.pad(f, ((0, max_frames - f.shape[0]), (0, 0)), constant_values=self.feature_padding_value)
                for f in features
            ]
        )
        mask = self._get_mask([(0, length) for length in frame_lengths], max_frames)

        batch_size, num_frames, num_mel_bins = batch.shape
        output["audio_features"] = batch.reshape(batch_size, num_frames // self.stride, num_mel_bins * self.stride)
        output["audio_features_mask"] = mask.reshape(batch_size, num_frames // self.stride, self.stride).min(axis=-1)
        return output


__all__ = ["NeuCodecAudioProcessorNumpy"]
