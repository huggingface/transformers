# Copyright 2026 IBM and The HuggingFace Team. All rights reserved.
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
from .audio_processing_granite_speech5 import GraniteSpeech5AudioProcessorMixin


class GraniteSpeech5AudioProcessorNumpy(GraniteSpeech5AudioProcessorMixin, NumpyAudioBackend):
    def _compute_deltas(self, features):
        """See the torch sibling: torchaudio's delta filter, implemented natively."""
        n = (self.delta_win_length - 1) // 2
        denominator = n * (n + 1) * (2 * n + 1) / 3
        padded = np.pad(features, ((0, 0), (0, 0), (n, n)), mode="edge")
        kernel = np.arange(-n, n + 1, dtype=features.dtype)
        # Correlate along time for each mel bin; `kernel` is applied in index order, matching
        # conv1d with a pre-reversed kernel in the torch path.
        windows = np.lib.stride_tricks.sliding_window_view(padded, kernel.size, axis=-1)
        return (windows * kernel).sum(-1) / denominator

    def _postprocess_output(self, output, audio_ranges=None, **kwargs):
        logmel = output.pop("audio_features")
        stacking = self.frame_stacking

        num_frames = stacking * -(-(logmel.shape[-1] - 1) // stacking)
        if logmel.shape[-1] < num_frames:
            logmel = np.pad(logmel, ((0, 0), (0, 0), (0, num_frames - logmel.shape[-1])))
        else:
            logmel = logmel[..., :num_frames]

        logmel = np.concatenate((logmel, self._compute_deltas(logmel)), axis=-2)
        logmel = np.swapaxes(logmel, -1, -2)
        batch_size = logmel.shape[0]
        output["audio_features"] = logmel.reshape(batch_size, -1, stacking * logmel.shape[-1])

        if audio_ranges is not None:
            hop = self.spectrogram_config.stft_config.hop_length
            lengths = np.array([-(-((end - start) // hop) // stacking) for start, end in audio_ranges])
            max_frames = output["audio_features"].shape[1]
            output["audio_features_mask"] = (np.arange(max_frames)[None, :] < lengths[:, None]).astype(np.int64)
        return output


__all__ = ["GraniteSpeech5AudioProcessorNumpy"]
