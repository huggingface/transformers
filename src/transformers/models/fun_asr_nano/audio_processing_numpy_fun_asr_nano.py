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

from ...audio_processing_backends import NumpyAudioBackend
from .audio_processing_fun_asr_nano import FunAsrNanoAudioProcessorMixin


class FunAsrNanoAudioProcessorNumpy(FunAsrNanoAudioProcessorMixin, NumpyAudioBackend):
    def _apply_lfr(self, features):
        """See the torch sibling: stack `num_frames_lfr` frames, hop by `stride_lfr`, repeating
        the edge frames rather than zero-padding."""
        num_input_frames = features.shape[0]
        left_pad = (self.num_frames_lfr - 1) // 2
        right_pad = self.num_frames_lfr - 1 - left_pad
        padded = np.concatenate(
            [np.repeat(features[0:1], left_pad, axis=0), features, np.repeat(features[-1:], right_pad, axis=0)],
            axis=0,
        )
        num_output_frames = -(-num_input_frames // self.stride_lfr)
        required = (num_output_frames - 1) * self.stride_lfr + self.num_frames_lfr
        if required > padded.shape[0]:
            padded = np.concatenate([padded, np.repeat(padded[-1:], required - padded.shape[0], axis=0)], axis=0)
        windows = np.lib.stride_tricks.sliding_window_view(padded, self.num_frames_lfr, axis=0)
        windows = windows[:: self.stride_lfr].transpose(0, 2, 1)
        return windows.reshape(num_output_frames, -1)

    def _postprocess_output(self, output, audio_ranges=None, feature_ranges=None, **kwargs):
        # Per-clip fbank, for the reason given in the torch sibling.
        audio_values = output.pop("audio_values")

        stacked = []
        for i, (start, end) in enumerate(audio_ranges):
            waveform = audio_values[i, ..., start:end].reshape(-1)
            features = self.extract_spectrogram([waveform], spectrogram_config=self.spectrogram_config)[0]
            stacked.append(self._apply_lfr(features.T))

        frame_counts = [f.shape[0] for f in stacked]
        max_frames = max(frame_counts)
        output["audio_features"] = np.stack(
            [np.pad(f, ((0, max_frames - f.shape[0]), (0, 0))) for f in stacked]
        )
        output["audio_features_mask"] = self._get_mask([(0, n) for n in frame_counts], max_frames).astype(np.int64)
        return output


__all__ = ["FunAsrNanoAudioProcessorNumpy"]
