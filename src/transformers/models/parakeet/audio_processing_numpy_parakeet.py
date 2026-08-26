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
from .audio_processing_parakeet import ParakeetAudioProcessorMixin


class ParakeetAudioProcessorNumpy(ParakeetAudioProcessorMixin, NumpyAudioBackend):
    def _postprocess_output(self, output, audio_ranges=None, **kwargs):
        if audio_ranges is None or "audio_features" not in output:
            return output

        features = output["audio_features"]
        stft_cfg = self.spectrogram_config.stft_config
        audio_lengths = np.asarray([end - start for start, end in audio_ranges])
        features_lengths = np.floor_divide(
            audio_lengths + stft_cfg.n_fft // 2 * 2 - stft_cfg.n_fft, stft_cfg.hop_length
        )
        attention_mask = np.arange(features.shape[1])[None, :] < features_lengths[:, None]
        mask = np.expand_dims(attention_mask, axis=-1)
        # NumPy promotes float32 / int64 → float64; cast lengths to the feature dtype to keep
        # parity with torch (which preserves the floating dtype across float/int division).
        features_lengths_f = features_lengths.astype(features.dtype)
        mel_masked = features * mask
        mean = np.expand_dims(mel_masked.sum(axis=1) / np.expand_dims(features_lengths_f, axis=-1), axis=1)
        variance = ((mel_masked - mean) ** 2 * mask).sum(axis=1) / np.expand_dims(features_lengths_f - 1, axis=-1)
        std = np.expand_dims(np.sqrt(variance), axis=1)
        output["audio_features"] = (features - mean) / (std + 1e-5) * mask
        return output


__all__ = ["ParakeetAudioProcessorNumpy"]
