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
from .audio_processing_musicgen_melody import MusicgenMelodyAudioProcessorMixin


class MusicgenMelodyAudioProcessorNumpy(MusicgenMelodyAudioProcessorMixin, NumpyAudioBackend):
    def extract_spectrogram(self, audio, **kwargs):
        waveform = self._pad_for_fft(audio)
        # normalized power spectrogram, matching `torchaudio.transforms.Spectrogram(normalized=True)`
        spec = self._stft(waveform, spectrogram_config=self.power_spectrogram_config)

        raw_chroma = np.matmul(self.chroma_filters, spec)
        # inf-norm over the chroma axis, as `F.normalize(p=inf, dim=-2, eps=1e-6)` does
        denom = np.maximum(np.abs(raw_chroma).max(axis=-2, keepdims=True), 1e-6)
        norm_chroma = np.swapaxes(raw_chroma / denom, 1, 2)

        idx = norm_chroma.argmax(-1)
        one_hot = np.zeros_like(norm_chroma)
        np.put_along_axis(one_hot, idx[..., None], 1.0, axis=-1)
        return one_hot


__all__ = ["MusicgenMelodyAudioProcessorNumpy"]
