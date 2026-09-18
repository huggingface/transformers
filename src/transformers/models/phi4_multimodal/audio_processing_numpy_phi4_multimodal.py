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
from .audio_processing_phi4_multimodal import Phi4MultimodalAudioProcessorMixin


class Phi4MultimodalAudioProcessorNumpy(Phi4MultimodalAudioProcessorMixin, NumpyAudioBackend):
    def _apply_frame_processing(self, frames, *, spectrogram_config, audio_ranges=None, **kwargs):
        # Mask frames that overlap the boundary between real audio and padding
        stft_cfg = spectrogram_config.stft_config
        win_length = stft_cfg.win_length or stft_cfg.n_fft
        hop_length = stft_cfg.hop_length or win_length // 2
        batch_size = frames.shape[0]

        if audio_ranges is not None and batch_size > 1:
            audio_lengths = np.array([end - start for start, end in audio_ranges])
            to_mask_idxs = np.arange(batch_size)[audio_lengths != audio_lengths.max()]
            if to_mask_idxs.size > 0:
                frames = frames.copy()
                down = (audio_lengths[to_mask_idxs] - win_length) // hop_length + 1
                up = audio_lengths[to_mask_idxs] // hop_length - 1
                offset = down.min()
                max_idx = up.max()

                mask_range = np.arange(max_idx - offset)[None, :]
                mask = ((down - offset)[:, None] <= mask_range) & (mask_range < (up - offset)[:, None])
                block = frames[to_mask_idxs, offset:max_idx]
                frames[to_mask_idxs, offset:max_idx] = np.where(mask[..., None], 0, block)

        frames_prev = np.roll(frames, 1, axis=-1)
        frames_prev[..., 0] = frames_prev[..., 1]
        return (frames - spectrogram_config.preemphasis * frames_prev) * 32768

    def _window_and_fft(self, frames, window, frame_length, n_fft, stft_cfg, audio_dtype=None):
        frames = frames * window
        if frame_length < n_fft:
            frames = np.pad(frames, [(0, 0)] * (frames.ndim - 1) + [(0, n_fft - frame_length)])
        # Cast to complex64 before abs() to match the FE's precision path
        spec = np.fft.rfft(frames, n=n_fft).astype(np.complex64)
        if stft_cfg.normalized:
            spec = spec / np.sqrt((window**2).sum())
        return spec.swapaxes(-2, -1)


__all__ = ["Phi4MultimodalAudioProcessorNumpy"]
