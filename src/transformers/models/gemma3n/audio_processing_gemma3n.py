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
from ...audio_utils import MelScaleConfig, SpectrogramConfig, StftConfig


def _unfold(array, dimension, size, step):
    """NumPy equivalent of PyTorch's unfold for 2D arrays along the last dim."""
    if array.ndim == 1:
        array = array[np.newaxis, :]
    batch_size, original_length = array.shape
    num_frames = (original_length - size) // step + 1
    if num_frames <= 0:
        return np.zeros((batch_size, 0, size), dtype=array.dtype)
    output_shape = (batch_size, num_frames, size)
    output_strides = (array.strides[0], array.strides[1] * step, array.strides[1])
    return np.lib.stride_tricks.as_strided(array, shape=output_shape, strides=output_strides)


class Gemma3nAudioProcessor(NumpyAudioBackend):
    sample_rate = 16000
    force_mono = True
    max_length = 480000  # 30 seconds
    truncation = True
    pad_to_multiple_of = 128
    preemphasis_htk_flavor = True

    # n_fft = 1024 (512 frame_length → next power of 2 → 512 → ×2 fft_overdrive)
    spectrogram_config = SpectrogramConfig(
        stft_config=StftConfig(
            n_fft=1024,
            win_length=512,
            hop_length=160,
            power=1.0,
            center=False,
        ),
        mel_scale_config=MelScaleConfig(
            n_mels=128,
            f_min=125.0,
            f_max=7600.0,
            mel_scale="htk",
        ),
        mel_floor=1e-5,
        log_mode="log",
        preemphasis=0.97,
    )

    def __init__(self, per_bin_mean=None, per_bin_stddev=None, **kwargs):
        super().__init__(**kwargs)

        # Pre-compute window from stft_config
        win_length = self.spectrogram_config.stft_config.win_length
        hann_arange = np.arange(win_length, dtype=np.float32)
        self.window = (0.5 * (1 - np.cos(2 * np.pi * hann_arange / win_length))).astype(np.float32)

        n_mels = self.spectrogram_config.mel_scale_config.n_mels
        if per_bin_mean is not None:
            self.per_bin_mean = np.array(per_bin_mean).reshape(1, n_mels)
        else:
            self.per_bin_mean = None

        if per_bin_stddev is not None:
            self.per_bin_stddev = np.array(per_bin_stddev).reshape(1, n_mels)
        else:
            self.per_bin_stddev = None

    def _extract_spectrogram(self, audio, *, spectrogram_config, **kwargs):
        stft_cfg = spectrogram_config.stft_config
        preemphasis = spectrogram_config.preemphasis

        frame_size_for_unfold = stft_cfg.win_length + 1
        frames_to_process = _unfold(audio, dimension=-1, size=frame_size_for_unfold, step=stft_cfg.hop_length)

        # Preemphasis
        if preemphasis is not None and preemphasis > 0.0:
            if self.preemphasis_htk_flavor:
                first_in_frame = frames_to_process[..., :1] * (1.0 - preemphasis)
                rest_in_frame = frames_to_process[..., 1:-1] - preemphasis * frames_to_process[..., :-2]
                frames = np.concatenate([first_in_frame, rest_in_frame], axis=-1)
            else:
                frames = frames_to_process[..., 1:] - preemphasis * frames_to_process[..., :-1]
        else:
            frames = frames_to_process[..., :-1]

        frames = frames * self.window  # Broadcasting window

        stft = np.fft.rfft(frames, n=stft_cfg.n_fft, axis=-1)
        return np.abs(stft)

    def _apply_mel_scale(self, features, *, spectrogram_config, **kwargs):
        """Apply mel filterbank. Features are in (batch, time, freq) format."""
        mel_spec = np.matmul(features, self.mel_filters)
        return np.maximum(spectrogram_config.mel_floor, mel_spec)

    def _normalize_magnitude(self, features, *, spectrogram_config, **kwargs):
        """Apply log compression and per-bin normalization."""
        result = super()._normalize_magnitude(features, spectrogram_config=spectrogram_config, **kwargs)

        if self.per_bin_mean is not None:
            result = result - self.per_bin_mean
        if self.per_bin_stddev is not None:
            result = result / self.per_bin_stddev

        return result.astype(np.float32)

    def _get_features_lengths(self, audio_lengths, spectrogram_config, include_center_frame=False):
        """Frame count for unfold-based STFT (no centering)."""
        hop_length = spectrogram_config.stft_config.hop_length
        frame_size = spectrogram_config.stft_config.win_length + 1
        return (audio_lengths - frame_size) // hop_length + 1


__all__ = ["Gemma3nAudioProcessor"]
