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

import math

import numpy as np

from ...audio_processing_backends import NumpyAudioBackend
from ...audio_utils import MelScaleConfig, SpectrogramConfig, StftConfig
from ...feature_extraction_utils import BatchFeature


def _create_fb_matrix(n_freqs, f_min, f_max, n_mels, sample_rate, fft_length, norm=None):
    """HTK-style mel filterbank matrix matching Gemma3n FE implementation."""
    all_freqs = np.arange(n_freqs, dtype=np.float32) * (sample_rate / fft_length)
    m_min = 2595.0 * math.log10(1.0 + (f_min / 700.0))
    m_max = 2595.0 * math.log10(1.0 + (f_max / 700.0))
    m_pts = np.linspace(m_min, m_max, n_mels + 2)
    f_pts = 700.0 * (10 ** (m_pts / 2595.0) - 1.0)
    f_diff = f_pts[1:] - f_pts[:-1]
    slopes = np.expand_dims(f_pts, 0) - np.expand_dims(all_freqs, 1)
    zero = np.zeros(1, dtype=np.float32)
    down_slopes = (-1.0 * slopes[:, :-2]) / f_diff[:-1]
    up_slopes = slopes[:, 2:] / f_diff[1:]
    fb = np.maximum(zero, np.minimum(down_slopes, up_slopes))
    if norm == "slaney":
        enorm = 2.0 / (f_pts[2 : n_mels + 2] - f_pts[:n_mels])
        fb *= np.expand_dims(enorm, 0)
    return fb


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

    def _mel_filter_bank(self, spectrogram_config):
        """Custom HTK-style mel filterbank matching the original Gemma3n FE."""
        sc = spectrogram_config
        msc = sc.mel_scale_config
        return _create_fb_matrix(
            n_freqs=sc.stft_config.n_fft // 2 + 1,
            f_min=msc.f_min,
            f_max=msc.f_max,
            n_mels=msc.n_mels,
            sample_rate=self.sample_rate,
            fft_length=sc.stft_config.n_fft,
        )

    def extract_spectrogram(self, audio, *, spectrogram_config=None, **kwargs):
        if spectrogram_config is None:
            spectrogram_config = self.spectrogram_config

        # Process all waveforms at once (bypass base class per-element iteration)
        if not isinstance(audio, list):
            audio = [audio]

        features = self._extract_spectrogram(audio, spectrogram_config=spectrogram_config, **kwargs)
        features = self._apply_mel_scale(features, spectrogram_config=spectrogram_config, **kwargs)
        # Skip _normalize_magnitude: _apply_mel_scale already applies log + per-bin normalization
        return features

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
        magnitude_spec = np.abs(stft)

    def _apply_mel_scale(self, features, *, spectrogram_config, **kwargs):
        """Apply mel filterbank, log compression, and per-bin normalization."""
        result = []
        for mag_spec in features:
            mel_spec = np.matmul(mag_spec, self.mel_filters)
            log_mel_spec = np.log(np.maximum(mel_spec, spectrogram_config.mel_floor))

            if self.per_bin_mean is not None:
                log_mel_spec = log_mel_spec - self.per_bin_mean
            if self.per_bin_stddev is not None:
                log_mel_spec = log_mel_spec / self.per_bin_stddev

            result.append(log_mel_spec.astype(np.float32))
        return result


__all__ = ["Gemma3nAudioProcessor"]
