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
from collections.abc import Sequence

import numpy as np

from ...audio_processing_backends import NumpyAudioBackend
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
    frame_length = 512  # 32ms at 16kHz
    hop_length = 160  # 10ms at 16kHz
    n_mels = 128
    min_frequency = 125.0
    max_frequency = 7600.0
    preemphasis_coeff = 0.97
    preemphasis_htk_flavor = True
    fft_overdrive = True
    mel_floor = 1e-5
    max_length = 480000  # 30 seconds
    truncation = True
    pad_to_multiple_of = 128

    def __init__(self, per_bin_mean=None, per_bin_stddev=None, **kwargs):
        super().__init__(**kwargs)

        fft_length = 2 ** math.ceil(math.log2(self.frame_length))
        if self.fft_overdrive:
            fft_length *= 2
        self.fft_length = fft_length

        hann_arange = np.arange(self.frame_length, dtype=np.float32)
        self.window = (0.5 * (1 - np.cos(2 * np.pi * hann_arange / self.frame_length))).astype(np.float32)

        self.mel_filters = _create_fb_matrix(
            n_freqs=self.fft_length // 2 + 1,
            f_min=self.min_frequency,
            f_max=self.max_frequency,
            n_mels=self.n_mels,
            sample_rate=self.sample_rate,
            fft_length=self.fft_length,
            norm=None,
        )

        if per_bin_mean is not None:
            self.per_bin_mean = np.array(per_bin_mean).reshape(1, 1, self.n_mels)
        else:
            self.per_bin_mean = None

        if per_bin_stddev is not None:
            self.per_bin_stddev = np.array(per_bin_stddev).reshape(1, 1, self.n_mels)
        else:
            self.per_bin_stddev = None

    def _extract_spectrogram(self, waveform):
        if waveform.ndim == 1:
            waveform = np.expand_dims(waveform, axis=0)

        frame_size_for_unfold = self.frame_length + 1
        frames_to_process = _unfold(waveform, dimension=-1, size=frame_size_for_unfold, step=self.hop_length)

        if self.preemphasis_coeff > 0.0:
            if self.preemphasis_htk_flavor:
                first_in_frame = frames_to_process[..., :1] * (1.0 - self.preemphasis_coeff)
                rest_in_frame = (
                    frames_to_process[..., 1:-1] - self.preemphasis_coeff * frames_to_process[..., :-2]
                )
                frames = np.concatenate([first_in_frame, rest_in_frame], axis=-1)
            else:
                frames = frames_to_process[..., 1:] - self.preemphasis_coeff * frames_to_process[..., :-1]
        else:
            frames = frames_to_process[..., :-1]

        frames = frames * self.window
        stft = np.fft.rfft(frames, n=self.fft_length, axis=-1)
        magnitude_spec = np.abs(stft)

        mel_spec = np.matmul(magnitude_spec, self.mel_filters)
        log_mel_spec = np.log(np.maximum(mel_spec, self.mel_floor))

        if self.per_bin_mean is not None:
            log_mel_spec = log_mel_spec - self.per_bin_mean
        if self.per_bin_stddev is not None:
            log_mel_spec = log_mel_spec / self.per_bin_stddev

        return log_mel_spec.squeeze(0)  # (frames, n_mels)

    def extract_spectrogram(self, audio, *, spectrogram_config):
        return [self._extract_spectrogram(waveform) for waveform in audio]

    def _preprocess(self, audio, padding, max_length, truncation, pad_to_multiple_of, return_tensors, **kwargs):
        # Use class defaults for max_length, truncation, pad_to_multiple_of if not overridden
        if max_length is None:
            max_length = self.max_length
        if truncation is None:
            truncation = self.truncation
        if pad_to_multiple_of is None:
            pad_to_multiple_of = self.pad_to_multiple_of

        # Truncate first (separate from padding, matching FE behavior)
        if truncation and max_length is not None:
            audio = [a[..., :max_length] for a in audio]

        # Pad to longest in batch (matching FE "longest" padding strategy)
        pad_length = max(a.shape[-1] for a in audio)
        if pad_to_multiple_of is not None and (pad_length % pad_to_multiple_of != 0):
            pad_length = ((pad_length // pad_to_multiple_of) + 1) * pad_to_multiple_of
        audio = [self.pad(a, pad_length) for a in audio]

        # Extract spectrogram
        features = self.extract_spectrogram(audio, spectrogram_config=None)

        # Cast to float32 to match FE output
        features = [f.astype(np.float32) for f in features]

        # Stack and return
        output_key = self.model_input_names[0]
        stacked = np.stack(features, axis=0)
        return BatchFeature(data={output_key: stacked}, tensor_type=return_tensors)


__all__ = ["Gemma3nAudioProcessor"]
