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

    def _extract_spectrogram(self, audio, *, spectrogram_config, **kwargs):
        """Custom STFT with HTK-flavor preemphasis."""
        stft_cfg = spectrogram_config.stft_config
        preemphasis = spectrogram_config.preemphasis

        features = []
        for waveform in audio:
            if waveform.ndim == 1:
                waveform = np.expand_dims(waveform, axis=0)

            frame_size_for_unfold = stft_cfg.win_length + 1
            frames_to_process = _unfold(waveform, dimension=-1, size=frame_size_for_unfold, step=stft_cfg.hop_length)

            if preemphasis is not None and preemphasis > 0.0:
                if self.preemphasis_htk_flavor:
                    first_in_frame = frames_to_process[..., :1] * (1.0 - preemphasis)
                    rest_in_frame = (
                        frames_to_process[..., 1:-1] - preemphasis * frames_to_process[..., :-2]
                    )
                    frames = np.concatenate([first_in_frame, rest_in_frame], axis=-1)
                else:
                    frames = frames_to_process[..., 1:] - preemphasis * frames_to_process[..., :-1]
            else:
                frames = frames_to_process[..., :-1]

            frames = frames * self.window
            stft = np.fft.rfft(frames, n=stft_cfg.n_fft, axis=-1)
            magnitude_spec = np.abs(stft)
            features.append(magnitude_spec.squeeze(0))  # (frames, n_freqs)

        return features

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

    def _preprocess(self, audio, padding, max_length, truncation, pad_to_multiple_of, return_tensors,
                    spectrogram_config=None, do_extract_spectrogram=None, **kwargs):
        if max_length is None:
            max_length = self.max_length
        if truncation is None:
            truncation = self.truncation
        if pad_to_multiple_of is None:
            pad_to_multiple_of = self.pad_to_multiple_of
        if spectrogram_config is None:
            spectrogram_config = self.spectrogram_config

        # Truncate then pad to longest in batch (matching FE "longest" padding strategy)
        if truncation and max_length is not None:
            audio = [a[..., :max_length] for a in audio]

        pad_length = max(a.shape[-1] for a in audio)
        if pad_to_multiple_of is not None and (pad_length % pad_to_multiple_of != 0):
            pad_length = ((pad_length // pad_to_multiple_of) + 1) * pad_to_multiple_of
        audio = [self._pad_single(a, pad_length) for a in audio]

        # Extract spectrogram via orchestrator (_extract_spectrogram + _apply_mel_scale)
        if do_extract_spectrogram is not False and spectrogram_config is not None:
            features = self.extract_spectrogram(audio, spectrogram_config=spectrogram_config)
        else:
            features = audio

        output_key = self.model_input_names[0]
        stacked = np.stack(features, axis=0)
        return BatchFeature(data={output_key: stacked}, tensor_type=return_tensors)


__all__ = ["Gemma3nAudioProcessor"]
