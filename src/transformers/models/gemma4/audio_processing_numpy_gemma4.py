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

import math
import warnings

import numpy as np

from ...audio_processing_backends import NumpyAudioBackend
from ...audio_utils import MelScaleConfig, SpectrogramConfig, StftConfig, mel_filter_bank


def _unfold(array, dimension, size, step):
    """NumPy equivalent of PyTorch's unfold along the last dim for 2-D arrays."""
    if array.ndim == 1:
        array = array[np.newaxis, :]
    batch_size, original_length = array.shape
    num_frames = (original_length - size) // step + 1
    if num_frames <= 0:
        return np.zeros((batch_size, 0, size), dtype=array.dtype)
    output_shape = (batch_size, num_frames, size)
    output_strides = (array.strides[0], array.strides[1] * step, array.strides[1])
    return np.lib.stride_tricks.as_strided(array, shape=output_shape, strides=output_strides)


def _gemma4_frame_length_ms_to_win_length(value, config_dict):
    sr = config_dict.get("sample_rate") or config_dict.get("sampling_rate") or 16000
    spec = config_dict.setdefault("spectrogram_config", {})
    stft = spec.setdefault("stft_config", {})
    stft.setdefault("win_length", int(round(sr * value / 1000.0)))


def _gemma4_hop_length_ms_to_hop_length(value, config_dict):
    sr = config_dict.get("sample_rate") or config_dict.get("sampling_rate") or 16000
    spec = config_dict.setdefault("spectrogram_config", {})
    stft = spec.setdefault("stft_config", {})
    stft.setdefault("hop_length", int(round(sr * value / 1000.0)))


class Gemma4AudioProcessorNumpy(NumpyAudioBackend):
    """NumPy sibling of [`Gemma4AudioProcessor`]. Bit-exact to the torch sibling within
    the float32 noise floor (ADR 0001) when ``dither=0`` — ``np.random.randn`` is not
    seeded for parity, so the parity fixture disables dither. See [`Gemma4AudioProcessor`]
    for the full pipeline description."""

    sample_rate = 16000
    force_mono = True
    padding = "longest"
    padding_value = 0.0
    max_length = 480_000
    truncation = True
    pad_to_multiple_of = 128

    preemphasis_htk_flavor: bool = True
    fft_overdrive: bool = False
    dither: float = 0.0
    input_scale_factor: float = 1.0

    legacy_field_mapping = {
        "feature_size": "spectrogram_config.mel_scale_config.n_mels",
        "frame_length_ms": _gemma4_frame_length_ms_to_win_length,
        "hop_length_ms": _gemma4_hop_length_ms_to_hop_length,
        "min_frequency": "spectrogram_config.mel_scale_config.f_min",
        "max_frequency": "spectrogram_config.mel_scale_config.f_max",
    }

    spectrogram_config = SpectrogramConfig(
        stft_config=StftConfig(
            n_fft=512,
            win_length=320,
            hop_length=160,
            window_fn="hann_window",
            power=1.0,
            center=False,
        ),
        mel_scale_config=MelScaleConfig(
            n_mels=128,
            f_min=0.0,
            f_max=8000.0,
            mel_scale="htk",
            matmul_order="features_first",
        ),
        preemphasis=0.0,
        mel_floor=1e-3,
        log_mode="log",
    )

    def __init__(
        self,
        preemphasis_htk_flavor: bool | None = None,
        fft_overdrive: bool | None = None,
        dither: float | None = None,
        input_scale_factor: float | None = None,
        per_bin_mean=None,
        per_bin_stddev=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if preemphasis_htk_flavor is not None:
            self.preemphasis_htk_flavor = preemphasis_htk_flavor
        if fft_overdrive is not None:
            self.fft_overdrive = fft_overdrive
        if dither is not None:
            self.dither = dither
        if input_scale_factor is not None:
            self.input_scale_factor = input_scale_factor

        self._maybe_rebuild_for_win_length()

        n_mels = self.spectrogram_config.mel_scale_config.n_mels
        self.per_bin_mean = (
            np.asarray(per_bin_mean).reshape(1, 1, n_mels) if per_bin_mean is not None else None
        )
        self.per_bin_stddev = (
            np.asarray(per_bin_stddev).reshape(1, 1, n_mels) if per_bin_stddev is not None else None
        )

        win_length = self.spectrogram_config.stft_config.win_length
        hann_arange = np.arange(win_length, dtype=np.float32)
        self.window = (0.5 * (1 - np.cos(2 * np.pi * hann_arange / win_length))).astype(np.float32)

    def _maybe_rebuild_for_win_length(self):
        from dataclasses import replace

        stft_cfg = self.spectrogram_config.stft_config
        win_length = stft_cfg.win_length
        expected_n_fft = 2 ** math.ceil(math.log2(win_length))
        if self.fft_overdrive:
            expected_n_fft *= 2
        if stft_cfg.n_fft != expected_n_fft:
            self.spectrogram_config = replace(
                self.spectrogram_config,
                stft_config=replace(stft_cfg, n_fft=expected_n_fft),
            )
            self.mel_filters = self._mel_filter_bank(self.spectrogram_config)

    def _mel_filter_bank(self, spectrogram_config):
        stft_cfg = spectrogram_config.stft_config
        mel_cfg = spectrogram_config.mel_scale_config
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return mel_filter_bank(
                num_frequency_bins=stft_cfg.n_fft // 2 + 1,
                num_mel_filters=mel_cfg.n_mels,
                min_frequency=mel_cfg.f_min,
                max_frequency=mel_cfg.f_max if mel_cfg.f_max is not None else self.sample_rate / 2,
                sampling_rate=self.sample_rate,
                norm=None,
                mel_scale=mel_cfg.mel_scale,
            )

    def _stft(self, audio, *, spectrogram_config, **kwargs):
        stft_cfg = spectrogram_config.stft_config
        win_length = stft_cfg.win_length
        hop_length = stft_cfg.hop_length
        pad_left = win_length // 2
        frame_size_for_unfold = win_length + 1

        if self.dither > 0.0:
            audio = audio + (self.dither * np.random.randn(*audio.shape)).astype(audio.dtype)

        if self.input_scale_factor != 1.0:
            audio = audio * self.input_scale_factor

        # Semicausal padding (legacy: `np.pad(waveform, ((0, 0), (pad_left, 0)), 'constant')`)
        if audio.ndim == 1:
            audio = audio[np.newaxis, :]
        audio = np.pad(audio, ((0, 0), (pad_left, 0)), mode="constant")

        frames = _unfold(audio, dimension=-1, size=frame_size_for_unfold, step=hop_length)
        frames = self._apply_htk_frame_processing(frames, spectrogram_config)

        frames = frames * self.window
        stft = np.fft.rfft(frames, n=stft_cfg.n_fft, axis=-1)
        return np.abs(stft).transpose(0, 2, 1) if stft.ndim == 3 else np.abs(stft).T

    def _apply_htk_frame_processing(self, frames, spectrogram_config):
        preemphasis = spectrogram_config.preemphasis
        if preemphasis is not None and preemphasis > 0.0:
            if self.preemphasis_htk_flavor:
                first = frames[..., :1] * (1.0 - preemphasis)
                rest = frames[..., 1:-1] - preemphasis * frames[..., :-2]
                return np.concatenate([first, rest], axis=-1)
            return frames[..., 1:] - preemphasis * frames[..., :-1]
        return frames[..., :-1]

    def _apply_mel_scale(self, features, *, spectrogram_config, **kwargs):
        mel_filters = self.mel_filters.astype(features.dtype, copy=False)
        # features shape (..., freq, num_frames). matmul_order=features_first.
        # Result (..., num_frames, n_mels).
        mel_spec = np.matmul(np.swapaxes(features, -2, -1), mel_filters)
        return mel_spec + spectrogram_config.mel_floor

    def _normalize_magnitude(self, features, *, spectrogram_config, **kwargs):
        return np.log(features).astype(np.float32)

    def _postprocess_output(self, output, audio_ranges=None, **kwargs):
        if audio_ranges is None or "audio_features" not in output:
            return output

        features = output["audio_features"]
        mask = output.get("audio_features_mask")

        if self.per_bin_mean is not None:
            features = features - self.per_bin_mean.astype(features.dtype)
        if self.per_bin_stddev is not None:
            features = features / self.per_bin_stddev.astype(features.dtype)

        if mask is not None:
            features = features * mask.astype(features.dtype)[..., None]

        output["audio_features"] = features
        return output

    def _get_features_lengths(self, audio_lengths, spectrogram_config, include_center_frame=False):
        stft_cfg = spectrogram_config.stft_config
        win_length = stft_cfg.win_length
        hop_length = stft_cfg.hop_length
        pad_left = win_length // 2
        frame_size_for_unfold = win_length + 1
        lengths = (audio_lengths + pad_left - frame_size_for_unfold) // hop_length + 1
        if isinstance(lengths, np.ndarray):
            lengths = np.maximum(0, lengths)
        else:
            lengths = max(0, int(lengths))
        return lengths


__all__ = ["Gemma4AudioProcessorNumpy"]
