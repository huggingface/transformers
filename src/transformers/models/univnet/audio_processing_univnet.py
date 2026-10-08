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

from typing import Annotated

from ...audio_processing_utils import BaseAudioProcessor
from ...processing_utils import AudioKwargs
from ...utils.type_validators import strictly_positive


class UnivNetAudioProcessorKwargs(AudioKwargs, total=False):
    r"""
    magnitude_floor (`float`, *optional*, defaults to 1e-09):
        Floor added to the squared STFT magnitude before the square root. This is UnivNet's
        `mel_floor` from the legacy feature extractor. It is a separate quantity from
        `spectrogram_config.mel_floor`, which clamps the features before `log()`.
    do_normalize (`bool`, *optional*, defaults to `False`):
        Whether to rescale the features to `[-1, 1]` using `normalize_min` and `normalize_max`.
    normalize_min (`float`, *optional*, defaults to -11.512925148010254):
        Lower bound of the range mapped to `-1` when `do_normalize` is `True`.
    normalize_max (`float`, *optional*, defaults to 2.3143386840820312):
        Upper bound of the range mapped to `1` when `do_normalize` is `True`.
    compression_factor (`float`, *optional*, defaults to 1.0):
        Multiplicative factor for dynamic range compression during spectral normalization.
    compression_clip_val (`float`, *optional*, defaults to 1e-05):
        Minimum mel magnitude before dynamic range compression.
    max_length_s (`int`, *optional*, defaults to 10):
        Default maximum waveform length in seconds when `max_length` is omitted.
    """

    magnitude_floor: float
    do_normalize: bool | None
    normalize_min: float
    normalize_max: float
    compression_factor: Annotated[float, strictly_positive]
    compression_clip_val: Annotated[float, strictly_positive]
    max_length_s: Annotated[int, strictly_positive]


class UnivNetAudioProcessor(BaseAudioProcessor):
    sampling_rate = 24000
    # The legacy FE saved `feature_size=1` (a raw-audio default) and kept the real mel count in
    # `num_mel_bins`, so the base mapping of `feature_size` must not apply here.
    # The legacy FE's `mel_floor` is UnivNet's *magnitude* floor (added inside the magnitude
    # before the sqrt), not the pre-`log()` clamp the base mapping assumes. Route it accordingly;
    # `spectrogram_config.mel_floor` is a separate value the port introduced.
    legacy_field_mapping = {
        "feature_size": None,
        "mel_floor": "magnitude_floor",
        # Belong to legacy noise-generation/pad-end helpers, not this feature workflow.
        "model_in_channels": None,
        "pad_end_length": None,
    }

    spectrogram_config = {
        "stft_config": {
            "n_fft": 1024,
            "hop_length": 256,
            "center": False,
            "window_fn": "hann",
            "periodic": True,
            "power": 1.0,
            # the legacy numpy `spectrogram()` wrote its FFT into a complex64 buffer
            "fft_dtype": "complex64",
        },
        "mel_scale_config": {
            "n_mels": 100,
            "f_min": 0.0,
            "f_max": 12000.0,
            "mel_scale": "slaney",
            "norm": "slaney",
            # the legacy extractor built its bank with `mel_filter_bank` (librosa's rounding, linspace bins)
            "bank_rounding": "librosa",
            "frequency_bin_mode": "linspace",
        },
        "log_mode": "log",
        # no clamp at the mel projection: the floor lives inside the magnitude (`magnitude_floor`) and the
        # clamp before `log()` is `compression_clip_val`
        "mel_floor": 0.0,
        "computation_dtype": "float64",
        "transpose_features": True,  # UnivNet consumes (frames, n_mels)
    }

    magnitude_floor = 1e-9
    do_normalize = False
    normalize_min = -11.512925148010254
    normalize_max = 2.3143386840820312
    compression_factor = 1.0
    compression_clip_val = 1e-5
    max_length_s = 10
    truncation = True
    valid_kwargs = UnivNetAudioProcessorKwargs

    def _preprocess(self, audio, *, max_length, max_length_s, **kwargs):
        return super()._preprocess(
            audio,
            max_length=max_length if max_length is not None else int(max_length_s * self.sampling_rate),
            **kwargs,
        )

    def _log_compress(self, features, *, spectrogram_config, compression_factor, compression_clip_val, **kwargs):
        features = self.ops.clamp_min(features, compression_clip_val) * compression_factor
        return super()._log_compress(features, spectrogram_config=spectrogram_config, **kwargs)

    def _padded_frame_count(self, padded_length, spectrogram_config, **kwargs) -> int:
        # UnivNet's frame count is exactly `samples // hop_length` — not the base's window
        # geometry. Deleting this override changes the mask width.
        return int(padded_length // spectrogram_config.stft_config.hop_length)

    def _valid_frame_counts(self, audio_lengths, spectrogram_config):
        # Same geometry as `_padded_frame_count`, per utterance.
        return audio_lengths // spectrogram_config.stft_config.hop_length

    def _waveform_to_spectrum(self, audio, *, spectrogram_config, **kwargs):
        # UnivNet reflect-pads by (n_fft - hop_length) / 2 instead of centring the frames
        stft_cfg = spectrogram_config.stft_config
        pad_amount = int((stft_cfg.n_fft - stft_cfg.hop_length) / 2)
        audio = self.ops.pad(audio, pad_amount, pad_amount, mode="reflect")
        return super()._waveform_to_spectrum(audio, spectrogram_config=spectrogram_config, **kwargs)

    def _spectrum_magnitude(self, stft_out, spectrogram_config, *, magnitude_floor, **kwargs):
        # The legacy extractor squares, sums and adds its floor in float32 on the complex64 buffer
        # (`fft_dtype`); `computation_dtype="float64"` is for the mel stage, so this override
        # deliberately does not widen. The float32 casts make that explicit for a backend whose
        # spectrum is not complex64.
        real = self.ops.astype(self.ops.real(stft_out), "float32")
        imag = self.ops.astype(self.ops.imag(stft_out), "float32")
        floored_power = real * real + imag * imag + magnitude_floor
        # the root is taken in float64 (`.double().sqrt()`) and rounded through float32 (`.float()`)
        magnitudes_f64 = self.ops.sqrt(self.ops.astype(floored_power, "float64"))
        magnitudes_f32 = self.ops.astype(magnitudes_f64, "float32")
        # float64 out: the legacy mel GEMM promoted to its float64 filters, and the base
        # `_project_to_mel` casts the bank to the features' dtype on every backend
        return self.ops.astype(magnitudes_f32, "float64")

    def _shape_log_features(
        self, features, spectrogram_config, *, do_normalize, normalize_min, normalize_max, **kwargs
    ):
        features = super()._shape_log_features(features, spectrogram_config)
        if do_normalize:
            features = 2.0 * ((features - normalize_min) / (normalize_max - normalize_min)) - 1.0
        return features


__all__ = ["UnivNetAudioProcessor"]
