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

import torch

from ...audio_processing_backends import TorchAudioBackend
from ...audio_utils import MelScaleConfig, SpectrogramConfig, StftConfig
from ...processing_utils import AudioKwargs


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
        Value the waveform is clipped to before dynamic range compression.
    max_length_s (`int`, *optional*, defaults to 10):
        Maximum waveform length in seconds, used to derive `num_max_samples`.
    """

    magnitude_floor: float
    do_normalize: bool
    normalize_min: float
    normalize_max: float
    compression_factor: float
    compression_clip_val: float
    max_length_s: int


class UnivNetAudioProcessorMixin:
    sampling_rate = 24000
    force_mono = True
    # The legacy FE saved `feature_size=1` (a raw-audio default) and kept the real mel count in
    # `num_mel_bins`, so the base mapping of `feature_size` must not apply here.
    # The legacy FE's `mel_floor` is UnivNet's *magnitude* floor (added inside the magnitude
    # before the sqrt), not the pre-`log()` clamp the base mapping assumes. Route it accordingly;
    # `spectrogram_config.mel_floor` is a separate value the port introduced.
    legacy_field_mapping = {"feature_size": None, "mel_floor": "magnitude_floor"}

    spectrogram_config = SpectrogramConfig(
        stft_config=StftConfig(
            n_fft=1024,
            hop_length=256,
            center=False,
            window_fn="hann",
            periodic=True,
            power=1.0,
        ),
        mel_scale_config=MelScaleConfig(
            n_mels=100,
            f_min=0.0,
            f_max=12000.0,
            mel_scale="slaney",
            norm="slaney",
        ),
        log_mode="log",
        # the clamp applied before `log()`; distinct from `magnitude_floor` above
        mel_floor=1e-5,
        computation_dtype="float64",
        transpose_features=True,  # UnivNet consumes (frames, n_mels)
    )

    magnitude_floor = 1e-9
    do_normalize = False
    normalize_min = -11.512925148010254
    normalize_max = 2.3143386840820312
    compression_factor = 1.0
    compression_clip_val = 1e-5
    max_length_s = 10
    valid_kwargs = UnivNetAudioProcessorKwargs

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.num_max_samples = self.max_length_s * self.sampling_rate

    def _get_mask_width(self, padded_length, spectrogram_config) -> int:
        return int(padded_length // spectrogram_config.stft_config.hop_length)

    def _get_valid_feature_lengths(self, audio_lengths, spectrogram_config):
        return audio_lengths // spectrogram_config.stft_config.hop_length

    def _stft(self, audio, *, spectrogram_config, **kwargs):
        # UnivNet reflect-pads by (n_fft - hop_length) / 2 instead of centring the frames
        stft_cfg = spectrogram_config.stft_config
        pad_amount = int((stft_cfg.n_fft - stft_cfg.hop_length) / 2)
        audio = self._reflect_pad(audio, pad_amount)
        return super()._stft(audio, spectrogram_config=spectrogram_config, **kwargs)

    def _normalize_magnitude(self, features, *, spectrogram_config, **kwargs):
        features = super()._normalize_magnitude(features, spectrogram_config=spectrogram_config, **kwargs)
        if self.do_normalize:
            features = 2 * ((features - self.normalize_min) / (self.normalize_max - self.normalize_min)) - 1
        return features


class UnivNetAudioProcessor(UnivNetAudioProcessorMixin, TorchAudioBackend):
    def _reflect_pad(self, audio, pad_amount):
        # torch reflect-pads the last axis only from a batched input, unlike `np.pad`
        if audio.ndim == 1:
            return torch.nn.functional.pad(audio[None], (pad_amount, pad_amount), mode="reflect")[0]
        return torch.nn.functional.pad(audio, (pad_amount, pad_amount), mode="reflect")

    def _compute_magnitudes(self, stft_out, power, spectrogram_config=None):
        # round-trip through complex64/float32 like the legacy FE, so the float64 magnitudes
        # match bit-exactly (the numpy sibling stays in float64 throughout)
        stft_out = stft_out.to(torch.complex64)
        presqrt = stft_out.real**2 + stft_out.imag**2 + self.magnitude_floor
        return presqrt.double().sqrt().float().double()

    def _apply_mel_scale(self, features, *, spectrogram_config, **kwargs):
        # No mel-scale clamp, as in the numpy sibling. Match the filters to the feature dtype:
        # unlike numpy, `torch.matmul` refuses mixed dtypes rather than promoting.
        mel_filters = self.mel_filters.to(device=features.device, dtype=features.dtype)
        return torch.matmul(mel_filters.T, features)


__all__ = ["UnivNetAudioProcessor"]
