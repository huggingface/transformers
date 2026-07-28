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
from dataclasses import replace

import numpy as np

from ...audio_processing_backends import NumpyAudioBackend
from ...audio_utils import MelScaleConfig, SpectrogramConfig, StftConfig


def _gemma4_frame_length_ms_to_win_length(value, config_dict):
    sr = config_dict.get("sampling_rate") or 16000
    spec = config_dict.setdefault("spectrogram_config", {})
    stft = spec.setdefault("stft_config", {})
    stft.setdefault("win_length", int(round(sr * value / 1000.0)))


def _gemma4_hop_length_ms_to_hop_length(value, config_dict):
    sr = config_dict.get("sampling_rate") or 16000
    spec = config_dict.setdefault("spectrogram_config", {})
    stft = spec.setdefault("stft_config", {})
    stft.setdefault("hop_length", int(round(sr * value / 1000.0)))


class Gemma4AudioProcessorMixin:
    """Gemma4 audio logic shared by the numpy and torch siblings: USM-style mel extractor
    (https://huggingface.co/papers/2303.01037), fully described by `spectrogram_config`."""

    sampling_rate = 16000
    force_mono = True
    padding = "longest"
    padding_value = 0.0
    max_length = 480_000
    truncation = True
    pad_to_multiple_of = 128

    # Gemma4-specific kwargs, folded into `spectrogram_config` by `_set_attributes`
    preemphasis_htk_flavor: bool = True
    fft_overdrive: bool = False
    dither: float = 0.0
    input_scale_factor: float = 1.0
    per_bin_mean = None
    per_bin_stddev = None

    legacy_field_mapping = {
        "feature_size": "spectrogram_config.mel_scale_config.n_mels",
        "frame_length_ms": _gemma4_frame_length_ms_to_win_length,
        "hop_length_ms": _gemma4_hop_length_ms_to_hop_length,
        "min_frequency": "spectrogram_config.mel_scale_config.f_min",
        "max_frequency": "spectrogram_config.mel_scale_config.f_max",
    }

    # `n_fft` is 2 ** ceil(log2(win_length)); `_maybe_rebuild_for_win_length` recomputes it
    # for non-default `win_length`/`fft_overdrive`. The base's "left"-center length formula
    # matches this framing, so no `_get_features_lengths` override is needed.
    spectrogram_config = SpectrogramConfig(
        stft_config=StftConfig(
            n_fft=512,
            win_length=320,
            hop_length=160,
            window_fn="hann_window_f32",
            power=1.0,
            center="left",
            frame_extension=1,
            fft_dtype="native",
        ),
        mel_scale_config=MelScaleConfig(
            n_mels=128,
            f_min=0.0,
            f_max=8000.0,
            mel_scale="htk",
            matmul_order="features_first",
        ),
        preemphasis=0.0,
        preemphasis_mode="htk_per_frame",
        mel_floor=0.0,  # no clamp; the log guard is pre_log_offset
        pre_log_offset=1e-3,
        log_mode="log",
        # float64 mel filters, kept float64 for the numpy matmul; torch casts to float32 at apply time
        computation_dtype="float64",
    )

    def _set_attributes(self, **kwargs):
        super()._set_attributes(**kwargs)
        updates = {}
        if not self.preemphasis_htk_flavor and self.spectrogram_config.preemphasis_mode == "htk_per_frame":
            updates["preemphasis_mode"] = "per_frame"
        if self.input_scale_factor != 1.0 and self.spectrogram_config.waveform_scale is None:
            updates["waveform_scale"] = self.input_scale_factor
        if updates:
            self.spectrogram_config = replace(self.spectrogram_config, **updates)
        self._maybe_rebuild_for_win_length()
        n_mels = self.spectrogram_config.mel_scale_config.n_mels
        if self.per_bin_mean is not None:
            self.per_bin_mean = self._as_backend_array(np.asarray(self.per_bin_mean)).reshape(1, 1, n_mels)
        if self.per_bin_stddev is not None:
            self.per_bin_stddev = self._as_backend_array(np.asarray(self.per_bin_stddev)).reshape(1, 1, n_mels)

    def _maybe_rebuild_for_win_length(self):
        stft_cfg = self.spectrogram_config.stft_config
        expected_n_fft = 2 ** math.ceil(math.log2(stft_cfg.win_length))
        if self.fft_overdrive:
            expected_n_fft *= 2
        if stft_cfg.n_fft != expected_n_fft:
            self.spectrogram_config = replace(
                self.spectrogram_config,
                stft_config=replace(stft_cfg, n_fft=expected_n_fft),
            )
            self.mel_filters = self._mel_filter_bank(self.spectrogram_config)


class Gemma4AudioProcessorNumpy(Gemma4AudioProcessorMixin, NumpyAudioBackend):
    """NumPy sibling of [`Gemma4AudioProcessor`], bit-exact with the legacy extractor."""

    def _postprocess_output(self, output, audio_ranges=None, **kwargs):
        if audio_ranges is None or "audio_features" not in output:
            return output
        features = output["audio_features"]
        # cast the float64 stats down BEFORE subtracting (legacy rounding, unlike gemma3n)
        if self.per_bin_mean is not None:
            features = features - self.per_bin_mean.astype(features.dtype)
        if self.per_bin_stddev is not None:
            features = features / self.per_bin_stddev.astype(features.dtype)
        mask = output.get("audio_features_mask")
        if mask is not None:
            features = features * mask.astype(features.dtype)[..., None]
        output["audio_features"] = features
        return output


__all__ = ["Gemma4AudioProcessorNumpy"]
