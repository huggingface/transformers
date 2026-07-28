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

from dataclasses import replace

import numpy as np

from ...audio_processing_backends import NumpyAudioBackend
from ...audio_utils import MelScaleConfig, SpectrogramConfig, StftConfig


class Gemma3nAudioProcessorMixin:
    """Gemma3n audio logic shared by the numpy and torch siblings; the USM-style pipeline
    is fully described by `spectrogram_config`."""

    sampling_rate = 16000
    force_mono = True
    max_length = 480000  # 30 seconds
    truncation = True
    pad_to_multiple_of = 128
    # Gemma3n-specific kwargs, folded into config/arrays by `_set_attributes`
    preemphasis_htk_flavor = True
    per_bin_mean = None
    per_bin_stddev = None

    spectrogram_config = SpectrogramConfig(
        stft_config=StftConfig(
            n_fft=1024,
            win_length=512,
            hop_length=160,
            power=1.0,
            center=False,
            window_fn="hann_window_f32",
            frame_extension=1,
            fft_dtype="float64",
        ),
        mel_scale_config=MelScaleConfig(
            n_mels=128,
            f_min=125.0,
            f_max=7600.0,
            mel_scale="htk",
            matmul_order="features_first",
        ),
        mel_floor=1e-5,
        log_mode="log",
        preemphasis=0.97,
        preemphasis_mode="htk_per_frame",
        computation_dtype="float64",
    )

    def _set_attributes(self, **kwargs):
        super()._set_attributes(**kwargs)
        if not self.preemphasis_htk_flavor and self.spectrogram_config.preemphasis_mode == "htk_per_frame":
            self.spectrogram_config = replace(self.spectrogram_config, preemphasis_mode="per_frame")
        n_mels = self.spectrogram_config.mel_scale_config.n_mels
        if self.per_bin_mean is not None:
            self.per_bin_mean = self._as_backend_array(np.asarray(self.per_bin_mean)).reshape(1, n_mels)
        if self.per_bin_stddev is not None:
            self.per_bin_stddev = self._as_backend_array(np.asarray(self.per_bin_stddev)).reshape(1, n_mels)

    def _get_features_lengths(self, audio_lengths, spectrogram_config, include_center_frame=False):
        """Extended-frame count for the mask width; per-utterance validity is
        ``ceil(L / hop)`` (legacy strided-sample-mask semantics)."""
        stft_cfg = spectrogram_config.stft_config
        if include_center_frame:
            frame_size = stft_cfg.win_length + 1
            return (audio_lengths - frame_size) // stft_cfg.hop_length + 1
        return (audio_lengths + stft_cfg.hop_length - 1) // stft_cfg.hop_length


class Gemma3nAudioProcessorNumpy(Gemma3nAudioProcessorMixin, NumpyAudioBackend):
    """NumPy sibling of [`Gemma3nAudioProcessor`], bit-exact with the legacy
    `Gemma3nAudioFeatureExtractor`."""

    def _normalize_magnitude(self, features, *, spectrogram_config, **kwargs):
        result = super()._normalize_magnitude(features, spectrogram_config=spectrogram_config, **kwargs)
        # float64 stats promote the result before the final float32 cast (legacy rounding)
        if self.per_bin_mean is not None:
            result = result - self.per_bin_mean
        if self.per_bin_stddev is not None:
            result = result / self.per_bin_stddev
        return result.astype(np.float32)


__all__ = ["Gemma3nAudioProcessorNumpy"]
