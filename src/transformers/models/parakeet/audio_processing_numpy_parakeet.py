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

import numpy as np

from ...audio_processing_backends import NumpyAudioBackend
from ...audio_utils import MelScaleConfig, SpectrogramConfig, StftConfig


class ParakeetAudioProcessorNumpy(NumpyAudioBackend):
    """NumPy sibling of [`ParakeetAudioProcessor`]. Bit-exact to the torch sibling within
    the float32 noise floor (ADR 0001)."""

    sample_rate = 16000
    force_mono = True
    spectrogram_config = SpectrogramConfig(
        stft_config=StftConfig(
            n_fft=512,
            hop_length=160,
            win_length=400,
            window_fn="hann_window",
            power=2.0,
            pad_mode="constant",
            periodic=False,
        ),
        mel_scale_config=MelScaleConfig(
            n_mels=80,
            f_min=0.0,
            norm="slaney",
            mel_scale="slaney",
        ),
        preemphasis=0.97,
        preemphasis_mode="waveform",
        log_mode="log",
        mel_floor=0.0,  # base clamp is a no-op; the log guard is pre_log_offset
        pre_log_offset=2**-24,
    )

    # The base numpy backend already builds librosa's per-band float32 filters and applies
    # the mel matmul / magnitude / `log(x + pre_log_offset)` forms this model needs.

    def _normalize_magnitude(self, features, *, spectrogram_config, **kwargs):
        # Base handles the legacy `log(x + guard)` form via `pre_log_offset`;
        # transpose to (batch, frames, mels).
        features = super()._normalize_magnitude(features, spectrogram_config=spectrogram_config, **kwargs)
        return np.transpose(features, axes=(0, 2, 1))

    def _postprocess_output(self, output, audio_ranges=None, **kwargs):
        if audio_ranges is None or "audio_features" not in output:
            return output

        features = output["audio_features"]
        stft_cfg = self.spectrogram_config.stft_config
        audio_lengths = np.asarray([end - start for start, end in audio_ranges])
        features_lengths = np.floor_divide(
            audio_lengths + stft_cfg.n_fft // 2 * 2 - stft_cfg.n_fft, stft_cfg.hop_length
        )
        attention_mask = np.arange(features.shape[1])[None, :] < features_lengths[:, None]
        mask = np.expand_dims(attention_mask, axis=-1)
        # NumPy promotes float32 / int64 → float64; cast lengths to the feature dtype to keep
        # parity with torch (which preserves the floating dtype across float/int division).
        features_lengths_f = features_lengths.astype(features.dtype)
        mel_masked = features * mask
        mean = np.expand_dims(mel_masked.sum(axis=1) / np.expand_dims(features_lengths_f, axis=-1), axis=1)
        variance = ((mel_masked - mean) ** 2 * mask).sum(axis=1) / np.expand_dims(
            features_lengths_f - 1, axis=-1
        )
        std = np.expand_dims(np.sqrt(variance), axis=1)
        output["audio_features"] = (features - mean) / (std + 1e-5) * mask
        return output


__all__ = ["ParakeetAudioProcessorNumpy"]
