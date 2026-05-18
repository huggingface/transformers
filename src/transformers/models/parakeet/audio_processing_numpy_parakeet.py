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
        log_mode="log",
        mel_floor=2**-24,
    )

    def _mel_filter_bank(self, spectrogram_config):
        """Replicate librosa's per-band float32 accumulation pattern for bit-exact FE parity."""
        from ...audio_utils import hertz_to_mel, mel_to_hertz

        stft_cfg = spectrogram_config.stft_config
        mel_cfg = spectrogram_config.mel_scale_config
        n_fft = stft_cfg.n_fft
        n_mels = mel_cfg.n_mels
        f_min = mel_cfg.f_min
        f_max = mel_cfg.f_max if mel_cfg.f_max is not None else self.sample_rate / 2

        mel_min = hertz_to_mel(f_min, mel_scale=mel_cfg.mel_scale)
        mel_max = hertz_to_mel(f_max, mel_scale=mel_cfg.mel_scale)
        mel_pts = np.linspace(mel_min, mel_max, n_mels + 2)
        filter_freqs = mel_to_hertz(mel_pts.copy(), mel_scale=mel_cfg.mel_scale)
        fft_freqs = np.linspace(0, self.sample_rate / 2, 1 + n_fft // 2)

        fdiff = np.diff(filter_freqs)
        ramps = np.subtract.outer(filter_freqs, fft_freqs)

        # Accumulate into f32 per-band to match librosa's truncation pattern
        weights = np.zeros((n_mels, 1 + n_fft // 2), dtype=np.float32)
        for i in range(n_mels):
            lower = -ramps[i] / fdiff[i]
            upper = ramps[i + 2] / fdiff[i + 1]
            weights[i] = np.maximum(0, np.minimum(lower, upper))

        if mel_cfg.norm == "slaney":
            enorm = 2.0 / (filter_freqs[2 : n_mels + 2] - filter_freqs[:n_mels])
            weights *= enorm[:, np.newaxis]

        # Match torch sibling: returns transposed filters as float32 ((1 + n_fft // 2, n_mels)).
        return weights.T.astype(np.float32)

    def _compute_magnitudes(self, stft_out, power, spectrogram_config=None):
        # `np.abs(z) ** power` is numerically equivalent to the torch sibling's explicit
        # `sqrt(real² + imag²) ** power` form on `view_as_real`. Both routes compute the
        # complex magnitude and then raise to `power`; remaining drift is float32 FFT noise.
        magnitudes = np.abs(stft_out)
        if power != 1.0:
            magnitudes = magnitudes ** power
        return magnitudes

    def _needs_manual_framing(self, spectrogram_config):
        # Preemphasis is handled waveform-level in _stft; no per-frame processing needed.
        return spectrogram_config.remove_dc_offset or spectrogram_config.stft_config.left_align_fft

    def _stft(self, audio, *, spectrogram_config, audio_ranges=None, **kwargs):
        audio_lengths = (
            np.asarray([end - start for start, end in audio_ranges]) if audio_ranges is not None else None
        )

        # Waveform-level preemphasis with masking to zero out padding
        preemphasis = spectrogram_config.preemphasis
        if preemphasis is not None:
            audio = np.concatenate(
                [audio[:, :1], audio[:, 1:] - preemphasis * audio[:, :-1]], axis=1
            )
            if audio_lengths is not None:
                timemask = np.expand_dims(np.arange(audio.shape[-1]), axis=0) < np.expand_dims(audio_lengths, axis=1)
                audio = np.where(timemask, audio, 0.0)

        return super()._stft(audio, spectrogram_config=spectrogram_config, **kwargs)

    def _apply_mel_scale(self, features, *, spectrogram_config, **kwargs):
        return np.matmul(self.mel_filters.T, features)

    def _normalize_magnitude(self, features, *, spectrogram_config, **kwargs):
        # Per-utterance mean/var normalization is applied later in `_postprocess_output` —
        # this hook is pointwise per ADR 0005. Match the legacy FE's `log(x + guard)` form
        # rather than `log(clamp(x, guard))`, then transpose to (batch, frames, mels).
        features = np.log(features + spectrogram_config.mel_floor)
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
