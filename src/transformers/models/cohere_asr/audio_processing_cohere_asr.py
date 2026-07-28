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

import torch

from ...audio_processing_backends import TorchAudioBackend
from ...audio_utils import _create_triangular_filter_bank, hertz_to_mel, mel_to_hertz
from .audio_processing_numpy_cohere_asr import CohereAsrAudioProcessorMixin


class CohereAsrAudioProcessor(CohereAsrAudioProcessorMixin, TorchAudioBackend):
    """Torch sibling of [`CohereAsrAudioProcessorNumpy`]: energy-based long-audio chunking,
    deterministic dither, waveform preemphasis, ``log(mel @ |X|^2 + 2^-24)`` features with
    per-utterance mean/variance normalization."""

    def _standard_mel_banks(
        self,
        num_mel_filters,
        num_frequency_bins,
        min_frequency,
        max_frequency,
        sampling_rate,
        n_fft,
        mel_cfg,
        computation_dtype,
    ):
        """Torch-native build of librosa's per-band float32 rounding: float64 weights cast
        to float32, slaney norm applied *after* that cast with a second float32 rounding —
        the only order that reproduces the legacy filters bit-exactly."""
        mel_min = hertz_to_mel(min_frequency, mel_scale=mel_cfg.mel_scale)
        mel_max = hertz_to_mel(max_frequency, mel_scale=mel_cfg.mel_scale)
        mel_freqs = torch.linspace(mel_min, mel_max, num_mel_filters + 2, dtype=torch.float64)
        filter_freqs = mel_to_hertz(mel_freqs, mel_scale=mel_cfg.mel_scale)
        fft_freqs = torch.linspace(0, sampling_rate // 2, num_frequency_bins, dtype=torch.float64)
        mel_filters = _create_triangular_filter_bank(fft_freqs, filter_freqs).to(torch.float32)
        if mel_cfg.norm == "slaney":
            enorm = 2.0 / (filter_freqs[2 : num_mel_filters + 2] - filter_freqs[:num_mel_filters])
            mel_filters = (mel_filters * enorm[None, :]).to(torch.float32)
        return mel_filters

    def _apply_dither(self, audio, audio_ranges=None):
        """Deterministic per-utterance dither: each row is seeded by its valid sample count,
        so dither is invariant to batch composition (matches the legacy FE)."""
        if self.dither <= 0 or audio_ranges is None:
            return audio
        audio = audio.clone()
        generator = torch.Generator(device=audio.device)
        for i, (start, end) in enumerate(audio_ranges):
            valid_samples = min(end - start, audio.shape[1])
            if valid_samples <= 0:
                continue
            generator.manual_seed(valid_samples)
            noise = torch.randn(valid_samples, dtype=audio.dtype, device=audio.device, generator=generator)
            audio[i, :valid_samples] = audio[i, :valid_samples] + self.dither * noise
        return audio

    def _compute_magnitudes(self, stft_out, power, spectrogram_config=None):
        # legacy view_as_real + sqrt(real² + imag²) ** power pattern
        magnitudes = torch.view_as_real(stft_out)
        magnitudes = torch.sqrt(magnitudes.pow(2).sum(-1))
        if power != 1.0:
            magnitudes = magnitudes.pow(power)
        return magnitudes

    def _apply_mel_scale(self, features, *, spectrogram_config, **kwargs):
        return torch.matmul(self.mel_filters.T, features)


__all__ = ["CohereAsrAudioProcessor"]
