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
from ...audio_utils import _create_triangular_filter_bank, hertz_to_mel, mel_to_hertz
from .audio_processing_numpy_parakeet import ParakeetAudioProcessorNumpy


class ParakeetAudioProcessor(TorchAudioBackend):
    sample_rate = 16000
    force_mono = True

    spectrogram_config = ParakeetAudioProcessorNumpy.spectrogram_config

    def _standard_mel_banks(self, num_mel_filters, num_frequency_bins, min_frequency,
                            max_frequency, sampling_rate, n_fft, mel_cfg, computation_dtype):
        """Torch-native build of librosa's per-band float32 rounding pattern.

        The legacy FE's filters are librosa's: triangular weights computed in float64,
        cast to float32, then the slaney area-norm applied *after* that cast with a
        second float32 rounding. The base torch leaf matches torchaudio instead
        (float32-native ops), and a float64 build with the norm applied before the
        final cast differs in the last ulp — only librosa's exact rounding order
        reproduces the legacy filters bit-exactly. Torch ops only (float64 linspace /
        exp match numpy's bitwise here); no numpy construction.
        """
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

    def _compute_magnitudes(self, stft_out, power, spectrogram_config=None):
        import torch

        magnitudes = torch.view_as_real(stft_out)
        magnitudes = torch.sqrt(magnitudes.pow(2).sum(-1))
        if power != 1.0:
            magnitudes = magnitudes.pow(power)
        return magnitudes

    def _apply_mel_scale(self, features, *, spectrogram_config, **kwargs):
        import torch

        return torch.matmul(self.mel_filters.T, features)

    def _normalize_magnitude(self, features, *, spectrogram_config, **kwargs):
        # Base handles the legacy `log(x + guard)` form via `pre_log_offset`;
        # transpose to (batch, frames, mels).
        features = super()._normalize_magnitude(features, spectrogram_config=spectrogram_config, **kwargs)
        return features.permute(0, 2, 1)

    def _postprocess_output(self, output, audio_ranges=None, **kwargs):
        import torch

        if audio_ranges is None or "audio_features" not in output:
            return output

        features = output["audio_features"]
        stft_cfg = self.spectrogram_config.stft_config
        audio_lengths = torch.tensor([end - start for start, end in audio_ranges])
        features_lengths = torch.floor_divide(
            audio_lengths + stft_cfg.n_fft // 2 * 2 - stft_cfg.n_fft, stft_cfg.hop_length
        )
        attention_mask = torch.arange(features.shape[1])[None, :] < features_lengths[:, None]
        mask = attention_mask.unsqueeze(-1)
        mel_masked = features * mask
        mean = (mel_masked.sum(dim=1) / features_lengths.unsqueeze(-1)).unsqueeze(1)
        variance = ((mel_masked - mean) ** 2 * mask).sum(dim=1) / (features_lengths - 1).unsqueeze(-1)
        std = torch.sqrt(variance).unsqueeze(1)
        output["audio_features"] = (features - mean) / (std + 1e-5) * mask
        return output


__all__ = ["ParakeetAudioProcessor"]
