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

from ...audio_processing_backends import TorchAudioBackend
from .audio_processing_numpy_parakeet import ParakeetAudioProcessorNumpy


class ParakeetAudioProcessor(TorchAudioBackend):
    sample_rate = 16000
    force_mono = True
    # Single source of truth for the config lives on the numpy sibling (importable without torch).
    spectrogram_config = ParakeetAudioProcessorNumpy.spectrogram_config

    def _mel_filter_bank(self, spectrogram_config):
        """Build the filters with librosa's per-band float32 pattern for bit-exact FE parity.

        `audio_utils.mel_filter_bank` with a float32 dtype replicates librosa's truncation;
        the base torch implementation instead matches torchaudio's float32 ops.
        """
        import numpy as np
        import torch

        from ...audio_utils import mel_filter_bank

        stft_cfg = spectrogram_config.stft_config
        mel_cfg = spectrogram_config.mel_scale_config
        filters = mel_filter_bank(
            num_frequency_bins=1 + stft_cfg.n_fft // 2,
            num_mel_filters=mel_cfg.n_mels,
            min_frequency=mel_cfg.f_min,
            max_frequency=mel_cfg.f_max if mel_cfg.f_max is not None else self.sample_rate / 2,
            sampling_rate=self.sample_rate,
            norm=mel_cfg.norm,
            mel_scale=mel_cfg.mel_scale,
            dtype=np.float32,
        )
        return torch.from_numpy(filters)

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
