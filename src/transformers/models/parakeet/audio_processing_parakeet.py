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

import librosa
import torch

from ...audio_processing_backends import TorchAudioBackend
from ...audio_utils import MelScaleConfig, SpectrogramConfig, StftConfig


class ParakeetAudioProcessor(TorchAudioBackend):
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
        ),
        preemphasis=0.97,
        log_mode="log",
        mel_floor=2**-24,
    )

    def _mel_filter_bank(self, spectrogram_config):
        """Use librosa mel filters for exact numerical match with the feature extractor."""
        stft_cfg = spectrogram_config.stft_config
        mel_cfg = spectrogram_config.mel_scale_config
        mel_filters = librosa.filters.mel(
            sr=self.sample_rate,
            n_fft=stft_cfg.n_fft,
            n_mels=mel_cfg.n_mels,
            fmin=mel_cfg.f_min,
            fmax=mel_cfg.f_max if mel_cfg.f_max is not None else self.sample_rate / 2,
            norm=mel_cfg.norm,
        )
        # librosa returns (n_mels, freq); transpose to (freq, n_mels) for base class convention
        return torch.from_numpy(mel_filters.T).to(torch.float32)        
    
    def _pre_stft(self, audio, *, spectrogram_config, **kwargs):
        preemphasis = spectrogram_config.preemphasis
        if preemphasis is not None:
            timemask = torch.arange(audio.shape[-1], device=audio.device).unsqueeze(0) < self._audio_lengths.unsqueeze(1)
            audio = torch.cat(
                [audio[:, :1], audio[:, 1:] - preemphasis * audio[:, :-1]], dim=1
            )
            audio = audio.masked_fill(~timemask, 0.0)
        return audio

    def _extract_spectrogram(self, audio, *, spectrogram_config, **kwargs):
        # Detect audio lengths from zero-padded waveform for preemphasis masking and normalization
        if audio.ndim == 2:
            indices = torch.arange(audio.shape[-1], device=audio.device).expand_as(audio)
            self._audio_lengths = indices.masked_fill(audio == 0, -1).max(dim=-1).values + 1

        audio = self._pre_stft(audio, spectrogram_config=spectrogram_config, **kwargs)

        # Compute STFT matching the FE's magnitude computation for exact numerical match
        stft_cfg = spectrogram_config.stft_config
        window = torch.hann_window(stft_cfg.win_length, periodic=stft_cfg.periodic, device=audio.device)
        stft = torch.stft(
            audio,
            stft_cfg.n_fft,
            hop_length=stft_cfg.hop_length,
            win_length=stft_cfg.win_length,
            window=window,
            return_complex=True,
            pad_mode=stft_cfg.pad_mode,
        )
        magnitudes = torch.view_as_real(stft)
        magnitudes = torch.sqrt(magnitudes.pow(2).sum(-1))
        magnitudes = magnitudes.pow(2)
        return magnitudes

    def _apply_mel_scale(self, features, *, spectrogram_config, **kwargs):
        return torch.matmul(self.mel_filters.T, features)

    def _normalize_magnitude(self, features, *, spectrogram_config, **kwargs):
        # Match FE: log(mel_spec + guard_value) instead of log(clamp(mel_spec, guard_value))
        features = torch.log(features + spectrogram_config.mel_floor)

        # (batch, mels, frames) -> (batch, frames, mels)
        features = features.permute(0, 2, 1)

        # Per-utterance normalization
        stft_cfg = spectrogram_config.stft_config
        audio_lengths = self._audio_lengths
        features_lengths = torch.floor_divide(
            audio_lengths + stft_cfg.n_fft // 2 * 2 - stft_cfg.n_fft, stft_cfg.hop_length
        )
        attention_mask = torch.arange(features.shape[1])[None, :] < features_lengths[:, None]
        mask = attention_mask.unsqueeze(-1)
        mel_masked = features * mask
        mean = mel_masked.sum(dim=1) / features_lengths.unsqueeze(-1)
        mean = mean.unsqueeze(1)
        variance = ((mel_masked - mean) ** 2 * mask).sum(dim=1) / (features_lengths - 1).unsqueeze(-1)
        std = torch.sqrt(variance).unsqueeze(1)
        features = (features - mean) / (std + 1e-5)
        features *= mask

        return features


__all__ = ["ParakeetAudioProcessor"]
