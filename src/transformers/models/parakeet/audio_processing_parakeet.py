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
            mel_scale="slaney",
        ),
        preemphasis=0.97,
        log_mode="log",
        mel_floor=2**-24,
    )

    def _mel_filter_bank(self, spectrogram_config):
        """Compute mel filters via numpy for exact numerical match with the feature extractor.

        The FE uses librosa which accumulates into a float32 array per-band.
        Replicating that truncation pattern is needed for bit-exact results.
        """
        import numpy as np
        import torch

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

        return torch.from_numpy(weights.T).to(torch.float32)

    def _compute_magnitudes(self, stft_out, power):
        import torch

        magnitudes = torch.view_as_real(stft_out)
        magnitudes = torch.sqrt(magnitudes.pow(2).sum(-1))
        if power != 1.0:
            magnitudes = magnitudes.pow(power)
        return magnitudes

    def _needs_manual_framing(self, spectrogram_config):
        # Preemphasis is handled waveform-level in _pre_stft; no per-frame processing needed.
        return spectrogram_config.remove_dc_offset or spectrogram_config.stft_config.left_align_fft

    def _pre_stft(self, audio, *, spectrogram_config, **kwargs):
        import torch

        if not isinstance(self._audio_lengths, torch.Tensor):
            self._audio_lengths = torch.tensor(self._audio_lengths, device=audio.device)

        preemphasis = spectrogram_config.preemphasis
        if preemphasis is not None:
            audio = torch.cat(
                [audio[:, :1], audio[:, 1:] - preemphasis * audio[:, :-1]], dim=1
            )
            timemask = torch.arange(audio.shape[-1], device=audio.device).unsqueeze(0) < self._audio_lengths.unsqueeze(1)
            audio = audio.masked_fill(~timemask, 0.0)
        return audio

    def _apply_mel_scale(self, features, *, spectrogram_config, **kwargs):
        import torch

        return torch.matmul(self.mel_filters.T, features)

    def _normalize_magnitude(self, features, *, spectrogram_config, **kwargs):
        import torch

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
