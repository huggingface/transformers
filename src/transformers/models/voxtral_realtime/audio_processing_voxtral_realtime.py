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
from ...audio_utils import MelScaleConfig, SpectrogramConfig, StftConfig, mel_filter_bank


class VoxtralRealtimeAudioProcessor(TorchAudioBackend):
    sample_rate = 16000
    force_mono = True
    spectrogram_config = SpectrogramConfig(
        stft_config=StftConfig(
            n_fft=400,
            hop_length=160,
            power=2.0,
        ),
        mel_scale_config=MelScaleConfig(
            n_mels=128,
            mel_scale="slaney",
            norm="slaney",
        ),
        log_mode="log10",
        global_log_mel_max=1.5,
    )

    def extract_spectrogram(self, audio, *, spectrogram_config=None, **kwargs):
        if spectrogram_config is None:
            spectrogram_config = self.spectrogram_config

        stft_cfg = spectrogram_config.stft_config
        global_log_mel_max = spectrogram_config.global_log_mel_max

        if isinstance(audio, list):
            waveform = torch.stack(audio)
        else:
            waveform = audio

        device = waveform.device
        window = torch.hann_window(stft_cfg.n_fft, device=device)
        stft = torch.stft(
            waveform, stft_cfg.n_fft, stft_cfg.hop_length,
            window=window, return_complex=True, center=True,
        )
        magnitudes = stft[..., :-1].abs() ** 2

        mel_filters = self.mel_filters.to(device, torch.float32)
        mel_spec = mel_filters.T @ magnitudes

        log_spec = torch.clamp(mel_spec, min=1e-10).log10()

        processed = []
        for i in range(log_spec.shape[0]):
            spec = log_spec[i]
            if global_log_mel_max is not None:
                spec_max = torch.tensor(global_log_mel_max, device=spec.device, dtype=spec.dtype)
            else:
                spec_max = spec.max()
            spec = torch.maximum(spec, spec_max - 8.0)
            spec = (spec + 4.0) / 4.0
            processed.append(spec)
        return processed

    def _mel_filter_bank(self, spectrogram_config):
        """Override to use numpy mel_filter_bank for exact match with feature extractor."""
        stft_cfg = spectrogram_config.stft_config
        mel_cfg = spectrogram_config.mel_scale_config
        mel_filters_np = mel_filter_bank(
            num_frequency_bins=1 + stft_cfg.n_fft // 2,
            num_mel_filters=mel_cfg.n_mels,
            min_frequency=mel_cfg.f_min,
            max_frequency=mel_cfg.f_max if mel_cfg.f_max is not None else self.sample_rate / 2,
            sampling_rate=self.sample_rate,
            norm=mel_cfg.norm,
            mel_scale=mel_cfg.mel_scale,
        )
        return torch.from_numpy(mel_filters_np).to(torch.float32)


__all__ = ["VoxtralRealtimeAudioProcessor"]
