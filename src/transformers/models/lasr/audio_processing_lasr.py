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

import numpy as np

from ...audio_processing_backends import TorchAudioBackend
from ...audio_utils import MelScaleConfig, SpectrogramConfig, StftConfig, hertz_to_mel


def _linear_to_mel_weight_matrix(num_mel_bins, num_spectrogram_bins, sample_rate, lower_edge_hertz, upper_edge_hertz):
    """Kaldi-style mel weight matrix matching the LASR FE implementation."""
    internal_dtype = np.float64
    bands_to_zero = 1
    nyquist_hertz = sample_rate / 2.0
    linear_frequencies = np.linspace(0.0, nyquist_hertz, num_spectrogram_bins, dtype=internal_dtype)[bands_to_zero:]
    spectrogram_bins_mel = hertz_to_mel(linear_frequencies, mel_scale="kaldi")[:, np.newaxis]

    edges = np.linspace(
        hertz_to_mel(lower_edge_hertz, mel_scale="kaldi"),
        hertz_to_mel(upper_edge_hertz, mel_scale="kaldi"),
        num_mel_bins + 2,
        dtype=internal_dtype,
    )
    lower_edge_mel = edges[:-2][np.newaxis, :]
    center_mel = edges[1:-1][np.newaxis, :]
    upper_edge_mel = edges[2:][np.newaxis, :]

    lower_slopes = (spectrogram_bins_mel - lower_edge_mel) / (center_mel - lower_edge_mel)
    upper_slopes = (upper_edge_mel - spectrogram_bins_mel) / (upper_edge_mel - center_mel)
    mel_weights = np.maximum(0.0, np.minimum(lower_slopes, upper_slopes))
    return np.pad(mel_weights, [[bands_to_zero, 0], [0, 0]]).astype(np.float64)


class LasrAudioProcessor(TorchAudioBackend):
    sample_rate = 16000
    force_mono = True
    add_channel_dim = True
    spectrogram_config = SpectrogramConfig(
        stft_config=StftConfig(n_fft=512, hop_length=160, win_length=400, power=2.0),
        mel_scale_config=MelScaleConfig(n_mels=128, f_min=125.0, f_max=7500.0, mel_scale="kaldi"),
        log_mode="log",
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.mel_filters = _linear_to_mel_weight_matrix(
            num_mel_bins=128,
            num_spectrogram_bins=512 // 2 + 1,
            sample_rate=self.sample_rate,
            lower_edge_hertz=125.0,
            upper_edge_hertz=7500.0,
        )

    def _get_features_lengths(self, audio_lengths, spectrogram_config, include_center_frame=False):
        stft_cfg = spectrogram_config.stft_config
        win_length = stft_cfg.win_length or stft_cfg.n_fft
        return (audio_lengths - win_length) // stft_cfg.hop_length + 1

    def extract_spectrogram(self, audio, *, spectrogram_config=None, **kwargs):
        import torch

        if spectrogram_config is None:
            spectrogram_config = self.spectrogram_config

        stft_cfg = spectrogram_config.stft_config
        n_fft = stft_cfg.n_fft
        hop_length = stft_cfg.hop_length
        win_length = stft_cfg.win_length or n_fft

        if isinstance(audio, list):
            waveform = torch.stack(audio, dim=0).to(torch.float64)
        else:
            waveform = audio.to(torch.float64)

        device = waveform.device

        window = torch.hann_window(win_length, periodic=False, device=device, dtype=torch.float64)
        frames = waveform.unfold(-1, win_length, hop_length)
        stft = torch.fft.rfft(window * frames, n=n_fft)
        power_spec = torch.abs(stft) ** 2

        mel_filters = torch.from_numpy(self.mel_filters).to(device)
        mel_spec = torch.clamp(power_spec @ mel_filters, min=1e-5)
        mel_spec = torch.log(mel_spec)

        return [mel_spec[i].to(torch.float32) for i in range(mel_spec.shape[0])]


__all__ = ["LasrAudioProcessor"]
