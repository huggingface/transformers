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
from ...utils.import_utils import requires


class MusicgenMelodyAudioProcessor(TorchAudioBackend):
    sample_rate = 32000
    force_mono = True
    do_extract_spectrogram = True
    return_padding_mask = False
    n_fft = 16384
    hop_length = 4096
    n_chroma = 12
    chunk_length = 30

    @requires(backends=("librosa", "torch"))
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        import librosa
        import torch

        self.chroma_filters = torch.from_numpy(
            librosa.filters.chroma(sr=self.sample_rate, n_fft=self.n_fft, tuning=0, n_chroma=self.n_chroma)
        ).float()

    def extract_spectrogram(self, audio, **kwargs):
        import torch
        import torchaudio

        waveform = audio  # Already a batched tensor from _to_batch
        device = waveform.device
        batch_size = waveform.shape[0]

        # Pad if too short for FFT
        if waveform.shape[-1] < self.n_fft:
            pad = self.n_fft - waveform.shape[-1]
            rest = 0 if pad % 2 == 0 else 1
            waveform = torch.nn.functional.pad(waveform, (pad // 2, pad // 2 + rest), "constant", 0)

        # Add channel dim for spectrogram: (batch, 1, length)
        waveform = waveform.unsqueeze(1)

        # Power spectrogram (normalized)
        spec_transform = torchaudio.transforms.Spectrogram(
            n_fft=self.n_fft, win_length=self.n_fft, hop_length=self.hop_length,
            power=2, center=True, pad=0, normalized=True,
        ).to(device)
        spec = spec_transform(waveform).squeeze(1)

        # Chroma features
        chroma_filters = self.chroma_filters.to(device)
        raw_chroma = torch.einsum("cf, ...ft->...ct", chroma_filters, spec)

        # Normalize with inf norm
        norm_chroma = torch.nn.functional.normalize(raw_chroma, p=float("inf"), dim=-2, eps=1e-6)

        # Transpose: (batch, chroma, frames) -> (batch, frames, chroma)
        norm_chroma = norm_chroma.transpose(1, 2)

        # One-hot encoding: argmax along chroma dim
        idx = norm_chroma.argmax(-1, keepdim=True)
        norm_chroma[:] = 0
        norm_chroma.scatter_(dim=-1, index=idx, value=1)

        return norm_chroma


__all__ = ["MusicgenMelodyAudioProcessor"]
