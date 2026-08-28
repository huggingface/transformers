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
from ...audio_utils import SpectrogramConfig, StftConfig
from ...processing_utils import AudioKwargs
from ...utils.import_utils import requires


class MusicgenMelodyAudioProcessorKwargs(AudioKwargs, total=False):
    r"""
    n_fft (`int`, *optional*, defaults to 16384):
        Size of the FFT used for the chroma spectrogram.
    hop_length (`int`, *optional*, defaults to 4096):
        Hop between successive chroma frames, in samples.
    n_chroma (`int`, *optional*, defaults to 12):
        Number of chroma bins.
    chunk_length (`int`, *optional*, defaults to 30):
        Length, in seconds, of the audio window the model consumes.
    """

    n_fft: int
    hop_length: int
    n_chroma: int
    chunk_length: int


class MusicgenMelodyAudioProcessorMixin:
    sampling_rate = 32000
    force_mono = True
    do_extract_spectrogram = True
    return_padding_mask = False
    # `chroma_filters` is an array and `power_spectrogram_config` is derived, so neither may
    # reach `to_json_string()`
    _excluded_dict_keys = {"mel_filters", "window", "chroma_filters", "power_spectrogram_config"}
    # The legacy FE mapped its chroma count to `num_chroma`.
    legacy_field_mapping = {"num_chroma": "n_chroma"}

    n_fft = 16384
    hop_length = 4096
    n_chroma = 12
    chunk_length = 30
    valid_kwargs = MusicgenMelodyAudioProcessorKwargs

    @requires(backends=("librosa",))
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        import librosa

        # Only used by the numpy sibling; the torch one goes through `torchaudio.transforms.Spectrogram`.
        self.power_spectrogram_config = SpectrogramConfig(
            stft_config=StftConfig(
                n_fft=self.n_fft,
                win_length=self.n_fft,
                hop_length=self.hop_length,
                power=2.0,
                center=True,
                normalized=True,
                window_fn="hann_window",
                periodic=True,
            ),
        )

        filters = librosa.filters.chroma(sr=self.sampling_rate, n_fft=self.n_fft, tuning=0, n_chroma=self.n_chroma)
        self.chroma_filters = self._astype(self._as_backend_array(filters), "float32")

    def _pad_for_fft(self, waveform):
        if waveform.shape[-1] >= self.n_fft:
            return waveform
        pad = self.n_fft - waveform.shape[-1]
        return self._pad_axis(waveform, pad // 2, pad // 2 + pad % 2, axis=-1)


class MusicgenMelodyAudioProcessor(MusicgenMelodyAudioProcessorMixin, TorchAudioBackend):
    def extract_spectrogram(self, audio, **kwargs):
        import torch
        import torchaudio

        waveform = audio  # Already a batched tensor from _to_batch
        device = waveform.device

        # Pad if too short for FFT
        if waveform.shape[-1] < self.n_fft:
            pad = self.n_fft - waveform.shape[-1]
            rest = 0 if pad % 2 == 0 else 1
            waveform = torch.nn.functional.pad(waveform, (pad // 2, pad // 2 + rest), "constant", 0)

        # Add channel dim for spectrogram: (batch, 1, length)
        waveform = waveform.unsqueeze(1)

        # Power spectrogram (normalized)
        spec_transform = torchaudio.transforms.Spectrogram(
            n_fft=self.n_fft,
            win_length=self.n_fft,
            hop_length=self.hop_length,
            power=2,
            center=True,
            pad=0,
            normalized=True,
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
