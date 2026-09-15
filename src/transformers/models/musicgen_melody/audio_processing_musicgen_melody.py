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
from ...audio_processing_base import BatchFeature
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
    model_input_names = ["audio_features"]
    # `chroma_filters` is an array and `power_spectrogram_config` is derived, so neither may
    # reach `to_json_string()`
    _excluded_dict_keys = {
        "mel_filters",
        "window",
        "chroma_filters",
        "power_spectrogram_config",
        "_chroma_key",
        "_cached_stft_window",
    }
    # The legacy FE mapped its chroma count to `num_chroma`.
    legacy_field_mapping = {"num_chroma": "n_chroma", "return_attention_mask": None, "return_padding_mask": None}

    n_fft = 16384
    hop_length = 4096
    n_chroma = 12
    chunk_length = 30
    valid_kwargs = MusicgenMelodyAudioProcessorKwargs

    @requires(backends=("librosa",))
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._chroma_key = (self.n_fft, self.hop_length, self.n_chroma)
        self.power_spectrogram_config, self.chroma_filters = self._make_chroma_recipe(*self._chroma_key)

    def _make_chroma_recipe(self, n_fft, hop_length, n_chroma):
        import librosa

        config = SpectrogramConfig(
            stft_config=StftConfig(
                n_fft=n_fft,
                win_length=n_fft,
                hop_length=hop_length,
                power=2.0,
                center=True,
                normalized=True,
                window_fn="hann_window",
                periodic=True,
            ),
        )
        filters = librosa.filters.chroma(sr=self.sampling_rate, n_fft=n_fft, tuning=0, n_chroma=n_chroma)
        return config, self._astype(self._as_backend_array(filters), "float32")

    def _validate_preprocess_kwargs(self, *, n_fft, hop_length, n_chroma, chunk_length, max_length, **kwargs):
        if min(n_fft, hop_length, n_chroma, chunk_length) <= 0:
            raise ValueError("MusicGen Melody FFT, hop, chroma count and chunk length must be positive.")
        super()._validate_preprocess_kwargs(
            max_length=max_length if max_length is not None else chunk_length * self.sampling_rate, **kwargs
        )

    def _preprocess(
        self,
        audio,
        *,
        n_fft,
        hop_length,
        n_chroma,
        chunk_length,
        padding,
        max_length,
        truncation,
        pad_to_multiple_of,
        padding_side,
        padding_value,
        return_tensors,
        **kwargs,
    ):
        """Pad waveforms and compute chroma using one recipe resolved from this call's options."""
        if max_length is None:
            max_length = chunk_length * self.sampling_rate
        padded, _ = self.pad(
            audio,
            padding,
            max_length,
            truncation,
            pad_to_multiple_of,
            padding_side=padding_side,
            padding_value=padding_value,
        )
        if (n_fft, hop_length, n_chroma) == self._chroma_key:
            config, filters = self.power_spectrogram_config, self.chroma_filters
        else:
            config, filters = self._make_chroma_recipe(n_fft, hop_length, n_chroma)
        features = self._compute_chroma(
            self._stack(padded),
            n_fft=n_fft,
            hop_length=hop_length,
            power_spectrogram_config=config,
            chroma_filters=filters,
        )
        return BatchFeature({"audio_features": features}, tensor_type=return_tensors)

    def _pad_for_fft(self, waveform, *, n_fft):
        if waveform.shape[-1] >= n_fft:
            return waveform
        pad = n_fft - waveform.shape[-1]
        return self._pad_axis(waveform, pad // 2, pad // 2 + pad % 2, axis=-1)


class MusicgenMelodyAudioProcessor(MusicgenMelodyAudioProcessorMixin, TorchAudioBackend):
    def _compute_chroma(self, audio, *, n_fft, hop_length, chroma_filters, power_spectrogram_config):
        import torch
        import torchaudio

        waveform = audio  # Already a batched tensor from _stack_waveforms
        device = waveform.device

        # Pad if too short for FFT
        if waveform.shape[-1] < n_fft:
            pad = n_fft - waveform.shape[-1]
            rest = 0 if pad % 2 == 0 else 1
            waveform = torch.nn.functional.pad(waveform, (pad // 2, pad // 2 + rest), "constant", 0)

        # Add channel dim for spectrogram: (batch, 1, length)
        waveform = waveform.unsqueeze(1)

        # Power spectrogram (normalized)
        spec_transform = torchaudio.transforms.Spectrogram(
            n_fft=n_fft,
            win_length=n_fft,
            hop_length=hop_length,
            power=2,
            center=True,
            pad=0,
            normalized=True,
        ).to(device)
        spec = spec_transform(waveform).squeeze(1)

        # Chroma features
        chroma_filters = chroma_filters.to(device)
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
