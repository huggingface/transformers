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

from typing import Annotated

import numpy as np

from ...audio_processing_base import BatchFeature
from ...audio_processing_utils import BaseAudioProcessor
from ...audio_utils import SpectrogramConfig, StftConfig
from ...processing_utils import AudioKwargs
from ...utils.import_utils import requires
from ...utils.type_validators import strictly_positive


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

    n_fft: Annotated[int, strictly_positive]
    hop_length: Annotated[int, strictly_positive]
    n_chroma: Annotated[int, strictly_positive]
    chunk_length: Annotated[int, strictly_positive]


class MusicgenMelodyAudioProcessor(BaseAudioProcessor):
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
        "ops",
        "backend",
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
        return config, filters.astype(np.float32)

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
            self.ops.stack(padded),
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
        return self.ops.pad(waveform, pad // 2, pad // 2 + pad % 2, axis=-1)

    def _compute_chroma(self, audio, *, n_fft, hop_length, chroma_filters, power_spectrogram_config):
        """One-hot chroma: a normalized power spectrogram (`torchaudio.transforms.Spectrogram(normalized=True)`)
        projected on the host chroma filters, L-inf normalized over the chroma axis as
        `F.normalize(p=inf, dim=-2, eps=1e-6)` does, then the argmax of each frame (the first on ties)."""
        waveform = self._pad_for_fft(audio, n_fft=n_fft)
        spec = self._waveform_to_spectrum(waveform, spectrogram_config=power_spectrogram_config, dither=0.0)
        filters = self.ops.astype(self.ops.asarray(chroma_filters, like=spec), self.ops.dtype_name(spec))
        raw_chroma = self.ops.matmul(filters, spec)
        denom = self.ops.clamp_min(self.ops.amax(self.ops.abs(raw_chroma), -2, keepdims=True), 1e-6)
        norm_chroma = self.ops.swapaxes(raw_chroma / denom, 1, 2)
        idx = self.ops.argmax(norm_chroma, -1)
        one_hot = self.ops.arange(norm_chroma.shape[-1], like=norm_chroma)[None, None, :] == idx[..., None]
        # float32 like the legacy one-hot, on every backend: numpy's spectrum (float64 window) would make it float64.
        return self.ops.astype(one_hot, "float32")


__all__ = ["MusicgenMelodyAudioProcessor"]
