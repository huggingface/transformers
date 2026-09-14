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

from dataclasses import replace
from typing import Annotated

import numpy as np
import torch

from ...audio_processing_backends import TorchAudioBackend
from ...processing_utils import AudioKwargs
from ...utils import PaddingStrategy
from ...utils.type_validators import padding_validator


# CLAP's legacy spelling of `padding` names the *fill method* for short audio rather than the target
# length, so `repeatpad`/`repeat`/`pad` are legal values here and `_resolve_padding_strategy`
# translates them. The base validator knows only the three length strategies and rejects them, which
# it did silently until the `Annotated` validators were activated. Widen it for this model rather
# than teaching the shared validator a CLAP-only vocabulary.
def clap_padding_validator(value: bool | str | PaddingStrategy | None = None):
    if value in ("repeatpad", "repeat", "pad"):
        return
    padding_validator(value)


class ClapAudioProcessorKwargs(AudioKwargs, total=False):
    r"""
    truncation_mode (`str`, *optional*, defaults to `"rand_trunc"`):
        Strategy used to truncate audio longer than `max_length`.
    padding_mode (`str`, *optional*, defaults to `"repeatpad"`):
        Strategy used to pad audio shorter than `max_length`.
    padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*):
        As the base field, and additionally accepts CLAP's legacy fill-method spellings
        `"repeatpad"`, `"repeat"` and `"pad"`.
    """

    truncation_mode: str
    padding_mode: str
    padding: Annotated[bool | str | PaddingStrategy | None, clap_padding_validator]


class ClapAudioProcessorMixin:
    sampling_rate = 48000
    max_length = 480000
    return_padding_mask = False
    # Released checkpoints tile short audio (`padding_mode="repeatpad"`). In `rand_trunc` mode HTSAT
    # mels the waveform itself with librosa defaults (slaney scale, slaney norm); `fusion`
    # checkpoints were trained on precomputed torchaudio-default mels (htk scale, no norm), which
    # `_set_attributes` swaps in.
    spectrogram_config = {
        "stft_config": {"n_fft": 1024, "hop_length": 480, "power": 2.0, "fft_dtype": "complex64"},
        "mel_scale_config": {
            "n_mels": 64,
            "f_min": 50,
            "f_max": 14000,
            "mel_scale": "slaney",
            "norm": "slaney",
            "frequency_bin_mode": "linspace",
            "computation_dtype": "float64",
        },
        "log_mode": "dB",
        "computation_dtype": "float64",
        "transpose_features": True,
    }
    legacy_field_mapping = {
        # Hub configs spell these `padding`/`truncation`; the modern names are the CLAP-specific
        # `padding_mode`/`truncation_mode`, leaving `padding`/`truncation` their base meaning.
        "padding": "padding_mode",
        "truncation": "truncation_mode",
        "top_db": "spectrogram_config.floor_below_peak",
        "chunk_length_s": None,
        "max_length_s": None,
    }
    extra_model_input_names = ["is_longer"]

    truncation_mode = "rand_trunc"
    padding_mode = "repeatpad"
    valid_kwargs = ClapAudioProcessorKwargs

    def _set_attributes(self, **kwargs):
        super()._set_attributes(**kwargs)
        if self.truncation_mode == "fusion":
            mel_scale_config = replace(self.spectrogram_config.mel_scale_config, mel_scale="htk", norm=None)
            self.spectrogram_config = replace(self.spectrogram_config, mel_scale_config=mel_scale_config)
            self.mel_filters = self._mel_filter_bank(self.spectrogram_config)
        # `rand_trunc` crops the waveform in `pad`; `fusion` keeps it whole and crops the mel instead.
        self.truncation = self.truncation_mode == "rand_trunc"

    def _resolve_padding_strategy(self, padding=False, max_length=None, *, padding_value, **kwargs):
        if padding in ("repeatpad", "repeat", "pad"):
            padding = True
        if padding is True and max_length is not None:
            return PaddingStrategy.MAX_LENGTH
        return super()._resolve_padding_strategy(padding=padding, max_length=max_length, padding_value=padding_value)

    def pad(self, audio, padding=True, *args, padding_mode, **kwargs):
        self._is_longer_flags = []
        if padding in ("repeatpad", "repeat", "pad"):
            # legacy spelling: `padding` named the fill method for short audio, not the target length
            padding_mode = padding
        return super().pad(audio, padding, *args, padding_mode=padding_mode, **kwargs)

    def _stack_waveforms(self, audio, **kwargs):
        # one mel per clip, so the clips stay a list rather than a (batch, samples) array
        return audio

    def _pad_waveform(self, audio, max_length, *, padding_mode, **kwargs):
        """Tile short audio before the base class zero-pads whatever remains."""
        if padding_mode in ("repeat", "repeatpad") and audio.shape[-1] < max_length:
            n_repeat = max_length // audio.shape[-1] + (padding_mode == "repeat")
            audio = self._concat_last([audio] * n_repeat)[..., :max_length]
        return super()._pad_waveform(audio, max_length, **kwargs)

    def _truncate_waveform(self, audio_el, max_length, **kwargs):
        """Random crop to `max_length` (rand_trunc mode), remembering which clips were longer."""
        overflow = audio_el.shape[-1] - max_length
        self._is_longer_flags.append(overflow > 0)
        idx = np.random.randint(0, overflow + 1) if overflow > 0 else 0
        return audio_el[..., idx : idx + max_length]

    def compute_features(self, audio, *, spectrogram_config, truncation_mode, max_length, **kwargs):
        """One (1, frames, 64) mel per clip in `rand_trunc` mode; four views per clip in `fusion` mode."""
        if not isinstance(audio, list):
            audio = list(audio) if audio.ndim == 2 else [audio]
        mels = super().compute_features(audio, spectrogram_config=spectrogram_config, **kwargs)
        if truncation_mode != "fusion":
            return [mel[None] for mel in mels]
        # the *merged* `max_length`, not `self.max_length`: `pad` has already used the per-call value,
        # and reading the instance here made the two disagree whenever a caller passed one (issue 39)
        chunk_frames = max_length // spectrogram_config.stft_config.hop_length + 1
        self._is_longer_flags = [mel.shape[0] > chunk_frames for mel in mels]
        return [
            self._random_mel_fusion(mel, chunk_frames) if mel.shape[0] > chunk_frames else self._stack([mel] * 4)
            for mel in mels
        ]

    def _random_mel_fusion(self, mel, chunk_frames):
        """A bilinear shrink of the whole mel plus three random `chunk_frames` crops (front, middle, back)."""
        ranges = np.array_split(list(range(0, mel.shape[0] - chunk_frames + 1)), 3)
        starts = [np.random.choice(r if len(r) else [0]) for r in ranges]
        crops = [mel[start : start + chunk_frames] for start in starts]
        return self._stack([self._bilinear_shrink(mel, chunk_frames)] + crops)

    def _bilinear_shrink(self, mel, chunk_frames):
        mel_tensor = torch.as_tensor(mel)[None, None]
        shrunk = torch.nn.functional.interpolate(
            mel_tensor, size=[chunk_frames, mel.shape[-1]], mode="bilinear", align_corners=False
        )
        return self._as_backend_array(shrunk[0, 0])

    def _finalize_output(self, output, audio_ranges=None, *, truncation_mode, **kwargs):
        """`is_longer` stands in for the padding mask: it tells HTSAT which clips carry fusion crops."""
        is_longer = self._is_longer_flags or [False] * len(audio_ranges)
        if truncation_mode == "fusion" and not any(is_longer):
            is_longer[np.random.randint(0, len(is_longer))] = True
        output["is_longer"] = [[longer] for longer in is_longer]
        return output


class ClapAudioProcessor(ClapAudioProcessorMixin, TorchAudioBackend):
    pass


__all__ = ["ClapAudioProcessor"]
