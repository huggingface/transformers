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

from typing import Annotated, Literal

import numpy as np
import torch

from ...audio_processing_backends import TorchAudioBackend
from ...audio_processing_base import BatchFeature
from ...processing_utils import AudioKwargs
from ...utils import PaddingStrategy


# CLAP always fills clips to `max_length`, so `padding` cannot name a length strategy other than
# `max_length`. Its legacy spelling names the *fill method* for short audio instead
# (`repeatpad`/`repeat`/`pad`), which the workflow translates to `padding_mode`. The shared
# `padding_validator` knows only the three length strategies; this one replaces it for CLAP.
def clap_padding_validator(value: bool | str | PaddingStrategy | None = None):
    if value not in (True, "max_length", PaddingStrategy.MAX_LENGTH, "repeatpad", "repeat", "pad"):
        raise ValueError(
            "CLAP fills clips to max_length: `padding` must be True or 'max_length', or one of the fill "
            "methods 'repeatpad', 'repeat', 'pad' (preferably via `padding_mode`)."
        )


def _clap_truncation_to_mode_and_mel_bank(value, config_dict):
    # Legacy configs name the mode `truncation` and carry no mel-bank description; the bank was
    # implied by the mode. Fusion checkpoints were trained on torchaudio-default mels (htk scale,
    # no norm), the others on librosa defaults (slaney/slaney, the class default). State the bank
    # once, at load, so the processor never re-derives it: the config decides which bank exists.
    config_dict.setdefault("truncation_mode", value)
    if value == "fusion":
        mel = config_dict.setdefault("spectrogram_config", {}).setdefault("mel_scale_config", {})
        mel.setdefault("mel_scale", "htk")
        mel.setdefault("norm", None)


_clap_truncation_to_mode_and_mel_bank.legacy_target = "truncation_mode"


class ClapAudioProcessorKwargs(AudioKwargs, total=False):
    r"""
    spectrogram_config (`dict` or [`~audio_utils.SpectrogramConfig`], *optional*):
        STFT and mel geometry, including which mel bank the checkpoint was trained with.
    max_length (`int`, *optional*, defaults to 480000):
        Target clip length in waveform samples.
    dither (`float`, *optional*, defaults to 0.0):
        Waveform dithering before feature extraction.
    truncation_mode (`str`, *optional*, defaults to `"rand_trunc"`):
        Strategy used to truncate audio longer than `max_length`.
    padding_mode (`str`, *optional*, defaults to `"repeatpad"`):
        Strategy used to pad audio shorter than `max_length`.
    padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*):
        `True` or `"max_length"` fills to `max_length` using `padding_mode`. The legacy
        spellings `"repeatpad"`, `"repeat"` and `"pad"` override the fill method for this call.
    """

    truncation_mode: Literal["rand_trunc", "fusion"]
    padding_mode: Literal["repeatpad", "repeat", "pad"]
    padding: Annotated[bool | str | PaddingStrategy | None, clap_padding_validator]


class ClapAudioProcessorMixin:
    """CLAP's HTSAT encoder consumes a fixed 1001 x 64 log-mel image: 10 s at 48 kHz, hop 480.

    `_preprocess` makes one from any clip. A short clip is filled to the clip length (tiled, then
    zero-filled). A long clip follows the checkpoint's mode: `rand_trunc` mels a random 10 s
    window, `fusion` mels the whole clip and returns four views of it (a bilinear shrink plus three
    random crops). `is_longer` flags the clips whose fusion crops are real.
    """

    sampling_rate = 48000
    max_length = 480000
    return_padding_mask = False
    # Released checkpoints tile short audio (`padding_mode="repeatpad"`). The mel bank is the
    # checkpoint's: librosa defaults (slaney scale, slaney norm) below; fusion checkpoints carry
    # torchaudio defaults (htk, no norm) in their config, written there by the legacy mapping.
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
        "truncation": _clap_truncation_to_mode_and_mel_bank,
        "top_db": "spectrogram_config.floor_below_peak",
        # The original CLAP recipe always zero-fills and emits is_longer rather than a mask.
        "padding_value": None,
        "return_attention_mask": None,
        "chunk_length_s": None,
        "max_length_s": None,
    }
    extra_model_input_names = ["is_longer"]

    truncation_mode = "rand_trunc"
    padding_mode = "repeatpad"
    valid_kwargs = ClapAudioProcessorKwargs

    def _standardize_kwargs(self, **kwargs):
        kwargs = super()._standardize_kwargs(**kwargs)
        # Legacy call spelling: `padding` naming the fill method rather than a length strategy.
        if kwargs.get("padding") in ("repeatpad", "repeat", "pad"):
            kwargs["padding_mode"] = kwargs["padding"]
            kwargs["padding"] = True
        return kwargs

    def _preprocess(
        self, audio, *, max_length, truncation_mode, padding_mode, spectrogram_config, return_tensors, **kwargs
    ):
        chunk_frames = max_length // spectrogram_config.stft_config.hop_length + 1
        features, is_longer = [], []
        for waveform in audio:
            longer = waveform.shape[-1] > max_length
            if longer and truncation_mode == "rand_trunc":
                waveform = self._random_window(waveform, max_length)
            else:
                waveform = self._fill(waveform, max_length, padding_mode)
            mel = self.spectrogram(waveform, spectrogram_config=spectrogram_config, **kwargs)
            if truncation_mode == "fusion":
                mel, longer = self._fusion_views(mel, chunk_frames)
            else:
                mel = mel[None]
            features.append(mel)
            is_longer.append(longer)
        if truncation_mode == "fusion" and not any(is_longer):
            # HTSAT's fusion path expects at least one selected clip even in an all-short batch.
            is_longer[np.random.randint(0, len(is_longer))] = True
        return BatchFeature(
            {"audio_features": self._stack(features), "is_longer": [[longer] for longer in is_longer]},
            tensor_type=return_tensors,
        )

    def _random_window(self, waveform, max_length):
        start = np.random.randint(0, waveform.shape[-1] - max_length + 1)
        return waveform[..., start : start + max_length]

    def _fill(self, waveform, max_length, padding_mode):
        """Bring a short clip to `max_length`: tile it (`repeat`, `repeatpad`), then zero-fill."""
        length = waveform.shape[-1]
        if length >= max_length:
            return waveform
        if padding_mode in ("repeat", "repeatpad"):
            repeats = max_length // length + (padding_mode == "repeat")
            waveform = self._concat_last([waveform] * repeats)[..., :max_length]
        return self._pad_axis(waveform, 0, max_length - waveform.shape[-1], axis=-1, value=0.0)

    def _fusion_views(self, mel, chunk_frames):
        """Four `chunk_frames` views of a whole-clip mel, and whether the crops are real."""
        # An overlong clip can still fit in `chunk_frames`; it is then shown four times, unfused.
        if mel.shape[0] <= chunk_frames:
            return self._stack([mel] * 4), False
        ranges = np.array_split(list(range(0, mel.shape[0] - chunk_frames + 1)), 3)
        starts = [np.random.choice(r if len(r) else [0]) for r in ranges]
        crops = [mel[start : start + chunk_frames] for start in starts]
        return self._stack([self._bilinear_shrink(mel, chunk_frames)] + crops), True

    def _bilinear_shrink(self, mel, chunk_frames):
        mel_tensor = torch.as_tensor(mel)[None, None]
        shrunk = torch.nn.functional.interpolate(
            mel_tensor, size=[chunk_frames, mel.shape[-1]], mode="bilinear", align_corners=False
        )
        return self._as_backend_array(shrunk[0, 0])


class ClapAudioProcessor(ClapAudioProcessorMixin, TorchAudioBackend):
    pass


__all__ = ["ClapAudioProcessor"]
