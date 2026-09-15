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
from ...audio_processing_base import AudioProcessingMixin, BatchFeature
from ...processing_utils import AudioKwargs
from ...utils import PaddingStrategy
from ...utils.type_validators import padding_validator


# CLAP's legacy spelling of `padding` names the *fill method* for short audio rather than the target
# length, so `repeatpad`/`repeat`/`pad` are legal values here and the CLAP workflow
# translates them. The base validator knows only the three length strategies and rejects them, which
# it did silently until the `Annotated` validators were activated. Widen it for this model rather
# than teaching the shared validator a CLAP-only vocabulary.
def clap_padding_validator(value: bool | str | PaddingStrategy | None = None):
    if value in ("repeatpad", "repeat", "pad"):
        return
    padding_validator(value)


class ClapAudioProcessorKwargs(AudioKwargs, total=False):
    r"""
    spectrogram_config (`dict` or [`~audio_utils.SpectrogramConfig`], *optional*):
        STFT and mel geometry. The truncation mode selects the mel scale and normalization.
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
        # The original CLAP recipe always zero-fills and emits is_longer rather than a mask.
        "padding_value": None,
        "return_attention_mask": None,
        "chunk_length_s": None,
        "max_length_s": None,
    }
    extra_model_input_names = ["is_longer"]
    _excluded_dict_keys = AudioProcessingMixin._excluded_dict_keys | {"_clap_mel_bank"}

    truncation_mode = "rand_trunc"
    padding_mode = "repeatpad"
    valid_kwargs = ClapAudioProcessorKwargs

    def _set_attributes(self, **kwargs):
        super()._set_attributes(**kwargs)
        if self.truncation_mode == "fusion":
            mel_scale_config = replace(self.spectrogram_config.mel_scale_config, mel_scale="htk", norm=None)
            self.spectrogram_config = replace(self.spectrogram_config, mel_scale_config=mel_scale_config)
            self.mel_filters = self._mel_filter_bank(self.spectrogram_config)
        self._clap_mel_bank = (self.spectrogram_config, self.mel_filters)

    def _validate_preprocess_kwargs(self, *, truncation_mode, padding_mode, max_length, padding, **kwargs):
        """Validate the options used by CLAP's crop/fusion workflow."""
        if truncation_mode not in ("rand_trunc", "fusion"):
            raise ValueError("CLAP truncation_mode must be 'rand_trunc' or 'fusion'.")
        if padding not in ("repeatpad", "repeat", "pad", True, "max_length", PaddingStrategy.MAX_LENGTH):
            raise ValueError("CLAP fills clips to max_length; use padding_mode='repeatpad', 'repeat' or 'pad'.")
        if padding_mode not in ("repeatpad", "repeat", "pad"):
            raise ValueError("CLAP padding_mode must be 'repeatpad', 'repeat' or 'pad'.")
        if max_length is None:
            raise ValueError("CLAP requires max_length in waveform samples.")

    def _preprocess(
        self,
        audio,
        *,
        truncation_mode,
        padding_mode,
        max_length,
        spectrogram_config,
        return_tensors,
        padding,
        **kwargs,
    ):
        """CLAP's two recipes: crop waveforms for one view, or fuse full-clip mels into four views.

        Each clip returns its features and metadata together. The backend handles the numerical
        extraction; CLAP owns the order of cropping, filling, extraction and view construction.
        """
        if padding in ("repeatpad", "repeat", "pad"):
            padding_mode = padding
        # The original extractor chooses its filter bank by the *call's* mode. Keep that choice
        # local, including when a caller switches modes or supplies a spectrogram config.
        mel_scale, norm = ("htk", None) if truncation_mode == "fusion" else ("slaney", "slaney")
        mel_config = spectrogram_config.mel_scale_config
        if mel_config.mel_scale != mel_scale or mel_config.norm != norm:
            spectrogram_config = replace(
                spectrogram_config, mel_scale_config=replace(mel_config, mel_scale=mel_scale, norm=norm)
            )
        cached_config, cached_filters = self._clap_mel_bank
        mel_filters = (
            cached_filters if spectrogram_config is cached_config else self._mel_filter_bank(spectrogram_config)
        )
        features, is_longer = [], []
        for waveform in audio:
            views, longer = self._get_input_mel(
                waveform,
                max_length=max_length,
                truncation_mode=truncation_mode,
                padding_mode=padding_mode,
                spectrogram_config=spectrogram_config,
                mel_filters=mel_filters,
                **kwargs,
            )
            features.append(views)
            is_longer.append(longer)

        # HTSAT's fusion path expects at least one selected clip even in an all-short batch.
        if truncation_mode == "fusion" and not any(is_longer):
            is_longer[np.random.randint(0, len(is_longer))] = True
        return BatchFeature(
            {"audio_features": features, "is_longer": [[longer] for longer in is_longer]},
            tensor_type=return_tensors,
        )

    def _get_input_mel(self, waveform, *, max_length, truncation_mode, padding_mode, spectrogram_config, **kwargs):
        """Return one clip's mel views and fusion/cropping metadata without storing call state."""
        length = waveform.shape[-1]
        if length == 0:
            raise ValueError("CLAP requires a non-empty waveform.")
        longer = length > max_length
        if longer and truncation_mode == "rand_trunc":
            start = np.random.randint(0, length - max_length + 1)
            waveform = waveform[start : start + max_length]
        elif length < max_length:
            if padding_mode in ("repeat", "repeatpad"):
                repeats = max_length // length + (padding_mode == "repeat")
                waveform = self._concat_last([waveform] * repeats)[:max_length]
            waveform = self._pad_axis(waveform, 0, max_length - waveform.shape[-1], axis=-1, value=0.0)

        mel = super().compute_features(waveform, spectrogram_config=spectrogram_config, **kwargs)
        if truncation_mode == "rand_trunc":
            return mel[None], longer
        chunk_frames = max_length // spectrogram_config.stft_config.hop_length + 1
        # An overlong waveform can still fit in the same number of mel frames.
        if mel.shape[0] <= chunk_frames:
            return self._stack([mel] * 4), False
        return self._random_mel_fusion(mel, chunk_frames), True

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


class ClapAudioProcessor(ClapAudioProcessorMixin, TorchAudioBackend):
    pass


__all__ = ["ClapAudioProcessor"]
