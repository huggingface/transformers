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

import numpy as np

from ...audio_processing_utils import BaseAudioProcessor
from ...processing_utils import AudioKwargs


class ClvpAudioProcessorKwargs(AudioKwargs, total=False):
    r"""
    mel_norms (`list[float]`, *optional*):
        Per-mel-bin normalization factors the features are divided by. `None` disables it.
    """

    mel_norms: list[float] | None


def _max_length_from_default_audio_length(value, config_dict):
    """CLVP pads to `default_audio_length * sampling_rate`, not to `n_samples`.

    Its legacy `__call__` computes `max_length = self.default_audio_length * self.sampling_rate`
    when the caller passes none. It *also* sets `n_samples = chunk_length * sampling_rate` at init
    and then never reads it — so the base mapping's `n_samples`/`chunk_length` route sends CLVP's
    padding to 30 s where the extractor uses 6 s. Both are neutralised below.
    """
    sampling_rate = config_dict.get("sampling_rate") or ClvpAudioProcessor.sampling_rate
    config_dict.setdefault("max_length", value * sampling_rate)


_max_length_from_default_audio_length.legacy_target = "max_length"


class ClvpAudioProcessor(BaseAudioProcessor):
    legacy_field_mapping = {
        "default_audio_length": _max_length_from_default_audio_length,
        "n_samples": None,
        "chunk_length": None,
    }
    max_length = 132300
    # and never masks it (the legacy FE defaulted to `return_attention_mask=False` too).
    return_padding_mask = False
    sampling_rate = 22050
    spectrogram_config = {
        "stft_config": {
            "n_fft": 1024,
            "hop_length": 256,
            "window_fn": "hann_window",
            "power": 2.0,
            "fft_dtype": "complex64",
        },
        "mel_scale_config": {
            "n_mels": 80,
            "f_min": 0.0,
            "f_max": 8000.0,
            "norm": "slaney",
            "mel_scale": "htk",
            # the legacy extractor built its bank with `mel_filter_bank` (librosa's rounding, linspace bins)
            "bank_rounding": "librosa",
            "frequency_bin_mode": "linspace",
        },
        "log_mode": "log",
        "mel_floor": 1e-5,
        "computation_dtype": "float64",
    }
    truncation = True

    mel_norms = None
    valid_kwargs = ClvpAudioProcessorKwargs

    def _log_compress(self, features, *, spectrogram_config, mel_norms, **kwargs):
        # log and the mel_norms division in the features' dtype (float64 here), rounded once at the end,
        # as the legacy extractor did
        features = self.ops.log(self.ops.clamp_min(features, spectrogram_config.mel_floor))
        if mel_norms is not None:
            # via float64 numpy: a raw list would be read as float32 on torch
            norms = self.ops.asarray(np.asarray(mel_norms, dtype=np.float64), like=features)
            norms = self.ops.astype(norms, self.ops.dtype_name(features))
            features = features / norms[:, None]
        return self.ops.astype(features, "float32")


__all__ = ["ClvpAudioProcessor"]
