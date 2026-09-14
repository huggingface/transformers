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
    sampling_rate = config_dict.get("sampling_rate") or ClvpAudioProcessorMixin.sampling_rate
    config_dict.setdefault("max_length", value * sampling_rate)


_max_length_from_default_audio_length.legacy_target = "max_length"


class ClvpAudioProcessorMixin:
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
            "frequency_bin_mode": "linspace",
        },
        "log_mode": "log",
        "mel_floor": 1e-5,
        "computation_dtype": "float64",
    }
    truncation = True

    mel_norms = None
    valid_kwargs = ClvpAudioProcessorKwargs


class ClvpAudioProcessor(ClvpAudioProcessorMixin, TorchAudioBackend):
    def _log_compress(self, features, *, spectrogram_config, mel_norms, **kwargs):
        # Compute log and mel_norms division in float64 before casting to float32
        # to match the legacy feature extractor's precision (same recipe as the numpy sibling).
        mel_floor = spectrogram_config.mel_floor
        features = torch.log(
            torch.maximum(torch.tensor(mel_floor, dtype=features.dtype, device=features.device), features)
        )
        if mel_norms is not None:
            mel_norms = torch.as_tensor(mel_norms, dtype=features.dtype, device=features.device)[:, None]
            features = features / mel_norms
        return features.to(torch.float32)


__all__ = ["ClvpAudioProcessor"]
