# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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


def _stft_frame_from_window_size(value, config_dict):
    """`window_size` is the STFT frame: the legacy extractor passes it as `frame_length` and sizes
    its filterbank with `(window_size // 2) + 1` frequency bins, i.e. it is `n_fft` as well."""
    stft_config = config_dict.setdefault("spectrogram_config", {}).setdefault("stft_config", {})
    stft_config.setdefault("n_fft", value)
    stft_config.setdefault("win_length", value)


_stft_frame_from_window_size.legacy_target = "spectrogram_config.stft_config.n_fft"


class Pop2PianoAudioProcessorMixin:
    # Tokenizer and model-head parameters that share this processor's config file. The legacy
    # extractor takes only (sampling_rate, padding_value, window_size, hop_length, min_frequency,
    # feature_size, num_bars), so none of these reach feature extraction on either side.
    legacy_field_mapping = {
        "default_velocity": None,
        "eos_token_id": None,
        "input_length": None,
        "mel_is_conditioned": None,
        "start_token_id": None,
        "target_length": None,
        "window_size": _stft_frame_from_window_size,
        # `num_bars` drives the beat-step extrapolation in the legacy `__call__`, which this
        # processor does not port -- the beat path needs Essentia and is the reason pop2piano is
        # the one parity xfail. Dropped rather than declared, since nothing here reads it.
        "num_bars": None,
    }
    sampling_rate = 22050
    spectrogram_config = {
        "stft_config": {"n_fft": 4096, "hop_length": 1024, "power": 2.0},
        "mel_scale_config": {"n_mels": 512, "f_min": 10.0, "mel_scale": "htk"},
        "log_mode": "log10",
    }


class Pop2PianoAudioProcessor(Pop2PianoAudioProcessorMixin, TorchAudioBackend):
    pass


__all__ = ["Pop2PianoAudioProcessor"]
