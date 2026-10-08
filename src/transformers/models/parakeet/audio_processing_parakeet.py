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

from ...audio_processing_utils import BaseAudioProcessor


class ParakeetAudioProcessor(BaseAudioProcessor):
    sampling_rate = 16000
    spectrogram_config = {
        "stft_config": {
            "n_fft": 512,
            "hop_length": 160,
            "win_length": 400,
            "window_fn": "hann_window",
            "power": 2.0,
            "pad_mode": "constant",
            "periodic": False,
            "magnitude_mode": "sqrt_sum_squares",
        },
        "mel_scale_config": {
            "n_mels": 80,
            "f_min": 0.0,
            "norm": "slaney",
            "mel_scale": "slaney",
            "matmul_order": "filters_first_matmul",  # the legacy extractor does `mel_filters @ magnitudes`
            "bank_rounding": "librosa",
            "frequency_bin_mode": "linspace",
        },
        "preemphasis": 0.97,
        "preemphasis_mode": "waveform",
        "log_mode": "log",
        "mel_floor": 0.0,
        "pre_log_offset": 2**-24,
        "transpose_features": True,
    }
    feature_normalization = "per_feature_standardize"


__all__ = ["ParakeetAudioProcessor"]
