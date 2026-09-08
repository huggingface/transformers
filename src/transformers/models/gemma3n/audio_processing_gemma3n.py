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


class Gemma3nAudioProcessorMixin:
    max_length = 480000
    pad_to_multiple_of = 128
    sampling_rate = 16000
    spectrogram_config = {
        "stft_config": {
            "n_fft": 1024,
            "win_length": 512,
            "hop_length": 160,
            "power": 1.0,
            "center": False,
            "window_fn": "hann_window_f32",
            "frame_extension": 1,
            "fft_dtype": "float64",
        },
        "mel_scale_config": {
            "n_mels": 128,
            "f_min": 125.0,
            "f_max": 7600.0,
            "mel_scale": "htk",
            "matmul_order": "features_first",
        },
        "mel_floor": 1e-5,
        "log_mode": "log",
        "preemphasis": 0.97,
        "preemphasis_mode": "htk_per_frame",
        "count_partial_frames": True,
        "computation_dtype": "float64",
    }
    truncation = True


class Gemma3nAudioProcessor(Gemma3nAudioProcessorMixin, TorchAudioBackend):
    pass


__all__ = ["Gemma3nAudioProcessor"]
