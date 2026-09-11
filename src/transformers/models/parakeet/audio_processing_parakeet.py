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

import numpy as np

from ...audio_processing_backends import TorchAudioBackend


class ParakeetAudioProcessorMixin:
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
            "matmul_order": "filters_first_matmul",
            "bank_rounding": "librosa",
        },
        "preemphasis": 0.97,
        "preemphasis_mode": "waveform",
        "log_mode": "log",
        "mel_floor": 0.0,
        "pre_log_offset": 2**-24,
        "transpose_features": True,
    }

    def _finalize_output(self, output, audio_ranges=None, **kwargs):
        if audio_ranges is None or "audio_features" not in output:
            return output
        audio_lengths = np.asarray([end - start for start, end in audio_ranges])
        frame_counts = self._valid_frame_counts(audio_lengths, self.spectrogram_config)
        output["audio_features"] = self._standardize_features(output["audio_features"], frame_counts, eps=1e-5)
        return output


class ParakeetAudioProcessor(ParakeetAudioProcessorMixin, TorchAudioBackend):
    pass


__all__ = ["ParakeetAudioProcessor"]
