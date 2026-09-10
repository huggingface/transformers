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

from ...audio_processing_backends import TorchAudioBackend


class Gemma4AudioProcessorMixin:
    legacy_field_mapping = {
        "mel_floor": "spectrogram_config.pre_log_offset",
    }
    max_length = 480_000
    pad_to_multiple_of = 128
    padding = "longest"
    sampling_rate = 16000
    spectrogram_config = {
        "stft_config": {
            "n_fft": 512,
            "win_length": 320,
            "hop_length": 160,
            # Gemma4's legacy extractor uses `window_function(...).astype(np.float32)` -- float64
            # cosine, stored as float32. Gemma3n instead builds the window inline from a float32
            # arange, which is what `hann_window_f32` reproduces; the two differ by 2.4e-07.
            "window_fn": "hann_window_f64_as_f32",
            "power": 1.0,
            "center": "left",
            "frame_extension": 1,
            # `np.fft.rfft` always promotes to complex128 regardless of float32 input, so the
            # legacy magnitudes are float64. "native" left the FFT in complex64.
            "fft_dtype": "float64",
        },
        "mel_scale_config": {
            "n_mels": 128,
            "f_min": 0.0,
            "f_max": 8000.0,
            "mel_scale": "htk",
            "matmul_order": "features_first",
        },
        "preemphasis": 0.0,
        "preemphasis_mode": "htk_per_frame",
        "mel_floor": 0.0,
        "pre_log_offset": 1e-3,
        "log_mode": "log",
        "computation_dtype": "float64",
    }
    truncation = True


class Gemma4AudioProcessor(Gemma4AudioProcessorMixin, TorchAudioBackend):
    def _postprocess_output(self, output, audio_ranges=None, **kwargs):
        # Zero the padded frames, as the legacy extractor does.
        mask = output.get("audio_features_mask")
        if mask is not None and "audio_features" in output:
            features = output["audio_features"]
            output["audio_features"] = features * mask.to(features.dtype).unsqueeze(-1)
        return output


__all__ = ["Gemma4AudioProcessor"]
