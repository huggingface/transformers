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

from ...audio_processing_backends import NumpyAudioBackend
from ...audio_utils import MelScaleConfig, SpectrogramConfig, StftConfig


class Gemma4AudioProcessorNumpy(NumpyAudioBackend):
    sampling_rate = 16000
    force_mono = True
    padding = "longest"
    padding_value = 0.0
    max_length = 480_000
    truncation = True
    pad_to_multiple_of = 128

    legacy_field_mapping = {
        "feature_size": "spectrogram_config.mel_scale_config.n_mels",
        "mel_floor": "spectrogram_config.pre_log_offset",
        "frame_length": "spectrogram_config.stft_config.win_length",
        "fft_length": "spectrogram_config.stft_config.n_fft",
        "min_frequency": "spectrogram_config.mel_scale_config.f_min",
        "max_frequency": "spectrogram_config.mel_scale_config.f_max",
    }

    spectrogram_config = SpectrogramConfig(
        stft_config=StftConfig(
            n_fft=512,
            win_length=320,
            hop_length=160,
            window_fn="hann_window_f32",
            power=1.0,
            center="left",
            frame_extension=1,
            fft_dtype="native",
        ),
        mel_scale_config=MelScaleConfig(
            n_mels=128,
            f_min=0.0,
            f_max=8000.0,
            mel_scale="htk",
            matmul_order="features_first",
        ),
        preemphasis=0.0,
        preemphasis_mode="htk_per_frame",
        mel_floor=0.0,  # no clamp; the log guard is pre_log_offset
        pre_log_offset=1e-3,
        log_mode="log",
        computation_dtype="float64",
    )

    def _postprocess_output(self, output, audio_ranges=None, **kwargs):
        # Zero the padded frames, as the legacy extractor does.
        mask = output.get("audio_features_mask")
        if mask is not None and "audio_features" in output:
            features = output["audio_features"]
            output["audio_features"] = features * mask.astype(features.dtype)[..., None]
        return output


__all__ = ["Gemma4AudioProcessorNumpy"]
