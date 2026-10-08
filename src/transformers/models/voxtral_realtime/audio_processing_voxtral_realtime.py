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
from ...processing_utils import AudioKwargs


class VoxtralRealtimeAudioProcessorKwargs(AudioKwargs, total=False):
    r"""
    global_log_mel_max (`float`, *optional*, defaults to 1.5):
        Fixed peak for the log-mel floor: the features are floored at `global_log_mel_max - 8` and then rescaled
        as `(x + 4) / 4`. `None` uses each spectrogram's own peak (its max over the last two axes).
    """

    global_log_mel_max: float


class VoxtralRealtimeAudioProcessor(BaseAudioProcessor):
    sampling_rate = 16000
    spectrogram_config = {
        "stft_config": {
            "n_fft": 400,
            "hop_length": 160,
            "power": 2.0,
        },
        "mel_scale_config": {
            "n_mels": 128,
            "mel_scale": "slaney",
            "norm": "slaney",
            "computation_dtype": "float64",
            # the legacy extractor does `mel_filters @ magnitudes`; torchaudio's order (the default) differs
            # from it in the last ulp
            "matmul_order": "filters_first_matmul",
        },
        "log_mode": "log10",
        "drop_last_frame": True,
    }

    global_log_mel_max = 1.5
    valid_kwargs = VoxtralRealtimeAudioProcessorKwargs

    def _shape_log_features(self, features, spectrogram_config, *, global_log_mel_max, **kwargs):
        # Floors at `peak - 8` with the fixed `global_log_mel_max` as the peak (the per-spectrogram peak when
        # None), then rescales `(x + 4) / 4`. The base `floor_below_peak` only knows the per-spectrogram peak,
        # so this is written out here.
        features = super()._shape_log_features(features, spectrogram_config)
        if global_log_mel_max is not None:
            features = self.ops.clamp_min(features, global_log_mel_max - 8.0)
        else:
            peak = self.ops.amax(features, (-2, -1), keepdims=True)
            features = self.ops.maximum(features, peak - 8.0)
        return (features + 4.0) / 4.0


__all__ = ["VoxtralRealtimeAudioProcessor"]
