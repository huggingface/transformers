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
from ...audio_utils import _clamp_min
from ...processing_utils import AudioKwargs


class VoxtralRealtimeAudioProcessorKwargs(AudioKwargs, total=False):
    r"""
    global_log_mel_max (`float`, *optional*, defaults to 1.5):
        Fixed ceiling used to clamp the log-mel features, in place of a per-clip maximum.
    """

    global_log_mel_max: float


class VoxtralRealtimeAudioProcessorMixin:
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
            # the legacy extractor does `mel_filters @ magnitudes`; `F.linear` (the default)
            # differs from it in the last ulp
            "matmul_order": "filters_first_matmul",
        },
        "log_mode": "log10",
        "skip_last_frame": True,
    }

    global_log_mel_max = 1.5
    valid_kwargs = VoxtralRealtimeAudioProcessorKwargs

    def _log_compress(self, features, *, spectrogram_config, **kwargs):
        # Voxtral uses a *fixed* `global_log_mel_max` as the upper bound (rather than the
        # per-utterance amax that the base `clip_max_offset` field expects), so we don't set
        # the post-log fields on `spectrogram_config` and handle the whole rescale here.
        features = super()._log_compress(features, spectrogram_config=spectrogram_config, **kwargs)
        spec_max = (
            self.global_log_mel_max if self.global_log_mel_max is not None else self._amax_over_features(features)
        )
        features = _clamp_min(features, spec_max - 8.0)
        return (features + 4.0) / 4.0


class VoxtralRealtimeAudioProcessor(VoxtralRealtimeAudioProcessorMixin, TorchAudioBackend):
    pass


__all__ = ["VoxtralRealtimeAudioProcessor"]
