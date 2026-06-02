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

from ...audio_processing_backends import NumpyAudioBackend
from ...audio_utils import MelScaleConfig, SpectrogramConfig, StftConfig
from .audio_processing_whisper import _whisper_chunk_length_to_max_length


class WhisperAudioProcessorNumpy(NumpyAudioBackend):
    """NumPy sibling of [`WhisperAudioProcessor`]. Required to produce bit-exact outputs
    against the torch sibling (ADR 0001)."""

    sample_rate = 16000
    force_mono = True
    return_padding_mask = False
    truncation = True
    max_length = 480000  # 30 seconds at 16000 Hz
    spectrogram_config = SpectrogramConfig(
        stft_config=StftConfig(
            n_fft=400,
            hop_length=160,
            power=2.0,
        ),
        mel_scale_config=MelScaleConfig(
            n_mels=80,
            mel_scale="slaney",
            norm="slaney",
            computation_dtype="float64",
        ),
        log_mode="log10",
        skip_last_frame=True,
        clip_max_offset=8.0,
        post_log_shift=4.0,
        post_log_scale=0.25,
    )

    legacy_field_mapping = {
        "feature_size": "spectrogram_config.mel_scale_config.n_mels",
        "chunk_length": _whisper_chunk_length_to_max_length,
        "n_samples": "max_length",
    }

    def _apply_mel_scale(self, features, *, spectrogram_config, **kwargs):
        # `filters_first` matmul order with mel_floor clamp, matching the torch sibling.
        return np.maximum(spectrogram_config.mel_floor, np.matmul(self.mel_filters.T, features))


__all__ = ["WhisperAudioProcessorNumpy"]
