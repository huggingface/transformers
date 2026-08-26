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
from ...audio_processing_base import legacy_chunk_length_to_max_length
from ...audio_utils import MelScaleConfig, SpectrogramConfig, StftConfig


class WhisperAudioProcessorMixin:
    force_mono = True
    legacy_field_mapping = {
        "chunk_length": legacy_chunk_length_to_max_length,
    }
    max_length = 480000
    return_padding_mask = False
    sampling_rate = 16000
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
    truncation = True


class WhisperAudioProcessor(WhisperAudioProcessorMixin, TorchAudioBackend):
    def _apply_mel_scale(self, features, *, spectrogram_config, **kwargs):
        mel_filters = self.mel_filters.to(device=features.device)
        return torch.clamp(torch.matmul(mel_filters.T, features), min=spectrogram_config.mel_floor)


__all__ = ["WhisperAudioProcessor"]
