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
from ...audio_utils import MelScaleConfig, SpectrogramConfig, StftConfig
from ...processing_utils import AudioKwargs


class WhisperAudioProcessorKwargs(AudioKwargs, total=False):
    r"""
    chunk_length (`int`, *optional*, defaults to 30):
        Length, in seconds, of the window the encoder consumes. This is model geometry, not a
        padding policy: read it for frame-rate maths (`chunk_length / max_source_positions`),
        and use `max_length` to control how much audio is actually padded or truncated. The two
        coincide for Whisper but need not in general.
    """

    chunk_length: int


class WhisperAudioProcessorMixin:
    force_mono = True
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

    chunk_length = 30
    valid_kwargs = WhisperAudioProcessorKwargs


class WhisperAudioProcessor(WhisperAudioProcessorMixin, TorchAudioBackend):
    def _apply_mel_scale(self, features, *, spectrogram_config, **kwargs):
        mel_filters = self.mel_filters.to(device=features.device)
        return torch.clamp(torch.matmul(mel_filters.T, features), min=spectrogram_config.mel_floor)


__all__ = ["WhisperAudioProcessor"]
