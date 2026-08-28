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

import torch

from ...audio_processing_backends import TorchAudioBackend
from ...processing_utils import AudioKwargs


def _gemma4_unified_feature_size_to_samples_per_token(value, config_dict):
    # Legacy configs carry the frame size both as `feature_size` and
    config_dict.setdefault("audio_samples_per_token", value)


class Gemma4UnifiedAudioProcessorKwargs(AudioKwargs, total=False):
    r"""
    audio_samples_per_token (`int`, *optional*, defaults to 640):
        Number of waveform samples represented by a single audio token.
    """

    audio_samples_per_token: int


class Gemma4UnifiedAudioProcessorMixin:
    do_batch_spectrogram = False
    # for non-spectrogram models) and padded at the token level, matching the legacy
    do_extract_spectrogram = True
    force_mono = True
    legacy_field_mapping = {
        "feature_size": _gemma4_unified_feature_size_to_samples_per_token,
    }
    padding = "longest"
    padding_value = 0.0
    sampling_rate = 16000

    audio_samples_per_token = 640
    valid_kwargs = Gemma4UnifiedAudioProcessorKwargs

    def extract_spectrogram(self, audio, **kwargs):
        return [self._chunk_waveform(waveform) for waveform in audio]


class Gemma4UnifiedAudioProcessor(Gemma4UnifiedAudioProcessorMixin, TorchAudioBackend):
    def _chunk_waveform(self, waveform):
        """Chunk a 1-D waveform into fixed-length frames of `audio_samples_per_token`
        samples, zero-padding the tail so the last (partial) frame is kept."""
        pad_len = (-waveform.shape[-1]) % self.audio_samples_per_token
        if pad_len:
            waveform = torch.nn.functional.pad(waveform, (0, pad_len))
        num_tokens = waveform.shape[-1] // self.audio_samples_per_token
        return waveform.reshape(num_tokens, self.audio_samples_per_token).to(torch.float32)


__all__ = ["Gemma4UnifiedAudioProcessor"]
