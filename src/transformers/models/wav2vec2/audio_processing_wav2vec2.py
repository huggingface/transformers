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
from ...processing_utils import AudioKwargs


class Wav2Vec2AudioProcessorKwargs(AudioKwargs, total=False):
    r"""
    do_normalize (`bool`, *optional*, defaults to `True`):
        Whether to zero-mean unit-variance normalize the raw waveform.
    """

    do_normalize: bool


class Wav2Vec2AudioProcessorMixin:
    force_mono = True
    sampling_rate = 16000

    do_normalize = True
    valid_kwargs = Wav2Vec2AudioProcessorKwargs


class Wav2Vec2AudioProcessor(Wav2Vec2AudioProcessorMixin, TorchAudioBackend):
    def _process_audio(self, audio_el):
        audio_el = super()._process_audio(audio_el)

        if self.do_normalize:
            audio_el = (audio_el - audio_el.mean()) / torch.sqrt(audio_el.var(correction=0) + 1e-7)

        return audio_el


__all__ = ["Wav2Vec2AudioProcessor"]
