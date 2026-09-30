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
from ...processing_utils import AudioKwargs


class PeAudioAudioProcessorKwargs(AudioKwargs, total=False):
    r"""
    reflect_pad_to_multiple_of (`int`, *optional*, defaults to 1920):
        Each waveform is reflect-padded up to a multiple of this many samples (the encoder's hop),
        before batch padding. The reflected samples count as audio in the padding mask.
    """

    reflect_pad_to_multiple_of: int


class PeAudioAudioProcessorMixin:
    sampling_rate = 48000
    # The legacy FE calls it `hop_length`, misleading name
    legacy_field_mapping = {"hop_length": "reflect_pad_to_multiple_of"}
    reflect_pad_to_multiple_of = 1920
    pad_to_multiple_of = 1920
    valid_kwargs = PeAudioAudioProcessorKwargs

    def _prepare_waveform(self, audio_el, *, reflect_pad_to_multiple_of, **kwargs):
        length = audio_el.shape[-1]
        pad = -length % reflect_pad_to_multiple_of
        if pad == 0:
            return audio_el
        edge = length - 1
        positions = (self._arange(pad, like=audio_el) + length) % max(2 * edge, 1)
        return self._concat_last([audio_el, audio_el[..., edge - abs(positions - edge)]])


class PeAudioAudioProcessor(PeAudioAudioProcessorMixin, TorchAudioBackend):
    pass


__all__ = ["PeAudioAudioProcessor"]
