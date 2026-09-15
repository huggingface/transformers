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
from ...processing_utils import AudioKwargs


class KyutaiSpeechToTextAudioProcessorKwargs(AudioKwargs, total=False):
    r"""
    audio_silence_prefix_seconds (`float`, *optional*, defaults to 1.0):
        Duration of silence, in seconds, prepended to the waveform.
    audio_delay_seconds (`float`, *optional*, defaults to 2.5):
        Duration of silence, in seconds, appended to the waveform (plus one further second).
    """

    audio_silence_prefix_seconds: float
    audio_delay_seconds: float


class KyutaiSpeechToTextAudioProcessorMixin:
    add_channel_dim = True
    sampling_rate = 24000

    audio_silence_prefix_seconds = 1.0
    audio_delay_seconds = 2.5
    valid_kwargs = KyutaiSpeechToTextAudioProcessorKwargs

    def _validate_preprocess_kwargs(self, *, do_extract_spectrogram, **kwargs):
        if do_extract_spectrogram:
            raise ValueError("Kyutai consumes padded waveforms, without spectrogram extraction.")
        super()._validate_preprocess_kwargs(do_extract_spectrogram=do_extract_spectrogram, **kwargs)

    def _finalize_output(self, output, *, audio_silence_prefix_seconds, audio_delay_seconds, **kwargs):
        pad_left = int(audio_silence_prefix_seconds * self.sampling_rate)
        pad_right = int((audio_delay_seconds + 1.0) * self.sampling_rate)
        if pad_left > 0 or pad_right > 0:
            for key in ("audio_values", "audio_values_mask"):
                if key in output:
                    output[key] = self._pad_axis(output[key], pad_left, pad_right, axis=-1, value=0)
        return output


class KyutaiSpeechToTextAudioProcessor(KyutaiSpeechToTextAudioProcessorMixin, TorchAudioBackend):
    pass


__all__ = ["KyutaiSpeechToTextAudioProcessor"]
