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
from .audio_processing_numpy_gemma3n import Gemma3nAudioProcessorNumpy


class Gemma3nAudioProcessor(TorchAudioBackend):
    sampling_rate = 16000
    force_mono = True
    max_length = 480000  # 30 seconds
    truncation = True
    pad_to_multiple_of = 128

    spectrogram_config = Gemma3nAudioProcessorNumpy.spectrogram_config
    legacy_field_mapping = Gemma3nAudioProcessorNumpy.legacy_field_mapping


__all__ = ["Gemma3nAudioProcessor"]
