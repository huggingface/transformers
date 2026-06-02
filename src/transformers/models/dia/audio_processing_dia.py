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
from ...audio_processing_base import make_legacy_audio_processor_alias


class DiaAudioProcessor(TorchAudioBackend):
    sample_rate = 44100
    force_mono = True
    add_channel_dim = True
    pad_to_multiple_of = 512


DiaFeatureExtractor = make_legacy_audio_processor_alias(DiaAudioProcessor, "DiaFeatureExtractor")


__all__ = ["DiaAudioProcessor", "DiaFeatureExtractor"]
