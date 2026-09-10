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


class PeAudioAudioProcessorMixin:
    # PE-Audio is a 48 kHz model: both the legacy `PeAudioFeatureExtractor` and every published
    # checkpoint (e.g. facebook/pe-a-frame-large) declare 48000. A 16000 default made the
    # processor silently resample 48 kHz input down and emit a third of the expected samples.
    sampling_rate = 48000
    # The legacy `PeAudioFeatureExtractor` declares feature_size=1 and always returns a channel
    # axis, so `audio_values` is (batch, 1, samples) rather than (batch, samples).
    add_channel_dim = True


class PeAudioAudioProcessor(PeAudioAudioProcessorMixin, TorchAudioBackend):
    pass


__all__ = ["PeAudioAudioProcessor"]
