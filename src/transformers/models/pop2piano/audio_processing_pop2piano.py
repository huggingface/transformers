# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

# NOTE: Full Pop2Piano feature extraction requires the Essentia library for
# beat detection (RhythmExtractor2013) and scipy for beat interpolation.
# This audio processor provides the basic mel spectrogram configuration but
# does not implement the complete beat-aligned segmentation pipeline.

from ...audio_processing_backends import TorchAudioBackend
from .audio_processing_numpy_pop2piano import Pop2PianoAudioProcessorNumpy


class Pop2PianoAudioProcessor(TorchAudioBackend):
    sample_rate = 22050
    force_mono = True
    # Single source of truth for the config lives on the numpy sibling (importable without torch).
    spectrogram_config = Pop2PianoAudioProcessorNumpy.spectrogram_config


__all__ = ["Pop2PianoAudioProcessor"]
