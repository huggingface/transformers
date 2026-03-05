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

from ...audio_processing_backends import NumpyAudioBackend
from ...audio_utils import MelScaleConfig, SpectrogramConfig, StftConfig


class Pop2PianoAudioProcessor(NumpyAudioBackend):
    sample_rate = 22050
    force_mono = True
    spectrogram_config = SpectrogramConfig(
        stft_config=StftConfig(n_fft=4096, hop_length=1024, power=2.0),
        mel_scale_config=MelScaleConfig(n_mels=512, f_min=10.0, mel_scale="htk"),
        log_mode="log10",
    )


__all__ = ["Pop2PianoAudioProcessor"]
