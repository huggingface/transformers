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

import numpy as np
import torch

from ...audio_processing_backends import NumpyAudioBackend
from .audio_processing_clap import ClapAudioProcessorMixin


class ClapAudioProcessorNumpy(ClapAudioProcessorMixin, NumpyAudioBackend):
    def _tile(self, audio, n_repeat):
        return np.tile(audio, n_repeat)

    def _bilinear_shrink(self, mel, chunk_frames):
        # legacy numpy dtype path: float64 straight through interpolate (torch sibling
        # round-trips through float32 instead)

        mel_tensor = torch.tensor(mel[None, None, :])
        mel_shrink = torch.nn.functional.interpolate(
            mel_tensor, size=[chunk_frames, 64], mode="bilinear", align_corners=False
        )
        return mel_shrink[0][0].numpy()


__all__ = ["ClapAudioProcessorNumpy"]
