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
from .audio_processing_numpy_cohere_asr import CohereAsrAudioProcessorMixin


class CohereAsrAudioProcessor(CohereAsrAudioProcessorMixin, TorchAudioBackend):
    extra_model_input_names = ["audio_chunk_index"]

    def _seeded_noise(self, length, seed, like):
        generator = torch.Generator(device=like.device).manual_seed(seed)
        return torch.randn(length, dtype=like.dtype, device=like.device, generator=generator)


__all__ = ["CohereAsrAudioProcessor"]
