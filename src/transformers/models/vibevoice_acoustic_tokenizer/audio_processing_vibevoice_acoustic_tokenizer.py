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


class VibevoiceAcousticTokenizerAudioProcessor(TorchAudioBackend):
    sample_rate = 24000
    force_mono = True
    add_channel_dim = True

    target_dB_FS = -25
    eps = 1e-6

    def _process_audio(self, audio_el):
        audio_el = super()._process_audio(audio_el)
        rms = torch.sqrt(torch.mean(audio_el**2))
        audio_el = audio_el * (10 ** (self.target_dB_FS / 20) / (rms + self.eps))
        max_val = torch.max(torch.abs(audio_el))
        if max_val > 1.0:
            audio_el = audio_el / (max_val + self.eps)
        return audio_el


__all__ = ["VibevoiceAcousticTokenizerAudioProcessor"]
