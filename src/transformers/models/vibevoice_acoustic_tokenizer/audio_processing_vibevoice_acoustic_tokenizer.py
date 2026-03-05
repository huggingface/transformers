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
from ...audio_utils import NormalizationConfig


class VibevoiceAcousticTokenizerAudioProcessor(TorchAudioBackend):
    sample_rate = 24000
    force_mono = True
    add_channel_dim = True
    do_values_normalize = True
    normalization_config = NormalizationConfig(method="rms_normalize", normalize_before_pad=True)

    def __init__(self, target_dB_FS=-25, eps=1e-6, **kwargs):
        self.target_dB_FS = target_dB_FS
        self.eps = eps
        super().__init__(**kwargs)

    def values_normalize(self, audio, *, normalization_config):
        import torch

        if normalization_config.method == "rms_normalize":
            normalized = []
            for a in audio:
                rms = torch.sqrt(torch.mean(a**2))
                a = a * (10 ** (self.target_dB_FS / 20) / (rms + self.eps))
                max_val = torch.max(torch.abs(a))
                if max_val > 1.0:
                    a = a / (max_val + self.eps)
                normalized.append(a)
            return normalized
        return super().values_normalize(audio, normalization_config=normalization_config)


__all__ = ["VibevoiceAcousticTokenizerAudioProcessor"]
