# Copyright 2025 Mistral AI and The HuggingFace Inc. team. All rights reserved.
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
import torch.nn as nn

from ..mistral.modeling_mistral import (
    MistralAttention,
    MistralDecoderLayer,
    MistralMLP,
    MistralModel,
    MistralPreTrainedModel,
    MistralRMSNorm,
    MistralRotaryEmbedding,
)


class VoxtralTtsMLP(MistralMLP):
    pass


class VoxtralTtsAttention(MistralAttention):
    pass


class VoxtralTtsRMSNorm(MistralRMSNorm):
    pass


class VoxtralTtsDecoderLayer(MistralDecoderLayer):
    pass


class VoxtralTtsRotaryEmbedding(MistralRotaryEmbedding):
    pass


class VoxtralTtsAudioEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_codebooks = config.num_codebooks
        self.embed_audio_tokens = nn.Embedding(config.audio_vocab_size, config.hidden_size)
        offsets = self._compute_offsets(config)
        self.register_buffer("audio_tokens_offsets", offsets, persistent=False)

    @staticmethod
    def _compute_offsets(config):
        offsets = torch.zeros(config.num_codebooks, dtype=torch.long)
        if config.n_acoustic_codebook > 1:
            acoustic_stride = (
                config.audio_vocab_size - config.semantic_codebook_size - config.acoustic_codebook_size
            ) // (config.n_acoustic_codebook - 1)
        else:
            acoustic_stride = config.acoustic_codebook_size
        for i in range(config.n_acoustic_codebook):
            offsets[i + 1] = config.semantic_codebook_size + i * acoustic_stride
        return offsets

    def forward(self, input_ids):
        inputs_embeds = self.embed_audio_tokens(input_ids + self.audio_tokens_offsets)
        inputs_embeds = inputs_embeds.sum(dim=2)
        return inputs_embeds


class VoxtralTtsPreTrainedModel(MistralPreTrainedModel):
    pass


class VoxtralTtsBackboneModel(MistralModel):
    def __init__(self, config):
        super().__init__(config)
        self.embed_tokens = VoxtralTtsAudioEmbeddings(config)


__all__ = [
    "VoxtralTtsPreTrainedModel",
    "VoxtralTtsBackboneModel",
]
