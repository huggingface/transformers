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

import math

import torch
import torch.nn as nn

from .configuration_voxtral_tts import VoxtralTtsFlowMatchingConfig
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


class VoxtralTtsFlowMatchingAttention(VoxtralTtsAttention):
    def __init__(self, config, layer_idx: int):
        super().__init__(config, layer_idx)
        self.is_causal = False


class VoxtralTtsFlowMatchingDecoderLayer(VoxtralTtsDecoderLayer):
    def __init__(self, config, layer_idx: int):
        super().__init__(config, layer_idx)
        self.self_attn = VoxtralTtsFlowMatchingAttention(config=config, layer_idx=layer_idx)


class VoxtralTtsFlowMatchingTransformer(nn.Module):
    def __init__(self, config: VoxtralTtsFlowMatchingConfig):
        super().__init__()
        self.config = config

        self.llm_projection = nn.Linear(config.input_dim, config.hidden_size, bias=False)
        self.time_projection = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.input_projection = nn.Linear(config.acoustic_dim, config.hidden_size, bias=False)

        self.layers = nn.ModuleList(
            [VoxtralTtsFlowMatchingDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = VoxtralTtsRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = VoxtralTtsRotaryEmbedding(config=config)

        self.semantic_codebook_output = nn.Linear(config.hidden_size, config.semantic_vocab_size, bias=False)
        self.acoustic_codebook_output = nn.Linear(config.hidden_size, config.acoustic_dim, bias=False)

    @staticmethod
    def _get_timestep_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
        half_dim = dim // 2
        exponent = -math.log(10000.0) * torch.arange(half_dim, device=timesteps.device, dtype=torch.float32) / half_dim
        emb = timesteps.float().unsqueeze(-1) * torch.exp(exponent).unsqueeze(0)
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        timesteps: torch.Tensor,
        acoustic_embeddings: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.llm_projection(hidden_states)
        t = self.time_projection(self._get_timestep_embedding(timesteps, self.config.hidden_size))
        x = self.input_projection(acoustic_embeddings)

        combined = h + t.unsqueeze(1) + x

        position_ids = torch.arange(combined.shape[1], device=combined.device).unsqueeze(0)
        position_embeddings = self.rotary_emb(combined, position_ids=position_ids)

        for layer in self.layers:
            combined = layer(
                combined,
                position_embeddings=position_embeddings,
            )

        combined = self.norm(combined)

        semantic_logits = self.semantic_codebook_output(combined)
        acoustic_output = self.acoustic_codebook_output(combined)
        return semantic_logits, acoustic_output


__all__ = [
    "VoxtralTtsPreTrainedModel",
    "VoxtralTtsBackboneModel",
]
