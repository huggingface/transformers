# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

from __future__ import annotations

import torch
from torch import nn

from ... import initialization as init
from ...activations import ACT2FN
from ...integrations import use_experts_implementation
from ...modeling_layers import GradientCheckpointingLayer
from ...models.deepseek_v3.modeling_deepseek_v3 import DeepseekV3MoE, DeepseekV3TopkRouter
from ...models.jamba.modeling_jamba import JambaPreTrainedModel
from ...models.mixtral.modeling_mixtral import MixtralExperts
from ...models.nemotron_h_dense.modeling_nemotron_h_dense import (
    NemotronHDenseAttention,
    NemotronHDenseAttentionDecoderLayer,
    NemotronHDenseForCausalLM,
    NemotronHDenseMamba,
    NemotronHDenseMambaDecoderLayer,
    NemotronHDenseMLP,
    NemotronHDenseModel,
    NemotronHDensePreTrainedModel,
    NemotronHDenseRMSNorm,
)
from ...utils import logging
from .configuration_nemotron_h_sparse import NemotronHSparseConfig


logger = logging.get_logger(__name__)


class NemotronHSparseMamba(NemotronHDenseMamba):
    pass


class NemotronHSparseRMSNorm(NemotronHDenseRMSNorm):
    pass


class NemotronHSparseMLP(NemotronHDenseMLP):
    pass


class NemotronHSparseAttention(NemotronHDenseAttention):
    pass


@use_experts_implementation(has_gate=False)
class NemotronHSparseExperts(MixtralExperts):
    """Non-gated MoE experts — up_proj → act → down_proj, otherwise identical to :class:`MixtralExperts`."""

    def __init__(self, config: NemotronHSparseConfig):
        nn.Module.__init__(self)
        self.num_experts = config.n_routed_experts
        self.hidden_dim = config.hidden_size
        self.intermediate_dim = config.moe_intermediate_size
        self.up_proj = nn.Parameter(torch.empty(self.num_experts, self.intermediate_dim, self.hidden_dim))
        self.down_proj = nn.Parameter(torch.empty(self.num_experts, self.hidden_dim, self.intermediate_dim))
        self.act_fn = ACT2FN[config.mlp_hidden_act]

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        final_hidden_states = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = torch.nn.functional.one_hot(top_k_index, num_classes=self.num_experts)
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        for expert_idx in expert_hit:
            expert_idx = expert_idx[0]
            if expert_idx == self.num_experts:
                continue
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]
            current_hidden_states = nn.functional.linear(current_state, self.up_proj[expert_idx])
            current_hidden_states = self.act_fn(current_hidden_states)
            current_hidden_states = nn.functional.linear(current_hidden_states, self.down_proj[expert_idx])
            current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))

        return final_hidden_states


class NemotronHSparseTopkRouter(DeepseekV3TopkRouter):
    pass


class NemotronHSparseMoE(DeepseekV3MoE):
    """Routed experts + one shared-expert MLP. Drop-in FFN for Nemotron-H Sparse layers."""

    def __init__(self, config: NemotronHSparseConfig):
        super().__init__(config)
        self.experts = NemotronHSparseExperts(config)
        self.gate = NemotronHSparseTopkRouter(config)
        self.shared_experts = NemotronHSparseMLP(
            config=config, intermediate_size=config.moe_shared_expert_intermediate_size
        )


class NemotronHSparseMambaDecoderLayer(NemotronHDenseMambaDecoderLayer):
    def __init__(self, config: NemotronHSparseConfig, layer_idx: int):
        GradientCheckpointingLayer.__init__(self)
        self.mamba = NemotronHSparseMamba(config, layer_idx=layer_idx)
        self.feed_forward = NemotronHSparseMoE(config)
        self.input_layernorm = NemotronHSparseRMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.pre_ff_layernorm = NemotronHSparseRMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)


class NemotronHSparseAttentionDecoderLayer(NemotronHDenseAttentionDecoderLayer):
    def __init__(self, config: NemotronHSparseConfig, layer_idx: int):
        GradientCheckpointingLayer.__init__(self)
        self.self_attn = NemotronHSparseAttention(config, layer_idx)
        self.feed_forward = NemotronHSparseMoE(config)
        self.input_layernorm = NemotronHSparseRMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.pre_ff_layernorm = NemotronHSparseRMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)


ALL_DECODER_LAYER_TYPES = {
    "mamba": NemotronHSparseMambaDecoderLayer,
    "attention": NemotronHSparseAttentionDecoderLayer,
}


class NemotronHSparsePreTrainedModel(NemotronHDensePreTrainedModel):
    config: NemotronHSparseConfig
    _no_split_modules = ["NemotronHSparseMambaDecoderLayer", "NemotronHSparseAttentionDecoderLayer"]
    _can_record_outputs = {
        "hidden_states": [NemotronHSparseMambaDecoderLayer, NemotronHSparseAttentionDecoderLayer],
        "attentions": NemotronHSparseAttention,
    }
    _keep_in_fp32_modules_strict = ["e_score_correction_bias"]

    @torch.no_grad()
    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, NemotronHSparseTopkRouter):
            init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            init.zeros_(module.e_score_correction_bias)
        elif isinstance(module, NemotronHSparseExperts):
            init.normal_(module.up_proj, mean=0.0, std=self.config.initializer_range)
            init.normal_(module.down_proj, mean=0.0, std=self.config.initializer_range)


class NemotronHSparseModel(NemotronHDenseModel):
    def __init__(self, config: NemotronHSparseConfig):
        JambaPreTrainedModel.__init__(self, config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [
                ALL_DECODER_LAYER_TYPES[config.layer_types[i]](config, layer_idx=i)
                for i in range(config.num_hidden_layers)
            ]
        )
        self.final_layernorm = NemotronHSparseRMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.gradient_checkpointing = False
        self.post_init()


class NemotronHSparseForCausalLM(NemotronHDenseForCausalLM):
    def __init__(self, config: NemotronHSparseConfig):
        JambaPreTrainedModel.__init__(self, config)
        self.model = NemotronHSparseModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()


__all__ = [
    "NemotronHSparsePreTrainedModel",
    "NemotronHSparseModel",
    "NemotronHSparseForCausalLM",
]
