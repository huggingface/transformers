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
from ...cache_utils import Cache
from ...integrations import use_experts_implementation
from ...modeling_layers import GradientCheckpointingLayer
from ...models.deepseek_v3.modeling_deepseek_v3 import DeepseekV3MoE, DeepseekV3TopkRouter
from ...models.nemotron_h_dense.modeling_nemotron_h_dense import (
    NemotronHDenseAttention,
    NemotronHDenseForCausalLM,
    NemotronHDenseMamba2Mixer,
    NemotronHDenseMLP,
    NemotronHDenseModel,
    NemotronHDensePreTrainedModel,
    NemotronHDenseRMSNorm,
)
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, logging
from .configuration_nemotron_h_sparse import NemotronHSparseConfig


logger = logging.get_logger(__name__)


class NemotronHSparseMamba2Mixer(NemotronHDenseMamba2Mixer):
    pass


class NemotronHSparseRMSNorm(NemotronHDenseRMSNorm):
    pass


class NemotronHSparseMLP(NemotronHDenseMLP):
    pass


class NemotronHSparseAttention(NemotronHDenseAttention):
    pass


@use_experts_implementation(has_gate=False)
class NemotronHSparseExperts(nn.Module):
    """
    Collection of expert weights stored as 3D tensors.

    **Architecture Note**: Unlike Mixtral or DeepSeek which use gated MLPs,
    Nemotron-H Sparse uses a standard MLP architecture with only up_proj and down_proj.
    """

    def __init__(self, config):
        super().__init__()
        self.num_experts = config.n_routed_experts
        self.hidden_dim = config.hidden_size
        self.intermediate_dim = config.moe_intermediate_size

        input_dim = config.moe_latent_size if config.moe_latent_size is not None else config.hidden_size

        self.up_proj = nn.Parameter(torch.empty(self.num_experts, self.intermediate_dim, input_dim))
        self.down_proj = nn.Parameter(torch.empty(self.num_experts, input_dim, self.intermediate_dim))

        self.act_fn = ACT2FN[config.mlp_hidden_act]

    def forward(self, hidden_states: torch.Tensor, top_k_index: torch.Tensor, top_k_weights: torch.Tensor):
        final_hidden_states = torch.zeros_like(hidden_states, dtype=top_k_weights.dtype)

        with torch.no_grad():
            expert_mask = torch.nn.functional.one_hot(top_k_index, num_classes=self.num_experts)
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero().squeeze(-1)

        for expert_idx in expert_hit:
            expert_idx = expert_idx.item()
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])

            if token_idx.numel() == 0:
                continue

            current_state = hidden_states[token_idx]

            current_hidden_states = torch.nn.functional.linear(current_state, self.up_proj[expert_idx])
            current_hidden_states = self.act_fn(current_hidden_states)
            current_hidden_states = torch.nn.functional.linear(current_hidden_states, self.down_proj[expert_idx])

            current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]

            final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))

        return final_hidden_states.to(hidden_states.dtype)


class NemotronHSparseTopkRouter(DeepseekV3TopkRouter):
    pass


class NemotronHSparseMoE(DeepseekV3MoE):
    """
    Mixture-of-Experts module for Nemotron-H Sparse.

    Unique architectures:
    - Uses non-gated MLP experts (NemotronHSparseExperts) instead of gated experts
    - Adds optional latent projection for computational efficiency
    """

    def __init__(self, config, layer_idx: int | None = None):
        super().__init__(config)

        self.experts = NemotronHSparseExperts(config)
        self.gate = NemotronHSparseTopkRouter(config)

        self.shared_experts = NemotronHSparseMLP(
            config=config, intermediate_size=config.moe_shared_expert_intermediate_size
        )

        if config.moe_latent_size is not None:
            self.fc1_latent_proj = nn.Linear(config.hidden_size, config.moe_latent_size, bias=config.mlp_bias)
            self.fc2_latent_proj = nn.Linear(config.moe_latent_size, config.hidden_size, bias=config.mlp_bias)
        else:
            self.fc1_latent_proj = nn.Identity()
            self.fc2_latent_proj = nn.Identity()

    def forward(self, hidden_states):
        residuals = hidden_states
        orig_shape = hidden_states.shape
        router_logits = self.gate(hidden_states)
        topk_indices, topk_weights = self.route_tokens_to_experts(router_logits)
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])

        hidden_states = self.fc1_latent_proj(hidden_states)
        hidden_states = self.experts(hidden_states, topk_indices, topk_weights)
        hidden_states = self.fc2_latent_proj(hidden_states)

        hidden_states = hidden_states.view(*orig_shape)
        hidden_states = hidden_states + self.shared_experts(residuals)
        return hidden_states


MIXER_TYPES = {
    "mamba": NemotronHSparseMamba2Mixer,
    "attention": NemotronHSparseAttention,
    "moe": NemotronHSparseMoE,
}


class NemotronHSparseBlock(GradientCheckpointingLayer):
    """
    A single transformer block in the sparse Nemotron-H model. Each block holds one of
    Mamba, Attention, or MoE mixer, applies pre-normalization then the mixer, and adds
    a residual connection.
    """

    def __init__(self, config, layer_idx):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.norm = NemotronHSparseRMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)

        self.block_type = config.layer_types[layer_idx]

        mixer_kwargs = {"config": config, "layer_idx": layer_idx}
        self.mixer = MIXER_TYPES[self.block_type](**mixer_kwargs)

    def forward(
        self,
        hidden_states,
        past_key_values: Cache | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        use_cache: bool | None = False,
        **kwargs: Unpack[TransformersKwargs],
    ):
        residual = hidden_states
        hidden_states = self.norm(hidden_states.to(dtype=self.norm.weight.dtype))

        if self.block_type == "mamba":
            hidden_states = self.mixer(hidden_states, cache_params=past_key_values, attention_mask=attention_mask)
        elif self.block_type == "attention":
            hidden_states, _ = self.mixer(
                hidden_states=hidden_states,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                position_ids=position_ids,
                user_cache=use_cache,
                **kwargs,
            )
        else:
            hidden_states = self.mixer(hidden_states)

        hidden_states = residual + hidden_states

        return hidden_states


class NemotronHSparsePreTrainedModel(NemotronHDensePreTrainedModel):
    config: NemotronHSparseConfig
    _no_split_modules = ["NemotronHSparseBlock"]
    _can_record_outputs = {
        "hidden_states": NemotronHSparseBlock,
        "attentions": NemotronHSparseAttention,
    }
    _keep_in_fp32_modules_strict = [
        "e_score_correction_bias",
    ]

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
    def __init__(self, config):
        # NemotronHDensePreTrainedModel.__init__(self, config) ... via super()
        super().__init__(config)
        # Rebuild layers with the sparse block.
        self.layers = nn.ModuleList(
            [NemotronHSparseBlock(config, layer_idx=idx) for idx in range(config.num_hidden_layers)]
        )
        self.post_init()


class NemotronHSparseForCausalLM(NemotronHDenseForCausalLM):
    pass


__all__ = [
    "NemotronHSparsePreTrainedModel",
    "NemotronHSparseModel",
    "NemotronHSparseForCausalLM",
]
