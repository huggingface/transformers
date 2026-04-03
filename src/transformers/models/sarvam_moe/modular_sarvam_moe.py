# Copyright 2025 Sarvam AI and the HuggingFace Inc. team. All rights reserved.
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
"""PyTorch SarvamMoe model."""

import torch
import torch.nn.functional as F
from torch import nn

from ... import initialization as init
from ...modeling_utils import PreTrainedModel
from ...utils import logging
from ...utils.output_capturing import OutputRecorder
from ..llama.modeling_llama import (
    LlamaDecoderLayer,
    LlamaForQuestionAnswering,
    LlamaForSequenceClassification,
    LlamaForTokenClassification,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
)
from ..mixtral.modeling_mixtral import (
    MixtralForCausalLM,
    MixtralModel,
    MixtralPreTrainedModel,
)
from ..qwen2_moe.modeling_qwen2_moe import Qwen2MoeExperts, Qwen2MoeMLP
from ..qwen3.modeling_qwen3 import Qwen3Attention
from .configuration_sarvam_moe import SarvamMoeConfig


logger = logging.get_logger(__name__)


class SarvamMoeRMSNorm(LlamaRMSNorm):
    pass


class SarvamMoeRotaryEmbedding(LlamaRotaryEmbedding):
    pass


class SarvamMoeMLP(Qwen2MoeMLP):
    pass


class SarvamMoeAttention(Qwen3Attention):
    def __init__(self, config: SarvamMoeConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        del self.layer_type
        self.sliding_window = None


class SarvamMoeExperts(Qwen2MoeExperts):
    pass


class SarvamMoeTopkRouter(nn.Module):
    """Router with sigmoid scoring, expert bias correction, and group-limited top-k selection."""

    def __init__(self, config: SarvamMoeConfig):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.num_experts
        self.n_group = config.n_group
        self.topk_group = config.topk_group
        self.norm_topk_prob = config.norm_topk_prob
        self.routed_scaling_factor = config.routed_scaling_factor
        self.hidden_dim = config.hidden_size
        self.weight = nn.Parameter(torch.empty(self.num_experts, self.hidden_dim))
        self.register_buffer("e_score_correction_bias", torch.zeros(self.num_experts))

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden_states = hidden_states.view(-1, self.hidden_dim)
        router_logits = F.linear(hidden_states.to(torch.float32), self.weight.to(torch.float32))

        scores = router_logits.sigmoid()
        scores_for_choice = scores + self.e_score_correction_bias

        # Group-limited top-k selection
        group_scores = (
            scores_for_choice.view(-1, self.n_group, self.num_experts // self.n_group).topk(2, dim=-1)[0].sum(dim=-1)
        )
        group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=False)[1]
        group_mask = torch.zeros_like(group_scores)
        group_mask.scatter_(1, group_idx, 1)
        score_mask = (
            group_mask.unsqueeze(-1)
            .expand(-1, self.n_group, self.num_experts // self.n_group)
            .reshape(-1, self.num_experts)
        )
        scores_for_choice = scores_for_choice.masked_fill(~score_mask.bool(), 0.0)
        topk_indices = torch.topk(scores_for_choice, k=self.top_k, dim=-1, sorted=False)[1]
        topk_weights = scores.gather(1, topk_indices)

        if self.norm_topk_prob:
            denominator = topk_weights.sum(dim=-1, keepdim=True) + 1e-20
            topk_weights = topk_weights / denominator

        topk_weights = topk_weights * self.routed_scaling_factor
        topk_weights = topk_weights.to(hidden_states.dtype)

        return router_logits, topk_weights, topk_indices


class SarvamMoeSparseMoeBlock(nn.Module):
    """Mixture of Experts block with shared experts."""

    def __init__(self, config: SarvamMoeConfig):
        super().__init__()
        self.experts = SarvamMoeExperts(config)
        self.gate = SarvamMoeTopkRouter(config)
        self.shared_experts = SarvamMoeMLP(
            config, intermediate_size=config.moe_intermediate_size * config.num_shared_experts
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residuals = hidden_states
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        _, routing_weights, selected_experts = self.gate(hidden_states)
        hidden_states = self.experts(hidden_states, selected_experts, routing_weights)
        hidden_states = hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        hidden_states = hidden_states + self.shared_experts(residuals)
        return hidden_states


class SarvamMoeDecoderLayer(LlamaDecoderLayer):
    def __init__(self, config: SarvamMoeConfig, layer_idx: int):
        nn.Module.__init__(self)
        self.self_attn = SarvamMoeAttention(config=config, layer_idx=layer_idx)
        if layer_idx >= config.first_k_dense_replace:
            self.mlp = SarvamMoeSparseMoeBlock(config)
        else:
            self.mlp = SarvamMoeMLP(config)
        self.input_layernorm = SarvamMoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = SarvamMoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hidden_size = config.hidden_size


class SarvamMoePreTrainedModel(MixtralPreTrainedModel):
    _can_record_outputs = {
        "router_logits": OutputRecorder(SarvamMoeTopkRouter, index=0),
        "hidden_states": SarvamMoeDecoderLayer,
        "attentions": SarvamMoeAttention,
    }
    _keep_in_fp32_modules_strict = ["e_score_correction_bias"]

    @torch.no_grad()
    def _init_weights(self, module):
        PreTrainedModel._init_weights(self, module)
        std = self.config.initializer_range
        if isinstance(module, SarvamMoeExperts):
            init.normal_(module.gate_up_proj, mean=0.0, std=std)
            init.normal_(module.down_proj, mean=0.0, std=std)
        elif isinstance(module, SarvamMoeTopkRouter):
            init.normal_(module.weight, mean=0.0, std=std)
            init.zeros_(module.e_score_correction_bias)


class SarvamMoeModel(MixtralModel):
    pass


class SarvamMoeForCausalLM(MixtralForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.model = SarvamMoeModel(config)
        self.num_experts = config.num_experts


class SarvamMoeForSequenceClassification(LlamaForSequenceClassification):
    pass


class SarvamMoeForTokenClassification(LlamaForTokenClassification):
    pass


class SarvamMoeForQuestionAnswering(LlamaForQuestionAnswering):
    pass


__all__ = [
    "SarvamMoeForCausalLM",
    "SarvamMoeForQuestionAnswering",
    "SarvamMoeModel",
    "SarvamMoePreTrainedModel",
    "SarvamMoeForSequenceClassification",
    "SarvamMoeForTokenClassification",
]
