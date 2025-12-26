# coding=utf-8
# Copyright 2025 The LG AI Research and HuggingFace Inc. team. All rights reserved.
#
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
"""LG AI Research EXAONE Lab"""

from typing import Optional

import torch
import torch.nn as nn

from ...activations import ACT2FN
from ...cache_utils import Cache
from ...configuration_utils import layer_type_validation
from ...processing_utils import Unpack
from ...utils import (
    TransformersKwargs,
)
from ..deepseek_v3.modeling_deepseek_v3 import (
    DeepseekV3MoE,
    DeepseekV3TopkRouter,
)
from ..exaone4.configuration_exaone4 import Exaone4Config
from ..exaone4.modeling_exaone4 import (
    Exaone4Attention,
    Exaone4ForCausalLM,
    Exaone4ForQuestionAnswering,
    Exaone4ForSequenceClassification,
    Exaone4ForTokenClassification,
    Exaone4MLP,
    Exaone4Model,
    Exaone4PreTrainedModel,
)
from ..olmoe.modeling_olmoe import (
    OlmoeDecoderLayer,
)


class ExaoneMoEConfig(Exaone4Config):
    model_type = "exaone_moe"

    def __init__(
        self,
        vocab_size=102400,
        hidden_size=4096,
        intermediate_size=16384,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=32,
        hidden_act="silu",
        max_position_embeddings=2048,
        initializer_range=0.02,
        rms_norm_eps=1e-5,
        use_cache=True,
        bos_token_id=0,
        eos_token_id=2,
        tie_word_embeddings=False,
        rope_parameters=None,
        attention_dropout=0.0,
        sliding_window=4096,
        sliding_window_pattern=4,
        is_moe_layer=None,
        layer_types=None,
        first_last_k_dense_replace=2,
        moe_intermediate_size=1024,
        num_experts=64,
        num_experts_per_tok=8,
        num_shared_experts=1,
        norm_topk_prob=False,
        routed_scaling_factor=2.5,
        n_group=1,
        topk_group=1,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.attention_dropout = attention_dropout
        self.sliding_window = sliding_window
        self.sliding_window_pattern = sliding_window_pattern
        self.moe_intermediate_size = moe_intermediate_size
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.num_shared_experts = num_shared_experts
        self.norm_topk_prob = norm_topk_prob
        self.routed_scaling_factor = routed_scaling_factor
        self.n_group = n_group
        self.topk_group = topk_group
        self.layer_types = layer_types

        self.is_moe_layer = is_moe_layer
        if self.is_moe_layer is None:
            self.is_moe_layer = [0] + [1] * (self.num_hidden_layers - 1)

        self.layer_types = layer_types
        if self.sliding_window is None:
            sliding_window_pattern = 0
        if self.layer_types is None:
            self.layer_types = [
                "sliding_attention"
                if ((i + 1) % (sliding_window_pattern) != 0 and i < self.num_hidden_layers)
                else "full_attention"
                for i in range(self.num_hidden_layers)
            ]
        if "sliding_window" in self.layer_types:
            self.cache_implementation = "hybrid"
        layer_type_validation(self.layer_types)

        super().__init__(
            bos_token_id=bos_token_id, eos_token_id=eos_token_id, tie_word_embeddings=tie_word_embeddings, **kwargs
        )


class ExaoneMoEAttention(Exaone4Attention):
    pass


class ExaoneMoEMLP(Exaone4MLP):
    def __init__(self, config, intermediate_size=None):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = intermediate_size if intermediate_size is not None else config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


class ExaoneMoETopkRouter(DeepseekV3TopkRouter):
    def __init__(self, config):
        nn.Module.__init__()
        self.config = config
        self.num_experts = config.num_experts
        self.weight = nn.Parameter(torch.empty((self.num_experts, config.hidden_size)))


class ExaoneMoESparseMoEBlock(DeepseekV3MoE):
    def __init__(self, config):
        nn.Module.__init__()
        self.config = config
        self.gate = ExaoneMoETopkRouter(config)
        self.experts = nn.ModuleList(
            [ExaoneMoEMLP(config, intermediate_size=config.moe_intermediate_size) for _ in range(config.num_experts)]
        )
        self.shared_experts = ExaoneMoEMLP(
            config=config, intermediate_size=config.moe_intermediate_size * config.num_shared_experts
        )
        self.num_experts = config.num_experts
        self.n_group = config.n_group
        self.topk_group = config.topk_group
        self.norm_topk_prob = config.norm_topk_prob
        self.routed_scaling_factor = config.routed_scaling_factor
        self.top_k = config.num_experts_per_tok
        self.register_buffer("e_score_correction_bias", torch.zeros(self.num_experts))

    def route_tokens_to_experts(self, router_logits):
        router_logits = router_logits.sigmoid()
        router_logits_for_choice = router_logits + self.e_score_correction_bias
        group_scores = (
            router_logits_for_choice.view(-1, self.n_group, self.num_experts // self.n_group)
            .topk(2, dim=-1)[0]
            .sum(dim=-1)
        )
        group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=False)[1]
        group_mask = torch.zeros_like(group_scores)
        group_mask.scatter_(1, group_idx, 1)
        score_mask = (
            group_mask.unsqueeze(-1)
            .expand(-1, self.n_group, self.num_experts // self.n_group)
            .reshape(-1, self.num_experts)
        )
        scores_for_choice = router_logits_for_choice.masked_fill(~score_mask.bool(), float("-inf"))
        topk_indices = torch.topk(scores_for_choice, k=self.top_k, dim=-1, sorted=False)[1]
        topk_weights = router_logits.gather(1, topk_indices)
        if self.norm_topk_prob:
            denominator = topk_weights.sum(dim=-1, keepdim=True) + 1e-20
            topk_weights = topk_weights / denominator
        topk_weights = topk_weights * self.routed_scaling_factor
        return topk_indices, topk_weights

    def experts_forward(
        self, hidden_states: torch.Tensor, top_k_index: torch.Tensor, top_k_weights: torch.Tensor
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
            expert_layer = self.experts[expert_idx]
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]
            current_hidden_states = expert_layer(current_state) * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))

        return final_hidden_states

    def forward(self, hidden_states):
        residuals = hidden_states
        orig_shape = hidden_states.shape
        router_logits = self.gate(hidden_states)
        topk_indices, topk_weights = self.route_tokens_to_experts(router_logits)
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        hidden_states = self.experts_forward(hidden_states, topk_indices, topk_weights).view(*orig_shape)
        hidden_states = hidden_states + self.shared_experts(residuals)
        return hidden_states


class ExaoneMoEDecoderLayer(OlmoeDecoderLayer):
    def __init__(self, config: ExaoneMoEConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.self_attn = ExaoneMoEAttention(config=config, layer_idx=layer_idx)
        self.mlp = ExaoneMoESparseMoEBlock(config) if config.is_moe_layer[layer_idx] else ExaoneMoEMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        # Self Attention
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class ExaoneMoEPreTrainedModel(Exaone4PreTrainedModel):
    config: ExaoneMoEConfig

    _can_record_outputs = {
        "hidden_states": ExaoneMoEDecoderLayer,
        "attentions": ExaoneMoEAttention,
        "router_logits": ExaoneMoESparseMoEBlock,
    }
    _can_compile_fullgraph = False
    _keep_in_fp32_modules_strict = ["e_score_correction_bias"]


class ExaoneMoEModel(Exaone4Model, ExaoneMoEPreTrainedModel):
    pass


class ExaoneMoEForCausalLM(Exaone4ForCausalLM):
    _keys_to_ignore_on_load_unexpected = [r"mtp.*"]


class ExaoneMoEForSequenceClassification(Exaone4ForSequenceClassification):
    pass


class ExaoneMoEForTokenClassification(Exaone4ForTokenClassification):
    pass


class ExaoneMoEForQuestionAnswering(Exaone4ForQuestionAnswering):
    pass


__all__ = [
    "ExaoneMoEConfig",
    "ExaoneMoEPreTrainedModel",
    "ExaoneMoEModel",
    "ExaoneMoEForCausalLM",
    "ExaoneMoEForSequenceClassification",
    "ExaoneMoEForTokenClassification",
    "ExaoneMoEForQuestionAnswering",
]
