# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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
from collections.abc import Callable
from dataclasses import field

import torch
import torch.nn.functional as F
from huggingface_hub.dataclasses import strict
from torch import nn

from ...activations import ACT2FN
from ...cache_utils import Cache
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring
from ...utils.generic import maybe_autocast
from ..llama.configuration_llama import LlamaConfig
from ..llama.modeling_llama import (
    LlamaAttention,
    LlamaDecoderLayer,
    LlamaForCausalLM,
    LlamaModel,
    LlamaPreTrainedModel,
    apply_rotary_pos_emb,
    eager_attention_forward,
)


@auto_docstring(checkpoint="openbmb/MiniCPM5-15B-A2.5B")
@strict
class MiniCPM5Config(LlamaConfig):
    r"""
    first_k_dense_replace (`int`, *optional*, defaults to 1):
        Number of initial decoder layers with a dense feed-forward network.
    n_routed_experts (`int`, *optional*, defaults to 160):
        Number of routed experts in each sparse layer.
    n_shared_experts (`int`, *optional*, defaults to 1):
        Number of expert widths in the always-active shared MLP.
    routed_scaling_factor (`float`, *optional*, defaults to 3.66):
        Scale applied to the selected expert weights after normalization.
    n_group (`int`, *optional*, defaults to 1):
        Number of expert groups used for top-k selection.
    topk_group (`int`, *optional*, defaults to 1):
        Number of expert groups retained during routing.
    use_gated_attention (`bool`, *optional*, defaults to True):
        Project one gate per query head dimension and apply it to the attention output.
    """

    model_type = "minicpm5_moe"
    default_theta = 5000000.0
    vocab_size: int = 130560
    hidden_size: int = 2048
    intermediate_size: int = 8192
    num_hidden_layers: int = 28
    num_attention_heads: int = 32
    num_key_value_heads: int = 2
    head_dim: int = 128
    max_position_embeddings: int = 524288
    pad_token_id: int = 1
    bos_token_id: int = 0
    eos_token_id: int | list[int] | None = field(default_factory=lambda: [1, 130073])
    first_k_dense_replace: int = 1
    n_routed_experts: int = 160
    n_shared_experts: int = 1
    routed_scaling_factor: float = 3.66
    num_experts_per_tok: int = 16
    moe_intermediate_size: int = 512
    n_group: int = 1
    topk_group: int = 1
    norm_topk_prob: bool = True
    use_gated_attention: bool = True


class MiniCPM5MLP(nn.Module):
    def __init__(self, config, intermediate_size=None):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size if intermediate_size is None else intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


class MiniCPM5TopkRouter(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_tok
        self.n_routed_experts = config.n_routed_experts
        self.routed_scaling_factor = config.routed_scaling_factor
        self.n_group = config.n_group
        self.topk_group = config.topk_group
        self.norm_topk_prob = config.norm_topk_prob

        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, config.hidden_size)))
        self.register_buffer("e_score_correction_bias", torch.zeros(self.n_routed_experts, dtype=torch.float32))

    @torch.no_grad()
    def get_topk_indices(self, scores):
        scores_for_choice = scores.view(-1, self.n_routed_experts) + self.e_score_correction_bias.unsqueeze(0)
        group_scores = (
            scores_for_choice.view(-1, self.n_group, self.n_routed_experts // self.n_group)
            .topk(2, dim=-1)[0]
            .sum(dim=-1)
        )
        group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=False)[1]
        group_mask = torch.zeros_like(group_scores)
        group_mask.scatter_(1, group_idx, 1)
        score_mask = (
            group_mask.unsqueeze(-1)
            .expand(-1, self.n_group, self.n_routed_experts // self.n_group)
            .reshape(-1, self.n_routed_experts)
        )
        scores_for_choice = scores_for_choice.masked_fill(~score_mask.bool(), float("-inf"))
        topk_indices = torch.topk(scores_for_choice, k=self.top_k, dim=-1, sorted=False)[1]
        return topk_indices

    def forward(self, hidden_states):
        hidden_states = hidden_states.view(-1, self.config.hidden_size)
        device_type = hidden_states.device.type if isinstance(hidden_states.device.type, str) else "cpu"
        with maybe_autocast(device_type=device_type, enabled=False):
            router_logits = F.linear(hidden_states.float(), self.weight.float())
        scores = router_logits.sigmoid()
        topk_indices = self.get_topk_indices(scores)
        topk_weights = scores.gather(1, topk_indices)
        if self.norm_topk_prob:
            denominator = topk_weights.sum(dim=-1, keepdim=True) + 1e-20
            topk_weights /= denominator
        topk_weights = topk_weights * self.routed_scaling_factor
        return router_logits, topk_weights, topk_indices


class MiniCPM5Experts(nn.ModuleList):
    def forward(self, hidden_states: torch.Tensor, topk_indices: torch.Tensor, topk_weights: torch.Tensor):
        final_hidden_states = torch.zeros_like(hidden_states, dtype=topk_weights.dtype)
        # Group the selected token/slot pairs without expanding them across all experts.
        expert_ids, positions = topk_indices.flatten().sort(stable=True)
        boundaries = torch.searchsorted(expert_ids, torch.arange(1, len(self), device=expert_ids.device)).tolist()
        flat_weights = topk_weights.flatten()
        start = 0
        for expert, end in zip(self, [*boundaries, positions.numel()]):
            expert_positions = positions[start:end]
            start = end
            # Keep empty experts in the autograd graph when gradients are enabled.
            if expert_positions.numel() == 0 and not torch.is_grad_enabled():
                continue
            token_indices = expert_positions // topk_indices.shape[-1]
            expert_weights = flat_weights[expert_positions]
            expert_input = hidden_states[token_indices]
            expert_output = expert(expert_input)
            weighted_output = expert_output * expert_weights.unsqueeze(-1)
            final_hidden_states.index_add_(0, token_indices, weighted_output)

        return final_hidden_states.to(hidden_states.dtype)


class MiniCPM5MoE(nn.Module):
    """
    A mixed expert module containing shared experts.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.experts = MiniCPM5Experts(
            [
                MiniCPM5MLP(config, intermediate_size=config.moe_intermediate_size)
                for _ in range(config.n_routed_experts)
            ]
        )
        self.gate = MiniCPM5TopkRouter(config)
        self.shared_experts = MiniCPM5MLP(
            config=config, intermediate_size=config.moe_intermediate_size * config.n_shared_experts
        )

    def forward(self, hidden_states):
        residuals = hidden_states
        orig_shape = hidden_states.shape
        _, topk_weights, topk_indices = self.gate(hidden_states)
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        hidden_states = self.experts(hidden_states, topk_indices, topk_weights).reshape(orig_shape)
        hidden_states = hidden_states + self.shared_experts(residuals)
        return hidden_states


class MiniCPM5Attention(LlamaAttention):
    def __init__(self, config: MiniCPM5Config, layer_idx: int):
        super().__init__(config, layer_idx)
        self.use_gated_attention = config.use_gated_attention
        self.q_proj = nn.Linear(
            config.hidden_size,
            config.num_attention_heads * self.head_dim * (2 if config.use_gated_attention else 1),
            bias=config.attention_bias,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        cache: Cache | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ):
        # Keep the three projections adjacent so backends can fuse them without repacking the checkpoint.
        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)
        input_shape = hidden_states.shape[:-1]
        head_shape = (*input_shape, -1, self.head_dim)
        if self.use_gated_attention:
            query, gate = query.reshape(*input_shape, -1, 2 * self.head_dim).chunk(2, dim=-1)
        query_states = query.reshape(head_shape).transpose(1, 2)
        key_states = key.reshape(head_shape).transpose(1, 2)
        value_states = value.reshape(head_shape).transpose(1, 2)
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        if cache is not None:
            key_states, value_states = cache.update(key_states, value_states, self.layer_idx, kwargs=kwargs)
        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )
        output, weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=self.attention_dropout if self.training else 0.0,
            scaling=self.scaling,
            **kwargs,
        )
        output = output.reshape(*input_shape, -1).contiguous()
        if self.use_gated_attention:
            output = output * gate.reshape(*input_shape, -1).sigmoid()
        return self.o_proj(output), weights


class MiniCPM5DecoderLayer(LlamaDecoderLayer):
    def __init__(self, config: MiniCPM5Config, layer_idx: int):
        super().__init__(config, layer_idx)
        self.mlp = MiniCPM5MLP(config) if layer_idx < config.first_k_dense_replace else MiniCPM5MoE(config)


class MiniCPM5PreTrainedModel(LlamaPreTrainedModel):
    _keep_in_fp32_modules_strict = ["e_score_correction_bias"]
    _can_compile_fullgraph = False

    def _init_weights(self, module):
        PreTrainedModel._init_weights(self, module)
        if isinstance(module, MiniCPM5TopkRouter):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            nn.init.zeros_(module.e_score_correction_bias)


class MiniCPM5Model(LlamaModel):
    pass


class MiniCPM5MoEForCausalLM(LlamaForCausalLM):
    def forward(self, **super_kwargs):
        r"""
        Example:

        ```python
        >>> from transformers import AutoModelForCausalLM, AutoTokenizer

        >>> checkpoint = "openbmb/MiniCPM5-15B-A2.5B"
        >>> tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        >>> model = AutoModelForCausalLM.from_pretrained(checkpoint, dtype="auto", device_map="auto")
        >>> inputs = tokenizer("What is 7 + 8?", return_tensors="pt").to(model.device)
        >>> outputs = model(**inputs, logits_to_keep=1)
        ```
        """
        return super().forward(**super_kwargs)


__all__ = [
    "MiniCPM5Config",
    "MiniCPM5PreTrainedModel",
    "MiniCPM5Model",
    "MiniCPM5MoEForCausalLM",
]
