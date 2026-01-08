# Copyright 2025 Arcee AI and the HuggingFace Inc. team. All rights reserved.
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
"""PyTorch AFMoE model."""

from collections.abc import Callable

import torch
from torch import nn

from ... import initialization as init
from ...cache_utils import Cache, DynamicCache
from ...generation import GenerationMixin
from ...masking_utils import create_causal_mask, create_sliding_window_causal_mask
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import MoeModelOutputWithPast
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, logging
from ...utils.generic import check_model_inputs
from ..gpt_oss.modeling_gpt_oss import GptOssRMSNorm
from ..llama.modeling_llama import (
    LlamaAttention,
    LlamaForCausalLM,
    LlamaRotaryEmbedding,
    apply_rotary_pos_emb,
    eager_attention_forward,
)
from ..qwen2_moe.modeling_qwen2_moe import Qwen2MoeMLP
from .configuration_afmoe import AfmoeConfig


logger = logging.get_logger(__name__)


class AfmoeRotaryEmbedding(LlamaRotaryEmbedding):
    pass


class AfmoeRMSNorm(GptOssRMSNorm):
    pass


class AfmoeMLP(Qwen2MoeMLP):
    pass


class AfmoeTokenChoiceRouter(nn.Module):
    """
    Token-choice top-K router for MoE routing.

    This router assigns each token to the top-K experts based on sigmoid scores, matching the released checkpoints.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.num_experts
        self.route_scale = config.route_scale
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)

    def forward(self, hidden_states: torch.Tensor, expert_bias: torch.Tensor):
        _, _, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)

        scores = torch.sigmoid(self.gate(hidden_states).to(torch.float32))

        _, selected_experts = torch.topk(scores + expert_bias, k=self.top_k, dim=1)
        top_scores = scores.gather(dim=1, index=selected_experts)
        denominator = top_scores.sum(dim=-1, keepdim=True) + 1e-20
        top_scores = top_scores / denominator
        top_scores = top_scores * self.route_scale
        return top_scores, selected_experts


class AfmoeExperts(nn.ModuleList):
    """
    Container holding the routed experts.

    This mirrors the Experts pattern used across other MoE models to ease checkpoint conversion.
    """

    def __init__(self, config: AfmoeConfig):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.num_experts
        for _ in range(self.num_experts):
            self.append(AfmoeMLP(config, intermediate_size=config.moe_intermediate_size))

    def forward(
        self, hidden_states: torch.Tensor, selected_experts: torch.Tensor, routing_weights: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: (batch, seq, hidden)
            selected_experts: (batch, seq, top_k)
            routing_weights: (batch, seq, top_k)
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape
        if seq_len == 0:
            return hidden_states.new_zeros(batch_size, 0, hidden_dim)
        hidden_states_flat = hidden_states.view(-1, hidden_dim)
        top_k = selected_experts.shape[-1]

        # Map every token routing decision to a unique position so we can process expert by expert.
        token_indices = torch.arange(
            hidden_states_flat.shape[0], device=hidden_states.device, dtype=torch.long
        ).repeat_interleave(top_k)
        expert_indices = selected_experts.reshape(-1)
        routing_weights = routing_weights.reshape(-1)

        sorting = torch.argsort(expert_indices, stable=True)
        token_indices = token_indices[sorting]
        expert_indices = expert_indices[sorting]
        routing_weights = routing_weights[sorting]

        dispatched_tokens = hidden_states_flat.index_select(0, token_indices)
        expert_outputs = torch.zeros_like(dispatched_tokens)

        unique_experts, counts = torch.unique_consecutive(expert_indices, return_counts=True)
        start = 0
        for expert_id, count in zip(unique_experts.tolist(), counts.tolist()):
            if count == 0:
                continue
            end = start + count
            expert_input = dispatched_tokens[start:end]
            expert_output = self[expert_id](expert_input)
            expert_outputs[start:end] = expert_output
            start = end

        weighted_outputs = (expert_outputs.to(torch.float32) * routing_weights.unsqueeze(-1)).to(hidden_states.dtype)
        aggregated = torch.zeros_like(hidden_states_flat)
        scatter_indices = token_indices.unsqueeze(-1).expand_as(weighted_outputs)
        aggregated.scatter_add_(0, scatter_indices, weighted_outputs)
        return aggregated.view(batch_size, seq_len, hidden_dim)


class AfmoeMoE(nn.Module):
    """
    Mixture of Experts (MoE) module for AFMoE.

    This module implements a sparse MoE layer with both shared experts (always active) and
    routed experts (activated based on token-choice routing).
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.router = AfmoeTokenChoiceRouter(config)
        self.shared_experts = AfmoeMLP(config, config.moe_intermediate_size * config.num_shared_experts)
        self.experts = AfmoeExperts(config)
        self.expert_bias = nn.Parameter(torch.zeros(config.num_experts, dtype=torch.float32), requires_grad=False)

    def forward(self, hidden_states):
        batch_size, seq_len, hidden_dim = hidden_states.shape
        hidden_states_flat = hidden_states.view(-1, hidden_dim)

        # Get routing decisions
        top_scores, selected_experts = self.router(hidden_states, self.expert_bias)
        top_scores = top_scores.view(batch_size, seq_len, self.config.num_experts_per_tok)
        selected_experts = selected_experts.view(batch_size, seq_len, self.config.num_experts_per_tok)

        # Process through shared experts
        shared_output = self.shared_experts(hidden_states_flat).view(batch_size, seq_len, hidden_dim)
        routed_output = self.experts(hidden_states, selected_experts, top_scores)
        return shared_output + routed_output


class AfmoeAttention(LlamaAttention):
    """
    Multi-headed attention module with optional sliding window and gating.

    This attention mechanism supports both full attention and sliding window attention,
    and includes Q/K normalization and gating of the output. It inherits from [`LlamaAttention`] to minimize the amount
    of custom logic we need to maintain.
    """

    def __init__(self, config: AfmoeConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        # Parent LlamaAttention already sets: layer_idx, num_heads, num_key_value_heads, num_key_value_groups, head_dim
        # We only add AFMoE-specific attributes
        self.is_local_attention = config.layer_types[layer_idx] == "sliding_attention"
        self.sliding_window = config.sliding_window if self.is_local_attention else None

        self.q_norm = AfmoeRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = AfmoeRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.gate_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_value: Cache | None = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape)
        key_states = self.k_proj(hidden_states).view(hidden_shape)
        value_states = self.v_proj(hidden_states).view(hidden_shape)
        gate_states = self.gate_proj(hidden_states)

        query_states = self.q_norm(query_states).transpose(1, 2)
        key_states = self.k_norm(key_states).transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        if self.is_local_attention:
            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            cache_kwargs = {"cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask=attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            **kwargs,
        )

        output = output.view(*input_shape, -1).contiguous()
        output = output * torch.sigmoid(gate_states)
        attn_output = self.o_proj(output)
        return attn_output, attn_weights


class AfmoeDecoderLayer(GradientCheckpointingLayer):
    """
    AFMoE decoder layer with dual normalization.

    This layer applies self-attention followed by either a dense MLP or MoE block,
    with dual normalization (pre and post) around each component.
    """

    def __init__(self, config: AfmoeConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx

        self.self_attn = AfmoeAttention(config=config, layer_idx=layer_idx)
        self.attention_type = config.layer_types[layer_idx]

        # Dual normalization for attention
        self.input_layernorm = AfmoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = AfmoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Dual normalization for FFN
        self.pre_mlp_layernorm = AfmoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_mlp_layernorm = AfmoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # MoE or dense FFN
        self.moe_enabled = layer_idx >= config.num_dense_layers
        if self.moe_enabled:
            self.mlp = AfmoeMoE(config)
        else:
            self.mlp = AfmoeMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_value: Cache | None = None,
        use_cache: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.FloatTensor:
        residual = hidden_states

        # Self Attention with dual normalization
        hidden_states = self.input_layernorm(hidden_states)
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
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        # FFN with dual normalization
        residual = hidden_states
        hidden_states = self.pre_mlp_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_mlp_layernorm(hidden_states)

        hidden_states = residual + hidden_states
        return hidden_states


class AfmoePreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config: AfmoeConfig
    base_model_prefix = "model"
    _no_split_modules = ["AfmoeDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _can_record_outputs = {
        "hidden_states": AfmoeDecoderLayer,
        "attentions": AfmoeAttention,
    }
    _keep_in_fp32_modules = [
        "input_layernorm",
        "post_attention_layernorm",
        "pre_mlp_layernorm",
        "post_mlp_layernorm",
        "q_norm",
        "k_norm",
        "norm",
        "expert_bias",
    ]
    _supports_sdpa = True
    _supports_flash_attn_2 = True
    _supports_flex_attn = True
    _supports_attention_backend = True
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        """Initialize the weights"""
        super()._init_weights(module)
        if isinstance(module, AfmoeTokenChoiceRouter):
            init.zeros_(module.gate.weight)
        elif isinstance(module, AfmoeMoE):
            init.zeros_(module.expert_bias)


@auto_docstring
class AfmoeModel(AfmoePreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`AfmoeDecoderLayer`]

    Args:
        config: AfmoeConfig
    """

    def __init__(self, config: AfmoeConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [AfmoeDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = AfmoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = AfmoeRotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        self.post_init()

    @auto_docstring
    @check_model_inputs
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        cache_position: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> MoeModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        # It may already have been prepared by e.g. `generate`
        if not isinstance(causal_mask_mapping := attention_mask, dict):
            mask_kwargs = {
                "config": self.config,
                "input_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
            }
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
                "sliding_attention": create_sliding_window_causal_mask(**mask_kwargs),
            }

        hidden_states = inputs_embeds

        # Apply muP input scaling if enabled
        if self.config.mup_enabled:
            hidden_states = hidden_states * (self.config.hidden_size**0.5)

        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for decoder_layer in self.layers:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping[decoder_layer.attention_type],
                position_ids=position_ids,
                past_key_value=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        return MoeModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
        )


class AfmoeForCausalLM(LlamaForCausalLM, AfmoePreTrainedModel, GenerationMixin):
    def __init__(self, config):
        AfmoePreTrainedModel.__init__(self, config)
        self.model = AfmoeModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()


__all__ = [
    "AfmoeForCausalLM",
    "AfmoeModel",
    "AfmoePreTrainedModel",
]
