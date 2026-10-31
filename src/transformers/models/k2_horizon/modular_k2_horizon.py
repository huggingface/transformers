# Copyright 2025 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
# Copyright 2026 The Institute of Foundation Models and the HuggingFace Inc. team. All rights reserved.
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
"""PyTorch K2 Horizon model."""

import math
from collections.abc import Callable

import torch
from torch import nn
from torch.nn import functional as F

from ...activations import ACT2FN
from ...cache_utils import Cache, DynamicCache
from ...masking_utils import create_causal_mask, create_sliding_window_causal_mask
from ...modeling_outputs import MoeCausalLMOutputWithPast, MoeModelOutputWithPast
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring
from ...utils.generic import can_return_tuple, merge_with_config_defaults, no_inherit_decorator
from ...utils.output_capturing import OutputRecorder, capture_outputs
from ..llama.modeling_llama import (
    LlamaAttention,
    LlamaDecoderLayer,
    LlamaForCausalLM,
    LlamaMLP,
    LlamaModel,
    LlamaPreTrainedModel,
    LlamaRotaryEmbedding,
    eager_attention_forward,
    rotate_half,
)
from ..mixtral.modeling_mixtral import load_balancing_loss_func
from .configuration_k2_horizon import K2HorizonConfig


class K2HorizonRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, n_groups: int, eps: float = 1e-6):
        super().__init__()
        self.n_groups = n_groups
        self.hidden_size = hidden_size
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        hidden_states = hidden_states.reshape(*hidden_states.shape[:-1], self.n_groups, -1)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        hidden_states = hidden_states.reshape(*hidden_states.shape[:-2], -1)
        # K2 applies the learned weight before casting back to the input dtype.
        hidden_states = self.weight * hidden_states
        return hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, n_groups={self.n_groups}, eps={self.variance_epsilon}"


class K2HorizonRotaryEmbedding(LlamaRotaryEmbedding):
    @staticmethod
    def compute_default_rope_parameters(config: K2HorizonConfig, device=None, **kwargs) -> tuple[torch.Tensor, float]:
        base = config.rope_parameters["rope_theta"]
        dim = config.rope_head_dim
        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim)
        )
        return inv_freq, 1.0


def split_to_interleaved(x):
    return x.reshape(*x.shape[:-1], 2, -1).transpose(-1, -2).reshape(*x.shape[:-1], -1)


def interleaved_to_split(x):
    return x.reshape(*x.shape[:-1], -1, 2).transpose(-1, -2).reshape(*x.shape[:-1], -1)


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


class K2HorizonMLP(LlamaMLP):
    def __init__(self, config, intermediate_size=None):
        nn.Module.__init__(self)
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size if intermediate_size is None else intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]


def compute_routing_weights(router_logits, router_bias, score_func, top_k, normalize, scaling_factor):
    if score_func == "softmax":
        routing_scores = F.softmax(router_logits, dim=-1, dtype=torch.float32)
    else:
        routing_scores = torch.sigmoid(router_logits.float())
    # Unlike a linear-layer bias, K2's correction bias only changes expert selection.
    selection_scores = routing_scores if router_bias is None else routing_scores + router_bias.to(routing_scores)
    selected_experts = torch.topk(selection_scores, top_k, dim=-1).indices
    routing_weights = routing_scores.gather(-1, selected_experts)
    if normalize:
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
    return routing_weights * scaling_factor, selected_experts


class K2HorizonTopKRouter(nn.Linear):
    def __init__(self, config, num_experts, top_k, normalize):
        super().__init__(config.hidden_size, num_experts, bias=config.moe_gate_bias)
        self.top_k = top_k
        self.normalize = normalize
        self.score_func = config.router_score_func
        self.scaling_factor = config.router_scaling_factor

    def forward(self, hidden_states):
        # Keep all parameter access inside forward so device offload hooks load both weight and bias.
        router_logits = F.linear(hidden_states, self.weight)
        routing_weights, selected_experts = compute_routing_weights(
            router_logits, self.bias, self.score_func, self.top_k, self.normalize, self.scaling_factor
        )
        return router_logits, routing_weights, selected_experts


class K2HorizonSparseMoeBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob
        self.router_score_func = config.router_score_func
        self.router_scaling_factor = config.router_scaling_factor
        self.num_shared_experts = config.num_shared_experts
        self.gate = K2HorizonTopKRouter(config, self.num_experts, self.top_k, self.norm_topk_prob)
        # Keep the published checkpoint names and per-expert projection shapes.
        self.experts = nn.ModuleList(
            [K2HorizonMLP(config, config.moe_intermediate_size) for _ in range(self.num_experts)]
        )
        if self.num_shared_experts:
            self.shared_experts = K2HorizonMLP(config, config.moe_intermediate_size * config.num_shared_experts)

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        input_shape = hidden_states.shape
        flat_hidden_states = hidden_states.reshape(-1, input_shape[-1])
        router_logits, routing_weights, selected_experts = self.gate(flat_hidden_states)
        routing_weights = routing_weights.to(hidden_states.dtype)
        final_hidden_states = torch.zeros_like(flat_hidden_states)
        expert_mask = F.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)
        for expert_idx in torch.nonzero(expert_mask.sum(dim=(-1, -2)), as_tuple=False).flatten():
            topk_positions, token_positions = torch.where(expert_mask[expert_idx])
            expert_states = self.experts[expert_idx](flat_hidden_states[token_positions])
            expert_states = expert_states * routing_weights[token_positions, topk_positions, None]
            final_hidden_states.index_add_(0, token_positions, expert_states.to(hidden_states.dtype))
        final_hidden_states = final_hidden_states.reshape(input_shape)
        if self.num_shared_experts:
            final_hidden_states = final_hidden_states + self.shared_experts(hidden_states)
        return final_hidden_states, router_logits


@no_inherit_decorator
class K2HorizonAttention(LlamaAttention):
    def __init__(self, config: K2HorizonConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.rope_head_dim = config.rope_head_dim
        self.gate_func = config.attention_gate_func
        if self.gate_func is not None:
            self.gate_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=False)
        if config.query_key_norm:
            self.q_norm = K2HorizonRMSNorm(
                config.num_attention_heads * self.head_dim, config.num_attention_heads, config.rms_norm_eps
            )
            self.k_norm = K2HorizonRMSNorm(
                config.num_key_value_heads * self.head_dim, config.num_key_value_heads, config.rms_norm_eps
            )
        self.sliding_window = config.sliding_window

    def compute_value_states(self, hidden_states):
        return self.v_proj(hidden_states)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        if self.config.query_key_norm:
            query_states = self.q_norm(query_states)
            key_states = self.k_norm(key_states)
        query_states = query_states.view(hidden_shape).transpose(1, 2)
        key_states = key_states.view(hidden_shape).transpose(1, 2)
        value_states = self.compute_value_states(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        if self.rope_head_dim == self.head_dim:
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        else:
            # Partial RoPE rotates complete pairs in the interleaved representation.
            query_states, query_pass = split_to_interleaved(query_states).split(
                [self.rope_head_dim, self.head_dim - self.rope_head_dim], dim=-1
            )
            key_states, key_pass = split_to_interleaved(key_states).split(
                [self.rope_head_dim, self.head_dim - self.rope_head_dim], dim=-1
            )
            query_states, key_states = apply_rotary_pos_emb(
                interleaved_to_split(query_states), interleaved_to_split(key_states), cos, sin
            )
            query_states = interleaved_to_split(torch.cat([split_to_interleaved(query_states), query_pass], dim=-1))
            key_states = interleaved_to_split(torch.cat([split_to_interleaved(key_states), key_pass], dim=-1))

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )
        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            **kwargs,
        )
        if self.gate_func is not None:
            gate = self.gate_proj(hidden_states).view(*input_shape, -1, self.head_dim)
            gate = F.silu(gate) if self.gate_func == "silu" else F.softplus(gate, beta=math.log(2))
            attn_output = attn_output * gate
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        return self.o_proj(attn_output), attn_weights


class K2HorizonMoVAAttention(K2HorizonAttention):
    def __init__(self, config: K2HorizonConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        del self.v_proj
        self.num_experts_per_tok = config.mova_num_experts_per_tok
        self.router_score_func = config.router_score_func
        self.router_scaling_factor = config.router_scaling_factor
        self.v_router = K2HorizonTopKRouter(
            config, config.mova_num_experts, self.num_experts_per_tok, self.num_experts_per_tok > 1
        )
        self.v_experts = nn.ModuleList(
            [
                nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=False)
                for _ in range(config.mova_num_experts)
            ]
        )

    def compute_value_states(self, hidden_states):
        flat_hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1])
        _, routing_weights, selected_experts = self.v_router(flat_hidden_states)
        value_states = hidden_states.new_zeros((flat_hidden_states.shape[0], self.v_experts[0].out_features))
        expert_mask = F.one_hot(selected_experts, num_classes=len(self.v_experts)).permute(2, 1, 0)
        for expert_idx in torch.nonzero(expert_mask.sum(dim=(-1, -2)), as_tuple=False).flatten():
            topk_positions, token_positions = torch.where(expert_mask[expert_idx])
            expert_states = F.silu(self.v_experts[expert_idx](flat_hidden_states[token_positions]))
            expert_states = expert_states * routing_weights[token_positions, topk_positions, None].to(
                expert_states.dtype
            )
            value_states.index_add_(0, token_positions, expert_states.to(hidden_states.dtype))
        return value_states.reshape(*hidden_states.shape[:-1], -1)


class K2HorizonDecoderLayer(LlamaDecoderLayer):
    def __init__(self, config: K2HorizonConfig, layer_idx: int):
        nn.Module.__init__(self)
        self.hidden_size = config.hidden_size
        is_sparse_layer = (
            config.num_experts > 0
            and layer_idx not in config.mlp_only_layers
            and (layer_idx + 1) % config.decoder_sparse_step == 0
        )
        attention_class = K2HorizonMoVAAttention if is_sparse_layer and config.mova_num_experts else K2HorizonAttention
        self.self_attn = attention_class(config, layer_idx)
        self.mlp = K2HorizonSparseMoeBlock(config) if is_sparse_layer else K2HorizonMLP(config)
        self.input_layernorm = K2HorizonRMSNorm(config.hidden_size, config.layernorm_num_groups, config.rms_norm_eps)
        self.post_attention_layernorm = K2HorizonRMSNorm(
            config.hidden_size, config.layernorm_num_groups, config.rms_norm_eps
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = False,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            position_ids=position_ids,
            use_cache=use_cache,
            **kwargs,
        )
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.mlp(self.post_attention_layernorm(hidden_states))
        if isinstance(hidden_states, tuple):
            hidden_states, _ = hidden_states
        return residual + hidden_states


class K2HorizonPreTrainedModel(LlamaPreTrainedModel):
    _can_record_outputs = {
        "router_logits": OutputRecorder(K2HorizonSparseMoeBlock, index=1),
        "hidden_states": K2HorizonDecoderLayer,
        "attentions": K2HorizonAttention,
    }


class K2HorizonModel(LlamaModel):
    def __init__(self, config: K2HorizonConfig):
        super().__init__(config)
        self.padding_idx = config.padding_idx
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.norm = K2HorizonRMSNorm(config.hidden_size, config.layernorm_num_groups, config.rms_norm_eps)
        # Routed experts use data-dependent indexing; the dense architecture remains compilable.
        self._can_compile_fullgraph = config.num_experts == 0

    @merge_with_config_defaults
    @capture_outputs
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> MoeModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)
        if position_ids is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
            position_ids = position_ids.unsqueeze(0)

        mask_function = create_causal_mask if self.config.sliding_window is None else create_sliding_window_causal_mask
        causal_mask = mask_function(
            config=self.config,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )
        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids=position_ids)
        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_embeddings=position_embeddings,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                **kwargs,
            )
        hidden_states = self.norm(hidden_states)
        return MoeModelOutputWithPast(last_hidden_state=hidden_states, past_key_values=past_key_values)


class K2HorizonForCausalLM(LlamaForCausalLM):
    def __init__(self, config: K2HorizonConfig):
        super().__init__(config)
        self._can_compile_fullgraph = self.model._can_compile_fullgraph

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        output_router_logits: bool | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> MoeCausalLMOutputWithPast:
        r"""
        Example:

        ```python
        >>> from transformers import AutoTokenizer, K2HorizonForCausalLM

        >>> model = K2HorizonForCausalLM.from_pretrained("IFM/K2-Horizon-0.9B")
        >>> tokenizer = AutoTokenizer.from_pretrained("IFM/K2-Horizon-0.9B")
        >>> inputs = tokenizer("The capital of France is", return_tensors="pt")
        >>> generated_ids = model.generate(**inputs, max_new_tokens=32, do_sample=False)
        >>> response = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        ```
        """
        output_router_logits = (
            self.config.output_router_logits if output_router_logits is None else output_router_logits
        )
        outputs: MoeModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_router_logits=output_router_logits,
            **kwargs,
        )
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(outputs.last_hidden_state[:, slice_indices, :])
        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)
        aux_loss = None
        if output_router_logits and outputs.router_logits:
            aux_loss = load_balancing_loss_func(
                outputs.router_logits,
                self.config.num_experts,
                self.config.num_experts_per_tok,
                attention_mask,
            )
            if loss is not None:
                loss = loss + self.config.router_aux_loss_coef * aux_loss.to(loss.device)
        return MoeCausalLMOutputWithPast(
            loss=loss,
            aux_loss=aux_loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            router_logits=outputs.router_logits,
        )


__all__ = ["K2HorizonModel", "K2HorizonForCausalLM", "K2HorizonPreTrainedModel"]
