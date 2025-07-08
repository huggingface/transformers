# Copyright (c) 2025 Baidu, Inc. All Rights Reserved.
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

import functools
from copy import deepcopy
from dataclasses import dataclass
from functools import partial
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...activations import ACT2FN
from ...cache_utils import Cache, DynamicCache
from ...generation import GenerationMixin
from ...masking_utils import create_causal_mask
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_outputs import ModelOutput, MoeCausalLMOutputWithPast
from ...modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...processing_utils import Unpack
from ...utils import LossKwargs, auto_docstring, can_return_tuple, logging
from .configuration_ernie4_5_moe import Ernie4_5_MoEConfig


logger = logging.get_logger(__name__)


class KwargsForCausalLM(FlashAttentionKwargs, LossKwargs): ...

@dataclass
class Erine4_5_MoeModelOutputWithPast(ModelOutput):
    last_hidden_state: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Cache] = None
    hidden_states: Optional[tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[tuple[torch.FloatTensor, ...]] = None
    router_loss: Optional[torch.FloatTensor] = None
    gate_logits: Optional[tuple[torch.FloatTensor, ...]] = None


@dataclass
class Ernie4_5_MoECausalLMOutputWithPast(MoeCausalLMOutputWithPast):
    router_loss: Optional[torch.FloatTensor] = None


# copy llama
class Ernie4_5_MoERMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


# copy qwen3 moe (except bias)
class Ernie4_5_MoEMLP(nn.Module):
    def __init__(self, config, intermediate_size=None):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = intermediate_size if intermediate_size is not None else config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.use_bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.use_bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.use_bias)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


# can become majority wise llama copy (except the fp32 consistency)
class Ernie4_5_MoERopeEmbedding(nn.Module):
    def __init__(self, config: Ernie4_5_MoEConfig, device=None):
        super().__init__()
        # BC: "rope_type" was originally "type"
        if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
            self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]
        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    @torch.no_grad()
    @dynamic_rope_update  # power user: used with advanced RoPE types (e.g. dynamic rope)
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            # key difference to llama rope happens here to force an even/odd pattern instead
            freqs = (inv_freq_expanded.float() * position_ids_expanded.float()).transpose(1, 2)
            emb = torch.stack((freqs, freqs), dim=-1).reshape(*freqs.shape[:2], -1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos, sin


def rotate_half(x):
    """Rotates half (in even/odd pattern) the hidden dims of the input."""
    input_shape = x.shape[:-1]
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).reshape(*input_shape, -1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    # key difference to llama is the forward in fp32
    original_dtype = q.dtype
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q.float() * cos) + (rotate_half(q).float() * sin)
    k_embed = (k.float() * cos) + (rotate_half(k).float() * sin)
    return q_embed.to(original_dtype), k_embed.to(original_dtype)


# copy llama
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


# copy llama
def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


# copy llama (config bias diff + no dropout)
class Ernie4_5_MoEAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: Ernie4_5_MoEConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = 0.0
        self.is_causal = True

        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.use_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.use_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.use_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.use_bias
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


# copy qwen3 moe (+ shared experts)
class Ernie4_5_MoESparseMoEBlock_v2(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.num_experts = config.moe_num_experts
        self.top_k = config.moe_k

        # TODO: apparently it's saved but not used - what's the purpose of this...
        self.moe_statics = Ernie4_5_MoEStatics(config)

        # gating
        self.gate = nn.Linear(config.hidden_size, config.moe_num_experts, bias=False)
        self.experts = nn.ModuleList(
            [Ernie4_5_MoEMLP(config, config.moe_intermediate_size) for _ in range(config.moe_num_experts)]
        )

        # shared experts for all forwards
        self.shared_experts = None
        if config.moe_num_shared_experts > 0:
            self.shared_experts = Ernie4_5_MoEMLP(config, config.moe_intermediate_size * config.moe_num_shared_experts)

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        # router_logits: (batch * sequence_length, n_experts)
        # TODO: deactivate autocast?
        router_logits = self.gate(hidden_states.float())

        # temporarily forward in fp32 and then cast back to the input dtype
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        routing_weights = routing_weights / torch.clamp(
            routing_weights.sum(dim=-1, keepdim=True), min=1e-12
        )
        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
        )

        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be sollicitated
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

        # Loop over all available experts in the model and perform the computation on each expert
        expert_hitted = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
        for expert_idx in expert_hitted:
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx].squeeze(0))

            # Index the correct hidden states and compute the expert hidden state for
            # the current expert. We need to make sure to multiply the output hidden
            # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
            current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
            current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]

            # However `index_add_` only support torch tensors for indexing so we'll use
            # the `top_x` tensor here.
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))

        # Add shared experts to the result
        if self.shared_experts is not None:
            final_hidden_states += self.shared_experts(hidden_states)

        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        return final_hidden_states, None, torch.zeros([1], dtype=torch.float32, device=hidden_states.device), router_logits


class Ernie4_5_MoEStatics(nn.Module):
    """
    Stores MoE (Mixture of Experts) statistics
    and expert usage information.
    """

    def __init__(self, config):
        """
        Initialize MoE statistics tracking.

        Args:
            config: Model configuration containing MoE parameters
        """
        super().__init__()

        num_experts = config.moe_num_experts
        num_experts_groups = 1

        self.e_score_correction_bias = nn.Parameter(
            torch.zeros(num_experts_groups, num_experts, dtype=torch.float32),
            requires_grad=False
        )


def topk_gate_func(
    module: nn.Module,
    hidden_states: torch.Tensor,
):
    capacity = module.get_capacity(hidden_states.shape[0])
    with torch.autocast(device_type='cuda', dtype=torch.float32):
        logits = module.gate(hidden_states.float())
    router_loss = torch.zeros([1], dtype=torch.float32, device=hidden_states.device)
    router_loss.detach()
    return logits, capacity, router_loss


class Ernie4_5_MoESparseMoEBlock(nn.Module):
    """Mixture of Experts (MoE) variant of ERNIE's MLP layer."""

    def __init__(self,config):
        super().__init__()
        self.config = config
        self.k = config.moe_k
        #self.sinkhorn_2gate = config.sinkhorn_2gate
        #self.sinkhorn_temp = config.sinkhorn_temp

        moe_intermediate_size = config.moe_intermediate_size if config.moe_intermediate_size else config.intermediate_size
        self.gate = nn.Linear(config.hidden_size, config.moe_num_experts, bias=False, dtype=torch.float32)
        #if config.moe_gate_act == "softmax":
        #    self.gate_act = partial(F.softmax, dim=-1)
        #elif config.moe_gate_act == "sigmoid":
        #    self.gate_act = F.sigmoid
        #else:
        #    raise ValueError(f"{config.moe_gate_act} is not supported.")
        self.gate_act = partial(F.softmax, dim=-1)

        self.experts = nn.ModuleList(
            [Ernie4_5_MoEMLP(config,moe_intermediate_size) for i in range(config.moe_num_experts)]
        )

        #if config.moe_use_aux_free:
        #    self.moe_statics = Ernie4_5_MoeStatics(config)

        #self.use_correction_bias = config.moe_use_aux_free
        # TODO: https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/incubate/nn/functional/moe_gate_dispatch.py#L104
        self.moe_statics = Ernie4_5_MoEStatics(config)  # TODO: apparently it's saved but not used
        self.num_local_experts = len(self.experts)

        self.shared_experts = self._init_shared_experts()

    def _init_shared_experts(self):
        """
        Initialize the shared expert module.

        Returns:
            shared_experts: Shared expert module, returns None if no shared experts are needed.

        """
        cfg = deepcopy(self.config)
        if getattr(cfg, 'moe_num_shared_experts', 0) > 0:
            if getattr(cfg, 'moe_intermediate_size', None):
                cfg.intermediate_size = cfg.moe_intermediate_size * cfg.moe_num_shared_experts
            else:
                cfg.intermediate_size = cfg.intermediate_size * cfg.moe_num_shared_experts
            shared_experts = Ernie4_5_MoEMLP(cfg, cfg.intermediate_size)
        else:
            shared_experts = None
        return shared_experts

    def forward(
        self,
        input: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through MoE layer.

        Args:
            input (Tensor): Input tensor of shape [s, d].
            token_type_ids: Optional tensor for token types.

        Returns:
            tuple: (output, combine_weights, router_loss, gate_logits)
        """

        if input.dim() == 3:
            orig_shape = input.shape
            input = input.reshape(-1, input.shape[-1])
        else:
            orig_shape = None
        assert input.dim() == 2, f"input Tensor must have dimensions: (s)equence, (d)im, got:{input.shape}"

        assert self.gate is not None

        gate_input = input

        (
            dispatched_input,
            combine_weights,
            dispatch_mask,
            scatter_index,
            router_loss,
            gate_logits,
            gate_prob
        ) = self.gate_and_dispatch(gate_input)

        expert_out = self.forward_experts(dispatched_input)

        combined_output = self.combine_expert_output(expert_out, combine_weights, scatter_index)

        if self.shared_experts is not None:
            shared_expert_out = self.shared_experts(gate_input)
            combined_output += shared_expert_out

        if orig_shape:
            combined_output = combined_output.reshape(orig_shape[:-1] + (combined_output.shape[-1],))

        return combined_output, combine_weights, router_loss, gate_logits

    def forward_experts(self, dispatched_input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through experts sequentially.

        Args:
            dispatched_input (Tensor): Input tensor of shape [num_experts, capacity, dim].

        Returns:
            Tensor: Expert outputs of shape [num_experts, capacity, dim].
        """
        true_experts = self.experts
        dispatched_input = dispatched_input.reshape(
            1, self.num_local_experts, -1, dispatched_input.shape[-1]
        )
        expert_outputs = []
        if isinstance(self.experts, nn.ModuleList):
            chunks = dispatched_input.permute(1, 0, 2, 3).contiguous().unbind(0)
            assert len(chunks) == len(true_experts), f"{len(chunks)}, {len(true_experts)}"
            for chunk, expert in zip(chunks, true_experts):
                expert_outputs.append(expert(chunk))
        else:
            dispatched_input = dispatched_input.permute(1, 0, 2, 3).contiguous()
            orig_shape = dispatched_input.shape
            chunks = dispatched_input.reshape(orig_shape[0], -1, orig_shape[-1])
            chunks = self.experts(chunks)
            chunks = chunks.reshape(orig_shape[:-1] + (chunks.shape[-1],)).unbind(0)
            expert_outputs.extend(chunks)

        expert_output = torch.stack(expert_outputs, dim=1)
        return expert_output

    def moe_gate_dispatch(
        self,
        x: torch.Tensor,
        gate_logits: torch.Tensor,
        k: int,
        capacity: Optional[int],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor,
               torch.Tensor, torch.Tensor]:

        S, H = x.shape
        E = gate_logits.shape[1]
        device = x.device
        topk_prob, topk_idx = torch.topk(gate_logits, k, dim=-1)
        combine_weights = topk_prob
        expert_id = topk_idx
        y = x.new_zeros((E, capacity, H))
        scatter_index = x.new_full((k, S), -1, dtype=torch.int32)

        # per-expert slot counters
        slot_counter = torch.zeros(E, dtype=torch.int32, device=device)

        for tok in range(S):
            for route in range(k):
                e = expert_id[tok, route].item()
                slot = slot_counter[e].item()
                if slot >= capacity:
                    combine_weights[tok, route] = 0.0
                    continue

                # record mapping & dispatch activation
                scatter_index[route, tok] = e * capacity + slot
                y[e, slot] = x[tok]
                slot_counter[e] += 1

        expert_offset = torch.cumsum(slot_counter, 0, dtype=torch.int64)

        return y, combine_weights, scatter_index, expert_offset, expert_id

    def combine_expert_output(self, expert_output: torch.Tensor, combine_weights: torch.Tensor, scatter_index: torch.Tensor) -> torch.Tensor:
        """
        Combine expert outputs using combination weights.

        Args:
            expert_output (Tensor): Expert outputs [num_experts, capacity, dim].
            combine_weights (Tensor): Combination weights.
            scatter_index (Tensor): Scatter indices.

        Returns:
            Tensor: Combined output [seqlen, dim].
        """
        expert_output = expert_output.reshape(-1, expert_output.shape[-1])
        combined_output = self.combining(expert_output, combine_weights, scatter_index)
        return combined_output

    def combining(self, x, combine_weights, scatter_index):
        """
        Combines and aggregates input matrix using combination weights.

        Args:
            x (Tensor): Input tensor of shape [num_experts * capacity, dim]
            combine_weights (Tensor): Combination weights of shape [seq, 2]
            scatter_index (Tensor): Scatter indices of shape [seq, 2]

        Returns:
            Tensor: Combined output tensor of shape [seq, dim]
        """
        dim = x.shape[-1]

        scatter_index = scatter_index.reshape([-1])
        num_k = combine_weights.shape[-1]

        combine_weights = combine_weights.unsqueeze(1)

        x = x[scatter_index].reshape([-1, num_k, dim])

        return torch.matmul(combine_weights, x).squeeze(1)

    def gate_and_dispatch(self, input):
        """
        Calculate gate and dispatch inputs.

        Args:
            input: Input tensor of shape [seq, dim]

        Returns:
            tuple: (dispatched_input, combine_weights, dispatch_mask,
            scatter_index, router_loss, gate_logits, gate_prob)
        """
        gate_logits, capacity, router_loss = topk_gate_func(self, input)

        # capacity no use
        prob = self.gate_act(gate_logits)
        (
            dispatched_input,
            combine_weights_unnorm,
            scatter_index,
            dispatch_mask,
            _,
        ) = self.moe_gate_dispatch(input, prob,  k=self.k, capacity=capacity)
        dispatch_mask = torch.diff(F.pad(dispatch_mask, (1, 0)))

        scatter_index.detach()
        dispatch_mask.detach()

        scatter_index = scatter_index.transpose(0, 1)  # [k, s] -> [s, k]
        combine_weights = combine_weights_unnorm / torch.clamp(
            combine_weights_unnorm.sum(dim=-1, keepdim=True), min=1e-12
        )
        combine_weights = combine_weights.to(dtype=dispatched_input.dtype)

        return dispatched_input, combine_weights, dispatch_mask, scatter_index, router_loss, gate_logits, prob

    def get_capacity(self, num_tokens, cap_factor=None):
        """
        Calculate capacity based on number of tokens.

        Args:
            num_tokens: Number of input tokens
            cap_factor: Optional capacity factor override

        Returns:
            int: Calculated capacity
        """
        num_experts = self.config.moe_num_experts
        if cap_factor is not None:
            cap = cap_factor
        else:
            if self.training:
                cap = self.config.moe_capacity[0]
            elif num_tokens < num_experts:
                cap = self.config.moe_capacity[2]
            else:
                cap = self.config.moe_capacity[1]

        capacity = int(cap * num_tokens // num_experts)
        assert capacity > 0, f"requires capacity to >= 0. cap={cap}, num_tokens={num_tokens}"
        return capacity


class Ernie4_5_MoEDecoderLayer(nn.Module):
    """A single transformer decoder layer in ERNIE-MoE model.

    Contains self-attention and feed-forward components with optional MoE (Mixture of Experts)
    support, residual connections, and layer normalization.
    """

    def __init__(self, config, layer_idx):
        """Initialize the decoder layer.

        Args:
            config (ErnieMoEConfig): Model configuration.
            layer_idx (int): Index of this layer in the transformer stack
        """
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        self.config = config
        self.use_moe = True
        self.self_attn = Ernie4_5_MoEAttention(config, layer_idx)

        moe_layer_start_index = (
            min(config.moe_layer_start_index)
            if isinstance(config.moe_layer_start_index, (tuple, list))
            else config.moe_layer_start_index
        )
        moe_layer_end_index = (
            max(config.moe_layer_end_index)
            if isinstance(config.moe_layer_end_index, (tuple, list))
            else config.moe_layer_end_index
        )

        if (
            self.use_moe
            and ((layer_idx + 1) % config.moe_layer_interval == 0)
            and layer_idx >= moe_layer_start_index
            and layer_idx <= moe_layer_end_index
        ):
            #self.mlp = Ernie4_5_MoESparseMoEBlock(config)
            self.mlp = Ernie4_5_MoESparseMoEBlock_v2(config)
        else:
            self.mlp = Ernie4_5_MoEMLP(config)

        self.input_layernorm = Ernie4_5_MoERMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = Ernie4_5_MoERMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        output_router_loss: bool = True,
        output_gate_logits: bool = True,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.FloatTensor, Optional[tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """Forward pass through the decoder layer.

        Args:
            hidden_states (torch.Tensor): Input tensor [batch_size, seq_len, hidden_size]
            attention_mask (Optional[torch.Tensor]): Attention mask tensor
            position_ids (Optional[torch.Tensor]): Position indices for rotary embeddings
            past_key_value (Optional[tuple[torch.Tensor]]): Cached key/value states
            output_attentions (Optional[bool]): Whether to return attention weights
            use_cache (Optional[bool]): Whether to cache key/value states
            cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
                Indices depicting the position of the input sequence tokens in the sequence.
            position_embeddings (`tuple[torch.FloatTensor, torch.FloatTensor]`, *optional*):
                tuple containing the cosine and sine positional embeddings of shape `(batch_size, seq_len, head_dim)`,
                with `head_dim` being the embedding dimension of each attention head.
            output_router_loss (bool): Whether to return MoE router loss
            output_gate_logits (bool): Whether to return MoE gate logits

        Returns:
            Union: Various output combinations depending on arguments:
                - Base case: Hidden states tensor
                - With attention: tuple of (hidden_states, attention_weights)
                - With router loss: May include gate logits in output tuple
                - With MoE gate logits: May include gate logits in output tuple
        """
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            position_ids=position_ids,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = hidden_states + residual

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        router_loss = None
        gate_logits = None

        #if isinstance(self.mlp, Ernie4_5_MoESparseMoEBlock):
        if isinstance(self.mlp, Ernie4_5_MoESparseMoEBlock_v2):
            hidden_states, _, router_loss, gate_logits = self.mlp(hidden_states)
        else:
            hidden_states = self.mlp(hidden_states)
        hidden_states = hidden_states + residual

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if output_router_loss:
            outputs += (router_loss,)

        if output_gate_logits:
            outputs += (gate_logits,)

        return outputs


@auto_docstring
class Ernie4_5_MoEPretrainedModel(PreTrainedModel):
    """Base class for ERNIE pretrained models."""
    config_class = Ernie4_5_MoEConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Ernie4_5_MoEDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _keep_in_fp32_modules_strict = ["gate", "moe_statics"]
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_flex_attn = True
    _supports_cache_class = True
    _supports_quantized_cache = True
    _supports_static_cache = False  # MoE models don't work with torch.compile (`torch.where(condition)` not supported)


def subbatch(f, arg_idx, axis, bs, out_idx, same_arg_idx={}):
    """
    Converts a function to one that applies to subbatch of an input dimension.
    Useful for processing large tensors in smaller chunks to reduce memory usage.

    Args:
        f (Callable): Function to be subbatched.
        arg_idx ([int]): Indices of the inputs to be subbatched.
        axis ([int]): Indices of the dimensions to be subbatched for each input.
        bs (int): Subbatch size.
        out_idx (int): Dimension to concatenate outputs along.
        same_arg_idx (dict): Mapping of argument indices that share the same tensor.

    Returns:
        Callable: New function that processes inputs in subbatches.
    """

    @functools.wraps(f)
    def wrapper(*args, **kwargs):

        assert len(arg_idx) == len(axis), "Number of batching args and number of batching dims should match."

        inps = [args[i] for i in arg_idx]
        axis_width = [inp.shape[d] for inp, d in zip(inps, axis)]
        assert len(set(axis_width)) == 1, "Batch sizes should be kept equal."

        inp_axis = {idx: d for idx, d in zip(arg_idx, axis)}

        axis_width = axis_width[0]
        if axis_width < bs:
            return f(*args, **kwargs)

        outs = []
        for slice_at in range(0, axis_width, bs):
            _args = []
            for i, inp in enumerate(args):
                if i in same_arg_idx:
                    assert (
                        i > same_arg_idx[i]
                    ), f"expect i > same_arg_idx[i], but got i: {i} and same_arg_idx[i]: {same_arg_idx[i]}"
                    _args.append(_args[same_arg_idx[i]])
                elif i in arg_idx:
                    d = inp_axis[i]
                    start = slice_at
                    end = min(inp.shape[d], slice_at + bs)
                    # Build slice for all dims, only slice along axis d
                    slices = [slice(None)] * inp.ndim
                    slices[d] = slice(start, end)
                    _args.append(inp[tuple(slices)])
                else:
                    _args.append(inp)

            out = f(*_args, **kwargs)
            outs.append(out)

        return torch.cat(outs, dim=out_idx)

    return wrapper


class ErniePretrainingCriterion(nn.Module):
    """Criterion for ERNIE pretraining task."""

    def __init__(self, config, return_tuple=True):
        """Initialize the pretraining criterion.

        Args:
            config (ErnieConfig): Model configuration.
            return_tuple (bool): Whether to return loss as tuple (loss, loss_sum). Defaults to True.
        """
        super().__init__()
        self.ignored_index = getattr(config, "ignored_index", -100)
        self.config = config
        self.return_tuple = return_tuple

        self.loss_func = nn.CrossEntropyLoss(reduction="none")

    def forward(self, prediction_scores, masked_lm_labels, loss_mask, router_loss=None):
        """Compute the combined pretraining loss.

        Args:
            prediction_scores: Prediction scores tensor, [batch_size, seq_len, vocab_size]
            masked_lm_labels: Target labels tensor [batch_size, seq_len]
            loss_mask: Optional mask for valid tokens
            router_loss: Optional MoE router loss tensor

        Returns:
            Union:
                - If return_tuple=True: tuple of (combined_loss, mlm_loss_sum)
                - If return_tuple=False: Combined loss tensor
        """
        res = self.forward_impl(prediction_scores, masked_lm_labels, loss_mask)

        if self.return_tuple:
            loss, loss_sum = res
        else:
            loss, loss_sum = res, None

        if router_loss is not None and isinstance(router_loss, torch.Tensor):
            loss = loss + router_loss - router_loss.detach()

        return loss, loss_sum


    def loss_impl(self, prediction_scores: torch.Tensor, masked_lm_labels: torch.Tensor) -> torch.Tensor:
        """
        Core loss computation without reduction (but per-token).

        Args:
            prediction_scores (torch.Tensor): Logits tensor [batch_size, seq_len, vocab_size].
            masked_lm_labels (torch.Tensor): Target labels tensor [batch_size, seq_len].

        Returns:
            torch.Tensor: Unreduced loss tensor of shape [batch_size, seq_len].
                          Losses are calculated in float32.
        """
        scores_float32 = prediction_scores.to(torch.float32)
        # prediction_scores: [batch_size, seq_len, vocab_size]
        # masked_lm_labels: [batch_size, seq_len]
        # Transpose prediction_scores to [batch_size, vocab_size, seq_len]
        unreduced_loss = self.loss_func(
            scores_float32.transpose(1, 2),  # Shape: [batch_size, vocab_size, seq_len]
            masked_lm_labels.long()          # Shape: [batch_size, seq_len], ensure long type
        )
        # unreduced_loss will be of shape [batch_size, seq_len] and dtype float32
        return unreduced_loss

    def forward_impl(self, prediction_scores, masked_lm_labels, loss_mask=None):
        prediction_scores_dims = len(prediction_scores.shape)

        loss_subbatch_seqlen_config_key = "loss_subbatch_seqlen"
        default_loss_subbatch_seqlen = 32768

        current_loss_subbatch_seqlen = self.config.get(
            loss_subbatch_seqlen_config_key, default_loss_subbatch_seqlen
        )

        if prediction_scores_dims == 2 and prediction_scores.shape[0] > current_loss_subbatch_seqlen:
            sb_loss_func = subbatch(
                self.loss_impl, [0, 1], [0, 0], current_loss_subbatch_seqlen, 0
            )
            masked_lm_loss = sb_loss_func(prediction_scores, masked_lm_labels)
        elif prediction_scores_dims == 3 and prediction_scores.shape[1] > current_loss_subbatch_seqlen:
            sb_loss_func = subbatch(
                self.loss_impl, [0, 1], [1, 1], current_loss_subbatch_seqlen, 1
            )
            masked_lm_loss = sb_loss_func(prediction_scores, masked_lm_labels)
        else:
            masked_lm_loss = self.loss_impl(prediction_scores, masked_lm_labels)

        if loss_mask is None:
            loss_mask = masked_lm_labels != self.ignored_index

        loss_mask = loss_mask.reshape(-1).to(torch.float32)

        masked_lm_loss = torch.sum(masked_lm_loss.to(torch.float32).reshape(-1) * loss_mask)

        # The division will be in float32
        loss = masked_lm_loss / loss_mask.sum()

        loss_sum = masked_lm_loss.sum().detach()

        if not self.return_tuple:
            if self.training:
                return loss
            return loss_sum
        return loss, loss_sum


@auto_docstring
class Ernie4_5_MoEModel(Ernie4_5_MoEPretrainedModel):
    """The core ERNIE transformer model with MoE (Mixture of Experts) support."""

    def __init__(self, config: Ernie4_5_MoEConfig):
        """Initialize the ERNIE model architecture."""
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        self.config = config

        self.embed_tokens = nn.Embedding(
            self.vocab_size,
            self.hidden_size,
        )

        self.layers = nn.ModuleList(
            [
                Ernie4_5_MoEDecoderLayer(config, i)
                for i in range(config.num_hidden_layers)
            ]
        )
        self.norm = Ernie4_5_MoERMSNorm(config.hidden_size, config.rms_norm_eps)
        self.rotary_emb = Ernie4_5_MoERopeEmbedding(config=config)

        self.gradient_checkpointing = False

        self.post_init()

    def get_input_embeddings(self):
        """Get the input embedding layer."""
        return self.embed_tokens

    def set_input_embeddings(self, value):
        """Set new input embeddings."""
        self.embed_tokens = value

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
    ):
        """Forward pass through the ERNIE model."""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = create_causal_mask(
            config=self.config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
        )

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_router_loss = torch.tensor(0.0, device=inputs_embeds.device)# if self.config.use_moe else None
        all_gate_logits = ()

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    partial(decoder_layer.__call__, **flash_attn_kwargs),
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                    position_embeddings,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                    position_embeddings,
                    **flash_attn_kwargs,
                )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

            #if self.config.use_moe:
            if True:
                layer_outputs, gate_logits = layer_outputs[:-1], layer_outputs[-1]
                all_gate_logits = all_gate_logits + (gate_logits,)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return Erine4_5_MoeModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            router_loss=all_router_loss,
            gate_logits=all_gate_logits,
        )


@auto_docstring
class Ernie4_5_MoEForCausalLM(Ernie4_5_MoEPretrainedModel,GenerationMixin):
    """ERNIE Mixture of Experts (MoE) model for causal language modeling."""

    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config):
        """
        Initializes the ERNIE MoE model for causal language modeling.

        Args:
            config (dict): Model configuration.
        """
        super().__init__(config)
        self.config = config
        self.model = Ernie4_5_MoEModel(config)
        #self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=config.weight_share_add_bias and config.use_bias) # TODO
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=config.use_bias)
        self.loss_function = ErniePretrainingCriterion(config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        """Returns the input embeddings layer."""
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        """Sets the input embeddings layer."""
        self.ernie.embed_tokens = value

    def get_output_embeddings(self):
        """Returns the output embeddings (LM head)."""
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        """Sets the output embeddings layer."""
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        """Sets the ERNIE decoder model."""
        self.model = decoder

    def get_decoder(self):
        """Get the transformer decoder."""
        return self.model

    @can_return_tuple
    def forward(
        self,
        input_ids,
        attention_mask=None,
        position_ids=None,
        past_key_values: Optional[list[torch.FloatTensor]] = None,
        inputs_embeds=None,
        labels=None,
        loss_mask=None,
        use_cache=False,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        **kwargs: Unpack[KwargsForCausalLM],
    ):
        """
        Forward pass for causal language modeling.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        outputs = self.model(
            input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        logits = self.lm_head(hidden_states)

        loss, router_loss = None, None
        if getattr(self.config, "use_moe", False):
            router_loss = outputs.router_loss

        if labels is not None:
            loss, _ = self.loss_function(logits, labels, loss_mask, router_loss)

        return Ernie4_5_MoECausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            router_loss=router_loss,
        )


__all__ = [
    "Ernie4_5_MoEModel",
    "Ernie4_5_MoEForCausalLM",
    "Ernie4_5_MoEPretrainedModel"
]
