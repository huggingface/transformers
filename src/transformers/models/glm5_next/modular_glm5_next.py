# Copyright 2026 the HuggingFace Team. All rights reserved.
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

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...activations import ACT2FN
from ...cache_utils import Cache, DynamicCache
from ...integrations import (
    use_kernel_forward_from_hub,
    use_kernel_func_from_hub,
    use_kernelized_func,
)
from ...integrations.accelerate import force_accelerate_hooks
from ...masking_utils import create_recurrent_attention_mask
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_outputs import MoeModelOutputWithPast
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, logging
from ...utils.output_capturing import OutputRecorder
from ..deepseek_v3.modeling_deepseek_v3 import DeepseekV3MoE
from ..glm4.modeling_glm4 import apply_rotary_pos_emb
from ..glm_moe_dsa.modeling_glm_moe_dsa import GlmMoeDsaAttention, GlmMoeDsaDecoderLayer, GlmMoeDsaIndexer
from ..llama.modeling_llama import (
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
    eager_attention_forward,
)
from ..minimax_m3_vl.modeling_minimax_m3_vl import MiniMaxM3VLExperts
from ..mixtral.modeling_mixtral import MixtralForCausalLM, MixtralModel
from ..qwen2_moe.modeling_qwen2_moe import Qwen2MoeMLP
from ..qwen3_5.modeling_qwen3_5 import Qwen3_5RMSNormGated
from ..qwen3_next.modeling_qwen3_next import apply_mask_to_padding_states
from .configuration_glm5_next import Glm5NextConfig


logger = logging.get_logger(__name__)


class Glm5NextRMSNorm(LlamaRMSNorm):
    pass


@use_kernel_forward_from_hub("RMSNormGated")
class Glm5NextRMSNormGated(Qwen3_5RMSNormGated):
    def __init__(self, hidden_size, eps=1e-6, **kwargs):
        super().__init__(hidden_size, eps, kwargs)
        self.activation = "sigmoid"

    def forward(self, hidden_states, gate=None):
        input_dtype = hidden_states.dtype

        # Strict FP32 norm (do not downcast on the weights)
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        hidden_states = self.weight.to(torch.float32) * hidden_states

        # Apply gating
        hidden_states = hidden_states * ACT2FN[self.activation](gate.to(torch.float32))

        return hidden_states.to(input_dtype)


class Glm5NextMLP(Qwen2MoeMLP):
    def __init__(self, config, intermediate_size=None):
        super().__init__(config)
        self.swiglu_limit = config.swiglu_limit

    def forward(self, x):
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        # Optional clamping
        if self.swiglu_limit is not None:
            gate = gate.clamp(min=None, max=self.swiglu_limit)
            up = up.clamp(min=-self.swiglu_limit, max=self.swiglu_limit)
        return self.down_proj(self.act_fn(gate) * up)


class Glm5NextExperts(MiniMaxM3VLExperts):
    def __init__(self, config):
        super().__init__(config)
        del self.limit
        del self.swiglu_alpha
        self.intermediate_dim = config.moe_intermediate_size

    def _apply_gate(self, gate_up: torch.Tensor) -> torch.Tensor:
        gate, up = gate_up.chunk(2, dim=-1)
        # Optional clamping
        if self.swiglu_limit is not None:
            gate = gate.clamp(min=None, max=self.swiglu_limit)
            up = up.clamp(min=-self.swiglu_limit, max=self.swiglu_limit)
        # Simple swiglu instead of alpha
        return F.silu(gate) * up


class Glm5NextMoE(DeepseekV3MoE):
    pass


class Glm5NextRotaryEmbedding(LlamaRotaryEmbedding):
    pass


# We reimplement it to use it via kernels, we need BC for the other models to rely on lazy load kernels
@use_kernel_func_from_hub("causal_conv1d_update")
def causal_conv1d_update(
    hidden_states,
    conv_state,
    weight,
    bias=None,
    activation=None,
):
    _, hidden_size, seq_len = hidden_states.shape
    state_len = conv_state.shape[-1]

    hidden_states_new = torch.cat([conv_state, hidden_states], dim=-1).to(weight.dtype)
    conv_state.copy_(hidden_states_new[:, :, -state_len:])
    out = F.conv1d(hidden_states_new, weight.unsqueeze(1), bias, padding=0, groups=hidden_size)
    out = out[:, :, -seq_len:]

    if activation is not None:
        out = ACT2FN[activation](out)

    return out.to(hidden_states.dtype)


@use_kernel_func_from_hub("causal_conv1d_fn")
def causal_conv1d_fn(
    hidden_states,
    weight,
    bias=None,
    activation=None,
    **kwargs,
):
    _, hidden_size, seq_len = hidden_states.shape
    padding = weight.shape[-1] - 1

    out = F.conv1d(
        hidden_states.to(weight.dtype),
        weight=weight.unsqueeze(1),
        bias=bias,
        padding=padding,
        groups=hidden_size,
    )[:, :, :seq_len]

    if activation is not None:
        out = ACT2FN[activation](out)

    return out.to(hidden_states.dtype)


def l2norm(x: torch.FloatTensor, dim: int = -1, eps: float = 1e-6):
    """
    This function is intended to align with the l2norm implementation in the FLA library.

    # NOTE: FLA compares against `F.normalize` but does + eps instead of max(..., eps) leading to a slight differences
    """
    # main difference to qwen's gdn variation: intentionally use sqrt and / to match original triton
    inv_norm = torch.sqrt((x * x).sum(dim=dim, keepdim=True) + eps)
    return x / inv_norm


@use_kernel_func_from_hub("recurrent_kimi_delta_attention")
def recurrent_kimi_delta_attention(
    query,
    key,
    value,
    g,
    beta,
    initial_state,
    output_final_state,
    use_qk_l2norm_in_kernel=False,
    **kwargs,
):
    # calculations happen in float as states are more susceptible to rounding errors
    initial_dtype = query.dtype
    query, key, value, g, beta = [x.to(torch.float32) for x in (query, key, value, g, beta)]

    # important: FLA calculates these in fp32 so we do this after the float casts
    if use_qk_l2norm_in_kernel:
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)

    # shapes and other metadata
    batch_size, sequence_length, num_heads, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    scale = 1 / (query.shape[-1] ** 0.5)
    query = query * scale

    core_attn_out = torch.zeros(
        batch_size, sequence_length, num_heads, v_head_dim, dtype=value.dtype, device=value.device
    )
    last_recurrent_state = (
        torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim, dtype=value.dtype, device=value.device)
        if initial_state is None
        else initial_state.to(value)
    )

    # recurrent iteration
    for i in range(sequence_length):
        q_i = query[:, i]
        k_i = key[:, i]
        v_i = value[:, i]
        g_i = g[:, i][..., None].exp()
        b_i = beta[:, i][..., None]

        last_recurrent_state = last_recurrent_state * g_i
        kv_mem = (last_recurrent_state * k_i[..., None]).sum(dim=-2)
        delta = (v_i - kv_mem) * b_i

        last_recurrent_state = last_recurrent_state + k_i.unsqueeze(-1) * delta.unsqueeze(-2)
        core_attn_out[:, i] = (last_recurrent_state * q_i.unsqueeze(-1)).sum(dim=-2)

    return core_attn_out.to(initial_dtype), last_recurrent_state if output_final_state else None


@use_kernel_func_from_hub("chunk_kimi_delta_attention")
def chunk_kimi_delta_attention(
    query,
    key,
    value,
    g,
    beta,
    chunk_size=64,
    initial_state=None,
    output_final_state=False,
    use_qk_l2norm_in_kernel=False,
    **kwargs,
):
    # calculations happen in float as states are more susceptible to rounding errors
    initial_dtype = query.dtype

    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)
    ]

    # important: FLA calculates these in fp32 so we do this after the float casts
    if use_qk_l2norm_in_kernel:
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)

    # shapes and other metadata
    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    scale = 1 / (query.shape[-1] ** 0.5)
    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    total_sequence_length = sequence_length + pad_size

    # prepare all the relevant input
    query = F.pad(query, (0, 0, 0, pad_size)) * scale
    key = F.pad(key, (0, 0, 0, pad_size))
    value = F.pad(value, (0, 0, 0, pad_size))
    g = F.pad(g, (0, 0, 0, pad_size))
    beta = F.pad(beta, (0, pad_size))
    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)

    # reshape to chunks
    query, key, value, g, k_beta, v_beta = [
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1]) for x in (query, key, value, g, k_beta, v_beta)
    ]
    beta = beta.reshape(beta.shape[0], beta.shape[1], -1, chunk_size)

    # Intra chunk
    # Main difference to GDN is the per head application of `g` which was broadcasted across heads instead
    g = g.cumsum(dim=-2)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=0)
    decay_mask = (g.unsqueeze(-2) - g.unsqueeze(-3)).exp().float()
    attn = -(k_beta.unsqueeze(-2) * key.unsqueeze(-3) * decay_mask).sum(dim=-1).masked_fill(mask, 0)
    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)

    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp())

    last_recurrent_state = (
        torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim, dtype=value.dtype, device=value.device)
        if initial_state is None
        else initial_state.to(value)
    )
    core_attn_out = torch.zeros_like(value)

    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=1)
    for i in range(total_sequence_length // chunk_size):
        q_i = query[:, :, i]
        k_i = key[:, :, i]
        v_i = value[:, :, i]
        g_i = g[:, :, i]

        # Inter chunk
        attn_inter = (q_i * g_i.exp()) @ last_recurrent_state
        # Intra chunk
        attn_intra = (q_i.unsqueeze(-2) * k_i.unsqueeze(-3) * decay_mask[:, :, i]).sum(dim=-1).masked_fill(mask, 0)
        # New update rule
        v_prime = k_cumdecay[:, :, i] @ last_recurrent_state
        v_new = v_i - v_prime

        core_attn_out[:, :, i] = attn_inter + attn_intra @ v_new
        last_recurrent_state = (
            last_recurrent_state * g_i[:, :, -1].exp().unsqueeze(-1)
            + (k_i * (g_i[:, :, -1:] - g_i).exp()).transpose(-1, -2) @ v_new
        )

    if not output_final_state:
        last_recurrent_state = None

    core_attn_out = core_attn_out.reshape(core_attn_out.shape[0], core_attn_out.shape[1], -1, core_attn_out.shape[-1])
    core_attn_out = core_attn_out[:, :, :sequence_length]
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)

    return core_attn_out, last_recurrent_state


class Glm5NextForgetGate(nn.Module):
    def __init__(self, config: Glm5NextConfig):
        super().__init__()
        self.head_dim = config.linear_attn_config["head_dim"]
        self.num_heads = config.linear_attn_config["num_heads"]
        self.qkv_dim = self.head_dim * self.num_heads

        self.f_a_proj = nn.Linear(config.hidden_size, self.head_dim, bias=False)
        self.f_b_proj = nn.Linear(self.head_dim, self.qkv_dim, bias=False)
        self.dt_bias = nn.Parameter(torch.empty(self.qkv_dim, dtype=torch.float32))
        self.A_log = nn.Parameter(torch.empty(self.num_heads, dtype=torch.float32))

        self.safe_gate_lower_bound = config.linear_attn_config.get("lower_bound", None)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_shape = (*hidden_states.shape[:2], -1, self.head_dim)

        forget_gate = self.f_b_proj(self.f_a_proj(hidden_states))
        g = (forget_gate.float() + self.dt_bias.float().view(1, 1, -1)).view(hidden_shape)
        A_log = self.A_log.float().view(1, 1, self.num_heads, 1)
        decay_rate = torch.exp(A_log)

        # Safe lower bound decay
        if self.safe_gate_lower_bound is not None:
            return self.safe_gate_lower_bound * torch.sigmoid(decay_rate * g)

        # Softplus "log(1 + exp(x))" with uper bound restraint to avoid overflows
        # NOTE: Softplus for larger values (e.g. 20+), Softplus(x) == x
        g_softplus = torch.where(g > 20.0, g, torch.log(1.0 + torch.exp(g)))

        return -decay_rate * g_softplus


@use_kernelized_func(
    [chunk_kimi_delta_attention, recurrent_kimi_delta_attention, causal_conv1d_fn, causal_conv1d_update]
)
class Glm5NextLinearAttention(nn.Module):
    """Kimi-style KDA (Kimi Linear Attention) for GLM-5-Next."""

    def __init__(
        self,
        config: Glm5NextConfig,
        layer_idx: int,
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.linear_attn_config["num_heads"]
        self.head_dim = config.linear_attn_config["head_dim"]
        self.qkv_dim = self.head_dim * self.num_heads

        self.conv_kernel_size = config.linear_attn_config.get("short_conv_kernel_size", 4)
        self.layer_idx = layer_idx
        self.activation = config.hidden_act
        self.layer_norm_epsilon = config.rms_norm_eps

        self.q_proj = nn.Linear(self.hidden_size, self.qkv_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.qkv_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.qkv_dim, bias=False)

        self.conv_dim = self.qkv_dim * 3
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            bias=False,
            kernel_size=self.conv_kernel_size,
            groups=self.conv_dim,
            padding=self.conv_kernel_size - 1,
        )

        self.forget_gate = Glm5NextForgetGate(config)
        self.b_proj = nn.Linear(self.hidden_size, self.num_heads, bias=False)

        self.g_a_proj = nn.Linear(self.hidden_size, self.head_dim, bias=False)
        self.g_b_proj = nn.Linear(self.head_dim, self.qkv_dim, bias=False)
        self.o_norm = Glm5NextRMSNormGated(self.head_dim, eps=self.layer_norm_epsilon)
        self.o_proj = nn.Linear(self.qkv_dim, self.hidden_size, bias=False)

        self.layer_type = config.layer_types[layer_idx]

    @force_accelerate_hooks("conv1d")
    def forward(
        self,
        hidden_states: torch.Tensor,
        cache_params: Cache | None = None,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ):
        # Zero out padding
        hidden_states = apply_mask_to_padding_states(hidden_states, attention_mask)

        # Set up dimensions for reshapes later
        batch_size, seq_len = hidden_states.shape[:2]
        hidden_shape = (batch_size, seq_len, -1, self.head_dim)

        mixed_qkv = torch.cat(
            [
                self.q_proj(hidden_states),
                self.k_proj(hidden_states),
                self.v_proj(hidden_states),
            ],
            dim=-1,
        ).transpose(1, 2)

        # Acts for normal prefill but also for multi-token prefill continue
        use_precomputed_states = cache_params is not None and cache_params.has_previous_state(self.layer_idx)
        if use_precomputed_states:
            conv_state = cache_params.layers[self.layer_idx].conv_states
            recurrent_state = cache_params.layers[self.layer_idx].recurrent_states

        # Single token decode path
        if use_precomputed_states and seq_len == 1:
            mixed_qkv = causal_conv1d_update(
                mixed_qkv,
                conv_state,
                weight=self.conv1d.weight.squeeze(1),
                bias=self.conv1d.bias,
                activation=self.activation,
            )
        # Multi token prefill or simple "full" prefill
        else:
            # Concatenated state for prefill
            if use_precomputed_states:
                mixed_qkv = torch.cat([conv_state.to(mixed_qkv.dtype), mixed_qkv], dim=-1)

            if cache_params is not None:
                new_conv_state = F.pad(mixed_qkv, (self.conv_kernel_size - mixed_qkv.shape[-1], 0))
                cache_params.update_conv_state(new_conv_state, self.layer_idx)

            mixed_qkv = causal_conv1d_fn(
                mixed_qkv,
                weight=self.conv1d.weight.squeeze(1),
                bias=self.conv1d.bias,
                activation=self.activation,
                **kwargs,
            )

            # Cut out any tail
            if use_precomputed_states:
                mixed_qkv = mixed_qkv[:, :, -seq_len:]

        query, key, value = torch.split(
            mixed_qkv.transpose(1, 2),
            [self.qkv_dim] * 3,
            dim=-1,
        )

        query = query.view(hidden_shape)
        key = key.view(hidden_shape)
        value = value.view(hidden_shape)

        # Forget gate and input gate
        g = self.forget_gate(hidden_states)
        beta = torch.sigmoid(self.b_proj(hidden_states))

        # KDA
        if use_precomputed_states and seq_len == 1:
            core_attn_out, last_recurrent_state = recurrent_kimi_delta_attention(
                query,
                key,
                value,
                g=g,
                beta=beta,
                initial_state=recurrent_state,
                output_final_state=cache_params is not None,
                use_qk_l2norm_in_kernel=True,
                **kwargs,
            )
        else:
            core_attn_out, last_recurrent_state = chunk_kimi_delta_attention(
                query,
                key,
                value,
                g=g,
                beta=beta,
                initial_state=recurrent_state if use_precomputed_states else None,
                output_final_state=cache_params is not None,
                use_qk_l2norm_in_kernel=True,
                **kwargs,
            )

        if cache_params is not None:
            cache_params.update_recurrent_state(last_recurrent_state.to(torch.float32), self.layer_idx)

        # Final gated norm and proj
        gate = self.g_b_proj(self.g_a_proj(hidden_states)).view(hidden_shape)
        output = self.o_norm(core_attn_out, gate).reshape(batch_size, seq_len, -1)
        output = self.o_proj(output)

        return output


class Glm5NextIndexer(GlmMoeDsaIndexer):
    """
    Recompute k-pool indexer.

    The stock indexed cache gives us one tensor, so we pack everything the
    indexer needs into it:

        [indexer_key, compression_gate, valid_bit]

    The output is fixed-width topk_indices. A real key is represented by its raw
    KV-cache index. An unused, padded, or invisible slot is represented by -1.

    This module only computes routing indices. Shared sparse layers reuse the
    previous layer's topk_indices and only rebuild the backend attention mask.
    """

    def __init__(self, config, layer_idx: int):
        super().__init__(config, layer_idx)

        self.index_kpool = config.index_kpool
        self.index_kpool_always_select_tail = config.index_kpool_always_select_tail

        self.index_kpool_compress_ape = nn.Parameter(torch.zeros(self.index_kpool, self.head_dim))
        self.index_kpool_compress_gate = nn.Parameter(torch.zeros(self.head_dim, self.hidden_size))

    def get_token_visible(
        self,
        key_valid: torch.BoolTensor,
        local_valid: torch.BoolTensor,
        current_length,
    ) -> torch.BoolTensor:
        """
        Decide which cached key slots each current query is allowed to route to.

        The local mask only says whether the current query token is real or
        padding. It does not carry causal information, so we rebuild causality
        from cache positions.

        A key is visible only if:
            - the key slot contains a real token
            - the query slot contains a real token
            - the key position is not in the query's future
        """
        seq_len = local_valid.shape[-1]
        total_len = key_valid.shape[-1]
        device = key_valid.device

        key_pos = torch.arange(total_len, device=device)
        query_pos = current_length - seq_len + torch.arange(seq_len, device=device)

        causal = key_pos[None, None, :] <= query_pos[None, :, None]

        return causal & key_valid[:, None, :] & local_valid[:, :, None]

    def get_pooled_states(
        self,
        packed_states: torch.Tensor,
        key_valid: torch.BoolTensor,
    ) -> tuple[torch.Tensor, torch.LongTensor, torch.BoolTensor]:
        """
        Rebuild compressed k-pool candidates from the indexed cache.

        Each cached row stores:
            [indexer_key, compression_gate, valid_bit]

        Pooling starts at the first real token, not raw slot 0. This is the part
        that makes:

            [P, P, A, B, C, D, ...]

        behave like:

            [A, B, C, D, ...]

        for k-pool grouping.
        """
        keys, gate_scores, _ = torch.split(
            packed_states,
            [self.head_dim, self.head_dim, 1],
            dim=-1,
        )

        batch_size, total_len = keys.shape[:2]
        device = keys.device
        rate = self.index_kpool

        # For all-padding rows, use total_len so every generated pool lands out
        # of range and becomes invalid.
        first_key = torch.where(
            key_valid.any(-1),
            key_valid.long().argmax(-1),
            torch.full((batch_size,), total_len, dtype=torch.long, device=device),
        )

        n_pools = (total_len + rate - 1) // rate

        pool_offsets = torch.arange(n_pools, device=device) * rate
        slot_offsets = torch.arange(rate, device=device)

        # Raw cache indices for each pool:
        #   pool 0: first_key + [0, 1, ..., rate - 1]
        #   pool 1: first_key + rate + [0, 1, ..., rate - 1]
        #
        # Shape: [B, P, rate]
        pool_indices = first_key[:, None, None] + pool_offsets[None, :, None] + slot_offsets[None, None, :]

        slot_in_range = pool_indices < total_len
        safe_indices = pool_indices.clamp(0, total_len - 1)

        batch_idx = torch.arange(batch_size, device=device)[:, None, None]

        grouped_keys = keys[batch_idx, safe_indices]
        grouped_gate_scores = gate_scores[batch_idx, safe_indices]
        slot_valid = key_valid[batch_idx, safe_indices]

        # A compressed pool represents a full block. If any slot is padding or
        # outside the cache, the pool is incomplete and should not be selected as
        # a compressed candidate.
        slot_valid = slot_valid & slot_in_range
        pool_valid = slot_valid.all(-1)

        # Learn a weighted average over the tokens inside each complete pool.
        logits = grouped_gate_scores.float() + self.index_kpool_compress_ape.float()[None, None]
        logits = logits.masked_fill(~slot_valid[..., None], float("-inf"))

        weights = torch.nan_to_num(logits.softmax(2)).to(grouped_keys.dtype)
        pool_keys = (weights * grouped_keys).sum(2)

        pool_indices = pool_indices.masked_fill(~slot_valid, -1)

        # Static cache may contain many empty future pools. Drop columns that are
        # invalid for every batch row.
        keep = pool_valid.any(0)

        return pool_keys[:, keep], pool_indices[:, keep], pool_valid[:, keep]

    def append_visible_tail(
        self,
        topk_indices: torch.Tensor,
        token_visible: torch.BoolTensor,
        key_valid: torch.BoolTensor,
    ) -> torch.Tensor:
        """
        Append the current incomplete pool as raw token indices.

        Full pools are selected by score. The last partial pool is always
        appended so the most recent uncompressed tokens stay directly visible.

        Example with index_kpool=4:
            visible keys: A B C D E F
            full pools:   [A B C D]
            tail:                 E F
        """
        rate = self.index_kpool
        tail_width = rate - 1

        if tail_width == 0:
            return topk_indices

        batch_size, seq_len, total_len = token_visible.shape
        device = token_visible.device

        first_key = torch.where(
            key_valid.any(-1),
            key_valid.long().argmax(-1),
            torch.full((batch_size,), total_len, dtype=torch.long, device=device),
        )

        # Count how many real, causal-visible keys each query can see. The
        # remainder tells us how many tokens are in the current partial pool.
        visible_count = token_visible.long().sum(-1)
        tail_count = visible_count.remainder(rate)

        # The tail starts right after the last complete visible pool.
        tail_start = first_key[:, None] + visible_count - tail_count

        offsets = torch.arange(tail_width, device=device)
        tail_indices = tail_start[..., None] + offsets

        tail_valid = offsets[None, None, :] < tail_count[..., None]
        tail_valid = tail_valid & tail_indices.lt(total_len)

        safe_tail = tail_indices.clamp(0, total_len - 1)

        batch_idx = torch.arange(batch_size, device=device)[:, None, None]
        query_idx = torch.arange(seq_len, device=device)[None, :, None]

        # Tail tokens still pass the same per-query visibility check.
        tail_visible = token_visible[batch_idx, query_idx, safe_tail]
        tail_indices = tail_indices.masked_fill(~(tail_valid & tail_visible), -1)

        return torch.cat([topk_indices, tail_indices], dim=-1)

    @torch.no_grad()
    def forward(
        self,
        hidden_states: torch.Tensor,
        q_resid: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None,
        attention_mask: torch.BoolTensor,
        past_key_values: Cache,
    ) -> torch.LongTensor:
        """
        Compute sparse routing indices for a full DSA indexer layer.

        The caller guarantees attention_mask is local [B, S] bool. On prefill it
        marks real tokens. On decode it may just be all ones for the new token;
        old padding is remembered through the cached valid_bit.
        """
        batch_size, seq_len = hidden_states.shape[:2]
        cache_layer = past_key_values.layers[self.layer_idx]

        q = self.wq_b(q_resid).view(batch_size, seq_len, self.n_heads, self.head_dim)
        k = self.k_norm(self.wk(hidden_states)).view(batch_size, seq_len, -1, self.head_dim)

        if position_embeddings is not None:
            cos, sin = position_embeddings
            q, k = apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=2)

        k = k.squeeze(2)

        gate_scores = F.linear(hidden_states, self.index_kpool_compress_gate)
        valid_channel = attention_mask.to(k.dtype).unsqueeze(-1)

        # Store everything the indexer needs in the standard indexed cache.
        #
        # Dynamic cache:
        #   packed_states -> [B, current_len, 2D + 1]
        #
        # Static cache:
        #   packed_states -> [B, max_cache_len, 2D + 1]
        packed_states = torch.cat([k, gate_scores, valid_channel], dim=-1)
        packed_states = past_key_values.update_indexer(packed_states, self.layer_idx)

        # Mask width follows the raw attention KV cache. For static cache this is
        # max_cache_len, not just the logical sequence length.
        kv_len = cache_layer.keys.shape[-2]

        # Logical cache length after the raw KV update for this layer.
        current_length = cache_layer.get_seq_length()

        # The valid bit is cached with the keys. During decode the incoming mask
        # may only describe the new token, but old padding is still remembered.
        key_valid = packed_states[..., -1].gt(0)

        token_visible = self.get_token_visible(
            key_valid=key_valid,
            local_valid=attention_mask,
            current_length=current_length,
        )

        pool_keys, pool_indices, pool_valid = self.get_pooled_states(
            packed_states=packed_states,
            key_valid=key_valid,
        )

        # Score compressed pool candidates.
        #
        # q:         [B, S, H, D]
        # pool_keys: [B, P, D]
        # scores:    [B, S, H, P]
        scores = torch.matmul(q.float(), pool_keys.transpose(-1, -2).float().unsqueeze(1))
        scores = F.relu(scores) * self.softmax_scale

        weights = self.weights_proj(hidden_states.to(self.weights_proj.weight.dtype)).float()
        pool_scores = torch.einsum("bshp,bsh->bsp", scores, weights * (self.n_heads**-0.5))

        if pool_keys.shape[1] != 0:
            # A pool becomes usable for a query only when the query can see the
            # pool's final raw token. This keeps the whole compressed block
            # causal.
            pool_end = pool_indices[..., -1].clamp(0, kv_len - 1)

            batch_idx = torch.arange(batch_size, device=hidden_states.device)[:, None, None]
            query_idx = torch.arange(seq_len, device=hidden_states.device)[None, :, None]

            pool_visible = token_visible[batch_idx, query_idx, pool_end[:, None, :]]

            # Query-specific validity matters. Early queries may not be allowed
            # to use a pool that later queries can already see.
            candidate_valid = pool_visible & pool_valid[:, None]

            pool_scores = pool_scores.masked_fill(
                ~candidate_valid,
                torch.finfo(pool_scores.dtype).min,
            )
        else:
            candidate_valid = pool_valid[:, None].expand(batch_size, seq_len, -1)

        # Match the original budget: index_topk counts history tokens from
        # complete pools. If tail selection is enabled, the tail is appended on
        # top of that.
        group_budget = self.index_topk // self.index_kpool
        select_k = min(group_budget, pool_scores.shape[-1])

        if select_k == 0:
            topk_indices = torch.empty(
                batch_size,
                seq_len,
                0,
                dtype=torch.long,
                device=hidden_states.device,
            )
        else:
            selected = pool_scores.topk(select_k, dim=-1).indices

            batch_pool_idx = torch.arange(batch_size, device=hidden_states.device)[:, None, None]

            # Gather query-specific validity, not only global pool validity. If
            # topk picks from an all-masked row, the result still becomes -1.
            selected_valid = candidate_valid.gather(-1, selected)
            selected_indices = pool_indices[batch_pool_idx, selected]

            # Expand selected compressed pools back into raw token ids.
            #
            # selected_indices: [B, S, K, rate]
            # topk_indices:     [B, S, K * rate]
            topk_indices = selected_indices.flatten(-2)
            topk_indices = topk_indices.masked_fill(
                ~selected_valid[..., None].expand_as(selected_indices).flatten(-2),
                -1,
            )

        if self.index_kpool_always_select_tail:
            topk_indices = self.append_visible_tail(topk_indices, token_visible, key_valid)

            # With tail enabled, output can be wider than index_topk by up to
            # rate - 1. This matches the original k-pool behavior.
            output_width = self.index_topk + self.index_kpool - 1
        else:
            output_width = self.index_topk

        # Fixed-width output. Unused slots stay -1, and padded query rows are
        # forced to all -1 so the attention mask builder can treat topk_indices
        # as the source of truth.
        if topk_indices.shape[-1] < output_width:
            topk_indices = F.pad(topk_indices, (0, output_width - topk_indices.shape[-1]), value=-1)

        topk_indices = topk_indices[..., :output_width]
        topk_indices = topk_indices.masked_fill(~attention_mask[..., None], -1)

        return topk_indices.long()


class Glm5NextAttention(GlmMoeDsaAttention):
    def __init__(self, config: Glm5NextConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.q_a_layernorm = (
            Glm5NextRMSNorm(config.q_lora_rank, eps=config.rms_norm_eps) if self.q_lora_rank is not None else None
        )
        self.kv_a_layernorm = Glm5NextRMSNorm(self.kv_lora_rank, eps=config.rms_norm_eps)
        self.indexer = None if self.skip_topk else Glm5NextIndexer(config, layer_idx)
        self.next_skip_topk = (
            not self.skip_topk and config.indexer_types[min(layer_idx + 1, len(config.indexer_types) - 1)] == "shared"
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None,
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None = None,
        prev_topk_indices: torch.Tensor | None = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        batch_size, seq_length = hidden_states.shape[:-1]
        query_shape = (batch_size, seq_length, -1, self.qk_head_dim)
        key_shape = (batch_size, seq_length, -1, self.qk_nope_head_dim + self.v_head_dim)

        # LoRA based path is guaranteed based on the config validation
        q_resid = self.q_a_layernorm(self.q_a_proj(hidden_states))
        query_states = self.q_b_proj(q_resid).view(query_shape).transpose(1, 2)

        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        k_pass, k_rot = torch.split(compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        k_pass = self.kv_b_proj(self.kv_a_layernorm(k_pass)).view(key_shape).transpose(1, 2)
        key_states, value_states = torch.split(k_pass, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)

        k_rot = k_rot.view(batch_size, 1, seq_length, self.qk_rope_head_dim)
        k_rot = k_rot.expand(*k_pass.shape[:-1], -1)
        key_states = torch.cat([key_states, k_rot], dim=-1)

        # Cache update
        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

        if self.indexer is not None:
            topk_indices = self.indexer(
                hidden_states=hidden_states,
                q_resid=q_resid,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
            )
        else:
            if prev_topk_indices is None:
                raise ValueError("Shared DSA layers require top-k indices from a previous full indexer layer.")
            topk_indices = prev_topk_indices

        attention_mask = self.build_attention_mask_from_topk(
            topk_indices=topk_indices,
            query_states=query_states,
            kv_length=key_states.shape[2],
        )

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
            **kwargs,
        )

        attn_output = attn_output.reshape(batch_size, seq_length, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights, topk_indices if self.next_skip_topk else None

    def build_attention_mask_from_topk(
        self,
        topk_indices: torch.Tensor,
        query_states: torch.Tensor,
        kv_length: int,
    ) -> torch.Tensor | None:
        """
        Convert topk_indices into the mask expected by the active backend.

        Only supporting SDPA and Eager as we have a 3D dependency which cannot be mapped to FA
        without a custom kernel that can select on a per indices bases per row (query -> topk keys).
        """
        # -1 is invalid as per convention in the indexer
        # NOTE: The indexer already took care of also excluding padding tokens and causality
        topk_valid = topk_indices.ge(0) & topk_indices.lt(kv_length)

        # Clamp only so scatter has a legal index
        safe_indices = topk_indices.clamp(0, kv_length - 1)
        selected_counts = torch.zeros(
            topk_indices.shape[0],  # batch size
            topk_indices.shape[1],  # q_length
            kv_length,              # kv_length
            dtype=torch.int32,
            device=topk_indices.device,
        )
        selected_counts.scatter_add_(-1, safe_indices, topk_valid.to(torch.int32))

        # Final mask 0 == False (not visible), 1 == True (visible)
        mask = selected_counts.ne(0).unsqueeze(1)

        # SDPA
        if self.config._attn_implementation == "sdpa":
            return mask

        # Eager
        min_dtype = torch.finfo(query_states.dtype).min
        # we need 0s where the tokens should be taken into account, and -inf otherwise (mask is already of boolean type)
        mask = torch.where(mask, torch.tensor(0.0, device=query_states.device, dtype=query_states.dtype), min_dtype)
        return mask


class Glm5NextDecoderLayer(GlmMoeDsaDecoderLayer):
    def __init__(self, config: Glm5NextConfig, layer_idx: int):
        self.block_type = config.layer_types[layer_idx]

        super().__init__(config, layer_idx)
        self.self_attn = (
            Glm5NextLinearAttention(config, layer_idx)
            if self.block_type == "linear_attention"
            else Glm5NextAttention(config, layer_idx)
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = False,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        prev_topk_indices: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, None]:
        residual = hidden_states
        # Self attn
        topk_indices = None
        hidden_states = self.input_layernorm(hidden_states)
        if self.block_type == "linear_attention":
            hidden_states = self.self_attn(
                hidden_states=hidden_states,
                cache_params=past_key_values,
                attention_mask=attention_mask,
                **kwargs,
            )
        else:
            hidden_states, _, topk_indices = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                position_embeddings=position_embeddings,
                prev_topk_indices=prev_topk_indices,
                **kwargs,
            )
        hidden_states = residual + hidden_states

        residual = hidden_states
        # Feed forward
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, topk_indices


@auto_docstring
class Glm5NextPreTrainedModel(PreTrainedModel):
    config: Glm5NextConfig
    base_model_prefix = "model"
    _no_split_modules = ["Glm5NextDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]

    # needs index based kernel
    _supports_flash_attn = False
    _supports_sdpa = True
    # needs per layer creation, too expensive
    _supports_flex_attn = False
    _supports_attention_backend = True

    supports_gradient_checkpointing = True
    _can_compile_fullgraph = True

    _can_record_outputs = {
        "attentions": Glm5NextAttention,
        "hidden_states": Glm5NextDecoderLayer,
        "router_logits": OutputRecorder(Glm5NextTopkRouter, index=0),  # noqa: F821
    }
    _keep_in_fp32_modules_strict = ["e_score_correction_bias", "conv1d"]
    _keys_to_ignore_on_load_unexpected = [r"layers\.45\.", r"layers\.\d+\.shared_head\."]

    @torch.no_grad()
    def _init_weights(self, module):
        super()._init_weights(module)
        # if isinstance(module, Glm5NextTopkRouter):
        #    nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
        #    nn.init.zeros_(module.e_score_correction_bias)
        # elif isinstance(module, Glm5NextForgetGate):
        #    nn.init.normal_(module.A_log, mean=0.0, std=0.02)
        #    nn.init.zeros_(module.dt_bias)
        # elif isinstance(module, Glm5NextLinearAttention):
        #    nn.init.ones_(module.o_norm.weight)
        # elif isinstance(module, Glm5NextHyperConnection):
        #    nn.init.normal_(module.fn, mean=0.0, std=0.02)
        #    nn.init.zeros_(module.base)
        #    nn.init.ones_(module.scale)


@auto_docstring
class Glm5NextModel(MixtralModel):
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
            inputs_embeds: torch.Tensor = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if position_ids is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
            position_ids = position_ids.unsqueeze(0)

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids=position_ids)

        if not isinstance(causal_mask_mapping := attention_mask, dict):
            attention_mask = create_recurrent_attention_mask(
                config=self.config,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
            )
            # Guarantee the mask to exist for the indexer
            if attention_mask is None:
                attention_mask = torch.ones(
                    inputs_embeds.shape[0],
                    inputs_embeds.shape[1],
                    dtype=torch.bool,
                    device=inputs_embeds.device,
                )

            causal_mask_mapping = {
                # The model creates its mask based on the topk indices so we only need to know where the padding is
                "deepseek_sparse_attention": attention_mask,
                "linear_attention": attention_mask
            }

        topk_indices = None
        for i, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            hidden_states, topk_indices = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping[self.config.layer_types[i]],
                position_embeddings=position_embeddings,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                prev_topk_indices=topk_indices,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)

        return MoeModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )


@auto_docstring
class Glm5NextForCausalLM(MixtralForCausalLM):
    def forward(**super_kwargs):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Example:

        ```python
        >>> from transformers import AutoTokenizer, Glm5NextForCausalLM

        >>> model = Glm5NextForCausalLM.from_pretrained("zai-org/GLM-5-Next")
        >>> tokenizer = AutoTokenizer.from_pretrained("zai-org/GLM-5-Next")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        ```"""
        super().forward(**super_kwargs)

    @staticmethod
    def create_masks_for_generate(config, inputs_embeds, attention_mask, past_key_values, **_):
        # We only use the base 2D mask as the indexer is reliant on the padding, not the expanded masks
        # I.e. 4D masks are build afterwards after subsets have been selected in the indexer
        #
        # Linear attention can reuse the mask as is as well then making the layer type difference only
        # be necessary for the cache
        attention_mask = create_recurrent_attention_mask(
            config=config.get_text_config(),
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
        )
        # Guarantee the mask to exist for the indexer
        if attention_mask is None:
            attention_mask = torch.ones(
                inputs_embeds.shape[0],
                inputs_embeds.shape[1],
                dtype=torch.bool,
                device=inputs_embeds.device,
            )

        return {"deepseek_sparse_attention": attention_mask, "linear_attention": attention_mask}


__all__ = [
    "Glm5NextPreTrainedModel",
    "Glm5NextModel",
    "Glm5NextForCausalLM",
]
