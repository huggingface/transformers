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
from ...cache_utils import Cache, DynamicCache, DynamicLayer, LinearAttentionLayer
from ...integrations import lazy_load_kernel
from ...masking_utils import create_causal_mask, create_recurrent_attention_mask
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_outputs import MoeModelOutputWithPast
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, logging
from ...utils.generic import is_flash_attention_requested
from ...utils.import_utils import resolve_internal_import
from ...utils.output_capturing import OutputRecorder
from ..deepseek_v3.modeling_deepseek_v3 import DeepseekV3MoE
from ..deepseek_v4.modeling_deepseek_v4 import DeepseekV4HyperConnection, DeepseekV4Model
from ..glm4.modeling_glm4 import apply_rotary_pos_emb
from ..glm_moe_dsa.modeling_glm_moe_dsa import GlmMoeDsaAttention, GlmMoeDsaDecoderLayer
from ..llama.modeling_llama import (
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
    eager_attention_forward,
)
from ..minimax_m3_vl.modeling_minimax_m3_vl import MiniMaxM3VLExperts
from ..mixtral.modeling_mixtral import MixtralForCausalLM
from ..qwen2_moe.modeling_qwen2_moe import Qwen2MoeMLP
from ..qwen3_5.modeling_qwen3_5 import Qwen3_5RMSNormGated
from ..qwen3_next.modeling_qwen3_next import (
    apply_mask_to_padding_states,
    torch_causal_conv1d_fn,
    torch_causal_conv1d_update,
)
from .configuration_glm5_next import Glm5NextConfig


logger = logging.get_logger(__name__)


# =============================================================================
# RMSNorm(Gated), MLP, MoE, RoPE
# =============================================================================


class Glm5NextRMSNorm(LlamaRMSNorm):
    pass


# TODO: Wrap with FLA layer kernel
class Glm5NextRMSNormGated(Qwen3_5RMSNormGated):
    def __init__(self, hidden_size, eps=1e-6, **kwargs):
        super().__init__(hidden_size, eps, kwargs)
        self.activation = "sigmoid"  # TODO: config value?

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


# =============================================================================
# MHC (Manifold-Constrained Hyper-Connection) helpers
# =============================================================================


class Glm5NextHyperConnection(DeepseekV4HyperConnection):
    pass


class Glm5NextHyperHead(nn.Module):
    """Final GLM-5-Next HC-stream collapse. Unlike DeepSeek-V4, this is an unweighted mean."""

    def forward(self, hidden_streams: torch.Tensor) -> torch.Tensor:
        return hidden_streams.mean(dim=2)


# =============================================================================
# KDA Linear Attention
# =============================================================================


def l2norm(x: torch.FloatTensor, dim: int = -1, eps: float = 1e-6):
    """
    This function is intended to align with the l2norm implementation in the FLA library.

    # NOTE: FLA compares against `F.normalize` but does + eps instead of max(..., eps) leading to a slight differences
    """
    # main difference to qwen's gdn variation: intentionally use sqrt and / to match original triton
    inv_norm = torch.sqrt((x * x).sum(dim=dim, keepdim=True) + eps)
    return x / inv_norm


def torch_recurrent_kimi_delta_attention(
    query, key, value, g, beta, initial_state, output_final_state, use_qk_l2norm_in_kernel=False
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


def torch_chunk_kimi_delta_attention(
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
            dtype=torch.float32,  # TODO: check if this was intended
        )

        self.forget_gate = Glm5NextForgetGate(config)
        self.b_proj = nn.Linear(self.hidden_size, self.num_heads, bias=False)

        self.g_a_proj = nn.Linear(self.hidden_size, self.head_dim, bias=False)
        self.g_b_proj = nn.Linear(self.head_dim, self.qkv_dim, bias=False)
        self.o_norm = Glm5NextRMSNormGated(self.head_dim, eps=self.layer_norm_epsilon)
        self.o_proj = nn.Linear(self.qkv_dim, self.hidden_size, bias=False)

        self.layer_type = config.layer_types[layer_idx]

        # Check kernels or torch availability
        global causal_conv1d_update, causal_conv1d_fn, chunk_kda, recurrent_kda
        fla = lazy_load_kernel("fla")
        causal_conv1d_update, causal_conv1d_fn, chunk_kda, recurrent_kda = (
            resolve_internal_import(fla, chained_path=path)
            for path in [
                "modules.convolution.causal_conv1d_update",
                "modules.convolution.causal_conv1d",
                "ops.kda.chunk.chunk_kda",
                "ops.kda.fused_recurrent.fused_recurrent_kda",
            ]
        )

        # TODO: fixup causal conv -> FLA always returns a tuple and needs transposed layout (T, D) not (D, T) as currently
        # self.causal_conv1d_fn = causal_conv1d_fn or torch_causal_conv1d_fn
        # self.causal_conv1d_update = causal_conv1d_update or torch_causal_conv1d_update
        # TODO: kernels drift a bit which is expected, keeping torch for easier comparisons
        # self.chunk_kimi_delta_attention = chunk_kda or torch_chunk_kimi_delta_attention
        # self.recurrent_kimi_delta_attention = recurrent_kda or torch_recurrent_kimi_delta_attention
        self.causal_conv1d_fn = torch_causal_conv1d_fn
        self.causal_conv1d_update = torch_causal_conv1d_update
        self.chunk_kimi_delta_attention = torch_chunk_kimi_delta_attention
        self.recurrent_kimi_delta_attention = torch_recurrent_kimi_delta_attention

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
            mixed_qkv = self.causal_conv1d_update(
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

            mixed_qkv = self.causal_conv1d_fn(
                mixed_qkv,
                weight=self.conv1d.weight.squeeze(1),
                bias=self.conv1d.bias,
                activation=self.activation,
                seq_idx=kwargs.get("seq_idx"),  # TODO: cu seqlens under FLA
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
            core_attn_out, last_recurrent_state = self.recurrent_kimi_delta_attention(
                query,
                key,
                value,
                g=g,
                beta=beta,
                initial_state=recurrent_state,
                output_final_state=cache_params is not None,
                use_qk_l2norm_in_kernel=True,
            )
        else:
            core_attn_out, last_recurrent_state = self.chunk_kimi_delta_attention(
                query,
                key,
                value,
                g=g,
                beta=beta,
                initial_state=recurrent_state if use_precomputed_states else None,
                output_final_state=cache_params is not None,
                use_qk_l2norm_in_kernel=True,
                # The chunked FLA kernel takes a single `cu_seqlens` arg; for packed self-attention this matches q-side lengths.
                cu_seqlens=kwargs.get("cu_seq_lens_q"),
            )

        if cache_params is not None:
            cache_params.update_recurrent_state(last_recurrent_state.to(torch.float32), self.layer_idx)

        # Final gated norm and proj
        gate = self.g_b_proj(self.g_a_proj(hidden_states)).view(hidden_shape)
        output = self.o_norm(core_attn_out, gate).reshape(batch_size, seq_len, -1)
        output = self.o_proj(output)

        return output


# =============================================================================
# Cache
# =============================================================================


class Glm5NextAttentionCacheLayer(DynamicLayer):
    def reorder_cache(self, beam_idx: torch.LongTensor) -> None:
        super().reorder_cache(beam_idx)
        for attribute_name in ("indexer_keys", "indexer_gate_scores"):
            state = getattr(self, attribute_name, None)
            if state is not None:
                setattr(
                    self,
                    attribute_name,
                    state.index_select(0, beam_idx.to(state.device)),
                )

    def crop(self, max_length: int) -> None:
        if max_length < 0:
            max_length = self.get_seq_length() - abs(max_length)
        super().crop(max_length)
        for attribute_name in ("indexer_keys", "indexer_gate_scores"):
            state = getattr(self, attribute_name, None)
            if state is not None:
                setattr(self, attribute_name, state[:, :max_length])


class Glm5NextDynamicCache(DynamicCache):
    def __init__(self, *args, config: Glm5NextConfig | None = None, **kwargs):
        super().__init__(*args, config=config, **kwargs)
        if config is not None:
            for layer_idx, layer_type in enumerate(config.layer_types):
                if layer_type == "linear_attention":
                    self.layers[layer_idx] = LinearAttentionLayer(config)
                else:
                    self.layers[layer_idx] = Glm5NextAttentionCacheLayer(config)


# =============================================================================
# MLA (Multi-head Latent Attention) with optional DSA indexer scaffold
# =============================================================================


class Glm5NextIndexer(nn.Module):
    """GLM-5-Next DSA indexer (`wq_b`, `wk`, `weights_proj`, `k_norm`)."""

    def __init__(self, config: Glm5NextConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.hidden_size: int = config.hidden_size
        self.n_heads: int = config.index_n_heads
        self.head_dim: int = config.index_head_dim
        self.index_topk: int = config.index_topk
        self.q_lora_rank: int = config.q_lora_rank
        self.index_kpool: int = config.index_kpool
        self.index_kpool_compress: bool = config.index_kpool_compress
        self.kpool_enabled = self.index_kpool > 1 and self.index_kpool_compress
        self.index_kpool_always_select_tail: bool = config.index_kpool_always_select_tail
        self.qk_rope_head_dim: int = config.qk_rope_head_dim

        if self.kpool_enabled:
            self.index_kpool_compress_ape = nn.Parameter(torch.zeros(self.index_kpool, self.head_dim))
            self.index_kpool_compress_gate = nn.Parameter(torch.zeros(self.head_dim, self.hidden_size))
        else:
            self.register_parameter("index_kpool_compress_ape", None)
            self.register_parameter("index_kpool_compress_gate", None)

        self.wq_b = nn.Linear(self.q_lora_rank, self.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(self.hidden_size, self.head_dim, bias=False)
        self.k_norm = nn.LayerNorm(self.head_dim, eps=1e-6) if config.index_dsa_use_layernorm else nn.Identity()
        self.weights_proj = nn.Linear(self.hidden_size, self.n_heads, bias=False)
        self.softmax_scale = self.head_dim**-0.5

    def _compress_kpool_keys(self, keys: torch.Tensor, gate_scores: torch.Tensor) -> torch.Tensor:
        num_complete_pools = keys.shape[1] // self.index_kpool
        if num_complete_pools == 0:
            return keys.new_empty(keys.shape[0], 0, self.head_dim)

        pooled_length = num_complete_pools * self.index_kpool
        grouped_keys = keys[:, :pooled_length].reshape(
            keys.shape[0], num_complete_pools, self.index_kpool, self.head_dim
        )
        grouped_gate_scores = gate_scores[:, :pooled_length].reshape_as(grouped_keys)
        compression_logits = grouped_gate_scores.float() + self.index_kpool_compress_ape.float()[None, None]
        probabilities = compression_logits.softmax(dim=2).to(grouped_keys.dtype)
        return (probabilities * grouped_keys).sum(dim=2)

    def _update_key_cache(self, k: torch.Tensor, past_key_values: Cache | None) -> torch.Tensor:
        if past_key_values is None:
            return k

        cache_layer = past_key_values.layers[self.layer_idx]
        cached_keys = getattr(cache_layer, "indexer_keys", None)
        if k.shape[1] > 1 or cached_keys is None or cached_keys.shape[0] != k.shape[0]:
            k_cached = k
        else:
            k_cached = torch.cat([cached_keys.to(k.device), k], dim=1)
        cache_layer.indexer_keys = k_cached
        return k_cached

    def _update_kpool_cache(
        self,
        keys: torch.Tensor,
        gate_scores: torch.Tensor,
        past_key_values: Cache | None,
        total_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if past_key_values is None:
            return keys, gate_scores

        cache_layer = past_key_values.layers[self.layer_idx]
        cached_keys = getattr(cache_layer, "indexer_keys", None)
        cached_gate_scores = getattr(cache_layer, "indexer_gate_scores", None)
        prefix_len = total_len - keys.shape[1]
        can_append = (
            cached_keys is not None
            and cached_gate_scores is not None
            and cached_keys.shape[0] == keys.shape[0]
            and cached_keys.shape[1] == prefix_len
            and cached_gate_scores.shape[:2] == cached_keys.shape[:2]
        )
        if can_append:
            assert cached_keys is not None and cached_gate_scores is not None
            keys = torch.cat((cached_keys.to(keys.device), keys), dim=1)
            gate_scores = torch.cat(
                (cached_gate_scores.to(gate_scores.device), gate_scores),
                dim=1,
            )

        cache_layer.indexer_keys = keys
        cache_layer.indexer_gate_scores = gate_scores
        return keys, gate_scores

    def _indexer_score_mask(self, attention_mask: torch.Tensor | None, total_len: int) -> torch.Tensor | None:
        if attention_mask is None:
            return None
        if attention_mask.dim() == 4:
            return attention_mask[:, 0, :, :total_len]
        if attention_mask.dim() == 2:
            return attention_mask[:, None, :total_len]
        return attention_mask[..., :total_len]

    def _expand_kpool_topk(
        self,
        pool_scores: torch.Tensor,
        visible_lengths: torch.Tensor,
    ) -> torch.LongTensor:
        batch_size, query_length, num_pools = pool_scores.shape
        complete_pools = torch.div(visible_lengths, self.index_kpool, rounding_mode="floor")
        group_budget = self.index_topk // self.index_kpool
        select_k = min(group_budget, num_pools)

        if select_k > 0:
            pool_ids = torch.arange(num_pools, device=pool_scores.device)
            pool_valid = pool_ids.view(1, 1, -1) < complete_pools.unsqueeze(-1)
            masked_scores = pool_scores.masked_fill(~pool_valid, torch.finfo(pool_scores.dtype).min)
            selected_groups = masked_scores.topk(select_k, dim=-1).indices
            selected_valid = pool_valid.gather(-1, selected_groups)
            offsets = torch.arange(self.index_kpool, device=pool_scores.device)
            expanded = (selected_groups.unsqueeze(-1) * self.index_kpool + offsets).flatten(-2)
            expanded_valid = selected_valid.unsqueeze(-1).expand(
                batch_size,
                query_length,
                select_k,
                self.index_kpool,
            )
            expanded = expanded.masked_fill(~expanded_valid.flatten(-2), -1)
        else:
            expanded = torch.empty(
                batch_size,
                query_length,
                0,
                dtype=torch.long,
                device=pool_scores.device,
            )

        expanded = F.pad(expanded, (0, self.index_topk - expanded.shape[-1]), value=-1)
        if not self.index_kpool_always_select_tail:
            return expanded

        output_width = self.index_topk + self.index_kpool - 1
        output_positions = torch.arange(output_width, device=pool_scores.device).view(1, 1, -1)
        history_length = (complete_pools * self.index_kpool).clamp(max=self.index_topk)
        history_indices = output_positions.clamp(max=self.index_topk - 1).expand(
            batch_size,
            query_length,
            -1,
        )
        history_values = expanded.gather(-1, history_indices)
        is_history = output_positions < history_length.unsqueeze(-1)

        tail_offset = output_positions - history_length.unsqueeze(-1)
        tail_count = visible_lengths.remainder(self.index_kpool)
        is_tail = tail_offset.ge(0) & tail_offset.lt(tail_count.unsqueeze(-1))
        tail_values = complete_pools.unsqueeze(-1) * self.index_kpool + tail_offset

        result = torch.full(
            (batch_size, query_length, output_width),
            -1,
            dtype=torch.long,
            device=pool_scores.device,
        )
        result = torch.where(is_history, history_values, result)
        return torch.where(is_tail, tail_values, result)

    def build_attention_mask(
        self,
        topk_indices: torch.Tensor,
        attention_mask: torch.Tensor | None,
        batch_size: int,
        seq_length: int,
        total_len: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        valid_indices = topk_indices.ge(0) & topk_indices.lt(total_len)
        safe_indices = topk_indices.clamp(min=0, max=total_len - 1)
        selected_counts = torch.zeros(
            (batch_size, seq_length, total_len),
            dtype=torch.int32,
            device=device,
        )
        selected_counts.scatter_add_(-1, safe_indices, valid_indices.to(torch.int32))
        index_mask = torch.zeros_like(selected_counts, dtype=dtype)
        index_mask = index_mask.masked_fill(selected_counts.eq(0), float("-inf")).unsqueeze(1)
        if attention_mask is not None and attention_mask.dim() == 4:
            return index_mask + attention_mask[..., :total_len]
        if attention_mask is not None:
            return attention_mask.masked_fill(index_mask == float("-inf"), float("-inf"))
        return index_mask

    @torch.no_grad()
    def forward(
        self,
        hidden_states: torch.Tensor,
        q_resid: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None,
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None = None,
        mask_dtype: torch.dtype | None = None,
        total_len: int | None = None,
        topk_indices: torch.Tensor | None = None,
    ) -> tuple[torch.LongTensor, torch.Tensor]:
        batch_size, seq_len, _ = hidden_states.shape
        total_len = seq_len if total_len is None else total_len
        mask_dtype = hidden_states.dtype if mask_dtype is None else mask_dtype

        # Create new indices or use past topk indices
        if topk_indices is None:
            hidden_shape = (batch_size, seq_len, -1, self.head_dim)
            q = self.wq_b(q_resid).view(hidden_shape)
            k = self.k_norm(self.wk(hidden_states)).view(hidden_shape)

            # Partial RoPE or NoPE
            # 70B (`qk_rope_head_dim=64`) rotates it; 300B (`qk_rope_head_dim=0`)
            if position_embeddings is not None:
                cos, sin = position_embeddings
                q, k = apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=2)

            # We temporarily treated it as `num_heads == 1` to apply RoPE
            k = k.squeeze(2)

            if self.kpool_enabled:
                gate_scores = F.linear(hidden_states, self.index_kpool_compress_gate)
                k_cached, gate_scores = self._update_kpool_cache(
                    k,
                    gate_scores,
                    past_key_values,
                    total_len,
                )
            else:
                k_cached = self._update_key_cache(k, past_key_values)

            weights = self.weights_proj(hidden_states).float() * (self.n_heads**-0.5)
            if self.kpool_enabled:
                pooled_keys = self._compress_kpool_keys(k_cached, gate_scores)
                scores = torch.einsum("bshd,bpd->bshp", q.float(), pooled_keys.float()) * self.softmax_scale
                pool_scores = torch.einsum("bshp,bsh->bsp", F.relu(scores), weights)
                prefix_len = total_len - seq_len
                visible_lengths = prefix_len + torch.arange(
                    1,
                    seq_len + 1,
                    device=hidden_states.device,
                )
                visible_lengths = visible_lengths.unsqueeze(0).expand(batch_size, -1)
                topk_indices = self._expand_kpool_topk(pool_scores, visible_lengths)
            else:
                scores = torch.einsum("bshd,btd->bsht", q.float(), k_cached.float()) * self.softmax_scale
                index_scores = torch.einsum("bsht,bsh->bst", F.relu(scores), weights)
                score_mask = self._indexer_score_mask(attention_mask, index_scores.shape[-1])
                if score_mask is not None:
                    index_scores = index_scores + score_mask
                topk = min(self.index_topk, index_scores.shape[-1])
                topk_indices = index_scores.topk(topk, dim=-1).indices

        attention_mask = self.build_attention_mask(
            topk_indices,
            attention_mask,
            batch_size,
            seq_len,
            total_len,
            mask_dtype,
            hidden_states.device,
        )
        return topk_indices, attention_mask


class Glm5NextAttention(GlmMoeDsaAttention):
    def __init__(self, config: Glm5NextConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.q_a_layernorm = (
            Glm5NextRMSNorm(config.q_lora_rank, eps=config.rms_norm_eps) if self.q_lora_rank is not None else None
        )
        self.kv_a_layernorm = Glm5NextRMSNorm(self.kv_lora_rank, eps=config.rms_norm_eps)
        # TODO: split kpool indexing vs normal full token indexing
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

        # ===== Query path =====
        if self.q_lora_rank is None:
            query_states = self.q_proj(hidden_states)
            q_resid = None
        else:
            q_resid = self.q_a_layernorm(self.q_a_proj(hidden_states))
            query_states = self.q_b_proj(q_resid)
        query_states = query_states.view(batch_size, seq_length, -1, self.qk_head_dim).transpose(1, 2)
        # NoPE MLA: GLM-5-Next full-attention layers keep the `qk_rope_head_dim` query/key
        # slice but do NOT rotate it (matches sglang `skip_rope=mla_nope`). Position
        # information is carried by the KDA linear-attention layers and the DSA indexer,
        # which applies RoPE on its own projections.

        # ===== KV path =====
        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        k_compressed, k_pe = torch.split(compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        k_compressed = self.kv_a_layernorm(k_compressed)

        kv_expanded = self.kv_b_proj(k_compressed)
        kv_expanded = kv_expanded.view(batch_size, seq_length, -1, self.qk_nope_head_dim + self.v_head_dim)
        key_states, value_states = torch.split(kv_expanded, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        if position_embeddings is not None:
            # Single-head (MQA) key-pe stream broadcast to all heads, unrotated (NoPE).
            k_pe = k_pe.view(batch_size, 1, seq_length, self.qk_rope_head_dim)
            k_pe = k_pe.expand(-1, key_states.shape[1], -1, -1)
            key_states = torch.cat([key_states, k_pe], dim=-1)

        # Cache update
        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

        topk_indices = None
        if self.indexer is not None:
            if q_resid is None:
                raise ValueError("GLM-5-Next DSA indexer requires q_lora_rank to be set.")
            reused_topk_indices = prev_topk_indices if self.skip_topk else None
            topk_indices, attention_mask = self.indexer(
                hidden_states,
                q_resid,
                position_embeddings,
                attention_mask,
                past_key_values=past_key_values,
                mask_dtype=query_states.dtype,
                total_len=key_states.shape[2],
                topk_indices=reused_topk_indices,
            )

        # Flash attention head_dim padding
        if is_flash_attention_requested(self.config) and self.qk_head_dim != self.v_head_dim:
            value_states = F.pad(value_states, [0, self.qk_head_dim - self.v_head_dim])

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

        if is_flash_attention_requested(self.config) and self.qk_head_dim != self.v_head_dim:
            attn_output = attn_output[:, :, :, : self.v_head_dim]

        attn_output = attn_output.reshape(batch_size, seq_length, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights, topk_indices if self.next_skip_topk else None


# =============================================================================
# Decoder Layer
# =============================================================================


class Glm5NextDecoderLayer(GlmMoeDsaDecoderLayer):
    def __init__(self, config: Glm5NextConfig, layer_idx: int):
        self.block_type = config.layer_types[layer_idx]

        super().__init__(config, layer_idx)
        self.self_attn = (
            Glm5NextLinearAttention(config, layer_idx)
            if self.block_type == "linear_attention"
            else Glm5NextAttention(config, layer_idx)
        )

        self.uses_mhc = config.mhc
        self.attn_hc = Glm5NextHyperConnection(config) if config.mhc else None
        self.ffn_hc = Glm5NextHyperConnection(config) if config.mhc else None

    def apply_residual(self, post, comb, hidden_states, residual, dtype=None):
        """Either apply normal additive residual stream or MHC residual stream"""
        if post is None and comb is None:
            return hidden_states + residual

        return post.to(dtype).unsqueeze(-1) * hidden_states.unsqueeze(-2) + torch.matmul(
            comb.to(dtype).transpose(-1, -2), residual
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
        dtype = hidden_states.dtype

        residual = hidden_states
        post, comb, hidden_states = self.attn_hc(hidden_states) if self.uses_mhc else (None, None, hidden_states)
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
        hidden_states = self.apply_residual(post, comb, hidden_states, residual, dtype=dtype)

        residual = hidden_states
        post, comb, hidden_states = self.ffn_hc(hidden_states) if self.uses_mhc else (None, None, hidden_states)
        # Feed forward
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.apply_residual(post, comb, hidden_states, residual, dtype=dtype)

        return hidden_states, topk_indices


# =============================================================================
# PreTrainedModel, Model, CausalLM
# =============================================================================


@auto_docstring
class Glm5NextPreTrainedModel(PreTrainedModel):
    config: Glm5NextConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Glm5NextDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = False
    _supports_sdpa = True
    _supports_flex_attn = False

    _can_compile_fullgraph = True
    _supports_attention_backend = True
    _can_record_outputs = {
        "attentions": Glm5NextAttention,
        "hidden_states": Glm5NextDecoderLayer,
        "router_logits": OutputRecorder(Glm5NextTopkRouter, index=0),  # noqa: F821
    }
    _keep_in_fp32_modules_strict = ["e_score_correction_bias"]
    _keys_to_ignore_on_load_unexpected = [r"model\.layers\.45\.", r"model\.layers\.\d+\.shared_head\."]
    _keep_in_fp32_modules = []
    _compatible_flash_implementations = ["kernels-community/flash-mla"]

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
class Glm5NextModel(DeepseekV4Model):
    def __init__(self, config: Glm5NextConfig):
        super().__init__(config)
        # Potential NoPE we detect by the head dim (alias for `qk_rope_head_dim`)
        self.rotary_emb = Glm5NextRotaryEmbedding(config) if self.config.head_dim > 0 else None
        self.hc_head = Glm5NextHyperHead() if config.mhc else nn.Identity()

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
            past_key_values = Glm5NextDynamicCache(config=self.config)

        if position_ids is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
            position_ids = position_ids.unsqueeze(0)

        if not isinstance(causal_mask_mapping := attention_mask, dict):
            # Prepare mask arguments
            mask_kwargs = {
                "config": self.config,
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            # Create the masks
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
                "linear_attention": create_recurrent_attention_mask(**mask_kwargs),
            }

        hidden_states = inputs_embeds
        position_embeddings = (
            self.rotary_emb(hidden_states, position_ids=position_ids) if self.rotary_emb is not None else None
        )

        if self.config.mhc:
            hidden_states = hidden_states.unsqueeze(2).expand(-1, -1, self.config.hc_mult, -1).contiguous()

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

        hidden_states = self.hc_head(hidden_states)
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


__all__ = [
    "Glm5NextPreTrainedModel",
    "Glm5NextModel",
    "Glm5NextForCausalLM",
]
