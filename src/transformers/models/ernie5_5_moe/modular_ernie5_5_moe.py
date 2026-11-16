# Copyright 2026 Baidu and The HuggingFace Inc. team. All rights reserved.
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
import torch.nn.functional as F
from torch import nn

from ... import initialization as init
from ...cache_utils import Cache, DynamicCache
from ...integrations import use_experts_implementation
from ...masking_utils import create_causal_mask, create_sliding_window_causal_mask
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import MoeModelOutputWithPast
from ...modeling_rope_utils import ROPE_INIT_FUNCTIONS
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, logging
from ...utils.deprecation import deprecate_kwarg
from ...utils.generic import merge_with_config_defaults
from ...utils.output_capturing import OutputRecorder, capture_outputs
from ..deepseek_v4.modeling_deepseek_v4 import (
    DeepseekV4Experts,
    DeepseekV4HCACache,
    DeepseekV4HCACompressor,
    DeepseekV4HyperConnection,
    DeepseekV4HyperHead,
    DeepseekV4RMSNorm,
    DeepseekV4RotaryEmbedding,
    DeepseekV4UnweightedRMSNorm,
    apply_rotary_pos_emb,
)
from ..gpt_oss.modeling_gpt_oss import eager_attention_forward
from ..mixtral.modeling_mixtral import MixtralForCausalLM, MixtralPreTrainedModel
from .configuration_ernie5_5_moe import Ernie5_5_MoeConfig


logger = logging.get_logger(__name__)


def apply_rotary_pos_emb_prefix(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, unsqueeze_dim: int = 1):
    """Half-split (non-interleaved) RoPE on the *leading* rope slice of `x`, used by the lightning indexer.

    `cos` / `sin` hold one entry per frequency (as returned by `Ernie5_5_MoeRotaryEmbedding`), so the rotated slice
    is `2 * cos.shape[-1]` channels wide. The indexer lays its heads out as `[rope | nope]`.
    """
    cos = cos.unsqueeze(unsqueeze_dim).float()
    sin = sin.unsqueeze(unsqueeze_dim).float()
    rope_dim = 2 * cos.shape[-1]
    rope, nope = x[..., :rope_dim], x[..., rope_dim:]
    x1, x2 = rope.float().chunk(2, dim=-1)
    rotated = torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1).to(x.dtype)
    return torch.cat([rotated, nope], dim=-1)


def situ_glu(gate: torch.Tensor, up: torch.Tensor, beta: float, linear_beta: float | None = None) -> torch.Tensor:
    r"""SiTU-GLU: `beta * tanh(gate / beta) * sigmoid(gate) * linear_beta * tanh(up / linear_beta)`, in fp32.

    The tanh terms softly saturate both branches (at `beta` and `linear_beta`), where SwiGLU variants clamp them.
    """
    gate, up_fp32 = gate.float(), up.float()
    activated = beta * torch.tanh(gate / beta) * torch.sigmoid(gate)
    if linear_beta is not None:
        up_fp32 = linear_beta * torch.tanh(up_fp32 / linear_beta)
    return (activated * up_fp32).to(up.dtype)


def virtual_head_mix(attn_output: torch.Tensor, vha_u: torch.Tensor, vha_v: torch.Tensor) -> torch.Tensor:
    r"""Virtual Head Attention (VHA) post-mix over the head axis: `o_h <- o_h + sum_r V[h, r] sum_g U[g, r] o_g`,
    i.e. `o <- (I + V U^T) o` with the low-rank factors `U, V` of shape `[num_heads, rank]`.

    `attn_output` is `[batch, seq_len, num_heads, head_dim]`. Computed in fp32 like the reference.
    """
    mixed = torch.einsum("gr,bsgd->bsrd", vha_u.float(), attn_output.float())
    mixed = torch.einsum("hr,bsrd->bshd", vha_v.float(), mixed)
    return (attn_output.float() + mixed).to(attn_output.dtype)


class Ernie5_5_MoeRMSNorm(DeepseekV4RMSNorm):
    pass


class Ernie5_5_MoeUnweightedRMSNorm(DeepseekV4UnweightedRMSNorm):
    pass


class Ernie5_5_MoeRotaryEmbedding(DeepseekV4RotaryEmbedding):
    pass


class Ernie5_5_MoeHCACache(DeepseekV4HCACache):
    pass


class Ernie5_5_MoeHCACompressor(DeepseekV4HCACompressor):
    pass


class Ernie5_5_MoeHCAAttention(nn.Module):
    r"""
    Heavily Compressed Attention (HCA) layer. It is DeepSeek-V4's HCA block (shared-KV MQA over a sliding window
    plus one compressed KV entry per `compress_rate` tokens, a per-head attention sink, and a grouped low-rank
    output projection) with two ERNIE 5.5 additions:

    * Virtual Head Attention: a low-rank head mix `I + V U^T` applied to the core attention output.
    * A low-rank output gate computed from the query latent:
      `o_proj(grouped_o_proj(o) * sigmoid(gate_proj(q_latent)))`.

    The projections keep the names of the released checkpoint. The DeepSeek-V4 names (`q_a_proj`, `q_b_proj`) are
    already used by the MLA layers with a different meaning, which would make the conversion mapping ambiguous when
    saving back to the original format.
    """

    def __init__(self, config: Ernie5_5_MoeConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.layer_type = config.layer_types[layer_idx]
        self.rope_layer_type = "compress"
        self.num_heads = config.num_attention_heads
        self.num_key_value_groups = config.num_attention_heads  # single KV head, broadcast to all
        self.head_dim = config.head_dim
        self.o_groups = config.o_groups
        self.sliding_window = config.sliding_window
        self.attention_dropout = config.attention_dropout
        self.is_causal = True
        self.scaling = self.head_dim**-0.5

        self.linear_q_down_proj = nn.Linear(config.hidden_size, config.q_lora_rank, bias=False)
        self.q_layernorm = Ernie5_5_MoeRMSNorm(config.q_lora_rank, eps=config.rms_norm_eps)
        self.linear_q_up_proj = nn.Linear(config.q_lora_rank, self.num_heads * self.head_dim, bias=False)
        self.q_head_norm = Ernie5_5_MoeUnweightedRMSNorm(eps=config.rms_norm_eps)
        self.linear_kv_proj = nn.Linear(config.hidden_size, self.head_dim, bias=False)
        self.kv_layernorm = Ernie5_5_MoeRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        # Block-diagonal projection of each group of `num_heads / o_groups` heads to `o_lora_rank` channels, stored
        # as a plain `[o_groups * o_lora_rank, num_heads * head_dim / o_groups]` weight
        self.linear_o_group_proj = nn.Parameter(
            torch.empty(config.o_groups * config.o_lora_rank, self.num_heads * self.head_dim // config.o_groups)
        )
        self.o_proj = nn.Linear(config.o_groups * config.o_lora_rank, config.hidden_size, bias=False)
        self.gate_proj = nn.Linear(config.q_lora_rank, config.o_groups * config.o_lora_rank, bias=False)
        self.vha_postmix_U = nn.Parameter(torch.empty(self.num_heads, config.vha_rank))
        self.vha_postmix_V = nn.Parameter(torch.empty(self.num_heads, config.vha_rank))
        self.sinks = nn.Parameter(torch.empty(self.num_heads))
        self.compressor = Ernie5_5_MoeHCACompressor(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: dict[str, tuple[torch.Tensor, torch.Tensor]],
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
        cache: Cache | None = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        cos, sin = position_embeddings[self.rope_layer_type]

        q_residual = self.q_layernorm(self.linear_q_down_proj(hidden_states))
        q = self.linear_q_up_proj(q_residual).view(*hidden_shape).transpose(1, 2)
        q = self.q_head_norm(q)
        q = apply_rotary_pos_emb(q, cos, sin)

        kv = self.kv_layernorm(self.linear_kv_proj(hidden_states)).view(*hidden_shape).transpose(1, 2)
        kv = apply_rotary_pos_emb(kv, cos, sin)

        if cache is not None:  # sliding window where K == V
            kv = cache.update(kv, kv, self.layer_idx, kwargs=kwargs)[0]

        compressed_kv, block_bias = self.compressor(hidden_states, q_residual, position_ids, cache, self.layer_idx)
        kv = torch.cat([kv, compressed_kv], dim=2)

        # The compressed entries are appended after the sliding-window keys, so extend the mask with their
        # per-query causality (`block_bias`) instead of zero-padding it.
        if isinstance(attention_mask, torch.Tensor) and kv.shape[2] > attention_mask.shape[-1]:
            if block_bias is not None:
                attention_mask = torch.cat([attention_mask, block_bias.to(attention_mask.dtype)], dim=-1)
            else:
                attention_mask = F.pad(attention_mask, (0, kv.shape[2] - attention_mask.shape[-1]), value=0.0)

        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )
        attn_output, attn_weights = attention_interface(
            self,
            q,
            kv,
            kv,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            s_aux=self.sinks,
            **kwargs,
        )

        # K == V, so the values carry RoPE on their trailing rope slice: rotate the output back at the query
        # position. `apply_rotary_pos_emb` expects `[B, S, H, D]`.
        attn_output = apply_rotary_pos_emb(attn_output.transpose(1, 2), cos, -sin).transpose(1, 2)
        attn_output = virtual_head_mix(attn_output, self.vha_postmix_U, self.vha_postmix_V)

        # Grouped projection: [G, N, in] @ [G, in, out] with N = batch * seq_len
        group_weight = self.linear_o_group_proj.view(self.o_groups, -1, self.linear_o_group_proj.shape[-1])
        grouped = attn_output.reshape(-1, self.o_groups, group_weight.shape[-1]).transpose(0, 1)
        grouped = torch.bmm(grouped, group_weight.transpose(1, 2)).transpose(0, 1).reshape(*input_shape, -1)
        grouped = grouped * torch.sigmoid(self.gate_proj(q_residual))
        return self.o_proj(grouped), attn_weights


class Ernie5_5_MoeIndexer(nn.Module):
    r"""
    Lightning indexer of the MLA + DSA layers. It scores, for every query, the keys that lie *before* the sliding
    window, `score[t, s] = sum_h w[t, h] * relu(q[t, h] . k[s]) / sqrt(index_head_dim)`, and keeps the top
    `index_topk` of them. The layer then attends to those keys plus the sliding window.

    The reference rotates `q` and `k` with a Hadamard transform and scores them in FP8; the transform is orthogonal
    (it preserves dot products) and FP8 is a speed optimization, so both are skipped here. Like DeepSeek-V3.2, the
    indexer is trained by a separate objective upstream, so no gradient flows through it here.
    """

    def __init__(self, config: Ernie5_5_MoeConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_heads = config.index_n_heads
        self.head_dim = config.index_head_dim
        self.index_topk = config.index_topk
        self.softmax_scale = self.head_dim**-0.5
        self.weights_scale = self.n_heads**-0.5

        self.wq_b = nn.Linear(config.hybrid_mla_q_lora_rank, self.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(config.hidden_size, self.head_dim, bias=False)
        self.k_norm = nn.LayerNorm(self.head_dim, eps=1e-6)
        self.weights_proj = nn.Linear(config.hidden_size, self.n_heads, bias=False)

    @torch.no_grad()
    def forward(
        self,
        hidden_states: torch.Tensor,
        q_residual: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        remote: torch.Tensor,
        attention_mask: torch.Tensor | None,
        cache: Cache | None = None,
    ) -> torch.Tensor:
        """
        Args:
            remote (`torch.Tensor`): boolean `[seq_len, num_keys]`, `True` for the keys at least `sliding_window`
                tokens older than the query, the only candidates of the top-k.
            attention_mask (`torch.Tensor`, *optional*): the layer's `[batch, 1, seq_len, num_keys]` causal mask, so
                that padding tokens are never selected.

        Returns:
            `torch.Tensor`: boolean `[batch, seq_len, num_keys]`, `True` at the selected keys.
        """
        batch_size, seq_len, _ = hidden_states.shape
        cos, sin = position_embeddings

        q = self.wq_b(q_residual).view(batch_size, seq_len, self.n_heads, self.head_dim)
        q = apply_rotary_pos_emb_prefix(q, cos, sin, unsqueeze_dim=2)
        k = self.wk(hidden_states)
        # `k_norm` is kept in fp32 (see `_keep_in_fp32_modules_strict`)
        k = self.k_norm(k.to(self.k_norm.weight.dtype)).to(k.dtype)
        k = apply_rotary_pos_emb_prefix(k.unsqueeze(2), cos, sin, unsqueeze_dim=2).squeeze(2)  # [B, S, D]
        if cache is not None:
            k = cache.update_indexer(k, self.layer_idx)

        # [B, S, H, D] @ [B, 1, D, T] -> [B, S, H, T]
        scores = F.relu(torch.matmul(q.float(), k.float().transpose(-1, -2).unsqueeze(1))) * self.softmax_scale
        # Weight per head and sum across heads: [B, S, 1, H] @ [B, S, H, T] -> [B, S, T]
        weights = self.weights_proj(hidden_states).float() * self.weights_scale
        index_scores = torch.matmul(weights.unsqueeze(-2), scores).squeeze(-2)

        if attention_mask is not None:
            if attention_mask.dtype == torch.bool:
                index_scores = index_scores.masked_fill(~attention_mask[:, 0], float("-inf"))
            else:
                index_scores = index_scores + attention_mask[:, 0].float()
        index_scores = index_scores.masked_fill(~remote, float("-inf"))
        top_k = min(self.index_topk, index_scores.shape[-1])
        top_k_indices = index_scores.topk(top_k, dim=-1).indices
        selected = torch.zeros(index_scores.shape, dtype=torch.bool, device=index_scores.device)
        selected = selected.scatter(-1, top_k_indices, True)
        # With fewer candidates than `index_topk`, the top-k also returns non-candidates: drop them
        return selected & remote


class Ernie5_5_MoeDSAAttention(nn.Module):
    r"""
    Multi-head Latent Attention with DeepSeek Sparse Attention (MLA + DSA). Each query attends to the `index_topk`
    keys picked by the lightning indexer among those older than the sliding window, plus the sliding window itself,
    with a per-head attention sink. As in the HCA layers, the output goes through the Virtual Head Attention head mix
    and a low-rank sigmoid gate computed from the query latent before `o_proj`.
    """

    def __init__(self, config: Ernie5_5_MoeConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.rope_layer_type = "main"
        self.num_heads = config.hybrid_mla_num_attention_heads
        self.num_key_value_groups = 1  # keys and values are expanded to every head
        self.q_lora_rank = config.hybrid_mla_q_lora_rank
        self.kv_lora_rank = config.hybrid_mla_kv_lora_rank
        self.qk_nope_head_dim = config.hybrid_mla_qk_nope_head_dim
        self.qk_rope_head_dim = config.hybrid_mla_qk_rope_head_dim
        self.v_head_dim = config.hybrid_mla_v_head_dim
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.sliding_window = config.sliding_window
        self.attention_dropout = config.attention_dropout
        self.is_causal = True
        self.scaling = self.qk_head_dim**-0.5

        self.q_a_proj = nn.Linear(config.hidden_size, self.q_lora_rank, bias=False)
        self.q_a_layernorm = Ernie5_5_MoeRMSNorm(self.q_lora_rank, eps=config.rms_norm_eps)
        self.q_b_proj = nn.Linear(self.q_lora_rank, self.num_heads * self.qk_head_dim, bias=False)
        self.kv_a_proj_with_mqa = nn.Linear(config.hidden_size, self.kv_lora_rank + self.qk_rope_head_dim, bias=False)
        self.kv_a_layernorm = Ernie5_5_MoeRMSNorm(self.kv_lora_rank, eps=config.rms_norm_eps)
        self.kv_b_proj = nn.Linear(
            self.kv_lora_rank, self.num_heads * (self.qk_nope_head_dim + self.v_head_dim), bias=False
        )
        self.o_proj = nn.Linear(self.num_heads * self.v_head_dim, config.hidden_size, bias=False)
        self.gate_proj = nn.Linear(self.q_lora_rank, self.num_heads * self.v_head_dim, bias=False)
        self.vha_postmix_U = nn.Parameter(torch.empty(self.num_heads, config.vha_rank))
        self.vha_postmix_V = nn.Parameter(torch.empty(self.num_heads, config.vha_rank))
        # The attention sink, named as in the released checkpoint (`sinks` below is what the attention reads)
        self.softmax_offset = nn.Parameter(torch.empty(self.num_heads))
        self.indexer = Ernie5_5_MoeIndexer(config, layer_idx)

    @property
    def sinks(self) -> nn.Parameter:
        return self.softmax_offset

    def expand_kv(self, kv_nope: torch.Tensor, k_rot: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Expands the latents `[B, 1, T, kv_lora_rank]` / `[B, 1, T, qk_rope_head_dim]` into per-head keys/values."""
        batch_size, _, seq_length, _ = kv_nope.shape
        kv = self.kv_b_proj(kv_nope).view(batch_size, seq_length, -1, self.qk_nope_head_dim + self.v_head_dim)
        k_nope, value_states = torch.split(kv.transpose(1, 2), [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        key_states = torch.cat([k_nope, k_rot.expand(-1, k_nope.shape[1], -1, -1)], dim=-1)
        return key_states, value_states

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: dict[str, tuple[torch.Tensor, torch.Tensor]],
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
        cache: Cache | None = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        batch_size, seq_length = hidden_states.shape[:-1]
        cos, sin = position_embeddings[self.rope_layer_type]

        q_residual = self.q_a_layernorm(self.q_a_proj(hidden_states))
        q_states = self.q_b_proj(q_residual).view(batch_size, seq_length, -1, self.qk_head_dim).transpose(1, 2)
        q_pass, q_rot = torch.split(q_states, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        kv_nope, k_rot = torch.split(
            self.kv_a_proj_with_mqa(hidden_states), [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
        )
        kv_nope = self.kv_a_layernorm(kv_nope).view(batch_size, 1, seq_length, self.kv_lora_rank)
        k_rot = k_rot.view(batch_size, 1, seq_length, self.qk_rope_head_dim)
        # MLA rotates interleaved channel pairs; `apply_rotary_pos_emb` rotates the trailing rope slice, here all of it
        q_rot = apply_rotary_pos_emb(q_rot, cos, sin)
        k_rot = apply_rotary_pos_emb(k_rot, cos, sin)

        # Cache read / write is performed while the latent KV is still compressed
        if cache is not None:
            kv_nope, k_rot = cache.update(kv_nope, k_rot, self.layer_idx, kwargs=kwargs)

        query_states = torch.cat([q_pass, q_rot], dim=-1)
        key_states, value_states = self.expand_kv(kv_nope, k_rot)

        # Sparse selection: every query attends to its sliding window plus the indexer's top-k among older keys.
        # Distances are counted in cache slots, so they stay right with left padding.
        num_keys = key_states.shape[2]
        query_index = torch.arange(seq_length, device=key_states.device) + (num_keys - seq_length)
        distance = query_index[:, None] - torch.arange(num_keys, device=key_states.device)[None, :]  # [S, T]
        remote = distance >= self.sliding_window
        min_value = torch.finfo(query_states.dtype).min
        if attention_mask is None:
            attention_mask = torch.zeros(distance.shape, dtype=query_states.dtype, device=query_states.device)
            attention_mask = attention_mask.masked_fill(distance < 0, min_value)[None, None]
        selected = self.indexer(hidden_states, q_residual, (cos, sin), remote, attention_mask, cache=cache)
        # Keys inside the window are always kept, future ones are removed by the causal mask
        allowed = (selected | ~remote).unsqueeze(1)  # [B, 1, S, T]
        attention_mask = torch.where(allowed, attention_mask, min_value)

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
            s_aux=self.sinks,
            **kwargs,
        )

        attn_output = virtual_head_mix(attn_output, self.vha_postmix_U, self.vha_postmix_V)
        attn_output = attn_output.reshape(batch_size, seq_length, -1)
        attn_output = attn_output * torch.sigmoid(self.gate_proj(q_residual))
        return self.o_proj(attn_output), attn_weights


class Ernie5_5_MoeStreamNorm(nn.Module):
    r"""
    Unweighted RMS normalization of the flattened mHC streams with the epsilon outside the square root,
    `x / (sqrt(mean(x^2)) + eps)` (DeepSeek-V4 uses `x / sqrt(mean(x^2) + eps)`). The two differ the most on the
    small-norm embedding streams entering the first layer.
    """

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states / (hidden_states.pow(2).mean(-1, keepdim=True).sqrt() + self.eps)


class Ernie5_5_MoeHyperConnection(DeepseekV4HyperConnection):
    def __init__(self, config: Ernie5_5_MoeConfig):
        super().__init__(config)
        self.input_norm = Ernie5_5_MoeStreamNorm(eps=config.rms_norm_eps)


class Ernie5_5_MoeHyperHead(DeepseekV4HyperHead):
    # The final contraction keeps the DeepSeek-V4 normalization (the reference serving stack runs the DeepSeek-V4
    # kernel for it)
    pass


class Ernie5_5_MoeMLP(nn.Module):
    """SiTU-GLU feed-forward block, used by the dense layers and as the shared expert."""

    def __init__(self, config: Ernie5_5_MoeConfig, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, config.hidden_size, bias=False)
        self.beta = config.activation_situ_beta
        self.linear_beta = config.activation_situ_linear_beta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(situ_glu(self.gate_proj(x), self.up_proj(x), self.beta, self.linear_beta))


@use_experts_implementation
class Ernie5_5_MoeExperts(DeepseekV4Experts):
    """Routed experts, stored as 3D tensors. They operate in the `moe_latent_size` space with SiTU-GLU."""

    def __init__(self, config: Ernie5_5_MoeConfig):
        nn.Module.__init__(self)
        self.num_experts = config.n_routed_experts
        self.hidden_dim = config.moe_latent_size
        self.intermediate_dim = config.moe_intermediate_size
        self.gate_up_proj = nn.Parameter(torch.empty(self.num_experts, 2 * self.intermediate_dim, self.hidden_dim))
        self.down_proj = nn.Parameter(torch.empty(self.num_experts, self.hidden_dim, self.intermediate_dim))
        self.beta = config.activation_situ_beta
        self.linear_beta = config.activation_situ_linear_beta

    def _apply_gate(self, gate_up: torch.Tensor) -> torch.Tensor:
        # Lives on the class so the grouped_mm / batched_mm backends apply the same activation
        gate, up = gate_up.chunk(2, dim=-1)
        return situ_glu(gate, up, self.beta, self.linear_beta)


class Ernie5_5_MoeTopKRouter(nn.Module):
    r"""
    Multi-Router: two independently initialized sigmoid routers whose scores are summed,
    `p = sigmoid(x W_0^T) + sigmoid(x W_1^T)`. The top-k is picked on `p + e_score_correction_bias`
    (auxiliary-loss-free balancing), the selected scores are renormalized, and each expert's weight is multiplied by
    its own learned scale `routed_scaling_factor_param`. Routing runs in fp32.
    """

    def __init__(self, config: Ernie5_5_MoeConfig):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.n_routed_experts
        self.hidden_dim = config.hidden_size
        self.norm_topk_prob = config.norm_topk_prob
        self.weight = nn.Parameter(torch.empty(self.num_experts, self.hidden_dim))
        self.weight_1 = nn.Parameter(torch.empty(self.num_experts, self.hidden_dim))
        self.routed_scaling_factor_param = nn.Parameter(torch.empty(self.num_experts))
        self.e_score_correction_bias = nn.Buffer(torch.zeros(self.num_experts))

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        flat = hidden_states.reshape(-1, self.hidden_dim).float()
        router_logits = F.linear(flat, self.weight.float())
        scores = torch.sigmoid(router_logits) + torch.sigmoid(F.linear(flat, self.weight_1.float()))
        indices = torch.topk(scores + self.e_score_correction_bias.float(), self.top_k, dim=-1, sorted=False).indices
        weights = scores.gather(1, indices)
        if self.norm_topk_prob:
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-20)
        weights = weights * self.routed_scaling_factor_param.float()[indices]
        return router_logits, weights, indices


class Ernie5_5_MoeSparseMoeBlock(nn.Module):
    r"""
    Latent MoE: tokens are projected to `moe_latent_size` before the routed experts, and the combined expert output
    goes through an RMSNorm and back up to `hidden_size`. The shared expert sees the original hidden states.
    """

    def __init__(self, config: Ernie5_5_MoeConfig):
        super().__init__()
        self.gate = Ernie5_5_MoeTopKRouter(config)
        self.experts = Ernie5_5_MoeExperts(config)
        self.shared_experts = Ernie5_5_MoeMLP(config, config.moe_intermediate_size * config.n_shared_experts)
        self.fc1_latent_proj = nn.Linear(config.hidden_size, config.moe_latent_size, bias=False)
        self.latent_norm = Ernie5_5_MoeRMSNorm(config.moe_latent_size, eps=config.rms_norm_eps)
        self.fc2_latent_proj = nn.Linear(config.moe_latent_size, config.hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch, seq_len, hidden_dim = hidden_states.shape
        _, weights, indices = self.gate(hidden_states)
        latent = self.fc1_latent_proj(hidden_states).view(batch * seq_len, -1)
        routed = self.experts(latent, indices, weights)
        routed = self.fc2_latent_proj(self.latent_norm(routed)).view(batch, seq_len, hidden_dim)
        return routed + self.shared_experts(hidden_states)


class Ernie5_5_MoeDecoderLayer(GradientCheckpointingLayer):
    r"""
    ERNIE 5.5 decoder block. The residual is a stack of `hc_mult` streams `[B, S, hc_mult, D]`, mixed in and out of
    the attention and feed-forward blocks by two manifold-constrained Hyper-Connections, as in DeepSeek-V4. The
    attention is HCA or MLA + DSA depending on `config.layer_types`, the feed-forward block dense for the first
    `first_k_dense_replace` layers and a latent MoE afterwards.
    """

    def __init__(self, config: Ernie5_5_MoeConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.attention_type = config.layer_types[layer_idx]
        if self.attention_type == "indexed_attention":
            self.self_attn = Ernie5_5_MoeDSAAttention(config, layer_idx)
        else:
            self.self_attn = Ernie5_5_MoeHCAAttention(config, layer_idx)
        if layer_idx < config.first_k_dense_replace:
            self.mlp = Ernie5_5_MoeMLP(config, config.intermediate_size)
        else:
            self.mlp = Ernie5_5_MoeSparseMoeBlock(config)
        self.input_layernorm = Ernie5_5_MoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Ernie5_5_MoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attn_hc = Ernie5_5_MoeHyperConnection(config)
        self.ffn_hc = Ernie5_5_MoeHyperConnection(config)

    def forward(self, hidden_states: torch.Tensor, **kwargs: Unpack[TransformersKwargs]) -> torch.Tensor:
        # hidden_states throughout: [B, S, hc_mult, hidden]. `comb` mixes the streams transposed (sum over the
        # first hc axis); it is doubly-stochastic but not symmetric, so the direction matters. The fp32 `post` and
        # `comb` update the streams in fp32, rounded once to the stream dtype.
        post, comb, collapsed = self.attn_hc(hidden_states)
        attn_output, _ = self.self_attn(self.input_layernorm(collapsed), **kwargs)
        hidden_states = self._update_streams(hidden_states, attn_output, post, comb)

        post, comb, collapsed = self.ffn_hc(hidden_states)
        mlp_output = self.mlp(self.post_attention_layernorm(collapsed))
        return self._update_streams(hidden_states, mlp_output, post, comb)

    @staticmethod
    def _update_streams(
        hidden_states: torch.Tensor, block_output: torch.Tensor, post: torch.Tensor, comb: torch.Tensor
    ) -> torch.Tensor:
        mixed = torch.matmul(comb.transpose(-1, -2), hidden_states.float())
        return (post.unsqueeze(-1) * block_output.float().unsqueeze(-2) + mixed).to(hidden_states.dtype)


class Ernie5_5_MoePreTrainedModel(MixtralPreTrainedModel):
    config: Ernie5_5_MoeConfig
    base_model_prefix = "model"
    _no_split_modules = ["Ernie5_5_MoeDecoderLayer"]
    # Eager only, like DeepSeek-V4: the HCA heads are 512 wide (above the FlashAttention limit), SDPA cannot apply
    # the per-head sinks, and both attention types extend or rewrite the mask inside the layer.
    _supports_flash_attn = False
    _supports_sdpa = False
    _supports_flex_attn = False
    _can_compile_fullgraph = False
    _keep_in_fp32_modules_strict = [
        "attn_hc",
        "ffn_hc",
        "hc_head",
        "sinks",
        "softmax_offset",
        "position_bias",
        "indexer.k_norm",
        "mlp.gate.weight",
        "mlp.gate.weight_1",
        "routed_scaling_factor_param",
        "e_score_correction_bias",
    ]
    # The compressor's running-window state cannot be rolled back across drafts
    _is_stateful = True
    _can_record_outputs = {
        "router_logits": OutputRecorder(Ernie5_5_MoeTopKRouter, index=0),
        "hidden_states": Ernie5_5_MoeDecoderLayer,
        "attentions": [
            OutputRecorder(Ernie5_5_MoeHCAAttention, index=1),
            OutputRecorder(Ernie5_5_MoeDSAAttention, index=1),
        ],
    }

    @torch.no_grad()
    def _init_weights(self, module):
        PreTrainedModel._init_weights(self, module)
        std = self.config.initializer_range
        if isinstance(module, Ernie5_5_MoeTopKRouter):
            init.normal_(module.weight, mean=0.0, std=std)
            init.normal_(module.weight_1, mean=0.0, std=std)
            init.constant_(module.routed_scaling_factor_param, self.config.routed_scaling_factor)
            init.zeros_(module.e_score_correction_bias)
        elif isinstance(module, Ernie5_5_MoeExperts):
            init.normal_(module.gate_up_proj, mean=0.0, std=std)
            init.normal_(module.down_proj, mean=0.0, std=std)
        elif isinstance(module, (Ernie5_5_MoeHCAAttention, Ernie5_5_MoeDSAAttention)):
            init.zeros_(module.sinks)
            # `V = 0` makes the head mix start as the identity while keeping both factors trainable
            init.normal_(module.vha_postmix_U, mean=0.0, std=std)
            init.zeros_(module.vha_postmix_V)
            if isinstance(module, Ernie5_5_MoeHCAAttention):
                init.normal_(module.linear_o_group_proj, mean=0.0, std=std)
        elif isinstance(module, Ernie5_5_MoeHyperConnection):
            init.normal_(module.fn, mean=0.0, std=std)
            init.zeros_(module.base)
            init.ones_(module.scale)
        elif isinstance(module, Ernie5_5_MoeHyperHead):
            init.normal_(module.hc_fn, mean=0.0, std=std)
            init.zeros_(module.hc_base)
            init.ones_(module.hc_scale)
        elif isinstance(module, Ernie5_5_MoeHCACompressor):
            init.zeros_(module.position_bias)
        elif isinstance(module, Ernie5_5_MoeLMHead):
            init.zeros_(module.multimax_ranges)
            init.zeros_(module.multimax_ts)
        elif isinstance(module, Ernie5_5_MoeRotaryEmbedding):
            for layer_type in module.layer_types:
                rope_init_fn = module.compute_default_rope_parameters
                if module.rope_type[layer_type] != "default":
                    rope_init_fn = ROPE_INIT_FUNCTIONS[module.rope_type[layer_type]]
                curr_inv_freq, _ = rope_init_fn(module.config, layer_type=layer_type)
                init.copy_(getattr(module, f"{layer_type}_inv_freq"), curr_inv_freq)
                init.copy_(getattr(module, f"{layer_type}_original_inv_freq"), curr_inv_freq)


@auto_docstring
class Ernie5_5_MoeModel(Ernie5_5_MoePreTrainedModel):
    def __init__(self, config: Ernie5_5_MoeConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [Ernie5_5_MoeDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Ernie5_5_MoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Ernie5_5_MoeRotaryEmbedding(config)
        self.hc_head = Ernie5_5_MoeHyperHead(config)
        self.gradient_checkpointing = False
        # The checkpoint stores the ECHO multi-token-prediction layers right after the decoder layers (e.g.
        # `model.layers.43.*`); they are not used here
        self._keys_to_ignore_on_load_unexpected = [
            rf"(^|\.)layers\.{layer_idx}\."
            for layer_idx in range(
                config.num_hidden_layers, config.num_hidden_layers + config.num_nextn_predict_layers
            )
        ]
        self.post_init()

    @merge_with_config_defaults
    @capture_outputs
    @auto_docstring
    @deprecate_kwarg("past_key_values", "5.40.0", new_name="cache")
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        cache: Cache | None = None,
        inputs_embeds: torch.Tensor | None = None,
        use_cache: bool | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> MoeModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
        if use_cache and cache is None:
            cache = DynamicCache(config=self.config)
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        if position_ids is None:
            past_seen = cache.get_seq_length() if cache is not None else 0
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen
            position_ids = position_ids.unsqueeze(0)

        # `generate()` may pass the per-layer-type masks already built by `create_masks_for_generate`
        if not isinstance(causal_mask_mapping := attention_mask, dict):
            mask_kwargs = {
                "config": self.config,
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "cache": cache,
                "position_ids": position_ids,
            }
            causal_mask_mapping = {
                "heavily_compressed_attention": create_sliding_window_causal_mask(**mask_kwargs),
                "indexed_attention": create_causal_mask(**mask_kwargs, allow_is_causal_skip=False),
            }

        hidden_states = inputs_embeds.unsqueeze(2).expand(-1, -1, self.config.hc_mult, -1).contiguous()
        position_embeddings = {
            "main": self.rotary_emb(inputs_embeds, position_ids=position_ids, layer_type="main"),
            "compress": self.rotary_emb(inputs_embeds, position_ids=position_ids, layer_type="compress"),
        }

        for decoder_layer in self.layers:
            hidden_states = decoder_layer(
                hidden_states,
                position_embeddings=position_embeddings,
                position_ids=position_ids,
                attention_mask=causal_mask_mapping[decoder_layer.attention_type],
                cache=cache,
                **kwargs,
            )

        hidden_states = self.norm(self.hc_head(hidden_states))
        return MoeModelOutputWithPast(last_hidden_state=hidden_states, cache=cache)


class Ernie5_5_MoeLMHead(nn.Linear):
    r"""
    Language-model head followed by MultiMax, a learned piece-wise correction of every logit `x` with ranges `r` and
    scales `t` (four of each): `x + t0 relu(r0 - x) + t1 relu(x - r1) + t2 relu(r2 - x)^2 + t3 relu(x - r3)^2`. With
    `r = t = 0` it is the identity.

    Like the reference trainer, the projection and MultiMax run in fp32 and the logits are returned in fp32; the
    weight keeps the model dtype.
    """

    def __init__(self, hidden_size: int, vocab_size: int, use_multimax: bool = True):
        super().__init__(hidden_size, vocab_size, bias=False)
        self.use_multimax = use_multimax
        self.multimax_ranges = nn.Parameter(torch.zeros(4))
        self.multimax_ts = nn.Parameter(torch.zeros(4))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        logits = F.linear(hidden_states.float(), self.weight.float())
        if not self.use_multimax:
            return logits
        r, t = self.multimax_ranges.float(), self.multimax_ts.float()
        return (
            logits
            + t[0] * F.relu(r[0] - logits)
            + t[1] * F.relu(logits - r[1])
            + t[2] * F.relu(r[2] - logits).square()
            + t[3] * F.relu(logits - r[3]).square()
        )


class Ernie5_5_MoeForCausalLM(MixtralForCausalLM):
    def __init__(self, config: Ernie5_5_MoeConfig):
        Ernie5_5_MoePreTrainedModel.__init__(self, config)
        self.model = Ernie5_5_MoeModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = Ernie5_5_MoeLMHead(config.hidden_size, config.vocab_size, config.multimax == "lm_head")
        self.router_aux_loss_coef = config.router_aux_loss_coef
        self.num_experts = config.n_routed_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        self.post_init()


__all__ = [
    "Ernie5_5_MoePreTrainedModel",
    "Ernie5_5_MoeModel",
    "Ernie5_5_MoeForCausalLM",
]
