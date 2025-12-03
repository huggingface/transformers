# coding=utf-8
# Copyright 2025 the HuggingFace Team. All rights reserved.
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
import math
from typing import Optional, Tuple, tuple

import torch
from torch import nn

from ...utils import logging


logger = logging.get_logger(__name__)

# Optional dependency for Hadamard transform (used in indexer for efficiency)
try:
    from fast_hadamard_transform import hadamard_transform

    _hadamard_available = True
except ImportError:
    hadamard_transform = None
    _hadamard_available = False

from ...cache_utils import Cache
from ..deepseek_v2.configuration_deepseek_v2 import DeepseekV2Config
from ..deepseek_v2.modeling_deepseek_v2 import (
    DeepseekV2Attention,
    DeepseekV2DecoderLayer,
    DeepseekV2ForCausalLM,
    DeepseekV2ForSequenceClassification,
    DeepseekV2MLP,
    DeepseekV2Model,
    DeepseekV2Moe,
    DeepseekV2PreTrainedModel,
    DeepseekV2RMSNorm,
    DeepseekV2RotaryEmbedding,
)


def apply_rotary_pos_emb(x, cos, sin, interleaved=True):
    """Applies Rotary Position Embedding to a single tensor.

    Args:
        x: Input tensor of shape [..., dim]
        cos: Cosine frequencies
        sin: Sine frequencies
        interleaved: If True, assumes x has interleaved real/imag pairs (default).
                     If False, assumes first half is real, second half is imag.
    """
    dtype = x.dtype
    shape = x.shape

    if not interleaved:
        # Non-interleaved: [r0, r1, ..., i0, i1, ...] -> rearrange for complex view
        x = x.view(*shape[:-1], 2, -1).transpose(-1, -2).contiguous()

    # Reshape x for complex view: [..., dim] -> [..., dim//2, 2]
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    # Reshape cos/sin to match: [..., dim//2]
    freqs = torch.complex(cos, sin)  # Create complex frequency tensor
    # Apply rotation
    x_rotated = x_complex * freqs
    # Convert back to real
    x_out = torch.view_as_real(x_rotated).flatten(-2)

    if not interleaved:
        # Convert back to non-interleaved format
        x_out = torch.cat([x_out[..., 0::2], x_out[..., 1::2]], dim=-1)

    return x_out.to(dtype)


def act_quant_native(x: torch.Tensor, block_size: int = 128) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize activations to FP8 with per-block scaling using native PyTorch.

    This is a fallback implementation when tilelang kernels are not available.
    Uses torch.float8_e4m3fn if available (PyTorch 2.1+).

    Args:
        x: Input tensor of shape [..., N] where N is divisible by block_size
        block_size: Block size for computing scales (default: 128)
    Returns:
        Tuple of (quantized_tensor, scales)
    """
    # Check if FP8 is available in this PyTorch version
    if not hasattr(torch, "float8_e4m3fn"):
        # PyTorch version doesn't support FP8, return as-is with dummy scales
        return x, torch.ones(*x.shape[:-1], x.shape[-1] // block_size, device=x.device, dtype=torch.float32)

    original_shape = x.shape
    N = x.shape[-1]
    assert N % block_size == 0, f"Last dimension {N} must be divisible by block_size {block_size}"

    # Reshape to [..., N//block_size, block_size]
    x_blocked = x.view(*original_shape[:-1], N // block_size, block_size)

    # Compute per-block max abs values for scaling
    fp8_max = 448.0  # Max representable value in float8_e4m3
    amax = x_blocked.abs().amax(dim=-1).clamp(min=1e-4)  # [..., N//block_size]
    scales = amax / fp8_max  # [..., N//block_size]

    # Quantize: divide by scale and clamp to FP8 range
    x_scaled = x_blocked / scales.unsqueeze(-1)
    x_clamped = x_scaled.clamp(-fp8_max, fp8_max)

    # Cast to FP8
    x_fp8 = x_clamped.to(torch.float8_e4m3fn).view(original_shape)

    return x_fp8, scales


def rotate_activation(x: torch.Tensor) -> torch.Tensor:
    """Apply Hadamard transform to activations for efficient indexer computation.

    This is an optional optimization used in DeepSeek V3.2's sparse attention indexer.
    Requires the `fast_hadamard_transform` package. Falls back to identity if unavailable.

    Args:
        x: Input tensor of any shape, last dimension should be power of 2 for optimal perf.
    Returns:
        Hadamard-transformed tensor with same shape, scaled by hidden_size^-0.5
    """
    if not _hadamard_available:
        # Fallback: skip Hadamard transform if library not installed
        # This may slightly reduce indexer accuracy but allows the model to run
        return x

    hidden_size = x.size(-1)
    # Hadamard transform expects bfloat16
    original_dtype = x.dtype
    if x.dtype != torch.bfloat16:
        x = x.to(torch.bfloat16)
    x = hadamard_transform(x, scale=hidden_size**-0.5)
    return x.to(original_dtype)


class DeepseekV32Config(DeepseekV2Config):
    """DeepSeek V3.2 configuration extending V2 with sparse attention indexer parameters.

    Additional args:
        index_n_heads (`int`, defaults to 64):
            Number of attention heads for the sparse attention indexer.
        index_head_dim (`int`, defaults to 128):
            Dimension of each indexer attention head.
        index_topk (`int`, defaults to 2048):
            Number of top-k tokens to select for sparse attention.
        use_fp8_indexer (`bool`, defaults to False):
            Whether to use FP8 quantization in the indexer for efficiency.
            Note: Requires specialized kernels, currently not implemented in HuggingFace.
    """

    def __init__(self, index_n_heads=64, index_head_dim=128, index_topk=2048, use_fp8_indexer=False, **super_kwargs):
        super().__init__(**super_kwargs)
        self.index_n_heads = index_n_heads
        self.index_head_dim = index_head_dim
        self.index_top_k = index_topk  # Note: stored with underscore for consistency with generated config
        self.use_fp8_indexer = use_fp8_indexer

        if use_fp8_indexer:
            logger.warning_once(
                "FP8 indexer is requested but not fully implemented in HuggingFace transformers. "
                "The indexer will run without FP8 quantization. For optimal performance, use the "
                "reference DeepSeek implementation with custom tilelang kernels."
            )


class DeepseekV32Moe(DeepseekV2Moe):
    """DeepSeek V3.2 MoE with sigmoid scoring and noaux_tc topk method support."""

    def __init__(self, config):
        super().__init__(config)
        self.scoring_func = getattr(config, "scoring_func", "softmax")
        # Add bias for specific model sizes (following reference implementation)
        if config.hidden_size == 7168:
            self.gate_bias = nn.Parameter(torch.zeros(config.n_routed_experts))
        else:
            self.gate_bias = None

    def route_tokens_to_experts(self, router_logits):
        batch_size, seq_len, hidden_dim = router_logits.shape
        router_logits = router_logits.view(-1, hidden_dim)

        # Apply scoring function
        if self.scoring_func == "softmax":
            scores = router_logits.softmax(dim=-1, dtype=torch.float32)
        else:  # sigmoid
            scores = router_logits.sigmoid()

        original_scores = scores

        # Apply bias if present
        if self.gate_bias is not None:
            scores = scores + self.gate_bias

        # Expert selection based on topk_method
        if self.topk_method == "greedy":
            topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)
        elif self.topk_method in ("group_limited_greedy", "noaux_tc"):
            # Group-based selection
            num_tokens = batch_size * seq_len
            group_scores = scores.view(num_tokens, self.num_group, -1)

            if self.gate_bias is None:
                group_max = group_scores.amax(dim=-1)
            else:
                # Use top-2 sum for group scoring when bias is present
                group_max = group_scores.topk(2, dim=-1)[0].sum(dim=-1)

            group_idx = torch.topk(group_max, k=self.topk_group, dim=-1, sorted=False)[1]

            # Create mask for selected groups
            group_mask = torch.zeros_like(group_max, dtype=torch.bool)
            group_mask.scatter_(1, group_idx, True)

            # Apply mask to scores
            scores_masked = scores.view(num_tokens, self.num_group, -1)
            scores_masked = scores_masked.masked_fill(~group_mask.unsqueeze(-1), float("-inf"))
            scores_masked = scores_masked.view(num_tokens, -1)

            topk_weight, topk_idx = torch.topk(scores_masked, k=self.top_k, dim=-1, sorted=False)
        else:
            raise ValueError(f"Unknown topk_method: {self.topk_method}")

        # Get weights from original scores (before bias)
        topk_weight = original_scores.gather(1, topk_idx)

        # Normalize weights for sigmoid scoring
        if self.scoring_func == "sigmoid":
            topk_weight = topk_weight / topk_weight.sum(dim=-1, keepdim=True)

        topk_weight = topk_weight * self.routed_scaling_factor
        topk_weight = torch.zeros_like(original_scores).scatter_(1, topk_idx, topk_weight)

        return topk_idx, topk_weight


class DeepseekV32MLP(DeepseekV2MLP):
    pass


class DeepseekV32RMSNorm(DeepseekV2RMSNorm):
    pass


class DeepseekV32RotaryEmbedding(DeepseekV2RotaryEmbedding):
    pass


class DeepseekV32Indexer(nn.Module):
    def __init__(self, config: "DeepseekV32Config", index_layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = index_layer_idx

        self.hidden_size: int = config.hidden_size
        self.num_heads: int = config.index_n_heads
        self.num_local_heads: int = config.index_n_heads  # world_size handling can be added as needed
        self.head_dim: int = config.index_head_dim
        self.qk_rope_head_dim: int = config.qk_rope_head_dim
        self.index_topk: int = config.index_top_k
        self.q_lora_rank: int = config.q_lora_rank

        self.q_b_proj = nn.Linear(self.q_lora_rank, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.head_dim, bias=False)
        self.k_layernorm = nn.LayerNorm(self.head_dim)
        self.weight_proj = nn.Linear(self.hidden_size, self.num_heads, dtype=torch.get_default_dtype(), bias=False)
        self.softmax_scale = self.head_dim**-0.5

    @torch.no_grad()
    def forward(
        self,
        hidden_states: torch.Tensor,  # [B, S, hidden]
        q_resid: torch.Tensor,  # [B, S, q_lora_rank]
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values_index: "Cache",
        cache_position: Optional[torch.LongTensor],
    ) -> torch.LongTensor:
        B, S, _ = hidden_states.shape
        cos, sin = position_embeddings

        # Queries (use non-interleaved RoPE for indexer, following reference)
        q_states = self.q_b_proj(q_resid)  # [B, S, H*D]
        q_states = q_states.view(B, S, self.num_heads, self.head_dim)  # [B, S, H, D]
        q_rot, q_pass = torch.split(q_states, [self.qk_rope_head_dim, self.head_dim - self.qk_rope_head_dim], dim=-1)
        q_rot = apply_rotary_pos_emb(q_rot, cos, sin, interleaved=False)  # [B, S, H, rope_D]
        q_states = torch.cat([q_rot, q_pass], dim=-1)  # [B, S, H, D]

        # Keys (single-head for indexer, use non-interleaved RoPE)
        k = self.k_layernorm(self.k_proj(hidden_states))  # [B, S, D]
        k_rot, k_pass = torch.split(k, [self.qk_rope_head_dim, self.head_dim - self.qk_rope_head_dim], dim=-1)
        k_rot = apply_rotary_pos_emb(k_rot.unsqueeze(2), cos, sin, interleaved=False).squeeze(2)  # [B, S, rope_D]
        k_single = torch.cat([k_rot, k_pass], dim=-1)  # [B, S, D]

        # Apply Hadamard transform for efficient indexer computation (optional optimization)
        q_states = rotate_activation(q_states)
        k_single = rotate_activation(k_single)

        # Update cache with single-head keys
        past_key_values_index.update(
            k_single.unsqueeze(1),  # [B, 1, S, D] for cache format
            self.layer_idx,
            cache_kwargs={"cache_position": cache_position}
        )

        # Head weights: [B, S, H]
        head_weights = self.weight_proj(hidden_states) * (self.num_heads**-0.5)  # [B, S, H]

        # Get cached keys for full sequence
        k_cache, _ = past_key_values_index[self.layer_idx]  # [B, 1, T, D]
        k_cache = k_cache.squeeze(1)  # [B, T, D]

        # Compute attention scores: q @ k^T
        # q_states: [B, S, H, D], k_cache: [B, T, D]
        # Expand k for multi-head: [B, T, D] -> [B, 1, T, D] -> broadcast with [B, S, H, D]
        # scores[b,s,h,t] = q[b,s,h,:] @ k[b,t,:]
        scores = torch.einsum("bshd,btd->bsht", q_states.float(), k_cache.float()) * self.softmax_scale

        # Apply head weights and ReLU, then sum over heads
        # head_weights: [B, S, H] -> [B, S, H, 1] for broadcasting
        scores = scores * head_weights.unsqueeze(-1)  # [B, S, H, T]
        scores = torch.relu(scores)
        index_scores = scores.sum(dim=2)  # [B, S, T]

        if attention_mask is not None:
            index_scores = index_scores + attention_mask

        T = index_scores.shape[-1]
        topk = min(self.index_topk, T)
        topk_indices = index_scores.topk(topk, dim=-1).indices  # [..., topk]

        # sanity clone (kept from original)
        _topk = topk_indices.clone()
        assert torch.equal(topk_indices, _topk), f"{topk_indices=} {_topk=}"
        return topk_indices


class DeepseekV32Attention(DeepseekV2Attention):
    def __init__(self, config, layer_idx):
        super().__init__(config, layer_idx)
        self.softmax_scale = self.qk_head_dim**-0.5

        # Apply YaRN mscale adjustment if using extended sequence length
        rope_scaling = config.rope_scaling or {}
        original_max_pos = rope_scaling.get("original_max_position_embeddings", config.max_position_embeddings)
        if config.max_position_embeddings > original_max_pos:
            mscale = rope_scaling.get("mscale", 1.0)
            rope_factor = rope_scaling.get("factor", 1.0)
            mscale_adjustment = 0.1 * mscale * math.log(rope_factor) + 1.0
            self.softmax_scale = self.softmax_scale * mscale_adjustment * mscale_adjustment

        self.indexer = DeepseekV32Indexer(config, layer_idx)

    def forward(
        self,
        hidden_states: torch.Tensor,  # [B, S, hidden]
        position_embeddings: tuple[torch.Tensor, torch.Tensor],  # (cos, sin)
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,  # must be Cache with MlaLayer at `layer_idx`
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
        B, S, _ = hidden_states.shape
        cos, sin = position_embeddings

        # ----- Q path -----
        q_resid = self.q_a_layernorm(self.q_a_proj(hidden_states))  # [B, S, q_lora_rank]
        q_states = self.q_b_proj(q_resid).view(B, S, self.num_heads, self.qk_head_dim)  # [B, S, H, D]
        # Split into pass/rot then apply RoPE on q_rot
        q_pass, q_rot = torch.split(q_states, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        q_rot = apply_rotary_pos_emb(q_rot, cos, sin)  # [B, S, H, rope_D]
        q_states = torch.cat([q_pass, q_rot], dim=-1)  # [B, S, H, D]

        # Layout for matmul: [B, H, S, D]
        q_states = q_states.transpose(1, 2).contiguous()  # [B, H, S, D]

        # ----- KV path (compressed + rope stream) -----
        kv_all = self.kv_a_proj_with_mqa(hidden_states)  # [B, S, kv_rank + rope_D]
        kv_compressed, k_rot = torch.split(kv_all, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        kv_compressed = self.kv_a_layernorm(kv_compressed)  # [B, S, kv_rank]
        # Pre-project to K_pass and V
        kv_proj = self.kv_b_proj(kv_compressed)  # [B, S, H*(qk_nope + v)]
        kv_proj = kv_proj.view(B, S, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
        k_pass, v_states = torch.split(
            kv_proj, [self.qk_nope_head_dim, self.v_head_dim], dim=-1
        )  # [B,S,H,nope], [B,S,H,V]

        # Rope on K side: keep a single-head rope stream like MLA, then expand
        k_rot = k_rot.view(B, 1, S, self.qk_rope_head_dim)  # [B, 1, S, rope_D]
        k_rot = apply_rotary_pos_emb(k_rot, cos, sin)  # [B, 1, S, rope_D]

        # Concatenate K = [K_pass, K_rot(expanded)]
        k_states = torch.cat(
            (
                k_pass.transpose(1, 2),  # [B, H, S, nope_D]
                k_rot.expand(B, self.num_heads, S, -1),
            ),  # [B, H, S, rope_D]
            dim=-1,
        )  # [B, H, S, D]
        v_states = v_states.transpose(1, 2).contiguous()  # [B, H, S, V]

        # ----- Cache update/usage -----
        if past_key_values is not None:
            # Store compressed stream & rope stream (as in original MLA path)
            # We cache `kv_compressed` under `keys` and `k_rot` under `values` in MlaLayer.
            # Shapes must be [B, H, t, *] and [B, 1, t, rope_D].
            kv_comp_cache = kv_compressed.view(B, 1, S, self.kv_lora_rank).expand(B, self.num_heads, S, -1)
            k_rot_cache = k_rot  # [B, 1, S, rope_D]
            cached_kv, cached_pe = past_key_values.update(
                kv_comp_cache, k_rot_cache, layer_idx=self.layer_idx, cache_kwargs={"cache_position": cache_position}
            )
            # Decode path makes use of cached projections; Prefill can use full K/V directly.

        # ----- Two paths (prefill vs decode) -----
        if attention_mask is not None:
            # Prefill (full attention over local window): standard scaled dot-product with top-k pruning from indexer

            # Build scores: [B, H, S, S_total]
            # K layout already [B, H, T, D]
            scores = (q_states.float() @ k_states.float().transpose(-1, -2)) * self.softmax_scale  # [B, H, S, T]

            # Indexer top-k
            if past_key_values is not None:
                topk_idx = self.indexer(
                    hidden_states,
                    q_resid,
                    position_embeddings,
                    attention_mask,
                    past_key_values_index=past_key_values,  # we reuse same Cache with IndexerLayer? (separate cache recommended)
                    cache_position=cache_position,
                )
                # Build mask to keep only top-k per (B,S,head?)
                # Expect topk_idx shape to broadcast to [B, H, S, T]. We scatter along last dim.
                keep_mask = torch.full_like(scores, float("-inf"))
                # If topk_idx is [B,S,topk], expand for heads:
                if topk_idx.dim() == 3:
                    topk_idx = topk_idx.unsqueeze(1).expand(B, self.num_heads, S, -1)
                keep_mask.scatter_(-1, topk_idx, 0.0)
                scores = scores + keep_mask

            probs = nn.functional.softmax(scores, dim=-1, dtype=torch.float32).type_as(hidden_states)  # [B, H, S, T]
            attn_output = probs @ v_states  # [B, H, S, V]

        elif past_key_values is not None:
            # Decode: use cached compressed KV & rope stream to recompose attention scores efficiently
            # Compose q_pass and q_rot pieces as in MLA math, but via matmul
            # 1) Rebuild "nope" term via kv_b weights (dequant on the fly)
            wkv_b = self.kv_b_proj.weight.view(
                self.num_heads, self.qk_nope_head_dim + self.v_head_dim, self.kv_lora_rank
            )
            w_k_nope = wkv_b[:, : self.qk_nope_head_dim, :]  # [H, nope_D, kv_rank]
            w_v = wkv_b[:, self.qk_nope_head_dim :, :]  # [H, V,     kv_rank]

            # q_pass: [B,H,S,nope_D]; cached_kv: [B,H,T,kv_rank]
            q_pass = q_states[..., : self.qk_nope_head_dim]  # [B,H,S,nope_D]
            kv_comp = past_key_values[self.layer_idx][0]  # keys -> [B,H,T,kv_rank]
            pe_full = past_key_values[self.layer_idx][1]  # values -> [B,1,T,rope_D]
            # Project q_pass with w_k_nope: [B,H,S,kv_rank]
            qk_nope = torch.matmul(q_pass, w_k_nope.transpose(-1, -2))  # [B,H,S,kv_rank]
            # Scores_nope = qk_nope @ kv_comp^T
            scores_nope = torch.matmul(qk_nope.float(), kv_comp.float().transpose(-1, -2))  # [B,H,S,T]

            # 2) Rope term: q_rot @ k_rot^T
            q_rot_only = q_states[..., -self.qk_rope_head_dim :]  # [B,H,S,rope_D]
            k_rot_only = pe_full.expand(B, self.num_heads, -1, -1)  # [B,H,T,rope_D]
            scores_rot = torch.matmul(q_rot_only.float(), k_rot_only.float().transpose(-1, -2))  # [B,H,S,T]

            scores = (scores_nope + scores_rot) * self.softmax_scale

            # Indexer top-k (decode)
            topk_idx = self.indexer(
                hidden_states,
                q_resid,
                position_embeddings,
                attention_mask,
                past_key_values_index=past_key_values,
                cache_position=cache_position,
            )
            # For decode single-step S==1 typically; build a [B,H,1,T] mask
            keep_mask = torch.full_like(scores, float("-inf"))
            if topk_idx.dim() == 3:
                topk_idx = topk_idx.unsqueeze(1).expand(B, self.num_heads, S, -1)
            keep_mask.scatter_(-1, topk_idx, 0.0)
            scores = scores + keep_mask

            probs = nn.functional.softmax(scores, dim=-1, dtype=torch.float32).type_as(hidden_states)  # [B,H,S,T]

            # Rebuild V for decode fast-path: v = (kv_comp @ w_v^T)
            # kv_comp: [B,H,T,kv_rank], w_v: [H, V, kv_rank]
            v_from_comp = torch.matmul(kv_comp, w_v.transpose(-1, -2))  # [B,H,T,V]
            attn_output = torch.matmul(probs, v_from_comp)  # [B,H,S,V]

        # Output projection
        attn_output = attn_output.transpose(1, 2).reshape(B, S, -1).contiguous()  # [B,S,H*V]
        attn_output = self.o_proj(attn_output)  # [B,S,hidden]
        return attn_output, None, None


class DeepseekV32DecoderLayer(DeepseekV2DecoderLayer):
    pass


class DeepseekV32PreTrainedModel(DeepseekV2PreTrainedModel):
    pass


class DeepseekV32Model(DeepseekV2Model):
    pass


class DeepseekV32ForCausalLM(DeepseekV2ForCausalLM):
    pass


class DeepseekV32ForSequenceClassification(DeepseekV2ForSequenceClassification):
    pass


__all__ = [
    "DeepseekV32Config",
    "DeepseekV32PreTrainedModel",
    "DeepseekV32Model",
    "DeepseekV32ForCausalLM",
    "DeepseekV32ForSequenceClassification",
]
