"""
Test script to compare HF DeepSeek V3.2 implementation against reference implementation.

This script creates small test models and compares outputs at various stages to identify
numerical differences between the implementations.

Usage:
    python test_numerical_equivalence.py [--device cuda|cpu] [--verbose]
"""

import argparse
import sys
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# Add parent directory to path for imports
sys.path.insert(0, "/home/jeffreyliu/transformers/src")

# Check for hadamard transform availability BEFORE importing HF module
HADAMARD_AVAILABLE = False
try:
    from fast_hadamard_transform import hadamard_transform as _cuda_hadamard
    HADAMARD_AVAILABLE = True
except ImportError:
    pass


def hadamard_transform_activation(x: torch.Tensor) -> torch.Tensor:
    """
    Hadamard transform with CPU fallback for testing.

    The CPU fallback uses scipy's Hadamard matrix, which is slower but correct.
    """
    if HADAMARD_AVAILABLE and x.is_cuda:
        hidden_size = x.size(-1)
        original_dtype = x.dtype
        if x.dtype != torch.bfloat16:
            x = x.to(torch.bfloat16)
        x = _cuda_hadamard(x, scale=hidden_size**-0.5)
        return x.to(original_dtype)
    else:
        # CPU fallback using scipy
        try:
            from scipy.linalg import hadamard
        except ImportError:
            raise ImportError("scipy is required for CPU hadamard transform testing")

        hidden_size = x.size(-1)
        # Hadamard matrix must be power of 2
        assert hidden_size & (hidden_size - 1) == 0, f"hidden_size must be power of 2, got {hidden_size}"

        H = torch.tensor(hadamard(hidden_size), dtype=x.dtype, device=x.device) * (hidden_size ** -0.5)
        return x @ H


# Monkey-patch the HF module to use our CPU-compatible hadamard transform
import transformers.models.deepseek_v32.modular_deepseek_v32 as hf_module


hf_module.hadamard_transform_activation = hadamard_transform_activation

from transformers.models.deepseek_v32.modular_deepseek_v32 import (
    DeepseekV32Attention,
    DeepseekV32Config,
    DeepseekV32Indexer,
    DeepseekV32RMSNorm,
)
from transformers.models.deepseek_v32.modular_deepseek_v32 import (
    apply_rotary_emb as hf_apply_rotary_emb,
)


# =============================================================================
# Reference Implementation (simplified, no FP8, no distributed)
# =============================================================================

@dataclass
class RefModelArgs:
    """Reference model arguments matching the reference implementation."""
    vocab_size: int = 1024
    dim: int = 256
    inter_dim: int = 512
    n_layers: int = 2
    n_heads: int = 4
    q_lora_rank: int = 64
    kv_lora_rank: int = 32
    qk_nope_head_dim: int = 32
    qk_rope_head_dim: int = 16
    v_head_dim: int = 32
    original_seq_len: int = 4096
    max_seq_len: int = 4096
    rope_theta: float = 10000.0
    rope_factor: float = 1.0
    beta_fast: int = 32
    beta_slow: int = 1
    mscale: float = 1.0
    index_n_heads: int = 4
    index_head_dim: int = 32
    index_topk: int = 64


def ref_precompute_freqs_cis(args: RefModelArgs) -> torch.Tensor:
    """Reference implementation of frequency computation."""
    dim = args.qk_rope_head_dim
    seqlen = args.max_seq_len
    base = args.rope_theta

    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    t = torch.arange(seqlen)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def ref_apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor, interleaved: bool = True) -> torch.Tensor:
    """Reference implementation of rotary embeddings."""
    dtype = x.dtype
    shape = x.shape
    if not interleaved:
        x = x.view(*shape[:-1], 2, -1).transpose(-1, -2).contiguous()
    x = torch.view_as_complex(x.float().view(*shape[:-1], -1, 2))
    freqs_cis = freqs_cis.view(1, x.size(1), 1, x.size(-1))
    y = torch.view_as_real(x * freqs_cis).flatten(3)
    if not interleaved:
        y = torch.cat([y[..., 0::2], y[..., 1::2]], dim=-1)
    return y.to(dtype)


class RefRMSNorm(nn.Module):
    """Reference RMSNorm implementation."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        var = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        return (self.weight * x).to(dtype)


class RefLayerNorm(nn.Module):
    """Reference LayerNorm for indexer."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.bias = nn.Parameter(torch.zeros(dim, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # F.layer_norm requires weight/bias to match input dtype on CPU
        return F.layer_norm(
            x.float(), (self.dim,), self.weight.float(), self.bias.float(), self.eps
        ).type_as(x)


class RefIndexer(nn.Module):
    """Reference Indexer implementation (without FP8)."""
    def __init__(self, args: RefModelArgs):
        super().__init__()
        self.dim = args.dim
        self.n_heads = args.index_n_heads
        self.head_dim = args.index_head_dim
        self.rope_head_dim = args.qk_rope_head_dim
        self.index_topk = args.index_topk
        self.q_lora_rank = args.q_lora_rank

        self.wq_b = nn.Linear(self.q_lora_rank, self.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(self.dim, self.head_dim, bias=False)
        self.k_norm = RefLayerNorm(self.head_dim)
        self.weights_proj = nn.Linear(self.dim, self.n_heads, bias=False)
        self.softmax_scale = self.head_dim ** -0.5

        # Cache for decode
        self._k_cache = None

    def reset_cache(self):
        self._k_cache = None

    def forward(
        self,
        x: torch.Tensor,
        qr: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        cache_position: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        bsz, seqlen, _ = x.size()

        q = self.wq_b(qr)
        q = q.view(bsz, seqlen, self.n_heads, self.head_dim)

        # Split: rope first, then nope
        q_pe, q_nope = torch.split(q, [self.rope_head_dim, self.head_dim - self.rope_head_dim], dim=-1)
        q_pe = ref_apply_rotary_emb(q_pe, freqs_cis, interleaved=False)
        q = torch.cat([q_pe, q_nope], dim=-1)

        k = self.wk(x)
        k = self.k_norm(k)
        k_pe, k_nope = torch.split(k, [self.rope_head_dim, self.head_dim - self.rope_head_dim], dim=-1)
        k_pe = ref_apply_rotary_emb(k_pe.unsqueeze(2), freqs_cis, interleaved=False).squeeze(2)
        k = torch.cat([k_pe, k_nope], dim=-1)

        # Hadamard transform
        q = hadamard_transform_activation(q)
        k = hadamard_transform_activation(k)

        # Update cache
        if cache_position is not None:
            start_pos = cache_position[0].item()
            end_pos = cache_position[-1].item() + 1

            if self._k_cache is None or start_pos == 0:
                if seqlen == end_pos:
                    self._k_cache = k.clone()
                else:
                    self._k_cache = torch.zeros(bsz, end_pos, self.head_dim, dtype=k.dtype, device=k.device)
                    self._k_cache[:, start_pos:end_pos] = k
            else:
                current_len = self._k_cache.shape[1]
                if end_pos > current_len:
                    new_cache = torch.zeros(bsz, end_pos, self.head_dim, dtype=k.dtype, device=k.device)
                    new_cache[:, :current_len] = self._k_cache
                    self._k_cache = new_cache
                self._k_cache[:, start_pos:end_pos] = k

            k_full = self._k_cache[:, :end_pos]
        else:
            k_full = k

        # Compute scores (non-FP8 path)
        weights = F.linear(x.float(), self.weights_proj.weight.float()) * (self.n_heads ** -0.5)

        # q: [B, S, H, D], k_full: [B, T, D]
        scores = torch.einsum("bshd,btd->bsht", q.float(), k_full.float()) * self.softmax_scale
        scores = torch.relu(scores)
        scores = scores * weights.unsqueeze(-1)
        index_scores = scores.sum(dim=2)  # [B, S, T]

        if mask is not None:
            if mask.dim() == 4:
                mask = mask.squeeze(1)
            index_scores = index_scores + mask

        topk = min(self.index_topk, index_scores.shape[-1])
        topk_indices = index_scores.topk(topk, dim=-1).indices

        return topk_indices


class RefMLA(nn.Module):
    """Reference MLA implementation (without FP8)."""
    def __init__(self, args: RefModelArgs):
        super().__init__()
        self.dim = args.dim
        self.n_heads = args.n_heads
        self.q_lora_rank = args.q_lora_rank
        self.kv_lora_rank = args.kv_lora_rank
        self.qk_nope_head_dim = args.qk_nope_head_dim
        self.qk_rope_head_dim = args.qk_rope_head_dim
        self.qk_head_dim = args.qk_nope_head_dim + args.qk_rope_head_dim
        self.v_head_dim = args.v_head_dim

        self.wq_a = nn.Linear(self.dim, self.q_lora_rank, bias=False)
        self.q_norm = RefRMSNorm(self.q_lora_rank)
        self.wq_b = nn.Linear(self.q_lora_rank, self.n_heads * self.qk_head_dim, bias=False)

        self.wkv_a = nn.Linear(self.dim, self.kv_lora_rank + self.qk_rope_head_dim, bias=False)
        self.kv_norm = RefRMSNorm(self.kv_lora_rank)
        self.wkv_b = nn.Linear(self.kv_lora_rank, self.n_heads * (self.qk_nope_head_dim + self.v_head_dim), bias=False)

        self.wo = nn.Linear(self.n_heads * self.v_head_dim, self.dim, bias=False)

        self.softmax_scale = self.qk_head_dim ** -0.5

        self.indexer = RefIndexer(args)

        # KV cache
        self._k_cache = None
        self._v_cache = None

    def reset_cache(self):
        self._k_cache = None
        self._v_cache = None
        self.indexer.reset_cache()

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        cache_position: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        bsz, seqlen, _ = x.size()

        # Q path
        qr = self.q_norm(self.wq_a(x))
        q = self.wq_b(qr)
        q = q.view(bsz, seqlen, self.n_heads, self.qk_head_dim)

        # Split: nope first, then rope (different from indexer!)
        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        q_pe = ref_apply_rotary_emb(q_pe, freqs_cis, interleaved=True)

        # KV path
        kv = self.wkv_a(x)
        kv, k_pe = torch.split(kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        kv = self.kv_norm(kv)
        k_pe = ref_apply_rotary_emb(k_pe.unsqueeze(2), freqs_cis, interleaved=True)

        # Project KV
        kv_proj = self.wkv_b(kv)
        kv_proj = kv_proj.view(bsz, seqlen, self.n_heads, self.qk_nope_head_dim + self.v_head_dim)
        k_nope, v = torch.split(kv_proj, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)

        # Combine Q and K
        q = torch.cat([q_nope, q_pe], dim=-1)
        k = torch.cat([k_nope, k_pe.expand(-1, -1, self.n_heads, -1)], dim=-1)

        # Transpose for attention
        q = q.transpose(1, 2)  # [B, H, S, D]
        k = k.transpose(1, 2)  # [B, H, S, D]
        v = v.transpose(1, 2)  # [B, H, S, D]

        # Update KV cache
        if cache_position is not None:
            start_pos = cache_position[0].item()
            end_pos = cache_position[-1].item() + 1

            if self._k_cache is None or start_pos == 0:
                self._k_cache = k.clone()
                self._v_cache = v.clone()
            else:
                self._k_cache = torch.cat([self._k_cache, k], dim=2)
                self._v_cache = torch.cat([self._v_cache, v], dim=2)

            k = self._k_cache
            v = self._v_cache

        # Attention scores
        scores = torch.matmul(q.float(), k.float().transpose(-1, -2)) * self.softmax_scale

        # Apply causal mask
        if mask is not None:
            if mask.dim() == 4:
                scores = scores + mask[:, :, :, :k.shape[-2]]
            else:
                scores = scores + mask[:, :k.shape[-2]]

        # Apply indexer
        topk_indices = self.indexer(x, qr, freqs_cis, mask, cache_position)
        index_mask = torch.full(
            (bsz, seqlen, k.shape[-2]),
            float("-inf"),
            device=x.device,
            dtype=scores.dtype,
        )
        index_mask.scatter_(-1, topk_indices, 0.0)
        scores = scores + index_mask.unsqueeze(1)

        # Softmax and output
        scores = F.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
        out = torch.matmul(scores, v)

        out = out.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        out = self.wo(out)

        return out


# =============================================================================
# Test Functions
# =============================================================================

def create_hf_config(args: RefModelArgs) -> DeepseekV32Config:
    """Create HF config matching reference args."""
    return DeepseekV32Config(
        vocab_size=args.vocab_size,
        hidden_size=args.dim,
        intermediate_size=args.inter_dim,
        num_hidden_layers=args.n_layers,
        num_attention_heads=args.n_heads,
        q_lora_rank=args.q_lora_rank,
        kv_lora_rank=args.kv_lora_rank,
        qk_nope_head_dim=args.qk_nope_head_dim,
        qk_rope_head_dim=args.qk_rope_head_dim,
        v_head_dim=args.v_head_dim,
        max_position_embeddings=args.max_seq_len,
        rope_theta=args.rope_theta,
        index_n_heads=args.index_n_heads,
        index_head_dim=args.index_head_dim,
        index_topk=args.index_topk,
        rms_norm_eps=1e-6,
        attention_bias=False,
        attention_dropout=0.0,
    )


def copy_indexer_weights(ref_indexer: RefIndexer, hf_indexer: DeepseekV32Indexer):
    """Copy weights from reference indexer to HF indexer."""
    hf_indexer.wq_b.weight.data.copy_(ref_indexer.wq_b.weight.data)
    hf_indexer.wk.weight.data.copy_(ref_indexer.wk.weight.data)
    hf_indexer.k_norm.weight.data.copy_(ref_indexer.k_norm.weight.data)
    hf_indexer.k_norm.bias.data.copy_(ref_indexer.k_norm.bias.data)
    hf_indexer.weights_proj.weight.data.copy_(ref_indexer.weights_proj.weight.data)


def copy_mla_weights(ref_mla: RefMLA, hf_attn: DeepseekV32Attention):
    """Copy weights from reference MLA to HF attention."""
    hf_attn.q_a_proj.weight.data.copy_(ref_mla.wq_a.weight.data)
    hf_attn.q_a_layernorm.weight.data.copy_(ref_mla.q_norm.weight.data)
    hf_attn.q_b_proj.weight.data.copy_(ref_mla.wq_b.weight.data)
    hf_attn.kv_a_proj_with_mqa.weight.data.copy_(ref_mla.wkv_a.weight.data)
    hf_attn.kv_a_layernorm.weight.data.copy_(ref_mla.kv_norm.weight.data)
    hf_attn.kv_b_proj.weight.data.copy_(ref_mla.wkv_b.weight.data)
    hf_attn.o_proj.weight.data.copy_(ref_mla.wo.weight.data)
    copy_indexer_weights(ref_mla.indexer, hf_attn.indexer)


def compare_tensors(name: str, ref: torch.Tensor, hf: torch.Tensor, rtol: float = 1e-4, atol: float = 1e-5) -> bool:
    """Compare two tensors and report differences."""
    if ref.shape != hf.shape:
        print(f"  {name}: SHAPE MISMATCH - ref {ref.shape} vs hf {hf.shape}")
        return False

    # Convert to same dtype for comparison
    ref_f = ref.float()
    hf_f = hf.float()

    max_diff = (ref_f - hf_f).abs().max().item()
    mean_diff = (ref_f - hf_f).abs().mean().item()
    is_close = torch.allclose(ref_f, hf_f, rtol=rtol, atol=atol)

    status = "✓ PASS" if is_close else "✗ FAIL"
    print(f"  {name}: {status} (max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e})")

    return is_close


def test_rotary_embedding(device: str, verbose: bool = False):
    """Test rotary embedding implementations."""
    print("\n" + "="*60)
    print("Testing Rotary Embeddings")
    print("="*60)

    args = RefModelArgs()

    # Create input
    batch_size, seq_len = 2, 16
    head_dim = args.qk_rope_head_dim
    x = torch.randn(batch_size, seq_len, 4, head_dim, device=device, dtype=torch.bfloat16)

    # Compute frequencies
    ref_freqs = ref_precompute_freqs_cis(args).to(device)[:seq_len]

    # Test interleaved=True
    print("\nInterleaved=True:")
    ref_out = ref_apply_rotary_emb(x, ref_freqs, interleaved=True)
    hf_out = hf_apply_rotary_emb(x, ref_freqs, interleaved=True)
    compare_tensors("RoPE output", ref_out, hf_out)

    # Test interleaved=False
    print("\nInterleaved=False:")
    ref_out = ref_apply_rotary_emb(x, ref_freqs, interleaved=False)
    hf_out = hf_apply_rotary_emb(x, ref_freqs, interleaved=False)
    compare_tensors("RoPE output", ref_out, hf_out)


def test_indexer(device: str, verbose: bool = False):
    """Test indexer implementations."""
    print("\n" + "="*60)
    print("Testing Indexer")
    print("="*60)

    args = RefModelArgs()
    config = create_hf_config(args)

    # Create models
    ref_indexer = RefIndexer(args).to(device).to(torch.bfloat16)
    hf_indexer = DeepseekV32Indexer(config, layer_idx=0).to(device).to(torch.bfloat16)

    # Copy weights
    copy_indexer_weights(ref_indexer, hf_indexer)

    # Create inputs
    batch_size, seq_len = 2, 32
    x = torch.randn(batch_size, seq_len, args.dim, device=device, dtype=torch.bfloat16)
    qr = torch.randn(batch_size, seq_len, args.q_lora_rank, device=device, dtype=torch.bfloat16)
    freqs_cis = ref_precompute_freqs_cis(args).to(device)[:seq_len]

    # Create causal mask
    mask = torch.full((seq_len, seq_len), float("-inf"), device=device)
    mask = torch.triu(mask, diagonal=1)
    mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, S, S]

    cache_position = torch.arange(seq_len, device=device)

    # Test prefill
    print("\nPrefill (seq_len > 1):")
    ref_indexer.reset_cache()
    hf_indexer.reset_cache()

    with torch.no_grad():
        ref_indices = ref_indexer(x, qr, freqs_cis, mask, cache_position)
        hf_indices = hf_indexer(x, qr, freqs_cis, mask.squeeze(1), cache_position)

    # Compare indices (they should be identical since we're selecting top-k)
    indices_match = torch.equal(ref_indices.sort(dim=-1).values, hf_indices.sort(dim=-1).values)
    print(f"  Top-k indices match: {'✓ PASS' if indices_match else '✗ FAIL'}")

    if not indices_match and verbose:
        print(f"    Ref indices sample: {ref_indices[0, 0, :10].tolist()}")
        print(f"    HF indices sample: {hf_indices[0, 0, :10].tolist()}")

    # Test decode
    print("\nDecode (seq_len == 1):")
    x_decode = torch.randn(batch_size, 1, args.dim, device=device, dtype=torch.bfloat16)
    qr_decode = torch.randn(batch_size, 1, args.q_lora_rank, device=device, dtype=torch.bfloat16)
    freqs_cis_decode = ref_precompute_freqs_cis(args).to(device)[seq_len:seq_len+1]
    cache_position_decode = torch.tensor([seq_len], device=device)

    with torch.no_grad():
        ref_indices_decode = ref_indexer(x_decode, qr_decode, freqs_cis_decode, None, cache_position_decode)
        hf_indices_decode = hf_indexer(x_decode, qr_decode, freqs_cis_decode, None, cache_position_decode)

    indices_match_decode = torch.equal(
        ref_indices_decode.sort(dim=-1).values,
        hf_indices_decode.sort(dim=-1).values
    )
    print(f"  Top-k indices match: {'✓ PASS' if indices_match_decode else '✗ FAIL'}")


def test_attention(device: str, verbose: bool = False):
    """Test attention implementations."""
    print("\n" + "="*60)
    print("Testing MLA Attention")
    print("="*60)

    from transformers.cache_utils import DynamicCache

    args = RefModelArgs()
    config = create_hf_config(args)

    # Create models
    ref_mla = RefMLA(args).to(device).to(torch.bfloat16)
    hf_attn = DeepseekV32Attention(config, layer_idx=0).to(device).to(torch.bfloat16)

    # Copy weights
    copy_mla_weights(ref_mla, hf_attn)

    # Create inputs
    batch_size, seq_len = 2, 32
    x = torch.randn(batch_size, seq_len, args.dim, device=device, dtype=torch.bfloat16)
    freqs_cis = ref_precompute_freqs_cis(args).to(device)[:seq_len]

    # Create causal mask
    mask = torch.full((seq_len, seq_len), float("-inf"), device=device, dtype=torch.bfloat16)
    mask = torch.triu(mask, diagonal=1)
    mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, S, S]

    cache_position = torch.arange(seq_len, device=device)

    # Test prefill
    print("\nPrefill:")
    ref_mla.reset_cache()
    hf_attn.indexer.reset_cache()

    # Create HF cache for prefill
    hf_cache = DynamicCache()

    with torch.no_grad():
        ref_out = ref_mla(x, freqs_cis, mask, cache_position)
        hf_out, _, hf_cache = hf_attn(x, freqs_cis, mask, hf_cache, False, cache_position)

    compare_tensors("Attention output", ref_out, hf_out, rtol=1e-3, atol=1e-4)

    # Test decode
    print("\nDecode:")
    x_decode = torch.randn(batch_size, 1, args.dim, device=device, dtype=torch.bfloat16)
    freqs_cis_decode = ref_precompute_freqs_cis(args).to(device)[seq_len:seq_len+1]
    cache_position_decode = torch.tensor([seq_len], device=device)

    # For decode, mask should cover all positions (prefill + current)
    mask_decode = torch.zeros(batch_size, 1, 1, seq_len + 1, device=device, dtype=torch.bfloat16)

    with torch.no_grad():
        ref_out_decode = ref_mla(x_decode, freqs_cis_decode, mask_decode, cache_position_decode)
        hf_out_decode, _, _ = hf_attn(x_decode, freqs_cis_decode, mask_decode, hf_cache, False, cache_position_decode)

    compare_tensors("Attention output (decode)", ref_out_decode, hf_out_decode, rtol=1e-3, atol=1e-4)


def test_rms_norm(device: str, verbose: bool = False):
    """Test RMSNorm implementations."""
    print("\n" + "="*60)
    print("Testing RMSNorm")
    print("="*60)

    dim = 256
    ref_norm = RefRMSNorm(dim).to(device)
    hf_norm = DeepseekV32RMSNorm(dim).to(device)

    # Copy weights
    hf_norm.weight.data.copy_(ref_norm.weight.data)

    # Test
    x = torch.randn(2, 32, dim, device=device, dtype=torch.bfloat16)

    with torch.no_grad():
        ref_out = ref_norm(x)
        hf_out = hf_norm(x)

    compare_tensors("RMSNorm output", ref_out, hf_out)


def main():
    parser = argparse.ArgumentParser(description="Test numerical equivalence between HF and reference implementations")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--verbose", action="store_true", help="Print detailed output")
    args = parser.parse_args()

    print(f"Running tests on device: {args.device}")
    print(f"PyTorch version: {torch.__version__}")

    # Check for hadamard transform
    if HADAMARD_AVAILABLE:
        print("fast_hadamard_transform: Available (CUDA)")
    else:
        try:
            from scipy.linalg import hadamard
            print("fast_hadamard_transform: NOT AVAILABLE - using scipy CPU fallback")
        except ImportError:
            print("fast_hadamard_transform: NOT AVAILABLE - scipy also not available!")
            print("Install scipy for CPU fallback: pip install scipy")

    # Run tests
    test_rms_norm(args.device, args.verbose)
    test_rotary_embedding(args.device, args.verbose)

    try:
        test_indexer(args.device, args.verbose)
        test_attention(args.device, args.verbose)
    except ImportError as e:
        print(f"\nSkipping indexer/attention tests: {e}")

    print("\n" + "="*60)
    print("Tests completed!")
    print("="*60)


if __name__ == "__main__":
    main()
