#!/usr/bin/env python3
"""
Reference Comparison Tests for DeepSeek V3.2 Implementation.

This test suite compares the HuggingFace Transformers implementation of DeepSeek V3.2
against the official reference implementation from DeepSeek to ensure numerical equivalence.

Reference: deepseek_files/inference/model.py

Key components tested:
1. RoPE (Rotary Position Embeddings) - both interleaved and non-interleaved
2. RMSNorm / LayerNorm
3. MLP and Expert layers
4. Gate routing mechanism
5. MLA (Multi-head Latent Attention)
6. Indexer (Lightning Indexer for DSA)
7. YaRN scaling for extended context

Usage:
    pytest src/transformers/models/deepseek_v32/test_reference_comparison.py -v
    python src/transformers/models/deepseek_v32/test_reference_comparison.py
"""

import math
import sys
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn


# =============================================================================
# Reference Implementation Components (copied/adapted from model.py)
# =============================================================================


@dataclass
class ReferenceModelArgs:
    """Reference model arguments matching deepseek_files/inference/model.py."""

    max_batch_size: int = 8
    max_seq_len: int = 4096 * 4
    dtype: Literal["bf16", "fp8"] = "bf16"
    vocab_size: int = 102400
    dim: int = 2048
    inter_dim: int = 10944
    moe_inter_dim: int = 1408
    n_layers: int = 27
    n_dense_layers: int = 1
    n_heads: int = 16
    # moe
    n_routed_experts: int = 64
    n_shared_experts: int = 2
    n_activated_experts: int = 6
    n_expert_groups: int = 1
    n_limited_groups: int = 1
    score_func: Literal["softmax", "sigmoid"] = "softmax"
    route_scale: float = 1.0
    # mla
    q_lora_rank: int = 0
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128
    # yarn
    original_seq_len: int = 4096
    rope_theta: float = 10000.0
    rope_factor: float = 40
    beta_fast: int = 32
    beta_slow: int = 1
    mscale: float = 1.0
    # index
    index_n_heads: int = 64
    index_head_dim: int = 128
    index_topk: int = 2048


def reference_precompute_freqs_cis(args: ReferenceModelArgs) -> torch.Tensor:
    """
    Reference implementation of precompute_freqs_cis from model.py lines 341-419.
    """
    dim = args.qk_rope_head_dim
    seqlen = args.max_seq_len
    beta_fast = args.beta_fast
    beta_slow = args.beta_slow
    base = args.rope_theta
    factor = args.rope_factor

    def find_correction_dim(num_rotations, dim, base, max_seq_len):
        return dim * math.log(max_seq_len / (num_rotations * 2 * math.pi)) / (2 * math.log(base))

    def find_correction_range(low_rot, high_rot, dim, base, max_seq_len):
        low = math.floor(find_correction_dim(low_rot, dim, base, max_seq_len))
        high = math.ceil(find_correction_dim(high_rot, dim, base, max_seq_len))
        return max(low, 0), min(high, dim - 1)

    def linear_ramp_factor(min_val, max_val, dim):
        if min_val == max_val:
            max_val += 0.001
        linear_func = (torch.arange(dim, dtype=torch.float32) - min_val) / (max_val - min_val)
        ramp_func = torch.clamp(linear_func, 0, 1)
        return ramp_func

    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    if seqlen > args.original_seq_len:
        low, high = find_correction_range(beta_fast, beta_slow, dim, base, args.original_seq_len)
        smooth = 1 - linear_ramp_factor(low, high, dim // 2)
        freqs = freqs / factor * (1 - smooth) + freqs * smooth

    t = torch.arange(seqlen)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def reference_apply_rotary_emb(
    x: torch.Tensor, freqs_cis: torch.Tensor, interleaved: bool = True
) -> torch.Tensor:
    """
    Reference implementation of apply_rotary_emb from model.py lines 422-442.
    """
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


class ReferenceRMSNorm(nn.Module):
    """Reference RMSNorm from model.py lines 287-322."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.float32))

    def forward(self, x: torch.Tensor):
        dtype = x.dtype
        x = x.float()
        var = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        return (self.weight * x).to(dtype)


class ReferenceLayerNorm(nn.Module):
    """Reference LayerNorm from model.py lines 325-338."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.bias = nn.Parameter(torch.zeros(dim, dtype=torch.float32))

    def forward(self, x: torch.Tensor):
        return F.layer_norm(x.float(), (self.dim,), self.weight, self.bias, self.eps).type_as(x)


class ReferenceGate(nn.Module):
    """Reference Gate from model.py lines 700-764."""

    def __init__(self, args: ReferenceModelArgs):
        super().__init__()
        self.dim = args.dim
        self.topk = args.n_activated_experts
        self.n_groups = args.n_expert_groups
        self.topk_groups = args.n_limited_groups
        self.score_func = args.score_func
        self.route_scale = args.route_scale
        self.weight = nn.Parameter(torch.empty(args.n_routed_experts, args.dim))
        self.bias = (
            nn.Parameter(torch.empty(args.n_routed_experts, dtype=torch.float32))
            if self.dim == 7168
            else None
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scores = F.linear(x.float(), self.weight.float())
        if self.score_func == "softmax":
            scores = scores.softmax(dim=-1)
        else:
            scores = scores.sigmoid()
        original_scores = scores
        if self.bias is not None:
            scores = scores + self.bias
        if self.n_groups > 1:
            scores = scores.view(x.size(0), self.n_groups, -1)
            if self.bias is None:
                group_scores = scores.amax(dim=-1)
            else:
                group_scores = scores.topk(2, dim=-1)[0].sum(dim=-1)
            indices = group_scores.topk(self.topk_groups, dim=-1)[1]
            mask = scores.new_ones(x.size(0), self.n_groups, dtype=bool).scatter_(1, indices, False)
            scores = scores.masked_fill_(mask.unsqueeze(-1), float("-inf")).flatten(1)
        indices = scores.topk(self.topk, dim=-1)[1]
        weights = original_scores.gather(1, indices)
        if self.score_func == "sigmoid":
            weights /= weights.sum(dim=-1, keepdim=True)
        weights *= self.route_scale
        return weights, indices


class ReferenceExpert(nn.Module):
    """Reference Expert from model.py lines 767-800."""

    def __init__(self, dim: int, inter_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, inter_dim, bias=False)
        self.w2 = nn.Linear(inter_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, inter_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2((F.silu(self.w1(x).float()) * self.w3(x).float()).type_as(x))


class ReferenceMLP(nn.Module):
    """Reference MLP from model.py lines 664-697."""

    def __init__(self, dim: int, inter_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, inter_dim, bias=False)
        self.w2 = nn.Linear(inter_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, inter_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2((F.silu(self.w1(x).float()) * self.w3(x).float()).type_as(x))


# =============================================================================
# Test Utilities
# =============================================================================


def get_small_test_config():
    """Create a small config for testing (matches test_deepseek_v32.py)."""
    from transformers.models.deepseek_v32.configuration_deepseek_v32 import DeepseekV32Config

    return DeepseekV32Config(
        vocab_size=1000,
        hidden_size=256,
        intermediate_size=512,
        moe_intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        q_lora_rank=64,
        kv_lora_rank=32,
        qk_nope_head_dim=32,
        qk_rope_head_dim=16,
        v_head_dim=32,
        n_routed_experts=4,
        n_shared_experts=1,
        num_experts_per_tok=2,
        n_group=2,
        topk_group=1,
        index_n_heads=4,
        index_head_dim=32,
        index_topk=128,
        first_k_dense_replace=1,
    )


def get_reference_args_from_config(config):
    """Convert HuggingFace config to reference ModelArgs."""
    return ReferenceModelArgs(
        dim=config.hidden_size,
        inter_dim=config.intermediate_size,
        moe_inter_dim=config.moe_intermediate_size,
        n_heads=config.num_attention_heads,
        n_routed_experts=config.n_routed_experts,
        n_activated_experts=config.num_experts_per_tok,
        n_expert_groups=config.n_group,
        n_limited_groups=config.topk_group,
        score_func=config.scoring_func,
        route_scale=config.routed_scaling_factor,
        q_lora_rank=config.q_lora_rank,
        kv_lora_rank=config.kv_lora_rank,
        qk_nope_head_dim=config.qk_nope_head_dim,
        qk_rope_head_dim=config.qk_rope_head_dim,
        v_head_dim=config.v_head_dim,
        rope_theta=config.rope_theta,
        index_n_heads=config.index_n_heads,
        index_head_dim=config.index_head_dim,
        index_topk=config.index_topk,
        max_seq_len=config.max_position_embeddings,
        original_seq_len=config.rope_scaling.get("original_max_position_embeddings", 4096)
        if config.rope_scaling
        else 4096,
        rope_factor=config.rope_scaling.get("factor", 1.0) if config.rope_scaling else 1.0,
        beta_fast=config.rope_scaling.get("beta_fast", 32) if config.rope_scaling else 32,
        beta_slow=config.rope_scaling.get("beta_slow", 1) if config.rope_scaling else 1,
        mscale=config.rope_scaling.get("mscale", 1.0) if config.rope_scaling else 1.0,
    )


# =============================================================================
# Comparison Tests
# =============================================================================


def test_rope_interleaved_equivalence():
    """
    Test 1: Verify interleaved RoPE matches reference implementation.

    Reference: model.py lines 422-442 (apply_rotary_emb with interleaved=True)
    HF impl: modeling_deepseek_v32.py apply_rotary_emb
    """
    print("=" * 60)
    print("Test 1: RoPE Interleaved Equivalence")
    print("=" * 60)

    from transformers.models.deepseek_v32.modeling_deepseek_v32 import apply_rotary_emb

    # Test with various shapes
    test_cases = [
        (1, 8, 4, 16),  # batch, seq, heads, head_dim
        (2, 16, 8, 32),
        (1, 32, 2, 64),
    ]

    for batch, seq, heads, head_dim in test_cases:
        torch.manual_seed(42)
        x = torch.randn(batch, seq, heads, head_dim)

        # Create freqs_cis
        freqs = torch.randn(seq, head_dim // 2)
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)

        # Reference implementation
        ref_output = reference_apply_rotary_emb(x, freqs_cis, interleaved=True)

        # HuggingFace implementation
        hf_output = apply_rotary_emb(x, freqs_cis, interleaved=True)

        if torch.allclose(ref_output, hf_output, atol=1e-5, rtol=1e-5):
            print(f"  ✓ Shape {(batch, seq, heads, head_dim)}: MATCH")
        else:
            diff = (ref_output - hf_output).abs().max().item()
            print(f"  ✗ Shape {(batch, seq, heads, head_dim)}: MISMATCH (max diff: {diff:.6f})")
            return False

    print("\n  Result: PASSED\n")
    return True


def test_rope_non_interleaved_equivalence():
    """
    Test 2: Verify non-interleaved RoPE matches reference implementation.

    Reference: model.py lines 422-442 (apply_rotary_emb with interleaved=False)
    Used in Indexer for rope on Q and K.
    """
    print("=" * 60)
    print("Test 2: RoPE Non-Interleaved Equivalence")
    print("=" * 60)

    from transformers.models.deepseek_v32.modeling_deepseek_v32 import apply_rotary_emb

    test_cases = [
        (1, 8, 4, 16),
        (2, 16, 8, 32),
        (1, 32, 2, 64),
    ]

    for batch, seq, heads, head_dim in test_cases:
        torch.manual_seed(42)
        x = torch.randn(batch, seq, heads, head_dim)

        freqs = torch.randn(seq, head_dim // 2)
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)

        ref_output = reference_apply_rotary_emb(x, freqs_cis, interleaved=False)
        hf_output = apply_rotary_emb(x, freqs_cis, interleaved=False)

        if torch.allclose(ref_output, hf_output, atol=1e-5, rtol=1e-5):
            print(f"  ✓ Shape {(batch, seq, heads, head_dim)}: MATCH")
        else:
            diff = (ref_output - hf_output).abs().max().item()
            print(f"  ✗ Shape {(batch, seq, heads, head_dim)}: MISMATCH (max diff: {diff:.6f})")
            return False

    print("\n  Result: PASSED\n")
    return True


def test_rope_interleaved_vs_non_interleaved_different():
    """
    Test 3: Verify interleaved and non-interleaved produce DIFFERENT results.

    This is critical because the Indexer uses non-interleaved while MLA uses interleaved.
    Mixing them up would cause silent numerical errors.
    """
    print("=" * 60)
    print("Test 3: RoPE Interleaved != Non-Interleaved")
    print("=" * 60)

    from transformers.models.deepseek_v32.modeling_deepseek_v32 import apply_rotary_emb

    torch.manual_seed(42)
    batch, seq, heads, head_dim = 1, 8, 4, 16
    x = torch.randn(batch, seq, heads, head_dim)

    freqs = torch.randn(seq, head_dim // 2)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)

    interleaved_output = apply_rotary_emb(x, freqs_cis, interleaved=True)
    non_interleaved_output = apply_rotary_emb(x, freqs_cis, interleaved=False)

    if not torch.allclose(interleaved_output, non_interleaved_output, atol=1e-5):
        print("  ✓ Interleaved and non-interleaved produce different results (CORRECT)")
    else:
        print("  ✗ Interleaved and non-interleaved produce same results (WRONG!)")
        return False

    print("\n  Result: PASSED\n")
    return True


def test_rms_norm_equivalence():
    """
    Test 4: Verify RMSNorm matches reference implementation.

    Reference: model.py lines 287-322
    """
    print("=" * 60)
    print("Test 4: RMSNorm Equivalence")
    print("=" * 60)

    from transformers.models.deepseek_v32.modeling_deepseek_v32 import DeepseekV32RMSNorm

    test_dims = [256, 512, 1024]
    test_shapes = [(1, 8), (2, 16), (4, 32)]

    for dim in test_dims:
        for batch, seq in test_shapes:
            torch.manual_seed(42)

            # Create both norms with same weights
            ref_norm = ReferenceRMSNorm(dim)
            hf_norm = DeepseekV32RMSNorm(dim)

            # Copy weights
            with torch.no_grad():
                hf_norm.weight.copy_(ref_norm.weight)

            x = torch.randn(batch, seq, dim)

            ref_output = ref_norm(x)
            hf_output = hf_norm(x)

            if torch.allclose(ref_output, hf_output, atol=1e-5, rtol=1e-5):
                print(f"  ✓ dim={dim}, shape=({batch}, {seq}): MATCH")
            else:
                diff = (ref_output - hf_output).abs().max().item()
                print(f"  ✗ dim={dim}, shape=({batch}, {seq}): MISMATCH (max diff: {diff:.6f})")
                return False

    print("\n  Result: PASSED\n")
    return True


def test_mlp_equivalence():
    """
    Test 5: Verify MLP matches reference implementation.

    Reference: model.py lines 664-697 (MLP class)
    Key detail: Uses float32 for SiLU intermediate computation.
    """
    print("=" * 60)
    print("Test 5: MLP Equivalence")
    print("=" * 60)

    from transformers.models.deepseek_v32.modeling_deepseek_v32 import DeepseekV32MLP

    config = get_small_test_config()

    torch.manual_seed(42)

    # Create both MLPs
    ref_mlp = ReferenceMLP(config.hidden_size, config.intermediate_size)
    hf_mlp = DeepseekV32MLP(config, intermediate_size=config.intermediate_size)

    # Copy weights
    with torch.no_grad():
        hf_mlp.gate_proj.weight.copy_(ref_mlp.w1.weight)
        hf_mlp.down_proj.weight.copy_(ref_mlp.w2.weight)
        hf_mlp.up_proj.weight.copy_(ref_mlp.w3.weight)

    # Test with different dtypes
    for dtype in [torch.float32, torch.bfloat16]:
        ref_mlp = ref_mlp.to(dtype)
        hf_mlp = hf_mlp.to(dtype)

        x = torch.randn(2, 8, config.hidden_size, dtype=dtype)

        ref_output = ref_mlp(x)
        hf_output = hf_mlp(x)

        # Use looser tolerance for bfloat16
        atol = 1e-3 if dtype == torch.bfloat16 else 1e-5
        rtol = 1e-3 if dtype == torch.bfloat16 else 1e-5

        if torch.allclose(ref_output, hf_output, atol=atol, rtol=rtol):
            print(f"  ✓ dtype={dtype}: MATCH")
        else:
            diff = (ref_output - hf_output).abs().max().item()
            print(f"  ✗ dtype={dtype}: MISMATCH (max diff: {diff:.6f})")
            return False

    print("\n  Result: PASSED\n")
    return True


def test_expert_equivalence():
    """
    Test 6: Verify Expert MLP matches reference implementation.

    Reference: model.py lines 767-800 (Expert class)
    """
    print("=" * 60)
    print("Test 6: Expert Equivalence")
    print("=" * 60)

    from transformers.models.deepseek_v32.modeling_deepseek_v32 import DeepseekV32Expert

    config = get_small_test_config()

    torch.manual_seed(42)

    ref_expert = ReferenceExpert(config.hidden_size, config.moe_intermediate_size)
    hf_expert = DeepseekV32Expert(config)

    # Copy weights
    with torch.no_grad():
        hf_expert.gate_proj.weight.copy_(ref_expert.w1.weight)
        hf_expert.down_proj.weight.copy_(ref_expert.w2.weight)
        hf_expert.up_proj.weight.copy_(ref_expert.w3.weight)

    for dtype in [torch.float32, torch.bfloat16]:
        ref_expert = ref_expert.to(dtype)
        hf_expert = hf_expert.to(dtype)

        x = torch.randn(2, 8, config.hidden_size, dtype=dtype)

        ref_output = ref_expert(x)
        hf_output = hf_expert(x)

        atol = 1e-3 if dtype == torch.bfloat16 else 1e-5
        rtol = 1e-3 if dtype == torch.bfloat16 else 1e-5

        if torch.allclose(ref_output, hf_output, atol=atol, rtol=rtol):
            print(f"  ✓ dtype={dtype}: MATCH")
        else:
            diff = (ref_output - hf_output).abs().max().item()
            print(f"  ✗ dtype={dtype}: MISMATCH (max diff: {diff:.6f})")
            return False

    print("\n  Result: PASSED\n")
    return True


def test_gate_routing_equivalence():
    """
    Test 7: Verify Gate routing matches reference implementation.

    Reference: model.py lines 700-764 (Gate class)
    Tests:
    - Sigmoid scoring
    - Group-based expert selection
    - Weight normalization
    - Scaling factor application
    """
    print("=" * 60)
    print("Test 7: Gate Routing Equivalence")
    print("=" * 60)

    from transformers.models.deepseek_v32.modeling_deepseek_v32 import DeepseekV32Gate

    config = get_small_test_config()
    ref_args = get_reference_args_from_config(config)

    torch.manual_seed(42)

    ref_gate = ReferenceGate(ref_args)
    hf_gate = DeepseekV32Gate(config)

    # Copy weights
    with torch.no_grad():
        hf_gate.weight.copy_(ref_gate.weight)
        # Note: bias is only present for hidden_size=7168

    # Test with multiple batches
    for batch_size in [1, 4, 16]:
        x = torch.randn(batch_size, config.hidden_size)

        ref_weights, ref_indices = ref_gate(x)
        hf_weights, hf_indices = hf_gate(x)

        # Check indices match
        if torch.equal(ref_indices.sort(dim=-1)[0], hf_indices.sort(dim=-1)[0]):
            print(f"  ✓ batch_size={batch_size}: Indices MATCH")
        else:
            print(f"  ✗ batch_size={batch_size}: Indices MISMATCH")
            return False

        # Check weights are close (order may differ due to topk)
        # Sort both by indices to compare
        ref_sorted_weights = ref_weights.gather(1, ref_indices.argsort(dim=-1))
        hf_sorted_weights = hf_weights.gather(1, hf_indices.argsort(dim=-1))

        if torch.allclose(ref_sorted_weights, hf_sorted_weights, atol=1e-5, rtol=1e-5):
            print(f"  ✓ batch_size={batch_size}: Weights MATCH")
        else:
            diff = (ref_sorted_weights - hf_sorted_weights).abs().max().item()
            print(f"  ✗ batch_size={batch_size}: Weights MISMATCH (max diff: {diff:.6f})")
            return False

    print("\n  Result: PASSED\n")
    return True


def test_yarn_freqs_equivalence():
    """
    Test 8: Verify YaRN frequency computation matches reference.

    Reference: model.py lines 341-419 (precompute_freqs_cis)
    """
    print("=" * 60)
    print("Test 8: YaRN Frequency Computation Equivalence")
    print("=" * 60)

    from transformers.models.deepseek_v32.configuration_deepseek_v32 import DeepseekV32Config
    from transformers.models.deepseek_v32.modeling_deepseek_v32 import DeepseekV32RotaryEmbedding

    # Create config with YaRN scaling
    config = DeepseekV32Config(
        vocab_size=1000,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=2,
        num_attention_heads=4,
        q_lora_rank=64,
        kv_lora_rank=32,
        qk_nope_head_dim=32,
        qk_rope_head_dim=16,  # This determines the freq dimension
        v_head_dim=32,
        max_position_embeddings=8192,
        rope_scaling={
            "type": "yarn",
            "factor": 4.0,
            "original_max_position_embeddings": 2048,
            "mscale": 1.0,
            "beta_fast": 32,
            "beta_slow": 1,
        },
    )

    ref_args = ReferenceModelArgs(
        qk_rope_head_dim=config.qk_rope_head_dim,
        max_seq_len=config.max_position_embeddings,
        original_seq_len=2048,
        rope_factor=4.0,
        beta_fast=32,
        beta_slow=1,
        rope_theta=config.rope_theta,
    )

    # Compute reference freqs_cis
    ref_freqs_cis = reference_precompute_freqs_cis(ref_args)

    # Compute HF freqs_cis
    hf_rotary = DeepseekV32RotaryEmbedding(config)
    x = torch.randn(1, 8192, config.hidden_size)
    position_ids = torch.arange(8192).unsqueeze(0)
    hf_freqs_cis = hf_rotary(x, position_ids)

    # Compare
    if torch.allclose(ref_freqs_cis, hf_freqs_cis, atol=1e-5, rtol=1e-5):
        print("  ✓ YaRN freqs_cis: MATCH")
    else:
        diff = (ref_freqs_cis - hf_freqs_cis).abs().max().item()
        print(f"  ✗ YaRN freqs_cis: MISMATCH (max diff: {diff:.6f})")
        return False

    print("\n  Result: PASSED\n")
    return True


def test_mscale_application():
    """
    Test 9: Verify mscale is applied correctly in attention softmax_scale.

    Reference: model.py lines 580-582 (MLA.__init__)
    mscale = 0.1 * args.mscale * math.log(args.rope_factor) + 1.0
    softmax_scale = softmax_scale * mscale * mscale
    """
    print("=" * 60)
    print("Test 9: mscale Application in Attention")
    print("=" * 60)

    from transformers.models.deepseek_v32.configuration_deepseek_v32 import DeepseekV32Config
    from transformers.models.deepseek_v32.modeling_deepseek_v32 import DeepseekV32ForCausalLM

    mscale = 1.2
    factor = 4.0

    config = DeepseekV32Config(
        vocab_size=1000,
        hidden_size=256,
        intermediate_size=512,
        moe_intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        q_lora_rank=64,
        kv_lora_rank=32,
        qk_nope_head_dim=32,
        qk_rope_head_dim=16,
        v_head_dim=32,
        n_routed_experts=4,
        n_shared_experts=1,
        num_experts_per_tok=2,
        n_group=2,
        topk_group=1,
        index_n_heads=4,
        index_head_dim=32,
        index_topk=128,
        first_k_dense_replace=1,
        max_position_embeddings=8192,
        rope_scaling={
            "type": "yarn",
            "factor": factor,
            "original_max_position_embeddings": 2048,
            "mscale": mscale,
            "beta_fast": 32,
            "beta_slow": 1,
        },
    )

    model = DeepseekV32ForCausalLM(config)
    attn = model.model.layers[0].self_attn

    # Compute expected softmax_scale
    qk_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim
    base_scale = qk_head_dim**-0.5
    mscale_adjustment = 0.1 * mscale * math.log(factor) + 1.0
    expected_scale = base_scale * mscale_adjustment * mscale_adjustment

    if abs(attn.softmax_scale - expected_scale) < 1e-6:
        print(f"  ✓ softmax_scale: {attn.softmax_scale:.6f} (expected {expected_scale:.6f})")
    else:
        print(f"  ✗ softmax_scale: {attn.softmax_scale:.6f}, expected {expected_scale:.6f}")
        return False

    print("\n  Result: PASSED\n")
    return True


def test_indexer_formula():
    """
    Test 10: Verify Indexer implements correct formula.

    Reference: model.py lines 482-519 (Indexer.forward)
    Formula: I_{t,s} = Σ_j w^I_{t,j} · ReLU(q^I_{t,j} · k^I_s)

    Key aspects:
    1. ReLU is applied BEFORE weight multiplication
    2. Non-interleaved RoPE is used
    3. RoPE dimension is at the START of head_dim (rope, then nope)
    """
    print("=" * 60)
    print("Test 10: Indexer Formula Verification")
    print("=" * 60)

    from transformers.models.deepseek_v32.modeling_deepseek_v32 import DeepseekV32Indexer

    config = get_small_test_config()

    torch.manual_seed(42)
    indexer = DeepseekV32Indexer(config, layer_idx=0)

    # Verify formula by checking source code structure
    import inspect

    source = inspect.getsource(indexer.forward)

    # Check 1: ReLU before weight multiplication
    relu_pos = source.find("torch.relu(scores)")
    weight_mult_pos = source.find("scores * head_weights")

    if relu_pos != -1 and weight_mult_pos != -1 and relu_pos < weight_mult_pos:
        print("  ✓ ReLU applied before weight multiplication")
    else:
        print("  ✗ ReLU order incorrect")
        return False

    # Check 2: Non-interleaved RoPE is used (interleaved=False)
    if "interleaved=False" in source:
        print("  ✓ Non-interleaved RoPE used (interleaved=False)")
    else:
        print("  ✗ Should use non-interleaved RoPE")
        return False

    # Check 3: RoPE dim split order (rope first)
    if "q_pe = q_states[..., : self.qk_rope_head_dim]" in source:
        print("  ✓ RoPE dimension is at START of head_dim")
    else:
        print("  ✗ RoPE dimension should be at start")
        return False

    print("\n  Result: PASSED\n")
    return True


def test_mla_dimension_splits():
    """
    Test 11: Verify MLA Q/K/V dimension splits match reference.

    Reference: model.py lines 609-628 (MLA.forward)
    - Q: [nope_dim, rope_dim] (nope first, then rope)
    - K: k_nope from kv_b_proj, k_pe from kv_a_proj
    - V: from kv_b_proj

    This is DIFFERENT from Indexer which has [rope_dim, nope_dim]!
    """
    print("=" * 60)
    print("Test 11: MLA Dimension Splits")
    print("=" * 60)

    from transformers.models.deepseek_v32.modeling_deepseek_v32 import DeepseekV32Attention

    config = get_small_test_config()
    attn = DeepseekV32Attention(config, layer_idx=0)

    # Verify Q projection output dimension
    q_out_dim = attn.q_b_proj.out_features
    expected_q_dim = config.num_attention_heads * (config.qk_nope_head_dim + config.qk_rope_head_dim)

    if q_out_dim == expected_q_dim:
        print(
            f"  ✓ Q projection: {q_out_dim} = heads({config.num_attention_heads}) × "
            f"(nope({config.qk_nope_head_dim}) + rope({config.qk_rope_head_dim}))"
        )
    else:
        print(f"  ✗ Q projection: {q_out_dim} != expected {expected_q_dim}")
        return False

    # Verify KV-B projection (k_nope + v)
    kv_b_out_dim = attn.kv_b_proj.out_features
    expected_kv_b_dim = config.num_attention_heads * (config.qk_nope_head_dim + config.v_head_dim)

    if kv_b_out_dim == expected_kv_b_dim:
        print(
            f"  ✓ KV-B projection: {kv_b_out_dim} = heads({config.num_attention_heads}) × "
            f"(k_nope({config.qk_nope_head_dim}) + v({config.v_head_dim}))"
        )
    else:
        print(f"  ✗ KV-B projection: {kv_b_out_dim} != expected {expected_kv_b_dim}")
        return False

    # Verify KV-A projection (kv_lora_rank + k_pe)
    kv_a_out_dim = attn.kv_a_proj_with_mqa.out_features
    expected_kv_a_dim = config.kv_lora_rank + config.qk_rope_head_dim

    if kv_a_out_dim == expected_kv_a_dim:
        print(
            f"  ✓ KV-A projection: {kv_a_out_dim} = lora({config.kv_lora_rank}) + "
            f"k_pe({config.qk_rope_head_dim})"
        )
    else:
        print(f"  ✗ KV-A projection: {kv_a_out_dim} != expected {expected_kv_a_dim}")
        return False

    # Verify split order in forward (nope first, then rope for Q)
    import inspect

    source = inspect.getsource(attn.forward)

    if "q_nope, q_pe = torch.split(q_states, [self.qk_nope_head_dim, self.qk_rope_head_dim]" in source:
        print("  ✓ Q split order: [nope, rope] (correct)")
    else:
        print("  ✗ Q split order may be incorrect")
        return False

    print("\n  Result: PASSED\n")
    return True


def test_mla_vs_indexer_rope_format():
    """
    Test 12: Verify MLA and Indexer use different RoPE formats correctly.

    MLA: interleaved=True (default)
    Indexer: interleaved=False

    This is a critical distinction from the reference implementation.
    """
    print("=" * 60)
    print("Test 12: MLA vs Indexer RoPE Format")
    print("=" * 60)

    from transformers.models.deepseek_v32.modeling_deepseek_v32 import (
        DeepseekV32Attention,
        DeepseekV32Indexer,
    )

    config = get_small_test_config()

    # Check MLA source code
    import inspect

    attn_source = inspect.getsource(DeepseekV32Attention.forward)
    indexer_source = inspect.getsource(DeepseekV32Indexer.forward)

    # MLA should use interleaved=True (default) or explicit True
    mla_uses_interleaved = (
        "apply_rotary_emb(q_pe, freqs_cis, interleaved=True)" in attn_source
        or ("apply_rotary_emb(q_pe, freqs_cis)" in attn_source and "interleaved=False" not in attn_source)
    )

    if mla_uses_interleaved:
        print("  ✓ MLA uses interleaved RoPE (correct)")
    else:
        print("  ✗ MLA should use interleaved RoPE")
        return False

    # Indexer should use interleaved=False
    if "interleaved=False" in indexer_source:
        print("  ✓ Indexer uses non-interleaved RoPE (correct)")
    else:
        print("  ✗ Indexer should use non-interleaved RoPE")
        return False

    print("\n  Result: PASSED\n")
    return True


def test_indexer_head_dim_split_order():
    """
    Test 13: Verify Indexer has ROPE FIRST in head_dim split.

    Reference: model.py lines 489-498
    q_pe, q_nope = torch.split(q, [self.rope_head_dim, self.head_dim - self.rope_head_dim], dim=-1)

    This is the OPPOSITE of MLA which has [nope, rope].
    """
    print("=" * 60)
    print("Test 13: Indexer Head Dim Split Order")
    print("=" * 60)

    from transformers.models.deepseek_v32.modeling_deepseek_v32 import DeepseekV32Indexer

    config = get_small_test_config()
    indexer = DeepseekV32Indexer(config, layer_idx=0)

    import inspect

    source = inspect.getsource(indexer.forward)

    # Check for correct split order: rope first, then nope
    if "q_pe = q_states[..., : self.qk_rope_head_dim]" in source:
        print("  ✓ Indexer Q split: rope FIRST (correct)")
    else:
        print("  ✗ Indexer Q split order may be incorrect")
        return False

    if "k_pe = k_states[..., : self.qk_rope_head_dim]" in source:
        print("  ✓ Indexer K split: rope FIRST (correct)")
    else:
        print("  ✗ Indexer K split order may be incorrect")
        return False

    print("\n  Result: PASSED\n")
    return True


def test_weight_normalization():
    """
    Test 14: Verify MoE weights are normalized correctly for sigmoid scoring.

    Reference: model.py lines 761-763
    if self.score_func == "sigmoid":
        weights /= weights.sum(dim=-1, keepdim=True)
    weights *= self.route_scale
    """
    print("=" * 60)
    print("Test 14: MoE Weight Normalization")
    print("=" * 60)

    from transformers.models.deepseek_v32.modeling_deepseek_v32 import DeepseekV32Gate

    config = get_small_test_config()

    torch.manual_seed(42)
    gate = DeepseekV32Gate(config)

    # Test with multiple batches
    for batch_size in [1, 8, 32]:
        x = torch.randn(batch_size, config.hidden_size)
        weights, indices = gate(x)

        # Check weights sum to route_scale
        weight_sums = weights.sum(dim=-1)
        expected_sum = config.routed_scaling_factor

        if torch.allclose(weight_sums, torch.full_like(weight_sums, expected_sum), atol=1e-5):
            print(f"  ✓ batch_size={batch_size}: weights sum to {expected_sum}")
        else:
            print(f"  ✗ batch_size={batch_size}: weights sum to {weight_sums.mean():.4f}, expected {expected_sum}")
            return False

    print("\n  Result: PASSED\n")
    return True


def test_hadamard_transform_handling():
    """
    Test 15: Verify Hadamard transform is handled correctly.

    Reference: model.py lines 445-450 (rotate_activation)
    The Hadamard transform is applied to both Q and K in the Indexer.
    When fast_hadamard_transform is not available, it should gracefully fallback.
    """
    print("=" * 60)
    print("Test 15: Hadamard Transform Handling")
    print("=" * 60)

    from transformers.models.deepseek_v32.modeling_deepseek_v32 import hadamard_transform_activation

    # Test with input
    x = torch.randn(2, 8, 128)

    try:
        result = hadamard_transform_activation(x)
        if result.shape == x.shape:
            # Check if transform was applied (library available) or skipped
            try:
                from fast_hadamard_transform import hadamard_transform

                print("  ✓ Hadamard transform applied (library available)")
            except ImportError:
                if torch.equal(result, x):
                    print("  ✓ Hadamard transform skipped (library not available, fallback works)")
                else:
                    print("  ✗ Unexpected behavior without library")
                    return False
        else:
            print(f"  ✗ Shape mismatch: {result.shape} vs {x.shape}")
            return False
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False

    print("\n  Result: PASSED\n")
    return True


def test_full_forward_consistency():
    """
    Test 16: End-to-end forward pass consistency test.

    Run the same input through the model multiple times and verify
    deterministic output (no randomness issues).
    """
    print("=" * 60)
    print("Test 16: Full Forward Pass Consistency")
    print("=" * 60)

    from transformers.models.deepseek_v32.modeling_deepseek_v32 import DeepseekV32ForCausalLM

    config = get_small_test_config()

    torch.manual_seed(42)
    model = DeepseekV32ForCausalLM(config)
    model.eval()

    input_ids = torch.randint(0, 1000, (1, 16))

    with torch.no_grad():
        output1 = model(input_ids).logits.clone()
        output2 = model(input_ids).logits.clone()
        output3 = model(input_ids).logits.clone()

    if torch.equal(output1, output2) and torch.equal(output2, output3):
        print("  ✓ Forward pass is deterministic")
    else:
        print("  ✗ Forward pass is not deterministic!")
        return False

    # Test with different batch sizes
    for batch_size in [1, 2, 4]:
        input_ids = torch.randint(0, 1000, (batch_size, 8))
        with torch.no_grad():
            output = model(input_ids)
        expected_shape = (batch_size, 8, config.vocab_size)
        if output.logits.shape == expected_shape:
            print(f"  ✓ batch_size={batch_size}: shape {expected_shape}")
        else:
            print(f"  ✗ batch_size={batch_size}: unexpected shape {output.logits.shape}")
            return False

    print("\n  Result: PASSED\n")
    return True


# =============================================================================
# Main Entry Point
# =============================================================================


def main():
    print("\n" + "=" * 60)
    print("DeepSeek V3.2 Reference Comparison Tests")
    print("=" * 60 + "\n")

    tests = [
        ("RoPE Interleaved Equivalence", test_rope_interleaved_equivalence),
        ("RoPE Non-Interleaved Equivalence", test_rope_non_interleaved_equivalence),
        ("RoPE Interleaved vs Non-Interleaved Different", test_rope_interleaved_vs_non_interleaved_different),
        ("RMSNorm Equivalence", test_rms_norm_equivalence),
        ("MLP Equivalence", test_mlp_equivalence),
        ("Expert Equivalence", test_expert_equivalence),
        ("Gate Routing Equivalence", test_gate_routing_equivalence),
        ("YaRN Frequency Computation", test_yarn_freqs_equivalence),
        ("mscale Application", test_mscale_application),
        ("Indexer Formula", test_indexer_formula),
        ("MLA Dimension Splits", test_mla_dimension_splits),
        ("MLA vs Indexer RoPE Format", test_mla_vs_indexer_rope_format),
        ("Indexer Head Dim Split Order", test_indexer_head_dim_split_order),
        ("Weight Normalization", test_weight_normalization),
        ("Hadamard Transform Handling", test_hadamard_transform_handling),
        ("Full Forward Consistency", test_full_forward_consistency),
    ]

    results = []
    for name, test_fn in tests:
        try:
            passed = test_fn()
            results.append((name, passed))
        except Exception as e:
            print(f"  ✗ Test crashed: {e}")
            import traceback

            traceback.print_exc()
            results.append((name, False))

    # Summary
    print("=" * 60)
    print("Summary")
    print("=" * 60)

    passed = sum(1 for _, p in results if p)
    total = len(results)

    for name, p in results:
        status = "✓ PASSED" if p else "✗ FAILED"
        print(f"  {status}: {name}")

    print(f"\n  Total: {passed}/{total} tests passed\n")

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
