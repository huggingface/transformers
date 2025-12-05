#!/usr/bin/env python3
"""
Fuzz testing for DeepSeek V3.2 implementation against reference.

This script runs randomized inputs through both implementations to find
numerical inconsistencies that structured tests might miss.
"""

import random
import sys
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn


# =============================================================================
# Reference Implementation (from deepseek_files/inference/model.py)
# =============================================================================


def ref_apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor, interleaved: bool = True) -> torch.Tensor:
    """Reference apply_rotary_emb from model.py lines 422-442."""
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
    """Reference RMSNorm from model.py."""

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


class RefLayerNorm(nn.Module):
    """Reference LayerNorm from model.py."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.bias = nn.Parameter(torch.zeros(dim, dtype=torch.float32))

    def forward(self, x: torch.Tensor):
        return F.layer_norm(x.float(), (self.dim,), self.weight, self.bias, self.eps).type_as(x)


class RefMLP(nn.Module):
    """Reference MLP from model.py."""

    def __init__(self, dim: int, inter_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, inter_dim, bias=False)
        self.w2 = nn.Linear(inter_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, inter_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2((F.silu(self.w1(x).float()) * self.w3(x).float()).type_as(x))


class RefExpert(nn.Module):
    """Reference Expert from model.py."""

    def __init__(self, dim: int, inter_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, inter_dim, bias=False)
        self.w2 = nn.Linear(inter_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, inter_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2((F.silu(self.w1(x).float()) * self.w3(x).float()).type_as(x))


class RefGate(nn.Module):
    """
    Reference Gate from model.py - training-correct version.

    The original reference (inference-only) has a subtle bug: it uses
    `original_scores = scores` (not a clone) followed by in-place `masked_fill_`,
    which corrupts `original_scores` through the shared view. This doesn't matter
    for inference but breaks training.

    This version fixes that bug for training while keeping all other logic identical.

    Key reference behaviors preserved:
    - sorted=True (default) in topk calls
    - torch.empty initialization for weight/bias
    - top-2 sum for group scoring when bias is present
    - No epsilon in sigmoid weight normalization
    """

    def __init__(self, dim: int, n_experts: int, topk: int, n_groups: int, topk_groups: int,
                 score_func: str = "softmax", route_scale: float = 1.0):
        super().__init__()
        self.dim = dim
        self.topk = topk
        self.n_groups = n_groups
        self.topk_groups = topk_groups
        self.score_func = score_func
        self.route_scale = route_scale
        # Reference: torch.empty (line 729)
        self.weight = nn.Parameter(torch.empty(n_experts, dim))
        # Reference: torch.empty with float32 dtype (line 730)
        self.bias = nn.Parameter(torch.empty(n_experts, dtype=torch.float32)) if dim == 7168 else None

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Reference: model.py lines 742-764
        scores = F.linear(x.float(), self.weight.float())
        if self.score_func == "softmax":
            scores = scores.softmax(dim=-1)
        else:
            scores = scores.sigmoid()
        # FIX: Clone to prevent corruption by masked_fill below
        # Reference bug: `original_scores = scores` shares storage with the view
        original_scores = scores.clone()
        if self.bias is not None:
            scores = scores + self.bias
        if self.n_groups > 1:
            scores = scores.view(x.size(0), self.n_groups, -1)
            if self.bias is None:
                group_scores = scores.amax(dim=-1)
            else:
                group_scores = scores.topk(2, dim=-1)[0].sum(dim=-1)
            # Reference uses sorted=True (default) - line 756
            indices = group_scores.topk(self.topk_groups, dim=-1)[1]
            # Reference uses scores.new_ones and scatter_ - line 757
            mask = scores.new_ones(x.size(0), self.n_groups, dtype=bool).scatter_(1, indices, False)
            # FIX: Use non-in-place masked_fill for gradient computation
            # Reference uses masked_fill_ which breaks gradients
            scores = scores.masked_fill(mask.unsqueeze(-1), float("-inf")).flatten(1)
        # Reference uses sorted=True (default) - line 759
        indices = scores.topk(self.topk, dim=-1)[1]
        weights = original_scores.gather(1, indices)
        if self.score_func == "sigmoid":
            # Reference: no epsilon, can produce NaN - line 762
            # HF adds 1e-8 to prevent NaN during training - this is a valid improvement
            weights /= weights.sum(dim=-1, keepdim=True)
        weights *= self.route_scale
        return weights, indices


class RefIndexer(nn.Module):
    """
    Reference Indexer from model.py lines 482-519.

    Computes index scores: I_{t,s} = Σ w^I_{t,j} · ReLU(q^I_{t,j} · k^I_s)
    and selects top-k tokens for sparse attention.

    Key behaviors from reference:
    - RoPE is NOT interleaved (line 490-491, 496-497)
    - Q split order: rope FIRST, then nope (line 489)
    - K split order: rope FIRST, then nope (line 495)
    - Hadamard transform is optional (skipped here for simplicity)
    """

    def __init__(self, hidden_size: int, q_lora_rank: int, num_heads: int,
                 head_dim: int, qk_rope_head_dim: int, index_topk: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.index_topk = index_topk

        # Reference: line 463-467
        self.wq_b = nn.Linear(q_lora_rank, num_heads * head_dim, bias=False)
        self.wk = nn.Linear(hidden_size, head_dim, bias=False)
        self.k_norm = nn.LayerNorm(head_dim)
        self.weights_proj = nn.Linear(hidden_size, num_heads, bias=False)

        self.softmax_scale = head_dim ** -0.5

    def forward(
        self,
        hidden_states: torch.Tensor,
        q_compressed: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute top-k indices for sparse attention.

        Args:
            hidden_states: [batch, seq_len, hidden_size]
            q_compressed: [batch, seq_len, q_lora_rank]
            freqs_cis: Complex frequencies [seq_len, rope_dim // 2]
            mask: Optional causal mask [batch, seq_len, seq_len]

        Returns:
            topk_indices: [batch, seq_len, topk]
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Reference: lines 487-488
        q_states = self.wq_b(q_compressed)
        q_states = q_states.view(batch_size, seq_len, self.num_heads, self.head_dim)

        # Reference: line 489 - rope FIRST, then nope (opposite of main attention!)
        q_pe = q_states[..., :self.qk_rope_head_dim]
        q_nope = q_states[..., self.qk_rope_head_dim:]

        # Reference: lines 490-492 - interleaved=False for indexer
        q_pe = ref_apply_rotary_emb(q_pe, freqs_cis, interleaved=False)
        q_states = torch.cat([q_pe, q_nope], dim=-1)

        # Reference: lines 493-494
        k_states = self.k_norm(self.wk(hidden_states))

        # Reference: line 495 - rope FIRST, then nope
        k_pe = k_states[..., :self.qk_rope_head_dim]
        k_nope = k_states[..., self.qk_rope_head_dim:]

        # Reference: lines 496-498 - interleaved=False, single-head
        k_pe = ref_apply_rotary_emb(k_pe.unsqueeze(2), freqs_cis, interleaved=False).squeeze(2)
        k_states = torch.cat([k_pe, k_nope], dim=-1)

        # Skip Hadamard transform (rotate_activation) - it's optional

        # Reference: line 505 - head weights
        weights = F.linear(hidden_states.float(), self.weights_proj.weight.float()) * (self.num_heads ** -0.5)

        # Compute scores: einsum("bshd,btd->bsht", q, k) * scale
        # Then apply ReLU and weight by head weights
        # Reference: fp8_index does: sum_j(w_j * relu(q_j @ k^T * scale))
        scores = torch.einsum("bshd,btd->bsht", q_states.float(), k_states.float()) * self.softmax_scale
        scores = torch.relu(scores)
        scores = scores * weights.unsqueeze(-1)
        index_scores = scores.sum(dim=2)  # [B, S, T]

        # Apply mask
        if mask is not None:
            index_scores = index_scores + mask

        # Select top-k
        topk = min(self.index_topk, seq_len)
        topk_indices = index_scores.topk(topk, dim=-1).indices

        return topk_indices


class RefMLA(nn.Module):
    """
    Reference Multi-head Latent Attention from model.py lines 594-661.

    This implements the prefill path (mask is not None) which is what we test.

    Key behaviors from reference:
    - Q split order: nope FIRST, then rope (line 612)
    - RoPE is interleaved (default, line 613, 617)
    - K is expanded to all heads (line 628)
    - Indexer mask is applied (lines 631-637)
    """

    def __init__(self, hidden_size: int, num_heads: int, q_lora_rank: int,
                 kv_lora_rank: int, qk_nope_head_dim: int, qk_rope_head_dim: int,
                 v_head_dim: int, index_n_heads: int, index_head_dim: int, index_topk: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = v_head_dim

        # Q path (reference: lines 571-573)
        self.wq_a = nn.Linear(hidden_size, q_lora_rank, bias=False)
        self.q_norm = RefRMSNorm(q_lora_rank)
        self.wq_b = nn.Linear(q_lora_rank, num_heads * self.qk_head_dim, bias=False)

        # KV path (reference: lines 574-576)
        self.wkv_a = nn.Linear(hidden_size, kv_lora_rank + qk_rope_head_dim, bias=False)
        self.kv_norm = RefRMSNorm(kv_lora_rank)
        self.wkv_b = nn.Linear(kv_lora_rank, num_heads * (qk_nope_head_dim + v_head_dim), bias=False)

        # Output (reference: line 577)
        self.wo = nn.Linear(num_heads * v_head_dim, hidden_size, bias=False)

        # Softmax scale (reference: line 578)
        self.softmax_scale = self.qk_head_dim ** -0.5

        # Indexer
        self.indexer = RefIndexer(
            hidden_size, q_lora_rank, index_n_heads, index_head_dim,
            qk_rope_head_dim, index_topk
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass for MLA (prefill path).

        Args:
            hidden_states: [batch, seq_len, hidden_size]
            freqs_cis: Complex frequencies [seq_len, rope_dim // 2]
            mask: Causal mask [batch, seq_len, seq_len]

        Returns:
            output: [batch, seq_len, hidden_size]
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Q path (reference: lines 609-611)
        qr = self.q_norm(self.wq_a(hidden_states))
        q = self.wq_b(qr)
        q = q.view(batch_size, seq_len, self.num_heads, self.qk_head_dim)

        # Reference: line 612 - nope FIRST, then rope
        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        # Reference: line 613 - interleaved=True (default)
        q_pe = ref_apply_rotary_emb(q_pe, freqs_cis, interleaved=True)

        # KV path (reference: lines 614-616)
        kv_all = self.wkv_a(hidden_states)
        kv, k_pe = torch.split(kv_all, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        kv = self.kv_norm(kv)

        # Reference: line 617 - interleaved=True (default)
        k_pe = ref_apply_rotary_emb(k_pe.unsqueeze(2), freqs_cis, interleaved=True)

        # Reference: line 624
        q = torch.cat([q_nope, q_pe], dim=-1)

        # Reference: lines 625-627
        kv_proj = self.wkv_b(kv)
        kv_proj = kv_proj.view(batch_size, seq_len, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
        k_nope, v = torch.split(kv_proj, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)

        # Reference: line 628 - expand k_pe to all heads
        k = torch.cat([k_nope, k_pe.expand(-1, -1, self.num_heads, -1)], dim=-1)

        # Reference: line 629
        scores = torch.einsum("bshd,bthd->bsht", q.float(), k.float()) * self.softmax_scale

        # Reference: lines 632-637 - indexer
        topk_indices = self.indexer(hidden_states, qr, freqs_cis, mask)
        index_mask = torch.full((batch_size, seq_len, seq_len), float("-inf"), device=hidden_states.device)
        index_mask.scatter_(-1, topk_indices, 0.0)
        index_mask = index_mask + mask
        # scores shape is [B, S, H, T], need to unsqueeze at dim=2 for head broadcast
        scores = scores + index_mask.unsqueeze(2)

        # Reference: lines 639-640
        scores = scores.softmax(dim=-1)
        output = torch.einsum("bsht,bthd->bshd", scores.to(v.dtype), v)

        # Reference: line 660
        output = self.wo(output.flatten(2))

        return output


# =============================================================================
# Fuzz Test Framework
# =============================================================================


class FuzzStats:
    """Track fuzz testing statistics."""

    def __init__(self):
        self.tests_run = 0
        self.tests_passed = 0
        self.failures = []

    def record_pass(self, test_name: str):
        self.tests_run += 1
        self.tests_passed += 1

    def record_fail(self, test_name: str, details: str):
        self.tests_run += 1
        self.failures.append((test_name, details))

    def summary(self):
        print(f"\n{'=' * 60}")
        print("Fuzz Test Summary")
        print(f"{'=' * 60}")
        print(f"  Tests run: {self.tests_run}")
        print(f"  Passed: {self.tests_passed}")
        print(f"  Failed: {len(self.failures)}")
        if self.failures:
            print("\nFailures:")
            for name, details in self.failures[:10]:  # Show first 10
                print(f"  - {name}: {details}")
            if len(self.failures) > 10:
                print(f"  ... and {len(self.failures) - 10} more")
        return len(self.failures) == 0


def random_tensor(shape, dtype=torch.float32, scale=1.0):
    """Generate random tensor with given shape."""
    return torch.randn(shape, dtype=dtype) * scale


def random_freqs_cis(seq_len: int, head_dim: int):
    """Generate random freqs_cis tensor."""
    freqs = torch.randn(seq_len, head_dim // 2)
    return torch.polar(torch.ones_like(freqs), freqs)


# =============================================================================
# Fuzz Tests
# =============================================================================


def fuzz_rope(stats: FuzzStats, num_iterations: int = 100):
    """Fuzz test RoPE implementation."""
    from transformers.models.deepseek_v32.modeling_deepseek_v32 import apply_rotary_emb as hf_apply_rotary_emb

    print("\nFuzzing RoPE...")

    for i in range(num_iterations):
        # Random dimensions
        batch = random.randint(1, 4)
        seq_len = random.randint(1, 64)
        n_heads = random.randint(1, 8)
        head_dim = random.choice([16, 32, 64, 128])
        interleaved = random.choice([True, False])
        dtype = random.choice([torch.float32, torch.bfloat16])

        torch.manual_seed(i)
        x = random_tensor((batch, seq_len, n_heads, head_dim), dtype=dtype)
        freqs_cis = random_freqs_cis(seq_len, head_dim)

        try:
            ref_out = ref_apply_rotary_emb(x, freqs_cis, interleaved=interleaved)
            hf_out = hf_apply_rotary_emb(x, freqs_cis, interleaved=interleaved)

            atol = 1e-3 if dtype == torch.bfloat16 else 1e-5
            if torch.allclose(ref_out, hf_out, atol=atol, rtol=1e-3):
                stats.record_pass(f"rope_{i}")
            else:
                max_diff = (ref_out - hf_out).abs().max().item()
                stats.record_fail(f"rope_{i}",
                    f"shape={x.shape}, interleaved={interleaved}, dtype={dtype}, max_diff={max_diff:.6f}")
        except Exception as e:
            stats.record_fail(f"rope_{i}", f"Exception: {e}")


def fuzz_rmsnorm(stats: FuzzStats, num_iterations: int = 100):
    """Fuzz test RMSNorm implementation."""
    from transformers.models.deepseek_v32.modeling_deepseek_v32 import DeepseekV32RMSNorm

    print("\nFuzzing RMSNorm...")

    for i in range(num_iterations):
        dim = random.choice([64, 128, 256, 512, 1024])
        batch = random.randint(1, 4)
        seq_len = random.randint(1, 64)
        dtype = random.choice([torch.float32, torch.bfloat16])
        eps = random.choice([1e-5, 1e-6, 1e-8])

        torch.manual_seed(i)

        ref_norm = RefRMSNorm(dim, eps=eps)
        hf_norm = DeepseekV32RMSNorm(dim, eps=eps)

        # Copy weights
        with torch.no_grad():
            hf_norm.weight.copy_(ref_norm.weight)

        ref_norm = ref_norm.to(dtype)
        hf_norm = hf_norm.to(dtype)

        x = random_tensor((batch, seq_len, dim), dtype=dtype)

        try:
            ref_out = ref_norm(x)
            hf_out = hf_norm(x)

            atol = 1e-3 if dtype == torch.bfloat16 else 1e-5
            if torch.allclose(ref_out, hf_out, atol=atol, rtol=1e-3):
                stats.record_pass(f"rmsnorm_{i}")
            else:
                max_diff = (ref_out - hf_out).abs().max().item()
                stats.record_fail(f"rmsnorm_{i}",
                    f"dim={dim}, shape={x.shape}, dtype={dtype}, max_diff={max_diff:.6f}")
        except Exception as e:
            stats.record_fail(f"rmsnorm_{i}", f"Exception: {e}")


def fuzz_mlp(stats: FuzzStats, num_iterations: int = 50):
    """Fuzz test MLP implementation."""
    from transformers.models.deepseek_v32.configuration_deepseek_v32 import DeepseekV32Config
    from transformers.models.deepseek_v32.modeling_deepseek_v32 import DeepseekV32MLP

    print("\nFuzzing MLP...")

    for i in range(num_iterations):
        hidden_size = random.choice([128, 256, 512])
        intermediate_size = random.choice([256, 512, 1024])
        batch = random.randint(1, 4)
        seq_len = random.randint(1, 32)
        dtype = random.choice([torch.float32, torch.bfloat16])

        torch.manual_seed(i)

        config = DeepseekV32Config(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_attention_heads=4,
            num_hidden_layers=1,
        )

        ref_mlp = RefMLP(hidden_size, intermediate_size)
        hf_mlp = DeepseekV32MLP(config, intermediate_size=intermediate_size)

        # Copy weights
        with torch.no_grad():
            hf_mlp.gate_proj.weight.copy_(ref_mlp.w1.weight)
            hf_mlp.down_proj.weight.copy_(ref_mlp.w2.weight)
            hf_mlp.up_proj.weight.copy_(ref_mlp.w3.weight)

        ref_mlp = ref_mlp.to(dtype)
        hf_mlp = hf_mlp.to(dtype)

        x = random_tensor((batch, seq_len, hidden_size), dtype=dtype)

        try:
            ref_out = ref_mlp(x)
            hf_out = hf_mlp(x)

            atol = 1e-2 if dtype == torch.bfloat16 else 1e-4
            if torch.allclose(ref_out, hf_out, atol=atol, rtol=1e-2):
                stats.record_pass(f"mlp_{i}")
            else:
                max_diff = (ref_out - hf_out).abs().max().item()
                stats.record_fail(f"mlp_{i}",
                    f"hidden={hidden_size}, inter={intermediate_size}, dtype={dtype}, max_diff={max_diff:.6f}")
        except Exception as e:
            stats.record_fail(f"mlp_{i}", f"Exception: {e}")


def fuzz_expert(stats: FuzzStats, num_iterations: int = 50):
    """Fuzz test Expert implementation."""
    from transformers.models.deepseek_v32.configuration_deepseek_v32 import DeepseekV32Config
    from transformers.models.deepseek_v32.modeling_deepseek_v32 import DeepseekV32Expert

    print("\nFuzzing Expert...")

    for i in range(num_iterations):
        hidden_size = random.choice([128, 256, 512])
        moe_inter_size = random.choice([64, 128, 256])
        batch = random.randint(1, 4)
        seq_len = random.randint(1, 32)
        dtype = random.choice([torch.float32, torch.bfloat16])

        torch.manual_seed(i)

        config = DeepseekV32Config(
            hidden_size=hidden_size,
            moe_intermediate_size=moe_inter_size,
            num_attention_heads=4,
            num_hidden_layers=1,
        )

        ref_expert = RefExpert(hidden_size, moe_inter_size)
        hf_expert = DeepseekV32Expert(config)

        # Copy weights
        with torch.no_grad():
            hf_expert.gate_proj.weight.copy_(ref_expert.w1.weight)
            hf_expert.down_proj.weight.copy_(ref_expert.w2.weight)
            hf_expert.up_proj.weight.copy_(ref_expert.w3.weight)

        ref_expert = ref_expert.to(dtype)
        hf_expert = hf_expert.to(dtype)

        x = random_tensor((batch, seq_len, hidden_size), dtype=dtype)

        try:
            ref_out = ref_expert(x)
            hf_out = hf_expert(x)

            atol = 1e-2 if dtype == torch.bfloat16 else 1e-4
            if torch.allclose(ref_out, hf_out, atol=atol, rtol=1e-2):
                stats.record_pass(f"expert_{i}")
            else:
                max_diff = (ref_out - hf_out).abs().max().item()
                stats.record_fail(f"expert_{i}",
                    f"hidden={hidden_size}, moe_inter={moe_inter_size}, dtype={dtype}, max_diff={max_diff:.6f}")
        except Exception as e:
            stats.record_fail(f"expert_{i}", f"Exception: {e}")


def fuzz_gate(stats: FuzzStats, num_iterations: int = 50):
    """Fuzz test Gate routing implementation."""
    from transformers.models.deepseek_v32.configuration_deepseek_v32 import DeepseekV32Config
    from transformers.models.deepseek_v32.modeling_deepseek_v32 import DeepseekV32Gate

    print("\nFuzzing Gate...")

    for i in range(num_iterations):
        # Use deterministic seed for each iteration - set both before params AND before tensors
        iter_seed = i + 10000
        random.seed(iter_seed)

        hidden_size = random.choice([128, 256, 512])
        n_experts = random.choice([4, 8, 16])
        topk = random.randint(1, min(4, n_experts))
        n_groups = random.choice([1, 2, 4])
        topk_groups = random.randint(1, n_groups)
        score_func = random.choice(["softmax", "sigmoid"])
        route_scale = random.uniform(0.5, 3.0)
        batch_size = random.randint(1, 16)

        # Ensure n_experts is divisible by n_groups
        n_experts = n_groups * (n_experts // n_groups)
        if n_experts == 0:
            n_experts = n_groups

        config = DeepseekV32Config(
            hidden_size=hidden_size,
            n_routed_experts=n_experts,
            num_experts_per_tok=topk,
            n_group=n_groups,
            topk_group=topk_groups,
            scoring_func=score_func,
            routed_scaling_factor=route_scale,
            num_attention_heads=4,
            num_hidden_layers=1,
        )

        # Create shared random weights first, then both gates use them
        torch.manual_seed(iter_seed)
        shared_weight = torch.randn(n_experts, hidden_size)

        ref_gate = RefGate(hidden_size, n_experts, topk, n_groups, topk_groups, score_func, route_scale)
        hf_gate = DeepseekV32Gate(config)

        # Copy shared weights to both
        with torch.no_grad():
            ref_gate.weight.copy_(shared_weight)
            hf_gate.weight.copy_(shared_weight)

        # Create input with different seed
        torch.manual_seed(iter_seed + 1)
        x = torch.randn(batch_size, hidden_size)

        try:
            ref_weights, ref_indices = ref_gate(x)
            hf_weights, hf_indices = hf_gate(x)

            # Check indices match (sorted, since order within topk may differ)
            ref_sorted = ref_indices.sort(dim=-1)[0]
            hf_sorted = hf_indices.sort(dim=-1)[0]

            indices_match = torch.equal(ref_sorted, hf_sorted)

            # Check expert->weight mapping matches (not positional weights)
            # Build expert weight tensors for comparison
            ref_expert_weights = torch.zeros(batch_size, n_experts)
            hf_expert_weights = torch.zeros(batch_size, n_experts)

            for b in range(batch_size):
                for j in range(topk):
                    ref_idx = ref_indices[b, j].item()
                    hf_idx = hf_indices[b, j].item()
                    ref_expert_weights[b, ref_idx] = ref_weights[b, j]
                    hf_expert_weights[b, hf_idx] = hf_weights[b, j]

            # Handle NaN by replacing with 0 for comparison
            ref_expert_weights = torch.nan_to_num(ref_expert_weights, nan=0.0)
            hf_expert_weights = torch.nan_to_num(hf_expert_weights, nan=0.0)

            # Note: For sigmoid scoring, HF adds 1e-8 epsilon to prevent NaN when weights
            # sum to ~0. This can cause small differences (~1e-3) in edge cases.
            # Using larger rtol for sigmoid to account for this intentional improvement.
            if score_func == "sigmoid":
                weights_match = torch.allclose(ref_expert_weights, hf_expert_weights, atol=1e-5, rtol=1e-2)
            else:
                weights_match = torch.allclose(ref_expert_weights, hf_expert_weights, atol=1e-5, rtol=1e-4)

            if indices_match and weights_match:
                stats.record_pass(f"gate_{i}")
            else:
                stats.record_fail(f"gate_{i}",
                    f"n_experts={n_experts}, topk={topk}, n_groups={n_groups}, "
                    f"score_func={score_func}, indices_match={indices_match}, weights_match={weights_match}")
        except Exception as e:
            stats.record_fail(f"gate_{i}", f"Exception: {e}")


def fuzz_attention_components(stats: FuzzStats, num_iterations: int = 30):
    """Fuzz test attention-related dimension splits and projections."""
    from transformers.models.deepseek_v32.configuration_deepseek_v32 import DeepseekV32Config
    from transformers.models.deepseek_v32.modeling_deepseek_v32 import DeepseekV32Attention

    print("\nFuzzing Attention Components...")

    for i in range(num_iterations):
        hidden_size = random.choice([128, 256, 512])
        num_heads = random.choice([2, 4, 8])
        q_lora_rank = random.choice([32, 64, 128])
        kv_lora_rank = random.choice([32, 64, 128])
        qk_nope_head_dim = random.choice([16, 32, 64])
        qk_rope_head_dim = random.choice([8, 16, 32])
        v_head_dim = random.choice([16, 32, 64])

        batch = random.randint(1, 2)
        seq_len = random.randint(1, 16)

        torch.manual_seed(i)

        config = DeepseekV32Config(
            hidden_size=hidden_size,
            num_attention_heads=num_heads,
            num_key_value_heads=num_heads,
            q_lora_rank=q_lora_rank,
            kv_lora_rank=kv_lora_rank,
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            v_head_dim=v_head_dim,
            num_hidden_layers=1,
            # Disable indexer for this test
            index_n_heads=4,
            index_head_dim=32,
            index_topk=8,
        )

        try:
            attn = DeepseekV32Attention(config, layer_idx=0)

            # Verify dimension calculations
            expected_q_dim = num_heads * (qk_nope_head_dim + qk_rope_head_dim)
            expected_kv_a_dim = kv_lora_rank + qk_rope_head_dim
            expected_kv_b_dim = num_heads * (qk_nope_head_dim + v_head_dim)

            actual_q_dim = attn.q_b_proj.out_features
            actual_kv_a_dim = attn.kv_a_proj_with_mqa.out_features
            actual_kv_b_dim = attn.kv_b_proj.out_features

            dims_correct = (
                actual_q_dim == expected_q_dim and
                actual_kv_a_dim == expected_kv_a_dim and
                actual_kv_b_dim == expected_kv_b_dim
            )

            if dims_correct:
                stats.record_pass(f"attn_dims_{i}")
            else:
                stats.record_fail(f"attn_dims_{i}",
                    f"Q: {actual_q_dim} vs {expected_q_dim}, "
                    f"KV-A: {actual_kv_a_dim} vs {expected_kv_a_dim}, "
                    f"KV-B: {actual_kv_b_dim} vs {expected_kv_b_dim}")
        except Exception as e:
            stats.record_fail(f"attn_dims_{i}", f"Exception: {e}")


def fuzz_attention_forward(stats: FuzzStats, num_iterations: int = 30):
    """
    Fuzz test attention forward pass behavior.

    This verifies that:
    1. Attention produces valid outputs (no NaN/Inf)
    2. With seq_len > 1, the indexer sparse mask is applied
    3. The sparse mask actually restricts attention to top-k positions
    """
    from transformers.models.deepseek_v32.configuration_deepseek_v32 import DeepseekV32Config
    from transformers.models.deepseek_v32.modeling_deepseek_v32 import DeepseekV32Attention

    print("\nFuzzing Attention Forward...")

    for i in range(num_iterations):
        torch.manual_seed(i)
        random.seed(i)

        hidden_size = random.choice([128, 256])
        num_heads = random.choice([2, 4])
        q_lora_rank = random.choice([32, 64])
        kv_lora_rank = random.choice([32, 64])
        qk_nope_head_dim = random.choice([16, 32])
        qk_rope_head_dim = random.choice([8, 16])
        v_head_dim = random.choice([16, 32])
        index_topk = random.choice([4, 8, 16])

        batch = random.randint(1, 2)
        # Use seq_len > 1 to trigger indexer path
        seq_len = random.randint(2, 16)

        config = DeepseekV32Config(
            hidden_size=hidden_size,
            num_attention_heads=num_heads,
            num_key_value_heads=num_heads,
            q_lora_rank=q_lora_rank,
            kv_lora_rank=kv_lora_rank,
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            v_head_dim=v_head_dim,
            num_hidden_layers=1,
            index_n_heads=2,
            index_head_dim=32,
            index_topk=index_topk,
        )

        try:
            attn = DeepseekV32Attention(config, layer_idx=0)
            attn.eval()

            x = torch.randn(batch, seq_len, hidden_size)
            freqs_cis = random_freqs_cis(seq_len, qk_rope_head_dim)

            # Create causal mask
            causal_mask = torch.full((seq_len, seq_len), float("-inf"))
            causal_mask = torch.triu(causal_mask, diagonal=1)
            causal_mask = causal_mask[None, None, :, :]

            with torch.no_grad():
                output, _, _ = attn(x, position_embeddings=freqs_cis, attention_mask=causal_mask)

            # Check output validity
            has_nan = torch.isnan(output).any().item()
            has_inf = torch.isinf(output).any().item()
            shape_correct = output.shape == x.shape

            if not has_nan and not has_inf and shape_correct:
                stats.record_pass(f"attn_forward_{i}")
            else:
                stats.record_fail(f"attn_forward_{i}",
                    f"NaN={has_nan}, Inf={has_inf}, shape={output.shape} vs {x.shape}")
        except Exception as e:
            stats.record_fail(f"attn_forward_{i}", f"Exception: {e}")


def fuzz_attention_indexer_effect(stats: FuzzStats, num_iterations: int = 20):
    """
    Fuzz test that the indexer actually affects attention output.

    This catches bugs like `if seq_len < 1:` which would disable the indexer.
    We verify by comparing outputs with different index_topk values.
    """
    from transformers.models.deepseek_v32.configuration_deepseek_v32 import DeepseekV32Config
    from transformers.models.deepseek_v32.modeling_deepseek_v32 import DeepseekV32Attention

    print("\nFuzzing Attention Indexer Effect...")

    for i in range(num_iterations):
        torch.manual_seed(i)

        hidden_size = 128
        num_heads = 4
        q_lora_rank = 32
        kv_lora_rank = 32
        qk_nope_head_dim = 16
        qk_rope_head_dim = 8
        v_head_dim = 16

        batch = 1
        # Must use seq_len > 1 and seq_len > index_topk for sparse effect
        seq_len = 16

        # Create two configs with very different topk values
        config_sparse = DeepseekV32Config(
            hidden_size=hidden_size,
            num_attention_heads=num_heads,
            num_key_value_heads=num_heads,
            q_lora_rank=q_lora_rank,
            kv_lora_rank=kv_lora_rank,
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            v_head_dim=v_head_dim,
            num_hidden_layers=1,
            index_n_heads=2,
            index_head_dim=32,
            index_topk=4,  # Very sparse - only attend to 4 positions
        )

        config_dense = DeepseekV32Config(
            hidden_size=hidden_size,
            num_attention_heads=num_heads,
            num_key_value_heads=num_heads,
            q_lora_rank=q_lora_rank,
            kv_lora_rank=kv_lora_rank,
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            v_head_dim=v_head_dim,
            num_hidden_layers=1,
            index_n_heads=2,
            index_head_dim=32,
            index_topk=seq_len,  # Dense - attend to all positions
        )

        try:
            torch.manual_seed(i)
            attn_sparse = DeepseekV32Attention(config_sparse, layer_idx=0)
            torch.manual_seed(i)
            attn_dense = DeepseekV32Attention(config_dense, layer_idx=0)

            # Copy weights to ensure identical except for topk behavior
            with torch.no_grad():
                for (name_s, param_s), (name_d, param_d) in zip(
                    attn_sparse.named_parameters(), attn_dense.named_parameters()
                ):
                    param_d.copy_(param_s)

            attn_sparse.eval()
            attn_dense.eval()

            torch.manual_seed(i + 1000)
            x = torch.randn(batch, seq_len, hidden_size)
            freqs_cis = random_freqs_cis(seq_len, qk_rope_head_dim)

            causal_mask = torch.full((seq_len, seq_len), float("-inf"))
            causal_mask = torch.triu(causal_mask, diagonal=1)
            causal_mask = causal_mask[None, None, :, :]

            with torch.no_grad():
                out_sparse, _, _ = attn_sparse(x, position_embeddings=freqs_cis, attention_mask=causal_mask)
                out_dense, _, _ = attn_dense(x, position_embeddings=freqs_cis, attention_mask=causal_mask)

            # The outputs SHOULD be different because sparse attention restricts
            # which positions can be attended to
            outputs_differ = not torch.allclose(out_sparse, out_dense, atol=1e-5)

            if outputs_differ:
                stats.record_pass(f"attn_indexer_effect_{i}")
            else:
                # If outputs are identical, the indexer isn't working!
                stats.record_fail(f"attn_indexer_effect_{i}",
                    "Sparse and dense attention produced identical outputs - indexer not applied!")
        except Exception as e:
            stats.record_fail(f"attn_indexer_effect_{i}", f"Exception: {e}")


def fuzz_indexer_components(stats: FuzzStats, num_iterations: int = 30):
    """Fuzz test indexer dimension calculations."""
    from transformers.models.deepseek_v32.configuration_deepseek_v32 import DeepseekV32Config
    from transformers.models.deepseek_v32.modeling_deepseek_v32 import DeepseekV32Indexer

    print("\nFuzzing Indexer Components...")

    for i in range(num_iterations):
        hidden_size = random.choice([128, 256, 512])
        q_lora_rank = random.choice([32, 64, 128])
        index_n_heads = random.choice([2, 4, 8])
        index_head_dim = random.choice([32, 64, 128])
        qk_rope_head_dim = random.choice([8, 16, 32])

        # Ensure rope dim fits in head dim
        if qk_rope_head_dim >= index_head_dim:
            qk_rope_head_dim = index_head_dim // 2

        torch.manual_seed(i)

        config = DeepseekV32Config(
            hidden_size=hidden_size,
            q_lora_rank=q_lora_rank,
            qk_rope_head_dim=qk_rope_head_dim,
            index_n_heads=index_n_heads,
            index_head_dim=index_head_dim,
            index_topk=64,
            num_attention_heads=4,
            num_hidden_layers=1,
        )

        try:
            indexer = DeepseekV32Indexer(config, layer_idx=0)

            # Check dimensions
            expected_q_out = index_n_heads * index_head_dim
            actual_q_out = indexer.wq_b.out_features

            expected_k_out = index_head_dim
            actual_k_out = indexer.wk.out_features

            dims_correct = (
                actual_q_out == expected_q_out and
                actual_k_out == expected_k_out
            )

            if dims_correct:
                stats.record_pass(f"indexer_dims_{i}")
            else:
                stats.record_fail(f"indexer_dims_{i}",
                    f"Q: {actual_q_out} vs {expected_q_out}, K: {actual_k_out} vs {expected_k_out}")
        except Exception as e:
            stats.record_fail(f"indexer_dims_{i}", f"Exception: {e}")


def fuzz_indexer_vs_reference(stats: FuzzStats, num_iterations: int = 30):
    """
    Fuzz test HF Indexer output against reference implementation.

    This is the core "mock inference" test for the indexer - it verifies
    that the HF implementation produces the exact same top-k indices as
    the reference implementation.
    """
    from transformers.models.deepseek_v32.configuration_deepseek_v32 import DeepseekV32Config
    from transformers.models.deepseek_v32.modeling_deepseek_v32 import DeepseekV32Indexer

    print("\nFuzzing Indexer vs Reference...")

    for i in range(num_iterations):
        torch.manual_seed(i)
        random.seed(i)

        hidden_size = random.choice([128, 256])
        q_lora_rank = random.choice([32, 64])
        index_n_heads = random.choice([2, 4])
        index_head_dim = random.choice([32, 64])
        qk_rope_head_dim = random.choice([8, 16])
        index_topk = random.choice([4, 8, 16])
        batch = random.randint(1, 2)
        seq_len = random.randint(4, 32)

        # Ensure rope fits in head dim
        if qk_rope_head_dim >= index_head_dim:
            qk_rope_head_dim = index_head_dim // 2

        # Ensure topk <= seq_len
        index_topk = min(index_topk, seq_len)

        config = DeepseekV32Config(
            hidden_size=hidden_size,
            q_lora_rank=q_lora_rank,
            qk_rope_head_dim=qk_rope_head_dim,
            index_n_heads=index_n_heads,
            index_head_dim=index_head_dim,
            index_topk=index_topk,
            num_attention_heads=4,
            num_hidden_layers=1,
        )

        try:
            # Create both implementations
            hf_indexer = DeepseekV32Indexer(config, layer_idx=0)
            ref_indexer = RefIndexer(
                hidden_size, q_lora_rank, index_n_heads, index_head_dim,
                qk_rope_head_dim, index_topk
            )

            # Copy weights from HF to reference
            with torch.no_grad():
                ref_indexer.wq_b.weight.copy_(hf_indexer.wq_b.weight)
                ref_indexer.wk.weight.copy_(hf_indexer.wk.weight)
                ref_indexer.k_norm.weight.copy_(hf_indexer.k_norm.weight)
                ref_indexer.k_norm.bias.copy_(hf_indexer.k_norm.bias)
                ref_indexer.weights_proj.weight.copy_(hf_indexer.weights_proj.weight)

            hf_indexer.eval()
            ref_indexer.eval()

            # Generate inputs
            torch.manual_seed(i + 1000)
            hidden_states = torch.randn(batch, seq_len, hidden_size)
            q_compressed = torch.randn(batch, seq_len, q_lora_rank)
            freqs_cis = random_freqs_cis(seq_len, qk_rope_head_dim)

            # Create causal mask
            causal_mask = torch.full((seq_len, seq_len), float("-inf"))
            causal_mask = torch.triu(causal_mask, diagonal=1)
            causal_mask = causal_mask.unsqueeze(0).expand(batch, -1, -1)

            with torch.no_grad():
                ref_indices = ref_indexer(hidden_states, q_compressed, freqs_cis, causal_mask)
                hf_indices, _ = hf_indexer(hidden_states, q_compressed, freqs_cis, causal_mask)

            # Compare indices (sorted, since order within topk may differ due to ties)
            ref_sorted = ref_indices.sort(dim=-1)[0]
            hf_sorted = hf_indices.sort(dim=-1)[0]

            indices_match = torch.equal(ref_sorted, hf_sorted)

            if indices_match:
                stats.record_pass(f"indexer_vs_ref_{i}")
            else:
                # Count how many indices differ
                diff_count = (ref_sorted != hf_sorted).sum().item()
                stats.record_fail(f"indexer_vs_ref_{i}",
                    f"Indices differ in {diff_count}/{ref_sorted.numel()} positions")
        except Exception as e:
            stats.record_fail(f"indexer_vs_ref_{i}", f"Exception: {e}")


def fuzz_attention_vs_reference(stats: FuzzStats, num_iterations: int = 30):
    """
    Fuzz test HF Attention output against reference MLA implementation.

    This is the core "mock inference" test for attention - it verifies
    that the HF implementation produces numerically equivalent outputs
    to the reference implementation.
    """
    from transformers.models.deepseek_v32.configuration_deepseek_v32 import DeepseekV32Config
    from transformers.models.deepseek_v32.modeling_deepseek_v32 import DeepseekV32Attention

    print("\nFuzzing Attention vs Reference...")

    for i in range(num_iterations):
        torch.manual_seed(i)
        random.seed(i)

        hidden_size = random.choice([128, 256])
        num_heads = random.choice([2, 4])
        q_lora_rank = random.choice([32, 64])
        kv_lora_rank = random.choice([32, 64])
        qk_nope_head_dim = random.choice([16, 32])
        qk_rope_head_dim = random.choice([8, 16])
        v_head_dim = random.choice([16, 32])
        index_n_heads = random.choice([2, 4])
        index_head_dim = random.choice([32, 64])
        index_topk = random.choice([4, 8, 16])

        batch = random.randint(1, 2)
        seq_len = random.randint(4, 16)

        # Ensure rope fits in index head dim
        if qk_rope_head_dim >= index_head_dim:
            qk_rope_head_dim = index_head_dim // 2

        # Ensure topk <= seq_len
        index_topk = min(index_topk, seq_len)

        config = DeepseekV32Config(
            hidden_size=hidden_size,
            num_attention_heads=num_heads,
            num_key_value_heads=num_heads,
            q_lora_rank=q_lora_rank,
            kv_lora_rank=kv_lora_rank,
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            v_head_dim=v_head_dim,
            index_n_heads=index_n_heads,
            index_head_dim=index_head_dim,
            index_topk=index_topk,
            num_hidden_layers=1,
        )

        try:
            # Create both implementations
            hf_attn = DeepseekV32Attention(config, layer_idx=0)
            ref_attn = RefMLA(
                hidden_size, num_heads, q_lora_rank, kv_lora_rank,
                qk_nope_head_dim, qk_rope_head_dim, v_head_dim,
                index_n_heads, index_head_dim, index_topk
            )

            # Copy weights from HF to reference
            with torch.no_grad():
                # Q path
                ref_attn.wq_a.weight.copy_(hf_attn.q_a_proj.weight)
                ref_attn.q_norm.weight.copy_(hf_attn.q_a_layernorm.weight)
                ref_attn.wq_b.weight.copy_(hf_attn.q_b_proj.weight)
                # KV path
                ref_attn.wkv_a.weight.copy_(hf_attn.kv_a_proj_with_mqa.weight)
                ref_attn.kv_norm.weight.copy_(hf_attn.kv_a_layernorm.weight)
                ref_attn.wkv_b.weight.copy_(hf_attn.kv_b_proj.weight)
                # Output
                ref_attn.wo.weight.copy_(hf_attn.o_proj.weight)
                # Indexer
                ref_attn.indexer.wq_b.weight.copy_(hf_attn.indexer.wq_b.weight)
                ref_attn.indexer.wk.weight.copy_(hf_attn.indexer.wk.weight)
                ref_attn.indexer.k_norm.weight.copy_(hf_attn.indexer.k_norm.weight)
                ref_attn.indexer.k_norm.bias.copy_(hf_attn.indexer.k_norm.bias)
                ref_attn.indexer.weights_proj.weight.copy_(hf_attn.indexer.weights_proj.weight)

            hf_attn.eval()
            ref_attn.eval()

            # Generate inputs
            torch.manual_seed(i + 1000)
            hidden_states = torch.randn(batch, seq_len, hidden_size)
            freqs_cis = random_freqs_cis(seq_len, qk_rope_head_dim)

            # Create causal mask for HF (4D) and reference (3D)
            causal_mask_3d = torch.full((seq_len, seq_len), float("-inf"))
            causal_mask_3d = torch.triu(causal_mask_3d, diagonal=1)
            causal_mask_3d = causal_mask_3d.unsqueeze(0).expand(batch, -1, -1)

            causal_mask_4d = causal_mask_3d.unsqueeze(1)  # [B, 1, S, S]

            with torch.no_grad():
                ref_out = ref_attn(hidden_states, freqs_cis, causal_mask_3d)
                hf_out, _, _ = hf_attn(hidden_states, position_embeddings=freqs_cis, attention_mask=causal_mask_4d)

            # Compare outputs
            if torch.allclose(ref_out, hf_out, atol=1e-4, rtol=1e-3):
                stats.record_pass(f"attn_vs_ref_{i}")
            else:
                max_diff = (ref_out - hf_out).abs().max().item()
                mean_diff = (ref_out - hf_out).abs().mean().item()
                stats.record_fail(f"attn_vs_ref_{i}",
                    f"max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")
        except Exception as e:
            stats.record_fail(f"attn_vs_ref_{i}", f"Exception: {e}")


def fuzz_indexer_backward_vs_reference(stats: FuzzStats, num_iterations: int = 30):
    """
    Fuzz test HF Indexer gradients against reference implementation.

    This is the "mock training" test for the indexer.
    Note: Indexer is typically used with @torch.no_grad() in inference,
    but we test gradients flow for completeness.
    """
    from transformers.models.deepseek_v32.configuration_deepseek_v32 import DeepseekV32Config
    from transformers.models.deepseek_v32.modeling_deepseek_v32 import DeepseekV32Indexer

    print("\nFuzzing Indexer Backward vs Reference...")

    for i in range(num_iterations):
        torch.manual_seed(i)
        random.seed(i)

        hidden_size = random.choice([128, 256])
        q_lora_rank = random.choice([32, 64])
        index_n_heads = random.choice([2, 4])
        index_head_dim = random.choice([32, 64])
        qk_rope_head_dim = random.choice([8, 16])
        index_topk = random.choice([4, 8])
        batch = random.randint(1, 2)
        seq_len = random.randint(4, 16)

        if qk_rope_head_dim >= index_head_dim:
            qk_rope_head_dim = index_head_dim // 2
        index_topk = min(index_topk, seq_len)

        config = DeepseekV32Config(
            hidden_size=hidden_size,
            q_lora_rank=q_lora_rank,
            qk_rope_head_dim=qk_rope_head_dim,
            index_n_heads=index_n_heads,
            index_head_dim=index_head_dim,
            index_topk=index_topk,
            num_attention_heads=4,
            num_hidden_layers=1,
        )

        try:
            hf_indexer = DeepseekV32Indexer(config, layer_idx=0)
            ref_indexer = RefIndexer(
                hidden_size, q_lora_rank, index_n_heads, index_head_dim,
                qk_rope_head_dim, index_topk
            )

            # Copy weights
            with torch.no_grad():
                ref_indexer.wq_b.weight.copy_(hf_indexer.wq_b.weight)
                ref_indexer.wk.weight.copy_(hf_indexer.wk.weight)
                ref_indexer.k_norm.weight.copy_(hf_indexer.k_norm.weight)
                ref_indexer.k_norm.bias.copy_(hf_indexer.k_norm.bias)
                ref_indexer.weights_proj.weight.copy_(hf_indexer.weights_proj.weight)

            # Generate inputs that require grad
            torch.manual_seed(i + 1000)
            hidden_ref = torch.randn(batch, seq_len, hidden_size, requires_grad=True)
            hidden_hf = hidden_ref.clone().detach().requires_grad_(True)
            q_comp_ref = torch.randn(batch, seq_len, q_lora_rank, requires_grad=True)
            q_comp_hf = q_comp_ref.clone().detach().requires_grad_(True)
            freqs_cis = random_freqs_cis(seq_len, qk_rope_head_dim)

            causal_mask = torch.full((seq_len, seq_len), float("-inf"))
            causal_mask = torch.triu(causal_mask, diagonal=1)
            causal_mask = causal_mask.unsqueeze(0).expand(batch, -1, -1)

            # Forward (indices are not differentiable, so we test intermediate scores)
            # We'll compute a loss based on the index computation and backprop

            # For ref, we need to expose intermediate scores
            # Simplified test: just verify no gradient issues
            ref_indices = ref_indexer(hidden_ref, q_comp_ref, freqs_cis, causal_mask)
            hf_indices, _ = hf_indexer(hidden_hf, q_comp_hf, freqs_cis, causal_mask)

            # Create a pseudo-loss from the float indices (for gradient testing)
            ref_loss = ref_indices.float().sum()
            hf_loss = hf_indices.float().sum()

            # Backward
            ref_loss.backward()
            hf_loss.backward()

            # Check gradients exist and are valid
            ref_h_grad = hidden_ref.grad
            hf_h_grad = hidden_hf.grad

            grads_valid = (
                ref_h_grad is not None and hf_h_grad is not None and
                not torch.isnan(ref_h_grad).any() and not torch.isnan(hf_h_grad).any()
            )

            if grads_valid:
                stats.record_pass(f"indexer_backward_{i}")
            else:
                stats.record_fail(f"indexer_backward_{i}",
                    f"ref_grad={ref_h_grad is not None}, hf_grad={hf_h_grad is not None}")
        except Exception as e:
            stats.record_fail(f"indexer_backward_{i}", f"Exception: {e}")


def fuzz_attention_backward_vs_reference(stats: FuzzStats, num_iterations: int = 30):
    """
    Fuzz test HF Attention gradients against reference MLA implementation.

    This is the core "mock training" test for attention - it verifies
    that gradients computed through the HF implementation match
    those computed through the reference implementation.
    """
    from transformers.models.deepseek_v32.configuration_deepseek_v32 import DeepseekV32Config
    from transformers.models.deepseek_v32.modeling_deepseek_v32 import DeepseekV32Attention

    print("\nFuzzing Attention Backward vs Reference...")

    for i in range(num_iterations):
        torch.manual_seed(i)
        random.seed(i)

        hidden_size = random.choice([128, 256])
        num_heads = random.choice([2, 4])
        q_lora_rank = random.choice([32, 64])
        kv_lora_rank = random.choice([32, 64])
        qk_nope_head_dim = random.choice([16, 32])
        qk_rope_head_dim = random.choice([8, 16])
        v_head_dim = random.choice([16, 32])
        index_n_heads = random.choice([2, 4])
        index_head_dim = random.choice([32, 64])
        index_topk = random.choice([4, 8])

        batch = random.randint(1, 2)
        seq_len = random.randint(4, 12)

        if qk_rope_head_dim >= index_head_dim:
            qk_rope_head_dim = index_head_dim // 2
        index_topk = min(index_topk, seq_len)

        config = DeepseekV32Config(
            hidden_size=hidden_size,
            num_attention_heads=num_heads,
            num_key_value_heads=num_heads,
            q_lora_rank=q_lora_rank,
            kv_lora_rank=kv_lora_rank,
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            v_head_dim=v_head_dim,
            index_n_heads=index_n_heads,
            index_head_dim=index_head_dim,
            index_topk=index_topk,
            num_hidden_layers=1,
        )

        try:
            hf_attn = DeepseekV32Attention(config, layer_idx=0)
            ref_attn = RefMLA(
                hidden_size, num_heads, q_lora_rank, kv_lora_rank,
                qk_nope_head_dim, qk_rope_head_dim, v_head_dim,
                index_n_heads, index_head_dim, index_topk
            )

            # Copy weights
            with torch.no_grad():
                ref_attn.wq_a.weight.copy_(hf_attn.q_a_proj.weight)
                ref_attn.q_norm.weight.copy_(hf_attn.q_a_layernorm.weight)
                ref_attn.wq_b.weight.copy_(hf_attn.q_b_proj.weight)
                ref_attn.wkv_a.weight.copy_(hf_attn.kv_a_proj_with_mqa.weight)
                ref_attn.kv_norm.weight.copy_(hf_attn.kv_a_layernorm.weight)
                ref_attn.wkv_b.weight.copy_(hf_attn.kv_b_proj.weight)
                ref_attn.wo.weight.copy_(hf_attn.o_proj.weight)
                ref_attn.indexer.wq_b.weight.copy_(hf_attn.indexer.wq_b.weight)
                ref_attn.indexer.wk.weight.copy_(hf_attn.indexer.wk.weight)
                ref_attn.indexer.k_norm.weight.copy_(hf_attn.indexer.k_norm.weight)
                ref_attn.indexer.k_norm.bias.copy_(hf_attn.indexer.k_norm.bias)
                ref_attn.indexer.weights_proj.weight.copy_(hf_attn.indexer.weights_proj.weight)

            # Generate inputs
            torch.manual_seed(i + 1000)
            hidden_ref = torch.randn(batch, seq_len, hidden_size, requires_grad=True)
            hidden_hf = hidden_ref.clone().detach().requires_grad_(True)
            freqs_cis = random_freqs_cis(seq_len, qk_rope_head_dim)

            causal_mask_3d = torch.full((seq_len, seq_len), float("-inf"))
            causal_mask_3d = torch.triu(causal_mask_3d, diagonal=1)
            causal_mask_3d = causal_mask_3d.unsqueeze(0).expand(batch, -1, -1)
            causal_mask_4d = causal_mask_3d.unsqueeze(1)

            # Forward
            ref_out = ref_attn(hidden_ref, freqs_cis, causal_mask_3d)
            hf_out, _, _ = hf_attn(hidden_hf, position_embeddings=freqs_cis, attention_mask=causal_mask_4d)

            # Create gradient target
            torch.manual_seed(i + 2000)
            grad_output = torch.randn_like(ref_out)

            # Backward
            ref_out.backward(grad_output)
            hf_out.backward(grad_output)

            # Compare input gradients
            ref_grad = hidden_ref.grad
            hf_grad = hidden_hf.grad

            if torch.allclose(ref_grad, hf_grad, atol=1e-4, rtol=1e-3):
                stats.record_pass(f"attn_backward_{i}")
            else:
                max_diff = (ref_grad - hf_grad).abs().max().item()
                mean_diff = (ref_grad - hf_grad).abs().mean().item()
                stats.record_fail(f"attn_backward_{i}",
                    f"max_grad_diff={max_diff:.6f}, mean_grad_diff={mean_diff:.6f}")
        except Exception as e:
            stats.record_fail(f"attn_backward_{i}", f"Exception: {e}")


def fuzz_full_forward(stats: FuzzStats, num_iterations: int = 20):
    """Fuzz test full forward pass for crashes and NaN/Inf."""
    from transformers.models.deepseek_v32.configuration_deepseek_v32 import DeepseekV32Config
    from transformers.models.deepseek_v32.modeling_deepseek_v32 import DeepseekV32ForCausalLM

    print("\nFuzzing Full Forward Pass...")

    for i in range(num_iterations):
        vocab_size = random.choice([100, 500, 1000])
        hidden_size = random.choice([128, 256])
        num_layers = random.choice([1, 2])
        num_heads = random.choice([2, 4])
        batch = random.randint(1, 4)
        seq_len = random.randint(1, 32)

        torch.manual_seed(i)

        config = DeepseekV32Config(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            intermediate_size=hidden_size * 2,
            moe_intermediate_size=hidden_size // 2,
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            num_key_value_heads=num_heads,
            q_lora_rank=hidden_size // 4,
            kv_lora_rank=hidden_size // 8,
            qk_nope_head_dim=hidden_size // num_heads // 2,
            qk_rope_head_dim=hidden_size // num_heads // 4,
            v_head_dim=hidden_size // num_heads // 2,
            n_routed_experts=4,
            n_shared_experts=1,
            num_experts_per_tok=2,
            n_group=2,
            topk_group=1,
            index_n_heads=2,
            index_head_dim=hidden_size // 4,
            index_topk=min(16, seq_len),
            first_k_dense_replace=1,
        )

        try:
            model = DeepseekV32ForCausalLM(config)
            model.eval()

            input_ids = torch.randint(0, vocab_size, (batch, seq_len))

            with torch.no_grad():
                output = model(input_ids)

            logits = output.logits

            # Check for NaN/Inf
            has_nan = torch.isnan(logits).any().item()
            has_inf = torch.isinf(logits).any().item()

            # Check shape
            expected_shape = (batch, seq_len, vocab_size)
            shape_correct = logits.shape == expected_shape

            if not has_nan and not has_inf and shape_correct:
                stats.record_pass(f"forward_{i}")
            else:
                stats.record_fail(f"forward_{i}",
                    f"NaN={has_nan}, Inf={has_inf}, shape={logits.shape} vs {expected_shape}")
        except Exception as e:
            stats.record_fail(f"forward_{i}", f"Exception: {e}")


def fuzz_rope_edge_cases(stats: FuzzStats, num_iterations: int = 50):
    """Fuzz test RoPE with edge cases."""
    from transformers.models.deepseek_v32.modeling_deepseek_v32 import apply_rotary_emb as hf_apply_rotary_emb

    print("\nFuzzing RoPE Edge Cases...")

    edge_cases = [
        # Very small/large values
        {"scale": 1e-6, "name": "tiny_values"},
        {"scale": 1e6, "name": "huge_values"},
        # Single elements
        {"batch": 1, "seq": 1, "heads": 1, "name": "single_elem"},
        # Large sequence
        {"seq": 512, "name": "long_seq"},
        # Many heads
        {"heads": 64, "name": "many_heads"},
    ]

    for i in range(num_iterations):
        edge = random.choice(edge_cases)

        batch = edge.get("batch", random.randint(1, 4))
        seq_len = edge.get("seq", random.randint(1, 64))
        n_heads = edge.get("heads", random.randint(1, 8))
        head_dim = random.choice([16, 32, 64])
        scale = edge.get("scale", 1.0)
        interleaved = random.choice([True, False])

        torch.manual_seed(i)
        x = random_tensor((batch, seq_len, n_heads, head_dim)) * scale
        freqs_cis = random_freqs_cis(seq_len, head_dim)

        try:
            ref_out = ref_apply_rotary_emb(x, freqs_cis, interleaved=interleaved)
            hf_out = hf_apply_rotary_emb(x, freqs_cis, interleaved=interleaved)

            # Use relative tolerance for scaled values
            if torch.allclose(ref_out, hf_out, atol=1e-4 * scale, rtol=1e-3):
                stats.record_pass(f"rope_edge_{edge['name']}_{i}")
            else:
                max_diff = (ref_out - hf_out).abs().max().item()
                stats.record_fail(f"rope_edge_{edge['name']}_{i}",
                    f"edge={edge['name']}, max_diff={max_diff:.6e}")
        except Exception as e:
            stats.record_fail(f"rope_edge_{edge['name']}_{i}", f"Exception: {e}")


def fuzz_moe_routing(stats: FuzzStats, num_iterations: int = 30):
    """Fuzz test MoE routing logic."""
    from transformers.models.deepseek_v32.configuration_deepseek_v32 import DeepseekV32Config
    from transformers.models.deepseek_v32.modeling_deepseek_v32 import DeepseekV32MoE

    print("\nFuzzing MoE Routing...")

    for i in range(num_iterations):
        hidden_size = random.choice([128, 256])
        n_routed = random.choice([4, 8])
        n_shared = random.choice([1, 2])
        topk = random.randint(1, min(3, n_routed))
        n_groups = random.choice([1, 2])
        batch = random.randint(1, 4)
        seq_len = random.randint(1, 16)

        # Ensure divisibility
        n_routed = n_groups * (n_routed // n_groups)
        if n_routed == 0:
            n_routed = n_groups

        torch.manual_seed(i)

        config = DeepseekV32Config(
            hidden_size=hidden_size,
            moe_intermediate_size=hidden_size // 2,
            n_routed_experts=n_routed,
            n_shared_experts=n_shared,
            num_experts_per_tok=topk,
            n_group=n_groups,
            topk_group=1,
            num_attention_heads=4,
            num_hidden_layers=1,
        )

        try:
            moe = DeepseekV32MoE(config)
            moe.eval()

            x = random_tensor((batch, seq_len, hidden_size))

            with torch.no_grad():
                output = moe(x)

            # Check output shape and no NaN/Inf
            has_nan = torch.isnan(output).any().item()
            has_inf = torch.isinf(output).any().item()
            shape_correct = output.shape == x.shape

            if not has_nan and not has_inf and shape_correct:
                stats.record_pass(f"moe_{i}")
            else:
                stats.record_fail(f"moe_{i}",
                    f"NaN={has_nan}, Inf={has_inf}, shape={output.shape}")
        except Exception as e:
            stats.record_fail(f"moe_{i}", f"Exception: {e}")


# =============================================================================
# Backward Pass Tests
# =============================================================================


def fuzz_rope_backward(stats: FuzzStats, num_iterations: int = 50):
    """Fuzz test RoPE backward pass - verify gradients match."""
    from transformers.models.deepseek_v32.modeling_deepseek_v32 import apply_rotary_emb as hf_apply_rotary_emb

    print("\nFuzzing RoPE Backward...")

    for i in range(num_iterations):
        batch = random.randint(1, 4)
        seq_len = random.randint(1, 32)
        n_heads = random.randint(1, 8)
        head_dim = random.choice([16, 32, 64])
        interleaved = random.choice([True, False])

        torch.manual_seed(i)

        # Create inputs that require grad
        x_ref = random_tensor((batch, seq_len, n_heads, head_dim), dtype=torch.float32)
        x_ref.requires_grad_(True)
        x_hf = x_ref.clone().detach().requires_grad_(True)

        freqs_cis = random_freqs_cis(seq_len, head_dim)

        try:
            # Forward
            ref_out = ref_apply_rotary_emb(x_ref, freqs_cis, interleaved=interleaved)
            hf_out = hf_apply_rotary_emb(x_hf, freqs_cis, interleaved=interleaved)

            # Create gradient target
            torch.manual_seed(i + 1000)
            grad_output = torch.randn_like(ref_out)

            # Backward
            ref_out.backward(grad_output)
            hf_out.backward(grad_output)

            ref_grad = x_ref.grad
            hf_grad = x_hf.grad

            if torch.allclose(ref_grad, hf_grad, atol=1e-5, rtol=1e-4):
                stats.record_pass(f"rope_backward_{i}")
            else:
                max_diff = (ref_grad - hf_grad).abs().max().item()
                stats.record_fail(f"rope_backward_{i}",
                    f"interleaved={interleaved}, max_grad_diff={max_diff:.6e}")
        except Exception as e:
            stats.record_fail(f"rope_backward_{i}", f"Exception: {e}")


def fuzz_rmsnorm_backward(stats: FuzzStats, num_iterations: int = 50):
    """Fuzz test RMSNorm backward pass."""
    from transformers.models.deepseek_v32.modeling_deepseek_v32 import DeepseekV32RMSNorm

    print("\nFuzzing RMSNorm Backward...")

    for i in range(num_iterations):
        batch = random.randint(1, 8)
        seq_len = random.randint(1, 64)
        dim = random.choice([128, 256, 512])
        eps = random.choice([1e-5, 1e-6, 1e-8])

        torch.manual_seed(i)

        ref_norm = RefRMSNorm(dim, eps)
        hf_norm = DeepseekV32RMSNorm(dim, eps)

        # Sync weights
        with torch.no_grad():
            hf_norm.weight.copy_(ref_norm.weight)

        x_ref = random_tensor((batch, seq_len, dim))
        x_ref.requires_grad_(True)
        x_hf = x_ref.clone().detach().requires_grad_(True)

        try:
            ref_out = ref_norm(x_ref)
            hf_out = hf_norm(x_hf)

            torch.manual_seed(i + 1000)
            grad_output = torch.randn_like(ref_out)

            ref_out.backward(grad_output)
            hf_out.backward(grad_output)

            # Check input gradients
            input_grad_match = torch.allclose(x_ref.grad, x_hf.grad, atol=1e-5, rtol=1e-4)

            # Check weight gradients
            weight_grad_match = torch.allclose(ref_norm.weight.grad, hf_norm.weight.grad, atol=1e-5, rtol=1e-4)

            if input_grad_match and weight_grad_match:
                stats.record_pass(f"rmsnorm_backward_{i}")
            else:
                input_diff = (x_ref.grad - x_hf.grad).abs().max().item() if not input_grad_match else 0
                weight_diff = (ref_norm.weight.grad - hf_norm.weight.grad).abs().max().item() if not weight_grad_match else 0
                stats.record_fail(f"rmsnorm_backward_{i}",
                    f"input_grad_match={input_grad_match} (diff={input_diff:.6e}), "
                    f"weight_grad_match={weight_grad_match} (diff={weight_diff:.6e})")
        except Exception as e:
            stats.record_fail(f"rmsnorm_backward_{i}", f"Exception: {e}")


def fuzz_mlp_backward(stats: FuzzStats, num_iterations: int = 50):
    """Fuzz test MLP backward pass."""
    from transformers.models.deepseek_v32.configuration_deepseek_v32 import DeepseekV32Config
    from transformers.models.deepseek_v32.modeling_deepseek_v32 import DeepseekV32MLP

    print("\nFuzzing MLP Backward...")

    for i in range(num_iterations):
        hidden_size = random.choice([128, 256])
        intermediate_size = random.choice([256, 512])
        batch = random.randint(1, 4)
        seq_len = random.randint(1, 16)

        torch.manual_seed(i)

        config = DeepseekV32Config(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_attention_heads=4,
            num_hidden_layers=1,
        )

        ref_mlp = RefMLP(hidden_size, intermediate_size)
        hf_mlp = DeepseekV32MLP(config)

        # Sync weights (RefMLP: w1=gate, w3=up, w2=down)
        with torch.no_grad():
            hf_mlp.gate_proj.weight.copy_(ref_mlp.w1.weight)
            hf_mlp.up_proj.weight.copy_(ref_mlp.w3.weight)
            hf_mlp.down_proj.weight.copy_(ref_mlp.w2.weight)

        x_ref = random_tensor((batch, seq_len, hidden_size))
        x_ref.requires_grad_(True)
        x_hf = x_ref.clone().detach().requires_grad_(True)

        try:
            ref_out = ref_mlp(x_ref)
            hf_out = hf_mlp(x_hf)

            torch.manual_seed(i + 1000)
            grad_output = torch.randn_like(ref_out)

            ref_out.backward(grad_output)
            hf_out.backward(grad_output)

            # Check input gradients
            input_grad_match = torch.allclose(x_ref.grad, x_hf.grad, atol=1e-5, rtol=1e-4)

            # Check weight gradients (RefMLP: w1=gate, w3=up, w2=down)
            gate_grad_match = torch.allclose(ref_mlp.w1.weight.grad, hf_mlp.gate_proj.weight.grad, atol=1e-5, rtol=1e-4)
            up_grad_match = torch.allclose(ref_mlp.w3.weight.grad, hf_mlp.up_proj.weight.grad, atol=1e-5, rtol=1e-4)
            down_grad_match = torch.allclose(ref_mlp.w2.weight.grad, hf_mlp.down_proj.weight.grad, atol=1e-5, rtol=1e-4)

            if input_grad_match and gate_grad_match and up_grad_match and down_grad_match:
                stats.record_pass(f"mlp_backward_{i}")
            else:
                stats.record_fail(f"mlp_backward_{i}",
                    f"input={input_grad_match}, gate={gate_grad_match}, up={up_grad_match}, down={down_grad_match}")
        except Exception as e:
            stats.record_fail(f"mlp_backward_{i}", f"Exception: {e}")


def fuzz_expert_backward(stats: FuzzStats, num_iterations: int = 50):
    """Fuzz test Expert backward pass."""
    from transformers.models.deepseek_v32.configuration_deepseek_v32 import DeepseekV32Config
    from transformers.models.deepseek_v32.modeling_deepseek_v32 import DeepseekV32Expert

    print("\nFuzzing Expert Backward...")

    for i in range(num_iterations):
        hidden_size = random.choice([128, 256])
        moe_intermediate_size = random.choice([64, 128])
        batch = random.randint(1, 4)
        seq_len = random.randint(1, 16)

        torch.manual_seed(i)

        config = DeepseekV32Config(
            hidden_size=hidden_size,
            moe_intermediate_size=moe_intermediate_size,
            num_attention_heads=4,
            num_hidden_layers=1,
        )

        ref_expert = RefExpert(hidden_size, moe_intermediate_size)
        hf_expert = DeepseekV32Expert(config)

        # Sync weights (RefExpert: w1=gate, w3=up, w2=down)
        with torch.no_grad():
            hf_expert.gate_proj.weight.copy_(ref_expert.w1.weight)
            hf_expert.up_proj.weight.copy_(ref_expert.w3.weight)
            hf_expert.down_proj.weight.copy_(ref_expert.w2.weight)

        x_ref = random_tensor((batch, seq_len, hidden_size))
        x_ref.requires_grad_(True)
        x_hf = x_ref.clone().detach().requires_grad_(True)

        try:
            ref_out = ref_expert(x_ref)
            hf_out = hf_expert(x_hf)

            torch.manual_seed(i + 1000)
            grad_output = torch.randn_like(ref_out)

            ref_out.backward(grad_output)
            hf_out.backward(grad_output)

            input_grad_match = torch.allclose(x_ref.grad, x_hf.grad, atol=1e-5, rtol=1e-4)

            if input_grad_match:
                stats.record_pass(f"expert_backward_{i}")
            else:
                max_diff = (x_ref.grad - x_hf.grad).abs().max().item()
                stats.record_fail(f"expert_backward_{i}",
                    f"max_input_grad_diff={max_diff:.6e}")
        except Exception as e:
            stats.record_fail(f"expert_backward_{i}", f"Exception: {e}")


def fuzz_gate_backward(stats: FuzzStats, num_iterations: int = 50):
    """Fuzz test Gate backward pass."""
    from transformers.models.deepseek_v32.configuration_deepseek_v32 import DeepseekV32Config
    from transformers.models.deepseek_v32.modeling_deepseek_v32 import DeepseekV32Gate

    print("\nFuzzing Gate Backward...")

    for i in range(num_iterations):
        iter_seed = i + 20000
        random.seed(iter_seed)

        hidden_size = random.choice([128, 256, 512])
        n_experts = random.choice([4, 8, 16])
        topk = random.randint(1, min(4, n_experts))
        n_groups = random.choice([1, 2, 4])
        topk_groups = random.randint(1, n_groups)
        score_func = random.choice(["softmax", "sigmoid"])
        route_scale = random.uniform(0.5, 3.0)
        batch_size = random.randint(1, 16)

        n_experts = n_groups * (n_experts // n_groups)
        if n_experts == 0:
            n_experts = n_groups

        config = DeepseekV32Config(
            hidden_size=hidden_size,
            n_routed_experts=n_experts,
            num_experts_per_tok=topk,
            n_group=n_groups,
            topk_group=topk_groups,
            scoring_func=score_func,
            routed_scaling_factor=route_scale,
            num_attention_heads=4,
            num_hidden_layers=1,
        )

        torch.manual_seed(iter_seed)
        shared_weight = torch.randn(n_experts, hidden_size)

        ref_gate = RefGate(hidden_size, n_experts, topk, n_groups, topk_groups, score_func, route_scale)
        hf_gate = DeepseekV32Gate(config)

        with torch.no_grad():
            ref_gate.weight.copy_(shared_weight)
            hf_gate.weight.copy_(shared_weight)

        torch.manual_seed(iter_seed + 1)
        x_ref = torch.randn(batch_size, hidden_size, requires_grad=True)
        x_hf = x_ref.clone().detach().requires_grad_(True)

        try:
            ref_weights, ref_indices = ref_gate(x_ref)
            hf_weights, hf_indices = hf_gate(x_hf)

            # Backward through weights (indices are not differentiable)
            torch.manual_seed(iter_seed + 2)
            grad_weights = torch.randn_like(ref_weights)

            ref_weights.backward(grad_weights)
            hf_weights.backward(grad_weights)

            # Compare input gradients
            input_grad_match = torch.allclose(x_ref.grad, x_hf.grad, atol=1e-5, rtol=1e-4)

            # Compare weight gradients
            weight_grad_match = torch.allclose(ref_gate.weight.grad, hf_gate.weight.grad, atol=1e-5, rtol=1e-4)

            if input_grad_match and weight_grad_match:
                stats.record_pass(f"gate_backward_{i}")
            else:
                input_diff = (x_ref.grad - x_hf.grad).abs().max().item() if not input_grad_match else 0
                weight_diff = (ref_gate.weight.grad - hf_gate.weight.grad).abs().max().item() if not weight_grad_match else 0
                stats.record_fail(f"gate_backward_{i}",
                    f"input_grad={input_grad_match} (diff={input_diff:.6e}), "
                    f"weight_grad={weight_grad_match} (diff={weight_diff:.6e})")
        except Exception as e:
            stats.record_fail(f"gate_backward_{i}", f"Exception: {e}")


def fuzz_full_backward(stats: FuzzStats, num_iterations: int = 20):
    """Fuzz test full model backward pass for training stability."""
    from transformers.models.deepseek_v32.configuration_deepseek_v32 import DeepseekV32Config
    from transformers.models.deepseek_v32.modeling_deepseek_v32 import DeepseekV32ForCausalLM

    print("\nFuzzing Full Backward Pass...")

    for i in range(num_iterations):
        vocab_size = random.choice([100, 500])
        hidden_size = random.choice([128, 256])
        num_layers = random.choice([1, 2])
        num_heads = random.choice([2, 4])
        batch = random.randint(1, 2)
        seq_len = random.randint(4, 16)

        torch.manual_seed(i)

        config = DeepseekV32Config(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            intermediate_size=hidden_size * 2,
            moe_intermediate_size=hidden_size // 2,
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            num_key_value_heads=num_heads,
            q_lora_rank=hidden_size // 4,
            kv_lora_rank=hidden_size // 8,
            qk_nope_head_dim=hidden_size // num_heads // 2,
            qk_rope_head_dim=hidden_size // num_heads // 4,
            v_head_dim=hidden_size // num_heads // 2,
            n_routed_experts=4,
            n_shared_experts=1,
            num_experts_per_tok=2,
            n_group=2,
            topk_group=1,
            index_n_heads=2,
            index_head_dim=hidden_size // 4,
            index_topk=min(8, seq_len),
            first_k_dense_replace=1,
        )

        try:
            model = DeepseekV32ForCausalLM(config)
            model.train()

            input_ids = torch.randint(0, vocab_size, (batch, seq_len))
            labels = torch.randint(0, vocab_size, (batch, seq_len))

            # Forward with labels for loss computation
            output = model(input_ids, labels=labels)
            loss = output.loss

            # Check loss is valid
            loss_valid = not (torch.isnan(loss).any() or torch.isinf(loss).any())

            if not loss_valid:
                stats.record_fail(f"backward_{i}", f"Invalid loss: {loss.item()}")
                continue

            # Backward
            loss.backward()

            # Check gradients are valid (no NaN/Inf)
            grads_valid = True
            grad_issues = []
            for name, param in model.named_parameters():
                if param.grad is not None:
                    if torch.isnan(param.grad).any():
                        grads_valid = False
                        grad_issues.append(f"{name}: NaN")
                    if torch.isinf(param.grad).any():
                        grads_valid = False
                        grad_issues.append(f"{name}: Inf")

            if grads_valid:
                stats.record_pass(f"backward_{i}")
            else:
                stats.record_fail(f"backward_{i}",
                    f"Invalid gradients: {grad_issues[:3]}...")
        except Exception as e:
            stats.record_fail(f"backward_{i}", f"Exception: {e}")


def fuzz_moe_backward(stats: FuzzStats, num_iterations: int = 30):
    """Fuzz test MoE backward pass."""
    from transformers.models.deepseek_v32.configuration_deepseek_v32 import DeepseekV32Config
    from transformers.models.deepseek_v32.modeling_deepseek_v32 import DeepseekV32MoE

    print("\nFuzzing MoE Backward...")

    for i in range(num_iterations):
        hidden_size = random.choice([128, 256])
        n_routed = random.choice([4, 8])
        n_shared = random.choice([1, 2])
        topk = random.randint(1, min(3, n_routed))
        n_groups = random.choice([1, 2])
        batch = random.randint(1, 4)
        seq_len = random.randint(1, 16)

        n_routed = n_groups * (n_routed // n_groups)
        if n_routed == 0:
            n_routed = n_groups

        torch.manual_seed(i)

        config = DeepseekV32Config(
            hidden_size=hidden_size,
            moe_intermediate_size=hidden_size // 2,
            n_routed_experts=n_routed,
            n_shared_experts=n_shared,
            num_experts_per_tok=topk,
            n_group=n_groups,
            topk_group=1,
            num_attention_heads=4,
            num_hidden_layers=1,
        )

        try:
            moe = DeepseekV32MoE(config)
            moe.train()

            x = torch.randn(batch, seq_len, hidden_size, requires_grad=True)

            output = moe(x)

            # Check forward is valid
            if torch.isnan(output).any() or torch.isinf(output).any():
                stats.record_fail(f"moe_backward_{i}", "NaN/Inf in forward")
                continue

            # Backward
            torch.manual_seed(i + 1000)
            grad_output = torch.randn_like(output)
            output.backward(grad_output)

            # Check input gradient
            input_grad_valid = x.grad is not None and not (torch.isnan(x.grad).any() or torch.isinf(x.grad).any())

            # Check parameter gradients
            param_grads_valid = True
            for name, param in moe.named_parameters():
                if param.grad is not None:
                    if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                        param_grads_valid = False
                        break

            if input_grad_valid and param_grads_valid:
                stats.record_pass(f"moe_backward_{i}")
            else:
                stats.record_fail(f"moe_backward_{i}",
                    f"input_grad_valid={input_grad_valid}, param_grads_valid={param_grads_valid}")
        except Exception as e:
            stats.record_fail(f"moe_backward_{i}", f"Exception: {e}")


# =============================================================================
# Real Config Tests (exact values from config.json / safetensors)
# =============================================================================

# Real DeepSeek V3.2 671B config values
REAL_CONFIG = {
    "hidden_size": 7168,
    "intermediate_size": 18432,
    "moe_intermediate_size": 2048,
    "num_attention_heads": 128,
    "num_key_value_heads": 128,
    "num_hidden_layers": 61,
    "q_lora_rank": 1536,
    "kv_lora_rank": 512,
    "qk_nope_head_dim": 128,
    "qk_rope_head_dim": 64,
    "v_head_dim": 128,
    "n_routed_experts": 256,
    "n_shared_experts": 1,
    "num_experts_per_tok": 8,
    "n_group": 8,
    "topk_group": 4,
    "scoring_func": "sigmoid",
    "routed_scaling_factor": 2.5,
    "index_n_heads": 64,
    "index_head_dim": 128,
    "index_topk": 2048,
    "first_k_dense_replace": 3,
    "rms_norm_eps": 1e-6,
    "vocab_size": 129280,
}


def test_real_config_dimensions(stats: FuzzStats):
    """Test that model components work with exact real config dimensions."""
    from transformers.models.deepseek_v32.configuration_deepseek_v32 import DeepseekV32Config
    from transformers.models.deepseek_v32.modeling_deepseek_v32 import (
        DeepseekV32Attention,
        DeepseekV32Expert,
        DeepseekV32Gate,
        DeepseekV32MLP,
        DeepseekV32RMSNorm,
    )

    print("\nTesting Real Config Dimensions...")

    # Test RMSNorm with real hidden_size
    try:
        norm = DeepseekV32RMSNorm(REAL_CONFIG["hidden_size"], REAL_CONFIG["rms_norm_eps"])
        x = torch.randn(1, 1, REAL_CONFIG["hidden_size"])
        out = norm(x)
        assert out.shape == x.shape, f"RMSNorm shape mismatch: {out.shape} vs {x.shape}"
        stats.record_pass("real_rmsnorm_dims")
    except Exception as e:
        stats.record_fail("real_rmsnorm_dims", str(e))

    # Test dense MLP (layers 0-2) with real dimensions
    try:
        config = DeepseekV32Config(
            hidden_size=REAL_CONFIG["hidden_size"],
            intermediate_size=REAL_CONFIG["intermediate_size"],
            num_attention_heads=REAL_CONFIG["num_attention_heads"],
            num_hidden_layers=1,
        )
        mlp = DeepseekV32MLP(config)
        x = torch.randn(1, 1, REAL_CONFIG["hidden_size"])
        out = mlp(x)
        assert out.shape == x.shape, f"MLP shape mismatch: {out.shape} vs {x.shape}"
        stats.record_pass("real_mlp_dims")
    except Exception as e:
        stats.record_fail("real_mlp_dims", str(e))

    # Test Expert with real moe_intermediate_size
    try:
        config = DeepseekV32Config(
            hidden_size=REAL_CONFIG["hidden_size"],
            moe_intermediate_size=REAL_CONFIG["moe_intermediate_size"],
            num_attention_heads=REAL_CONFIG["num_attention_heads"],
            num_hidden_layers=1,
        )
        expert = DeepseekV32Expert(config)
        x = torch.randn(1, 1, REAL_CONFIG["hidden_size"])
        out = expert(x)
        assert out.shape == x.shape, f"Expert shape mismatch: {out.shape} vs {x.shape}"
        stats.record_pass("real_expert_dims")
    except Exception as e:
        stats.record_fail("real_expert_dims", str(e))

    # Test Gate with real n_routed_experts, n_group, etc.
    try:
        config = DeepseekV32Config(
            hidden_size=REAL_CONFIG["hidden_size"],
            n_routed_experts=REAL_CONFIG["n_routed_experts"],
            num_experts_per_tok=REAL_CONFIG["num_experts_per_tok"],
            n_group=REAL_CONFIG["n_group"],
            topk_group=REAL_CONFIG["topk_group"],
            scoring_func=REAL_CONFIG["scoring_func"],
            routed_scaling_factor=REAL_CONFIG["routed_scaling_factor"],
            num_attention_heads=REAL_CONFIG["num_attention_heads"],
            num_hidden_layers=1,
        )
        gate = DeepseekV32Gate(config)
        x = torch.randn(1, REAL_CONFIG["hidden_size"])
        weights, indices = gate(x)
        assert weights.shape == (1, REAL_CONFIG["num_experts_per_tok"]), f"Gate weights shape: {weights.shape}"
        assert indices.shape == (1, REAL_CONFIG["num_experts_per_tok"]), f"Gate indices shape: {indices.shape}"
        stats.record_pass("real_gate_dims")
    except Exception as e:
        stats.record_fail("real_gate_dims", str(e))

    # Test Attention with real MLA dimensions
    try:
        config = DeepseekV32Config(
            hidden_size=REAL_CONFIG["hidden_size"],
            num_attention_heads=REAL_CONFIG["num_attention_heads"],
            num_key_value_heads=REAL_CONFIG["num_key_value_heads"],
            q_lora_rank=REAL_CONFIG["q_lora_rank"],
            kv_lora_rank=REAL_CONFIG["kv_lora_rank"],
            qk_nope_head_dim=REAL_CONFIG["qk_nope_head_dim"],
            qk_rope_head_dim=REAL_CONFIG["qk_rope_head_dim"],
            v_head_dim=REAL_CONFIG["v_head_dim"],
            index_n_heads=REAL_CONFIG["index_n_heads"],
            index_head_dim=REAL_CONFIG["index_head_dim"],
            index_topk=64,  # Use smaller for test
            num_hidden_layers=1,
        )
        attn = DeepseekV32Attention(config, layer_idx=0)
        seq_len = 8
        x = torch.randn(1, seq_len, REAL_CONFIG["hidden_size"])
        # Create position embeddings (freqs_cis)
        freqs_cis = random_freqs_cis(seq_len, REAL_CONFIG["qk_rope_head_dim"])
        out, _, _ = attn(x, position_embeddings=freqs_cis)
        assert out.shape == x.shape, f"Attention shape mismatch: {out.shape} vs {x.shape}"
        stats.record_pass("real_attention_dims")
    except Exception as e:
        stats.record_fail("real_attention_dims", str(e))


def test_real_config_gate_routing(stats: FuzzStats):
    """Test Gate routing with exact real config (256 experts, 8 groups, etc.)."""
    from transformers.models.deepseek_v32.configuration_deepseek_v32 import DeepseekV32Config
    from transformers.models.deepseek_v32.modeling_deepseek_v32 import DeepseekV32Gate

    print("\nTesting Real Config Gate Routing...")

    config = DeepseekV32Config(
        hidden_size=REAL_CONFIG["hidden_size"],
        n_routed_experts=REAL_CONFIG["n_routed_experts"],  # 256
        num_experts_per_tok=REAL_CONFIG["num_experts_per_tok"],  # 8
        n_group=REAL_CONFIG["n_group"],  # 8
        topk_group=REAL_CONFIG["topk_group"],  # 4
        scoring_func=REAL_CONFIG["scoring_func"],  # sigmoid
        routed_scaling_factor=REAL_CONFIG["routed_scaling_factor"],  # 2.5
        num_attention_heads=REAL_CONFIG["num_attention_heads"],
        num_hidden_layers=1,
    )

    torch.manual_seed(42)
    gate = DeepseekV32Gate(config)
    ref_gate = RefGate(
        REAL_CONFIG["hidden_size"],
        REAL_CONFIG["n_routed_experts"],
        REAL_CONFIG["num_experts_per_tok"],
        REAL_CONFIG["n_group"],
        REAL_CONFIG["topk_group"],
        REAL_CONFIG["scoring_func"],
        REAL_CONFIG["routed_scaling_factor"],
    )

    # Copy weights and bias
    with torch.no_grad():
        ref_gate.weight.copy_(gate.weight)
        if gate.e_score_correction_bias is not None and ref_gate.bias is not None:
            ref_gate.bias.copy_(gate.e_score_correction_bias)

    # Test with batch
    for batch_size in [1, 4, 16]:
        torch.manual_seed(batch_size)
        x = torch.randn(batch_size, REAL_CONFIG["hidden_size"])

        try:
            ref_weights, ref_indices = ref_gate(x)
            hf_weights, hf_indices = gate(x)

            # Check indices (sorted)
            ref_sorted = ref_indices.sort(dim=-1)[0]
            hf_sorted = hf_indices.sort(dim=-1)[0]
            indices_match = torch.equal(ref_sorted, hf_sorted)

            # Check expert->weight mapping
            ref_expert_weights = torch.zeros(batch_size, REAL_CONFIG["n_routed_experts"])
            hf_expert_weights = torch.zeros(batch_size, REAL_CONFIG["n_routed_experts"])

            for b in range(batch_size):
                for j in range(REAL_CONFIG["num_experts_per_tok"]):
                    ref_idx = ref_indices[b, j].item()
                    hf_idx = hf_indices[b, j].item()
                    ref_expert_weights[b, ref_idx] = ref_weights[b, j]
                    hf_expert_weights[b, hf_idx] = hf_weights[b, j]

            weights_match = torch.allclose(ref_expert_weights, hf_expert_weights, atol=1e-5, rtol=1e-4)

            if indices_match and weights_match:
                stats.record_pass(f"real_gate_routing_batch{batch_size}")
            else:
                stats.record_fail(f"real_gate_routing_batch{batch_size}",
                    f"indices_match={indices_match}, weights_match={weights_match}")
        except Exception as e:
            stats.record_fail(f"real_gate_routing_batch{batch_size}", str(e))


def test_real_config_backward(stats: FuzzStats):
    """Test backward pass with real config dimensions."""
    from transformers.models.deepseek_v32.configuration_deepseek_v32 import DeepseekV32Config
    from transformers.models.deepseek_v32.modeling_deepseek_v32 import (
        DeepseekV32Expert,
        DeepseekV32Gate,
        DeepseekV32MLP,
        DeepseekV32RMSNorm,
    )

    print("\nTesting Real Config Backward...")

    # Test RMSNorm backward
    try:
        norm = DeepseekV32RMSNorm(REAL_CONFIG["hidden_size"], REAL_CONFIG["rms_norm_eps"])
        x = torch.randn(1, 4, REAL_CONFIG["hidden_size"], requires_grad=True)
        out = norm(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None and not torch.isnan(x.grad).any()
        stats.record_pass("real_rmsnorm_backward")
    except Exception as e:
        stats.record_fail("real_rmsnorm_backward", str(e))

    # Test MLP backward
    try:
        config = DeepseekV32Config(
            hidden_size=REAL_CONFIG["hidden_size"],
            intermediate_size=REAL_CONFIG["intermediate_size"],
            num_attention_heads=REAL_CONFIG["num_attention_heads"],
            num_hidden_layers=1,
        )
        mlp = DeepseekV32MLP(config)
        x = torch.randn(1, 4, REAL_CONFIG["hidden_size"], requires_grad=True)
        out = mlp(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None and not torch.isnan(x.grad).any()
        stats.record_pass("real_mlp_backward")
    except Exception as e:
        stats.record_fail("real_mlp_backward", str(e))

    # Test Expert backward
    try:
        config = DeepseekV32Config(
            hidden_size=REAL_CONFIG["hidden_size"],
            moe_intermediate_size=REAL_CONFIG["moe_intermediate_size"],
            num_attention_heads=REAL_CONFIG["num_attention_heads"],
            num_hidden_layers=1,
        )
        expert = DeepseekV32Expert(config)
        x = torch.randn(1, 4, REAL_CONFIG["hidden_size"], requires_grad=True)
        out = expert(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None and not torch.isnan(x.grad).any()
        stats.record_pass("real_expert_backward")
    except Exception as e:
        stats.record_fail("real_expert_backward", str(e))

    # Test Gate backward
    try:
        config = DeepseekV32Config(
            hidden_size=REAL_CONFIG["hidden_size"],
            n_routed_experts=REAL_CONFIG["n_routed_experts"],
            num_experts_per_tok=REAL_CONFIG["num_experts_per_tok"],
            n_group=REAL_CONFIG["n_group"],
            topk_group=REAL_CONFIG["topk_group"],
            scoring_func=REAL_CONFIG["scoring_func"],
            routed_scaling_factor=REAL_CONFIG["routed_scaling_factor"],
            num_attention_heads=REAL_CONFIG["num_attention_heads"],
            num_hidden_layers=1,
        )
        gate = DeepseekV32Gate(config)
        x = torch.randn(4, REAL_CONFIG["hidden_size"], requires_grad=True)
        weights, indices = gate(x)
        loss = weights.sum()
        loss.backward()
        assert x.grad is not None and not torch.isnan(x.grad).any()
        stats.record_pass("real_gate_backward")
    except Exception as e:
        stats.record_fail("real_gate_backward", str(e))


def test_weight_names_match_safetensors(stats: FuzzStats):
    """Verify that model weight names match the safetensors index."""
    from transformers.models.deepseek_v32.configuration_deepseek_v32 import DeepseekV32Config
    from transformers.models.deepseek_v32.modeling_deepseek_v32 import DeepseekV32ForCausalLM

    print("\nTesting Weight Names Match Safetensors...")

    # Expected weight patterns from safetensors index
    expected_patterns = [
        # Embedding
        "model.embed_tokens.weight",
        # Attention (MLA)
        "model.layers.0.self_attn.q_a_proj.weight",
        "model.layers.0.self_attn.q_a_layernorm.weight",
        "model.layers.0.self_attn.q_b_proj.weight",
        "model.layers.0.self_attn.kv_a_proj_with_mqa.weight",
        "model.layers.0.self_attn.kv_a_layernorm.weight",
        "model.layers.0.self_attn.kv_b_proj.weight",
        "model.layers.0.self_attn.o_proj.weight",
        # Indexer
        "model.layers.0.self_attn.indexer.wq_b.weight",
        "model.layers.0.self_attn.indexer.wk.weight",
        "model.layers.0.self_attn.indexer.k_norm.weight",
        "model.layers.0.self_attn.indexer.k_norm.bias",
        "model.layers.0.self_attn.indexer.weights_proj.weight",
        # Dense MLP (layer 0-2)
        "model.layers.0.mlp.gate_proj.weight",
        "model.layers.0.mlp.up_proj.weight",
        "model.layers.0.mlp.down_proj.weight",
        # Layer norms
        "model.layers.0.input_layernorm.weight",
        "model.layers.0.post_attention_layernorm.weight",
    ]

    # MoE layer patterns (layer >= first_k_dense_replace)
    # Note: e_score_correction_bias is optional (only present in full 671B model)
    moe_patterns = [
        "model.layers.3.mlp.gate.weight",
        "model.layers.3.mlp.shared_experts.gate_proj.weight",
        "model.layers.3.mlp.shared_experts.up_proj.weight",
        "model.layers.3.mlp.shared_experts.down_proj.weight",
        "model.layers.3.mlp.experts.0.gate_proj.weight",
        "model.layers.3.mlp.experts.0.up_proj.weight",
        "model.layers.3.mlp.experts.0.down_proj.weight",
    ]

    # Optional patterns (may not be present in small test models)
    optional_patterns = [
        "model.layers.3.mlp.gate.e_score_correction_bias",  # Only in 671B model
    ]

    # Create small model to check weight names
    config = DeepseekV32Config(
        vocab_size=1000,
        hidden_size=256,
        intermediate_size=512,
        moe_intermediate_size=128,
        num_hidden_layers=4,  # Need >= 4 to test MoE layers
        num_attention_heads=4,
        num_key_value_heads=4,
        q_lora_rank=64,
        kv_lora_rank=32,
        qk_nope_head_dim=32,
        qk_rope_head_dim=16,
        v_head_dim=32,
        n_routed_experts=8,
        n_shared_experts=1,
        num_experts_per_tok=2,
        n_group=2,
        topk_group=1,
        index_n_heads=2,
        index_head_dim=32,
        index_topk=16,
        first_k_dense_replace=3,  # Layers 0,1,2 are dense; 3+ are MoE
    )

    try:
        model = DeepseekV32ForCausalLM(config)
        model_weights = set(model.state_dict().keys())

        # Check dense layer patterns
        for pattern in expected_patterns:
            if pattern in model_weights:
                stats.record_pass(f"weight_name_{pattern.split('.')[-1]}")
            else:
                # Check if pattern exists with different layer idx
                base = pattern.replace("layers.0", "layers.{}")
                found = any(base.format(i) in model_weights for i in range(4))
                if found:
                    stats.record_pass(f"weight_name_{pattern.split('.')[-1]}")
                else:
                    stats.record_fail(f"weight_name_{pattern.split('.')[-1]}", f"Missing: {pattern}")

        # Check MoE layer patterns
        for pattern in moe_patterns:
            if pattern in model_weights:
                stats.record_pass(f"weight_name_moe_{pattern.split('.')[-1]}")
            else:
                stats.record_fail(f"weight_name_moe_{pattern.split('.')[-1]}", f"Missing: {pattern}")

    except Exception as e:
        stats.record_fail("weight_names_check", str(e))


# =============================================================================
# Main
# =============================================================================


def main():
    print("=" * 60)
    print("DeepSeek V3.2 Fuzz Testing")
    print("=" * 60)

    # Set random seed for reproducibility
    seed = random.randint(0, 2**32 - 1)
    print(f"\nMaster seed: {seed}")
    print("(Use this seed to reproduce failures)")
    random.seed(seed)

    stats = FuzzStats()

    # Run forward fuzz tests
    print("\n" + "=" * 60)
    print("FORWARD PASS TESTS")
    print("=" * 60)
    fuzz_rope(stats, num_iterations=100)
    fuzz_rope_edge_cases(stats, num_iterations=50)
    fuzz_rmsnorm(stats, num_iterations=100)
    fuzz_mlp(stats, num_iterations=50)
    fuzz_expert(stats, num_iterations=50)
    fuzz_gate(stats, num_iterations=50)
    fuzz_attention_components(stats, num_iterations=30)
    fuzz_attention_forward(stats, num_iterations=30)
    fuzz_attention_indexer_effect(stats, num_iterations=20)
    fuzz_indexer_components(stats, num_iterations=30)
    fuzz_indexer_vs_reference(stats, num_iterations=30)
    fuzz_attention_vs_reference(stats, num_iterations=30)
    fuzz_moe_routing(stats, num_iterations=30)
    fuzz_full_forward(stats, num_iterations=20)

    # Run backward fuzz tests
    print("\n" + "=" * 60)
    print("BACKWARD PASS TESTS")
    print("=" * 60)
    fuzz_rope_backward(stats, num_iterations=50)
    fuzz_rmsnorm_backward(stats, num_iterations=50)
    fuzz_mlp_backward(stats, num_iterations=50)
    fuzz_expert_backward(stats, num_iterations=50)
    fuzz_gate_backward(stats, num_iterations=50)
    # Note: fuzz_indexer_backward_vs_reference skipped - indexer uses @torch.no_grad()
    # and topk indices are not differentiable. Gradients flow through attention instead.
    fuzz_attention_backward_vs_reference(stats, num_iterations=30)
    fuzz_moe_backward(stats, num_iterations=30)
    fuzz_full_backward(stats, num_iterations=20)

    # Real config tests (exact values from config.json / safetensors)
    print("\n" + "=" * 60)
    print("REAL CONFIG TESTS (671B model dimensions)")
    print("=" * 60)
    test_real_config_dimensions(stats)
    test_real_config_gate_routing(stats)
    test_real_config_backward(stats)
    test_weight_names_match_safetensors(stats)

    # Print summary
    success = stats.summary()

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
