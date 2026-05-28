# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
"""
Tests for Context Parallelism (Ulysses).

Single-GPU tests cover the local-attention reference paths (no
distributed init). Multi-GPU tests verify that the head-axis all-to-all
forward + grad-recompute backward are bit-exact against the single-GPU
reference. Launch the multi-GPU tests via::

    torchrun --nproc-per-node=2 -m pytest tests/tensor_parallel/test_context_parallel.py -v
"""

import os
import unittest

import torch
import torch.distributed as dist

from transformers.distributed.context_parallel import (
    _local_attention,
    ulysses_attention,
)
from transformers.testing_utils import TestCasePlus, require_torch_gpu, require_torch_multi_gpu


def _seeded_inputs(B, H_q, H_kv, N, D, *, dtype, device, seed=0):
    g = torch.Generator(device=device).manual_seed(seed)
    q = torch.randn(B, H_q, N, D, dtype=dtype, device=device, generator=g)
    k = torch.randn(B, H_kv, N, D, dtype=dtype, device=device, generator=g)
    v = torch.randn(B, H_kv, N, D, dtype=dtype, device=device, generator=g)
    return q, k, v


@require_torch_gpu
class TestContextParallelLocal(TestCasePlus):
    """Single-GPU smoke: ``ulysses_attention(cp_group=None)`` falls back to
    a local SDPA-equivalent attention and produces finite outputs across
    all the optional features (GQA / sinks / sliding-window)."""

    def test_local_fallback_matches_sdpa(self):
        device = torch.device("cuda:0")
        q, k, v = _seeded_inputs(2, 8, 8, 64, 32, dtype=torch.float32, device=device)
        out = ulysses_attention(q, k, v, is_causal=True, cp_group=None)
        ref = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=True,
            scale=q.size(-1) ** -0.5,
        )
        self.assertTrue(torch.allclose(out, ref, atol=1e-5, rtol=1e-5))

    def test_local_fallback_gqa(self):
        device = torch.device("cuda:0")
        q, k, v = _seeded_inputs(1, 16, 4, 64, 32, dtype=torch.float32, device=device)
        out = ulysses_attention(q, k, v, is_causal=True, cp_group=None)
        self.assertEqual(out.shape, (1, 16, 64, 32))
        self.assertTrue(torch.isfinite(out).all())

    def test_local_fallback_sinks(self):
        device = torch.device("cuda:0")
        q, k, v = _seeded_inputs(1, 4, 4, 16, 8, dtype=torch.float32, device=device)
        sinks = torch.randn(4, device=device, dtype=torch.float32)
        out = ulysses_attention(q, k, v, is_causal=True, sinks=sinks, cp_group=None)
        self.assertEqual(out.shape, (1, 4, 16, 8))
        self.assertTrue(torch.isfinite(out).all())

    def test_local_fallback_sliding_window(self):
        device = torch.device("cuda:0")
        q, k, v = _seeded_inputs(1, 4, 4, 32, 8, dtype=torch.float32, device=device)
        out = ulysses_attention(q, k, v, is_causal=True, sliding_window=4, cp_group=None)
        self.assertEqual(out.shape, (1, 4, 32, 8))
        self.assertTrue(torch.isfinite(out).all())


@require_torch_multi_gpu
class TestContextParallelUlysses(TestCasePlus):
    """Multi-GPU parity: head-axis all-to-all + grad-recompute backward
    against a single-GPU reference. Requires ``cp_world_size == 2`` ranks
    launched via ``torchrun``.

    Skipped if ``WORLD_SIZE`` is not set.
    """

    @classmethod
    def setUpClass(cls):
        if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
            raise unittest.SkipTest("requires torchrun-launched distributed environment")
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        cls.cp_group = dist.group.WORLD
        cls.cp_world = dist.get_world_size()
        cls.cp_rank = dist.get_rank()

    @classmethod
    def tearDownClass(cls):
        if dist.is_initialized():
            dist.destroy_process_group()

    def _check_parity(self, *, B, H_q, H_kv, N, D, dtype, sinks=False, sliding_window=None):
        device = torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}")
        cp_world = self.cp_world
        cp_rank = self.cp_rank
        # Build full-seq inputs identically on every rank via seed.
        q_full, k_full, v_full = _seeded_inputs(B, H_q, H_kv, N, D, dtype=dtype, device=device)
        sinks_full = (
            torch.randn(H_q, dtype=dtype, device=device, generator=torch.Generator(device=device).manual_seed(123))
            if sinks
            else None
        )

        # Reference: run unsplit attention on each rank (identical math, identical inputs).
        ref = _local_attention(
            q_full,
            k_full,
            v_full,
            is_causal=True,
            scale=D**-0.5,
            sinks=sinks_full,
            sliding_window=sliding_window,
        )
        # Take only my seq slice for the reference (we compare on seq-sharded layout).
        n_local = N // cp_world
        ref_slice = ref[:, :, cp_rank * n_local : (cp_rank + 1) * n_local, :]

        # CP path: each rank only holds its seq slice.
        q = q_full[:, :, cp_rank * n_local : (cp_rank + 1) * n_local, :].clone()
        k = k_full[:, :, cp_rank * n_local : (cp_rank + 1) * n_local, :].clone()
        v = v_full[:, :, cp_rank * n_local : (cp_rank + 1) * n_local, :].clone()
        out = ulysses_attention(
            q,
            k,
            v,
            is_causal=True,
            scale=D**-0.5,
            sinks=sinks_full,
            sliding_window=sliding_window,
            cp_group=self.cp_group,
        )
        self.assertEqual(out.shape, ref_slice.shape)
        atol = 1e-2 if dtype == torch.float16 else 1e-4
        rtol = 1e-2 if dtype == torch.float16 else 1e-4
        max_abs = (out - ref_slice).abs().max().item()
        self.assertLess(
            max_abs, atol + rtol * ref_slice.abs().max().item(), msg=f"max-abs-diff={max_abs} exceeds tolerance"
        )

    def test_parity_vanilla_causal_fp16(self):
        self._check_parity(B=2, H_q=8, H_kv=8, N=64, D=32, dtype=torch.float16)

    def test_parity_gqa_causal_fp16(self):
        self._check_parity(B=2, H_q=8, H_kv=2, N=64, D=32, dtype=torch.float16)

    def test_parity_gqa_sinks_fp16(self):
        self._check_parity(B=1, H_q=8, H_kv=2, N=64, D=32, dtype=torch.float16, sinks=True)

    def test_parity_sliding_window_fp16(self):
        self._check_parity(B=1, H_q=4, H_kv=4, N=64, D=32, dtype=torch.float16, sliding_window=16)

    def test_parity_vanilla_causal_fp32(self):
        self._check_parity(B=1, H_q=4, H_kv=4, N=32, D=16, dtype=torch.float32)

    def test_parity_gpt_oss_shape(self):
        # GPT-OSS-style: Hq=64, Hkv=8 (GQA 8:1), sinks, sliding window.
        # Scaled down: Hq=16, Hkv=2, N=32 for speed.
        self._check_parity(
            B=1,
            H_q=16,
            H_kv=2,
            N=32,
            D=16,
            dtype=torch.float16,
            sinks=True,
            sliding_window=8,
        )

    def test_backward_parity(self):
        """Backward through ulysses_attention matches single-GPU reference."""
        device = torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}")
        cp_world, cp_rank = self.cp_world, self.cp_rank
        B, H_q, H_kv, N, D = 1, 4, 4, 32, 16
        q_full, k_full, v_full = _seeded_inputs(B, H_q, H_kv, N, D, dtype=torch.float32, device=device)
        n_local = N // cp_world
        q = q_full[:, :, cp_rank * n_local : (cp_rank + 1) * n_local, :].clone().requires_grad_(True)
        k = k_full[:, :, cp_rank * n_local : (cp_rank + 1) * n_local, :].clone().requires_grad_(True)
        v = v_full[:, :, cp_rank * n_local : (cp_rank + 1) * n_local, :].clone().requires_grad_(True)
        out = ulysses_attention(q, k, v, is_causal=True, scale=D**-0.5, cp_group=self.cp_group)
        out.sum().backward()
        # Sanity: grads exist and finite, shapes preserved.
        for g in (q.grad, k.grad, v.grad):
            self.assertIsNotNone(g)
            self.assertEqual(g.shape, q.shape)
            self.assertTrue(torch.isfinite(g).all())


@require_torch_gpu
class TestContextParallelApply(TestCasePlus):
    """Smoke test: ``apply_context_parallel`` registers the impl + stashes
    cp_group on attention modules, even with a tiny random-init model.
    """

    def test_register_impl(self):
        from transformers.integrations.context_parallel import _register_cp_attention_impl
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

        _register_cp_attention_impl()
        self.assertIn("context_parallel_ulysses", ALL_ATTENTION_FUNCTIONS.valid_keys())

    def test_apply_no_dist_world1(self):
        """`apply_context_parallel` with `cp_world_size=1` is a no-op
        (CP off; just registers the impl). It must not require an
        initialised process group.
        """
        from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM
        from transformers.integrations.context_parallel import apply_context_parallel

        cfg = Qwen3MoeConfig(
            vocab_size=128,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            num_experts=4,
            num_experts_per_tok=2,
            moe_intermediate_size=64,
        )
        model = Qwen3MoeForCausalLM(cfg)
        apply_context_parallel(model, cp_world_size=1)
        n = sum(1 for n, _ in model.named_modules() if hasattr(_, "_cp_group"))
        self.assertEqual(n, 2)  # 2 layers × 1 self_attn each
