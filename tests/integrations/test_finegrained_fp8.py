# Copyright 2026 The HuggingFace Team. All rights reserved.
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
"""Integration tests for the finegrained-fp8 Triton kernel bindings.

Two layers, both mocking only what needs the `kernels-community/finegrained-fp8` hub build:

* `FinegrainedFp8LoaderTest` mocks `is_kernels_available` and `lazy_load_kernel` to drive
  `load_finegrained_fp8_kernel` with no hub download — asserting it gates correctly (raises when a
  precondition is unmet, returns the bundle otherwise) and stays torch-compile safe.
* `FinegrainedFp8ForwardTest` mocks only the loaded kernel bundle (`matmul` / `batched_matmul` /
  `grouped_matmul`) with capturing fakes that return correctly-shaped tensors, and runs the real
  `finegrained_fp8_linear`, `fp8_batched_mm_experts_forward` and `fp8_grouped_mm_experts_forward` on a
  CUDA device so their repeat-interleave / routing-flatten / sort-and-histogram / gating / sentinel-mask
  / reshape-sum glue executes for real; it then asserts the tensors handed to the kernel are what the
  kernel expects (flattened routing, unclamped sentinels, weight/scale pairing, block_size, ...) and
  the value-exact output the surrounding marshalling produces.
"""

import contextlib
import types
import unittest
from unittest import mock

import torch

import transformers.integrations.finegrained_fp8 as fg
from transformers.integrations.finegrained_fp8 import (
    finegrained_fp8_linear,
    fp8_batched_mm_experts_forward,
    fp8_grouped_mm_experts_forward,
)
from transformers.testing_utils import require_torch, require_torch_gpu


def _add_one(x, *args, **kwargs):
    return x + 1


# ── Fake kernels (a "good" one exposing every symbol the loader resolves, plus a variant missing one) ──


class _FakeFinegrainedKernel:
    matmul_2d = staticmethod(_add_one)
    matmul_batched = staticmethod(_add_one)
    matmul_grouped = staticmethod(_add_one)


class _FinegrainedKernelMissingSymbol:
    matmul_2d = staticmethod(_add_one)  # missing matmul_batched / matmul_grouped


@require_torch
class FinegrainedFp8LoaderTest(unittest.TestCase):
    def setUp(self):
        fg._FINEGRAINED_FP8 = None
        self.addCleanup(setattr, fg, "_FINEGRAINED_FP8", None)

    def _env(self, *, kernels_available=True, kernel=_FakeFinegrainedKernel):
        stack = contextlib.ExitStack()
        stack.enter_context(mock.patch.object(fg, "is_kernels_available", return_value=kernels_available))
        stack.enter_context(mock.patch.object(fg, "lazy_load_kernel", return_value=kernel))
        return stack

    def test_loads_when_environment_is_valid(self):
        with self._env():
            bundle = fg.load_finegrained_fp8_kernel()
        self.assertIsInstance(bundle, fg.FineGrainedFP8)
        self.assertIs(bundle.matmul, _add_one)

    def test_raises_without_kernels_package(self):
        with self._env(kernels_available=False), self.assertRaisesRegex(ImportError, "requires the .kernels. package"):
            fg.load_finegrained_fp8_kernel()

    def test_raises_when_kernel_fails_to_load(self):
        with self._env(kernel=None), self.assertRaisesRegex(ImportError, "Failed to load the finegrained-fp8 kernel"):
            fg.load_finegrained_fp8_kernel()

    def test_raises_on_missing_symbols(self):
        with (
            self._env(kernel=_FinegrainedKernelMissingSymbol),
            self.assertRaisesRegex(ImportError, "missing required symbols"),
        ):
            fg.load_finegrained_fp8_kernel()

    def test_loader_is_compile_safe(self):
        with self._env():
            torch.compiler.reset()

            @torch.compile(backend="aot_eager", fullgraph=True)
            def run(x):
                return fg.load_finegrained_fp8_kernel().matmul(x)

            out = run(torch.zeros(3))
        self.assertTrue(torch.equal(out, torch.ones(3)))


def _make_fp8_experts(
    *,
    num_experts=4,
    hidden=8,
    inter=16,
    has_gate=True,
    activation_scheme="dynamic",
    block_size=(128, 128),
    dtype=torch.float8_e4m3fn,
    device="cuda",
):
    """A minimal stand-in for `FP8Experts` carrying exactly what the experts forwards read.

    Weights/scales are arbitrary (the kernel is faked), but they are the same tensor objects the
    forward passes to the kernel, so `torch.equal` on them checks the weight/scale marshalling exactly.
    Gating uses the real `FP8Experts._apply_gate`, bound so the 2*inter -> inter collapse runs for real.
    """
    experts = types.SimpleNamespace(
        num_experts=num_experts,
        has_gate=has_gate,
        activation_scheme=activation_scheme,
        block_size=block_size,
        down_proj=torch.randn(num_experts, hidden, inter, device=device).to(dtype),
        down_proj_scale_inv=torch.randn(num_experts, 1, 1, dtype=torch.float32, device=device),
        act_fn=torch.nn.functional.silu,
        swiglu_alpha=None,
        swiglu_limit=None,
        limit=None,
    )
    if has_gate:
        experts.gate_up_proj = torch.randn(num_experts, 2 * inter, hidden, device=device).to(dtype)
        experts.gate_up_proj_scale_inv = torch.randn(num_experts, 2, 1, dtype=torch.float32, device=device)
    else:
        experts.up_proj = torch.randn(num_experts, inter, hidden, device=device).to(dtype)
        experts.up_proj_scale_inv = torch.randn(num_experts, 1, 1, dtype=torch.float32, device=device)
    experts._apply_gate = types.MethodType(fg.FP8Experts._apply_gate, experts)
    return experts


@require_torch_gpu
class FinegrainedFp8ForwardTest(unittest.TestCase):
    """Drives the real finegrained-fp8 forwards with only the loaded kernel bundle mocked."""

    def setUp(self):
        fg._FINEGRAINED_FP8 = None
        self.addCleanup(setattr, fg, "_FINEGRAINED_FP8", None)

    @contextlib.contextmanager
    def _mocked_kernel(self):
        # Each fake records every call and returns a rightly-shaped, deterministic tensor:
        # `matmul` fills 3.0 (so the bias add is value-checkable); the batched/grouped experts fakes
        # return ones (so the routing-weight * mask * reshape-sum reduction is analytically exact and
        # independent of the — non-stable on CUDA — expert sort).
        calls = {"matmul": [], "batched_matmul": [], "grouped_matmul": []}

        def fake_matmul(input, weight, weight_scale_inv, block_size, output_dtype, *, activation_scale=None):
            calls["matmul"].append(
                {
                    "input": input,
                    "weight": weight,
                    "weight_scale_inv": weight_scale_inv,
                    "block_size": block_size,
                    "output_dtype": output_dtype,
                    "activation_scale": activation_scale,
                }
            )
            return torch.full((input.shape[0], weight.shape[0]), 3.0, dtype=output_dtype, device=input.device)

        def fake_batched(input, weight, weight_scale, *, block_size, expert_ids):
            calls["batched_matmul"].append(
                {
                    "input": input,
                    "weight": weight,
                    "weight_scale": weight_scale,
                    "block_size": block_size,
                    "expert_ids": expert_ids,
                }
            )
            return torch.ones(input.shape[0], weight.shape[1], dtype=torch.float32, device=input.device)

        def fake_grouped(input, weight, weight_scale, *, offsets, tokens_per_expert, block_size):
            calls["grouped_matmul"].append(
                {
                    "input": input,
                    "weight": weight,
                    "weight_scale": weight_scale,
                    "offsets": offsets,
                    "tokens_per_expert": tokens_per_expert,
                    "block_size": block_size,
                }
            )
            return torch.ones(input.shape[0], weight.shape[1], dtype=torch.float32, device=input.device)

        bundle = fg.FineGrainedFP8(matmul=fake_matmul, batched_matmul=fake_batched, grouped_matmul=fake_grouped)
        with mock.patch.object(fg, "load_finegrained_fp8_kernel", return_value=bundle):
            yield calls

    @staticmethod
    def _expected_experts_output(hidden_states, top_k_index, top_k_weights, num_experts):
        # Mirror the shared tail of both experts forwards with the ones-returning down projection:
        # weighted_out = ones * routing_weight, sentinel rows zeroed, then per-token reduction.
        num_tokens, top_k = top_k_index.shape
        hidden = hidden_states.size(-1)
        sw = top_k_weights.reshape(-1).to(torch.float32)
        weighted = torch.ones(sw.shape[0], hidden, device=hidden_states.device) * sw.unsqueeze(-1)
        sentinel = (top_k_index.reshape(-1) >= num_experts).unsqueeze(-1)
        weighted = weighted.masked_fill(sentinel, 0.0)
        return weighted.view(num_tokens, top_k, hidden).sum(dim=1).to(hidden_states.dtype)

    # ── finegrained_fp8_linear ────────────────────────────────────────────────────────────────────

    def test_linear_marshals_args_and_defaults_output_dtype(self):
        input = torch.randn(3, 8, dtype=torch.bfloat16, device="cuda")
        weight = torch.randn(16, 8, device="cuda").to(torch.float8_e4m3fn)
        weight_scale_inv = torch.randn(1, 1, dtype=torch.float32, device="cuda")
        block_size = [128, 128]
        with self._mocked_kernel() as calls:
            out = finegrained_fp8_linear(input, weight, weight_scale_inv, block_size)
        call = calls["matmul"][0]
        # input / weight / scale / block_size pass straight through, positionally.
        self.assertIs(call["input"], input)
        self.assertIs(call["weight"], weight)
        self.assertIs(call["weight_scale_inv"], weight_scale_inv)
        self.assertIs(call["block_size"], block_size)
        # output_dtype defaults to input.dtype when the caller leaves it None, and no activation scale.
        self.assertEqual(call["output_dtype"], torch.bfloat16)
        self.assertIsNone(call["activation_scale"])
        # No bias -> the kernel output is returned untouched.
        self.assertEqual(out.shape, (3, 16))
        self.assertEqual(out.dtype, torch.bfloat16)
        self.assertTrue(torch.equal(out, torch.full((3, 16), 3.0, dtype=torch.bfloat16, device="cuda")))

    def test_linear_forwards_explicit_output_dtype_and_activation_scale(self):
        input = torch.randn(3, 8, dtype=torch.bfloat16, device="cuda")
        weight = torch.randn(16, 8, device="cuda").to(torch.float8_e4m3fn)
        weight_scale_inv = torch.randn(1, 1, dtype=torch.float32, device="cuda")
        activation_scale = torch.tensor(2.0, device="cuda")
        with self._mocked_kernel() as calls:
            out = finegrained_fp8_linear(
                input,
                weight,
                weight_scale_inv,
                block_size=None,
                activation_scale=activation_scale,
                output_dtype=torch.float32,
            )
        call = calls["matmul"][0]
        self.assertIsNone(call["block_size"])
        self.assertEqual(call["output_dtype"], torch.float32)
        self.assertIs(call["activation_scale"], activation_scale)
        self.assertEqual(out.dtype, torch.float32)

    def test_linear_adds_bias_in_place(self):
        input = torch.randn(3, 8, dtype=torch.bfloat16, device="cuda")
        weight = torch.randn(16, 8, device="cuda").to(torch.float8_e4m3fn)
        weight_scale_inv = torch.randn(1, 1, dtype=torch.float32, device="cuda")
        bias = torch.randn(16, dtype=torch.bfloat16, device="cuda")
        with self._mocked_kernel():
            out = finegrained_fp8_linear(input, weight, weight_scale_inv, [128, 128], bias=bias)
        # The kernel returned a 3.0-filled (3, 16) tensor; bias is broadcast-added onto it.
        self.assertTrue(torch.equal(out, torch.full((3, 16), 3.0, dtype=torch.bfloat16, device="cuda") + bias))

    # ── fp8_batched_mm_experts_forward ──────────────────────────────────────────────────────────────

    def test_batched_mm_kernel_inputs_and_output(self):
        experts = _make_fp8_experts(num_experts=4, hidden=8, inter=16)
        hidden_states = torch.randn(3, 8, dtype=torch.bfloat16, device="cuda")
        top_k_index = torch.randint(0, 4, (3, 2), device="cuda")
        top_k_weights = torch.rand(3, 2, dtype=torch.bfloat16, device="cuda")
        with self._mocked_kernel() as calls:
            out = fp8_batched_mm_experts_forward(experts, hidden_states, top_k_index, top_k_weights)

        up, down = calls["batched_matmul"]
        # Up projection: each token replicated top_k times (S = 6), routing flattened row-major.
        self.assertTrue(torch.equal(up["input"], hidden_states.repeat_interleave(2, dim=0)))
        self.assertEqual(up["input"].shape, (6, 8))
        self.assertEqual(up["expert_ids"].shape, (6,))
        self.assertTrue(torch.equal(up["expert_ids"], top_k_index.reshape(-1)))
        # Weight/scale pairing and block_size pass straight through (no copies).
        self.assertIs(up["weight"], experts.gate_up_proj)
        self.assertIs(up["weight_scale"], experts.gate_up_proj_scale_inv)
        self.assertIs(up["block_size"], experts.block_size)
        # Down projection sees the gated activation: 2*inter (32) collapsed to inter (16).
        self.assertEqual(down["input"].shape, (6, 16))
        self.assertIs(down["weight"], experts.down_proj)
        self.assertIs(down["weight_scale"], experts.down_proj_scale_inv)
        self.assertTrue(torch.equal(down["expert_ids"], top_k_index.reshape(-1)))

        # Output: (num_tokens, hidden), recast to the input dtype, value-exact reduction.
        self.assertEqual(out.shape, (3, 8))
        self.assertEqual(out.dtype, torch.bfloat16)
        expected = self._expected_experts_output(hidden_states, top_k_index, top_k_weights, experts.num_experts)
        self.assertTrue(torch.equal(out, expected))

    def test_batched_mm_non_gated_uses_up_proj_and_activation(self):
        experts = _make_fp8_experts(num_experts=4, hidden=8, inter=16, has_gate=False)
        hidden_states = torch.randn(3, 8, dtype=torch.bfloat16, device="cuda")
        top_k_index = torch.randint(0, 4, (3, 2), device="cuda")
        top_k_weights = torch.rand(3, 2, dtype=torch.bfloat16, device="cuda")
        with self._mocked_kernel() as calls:
            out = fp8_batched_mm_experts_forward(experts, hidden_states, top_k_index, top_k_weights)
        up, down = calls["batched_matmul"]
        self.assertIs(up["weight"], experts.up_proj)
        self.assertIs(up["weight_scale"], experts.up_proj_scale_inv)
        # Non-gated: act_fn keeps the inter dim (16), no 2*inter chunk.
        self.assertEqual(down["input"].shape, (6, 16))
        expected = self._expected_experts_output(hidden_states, top_k_index, top_k_weights, experts.num_experts)
        self.assertTrue(torch.equal(out, expected))

    def test_batched_mm_passes_sentinel_expert_ids_unclamped(self):
        # EP sentinels (expert_ids >= num_experts) reach the kernel unclamped; the post-mask zeroes the
        # matching output rows before the per-token reduction (the kernel leaves them uninitialized).
        experts = _make_fp8_experts(num_experts=4, hidden=8, inter=16)
        hidden_states = torch.randn(3, 8, dtype=torch.bfloat16, device="cuda")
        top_k_index = torch.tensor([[0, 4], [1, 4], [2, 4]], device="cuda")  # 4 == num_experts -> sentinel
        top_k_weights = torch.rand(3, 2, dtype=torch.bfloat16, device="cuda")
        with self._mocked_kernel() as calls:
            out = fp8_batched_mm_experts_forward(experts, hidden_states, top_k_index, top_k_weights)
        self.assertTrue(torch.equal(calls["batched_matmul"][0]["expert_ids"], top_k_index.reshape(-1)))
        self.assertEqual(int(calls["batched_matmul"][0]["expert_ids"].max()), 4)
        # Sentinel token-expert pairs contribute 0 to the reduction -> each token keeps only its non-sentinel weight.
        expected = self._expected_experts_output(hidden_states, top_k_index, top_k_weights, experts.num_experts)
        self.assertTrue(torch.equal(out, expected))
        self.assertTrue(torch.equal(out, top_k_weights[:, :1].to(torch.float32).expand(3, 8).to(torch.bfloat16)))

    def test_batched_mm_rejects_static_activation_scheme(self):
        # Static activation quant needs a per-tensor activation scale the batched kernel can't consume;
        # this guards a genuine unsupported-config path, not a trivial type/device check.
        experts = _make_fp8_experts(activation_scheme="static")
        hidden_states = torch.randn(3, 8, dtype=torch.bfloat16, device="cuda")
        top_k_index = torch.randint(0, 4, (3, 2), device="cuda")
        top_k_weights = torch.rand(3, 2, dtype=torch.bfloat16, device="cuda")
        with self._mocked_kernel(), self.assertRaisesRegex(NotImplementedError, "activation_scheme='static'"):
            fp8_batched_mm_experts_forward(experts, hidden_states, top_k_index, top_k_weights)

    # ── fp8_grouped_mm_experts_forward ────────────────────────────────────────────────────────────

    def test_grouped_mm_kernel_inputs_and_output(self):
        experts = _make_fp8_experts(num_experts=4, hidden=8, inter=16)
        hidden_states = torch.randn(3, 8, dtype=torch.bfloat16, device="cuda")
        top_k_index = torch.randint(0, 4, (3, 2), device="cuda")
        top_k_weights = torch.rand(3, 2, dtype=torch.bfloat16, device="cuda")
        with self._mocked_kernel() as calls:
            out = fp8_grouped_mm_experts_forward(experts, hidden_states, top_k_index, top_k_weights)

        up, down = calls["grouped_matmul"]
        # S = num_tokens * top_k selected pairs, gathered by the expert-sort permutation.
        self.assertEqual(up["input"].shape, (6, 8))
        self.assertIs(up["weight"], experts.gate_up_proj)
        self.assertIs(up["weight_scale"], experts.gate_up_proj_scale_inv)
        self.assertIs(up["block_size"], experts.block_size)
        # Down projection sees the gated activation: 2*inter (32) collapsed to inter (16).
        self.assertEqual(down["input"].shape, (6, 16))
        self.assertIs(down["weight"], experts.down_proj)
        self.assertIs(down["weight_scale"], experts.down_proj_scale_inv)

        # offsets / tokens_per_expert are the per-expert histogram over the sorted expert ids.
        expert_ids_g, _ = torch.sort(top_k_index.reshape(-1))
        expected_tpe = torch.histc(expert_ids_g.int(), bins=experts.num_experts, min=0, max=experts.num_experts - 1)
        expected_offsets = torch.cumsum(expected_tpe, dim=0, dtype=torch.int32)
        self.assertTrue(torch.equal(up["tokens_per_expert"], expected_tpe))
        self.assertTrue(torch.equal(up["offsets"], expected_offsets))
        # Both projections share the same offsets/tokens_per_expert tensors.
        self.assertIs(down["offsets"], up["offsets"])
        self.assertIs(down["tokens_per_expert"], up["tokens_per_expert"])

        # Output restored to original token order (inv_perm) then reduced; value-exact, dtype restored.
        self.assertEqual(out.shape, (3, 8))
        self.assertEqual(out.dtype, torch.bfloat16)
        expected = self._expected_experts_output(hidden_states, top_k_index, top_k_weights, experts.num_experts)
        self.assertTrue(torch.equal(out, expected))

    def test_grouped_mm_sentinels_dropped_from_histogram(self):
        # Sentinels are left unclamped so the sort pushes them to the tail and histc(max=num_experts-1)
        # drops them from tokens_per_expert -> no wasted GEMM rows; the post-mask zeroes their output.
        experts = _make_fp8_experts(num_experts=4, hidden=8, inter=16)
        hidden_states = torch.randn(3, 8, dtype=torch.bfloat16, device="cuda")
        top_k_index = torch.tensor([[0, 4], [1, 4], [2, 4]], device="cuda")  # three sentinels (== num_experts)
        top_k_weights = torch.rand(3, 2, dtype=torch.bfloat16, device="cuda")
        with self._mocked_kernel() as calls:
            out = fp8_grouped_mm_experts_forward(experts, hidden_states, top_k_index, top_k_weights)
        tpe = calls["grouped_matmul"][0]["tokens_per_expert"]
        # Only experts 0,1,2 got one token each; the 3 sentinels are absent from the histogram.
        self.assertTrue(torch.equal(tpe, torch.tensor([1.0, 1.0, 1.0, 0.0], device="cuda")))
        self.assertEqual(int(tpe.sum()), 3)
        expected = self._expected_experts_output(hidden_states, top_k_index, top_k_weights, experts.num_experts)
        self.assertTrue(torch.equal(out, expected))

    def test_grouped_mm_rejects_static_activation_scheme(self):
        experts = _make_fp8_experts(activation_scheme="static")
        hidden_states = torch.randn(3, 8, dtype=torch.bfloat16, device="cuda")
        top_k_index = torch.randint(0, 4, (3, 2), device="cuda")
        top_k_weights = torch.rand(3, 2, dtype=torch.bfloat16, device="cuda")
        with self._mocked_kernel(), self.assertRaisesRegex(NotImplementedError, "activation_scheme='static'"):
            fp8_grouped_mm_experts_forward(experts, hidden_states, top_k_index, top_k_weights)


if __name__ == "__main__":
    unittest.main()
