# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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
"""Tests for the call contract between the `finegrained` integration and `kernels-community/finegrained-kernels`.

The kernel bundle is mocked, so these tests run without a GPU.
"""

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

import transformers.integrations.deepgemm as deepgemm
import transformers.integrations.finegrained.layers as fg
from transformers.integrations.finegrained import (
    FineGrainedExperts,
    FineGrainedLinear,
    finegrained_linear,
    load_finegrained_kernel,
)
from transformers.testing_utils import require_torch
from transformers.utils.quantization_config import NVFP4Config

from ..integrations.finegrained.utils import _Cfg, _format_config, fake_finegrained_kernel, patch_finegrained_kernel


@require_torch
class FineGrainedLoaderTest(unittest.TestCase):
    def test_loader_requires_every_symbol(self):
        kernel, _ = fake_finegrained_kernel()
        del kernel.matmul_grouped
        # Mock auto-creates attributes; force the miss
        kernel.matmul_grouped = None
        with patch_finegrained_kernel(kernel), self.assertRaises(ImportError) as ctx:
            load_finegrained_kernel()
        self.assertIn("matmul_grouped", str(ctx.exception))

    def test_loader_requires_only_the_symbols_it_calls(self):
        """Check that a build with exactly the bundle's symbols loads."""
        kernel = SimpleNamespace(**{name: object() for name in fg.FineGrained.__dataclass_fields__})
        with patch_finegrained_kernel(kernel):
            self.assertIsInstance(load_finegrained_kernel(), fg.FineGrained)

    def test_loader_is_compile_safe_cold(self):
        """Check the loader under compile when the compiled call loads the kernel first."""
        kernel, _ = fake_finegrained_kernel()
        kernel.matmul_2d = lambda x, *a, **k: x + 1
        with patch_finegrained_kernel(kernel):
            torch.compiler.reset()

            @torch.compile(fullgraph=True)
            def run(x):
                return load_finegrained_kernel().matmul_2d(x)

            out = run(torch.zeros(3))
        self.assertTrue(torch.equal(out, torch.ones(3)))

    def test_loader_is_compile_safe_when_warm(self):
        """Check the loader under compile after an eager warm-up, the usual production order."""
        kernel, _ = fake_finegrained_kernel()
        kernel.matmul_2d = lambda x, *a, **k: x + 1
        with patch_finegrained_kernel(kernel):
            load_finegrained_kernel()
            torch.compiler.reset()

            @torch.compile(fullgraph=True)
            def run(x):
                return load_finegrained_kernel().matmul_2d(x)

            out = run(torch.zeros(3))
        self.assertTrue(torch.equal(out, torch.ones(3)))

    def test_loader_binds_all_symbols(self):
        kernel, _ = fake_finegrained_kernel()
        with patch_finegrained_kernel(kernel):
            bundle = load_finegrained_kernel()
        self.assertIs(bundle.matmul_2d, kernel.matmul_2d)
        self.assertIs(bundle.get_supported_act_fns, kernel.get_supported_act_fns)


@require_torch
class FineGrainedLinearMarshallingTest(unittest.TestCase):
    def _run(self, **linear_kwargs):
        kernel, rec = fake_finegrained_kernel()
        with patch_finegrained_kernel(kernel), mock.patch.object(deepgemm, "is_deepgemm_loadable", return_value=False):
            x = torch.randn(3, 5, 64, dtype=torch.bfloat16)
            w = torch.randn(32, 64).to(torch.float8_e4m3fn)
            ws = torch.randn(1, 1, dtype=torch.float32)
            out = finegrained_linear(x, w, ws, **linear_kwargs)
        return out, rec.calls["matmul_2d"][-1]

    def test_scale_rides_positionally_and_block_size_dies(self):
        out, call = self._run(block_size=[128, 128])
        # The input is flattened to 2D. The third positional slot is the activation scale.
        self.assertEqual(call.args[0].shape, (15, 64))
        self.assertIsNone(call.args[2])
        self.assertEqual(call.args[3].dtype, torch.float32)
        self.assertNotIn("block_size", call.kwargs)
        self.assertEqual(out.shape, (3, 5, 32))

    def test_static_activation_scale_is_As(self):
        scale = torch.tensor(0.5)
        _, call = self._run(activation_scale=scale)
        self.assertIs(call.args[2], scale)

    def test_nvfp4_global_and_weight_only_format(self):
        g = torch.tensor(2.0)
        _, call = self._run(weight_global_scale=g, activation_format="bf16")
        self.assertIs(call.kwargs["b_global_scale"], g)
        self.assertEqual(call.kwargs["activation_format"], "bf16")

    def test_module_forward_threads_everything(self):
        kernel, rec = fake_finegrained_kernel()
        with patch_finegrained_kernel(kernel), mock.patch.object(deepgemm, "is_deepgemm_loadable", return_value=False):
            m = FineGrainedLinear(
                64, 32, quantization_config=NVFP4Config(weight_block_size=None, activation_format="bf16")
            )
            m.weight.data = torch.randint(-127, 127, (32, 32), dtype=torch.int8)
            out = m(torch.randn(2, 64, dtype=torch.bfloat16))
        call = rec.calls["matmul_2d"][-1]
        self.assertIs(call.kwargs["b_global_scale"], m.weight_global_scale)
        self.assertEqual(call.kwargs["activation_format"], "bf16")
        self.assertEqual(out.shape, (2, 32))


@require_torch
class FineGrainedExpertsMarshallingTest(unittest.TestCase):
    def _experts(self, **kw):
        cfg = _Cfg()
        weight_format = kw.pop("weight_format", "fp8")
        block = {"weight_block_size": (4, 4)} if weight_format == "fp8" else {}
        quantized_type = _format_config(weight_format, activation_format=kw.pop("activation_format", None), **block)
        # the fused chain needs gate|up interleaved, which block FP8 holds on SM100 only
        with mock.patch.object(fg, "is_sm100", return_value=True):
            m = FineGrainedExperts(cfg, quantization_config=quantized_type, **kw)
        for name, p in m.named_parameters():
            if p is None:
                continue
            if p.dtype == torch.float8_e4m3fn:
                p.data = torch.randn(p.shape, dtype=torch.float32).to(torch.float8_e4m3fn)
            elif p.dtype.is_floating_point:
                p.data = torch.randn_like(p.data) if p.dim() else torch.ones_like(p.data)
        return m

    def _route(self, tokens=6):
        hs = torch.randn(tokens, 64, dtype=torch.bfloat16)
        idx = torch.randint(0, 4, (tokens, 2))
        wts = torch.rand(tokens, 2)
        return hs, idx, wts

    def test_batched_marshalling(self):
        kernel, rec = fake_finegrained_kernel()
        m = self._experts(has_gate=True)
        with patch_finegrained_kernel(kernel):
            out = fg.finegrained_batched_mm_experts_forward(m, *self._route())
        (call,) = rec.calls["moe_fused_batched"]
        hidden, top_k_index, top_k_weights = call.args
        self.assertEqual(hidden.shape[0], 6)  # the kernel gathers the routed rows itself
        self.assertEqual(top_k_index.shape, (6, 2))
        self.assertIs(call.kwargs["gate_up_proj"], m.gate_up_proj)
        self.assertIs(call.kwargs["gate_up_proj_scale_inv"], m.gate_up_proj_scale_inv)
        self.assertIsNone(call.kwargs["gate_up_proj_weight_global_scale"])
        self.assertEqual(call.kwargs["act_fn"], "silu")
        self.assertIs(call.kwargs["gate"], True)
        self.assertIsNone(call.kwargs["activation_format"])  # `None` means the weight format
        self.assertEqual(out.shape, (6, 64))

    def test_grouped_marshalling(self):
        kernel, rec = fake_finegrained_kernel()
        m = self._experts(has_gate=True)
        with patch_finegrained_kernel(kernel):
            fg.finegrained_grouped_mm_experts_forward(m, *self._route())
        (call,) = rec.calls["moe_fused_grouped"]
        self.assertIs(call.kwargs["down_proj"], m.down_proj)
        self.assertIs(call.kwargs["down_proj_scale_inv"], m.down_proj_scale_inv)
        self.assertNotIn("moe_fused_batched", rec.calls)

    def test_post_expert_norm_rides_the_chain_and_the_eager_loop(self):
        """Check that the kernel chain and the eager loop both apply the post-expert norm.

        A norm the kernels do not implement is passed as the module itself.
        """
        kernel, rec = fake_finegrained_kernel()
        m = self._experts(has_gate=True)
        m.post_expert_norm, m.has_post_expert_norm = torch.nn.LayerNorm(m.hidden_dim), True
        with patch_finegrained_kernel(kernel):
            fg.finegrained_grouped_mm_experts_forward(m, *self._route())
        self.assertEqual(rec.calls["moe_fused_grouped"][-1].kwargs["post_expert_norm"], m.post_expert_norm)

        class _Recording(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.rows = 0

            def forward(self, x):
                self.rows += x.shape[0]
                return x

        m.post_expert_norm = _Recording()
        with patch_finegrained_kernel(kernel), mock.patch.object(deepgemm, "is_deepgemm_loadable", return_value=False):
            m(*self._route())
        self.assertTrue(m.post_expert_norm.rows, "the eager loop never applied the post-expert norm")

    def test_post_expert_norm_rides_by_name_when_the_kernels_know_it(self):
        """Check that a norm the kernels implement is passed by name, with its weight and epsilon."""
        kernel, rec = fake_finegrained_kernel()
        for name, fused in (("rms_norm", True), ("input_scaled_rms_norm", True), ("a_models_own_norm", False)):
            # A module resolves its norm once, on its first forward
            m = self._experts(has_gate=True)
            m.post_expert_norm, m.has_post_expert_norm = torch.nn.RMSNorm(m.hidden_dim, eps=1e-4), True
            with (
                patch_finegrained_kernel(kernel),
                mock.patch.dict(fg.supported_fused_norms(), {torch.nn.RMSNorm: name}),
            ):
                fg.finegrained_grouped_mm_experts_forward(m, *self._route())
            call = rec.calls["moe_fused_grouped"][-1]
            if fused:
                self.assertEqual(call.kwargs["post_expert_norm"], name)
                self.assertIs(call.kwargs["post_expert_norm_weight"], m.post_expert_norm.weight)
                self.assertAlmostEqual(call.kwargs["post_expert_norm_eps"], 1e-4)
            else:
                self.assertEqual(call.kwargs["post_expert_norm"], m.post_expert_norm)
                self.assertIsNone(call.kwargs["post_expert_norm_weight"])

    def test_biases_ride_the_kernel_chain(self):
        kernel, rec = fake_finegrained_kernel()
        m = self._experts(has_gate=True, has_bias=True)
        with patch_finegrained_kernel(kernel):
            out = fg.finegrained_batched_mm_experts_forward(m, *self._route())
        self.assertTrue(torch.isfinite(out).all())
        (call,) = rec.calls["moe_fused_batched"]
        self.assertIs(call.kwargs["gate_up_proj_bias"], m.gate_up_proj_bias)
        self.assertIs(call.kwargs["down_proj_bias"], m.down_proj_bias)

    def test_unfusable_act_fn_is_passed_as_the_module_glu(self):
        kernel, rec = fake_finegrained_kernel()
        m = self._experts(has_gate=True, has_bias=True)
        m.act_fn_name = "quick_gelu"  # not supported by the kernels
        with patch_finegrained_kernel(kernel):
            fg.finegrained_batched_mm_experts_forward(m, *self._route())
        (call,) = rec.calls["moe_fused_batched"]
        # The kernels run the module's own GLU between the two GEMMs
        self.assertEqual(call.kwargs["act_fn"], m._apply_gate)
        self.assertIs(call.kwargs["gate_up_proj_bias"], m.gate_up_proj_bias)

    def test_eager_linear_threads_bias_global_and_activation_format(self):
        """Check that the eager loop passes the same operands as the fused forwards."""
        kernel, rec = fake_finegrained_kernel()
        m = self._experts(weight_format="nvfp4", has_bias=True, activation_format="bf16")
        linear = mock.Mock(wraps=fg.finegrained_linear)
        with (
            patch_finegrained_kernel(kernel),
            mock.patch.object(deepgemm, "is_deepgemm_loadable", return_value=False),
            mock.patch.object(fg, "finegrained_linear", linear),
        ):
            m(*self._route())
        self.assertTrue(linear.call_args_list)
        for call in linear.call_args_list:
            self.assertEqual(call.kwargs["bias"].ndim, 1)
            self.assertEqual(call.kwargs["weight_global_scale"].shape, ())
            self.assertEqual(call.kwargs["activation_format"], "bf16")
            # Weight-only keeps the activations in bf16
            self.assertIsNone(call.kwargs["input_global_scale"])
        for call in rec.calls["matmul_2d"]:
            self.assertIsNotNone(call.kwargs["b_global_scale"])
            self.assertEqual(call.kwargs["activation_format"], "bf16")

    def test_activation_format_passes_through(self):
        kernel, rec = fake_finegrained_kernel()
        # The weight format's own activations are passed as `None`
        for activation_format, handed in ((None, None), ("bf16", "bf16"), ("nvfp4", None)):
            m = self._experts(has_gate=True, weight_format="nvfp4", activation_format=activation_format)
            with patch_finegrained_kernel(kernel):
                fg.finegrained_batched_mm_experts_forward(m, *self._route())
            self.assertEqual(rec.calls["moe_fused_batched"][-1].kwargs["activation_format"], handed)

    def test_nvfp4_experts_thread_per_expert_globals(self):
        kernel, rec = fake_finegrained_kernel()
        m = self._experts(has_gate=True, weight_format="nvfp4")
        # The forwards drop the global when this is `None`
        self.assertIsNotNone(m.global_scale_dtype)
        with patch_finegrained_kernel(kernel):
            fg.finegrained_batched_mm_experts_forward(m, *self._route())
        (call,) = rec.calls["moe_fused_batched"]
        self.assertIs(call.kwargs["gate_up_proj_weight_global_scale"], m.gate_up_proj_weight_global_scale)
        self.assertIs(call.kwargs["down_proj_weight_global_scale"], m.down_proj_weight_global_scale)
        # Loading merges a per-half gate|up global into one global per expert
        self.assertEqual(tuple(m.gate_up_proj_weight_global_scale.shape), (m.num_experts,))
        self.assertEqual(tuple(m.down_proj_weight_global_scale.shape), (m.num_experts,))
        # The checkpoint's `input_scale` gives one activation global per expert
        self.assertIs(call.kwargs["gate_up_proj_input_global_scale"], m.gate_up_proj_activation_scale)
        self.assertIs(call.kwargs["down_proj_input_global_scale"], m.down_proj_activation_scale)
        self.assertEqual(tuple(m.gate_up_proj_activation_scale.shape), (m.num_experts,))
        self.assertEqual(tuple(m.down_proj_activation_scale.shape), (m.num_experts,))

    def test_weight_only_experts_hold_no_activation_global(self):
        """W4A16 keeps the activations in bf16, so the module allocates no activation global."""
        kernel, rec = fake_finegrained_kernel()
        m = self._experts(has_gate=True, weight_format="nvfp4", activation_format="bf16")
        with patch_finegrained_kernel(kernel):
            fg.finegrained_grouped_mm_experts_forward(m, *self._route())
        (call,) = rec.calls["moe_fused_grouped"]
        self.assertIsNone(m.gate_up_proj_activation_scale)
        self.assertIsNone(m.down_proj_activation_scale)
        self.assertIsNone(call.kwargs["gate_up_proj_input_global_scale"])
        self.assertIsNone(call.kwargs["down_proj_input_global_scale"])
        # The weight globals stay
        self.assertIsNotNone(call.kwargs["gate_up_proj_weight_global_scale"])
