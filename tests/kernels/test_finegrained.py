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
"""What the `finegrained` integration hands `kernels-community/finegrained-kernels`: the kernel bundle is
mocked, so these pin the call contract (As-positional / no-block_size / expert_start / b_global_scale), the
loader and the DeepGEMM dispatch without a GPU."""

import unittest
from dataclasses import dataclass
from unittest import mock

import torch

import transformers.integrations.deepgemm as deepgemm
import transformers.integrations.finegrained.core as fg
from transformers.integrations.finegrained import (
    FineGrainedExperts,
    FineGrainedLinear,
    finegrained_linear,
    load_finegrained_kernel,
)
from transformers.testing_utils import require_torch

from ..integrations.finegrained.test_core import _Cfg


@dataclass
class _Call:
    args: tuple
    kwargs: dict


class _Recorder:
    """Records every call and returns a correctly-shaped zero tensor for matmuls."""

    def __init__(self):
        self.calls = {}

    def _op(self, name):
        def op(*args, **kwargs):
            self.calls.setdefault(name, []).append(_Call(args, kwargs))
            if name == "swizzle_mx_scales":
                return args[0]
            if name.startswith("moe_fused"):  # (hidden, top_k_index, top_k_weights, gate_up, down, ...)
                hidden, down = args[0], kwargs["down_proj"]
                return hidden.new_zeros(hidden.shape[0], down.shape[1], dtype=torch.bfloat16)
            a, b = args[0], args[1]
            n = b.shape[-2]
            gather = kwargs.get("gather_idx")
            rows = gather.shape[0] if gather is not None else a.shape[0]
            return a.new_zeros(rows, n, dtype=torch.bfloat16)

        return op


def fake_finegrained_kernel():
    """A `finegrained-kernels` stand-in whose ops record their calls and return correctly shaped zeros."""
    rec = _Recorder()
    kernel = mock.Mock()
    for name in (
        "matmul_2d",
        "matmul_batched",
        "matmul_grouped",
        "moe_fused_batched",
        "moe_fused_grouped",
        "swizzle_mx_scales",
        "unswizzle_mx_scales",
        "mxfp8_act_quant",
        "mxfp4_act_quant",
        "nvfp4_act_quant",
    ):
        setattr(kernel, name, rec._op(name))
    kernel.get_supported_act_fns = lambda: ("silu", "gelu", "relu")
    kernel.get_supported_norms = lambda: ("rms_norm", "centered_rms_norm", "input_scaled_rms_norm")
    return kernel, rec


def patch_finegrained_kernel(kernel):
    """Patches that make the finegrained integration load `kernel` as its bundle."""
    return (
        mock.patch.object(fg, "_FINEGRAINED", None),
        mock.patch.object(fg, "is_kernels_available", return_value=True),
        mock.patch.object(fg, "lazy_load_kernel", return_value=kernel),
    )


@require_torch
class FineGrainedLoaderTest(unittest.TestCase):
    def test_loader_requires_every_symbol(self):
        kernel, _ = fake_finegrained_kernel()
        del kernel.matmul_grouped
        # Mock auto-creates attributes; force the miss
        kernel.matmul_grouped = None
        p1, p2, p3 = patch_finegrained_kernel(kernel)
        with p1, p2, p3, self.assertRaises(ImportError) as ctx:
            load_finegrained_kernel()
        self.assertIn("matmul_grouped", str(ctx.exception))

    def test_loader_is_compile_safe_cold(self):
        """Cold path: the compiled call is first to load, so the opaque loader node runs its full
        body under compile and must return None, never the bundle (`torch.* op returned
        non-Tensor`). The loader has no arch gate, so nothing here fakes a device."""
        kernel, _ = fake_finegrained_kernel()
        kernel.matmul_2d = lambda x, *a, **k: x + 1
        p1, p2, p3 = patch_finegrained_kernel(kernel)
        with p1, p2, p3:
            torch.compiler.reset()

            @torch.compile(fullgraph=True)
            def run(x):
                return load_finegrained_kernel().matmul_2d(x)

            out = run(torch.zeros(3))
        self.assertTrue(torch.equal(out, torch.ones(3)))

    def test_loader_is_compile_safe_when_warm(self):
        """Warm path, which is the production order: eager warm-up, then compile. The loader hits
        its short-circuit at trace time — the branch that must also return None."""
        kernel, _ = fake_finegrained_kernel()
        kernel.matmul_2d = lambda x, *a, **k: x + 1
        p1, p2, p3 = patch_finegrained_kernel(kernel)
        with p1, p2, p3:
            load_finegrained_kernel()
            torch.compiler.reset()

            @torch.compile(fullgraph=True)
            def run(x):
                return load_finegrained_kernel().matmul_2d(x)

            out = run(torch.zeros(3))
        self.assertTrue(torch.equal(out, torch.ones(3)))

    def test_loader_binds_all_symbols(self):
        kernel, _ = fake_finegrained_kernel()
        p1, p2, p3 = patch_finegrained_kernel(kernel)
        with p1, p2, p3:
            bundle = load_finegrained_kernel()
        self.assertIs(bundle.matmul_2d, kernel.matmul_2d)
        self.assertIs(bundle.get_supported_act_fns, kernel.get_supported_act_fns)


@require_torch
class FineGrainedLinearMarshallingTest(unittest.TestCase):
    def _run(self, **linear_kwargs):
        kernel, rec = fake_finegrained_kernel()
        p1, p2, p3 = patch_finegrained_kernel(kernel)
        with p1, p2, p3, mock.patch.object(deepgemm, "is_deepgemm_loadable", return_value=False):
            x = torch.randn(3, 5, 64, dtype=torch.bfloat16)
            w = torch.randn(32, 64).to(torch.float8_e4m3fn)
            ws = torch.randn(1, 1, dtype=torch.float32)
            out = finegrained_linear(x, w, ws, **linear_kwargs)
        return out, rec.calls["matmul_2d"][-1]

    def test_scale_rides_positionally_and_block_size_dies(self):
        out, call = self._run(block_size=[128, 128])
        # A is flattened 2D; As slot (positional 3rd) is the activation scale = None here
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
        p1, p2, p3 = patch_finegrained_kernel(kernel)
        with p1, p2, p3, mock.patch.object(deepgemm, "is_deepgemm_loadable", return_value=False):
            m = FineGrainedLinear(
                64,
                32,
                block_size=None,
                weight_format="nvfp4",
                activation_format="bf16",
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
        m = FineGrainedExperts(cfg, block_size=(4, 4), weight_format=weight_format, **kw)
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

    def test_eager_linear_threads_bias_global_and_activation_format(self):
        """The eager per-expert loop hands `matmul_2d` the same operands the fused forwards do:
        the expert's bias, its NVFP4 global and the module's activation format (W4A16 here)."""
        kernel, rec = fake_finegrained_kernel()
        m = self._experts(weight_format="nvfp4", has_bias=True, activation_format="bf16")
        p1, p2, p3 = patch_finegrained_kernel(kernel)
        linear = mock.Mock(wraps=fg.finegrained_linear)
        with (
            p1,
            p2,
            p3,
            mock.patch.object(deepgemm, "is_deepgemm_loadable", return_value=False),
            mock.patch.object(fg, "finegrained_linear", linear),
        ):
            m(*self._route())
        self.assertTrue(linear.call_args_list)
        for call in linear.call_args_list:
            self.assertEqual(call.kwargs["bias"].ndim, 1)  # this expert's bias (added after the matmul)
            # this expert's NVFP4 weight global: one value for the down, and (1, 2) for the
            # gate|up stack, whose halves modelopt calibrates separately
            self.assertIn(tuple(call.kwargs["weight_global_scale"].shape), {(), (1, 2)})
            self.assertEqual(call.kwargs["activation_format"], "bf16")
            # weight-only: the activations stay bf16, so no activation global is quantized against
            self.assertIsNone(call.kwargs["input_global_scale"])
        for call in rec.calls["matmul_2d"]:
            self.assertIsNotNone(call.kwargs["b_global_scale"])
            self.assertEqual(call.kwargs["activation_format"], "bf16")

    def test_batched_marshalling(self):
        kernel, rec = fake_finegrained_kernel()
        m = self._experts(has_gate=True)
        p1, p2, p3 = patch_finegrained_kernel(kernel)
        with p1, p2, p3:
            out = fg.finegrained_batched_mm_experts_forward(m, *self._route())
        (call,) = rec.calls["moe_fused_batched"]
        hidden, top_k_index, top_k_weights = call.args
        self.assertEqual(hidden.shape[0], 6)  # UNEXPANDED tokens — the kernel gathers
        self.assertEqual(top_k_index.shape, (6, 2))
        self.assertIs(call.kwargs["gate_up_proj"], m.gate_up_proj)
        # one scale tensor per projection: the affine Parameter, or the swizzled cache when the
        # post-load hook built one (not here — the hook runs on SM100 for dot_scaled chains)
        self.assertIs(call.kwargs["gate_up_proj_scale_inv"], m.gate_up_proj_scale_inv)
        self.assertIsNone(call.kwargs["gate_up_proj_weight_global_scale"])
        self.assertEqual(call.kwargs["act_fn"], "silu")  # fusable: passed by name
        self.assertIs(call.kwargs["gate"], True)
        self.assertIsNone(call.kwargs["activation_format"])  # None = the weight family's format
        self.assertEqual(out.shape, (6, 64))

    def test_grouped_marshalling(self):
        kernel, rec = fake_finegrained_kernel()
        m = self._experts(has_gate=True)
        p1, p2, p3 = patch_finegrained_kernel(kernel)
        with p1, p2, p3:
            fg.finegrained_grouped_mm_experts_forward(m, *self._route())
        (call,) = rec.calls["moe_fused_grouped"]
        self.assertIs(call.kwargs["down_proj"], m.down_proj)
        self.assertIs(call.kwargs["down_proj_scale_inv"], m.down_proj_scale_inv)
        self.assertNotIn("moe_fused_batched", rec.calls)

    def test_activation_format_passes_through(self):
        kernel, rec = fake_finegrained_kernel()
        for activation_format in (None, "bf16", "mxfp8"):
            m = self._experts(has_gate=True, activation_format=activation_format)
            p1, p2, p3 = patch_finegrained_kernel(kernel)
            with p1, p2, p3:
                fg.finegrained_batched_mm_experts_forward(m, *self._route())
            self.assertEqual(rec.calls["moe_fused_batched"][-1].kwargs["activation_format"], activation_format)

    def test_nvfp4_experts_thread_per_expert_globals(self):
        kernel, rec = fake_finegrained_kernel()
        m = self._experts(has_gate=True, weight_format="nvfp4")
        # the format table must resolve the ATTRIBUTE the forwards gate on, not just the
        # param allocation — a None here silently drops the global at every forward
        self.assertIsNotNone(m.global_scale_dtype)
        p1, p2, p3 = patch_finegrained_kernel(kernel)
        with p1, p2, p3:
            fg.finegrained_batched_mm_experts_forward(m, *self._route())
        (call,) = rec.calls["moe_fused_batched"]
        self.assertIs(call.kwargs["gate_up_proj_weight_global_scale"], m.gate_up_proj_weight_global_scale)
        self.assertIs(call.kwargs["down_proj_weight_global_scale"], m.down_proj_weight_global_scale)
        # one weight global per expert on both projections: a gate|up stack calibrated per half
        # is merged to that at load, so nothing downstream carries the pair
        self.assertEqual(tuple(m.gate_up_proj_weight_global_scale.shape), (m.num_experts,))
        self.assertEqual(tuple(m.down_proj_weight_global_scale.shape), (m.num_experts,))
        # the calibrated activation globals (the checkpoint's `input_scale`): one value for the
        # gate_up, whose rows are quantized once before routing, and one per expert for the down,
        # whose rows the gate_up epilogue requantizes per expert
        self.assertIs(call.kwargs["gate_up_proj_input_global_scale"], m.gate_up_proj_input_global_scale)
        self.assertIs(call.kwargs["down_proj_input_global_scale"], m.down_proj_input_global_scale)
        self.assertEqual(tuple(m.gate_up_proj_input_global_scale.shape), (1,))
        self.assertEqual(tuple(m.down_proj_input_global_scale.shape), (m.num_experts,))

    def test_weight_only_experts_hold_no_activation_global(self):
        """W4A16 keeps the activations bf16 and requantizes nothing, so there is no activation
        quant for a calibrated global to normalize: the module never allocates one, and the ops
        would refuse a two-level path without it."""
        kernel, rec = fake_finegrained_kernel()
        m = self._experts(has_gate=True, weight_format="nvfp4", activation_format="bf16")
        p1, p2, p3 = patch_finegrained_kernel(kernel)
        with p1, p2, p3:
            fg.finegrained_grouped_mm_experts_forward(m, *self._route())
        (call,) = rec.calls["moe_fused_grouped"]
        self.assertIsNone(m.gate_up_proj_input_global_scale)
        self.assertIsNone(m.down_proj_input_global_scale)
        self.assertIsNone(call.kwargs["gate_up_proj_input_global_scale"])
        self.assertIsNone(call.kwargs["down_proj_input_global_scale"])
        # the WEIGHT globals are the format's own second level and stay
        self.assertIsNotNone(call.kwargs["gate_up_proj_weight_global_scale"])

    def test_post_expert_norm_rides_the_chain_and_the_eager_loop(self):
        """A model whose experts norm the down output before the routing weights:
        the swap carries the norm over, the kernel chain runs it on the routed rows, and the
        eager loop applies it per expert application — the same place the reference forwards do.
        A form the kernels do not implement rides as the module's own ``post_expert_norm``."""
        kernel, rec = fake_finegrained_kernel()
        m = self._experts(has_gate=True)
        m.post_expert_norm, m.has_post_expert_norm = torch.nn.LayerNorm(m.hidden_dim), True
        p1, p2, p3 = patch_finegrained_kernel(kernel)
        with p1, p2, p3:
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
        with p1, p2, p3, mock.patch.object(deepgemm, "is_deepgemm_loadable", return_value=False):
            m(*self._route())
        self.assertTrue(m.post_expert_norm.rows, "the eager loop never applied the post-expert norm")

    def test_a_post_expert_norm_needs_nothing_but_the_norm(self):
        """A class that declares a post-expert norm only has to HOLD it: every backend calls
        `post_expert_norm` directly, so there is no per-class hook to define alongside it."""
        from transformers.integrations.moe import use_experts_implementation

        @use_experts_implementation(has_post_expert_norm=True)
        class _JustTheNorm(torch.nn.Module):
            def forward(self, hidden_states, top_k_index, top_k_weights):
                return hidden_states

        self.assertFalse(hasattr(_JustTheNorm, "_apply_post_norm"))

    def test_post_expert_norm_rides_by_name_when_the_kernels_know_it(self):
        """The norm follows `act_fn`'s shape: the name the model declared goes to the kernels
        when they implement it, with the weight and epsilon the fused form needs, so the chain
        folds it into its reduce instead of calling the module."""
        kernel, rec = fake_finegrained_kernel()
        m = self._experts(has_gate=True)
        m.post_expert_norm, m.has_post_expert_norm = torch.nn.RMSNorm(m.hidden_dim, eps=1e-4), True
        p1, p2, p3 = patch_finegrained_kernel(kernel)
        for name, fused in (("rms_norm", True), ("input_scaled_rms_norm", True), ("a_models_own_norm", False)):
            m.post_expert_norm_name = name
            with p1, p2, p3:
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
        p1, p2, p3 = patch_finegrained_kernel(kernel)
        with p1, p2, p3:
            out = fg.finegrained_batched_mm_experts_forward(m, *self._route())
        self.assertTrue(torch.isfinite(out).all())
        (call,) = rec.calls["moe_fused_batched"]
        # the kernels add gate_up's bias before the gated split and down's before the
        # routing-weight multiply; nothing is added host-side
        self.assertIs(call.kwargs["gate_up_proj_bias"], m.gate_up_proj_bias)
        self.assertIs(call.kwargs["down_proj_bias"], m.down_proj_bias)

    def test_static_activation_scheme_reaches_the_fused_chain(self):
        """Mistral-3 ships `activation_scheme="static"`: a calibrated scale per activation, and a
        MoE calibrates each expert separately, so the module holds one per expert. Both fused
        chains take them per projection and the kernels apply each tile's own expert scale in
        register — a host pre-quant could not, since top-k routes one row to several experts
        whose scales differ. A dynamic module holds no such slot and passes `None`."""
        static, dynamic = (
            FineGrainedExperts(_Cfg(), block_size=(128, 128), activation_scheme=scheme)
            for scheme in ("static", "dynamic")
        )
        held = static.gate_up_proj_activation_scale
        self.assertIsNotNone(held, "the calibrated scale the chain needs")
        self.assertEqual(held.numel(), _Cfg.num_local_experts, "one per expert, which the kernels index")
        # a dynamic module holds the slot empty, like every other optional parameter
        self.assertIsNone(dynamic.gate_up_proj_activation_scale)

        kernel = mock.Mock()
        kernel.get_supported_act_fns.return_value = ("silu",)
        kernel.get_supported_norms.return_value = ()
        for module, calibrated in ((static, True), (dynamic, False)):
            operands = module._moe_operands(kernel)
            for projection in ("gate_up_proj", "down_proj"):
                with self.subTest(scheme=module.activation_scheme, projection=projection):
                    scale = operands[f"{projection}_activation_scale"]
                    self.assertEqual(scale is not None, calibrated)
                    if calibrated:
                        self.assertEqual(scale.numel(), _Cfg.num_local_experts)

    def test_unfusable_act_fn_is_passed_as_the_module_glu(self):
        kernel, rec = fake_finegrained_kernel()
        m = self._experts(has_gate=True, has_bias=True)
        m.act_fn_name = "quick_gelu"  # not in the kernels' get_supported_act_fns()
        p1, p2, p3 = patch_finegrained_kernel(kernel)
        with p1, p2, p3:
            fg.finegrained_batched_mm_experts_forward(m, *self._route())
        (call,) = rec.calls["moe_fused_batched"]
        # the kernels run the module's own GLU on the host between the two GEMMs — a new
        # activation never waits for a kernel release; the bias still rides the GEMM
        self.assertEqual(call.kwargs["act_fn"], m._apply_gate)  # bound-method equality
        self.assertIs(call.kwargs["gate_up_proj_bias"], m.gate_up_proj_bias)


@require_torch
class FineGrainedDeepGemmDispatchTest(unittest.TestCase):
    """`prefers_deepgemm_linear` carries two independent gates: a correctness one (a pre-swizzled
    scale is not readable as row-major, so DeepGEMM would consume a permuted buffer as affine
    and silently return garbage) and an SM100 perf one. They cover different cases — block-FP8
    scales have no swizzled layout, so the first never fires for the shape the second catches."""

    def _routed_to(self, *, sm100, scale_ndim, trains=False):
        """Which backend a block-FP8 linear reaches, given arch, scale layout, and whether the call
        needs a gradient."""
        kernel, _ = fake_finegrained_kernel()
        w = torch.randn(32, 64).to(torch.float8_e4m3fn)
        s = torch.randn(*([1, 1, 1, 1, 1][:scale_ndim] if scale_ndim > 2 else [1, 1]))
        p1, p2, p3 = patch_finegrained_kernel(kernel)
        with (
            p1,
            p2,
            p3,
            mock.patch.object(deepgemm, "is_deepgemm_loadable", return_value=True),
            mock.patch.object(deepgemm, "is_sm100", return_value=sm100),
            mock.patch.object(fg, "deepgemm_fp8_fp4_linear") as dg,
        ):
            x = torch.randn(4, 64, dtype=torch.bfloat16, requires_grad=trains)
            fg.finegrained_linear(x, w, s, block_size=[128, 128])
        return "deepgemm" if dg.called else "triton"

    def test_sm100_never_prefers_deepgemm(self):
        self.assertEqual(self._routed_to(sm100=True, scale_ndim=2), "triton")

    def test_pre_sm100_still_uses_deepgemm(self):
        self.assertEqual(self._routed_to(sm100=False, scale_ndim=2), "deepgemm")

    def test_deepgemm_refuses_a_call_that_needs_a_gradient(self):
        """DeepGEMM has no backward pass: every entry point fails a call that needs a gradient,
        before touching the kernel."""
        cfg = _Cfg()
        cfg.hidden_size, cfg.intermediate_size = 256, 128
        experts = fg.FineGrainedExperts(cfg, block_size=(128, 128), weight_format="fp8")
        experts._deepgemm_disabled = False
        hs = torch.randn(2, 256, dtype=torch.bfloat16, requires_grad=True)
        routing = torch.zeros(2, 2, dtype=torch.long), torch.ones(2, 2)
        w, s = torch.randn(32, 256).to(torch.float8_e4m3fn), torch.ones(1, 2)
        calls = [
            lambda: deepgemm.deepgemm_fp8_fp4_linear(hs, w, s, block_size=(128, 128)),
            lambda: deepgemm.deepgemm_fp8_fp4_experts_forward(experts, hs, *routing),
            lambda: deepgemm.deepgemm_fp8_fp4_megamoe_experts_forward(experts, hs, *routing),
        ]
        with mock.patch.object(deepgemm, "load_deepgemm_kernel") as load:
            for call in calls:
                with self.assertRaises(NotImplementedError):
                    call()
        load.assert_not_called()

    def test_a_linear_that_needs_a_gradient_prefers_triton(self):
        """Where DeepGEMM would otherwise win, a call that needs a gradient still goes to triton."""
        self.assertEqual(self._routed_to(sm100=False, scale_ndim=2, trains=True), "triton")

    def test_swizzled_scales_never_reach_deepgemm(self):
        """Correctness gate, and it must hold on any arch — not just where the perf gate does."""
        self.assertEqual(self._routed_to(sm100=False, scale_ndim=5), "triton")

    def test_deepgemm_experts_refuse_swizzled_scales(self):
        """A module loaded for a triton backend holds swizzled scales; switching it to a DeepGEMM
        experts backend afterwards must fail loudly rather than read the permuted buffer as affine."""
        from transformers.integrations.deepgemm import (
            deepgemm_fp8_fp4_experts_forward,
            deepgemm_fp8_fp4_megamoe_experts_forward,
        )

        cfg = _Cfg()
        cfg.hidden_size, cfg.intermediate_size, cfg._experts_implementation = 256, 128, "grouped_mm"
        with (
            mock.patch.object(fg, "is_sm100", return_value=True),
            mock.patch("torch.cuda.is_available", return_value=True),
        ):
            experts = fg.FineGrainedExperts(cfg, weight_format="mxfp8")
        self.assertEqual(experts.gate_up_proj_scale_inv.ndim, 5)
        hs = torch.randn(2, 256, dtype=torch.bfloat16)
        idx, wts = torch.zeros(2, 2, dtype=torch.long), torch.ones(2, 2)
        for forward in (deepgemm_fp8_fp4_experts_forward, deepgemm_fp8_fp4_megamoe_experts_forward):
            with self.assertRaises(RuntimeError):
                forward(experts, hs, idx, wts)


class FineGrainedFusedNormGateTest(unittest.TestCase):
    """A norm the kernels can fuse must NOT be fused when a forward collective sits on it.

    Fusing runs the norm inside the launch that produced the rows, so an all-reduce that has to
    land between the two has nowhere to go — and under intra-expert TP those rows are a partial
    sum over the sharded intermediate, so the fused norm would normalize a fraction of the value.
    The wrapped module does the reduce in its own forward, so the fallback is the correct path.
    """

    class _Norm(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(4))
            self.eps = 1e-6

    def _operands(self, *, input_reduce: bool):
        norm = self._Norm()
        if input_reduce:
            norm._hf_tp_input_reduce = True  # what `ReplicatedWithInputAllReduce` marks

        module = mock.Mock()
        module.has_gate = True
        module.act_fn_name = "silu"
        module.swiglu_alpha = None
        module.has_post_expert_norm = True
        module.post_expert_norm = norm
        module.post_expert_norm_name = "input_scaled_rms_norm"
        module._projection_slots = fg.FineGrainedExperts._projection_slots
        kernel = mock.Mock()
        kernel.get_supported_act_fns.return_value = ("silu",)
        kernel.get_supported_norms.return_value = ("input_scaled_rms_norm",)
        return fg.FineGrainedExperts._moe_operands(module, kernel), module

    def test_a_fusable_norm_fuses_when_nothing_reduces_its_input(self):
        operands, _ = self._operands(input_reduce=False)
        self.assertEqual(operands["post_expert_norm"], "input_scaled_rms_norm")
        self.assertIsNotNone(operands["post_expert_norm_weight"], "the fused form needs the norm's weight")

    def test_a_fusable_norm_falls_back_when_its_input_is_all_reduced(self):
        operands, module = self._operands(input_reduce=True)
        self.assertIs(
            operands["post_expert_norm"],
            module.post_expert_norm,
            "fused past a collective: the kernel would normalize this rank's partial sum",
        )
        self.assertIsNone(operands["post_expert_norm_weight"])
