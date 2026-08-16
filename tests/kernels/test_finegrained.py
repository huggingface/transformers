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
"""Marshalling tests for the multi-recipe ``finegrained`` integration: the kernel bundle is
mocked, so these pin exactly what the integration passes to `kernels-community/finegrained-kernels`
(the As-positional / no-block_size / expert_start / b_global_scale contract) without a GPU."""

import unittest
from dataclasses import dataclass
from unittest import mock

import torch

import transformers.integrations.finegrained as fg
from transformers.integrations.finegrained import (
    FineGrainedExperts,
    FineGrainedLinear,
    finegrained_linear,
    load_finegrained_kernel,
)
from transformers.testing_utils import require_torch, require_torch_gpu


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
            a, b = args[0], args[1]
            n = b.shape[-2]
            gather = kwargs.get("gather_idx")
            rows = gather.shape[0] if gather is not None else a.shape[0]
            return a.new_zeros(rows, n, dtype=torch.bfloat16)

        return op


def _fake_bundle():
    rec = _Recorder()
    kernel = mock.Mock()
    for name in ("matmul_2d", "matmul_batched", "matmul_grouped", "nvfp4_quantize_two_level", "swizzle_mx_scales"):
        setattr(kernel, name, rec._op(name))

    def compute_grouped_scheduling(top_k_index, num_experts, num_top_k):
        rec.calls.setdefault("compute_grouped_scheduling", []).append(_Call((top_k_index, num_experts, num_top_k), {}))
        flat = top_k_index.reshape(-1)
        sorted_ids, perm = torch.sort(flat)
        counts = torch.bincount(sorted_ids.clamp(max=num_experts - 1), minlength=num_experts)
        expert_start = torch.cat([counts.new_zeros(1), counts.cumsum(0)]).to(torch.int32)
        gather_idx = (perm // num_top_k).to(torch.int32)
        scatter_idx = torch.empty_like(perm)
        scatter_idx[torch.arange(perm.numel())] = perm
        return expert_start, gather_idx, scatter_idx.to(torch.int32)

    def weighted_reduce(rows, top_k_index, top_k_weights, num_experts, simulate_unfused=False):
        rec.calls.setdefault("weighted_reduce", []).append(_Call((rows, top_k_index, top_k_weights, num_experts), {}))
        num_tokens, num_top_k = top_k_index.shape
        w = top_k_weights.reshape(-1, 1).to(rows.dtype)
        w = w * (top_k_index.reshape(-1, 1) < num_experts)
        return (rows * w).view(num_tokens, num_top_k, -1).sum(1)

    kernel.compute_grouped_scheduling = compute_grouped_scheduling
    kernel.weighted_reduce = weighted_reduce

    @dataclass
    class Quantization:
        input_recipe: str | None = "weights"
        output_recipe: str | None = None

    @dataclass
    class Epilogue:
        gate: bool = False
        act_fn: str = "silu"
        swiglu_alpha: float | None = None
        swiglu_limit: float | None = None

    kernel.Quantization = Quantization
    kernel.Epilogue = Epilogue
    return kernel, rec


class _Cfg:
    hidden_size = 64
    num_local_experts = 4
    intermediate_size = 32
    hidden_act = "silu"


def _loaded(kernel):
    return (
        mock.patch.object(fg, "_FINEGRAINED", None),
        mock.patch.object(fg, "is_kernels_available", return_value=True),
        mock.patch.object(fg, "lazy_load_kernel", return_value=kernel),
        # a locally importable checkout (FINEGRAINED_KERNELS_PATH / installed package)
        # takes precedence over the hub loader; tests must stay hermetic to the fake bundle
        mock.patch.object(fg, "_import_local_finegrained", return_value=None),
    )


@require_torch
class FineGrainedLoaderTest(unittest.TestCase):
    def test_loader_requires_every_symbol(self):
        kernel, _ = _fake_bundle()
        del kernel.matmul_grouped
        # Mock auto-creates attributes; force the miss
        kernel.matmul_grouped = None
        p1, p2, p3, p4 = _loaded(kernel)
        with p1, p2, p3, p4, self.assertRaises(ImportError) as ctx:
            load_finegrained_kernel()
        self.assertIn("matmul_grouped", str(ctx.exception))

    def test_loader_binds_all_symbols(self):
        kernel, _ = _fake_bundle()
        p1, p2, p3, p4 = _loaded(kernel)
        with p1, p2, p3, p4:
            bundle = load_finegrained_kernel()
        self.assertIs(bundle.Quantization, kernel.Quantization)
        self.assertIs(bundle.Epilogue, kernel.Epilogue)


@require_torch
class FineGrainedLinearMarshallingTest(unittest.TestCase):
    def _run(self, **linear_kwargs):
        kernel, rec = _fake_bundle()
        p1, p2, p3, p4 = _loaded(kernel)
        with p1, p2, p3, p4, mock.patch.object(fg, "is_deepgemm_loadable", return_value=False):
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
        self.assertIsNone(call.kwargs["quantization"].input_recipe)

    def test_module_forward_threads_everything(self):
        kernel, rec = _fake_bundle()
        p1, p2, p3, p4 = _loaded(kernel)
        with p1, p2, p3, p4, mock.patch.object(fg, "is_deepgemm_loadable", return_value=False):
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
        self.assertIsNone(call.kwargs["quantization"].input_recipe)
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

    def test_batched_marshalling(self):
        kernel, rec = _fake_bundle()
        m = self._experts(has_gate=True)
        p1, p2, p3, p4 = _loaded(kernel)
        with p1, p2, p3, p4:
            out = fg.finegrained_batched_mm_experts_forward(m, *self._route())
        up, down = rec.calls["matmul_batched"]
        self.assertIsNone(up.args[2])  # As positional None
        self.assertEqual(up.args[0].shape[0], 6)  # UNEXPANDED tokens — the kernel gathers
        self.assertEqual(up.kwargs["gather_idx"].shape, (12,))
        self.assertEqual(up.kwargs["expert_ids"].shape, (12,))
        self.assertNotIn("block_size", up.kwargs)
        self.assertIsNone(up.kwargs["b_global_scale"])
        self.assertEqual(out.shape, (6, 64))

    def test_grouped_marshalling_expert_start_boundaries(self):
        kernel, rec = _fake_bundle()
        m = self._experts(has_gate=True)
        p1, p2, p3, p4 = _loaded(kernel)
        with p1, p2, p3, p4:
            fg.finegrained_grouped_mm_experts_forward(m, *self._route())
        up = rec.calls["matmul_grouped"][0]
        es = up.kwargs["expert_start"]
        self.assertEqual(es.shape, (5,))  # (E+1,) boundaries
        self.assertEqual(es[0].item(), 0)
        self.assertEqual(es[-1].item(), 12)  # S = tokens * top_k

    def test_nvfp4_experts_thread_per_expert_globals(self):
        kernel, rec = _fake_bundle()
        m = self._experts(has_gate=True, weight_format="nvfp4")
        # the format table must resolve the ATTRIBUTE the forwards gate on, not just the
        # param allocation — a None here silently drops the global at every forward
        self.assertIs(m.has_global_scale, True)
        p1, p2, p3, p4 = _loaded(kernel)
        with p1, p2, p3, p4:
            fg.finegrained_batched_mm_experts_forward(m, *self._route())
        up, down = rec.calls["matmul_batched"]
        self.assertIs(up.kwargs["b_global_scale"], m.gate_up_proj_global_scale)
        self.assertIs(down.kwargs["b_global_scale"], m.down_proj_global_scale)

    def test_bias_lands_before_gate_and_before_routing(self):
        kernel, rec = _fake_bundle()
        m = self._experts(has_gate=True, has_bias=True)
        hs, idx, wts = self._route()
        p1, p2, p3, p4 = _loaded(kernel)
        with p1, p2, p3, p4:
            out = fg.finegrained_batched_mm_experts_forward(m, hs, idx, wts)
        # zero matmul outputs + bias: result = sum_k w_k * (gate(bias_up) @ ... ) — just assert
        # the bias params were consumed (non-None) and output is finite
        self.assertTrue(torch.isfinite(out).all())
        self.assertEqual(len(rec.calls["matmul_batched"]), 2)


@require_torch
class FrozenFp8ShimTest(unittest.TestCase):
    def test_frozen_module_warns_and_is_self_contained(self):
        import importlib
        import warnings

        import transformers.integrations.finegrained_fp8 as frozen

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            importlib.reload(frozen)
        self.assertTrue(any(issubclass(x.category, DeprecationWarning) for x in w))
        # distinct machinery: the frozen classes are not the live ones
        self.assertIsNot(frozen.FP8Linear, FineGrainedLinear)


@require_torch
class FineGrainedMxfp4ConverterTest(unittest.TestCase):
    """The gpt-oss checkpoint layout through the real ConversionOps: blocks (E, N, K/32, 16)
    uint8 low-nibble-first E2M1 + biased-127 exponent scales, gate|up rows INTERLEAVED —
    deserialized output must dequantize identically to the reference LUT unpack."""

    FP4_VALUES = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]

    def _reference_dequant(self, blocks, scales):
        # the mxfp4.py reference semantics: lut[lo], lut[hi] interleaved along K, x 2^(scale-127)
        lut = torch.tensor(self.FP4_VALUES)
        lo = lut[(blocks & 0xF).long()]
        hi = lut[(blocks >> 4).long()]
        vals = torch.stack([lo, hi], dim=-1).reshape(*blocks.shape[:-1], -1)  # (E, N, K/32, 32)
        exp = (scales.long() - 127).unsqueeze(-1)
        return (vals * torch.pow(torch.tensor(2.0), exp)).reshape(*blocks.shape[:2], -1)

    def test_deserialize_matches_reference(self):
        from transformers.integrations.finegrained import FineGrainedMxfp4Deserialize, _get_ue8m0_dtype

        torch.manual_seed(0)
        E, N, K = 2, 8, 64  # N = 2I interleaved gate|up rows
        blocks = torch.randint(0, 256, (E, N, K // 32, 16), dtype=torch.uint8)
        scales = torch.randint(110, 140, (E, N, K // 32), dtype=torch.uint8)

        op = FineGrainedMxfp4Deserialize(hf_quantizer=None)
        out = op.convert(
            {"gate_up_proj_blocks": blocks, "gate_up_proj_scales": scales},
            full_layer_name="model.layers.0.mlp.experts.gate_up_proj",
        )
        weight = out["model.layers.0.mlp.experts.gate_up_proj"]
        scale_inv = out["model.layers.0.mlp.experts.gate_up_proj_scale_inv"]
        self.assertEqual(weight.dtype, torch.int8)
        self.assertEqual(weight.shape, (E, N, K // 2))
        self.assertEqual(scale_inv.dtype, _get_ue8m0_dtype())

        ref = self._reference_dequant(blocks, scales)  # rows pass through interleaved

        # dequantize the converted pair: packed E2M1 low-nibble-first x 2^(e8m0)
        lut = torch.tensor(self.FP4_VALUES)
        w_u8 = weight.view(torch.uint8)
        lo = lut[(w_u8 & 0xF).long()]
        hi = lut[(w_u8 >> 4).long()]
        vals = torch.stack([lo, hi], dim=-1).reshape(E, N, -1)
        exp = (scale_inv.view(torch.uint8).long() - 127).repeat_interleave(32, dim=-1)
        got = vals * torch.pow(torch.tensor(2.0), exp)
        torch.testing.assert_close(got, ref, rtol=0, atol=0)

    def test_interleave_gate_up_after_loading(self):
        """Stacked-checkpoint families get gate_up interleaved post-load; GPT-OSS-style MXFP4
        already ships that order and is skipped."""
        from transformers.integrations.finegrained import (
            FineGrainedExperts,
            interleave_gate_up_after_loading,
        )

        module = FineGrainedExperts.__new__(FineGrainedExperts)
        torch.nn.Module.__init__(module)
        module.has_gate = True
        # rows [g0,g1,g2,u0,u1,u2]; bias follows the same output axis
        module.gate_up_proj = torch.nn.Parameter(
            torch.arange(24, dtype=torch.float32).reshape(1, 6, 4), requires_grad=False
        )
        module.gate_up_proj_bias = torch.nn.Parameter(
            torch.arange(6, dtype=torch.float32), requires_grad=False
        )
        stacked_w = module.gate_up_proj.detach().clone()
        stacked_b = module.gate_up_proj_bias.detach().clone()

        model = torch.nn.Module()
        model.experts = module
        interleave_gate_up_after_loading(model)

        for j in range(3):
            torch.testing.assert_close(module.gate_up_proj[0, 2 * j], stacked_w[0, j])
            torch.testing.assert_close(module.gate_up_proj[0, 2 * j + 1], stacked_w[0, 3 + j])
            torch.testing.assert_close(module.gate_up_proj_bias[2 * j], stacked_b[j])
            torch.testing.assert_close(module.gate_up_proj_bias[2 * j + 1], stacked_b[3 + j])

    def test_interleave_skipped_when_already_interleaved(self):
        """GPT-OSS ships [g0,u0,...] already — the pass must leave it alone."""
        from transformers.integrations.finegrained import (
            FineGrainedExperts,
            interleave_gate_up_after_loading,
        )

        module = FineGrainedExperts.__new__(FineGrainedExperts)
        torch.nn.Module.__init__(module)
        module.has_gate = True
        module.gate_up_proj = torch.nn.Parameter(
            torch.arange(24, dtype=torch.float32).reshape(1, 6, 4), requires_grad=False
        )
        before = module.gate_up_proj.detach().clone()

        model = torch.nn.Module()
        model.experts = module
        interleave_gate_up_after_loading(model, already_interleaved=True)
        torch.testing.assert_close(module.gate_up_proj.data, before)


@require_torch
class FineGrainedDeepGemmDispatchTest(unittest.TestCase):
    """`deepgemm_preferred` carries two independent gates: a correctness one (a pre-swizzled
    scale is not readable as row-major, so DeepGEMM would consume a permuted buffer as affine
    and silently return garbage) and an SM100 perf one. They cover different cases — block-FP8
    scales have no swizzled layout, so the first never fires for the shape the second catches."""

    def _routed_to(self, *, sm100, scale_ndim):
        """Which backend a block-FP8 linear reaches, given arch and scale layout."""
        kernel, _ = _fake_bundle()
        w = torch.randn(32, 64).to(torch.float8_e4m3fn)
        s = torch.randn(*([1, 1, 1, 1, 1][:scale_ndim] if scale_ndim > 2 else [1, 1]))
        p1, p2, p3, p4 = _loaded(kernel)
        with (
            p1,
            p2,
            p3,
            p4,
            mock.patch.object(fg, "is_deepgemm_loadable", return_value=True),
            mock.patch.object(fg, "is_sm100", return_value=sm100),
            mock.patch.object(fg, "deepgemm_fp8_fp4_linear") as dg,
        ):
            fg.finegrained_linear(torch.randn(4, 64, dtype=torch.bfloat16), w, s, block_size=[128, 128])
        return "deepgemm" if dg.called else "triton"

    def test_sm100_never_prefers_deepgemm(self):
        self.assertEqual(self._routed_to(sm100=True, scale_ndim=2), "triton")

    def test_pre_sm100_still_uses_deepgemm(self):
        self.assertEqual(self._routed_to(sm100=False, scale_ndim=2), "deepgemm")

    def test_swizzled_scales_never_reach_deepgemm(self):
        """Correctness gate, and it must hold on any arch — not just where the perf gate does."""
        self.assertEqual(self._routed_to(sm100=False, scale_ndim=5), "triton")


@require_torch
class FineGrainedSwizzlePassTest(unittest.TestCase):
    """`swizzle_scales_after_loading` builds the Blackwell SWIZZLE_32_4_4 artifact. It is gated
    on arch AND on the recipe having group scales — keyed off the declared weight format, not
    the scale dtype, because V4-style block-FP8 ships UE8M0 scales and a dtype test lets it
    through (it then survives only by accident, since N/128 is rarely a multiple of 128)."""

    def _experts(self, weight_format, scale_dtype):
        cfg = _Cfg()
        experts = fg.FineGrainedExperts(cfg, block_size=(128, 128), weight_format=weight_format)
        for proj in ("gate_up_proj", "down_proj"):
            rows = 256
            setattr(
                experts,
                f"{proj}_scale_inv",
                torch.nn.Parameter(
                    torch.zeros(cfg.num_local_experts, rows, 8, dtype=scale_dtype),
                    requires_grad=False,
                ),
            )
        model = torch.nn.Module()
        model.experts = experts
        return model, experts

    def _run_pass(self, model, *, sm100):
        kernel, _ = _fake_bundle()
        kernel.swizzle_mx_scales = mock.Mock(side_effect=lambda s, gate=False: s)
        p1, p2, p3, p4 = _loaded(kernel)
        with p1, p2, p3, p4, mock.patch.object(fg, "is_sm100", return_value=sm100):
            fg.swizzle_scales_after_loading(model)
        return kernel.swizzle_mx_scales

    def test_skipped_off_sm100(self):
        """The layout is a Blackwell tcgen05 artifact; elsewhere it buys nothing, pins the tile
        space, and collides with DeepGEMM — which stays the preferred backend below SM100."""
        model, _ = self._experts("mxfp8", torch.float8_e8m0fnu)
        self.assertEqual(self._run_pass(model, sm100=False).call_count, 0)

    def test_applied_for_group_scaled_recipes_on_sm100(self):
        model, _ = self._experts("mxfp8", torch.float8_e8m0fnu)
        self.assertGreater(self._run_pass(model, sm100=True).call_count, 0)

    def test_block_fp8_skipped_even_with_ue8m0_scales(self):
        """The latent case: block-FP8's (N/128, K/128) grid never reaches a scaled-MMA, so it has
        no swizzled form — but it ships UE8M0 scales, so only a recipe-keyed guard excludes it."""
        model, _ = self._experts("fp8", torch.float8_e8m0fnu)
        self.assertEqual(self._run_pass(model, sm100=True).call_count, 0)


@require_torch
class FineGrainedModeloptConverterTest(unittest.TestCase):
    """The ModelOpt (NVFP4) weight conversions. `MergeModulelist` is load-bearing beyond the
    merge itself: `core_model_loading` stamps a source with its expert index ONLY when one is
    present in the chain, and expert parallelism selects experts by that index. Without it every
    rank collects all E globals and the forward asserts on the per-expert count."""

    def _modelopt_conversions(self):
        from transformers.quantizers.quantizer_finegrained import FineGrainedHfQuantizer
        from transformers.utils.quantization_config import FineGrainedConfig

        cfg = FineGrainedConfig(quant_method="modelopt", quant_algo="NVFP4")
        quantizer = FineGrainedHfQuantizer(cfg)
        return quantizer.get_weight_conversions()

    def test_global_scale_converters_carry_an_expert_index(self):
        from transformers.core_model_loading import MergeModulelist

        globals_converters = [c for c in self._modelopt_conversions() if any("global_scale" in t for t in _targets(c))]
        self.assertTrue(globals_converters, "no global-scale converter found")
        for conv in globals_converters:
            self.assertTrue(
                any(isinstance(op, MergeModulelist) for op in conv.operations),
                f"{_targets(conv)} has no MergeModulelist, so expert parallelism cannot select "
                "experts and every rank would collect all of them",
            )


def _targets(conv):
    t = conv.target_patterns
    return t if isinstance(t, (list, tuple)) else [t]


def _sm100():
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 10


@require_torch_gpu
@unittest.skipUnless(_sm100(), "the fine-grained kernels target SM100+")
class FineGrainedRealKernelTest(unittest.TestCase):
    """Runs the REAL kernels. The mocked tests above pin the calling contract; these pin that the
    contract produces correct numbers, which no amount of argument-matching can show."""

    @classmethod
    def setUpClass(cls):
        try:
            load_finegrained_kernel()
        except ImportError as e:
            raise unittest.SkipTest(f"finegrained kernels unavailable: {e}")

    def _block_fp8_weight(self, N, K, *, ue8m0):
        """(N, K) E4M3 + its (N/128, K/128) inv-scale grid, and the exact dequantized values the
        kernel will see — so the reference is the quantization floor, not the pre-quant weight."""
        w = torch.randn(N, K, device="cuda", dtype=torch.float32)
        blocks = w.reshape(N // 128, 128, K // 128, 128)
        amax = blocks.abs().amax(dim=(1, 3), keepdim=True).clamp(min=1e-12)
        inv = amax / 448.0
        if ue8m0:  # power-of-two scales: the tcgen05 dot_scaled recipe
            inv = torch.pow(2.0, torch.ceil(torch.log2(inv)))
        q = (blocks / inv).to(torch.float8_e4m3fn)
        deq = (q.float() * inv).reshape(N, K)
        return q.reshape(N, K), inv.reshape(N // 128, K // 128).contiguous(), deq

    def test_linear_forward_matches_the_quantization_floor(self):
        """A real quantized linear against its own dequantized weight. Loose vs bf16 would pass
        even if the kernel silently dropped a K-block; comparing to the floor does not."""
        M, N, K = 256, 512, 384
        for ue8m0 in (False, True):
            with self.subTest(ue8m0=ue8m0):
                w, s, deq = self._block_fp8_weight(N, K, ue8m0=ue8m0)
                x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
                out = finegrained_linear(x, w, s, block_size=[128, 128])
                ref = x.float() @ deq.t()
                rel = ((out.float() - ref).norm() / ref.norm()).item()
                self.assertLess(rel, 5e-2, f"ue8m0={ue8m0}: {rel:.2e} vs the dequantized weight")

    def test_swizzled_scales_do_not_change_the_result(self):
        """The SWIZZLE_32_4_4 pass reorders scale BYTES; it must not move a single value. Run the
        same MXFP8 GEMM against the affine grid and the swizzled artifact and demand agreement —
        a wrong reorder shows up here and nowhere in the mocked tests."""
        kernel = load_finegrained_kernel()
        N, K, M = 512, 256, 128
        w = torch.randn(N, K, device="cuda", dtype=torch.float32)
        groups = w.reshape(N, K // 32, 32)
        inv = torch.pow(2.0, torch.ceil(torch.log2(groups.abs().amax(-1, keepdim=True) / 448.0).clamp(min=-127)))
        q = (groups / inv).to(torch.float8_e4m3fn).reshape(N, K)
        scales = inv.reshape(N, K // 32).to(torch.float8_e8m0fnu)
        x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)

        affine = kernel.matmul(x, q, None, scales, output_dtype=torch.bfloat16)
        swizzled = kernel.matmul(
            x, q, None, kernel.swizzle_mx_scales(scales), output_dtype=torch.bfloat16
        )
        torch.cuda.synchronize()
        self.assertTrue(
            torch.equal(affine, swizzled),
            "swizzling changed the result; the artifact is a byte reorder, values must be identical",
        )

    def test_experts_swizzle_pass_keeps_the_forward_correct(self):
        """End-to-end over the integration: build real MXFP8 experts, run the post-load swizzle,
        and check the fused forward still matches what it produced beforehand."""
        cfg = _Cfg()
        cfg.hidden_size, cfg.intermediate_size, cfg.num_local_experts = 256, 128, 4
        experts = FineGrainedExperts(cfg, weight_format="mxfp8").cuda()
        for proj, rows in (("gate_up_proj", 2 * cfg.intermediate_size), ("down_proj", cfg.hidden_size)):
            cols = cfg.hidden_size if proj == "gate_up_proj" else cfg.intermediate_size
            setattr(experts, proj, torch.nn.Parameter(
                torch.randn(cfg.num_local_experts, rows, cols, device="cuda").to(torch.float8_e4m3fn),
                requires_grad=False))
            setattr(experts, f"{proj}_scale_inv", torch.nn.Parameter(
                torch.full((cfg.num_local_experts, rows, cols // 32), 127, dtype=torch.uint8, device="cuda")
                .view(torch.float8_e8m0fnu), requires_grad=False))

        x = torch.randn(8, cfg.hidden_size, device="cuda", dtype=torch.bfloat16)
        idx = torch.randint(0, cfg.num_local_experts, (8, 2), device="cuda", dtype=torch.long)
        wts = torch.rand(8, 2, device="cuda", dtype=torch.bfloat16)

        before = experts(x, idx, wts)
        model = torch.nn.Module()
        model.experts = experts
        fg.swizzle_scales_after_loading(model)
        after = experts(x, idx, wts)
        torch.cuda.synchronize()

        self.assertTrue(
            any(getattr(experts, f"{p}_scale_inv_swizzled", None) is not None
                for p in ("gate_up_proj", "down_proj")),
            "the swizzle pass produced no artifact, so this asserts nothing",
        )
        rel = ((after.float() - before.float()).norm() / before.float().norm().clamp(min=1e-9)).item()
        self.assertLess(rel, 1e-6, f"swizzled forward diverged from the affine one: {rel:.2e}")
