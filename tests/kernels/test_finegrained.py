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
            if name.startswith("moe_fused"):  # (hidden, top_k_index, top_k_weights, gate_up, down, ...)
                hidden, down = args[0], kwargs["down_proj"]
                return hidden.new_zeros(hidden.shape[0], down.shape[1], dtype=torch.bfloat16)
            a, b = args[0], args[1]
            n = b.shape[-2]
            gather = kwargs.get("gather_idx")
            rows = gather.shape[0] if gather is not None else a.shape[0]
            return a.new_zeros(rows, n, dtype=torch.bfloat16)

        return op


def _fake_bundle():
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
class FineGrainedEmbeddingTest(unittest.TestCase):
    """A quantized embedding TABLE (Qwen4-Exp's n-gram table): FP8 rows with one per-tensor scale,
    rescaled on the rows a lookup gathers; swapped in for the names in `modules_to_convert`."""

    def test_lookup_rescales_the_gathered_rows(self):
        torch.manual_seed(0)
        table = torch.randn(64, 16)
        scale = table.abs().max() / 448.0
        embedding = fg.FineGrainedEmbedding(64, 16)
        embedding.weight = torch.nn.Parameter((table / scale).to(torch.float8_e4m3fn), requires_grad=False)
        embedding.weight_scale = torch.nn.Parameter(scale.to(torch.bfloat16).reshape(1), requires_grad=False)
        ids = torch.tensor([[3, 7], [60, 0]])
        out = embedding(ids)
        self.assertEqual(out.dtype, torch.bfloat16)
        expected = (table / scale).to(torch.float8_e4m3fn)[ids].to(torch.bfloat16) * scale.to(torch.bfloat16)
        torch.testing.assert_close(out, expected)

    def test_replacement_follows_the_patterns_and_the_skip_list(self):
        model = torch.nn.Module()
        model.ngram_embedding = torch.nn.Embedding(8, 4)
        model.embed_tokens = torch.nn.Embedding(8, 4)
        model.other = torch.nn.Embedding(8, 4)
        fg.replace_with_finegrained_embedding(model, ["ngram_embedding", "other"], modules_to_not_convert=["other"])
        self.assertIsInstance(model.ngram_embedding, fg.FineGrainedEmbedding)
        self.assertTrue(model.ngram_embedding._hf_quantized_needs_local_tp)
        self.assertIs(type(model.embed_tokens), torch.nn.Embedding)
        self.assertIs(type(model.other), torch.nn.Embedding)


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
        kernel, rec = _fake_bundle()
        m = self._experts(weight_format="nvfp4", has_bias=True, activation_format="bf16")
        p1, p2, p3, p4 = _loaded(kernel)
        linear = mock.Mock(wraps=fg.finegrained_linear)
        with (
            p1,
            p2,
            p3,
            p4,
            mock.patch.object(fg, "is_deepgemm_loadable", return_value=False),
            mock.patch.object(fg, "finegrained_linear", linear),
        ):
            m(*self._route())
        self.assertTrue(linear.call_args_list)
        for call in linear.call_args_list:
            self.assertEqual(call.kwargs["bias"].ndim, 1)  # this expert's bias (added after the matmul)
            self.assertEqual(call.kwargs["weight_global_scale"].ndim, 0)  # this expert's NVFP4 global
            self.assertEqual(call.kwargs["activation_format"], "bf16")
        for call in rec.calls["matmul_2d"]:
            self.assertIsNotNone(call.kwargs["b_global_scale"])
            self.assertIsNone(call.kwargs["quantization"].input_recipe)

    def test_batched_marshalling(self):
        kernel, rec = _fake_bundle()
        m = self._experts(has_gate=True)
        p1, p2, p3, p4 = _loaded(kernel)
        with p1, p2, p3, p4:
            out = fg.finegrained_batched_mm_experts_forward(m, *self._route())
        (call,) = rec.calls["moe_fused_batched"]
        hidden, top_k_index, top_k_weights = call.args
        self.assertEqual(hidden.shape[0], 6)  # UNEXPANDED tokens — the kernel gathers
        self.assertEqual(top_k_index.shape, (6, 2))
        self.assertIs(call.kwargs["gate_up_proj"], m.gate_up_proj)
        # one scale tensor per projection: the affine Parameter, or the swizzled cache when the
        # post-load hook built one (not here — the hook runs on SM100 for dot_scaled chains)
        self.assertIs(call.kwargs["gate_up_proj_scale_inv"], m.gate_up_proj_scale_inv)
        self.assertIsNone(call.kwargs["gate_up_proj_global_scale"])
        self.assertEqual(call.kwargs["act_fn"], "silu")  # fusable: passed by name
        self.assertIs(call.kwargs["gate"], True)
        self.assertEqual(call.kwargs["recipe"], "weights")  # activation_format None = weight family
        self.assertEqual(out.shape, (6, 64))

    def test_grouped_marshalling(self):
        kernel, rec = _fake_bundle()
        m = self._experts(has_gate=True)
        p1, p2, p3, p4 = _loaded(kernel)
        with p1, p2, p3, p4:
            fg.finegrained_grouped_mm_experts_forward(m, *self._route())
        (call,) = rec.calls["moe_fused_grouped"]
        self.assertIs(call.kwargs["down_proj"], m.down_proj)
        self.assertIs(call.kwargs["down_proj_scale_inv"], m.down_proj_scale_inv)
        self.assertNotIn("moe_fused_batched", rec.calls)

    def test_activation_format_maps_to_the_block_recipe(self):
        kernel, rec = _fake_bundle()
        for activation_format, recipe in ((None, "weights"), ("bf16", None), ("mxfp8", "mxfp8")):
            m = self._experts(has_gate=True, activation_format=activation_format)
            p1, p2, p3, p4 = _loaded(kernel)
            with p1, p2, p3, p4:
                fg.finegrained_batched_mm_experts_forward(m, *self._route())
            self.assertEqual(rec.calls["moe_fused_batched"][-1].kwargs["recipe"], recipe)

    def test_nvfp4_experts_thread_per_expert_globals(self):
        kernel, rec = _fake_bundle()
        m = self._experts(has_gate=True, weight_format="nvfp4")
        # the format table must resolve the ATTRIBUTE the forwards gate on, not just the
        # param allocation — a None here silently drops the global at every forward
        self.assertIs(m.has_global_scale, True)
        p1, p2, p3, p4 = _loaded(kernel)
        with p1, p2, p3, p4:
            fg.finegrained_batched_mm_experts_forward(m, *self._route())
        (call,) = rec.calls["moe_fused_batched"]
        self.assertIs(call.kwargs["gate_up_proj_global_scale"], m.gate_up_proj_global_scale)
        self.assertIs(call.kwargs["down_proj_global_scale"], m.down_proj_global_scale)

    def test_biases_ride_the_kernel_chain(self):
        kernel, rec = _fake_bundle()
        m = self._experts(has_gate=True, has_bias=True)
        p1, p2, p3, p4 = _loaded(kernel)
        with p1, p2, p3, p4:
            out = fg.finegrained_batched_mm_experts_forward(m, *self._route())
        self.assertTrue(torch.isfinite(out).all())
        (call,) = rec.calls["moe_fused_batched"]
        # the kernels add gate_up's bias before the gated split and down's before the
        # routing-weight multiply; nothing is added host-side
        self.assertIs(call.kwargs["gate_up_proj_bias"], m.gate_up_proj_bias)
        self.assertIs(call.kwargs["down_proj_bias"], m.down_proj_bias)

    def test_unfusable_act_fn_is_passed_as_the_module_glu(self):
        kernel, rec = _fake_bundle()
        m = self._experts(has_gate=True, has_bias=True)
        m.act_fn_name = "quick_gelu"  # not in the kernels' get_supported_act_fns()
        p1, p2, p3, p4 = _loaded(kernel)
        with p1, p2, p3, p4:
            fg.finegrained_batched_mm_experts_forward(m, *self._route())
        (call,) = rec.calls["moe_fused_batched"]
        # the kernels run the module's own GLU on the host between the two GEMMs — a new
        # activation never waits for a kernel release; the bias still rides the GEMM
        self.assertEqual(call.kwargs["act_fn"], m._apply_gate)  # bound-method equality
        self.assertIs(call.kwargs["gate_up_proj_bias"], m.gate_up_proj_bias)


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

    def test_blocks_and_scales_convert_to_the_kernel_pair_and_back(self):
        from transformers.integrations.finegrained import FineGrainedPackedBlocks, FineGrainedScaleContainer

        torch.manual_seed(0)
        E, N, K = 2, 8, 64  # N = 2I interleaved gate|up rows
        blocks = torch.randint(0, 256, (E, N, K // 32, 16), dtype=torch.uint8)
        scales = torch.randint(110, 140, (E, N, K // 32), dtype=torch.uint8)

        weight = FineGrainedPackedBlocks().convert({"gate_up_proj_blocks$": blocks}, target_patterns=["gate_up_proj"])[
            "gate_up_proj"
        ]
        self.assertEqual((weight.dtype, weight.shape), (torch.int8, (E, N, K // 2)))
        back = FineGrainedPackedBlocks().reverse_op.convert({"gate_up_proj": weight})["gate_up_proj"]
        self.assertTrue(torch.equal(back, blocks))

        # the scales are the container op's job: a module holding e8m0 receives the exponent bytes as is
        module = fg.FineGrainedExperts.__new__(fg.FineGrainedExperts)
        torch.nn.Module.__init__(module)
        module.gate_up_proj_scale_inv = torch.nn.Parameter(
            torch.zeros(E, N, K // 32, dtype=torch.float8_e8m0fnu), requires_grad=False
        )
        model = torch.nn.Module()
        model.experts = module
        scale_inv = FineGrainedScaleContainer().convert(
            {"gate_up_proj_scales$": scales},
            model=model,
            full_layer_name="experts.gate_up_proj_scale_inv",
            target_patterns=["gate_up_proj_scale_inv"],
        )["gate_up_proj_scale_inv"]
        self.assertEqual(scale_inv.dtype, torch.float8_e8m0fnu)

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
            with self.assertRaisesRegex(RuntimeError, "held swizzled"):
                forward(experts, hs, idx, wts)


@require_torch
class FineGrainedScaleLayoutTest(unittest.TestCase):
    """The module HOLDS its block scales swizzled (a 5-D Parameter) when its chain runs the tcgen05
    scaled-MMA under a triton dispatch on SM100 — keyed off the declared weight format, not the
    scale dtype (V4-style block-FP8 ships UE8M0 scales) — and affine otherwise; the loader's
    `FineGrainedSwizzleScales` op fills that layout and its reverse restores the checkpoint grid."""

    E, HIDDEN, INTER = 4, 256, 128  # gate_up rows 2*INTER = 256 and down rows HIDDEN = 256: whole 128-row blocks

    def _experts(self, weight_format, activation_format=None, impl="grouped_mm", sm100=True):
        cfg = _Cfg()
        cfg.hidden_size, cfg.intermediate_size, cfg.num_local_experts = self.HIDDEN, self.INTER, self.E
        cfg._experts_implementation = impl
        with (
            mock.patch.object(fg, "is_sm100", return_value=sm100),
            mock.patch("torch.cuda.is_available", return_value=True),
        ):
            experts = fg.FineGrainedExperts(
                cfg, block_size=(128, 128), weight_format=weight_format, activation_format=activation_format
            )
        model = torch.nn.Module()
        model.config = cfg
        model.experts = experts
        return model, experts

    def test_module_holds_swizzled_scales_for_the_scaled_mma_chain(self):
        _, experts = self._experts("mxfp8")
        self.assertEqual(experts.gate_up_proj_scale_inv.shape, (4, 2, 2, 2, 256))  # (E, 256/128, (256/32)/4, 2, 256)
        self.assertEqual(experts.down_proj_scale_inv.shape, (4, 2, 1, 2, 256))
        self.assertEqual(experts.gate_up_proj_scale_inv.dtype, torch.float8_e8m0fnu)
        self.assertTrue(experts._gate_up_interleaved)
        # every MX group-32 format holds UE8M0 scales whatever `scale_fmt` says (the kernels reject a
        # float32 grid there; GPT-OSS's config has no scale_fmt at all)
        _, experts = self._experts("mxfp4", activation_format="bf16")
        self.assertEqual(experts.gate_up_proj_scale_inv.dtype, torch.float8_e8m0fnu)

    def test_affine_off_sm100_weight_only_deepgemm_and_block_fp8(self):
        for kw in (
            {"weight_format": "mxfp8", "sm100": False},
            {"weight_format": "mxfp4", "activation_format": "bf16"},  # W4A16 reads scales per group affinely
            {"weight_format": "mxfp8", "impl": "deepgemm"},  # the DeepGEMM backends read affine scales
            {"weight_format": "nvfp4", "impl": "deepgemm_megamoe"},
        ):
            with self.subTest(**kw):
                _, experts = self._experts(**kw)
                self.assertEqual(
                    experts.gate_up_proj_scale_inv.shape,
                    (4, 256, 256 // (16 if kw["weight_format"] == "nvfp4" else 32)),
                )
        # block-FP8's (N/128, K/128) grid never reaches a scaled-MMA, whatever its scale dtype
        _, experts = self._experts("fp8")
        self.assertEqual(experts.gate_up_proj_scale_inv.shape, (4, 2, 2))
        with mock.patch.dict("os.environ", {"TRANSFORMERS_FINEGRAINED_NO_SWIZZLE": "1"}):
            _, experts = self._experts("mxfp8")
        self.assertEqual(experts.gate_up_proj_scale_inv.shape, (4, 256, 8))

    def test_megamoe_holds_gate_up_stacked(self):
        _, experts = self._experts("fp8", impl="deepgemm_megamoe")
        self.assertFalse(experts._gate_up_interleaved)

    def _op_kernel(self):
        kernel, _ = _fake_bundle()
        kernel.swizzle_mx_scales = mock.Mock(
            side_effect=lambda s: s.reshape(s.shape[0], s.shape[1] // 128, -1, 2, 256)
        )
        kernel.unswizzle_mx_scales = mock.Mock(
            side_effect=lambda s, rows, cols, num_experts=None: s.reshape(num_experts, rows, cols)
        )
        return kernel

    def test_swizzle_op_fills_only_the_scales_the_module_holds_swizzled(self):
        from transformers.integrations.finegrained import FineGrainedSwizzleScales

        model, experts = self._experts("mxfp8")
        kernel = self._op_kernel()
        op = FineGrainedSwizzleScales(hf_quantizer=None)
        grid = torch.zeros(4, 256, 8, dtype=torch.float8_e8m0fnu)
        weight = torch.zeros(4, 256, 256, dtype=torch.float8_e4m3fn)
        p1, p2, p3, p4 = _loaded(kernel)
        with p1, p2, p3, p4:
            # the loader hands a single target under its pattern, the full name (with suffix) alongside
            out = op.convert(
                {"mlp.experts.gate_up_proj": grid}, model=model, full_layer_name="experts.gate_up_proj_scale_inv"
            )
            self.assertEqual(out["mlp.experts.gate_up_proj"].shape, (4, 2, 2, 2, 256))
            # the same converter's weight passes through
            out = op.convert({"mlp.experts.gate_up_proj": weight}, model=model, full_layer_name="experts.gate_up_proj")
            self.assertIs(out["mlp.experts.gate_up_proj"], weight)
            # a multi-tensor converter (GPT-OSS deserialize) names its outputs fully
            out = op.convert(
                {
                    "experts.down_proj": weight,
                    "experts.down_proj_scale_inv": torch.zeros(4, 256, 4, dtype=torch.float8_e8m0fnu),
                },
                model=model,
                full_layer_name="experts.down_proj",
            )
            self.assertEqual(out["experts.down_proj_scale_inv"].shape, (4, 2, 1, 2, 256))
            self.assertIs(out["experts.down_proj"], weight)
            # a module that holds affine scales keeps them
            affine_model, _ = self._experts("mxfp4", activation_format="bf16")
            out = op.convert(
                {"mlp.experts.gate_up_proj": grid},
                model=affine_model,
                full_layer_name="experts.gate_up_proj_scale_inv",
            )
            self.assertIs(out["mlp.experts.gate_up_proj"], grid)
        self.assertEqual(kernel.swizzle_mx_scales.call_count, 2)

    def test_scale_container_op_round_trips_every_container(self):
        """dsv4-flash-base ships UE8M0 scales as float32 values, MiniMax as uint8 exponent bytes:
        the op brings both into the held e8m0 (exact cast / same bytes), records the container on
        the module, and its reverse restores that container on save; weights and native scales
        pass both ways."""
        from transformers.integrations.finegrained import FineGrainedScaleContainer

        op = FineGrainedScaleContainer(hf_quantizer=None)
        native = torch.pow(2.0, torch.randint(-8, 8, (4, 256, 8)).float()).to(torch.float8_e8m0fnu)
        for container, shipped in ((torch.float32, native.float()), (torch.uint8, native.view(torch.uint8))):
            with self.subTest(container=container):
                model, experts = self._experts("mxfp8", sm100=False)  # affine e8m0 scale Parameters
                out = op.convert(
                    {"mlp.experts.gate_up_proj": shipped},
                    model=model,
                    full_layer_name="experts.gate_up_proj_scale_inv",
                )
                held = out["mlp.experts.gate_up_proj"]
                self.assertEqual(held.dtype, torch.float8_e8m0fnu)
                self.assertTrue(torch.equal(held.view(torch.uint8), native.view(torch.uint8)))
                self.assertIs(experts.scale_container_dtype, container)
                back = op.reverse_op.convert({"x": held}, model=model)["x"]
                self.assertEqual(back.dtype, container)
                self.assertTrue(torch.equal(back, shipped))
                weight = torch.zeros(4, 256, 256, dtype=torch.float8_e4m3fn)
                self.assertIs(op.reverse_op.convert({"w": weight}, model=model)["w"], weight)
        model, experts = self._experts("mxfp8", sm100=False)
        self.assertIs(
            op.convert({"mlp.experts.down_proj": native}, model=model, full_layer_name="experts.down_proj_scale_inv")[
                "mlp.experts.down_proj"
            ],
            native,
        )
        self.assertIsNone(experts.scale_container_dtype)
        self.assertIs(op.reverse_op.convert({"x": native}, model=model)["x"], native)  # nothing recorded: native stays

    def test_reverse_op_restores_the_affine_grid_from_the_artifact_alone(self):
        from transformers.integrations.finegrained import FineGrainedSwizzleScales

        kernel = self._op_kernel()
        reverse = FineGrainedSwizzleScales(hf_quantizer=None).reverse_op
        artifact = torch.zeros(4, 2, 2, 2, 256, dtype=torch.float8_e8m0fnu)
        affine = torch.zeros(4, 256, 8, dtype=torch.float8_e8m0fnu)
        p1, p2, p3, p4 = _loaded(kernel)
        with p1, p2, p3, p4:
            out = reverse.convert({"experts.gate_up_proj_scale_inv": artifact}, model=None, full_layer_name="x")
            self.assertEqual(out["experts.gate_up_proj_scale_inv"].shape, (4, 256, 8))
            kernel.unswizzle_mx_scales.assert_called_once_with(artifact, 256, 8, num_experts=4)
            out = reverse.convert({"experts.gate_up_proj_scale_inv": affine}, model=None, full_layer_name="x")
            self.assertIs(out["experts.gate_up_proj_scale_inv"], affine)

    def _quantizer(self, quant_method):
        from transformers.quantizers.quantizer_finegrained import FineGrainedHfQuantizer
        from transformers.utils.quantization_config import FineGrainedConfig

        return FineGrainedHfQuantizer(FineGrainedConfig(quant_method=quant_method)).update_weight_conversions

    def _arch_converters(self):
        from transformers.core_model_loading import Concatenate, MergeModulelist, WeightConverter

        return [
            WeightConverter(
                source_patterns=["mlp.experts.*.gate_proj.weight", "mlp.experts.*.up_proj.weight"],
                target_patterns="mlp.experts.gate_up_proj",
                operations=[MergeModulelist(dim=0), Concatenate(dim=1)],
            ),
            WeightConverter(
                source_patterns="mlp.experts.*.down_proj.weight",
                target_patterns="mlp.experts.down_proj",
                operations=[MergeModulelist(dim=0)],
            ),
            WeightConverter(
                source_patterns=["mlp.fc1.weight", "mlp.fc2.weight"],
                target_patterns="mlp.fc.weight",
                operations=[Concatenate(dim=0)],
            ),
        ]

    def test_quantizer_attaches_the_layout_ops_to_expert_converters(self):
        from transformers.core_model_loading import WeightConverter
        from transformers.integrations.finegrained import (
            FineGrainedInterleaveGateUp,
            FineGrainedScaleContainer,
            FineGrainedSwizzleScales,
        )

        converters = self._quantizer("mxfp8")(self._arch_converters())
        by_target = {_targets(c)[0]: c for c in converters if isinstance(c, WeightConverter)}
        gate_up_ops = by_target["mlp.experts.gate_up_proj"].operations
        # stacked [gate; up] -> [g0, u0, ...], the scale into the held dtype, then the swizzle packs
        # the final row order (it reads 1-byte scales)
        self.assertIsInstance(gate_up_ops[-3], FineGrainedInterleaveGateUp)
        self.assertFalse(gate_up_ops[-3].inverse)
        self.assertIsInstance(gate_up_ops[-2], FineGrainedScaleContainer)
        self.assertIsInstance(gate_up_ops[-1], FineGrainedSwizzleScales)
        self.assertIsInstance(by_target["mlp.experts.down_proj"].operations[-1], FineGrainedSwizzleScales)
        self.assertIsInstance(by_target["weight_scale_inv"].operations[0], FineGrainedScaleContainer)  # dense linears
        self.assertEqual(len(by_target["mlp.fc.weight"].operations), 1)  # dense converters untouched
        # keys arriving under the fused names get the same layout
        self.assertIn("experts.gate_up_proj_bias", by_target)  # patterns are stored anchor-stripped
        self.assertIsInstance(by_target["experts.down_proj_scale_inv"].operations[-1], FineGrainedSwizzleScales)
        # ...and saving reverses in the opposite order: unswizzle first, then de-interleave
        reverse_ops = by_target["mlp.experts.gate_up_proj"].reverse_transform().operations
        self.assertIsInstance(reverse_ops[0], FineGrainedSwizzleScales)
        self.assertTrue(reverse_ops[0].inverse)
        self.assertIsInstance(reverse_ops[2], FineGrainedInterleaveGateUp)
        self.assertTrue(reverse_ops[2].inverse)

    def test_blocks_scales_converters_carry_the_layout_ops(self):
        from transformers.core_model_loading import WeightConverter
        from transformers.integrations.finegrained import FineGrainedInterleaveGateUp, FineGrainedSwizzleScales

        converters = self._quantizer("mxfp4")([])
        by_target = {_targets(c)[0]: c for c in converters if isinstance(c, WeightConverter)}
        ops = by_target["gate_up_proj"].operations
        self.assertIsInstance(ops[-2], FineGrainedInterleaveGateUp)
        self.assertIsInstance(ops[-1], FineGrainedSwizzleScales)
        self.assertIsInstance(by_target["gate_up_proj_scale_inv"].operations[0], FineGrainedInterleaveGateUp)
        self.assertIsInstance(by_target["down_proj"].operations[-1], FineGrainedSwizzleScales)
        self.assertIsInstance(by_target["down_proj_scale_inv"].operations[-1], FineGrainedSwizzleScales)

    def test_catch_all_converters_deliver_the_full_parameter_name(self):
        """A key already under the fused name (GPT-OSS's gate_up_proj_bias, a renamed MiniMax
        down_proj_scale_inv) goes through the real converter plumbing: the ops receive the tensor
        under the source PATTERN and must hand back the target, which the loader expands to the
        full name."""
        from transformers.core_model_loading import WeightConverter, rename_source_key

        model, _ = self._experts("mxfp8")
        converters = [c for c in self._quantizer("mxfp8")([]) if isinstance(c, WeightConverter)]
        for key, shape in (("experts.gate_up_proj_bias", (4, 256)), ("experts.down_proj_scale_inv", (4, 256, 4))):
            renamed, source_pattern = rename_source_key(key, [], converters)
            self.assertEqual(renamed, key)
            converter = next(c for c in converters if source_pattern in c.source_patterns)
            tensor = torch.zeros(shape, dtype=torch.float8_e8m0fnu if "scale" in key else torch.float32)
            converter.add_tensor(renamed, key, source_pattern, lambda t=tensor: t)
            kernel = self._op_kernel()
            p1, p2, p3, p4 = _loaded(kernel)
            with p1, p2, p3, p4:
                out = converter.convert(renamed, model=model)
            self.assertEqual(list(out), [key])

    def test_replacement_keeps_the_models_gate_up_convention(self):
        """`replace_with_finegrained_layer` goes through `use_experts_implementation`, which stamps
        the layout flags on the instance after __init__ — GPT-OSS's `is_concatenated=False` must
        survive it (it decides whether the loader interleaves)."""
        from transformers import GptOssConfig, GptOssForCausalLM, Qwen3MoeConfig, Qwen3MoeForCausalLM
        from transformers.utils.quantization_config import FineGrainedConfig

        gpt_oss = GptOssConfig(
            vocab_size=64,
            hidden_size=64,
            intermediate_size=32,
            num_local_experts=2,
            num_experts_per_tok=1,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=32,
            sliding_window=16,
        )
        qwen = Qwen3MoeConfig(
            vocab_size=64,
            hidden_size=64,
            intermediate_size=64,
            moe_intermediate_size=32,
            num_experts=2,
            num_experts_per_tok=1,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=32,
        )
        for cls, cfg, quant_method, expected in (
            (GptOssForCausalLM, gpt_oss, "mxfp4", False),
            (Qwen3MoeForCausalLM, qwen, "mxfp8", True),
        ):
            with self.subTest(model=cls.__name__), torch.device("meta"):
                model = cls(cfg)
                fg.replace_with_finegrained_layer(
                    model, modules_to_not_convert=[], quantization_config=FineGrainedConfig(quant_method=quant_method)
                )
                experts = model.model.layers[0].mlp.experts
                self.assertIsInstance(experts, fg.FineGrainedExperts)
                self.assertIs(experts.is_concatenated, expected)

    def test_interleave_op_follows_the_model_and_the_experts_backend(self):
        """The op reads the model's experts modules: a model whose own rows already come
        interleaved (`is_concatenated=False`, GPT-OSS) keeps them, Mega MoE packs gate|up itself
        and keeps the stack; every other case gets the interleaved order, and the reverse restores
        the stack."""
        from transformers.integrations.finegrained import FineGrainedInterleaveGateUp

        stacked = torch.arange(2 * 6 * 4, dtype=torch.float32).reshape(2, 6, 4)  # rows [g0,g1,g2,u0,u1,u2]
        op = FineGrainedInterleaveGateUp(hf_quantizer=None)
        model, experts = self._experts("mxfp8")
        out = op.convert({"mlp.experts.gate_up_proj": stacked}, model=model)["mlp.experts.gate_up_proj"]
        torch.testing.assert_close(out, stacked[:, [0, 3, 1, 4, 2, 5]])
        back = op.reverse_op.convert({"mlp.experts.gate_up_proj": out}, model=model)["mlp.experts.gate_up_proj"]
        torch.testing.assert_close(back, stacked)
        experts.is_concatenated = False
        self.assertIs(op.convert({"x": stacked}, model=model)["x"], stacked)
        experts.is_concatenated = True
        experts._gate_up_interleaved = False
        self.assertIs(op.convert({"x": stacked}, model=model)["x"], stacked)


@require_torch
class FineGrainedOnTheFlyQuantizeTest(unittest.TestCase):
    """`FineGrainedQuantize` reads the module's format: block-FP8 in torch (the group formats run the
    kernels' quantizers — covered with real kernels below), emitted in the module's layout."""

    def test_block_fp8_round_trips_within_its_floor(self):
        from transformers.integrations.finegrained import FineGrainedDequantize, FineGrainedQuantize

        torch.manual_seed(0)
        cfg = _Cfg()
        cfg.hidden_size, cfg.intermediate_size, cfg.num_local_experts = 256, 128, 4
        for scale_fmt in ("float", "ue8m0"):
            with self.subTest(scale_fmt=scale_fmt), mock.patch.object(fg, "is_sm100", return_value=False):
                experts = fg.FineGrainedExperts(cfg, block_size=(128, 128), weight_format="fp8", scale_fmt=scale_fmt)
                model = torch.nn.Module()
                model.experts = experts
                original = torch.randn(4, 256, 256) * 0.02
                out = FineGrainedQuantize(hf_quantizer=None).convert({"experts.gate_up_proj": original}, model=model)
                weight, scale = out["experts.gate_up_proj"], out["experts.gate_up_proj_scale_inv"]
                self.assertEqual(
                    (weight.dtype, weight.shape), (experts.gate_up_proj.dtype, experts.gate_up_proj.shape)
                )
                self.assertEqual(
                    (scale.dtype, scale.shape),
                    (experts.gate_up_proj_scale_inv.dtype, experts.gate_up_proj_scale_inv.shape),
                )
                deq = FineGrainedDequantize(None)._dequantize_one(weight, scale.float(), output_dtype=torch.float32)
                rel = ((deq - original).norm() / original.norm()).item()
                self.assertLess(rel, 4e-2, f"{scale_fmt}: {rel:.3f}")


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

    def test_on_the_fly_group_formats_round_trip_within_their_floor(self):
        """MXFP8 / MXFP4 / NVFP4 on-the-fly quantization through the kernels' quantizers, emitted in
        the module's layout (swizzled where held so); dequantizing them back lands within each
        format's floor."""
        from transformers.integrations.finegrained import FineGrainedDequantize, FineGrainedQuantize

        torch.manual_seed(0)
        cfg = _Cfg()
        cfg.hidden_size, cfg.intermediate_size, cfg.num_local_experts = 512, 256, 4
        cfg._experts_implementation = "grouped_mm"
        for fmt, floor in (("mxfp8", 4e-2), ("mxfp4", 2e-1), ("nvfp4", 2e-1)):
            with self.subTest(fmt=fmt):
                experts = FineGrainedExperts(cfg, weight_format=fmt).cuda()
                model = torch.nn.Module()
                model.experts = experts
                original = (torch.randn(4, 512, 512, device="cuda") * 0.02).to(torch.bfloat16)
                out = FineGrainedQuantize(hf_quantizer=None).convert({"experts.gate_up_proj": original}, model=model)
                weight, scale = out["experts.gate_up_proj"], out["experts.gate_up_proj_scale_inv"]
                self.assertEqual(
                    (weight.dtype, weight.shape), (experts.gate_up_proj.dtype, experts.gate_up_proj.shape)
                )
                self.assertEqual(
                    (scale.dtype, scale.shape),
                    (experts.gate_up_proj_scale_inv.dtype, experts.gate_up_proj_scale_inv.shape),
                )
                if scale.ndim == 5:
                    kernel = load_finegrained_kernel()
                    scale = kernel.unswizzle_mx_scales(scale, 512, scale.shape[2] * 4, num_experts=4)
                deq = FineGrainedDequantize(None)._dequantize_one(weight, scale.float(), output_dtype=torch.float32)
                if fmt == "nvfp4":
                    deq = deq * out["experts.gate_up_proj_global_scale"].reshape(-1, 1, 1)
                else:
                    self.assertNotIn("experts.gate_up_proj_global_scale", out)
                rel = ((deq - original.float()).norm() / original.float().norm()).item()
                self.assertLess(rel, floor, f"{fmt}: {rel:.3f}")
                self.assertGreater(rel, 0.0)

    def test_experts_scale_layout_op_keeps_the_forward_correct(self):
        """End-to-end over the integration: real MXFP8 experts hold swizzled scales, the loader's op
        fills them from the affine grid, the fused forward matches the affine module's, and the
        reverse op hands the affine grid back bitwise."""
        from transformers.integrations.finegrained import FineGrainedSwizzleScales

        cfg = _Cfg()
        cfg.hidden_size, cfg.intermediate_size, cfg.num_local_experts = (
            512,
            256,
            4,
        )  # K >= 256: the swizzled arm has tiles
        cfg._experts_implementation = "grouped_mm"
        experts = FineGrainedExperts(cfg, weight_format="mxfp8").cuda()
        with mock.patch.dict("os.environ", {"TRANSFORMERS_FINEGRAINED_NO_SWIZZLE": "1"}):
            affine_experts = FineGrainedExperts(cfg, weight_format="mxfp8").cuda()
        model = torch.nn.Module()
        model.experts = experts
        op = FineGrainedSwizzleScales(hf_quantizer=None)

        affine = {}
        for proj, rows in (("gate_up_proj", 2 * cfg.intermediate_size), ("down_proj", cfg.hidden_size)):
            cols = cfg.hidden_size if proj == "gate_up_proj" else cfg.intermediate_size
            weight = torch.randn(cfg.num_local_experts, rows, cols, device="cuda").to(torch.float8_e4m3fn)
            grid = torch.randint(120, 134, (cfg.num_local_experts, rows, cols // 32), dtype=torch.uint8, device="cuda")
            affine[proj] = grid.view(torch.float8_e8m0fnu)
            for module in (experts, affine_experts):
                setattr(module, proj, torch.nn.Parameter(weight.clone(), requires_grad=False))
            setattr(affine_experts, f"{proj}_scale_inv", torch.nn.Parameter(affine[proj].clone(), requires_grad=False))
            self.assertEqual(
                getattr(experts, f"{proj}_scale_inv").dim(), 5, "the module should hold this projection swizzled"
            )
            out = op.convert(
                {f"mlp.experts.{proj}": affine[proj]}, model=model, full_layer_name=f"experts.{proj}_scale_inv"
            )
            swizzled = out[f"mlp.experts.{proj}"]
            self.assertEqual(swizzled.shape, getattr(experts, f"{proj}_scale_inv").shape)
            setattr(experts, f"{proj}_scale_inv", torch.nn.Parameter(swizzled, requires_grad=False))

        x = torch.randn(8, cfg.hidden_size, device="cuda", dtype=torch.bfloat16)
        idx = torch.randint(0, cfg.num_local_experts, (8, 2), device="cuda", dtype=torch.long)
        wts = torch.rand(8, 2, device="cuda", dtype=torch.bfloat16)
        reference = fg.finegrained_grouped_mm_experts_forward(affine_experts, x, idx, wts)
        out = fg.finegrained_grouped_mm_experts_forward(experts, x, idx, wts)
        torch.cuda.synchronize()
        rel = ((out.float() - reference.float()).norm() / reference.float().norm().clamp(min=1e-9)).item()
        # different tuned arms reorder the fp32 reduction (~1e-5); a layout bug is O(1)
        self.assertLess(rel, 1e-3, f"swizzled forward diverged from the affine one: {rel:.2e}")
        # the eager per-expert loop reads one expert's slice of the swizzled stack directly
        eager_reference = affine_experts(x, idx, wts)
        eager_out = experts(x, idx, wts)
        rel = (
            (eager_out.float() - eager_reference.float()).norm() / eager_reference.float().norm().clamp(min=1e-9)
        ).item()
        self.assertLess(rel, 1e-3, f"eager forward on swizzled scales diverged from affine: {rel:.2e}")

        reverse = op.reverse_op
        for proj, grid in affine.items():
            restored = reverse.convert({proj: getattr(experts, f"{proj}_scale_inv").data})[proj]
            self.assertTrue(torch.equal(restored.view(torch.uint8), grid.view(torch.uint8)), proj)
