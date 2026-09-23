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
"""Marshalling tests for the multi-format ``finegrained`` integration: the kernel bundle is
mocked, so these pin exactly what the integration passes to `kernels-community/finegrained-kernels`
(the As-positional / no-block_size / expert_start / b_global_scale contract) without a GPU."""

import os
import re
import socket
import subprocess
import tempfile
import unittest
import warnings
from contextlib import ExitStack, contextmanager
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
from transformers.testing_utils import (
    TestCasePlus,
    require_torch,
    require_torch_gpu,
    require_torch_multi_accelerator,
)


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
    kernel.get_supported_norms = lambda: ("rms_norm", "centered_rms_norm", "input_scaled_rms_norm")
    return kernel, rec


class _Cfg:
    hidden_size = 64
    num_local_experts = 4
    intermediate_size = 32
    hidden_act = "silu"
    _experts_implementation = None


def _loaded(kernel):
    return (
        mock.patch.object(fg, "_FINEGRAINED", None),
        mock.patch.object(fg, "is_kernels_available", return_value=True),
        mock.patch.object(fg, "lazy_load_kernel", return_value=kernel),
        # a locally importable checkout (FINEGRAINED_KERNELS_PATH / installed package)
        # takes precedence over the hub loader; tests must stay hermetic to the fake bundle
        mock.patch.object(fg, "_import_local_finegrained", return_value=None),
    )


def _quantizer_for(config):
    """The quantizer a `from_pretrained` would pick for this config. Since the arms were split
    per scheme, constructing the base directly only exercises what every scheme shares."""
    from transformers.quantizers.auto import AUTO_QUANTIZER_MAPPING

    method = getattr(config.quant_method, "value", config.quant_method)
    return AUTO_QUANTIZER_MAPPING[method](config)


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

    def test_loader_is_compile_safe_cold(self):
        """Cold path: the compiled call is first to load, so the opaque loader node runs its full
        body under compile and must return None, never the bundle (`torch.* op returned
        non-Tensor`). The loader has no arch gate, so nothing here fakes a device."""
        kernel, _ = _fake_bundle()
        kernel.matmul_2d = lambda x, *a, **k: x + 1
        p1, p2, p3, p4 = _loaded(kernel)
        with p1, p2, p3, p4:
            torch.compiler.reset()

            @torch.compile(fullgraph=True)
            def run(x):
                return load_finegrained_kernel().matmul_2d(x)

            out = run(torch.zeros(3))
        self.assertTrue(torch.equal(out, torch.ones(3)))

    def test_loader_is_compile_safe_when_warm(self):
        """Warm path, which is the production order: eager warm-up, then compile. The loader hits
        its short-circuit at trace time — the branch that must also return None."""
        kernel, _ = _fake_bundle()
        kernel.matmul_2d = lambda x, *a, **k: x + 1
        p1, p2, p3, p4 = _loaded(kernel)
        with p1, p2, p3, p4:
            load_finegrained_kernel()
            torch.compiler.reset()

            @torch.compile(fullgraph=True)
            def run(x):
                return load_finegrained_kernel().matmul_2d(x)

            out = run(torch.zeros(3))
        self.assertTrue(torch.equal(out, torch.ones(3)))

    def test_loader_binds_all_symbols(self):
        kernel, _ = _fake_bundle()
        p1, p2, p3, p4 = _loaded(kernel)
        with p1, p2, p3, p4:
            bundle = load_finegrained_kernel()
        self.assertIs(bundle.matmul_2d, kernel.matmul_2d)
        self.assertIs(bundle.get_supported_act_fns, kernel.get_supported_act_fns)


@require_torch
class FineGrainedLinearMarshallingTest(unittest.TestCase):
    def _run(self, **linear_kwargs):
        kernel, rec = _fake_bundle()
        p1, p2, p3, p4 = _loaded(kernel)
        with p1, p2, p3, p4, mock.patch.object(deepgemm, "is_deepgemm_loadable", return_value=False):
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
        kernel, rec = _fake_bundle()
        p1, p2, p3, p4 = _loaded(kernel)
        with p1, p2, p3, p4, mock.patch.object(deepgemm, "is_deepgemm_loadable", return_value=False):
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
@require_torch
class FineGrainedGroupedLinearSwapTest(unittest.TestCase):
    def test_a_grouped_linear_is_recognised_by_its_groups_not_its_name(self):
        """`replace_with_finegrained_layer` picks `FineGrainedGroupedLinear` for a block-diagonal
        linear — a plain one would collapse the groups into a single giant matmul and return the
        wrong output dim. The branch keys on `n_groups`, the attribute the swap consumes: a class
        NAME substring would miss a grouped linear named anything else, and would match (then
        `AttributeError` on) one that carries its group count under another name.
        """
        from types import SimpleNamespace

        import torch.nn as nn

        from transformers.integrations.finegrained import FineGrainedGroupedLinear, replace_with_finegrained_layer
        from transformers.utils.quantization_config import FineGrainedConfig

        class OddlyNamedBlockDiagonal(nn.Linear):
            def __init__(self):
                super().__init__(8, 16, bias=False)
                self.n_groups = 2

        model = nn.Module()
        model.config = SimpleNamespace(get_text_config=lambda: SimpleNamespace())
        model.grouped = OddlyNamedBlockDiagonal()
        model.plain = nn.Linear(8, 16, bias=False)

        with torch.device("meta"):
            replace_with_finegrained_layer(model, None, FineGrainedConfig(quant_method="fp8"))
        self.assertIsInstance(model.grouped, FineGrainedGroupedLinear)
        self.assertNotIsInstance(model.plain, FineGrainedGroupedLinear)


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
        self.assertIsNone(call.kwargs["gate_up_proj_weight_global_scale"])
        self.assertEqual(call.kwargs["act_fn"], "silu")  # fusable: passed by name
        self.assertIs(call.kwargs["gate"], True)
        self.assertIsNone(call.kwargs["activation_format"])  # None = the weight family's format
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

    def test_activation_format_passes_through(self):
        kernel, rec = _fake_bundle()
        for activation_format in (None, "bf16", "mxfp8"):
            m = self._experts(has_gate=True, activation_format=activation_format)
            p1, p2, p3, p4 = _loaded(kernel)
            with p1, p2, p3, p4:
                fg.finegrained_batched_mm_experts_forward(m, *self._route())
            self.assertEqual(rec.calls["moe_fused_batched"][-1].kwargs["activation_format"], activation_format)

    def test_nvfp4_experts_thread_per_expert_globals(self):
        kernel, rec = _fake_bundle()
        m = self._experts(has_gate=True, weight_format="nvfp4")
        # the format table must resolve the ATTRIBUTE the forwards gate on, not just the
        # param allocation — a None here silently drops the global at every forward
        self.assertIsNotNone(m.global_scale_dtype)
        p1, p2, p3, p4 = _loaded(kernel)
        with p1, p2, p3, p4:
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
        kernel, rec = _fake_bundle()
        m = self._experts(has_gate=True, weight_format="nvfp4", activation_format="bf16")
        p1, p2, p3, p4 = _loaded(kernel)
        with p1, p2, p3, p4:
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
        kernel, rec = _fake_bundle()
        m = self._experts(has_gate=True)
        m.post_expert_norm, m.has_post_expert_norm = torch.nn.LayerNorm(m.hidden_dim), True
        p1, p2, p3, p4 = _loaded(kernel)
        with p1, p2, p3, p4:
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
        with p1, p2, p3, p4, mock.patch.object(deepgemm, "is_deepgemm_loadable", return_value=False):
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
        kernel, rec = _fake_bundle()
        m = self._experts(has_gate=True)
        m.post_expert_norm, m.has_post_expert_norm = torch.nn.RMSNorm(m.hidden_dim, eps=1e-4), True
        p1, p2, p3, p4 = _loaded(kernel)
        for name, fused in (("rms_norm", True), ("input_scaled_rms_norm", True), ("a_models_own_norm", False)):
            m.post_expert_norm_name = name
            with p1, p2, p3, p4:
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
            operands = fg._moe_operands(kernel, module)
            for projection in ("gate_up_proj", "down_proj"):
                with self.subTest(scheme=module.activation_scheme, projection=projection):
                    scale = operands[f"{projection}_activation_scale"]
                    self.assertEqual(scale is not None, calibrated)
                    if calibrated:
                        self.assertEqual(scale.numel(), _Cfg.num_local_experts)

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
class FineGrainedScaleDtypeTest(unittest.TestCase):
    """A scale's dtype is the format's, never the ambient default.

    `from_pretrained` sets the default dtype to the checkpoint's for the duration of model
    construction, so a scale allocated as `torch.ones(n)` comes out bf16 and the kernels read it
    as fp32 — NaN logits, no error. Building under both defaults and comparing is what catches
    that, whatever the format decides each scale should be.
    """

    def _dtypes(self, build, default):
        previous = torch.get_default_dtype()
        torch.set_default_dtype(default)
        try:
            with torch.device("meta"):
                module = build()
            return {name: p.dtype for name, p in module.named_parameters() if p is not None}
        finally:
            torch.set_default_dtype(previous)

    def _assert_default_dtype_invariant(self, build):
        under_fp32 = self._dtypes(build, torch.float32)
        under_bf16 = self._dtypes(build, torch.bfloat16)
        self.assertTrue(under_fp32)
        # the weight and bias DO follow the model dtype; every scale beside them must not
        scales = {name for name in under_fp32 if "scale" in name}
        self.assertTrue(scales)
        for name in sorted(scales):
            self.assertEqual(under_bf16[name], under_fp32[name], f"{name} followed the default dtype")

    def test_linear_scales_ignore_the_default_dtype(self):
        for weight_format, scheme in (("fp8", "static"), ("fp8", "dynamic"), ("nvfp4", "dynamic")):
            with self.subTest(weight_format=weight_format, activation_scheme=scheme):
                self._assert_default_dtype_invariant(
                    lambda f=weight_format, s=scheme: FineGrainedLinear(
                        in_features=64, out_features=32, block_size=(4, 4), weight_format=f, activation_scheme=s
                    )
                )

    def test_experts_scales_ignore_the_default_dtype(self):
        for weight_format, scheme in (("fp8", "static"), ("fp8", "dynamic"), ("nvfp4", "dynamic")):
            with self.subTest(weight_format=weight_format, activation_scheme=scheme):
                self._assert_default_dtype_invariant(
                    lambda f=weight_format, s=scheme: FineGrainedExperts(
                        _Cfg(), block_size=(4, 4), weight_format=f, activation_scheme=s
                    )
                )


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
        kernel = mock.Mock()
        kernel.get_supported_act_fns.return_value = ("silu",)
        kernel.get_supported_norms.return_value = ("input_scaled_rms_norm",)
        return fg._moe_operands(kernel, module), module

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


@require_torch
class FrozenFp8ShimTest(unittest.TestCase):
    def test_frozen_module_warns_and_is_self_contained(self):
        import importlib

        import transformers.integrations.finegrained_fp8 as frozen
        from transformers.quantizers.quantizer_finegrained_fp8 import FineGrainedFP8HfQuantizer
        from transformers.utils.quantization_config import FineGrainedFP8Config

        # the whole FILE is deprecated, so importing anything out of it is what warns. Safe to
        # do at module scope only because `_LazyModule` never executes these unless asked: a
        # plain `import transformers` does not reach them.
        with warnings.catch_warnings(record=True) as raised:
            warnings.simplefilter("always")
            importlib.reload(frozen)
        self.assertTrue(
            [w for w in raised if issubclass(w.category, DeprecationWarning) and "frozen" in str(w.message)],
            [str(w.message) for w in raised],
        )
        # the quantizer is still constructible — frozen means no new recipes, not removed
        FineGrainedFP8HfQuantizer(FineGrainedFP8Config())
        # distinct machinery: the frozen classes are not the live ones
        self.assertIsNot(frozen.FP8Linear, FineGrainedLinear)


@require_torch
class FineGrainedValidateEnvironmentTest(unittest.TestCase):
    """`validate_environment` on the LIVE quantizer. The only tests that covered a
    `validate_environment` sat on `FineGrainedFP8HfQuantizer`, which no `from_pretrained` can
    reach any more (`AUTO_QUANTIZER_MAPPING["fp8"]` is `FineGrainedHfQuantizer`)."""

    @contextmanager
    def _no_accelerator(self):
        with ExitStack() as stack:
            stack.enter_context(mock.patch("torch.cuda.is_available", return_value=False))
            stack.enter_context(
                mock.patch("transformers.quantizers.finegrained.base.is_torch_xpu_available", return_value=False)
            )
            yield

    def _quantizer(self, pre_quantized, **cfg_kwargs):
        # through the registry, so the test sees the class a `from_pretrained` would pick
        from transformers.quantizers.auto import AUTO_QUANTIZER_MAPPING
        from transformers.utils.quantization_config import FineGrainedConfig

        config = FineGrainedConfig(**cfg_kwargs)
        method = getattr(config.quant_method, "value", config.quant_method)
        quantizer = AUTO_QUANTIZER_MAPPING[method](config)
        quantizer.pre_quantized = pre_quantized
        return quantizer

    def test_no_accelerator_refuses_to_quantize(self):
        # quantizing a bf16 checkpoint needs the hardware; there is nothing to fall back to
        quantizer = self._quantizer(pre_quantized=False)
        with self._no_accelerator(), self.assertRaises(RuntimeError):
            quantizer.validate_environment()

    def test_no_accelerator_dequantizes_a_quantized_checkpoint(self):
        quantizer = self._quantizer(pre_quantized=True)
        with self._no_accelerator():
            quantizer.validate_environment()
        self.assertTrue(quantizer.quantization_config.dequantize)

    def test_nvfp4_refuses_to_dequantize(self):
        # two-level scales have no slot in the dequantize chain, so the CPU fallback that rescues
        # block-FP8 must refuse here rather than hand back a weight scaled by `1 / global`
        for pre_quantized, kwargs in ((True, {}), (False, {"dequantize": True})):
            with self.subTest(pre_quantized=pre_quantized):
                quantizer = self._quantizer(pre_quantized=pre_quantized, quant_method="nvfp4", **kwargs)
                with self._no_accelerator(), self.assertRaises(NotImplementedError):
                    quantizer.validate_environment()


class FineGrainedParallelPlanTest(unittest.TestCase):
    """`update_tp_plan`: a model's plan as a quantized experts module needs it."""

    BASE_EP = {
        "layers.*.mlp.gate": "ep_router",
        "layers.*.mlp.experts.gate_up_proj": "grouped_gemm",
        "layers.*.mlp.experts.down_proj": "grouped_gemm",
        "layers.*.mlp.experts": "moe_tp_experts",
        "layers.*.self_attn.q_proj": "colwise",
    }

    def _planned(self, impl=None, raw=None):
        from types import SimpleNamespace

        from transformers.quantizers.finegrained.base import FineGrainedHfQuantizer
        from transformers.utils.quantization_config import FineGrainedConfig

        quantizer = FineGrainedHfQuantizer(FineGrainedConfig())
        config = SimpleNamespace(base_model_tp_plan=dict(raw or self.BASE_EP), _experts_implementation=impl)
        quantizer.update_tp_plan(config)
        return config.base_model_tp_plan

    def test_every_companion_shards_with_its_expert_weight(self):
        """The plan matcher keys on the exact parameter name and falls back only to the owning
        module, whose entry shards nothing — so a scale with no entry of its own stays whole while
        its weight is this rank's expert slice, and every rank but the first reads another rank's
        scales."""
        from transformers.distributed.tensor_parallel import _get_parameter_tp_plan

        plan = self._planned()
        style = lambda name: _get_parameter_tp_plan(f"layers.3.mlp.experts.{name}", plan)  # noqa: E731

        for name in ("gate_up_proj", "gate_up_proj_scale_inv", "gate_up_proj_weight_global_scale"):
            self.assertEqual(style(name), "grouped_gemm", name)
        for name in ("down_proj_scale_inv", "down_proj_bias", "down_proj_input_global_scale"):
            self.assertEqual(style(name), "grouped_gemm", name)
        # one per-tensor value for the pre-routing hidden states: replicated, so no entry of its own
        self.assertEqual(style("gate_up_proj_input_global_scale"), "moe_tp_experts")
        self.assertEqual(plan["layers.*.self_attn.q_proj"], "colwise")

    def test_intra_expert_entries_follow_the_projection_axis(self):
        """Under intra-expert TP the experts stay whole and the projection's own axis splits, so
        the scale grid must follow its weight while the per-expert globals stay replicated. And
        `packed_colwise` splits the output axis as two packed halves — the `[gate; up]` STACK —
        where a quantized module holds interleaved rows, so the weight's own style changes too."""
        from transformers.distributed.tensor_parallel import _get_parameter_tp_plan

        base = {
            "layers.*.mlp.experts.gate_up_proj": "packed_colwise",
            "layers.*.mlp.experts.down_proj": "rowwise",
            "layers.*.mlp.experts": "moe_tp_experts",
        }
        plan = self._planned(raw=base)
        style = lambda n: _get_parameter_tp_plan(f"layers.3.mlp.experts.{n}", plan)  # noqa: E731

        # the weight keeps the style the MODEL declared — `packed_colwise` is the strided split a
        # concatenated `[gate; up]` checkpoint needs, since the interleave into the kernels' row
        # order runs on each rank's shard, after sharding. The companions follow it.
        self.assertEqual(style("gate_up_proj"), "packed_colwise")
        for name in ("gate_up_proj_scale_inv", "gate_up_proj_bias"):
            self.assertEqual(style(name), "moe_experts_packed_colwise", name)
        # the down scale splits on dim 2 — the reduce axis affine and 5-D swizzled alike
        self.assertEqual(style("down_proj"), "rowwise")
        self.assertEqual(style("down_proj_scale_inv"), "moe_experts_rowwise")
        # TP does not split experts, so anything indexed per expert stays whole
        for name in ("gate_up_proj_weight_global_scale", "down_proj_weight_global_scale", "down_proj_bias"):
            self.assertEqual(style(name), "moe_tp_experts", name)

        # a natively INTERLEAVED layout declares `colwise`, and its companions split contiguously
        inter = dict(base, **{"layers.*.mlp.experts.gate_up_proj": "colwise"})
        plan_i = self._planned(raw=inter)
        self.assertEqual(_get_parameter_tp_plan("layers.3.mlp.experts.gate_up_proj", plan_i), "colwise")
        self.assertEqual(
            _get_parameter_tp_plan("layers.3.mlp.experts.gate_up_proj_scale_inv", plan_i),
            "moe_experts_colwise",
        )

    def test_dense_projections_that_share_a_name_are_left_alone(self):
        """`mlp.up_proj` and `mlp.shared_experts.down_proj` end in the same words as the stacked
        expert projections but are plain 2-D linears that colwise/rowwise already shard right.
        Rewriting them to an expert-axis style sharded a dim they do not have — a CUDA illegal
        memory access in the dense FP8 matmul, seen on DeepSeek-V3 under TP."""
        from transformers.distributed.tensor_parallel import _get_parameter_tp_plan

        plan = self._planned(
            raw={
                "layers.*.mlp.experts.gate_up_proj": "packed_colwise",
                "layers.*.mlp.experts.down_proj": "rowwise",
                "layers.*.mlp.experts": "moe_tp_experts",
                "layers.*.mlp.up_proj": "colwise",
                "layers.*.mlp.down_proj": "rowwise",
                "layers.*.mlp.shared_experts.up_proj": "colwise",
                "layers.*.mlp.shared_experts.down_proj": "rowwise",
            }
        )
        for dense, expected in (
            ("layers.*.mlp.up_proj", "colwise"),
            ("layers.*.mlp.down_proj", "rowwise"),
            ("layers.*.mlp.shared_experts.up_proj", "colwise"),
            ("layers.*.mlp.shared_experts.down_proj", "rowwise"),
        ):
            self.assertEqual(plan[dense], expected, dense)
        # and no scale entry was invented for a dense linear, whose scale `rowwise`/`colwise`
        # already reaches through the module fallback
        strays = [k for k in plan if k.endswith("_scale_inv") and ".experts." not in k]
        self.assertEqual(strays, [], f"invented dense scale entries: {strays}")
        # the stacked experts still get theirs
        self.assertEqual(
            _get_parameter_tp_plan("layers.3.mlp.experts.gate_up_proj_scale_inv", plan),
            "moe_experts_packed_colwise",
        )

    def test_a_multimodal_models_expert_plans_are_reached(self):
        """A multimodal model keeps its experts' plans on a SUB-config; the config the quantizer
        is handed carries a few projector entries and often no `base_model_ep_plan` at all.
        Rewriting only what we were handed adds no companion anywhere, while the weights still
        shard from the sub-config's own plan — so the scale stays whole against a sharded weight,
        which is a wrong answer rather than a crash."""
        from transformers import Glm4vMoeConfig
        from transformers.quantizers.finegrained.base import FineGrainedHfQuantizer
        from transformers.utils.quantization_config import FineGrainedConfig

        config = Glm4vMoeConfig()
        self.assertFalse(
            getattr(config, "base_model_ep_plan", None),
            "this model no longer nests its plans, so it cannot guard the sub-config walk",
        )
        before = dict(getattr(config.text_config, "base_model_ep_plan", None) or {})
        FineGrainedHfQuantizer(FineGrainedConfig()).update_tp_plan(config)
        after = getattr(config.text_config, "base_model_ep_plan", None) or {}

        companions = {
            k
            for k in after
            if k.endswith(("_scale_inv", "_bias", "_weight_global_scale", "_input_global_scale", "_activation_scale"))
        }
        self.assertTrue(companions, "no companion reached the sub-config's plan")
        self.assertEqual(set(before), set(after) - companions, "the model's own entries were disturbed")

    def test_dequantize_folds_every_quantized_key_to_full_precision(self):
        """`dequantize=True` is the escape hatch for hardware that cannot serve the format. It
        must claim BOTH checkpoint key layouts — the `{proj}_blocks` + `{proj}_scales` pair
        GPT-OSS ships and the `weight` + `weight_scale_inv` pair everything else does — and
        land on the plain weight, since a module that was never swapped has no scale slot to
        write into."""
        import re

        from transformers.utils.quantization_config import FineGrainedConfig

        for method, sources in (
            ("mxfp4", ("experts.gate_up_proj_blocks", "experts.gate_up_proj_scales")),
            ("fp8", ("mlp.down_proj.weight", "mlp.down_proj.weight_scale_inv")),
        ):
            quantizer = _quantizer_for(FineGrainedConfig(quant_method=method, dequantize=True))
            quantizer.pre_quantized = True
            converters = quantizer.get_weight_conversions()
            with self.subTest(quant_method=method):
                self.assertTrue(converters, f"{method}: dequantize produced no converter")
                for source in sources:
                    self.assertTrue(
                        any(
                            re.search(str(pattern), source)
                            for c in converters
                            for pattern in (
                                c.source_patterns if isinstance(c.source_patterns, list) else [c.source_patterns]
                            )
                        ),
                        f"{method}: nothing dequantizes {source}",
                    )

    def test_the_impl_rewrites_the_layer_kinds(self):
        megamoe = self._planned("deepgemm_megamoe")
        self.assertEqual(megamoe["layers.*.mlp.gate"], "megamoe_router")
        self.assertEqual(megamoe["layers.*.mlp.experts"], "megamoe_experts")

    def test_a_models_own_plan_survives_the_update(self):
        """`update_tp_plan` ADDS companions; it must never drop what the model ships.

        The config here is a REAL one, which the `SimpleNamespace` the tests above use cannot
        substitute for: a branch keyed on the config CLASS is invisible to a fake. A hardcoded
        `"Qwen3" in config.__class__.__name__` override used to assign a dense plan over
        `base_model_tp_plan`, dropping `q_norm`/`k_norm` on Qwen3 and — because the check is a
        substring — the whole expert plan on Qwen3-MoE.
        """
        from transformers import Qwen3Config, Qwen3MoeConfig, Qwen3VLMoeConfig
        from transformers.quantizers.finegrained.base import FineGrainedHfQuantizer
        from transformers.utils.quantization_config import FineGrainedConfig

        for config_cls in (Qwen3Config, Qwen3MoeConfig):
            with self.subTest(config=config_cls.__name__):
                config = config_cls()
                shipped = dict(config.base_model_tp_plan)
                self.assertTrue(shipped, f"{config_cls.__name__} ships no plan to preserve")
                FineGrainedHfQuantizer(FineGrainedConfig()).update_tp_plan(config)
                for key, style in shipped.items():
                    self.assertEqual(config.base_model_tp_plan.get(key), style, key)

        # and a multimodal one keeps its plan on the SUB-config: the outer carries none, so
        # nothing may be invented there. 25 config classes contain "Qwen3" — a name-substring
        # branch reaches every one of them, including these.
        multimodal = Qwen3VLMoeConfig()
        FineGrainedHfQuantizer(FineGrainedConfig()).update_tp_plan(multimodal)
        self.assertFalse(getattr(multimodal, "base_model_tp_plan", None) or {})
        self.assertTrue(multimodal.text_config.base_model_tp_plan)

    def test_a_real_moe_models_experts_gain_their_companions(self):
        """The other half of the same invariant: preserving the model's entries is what lets the
        companion pass find `moe_tp_experts` and give the expert scales an entry of their own."""
        from transformers import Qwen3MoeConfig
        from transformers.quantizers.finegrained.base import FineGrainedHfQuantizer
        from transformers.utils.quantization_config import FineGrainedConfig

        config = Qwen3MoeConfig()
        FineGrainedHfQuantizer(FineGrainedConfig()).update_tp_plan(config)
        plan = config.base_model_tp_plan
        self.assertEqual(plan["layers.*.mlp.experts"], "moe_tp_experts")
        for companion in ("gate_up_proj_scale_inv", "down_proj_scale_inv", "gate_up_proj_bias"):
            self.assertIn(f"layers.*.mlp.experts.{companion}", plan, companion)


class FineGrainedGroupsTest(unittest.TestCase):
    """A quant config names a format per module SUBSET, because a checkpoint can be more than one."""

    def _config(self, **kwargs):
        from transformers.utils.quantization_config import FineGrainedConfig

        return FineGrainedConfig(**kwargs)

    def test_a_flat_config_is_one_group_over_everything(self):
        groups = self._config(quant_method="mxfp8").groups
        self.assertEqual({name: group.quant_method for name, group in groups.items()}, {"default": "mxfp8"})

    def test_a_legacy_expert_dtype_becomes_the_two_groups_it_meant(self):
        """DeepSeek-V4 ships `quant_method="fp8"` and declares its mxfp4 experts on the MODEL
        config, because a flat config cannot say "the experts differ". `shared_experts` is DENSE
        despite the name, which is why the pattern matches a path segment and not a substring."""
        from transformers.utils.quantization_config import groups_with_expert_dtype

        config = self._config(quant_method="fp8", weight_block_size=(128, 128), scale_fmt="ue8m0")
        config.groups = groups_with_expert_dtype(config.groups, "fp4")
        for module, expected in (
            ("model.layers.3.mlp.experts", "mxfp4"),
            ("model.layers.3.mlp.experts.gate_up_proj", "mxfp4"),
            ("model.layers.3.self_attn.q_proj", "fp8"),
            ("model.layers.3.mlp.shared_experts.up_proj", "fp8"),
        ):
            self.assertEqual(config.group_for(module).quant_method, expected, module)

    def test_resolution_survives_a_round_trip_through_config_json(self):
        """`to_json_string` SORTS keys, so a config read back has lost the order it was written
        in. Resolution must not depend on it: the catch-all sorts first here and would swallow
        every module if targeted groups were not preferred outright."""
        import json

        from transformers.utils.quantization_config import FineGrainedConfig, FineGrainedGroup

        config = FineGrainedConfig(
            groups={
                "experts": FineGrainedGroup(quant_method="nvfp4", targets=[r"\.experts($|\.)"]),
                "dense": FineGrainedGroup(quant_method="fp8", weight_block_size=(128, 128)),
            }
        )
        back = FineGrainedConfig.from_dict(json.loads(config.to_json_string()))
        self.assertEqual(back.group_for("model.layers.0.mlp.experts").quant_method, "nvfp4")
        self.assertEqual(back.group_for("model.layers.0.self_attn.q_proj").quant_method, "fp8")
        # a list survives JSON where a tuple does not, and the block size is compared as a tuple
        self.assertEqual(back.groups["dense"].weight_block_size, (128, 128))

    def test_two_targeted_groups_claiming_one_module_is_an_error(self):
        from transformers.utils.quantization_config import FineGrainedGroup

        config = self._config(
            groups={
                "a": FineGrainedGroup(quant_method="fp8", targets=[r"\.experts"]),
                "b": FineGrainedGroup(quant_method="nvfp4", targets=[r"mlp\."]),
            }
        )
        with self.assertRaises(ValueError):
            config.group_for("model.layers.0.mlp.experts")

    def test_a_producers_config_groups_are_normalized(self):
        """modelopt describes a format by its parameters; we name it. GLM-5.2-NVFP4 ships exactly
        this, and `targets: ["Linear"]` is its spelling of the catch-all."""
        config = self._config(
            quant_method="modelopt",
            quant_algo="NVFP4",
            config_groups={
                "group_0": {
                    "weights": {"num_bits": 4, "type": "float", "group_size": 16},
                    "input_activations": {"num_bits": 4, "type": "float", "group_size": 16, "dynamic": False},
                    "targets": ["Linear"],
                }
            },
        )
        group = config.group_for("model.layers.0.self_attn.q_proj")
        self.assertEqual((group.quant_method, group.activation_format), ("nvfp4", "nvfp4"))
        self.assertEqual(group.activation_scheme, "static")  # `dynamic: False`

    def test_a_format_we_do_not_serve_falls_back_rather_than_guessing(self):
        """Half a translation would quantize modules by a rule the producer did not write."""
        config = self._config(
            quant_method="modelopt",
            quant_algo="NVFP4",
            config_groups={
                "group_0": {"weights": {"num_bits": 3, "type": "int", "group_size": 64}, "targets": ["Linear"]}
            },
        )
        self.assertEqual(list(config.groups), ["default"])


class SubtreePatternTest(unittest.TestCase):
    def test_a_checkpoints_glob_skip_list_is_bounded(self):
        """modelopt names skipped subtrees with globs. Passed through as regexes the dots match any
        character and the star is greedy, so "model.layers.1.*" also takes layers 10-19."""
        from transformers.quantizers.quantizers_utils import should_convert_module
        from transformers.utils.quantization_config import FineGrainedConfig

        # as a modelopt checkpoint ships them, through the config that normalizes them
        skip = FineGrainedConfig(
            quant_method="modelopt",
            quant_algo="NVFP4",
            ignore=["model.layers.0*", "model.layers.1.*", "self_attn"],
        ).modules_to_not_convert
        for name in ("model.layers.0.mlp", "model.layers.1.mlp.down_proj", "model.layers.2.self_attn.q_proj"):
            self.assertFalse(should_convert_module(name, skip), name)
        for name in ("model.layers.16.mlp.down_proj", "model.layers.2.mlp.gate_proj", "model.my_self_attn.q_proj"):
            self.assertTrue(should_convert_module(name, skip), name)


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
        from transformers.integrations.finegrained.conversions import (
            FineGrainedPackedBlocks,
            FineGrainedScaleContainer,
        )

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
    """`prefers_deepgemm_linear` carries two independent gates: a correctness one (a pre-swizzled
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
            mock.patch.object(deepgemm, "is_deepgemm_loadable", return_value=True),
            mock.patch.object(deepgemm, "is_sm100", return_value=sm100),
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
        self.assertTrue(experts.holds_interleaved_gate_up)
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
        # block-FP8 has no per-group scale grid to swizzle: one scalar per (N/128, K/128) block
        _, experts = self._experts("fp8")
        self.assertEqual(experts.gate_up_proj_scale_inv.shape, (4, 2, 2))
        with mock.patch.object(fg, "_holds_swizzled_scales", return_value=False):
            _, experts = self._experts("mxfp8")
        self.assertEqual(experts.gate_up_proj_scale_inv.shape, (4, 256, 8))

    def test_megamoe_holds_gate_up_stacked(self):
        _, experts = self._experts("fp8", impl="deepgemm_megamoe")
        self.assertFalse(experts.holds_interleaved_gate_up)

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
        from transformers.integrations.finegrained.conversions import FineGrainedSwizzleScales

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
        from transformers.integrations.finegrained.conversions import FineGrainedScaleContainer

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
        from transformers.integrations.finegrained.conversions import FineGrainedSwizzleScales

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
        from transformers.utils.quantization_config import FineGrainedConfig

        return _quantizer_for(FineGrainedConfig(quant_method=quant_method)).update_weight_conversions

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
        from transformers.integrations.finegrained.conversions import (
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
        from transformers.integrations.finegrained.conversions import (
            FineGrainedInterleaveGateUp,
            FineGrainedSwizzleScales,
        )

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
        from transformers.integrations.finegrained.conversions import FineGrainedInterleaveGateUp

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
        experts.holds_interleaved_gate_up = False
        self.assertIs(op.convert({"x": stacked}, model=model)["x"], stacked)


@require_torch
class FineGrainedOnTheFlyQuantizeTest(unittest.TestCase):
    """`FineGrainedQuantize` reads the module's format: block-FP8 in torch (the group formats run the
    kernels' quantizers — covered with real kernels below), emitted in the module's layout."""

    def test_mxfp4_blocks_dequantize_like_the_integration_they_replace(self):
        """`dequantize=True` on a GPT-OSS MXFP4 checkpoint used to be the mxfp4 integration's job.
        The finegrained chain has to land the same bf16 tensor, in the (E, hidden, 2I) orientation
        the unquantized experts hold — so it is compared against that integration directly."""
        from transformers.core_model_loading import Transpose
        from transformers.integrations.finegrained.conversions import FineGrainedDequantize, FineGrainedPackedBlocks
        from transformers.integrations.mxfp4 import convert_moe_packed_tensors

        torch.manual_seed(0)
        num_experts, rows, hidden = 2, 8, 64
        blocks = torch.randint(0, 256, (num_experts, rows, hidden // 32, 16), dtype=torch.uint8)
        scales = torch.randint(120, 134, (num_experts, rows, hidden // 32), dtype=torch.uint8)
        reference = convert_moe_packed_tensors(blocks.clone(), scales.clone(), dtype=torch.bfloat16)

        sources = ["gate_up_proj_blocks$", "gate_up_proj_scales$"]
        tensors = {sources[0]: blocks.clone(), sources[1]: scales.clone()}
        for op in (FineGrainedPackedBlocks(None), FineGrainedDequantize(None), Transpose(1, 2)):
            tensors = op.convert(tensors, source_patterns=sources, target_patterns=["gate_up_proj"])
        self.assertEqual(len(tensors), 1)  # the scale is consumed, not passed down the chain
        torch.testing.assert_close(next(iter(tensors.values())).float(), reference.float(), rtol=0, atol=0)

    def test_a_static_scheme_gets_its_activation_scales_written(self):
        """A calibration-fed slot the checkpoint does not supply is still WRITTEN, as the identity.

        The loader materializes a missing key with `torch.empty_like` and `_init_weights` has no
        branch for a scale, so a slot this op leaves out reaches the kernels as uninitialized
        memory: a zero divides the activations by zero and a negative flips their sign, which
        showed up as NaN logits from one fixture and not another, run to run.
        """
        from transformers.integrations.finegrained.conversions import FineGrainedQuantize

        torch.manual_seed(0)
        cfg = _Cfg()
        cfg.hidden_size, cfg.intermediate_size, cfg.num_local_experts = 256, 128, 4
        model = torch.nn.Module()
        model.experts = fg.FineGrainedExperts(
            cfg, block_size=(128, 128), weight_format="fp8", activation_scheme="static"
        )
        model.proj = fg.FineGrainedLinear(
            in_features=256,
            out_features=256,
            block_size=(128, 128),
            weight_format="fp8",
            activation_scheme="static",
        )
        op = FineGrainedQuantize(hf_quantizer=None)
        for key, tensor, slot in (
            ("experts.gate_up_proj", torch.randn(4, 256, 256), "experts.gate_up_proj_activation_scale"),
            ("proj.weight", torch.randn(256, 256), "proj.activation_scale"),
        ):
            with self.subTest(key=key):
                out = op.convert({key: tensor}, model=model)
                self.assertIn(slot, out, f"{key}: the static activation scale was not written")
                held = model.get_parameter(slot)
                self.assertEqual((out[slot].shape, out[slot].dtype), (held.shape, held.dtype))
                torch.testing.assert_close(out[slot], torch.ones_like(out[slot]), rtol=0, atol=0)

    def test_an_expert_bias_is_passed_through_not_quantized(self):
        """A GPT-OSS expert bias is `(E, rows)` — 2-D, like a dense weight — so the rank guard
        alone lets it through. It has no scale slot, and emitting one gives the loader a
        `<proj>_bias_scale_inv` no module holds."""
        from transformers.integrations.finegrained.conversions import FineGrainedQuantize

        torch.manual_seed(0)
        cfg = _Cfg()
        cfg.hidden_size, cfg.intermediate_size, cfg.num_local_experts = 256, 128, 4
        model = torch.nn.Module()
        model.experts = fg.FineGrainedExperts(cfg, block_size=(128, 128), weight_format="fp8", has_bias=True)
        bias = torch.randn(4, 256)
        out = FineGrainedQuantize(hf_quantizer=None).convert({"experts.down_proj_bias": bias}, model=model)
        self.assertEqual(list(out), ["experts.down_proj_bias"])
        self.assertIs(out["experts.down_proj_bias"], bias)

    def test_a_partial_trailing_block_is_padded_not_refused(self):
        """DeepSeek-V3 ships `kv_a_proj_with_mqa` as `(576, 7168)` against a 128x128 block with a
        `(5, 56)` scale grid, so the format rounds the grid UP and quantizes the short block on
        its own values. Refusing the shape instead left the weight full precision."""
        from transformers.integrations.finegrained.conversions import FineGrainedQuantize

        torch.manual_seed(0)
        for rows, cols in ((576, 256), (192, 256), (256, 256)):
            with self.subTest(shape=(rows, cols)):
                weight, scale = FineGrainedQuantize._quantize_block_fp8(
                    torch.randn(rows, cols), (128, 128), ue8m0=False
                )
                self.assertEqual(weight.shape, (rows, cols))
                self.assertEqual(weight.dtype, torch.float8_e4m3fn)
                self.assertEqual(scale.shape, (-(-rows // 128), -(-cols // 128)))

    def test_block_fp8_round_trips_within_its_floor(self):
        from transformers.integrations.finegrained.conversions import FineGrainedDequantize, FineGrainedQuantize

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

    def _modelopt_conversions(self, **cfg_kwargs):
        from transformers.quantizers.finegrained.nvfp4 import FineGrainedNvfp4HfQuantizer
        from transformers.utils.quantization_config import FineGrainedConfig

        cfg = FineGrainedConfig(quant_method="modelopt", quant_algo="NVFP4", **cfg_kwargs)
        return FineGrainedNvfp4HfQuantizer(cfg).get_weight_conversions()

    def test_every_declared_target_is_a_slot_the_module_holds(self):
        """A converter target that no module holds is a load failure, and the activation global is
        exactly such a target under a weight-only run: the experts allocate one only when they
        quantize activations. Both runs are checked against a real module's parameter names, since
        the declarations are the only thing standing between the two."""
        cfg = _Cfg()
        for activation_format, holds_global in ((None, True), ("bf16", False)):
            experts = FineGrainedExperts(
                cfg, block_size=(4, 4), weight_format="nvfp4", activation_format=activation_format
            )
            slots = {name for name, _ in experts.named_parameters()} | set(experts._buffers)
            self.assertEqual(
                any("input_global_scale" in slot for slot in slots),
                holds_global,
                f"module with activation_format={activation_format!r} disagrees about the slot",
            )
            targets = {
                t.split("experts.")[-1].rstrip("$")
                for c in self._modelopt_conversions(activation_format=activation_format)
                for t in _targets(c)
            }
            for target in targets:
                self.assertIn(target, slots, f"activation_format={activation_format!r} converts onto a missing slot")

    def test_global_scale_converters_carry_an_expert_index(self):
        from transformers.core_model_loading import MergeModulelist

        # the per-expert layout only (`experts.*.`): a checkpoint that ships one already-stacked
        # tensor per layer has no per-expert sources to stamp — it is sharded like the fused
        # expert weight next to it
        globals_converters = [
            c
            for c in self._modelopt_conversions()
            if any("global_scale" in t for t in _targets(c))
            and any(re.search(p, "model.layers.0.mlp.experts.7.gate_proj.weight_scale_2") for p in c.source_patterns)
        ]
        self.assertTrue(globals_converters, "no global-scale converter found")
        for conv in globals_converters:
            self.assertTrue(
                any(isinstance(op, MergeModulelist) for op in conv.operations),
                f"{_targets(conv)} has no MergeModulelist, so expert parallelism cannot select "
                "experts and every rank would collect all of them",
            )

    def test_both_modelopt_scale_layouts_have_converters(self):
        """modelopt ships the scales either as one tensor per expert per projection (GLM-5.2) or
        as one already-stacked tensor per layer (the fused vLLM layout). Both are
        converted; the block scale takes the module's layout ops either way."""
        sources = {p for c in self._modelopt_conversions() for p in c.source_patterns}
        for suffix in ("weight_scale", "weight_scale_2", "input_scale"):
            per_expert = f"model.layers.0.mlp.experts.7.gate_proj.{suffix}"
            fused = f"model.layers.0.mlp.experts.gate_up_proj_{suffix}"
            self.assertTrue(any(re.search(p, per_expert) for p in sources), f"no per-expert converter for {suffix}")
            self.assertTrue(any(re.search(p, fused) for p in sources), f"no fused converter for {suffix}")
            # the per-expert patterns are REGEXES: unescaped dots matched `_` too, so
            # `mlp.experts.*.up_proj.weight_scale_2` also claimed the fused key and the
            # gate|up globals were flattened instead of folded to one per expert
            claimed = [p for p in sources if re.search(p, fused) and re.search(p, per_expert)]
            self.assertEqual(claimed, [], f"one pattern claims BOTH layouts for {suffix}: {claimed}")

    def test_a_checkpoint_key_is_claimed_by_exactly_one_converter(self):
        """Two converters for one key is not additive — the later one WINS and silently drops
        whatever ops the first carried. That is how a fused modelopt checkpoint lost its packed
        uint8 -> int8 view (a catch-all duplicated the fused converter) and how the per-expert
        patterns, which are REGEXES whose unescaped dots also match `_`, claimed the fused
        globals and flattened the gate|up pair instead of folding it."""
        import re

        from transformers.quantizers.finegrained.base import FineGrainedHfQuantizer
        from transformers.utils.quantization_config import FineGrainedConfig

        for quant_kwargs in (
            {"quant_method": "fp8"},
            {"quant_method": "mxfp4"},
            {"quant_method": "modelopt", "quant_algo": "NVFP4"},
        ):
            quantizer = FineGrainedHfQuantizer(FineGrainedConfig(**quant_kwargs))
            quantizer.pre_quantized = True
            converters = quantizer.update_weight_conversions([])
            for key in (
                "model.layers.3.mlp.experts.gate_up_proj",
                "model.layers.3.mlp.experts.down_proj",
                "model.layers.3.mlp.experts.gate_up_proj_weight_scale_2",
                "model.layers.3.mlp.experts.7.up_proj.weight_scale_2",
            ):
                claimed = [
                    c
                    for c in converters
                    for pattern in (c.source_patterns if isinstance(c.source_patterns, list) else [c.source_patterns])
                    if re.search(str(pattern), key)
                ]
                with self.subTest(quant=quant_kwargs["quant_method"], key=key.rsplit(".", 1)[1]):
                    self.assertLessEqual(
                        len(claimed),
                        1,
                        f"{len(claimed)} converters claim this key; the last one wins and drops "
                        f"the others' ops: {[[type(o).__name__ for o in c.operations] for c in claimed]}",
                    )

    def test_a_calibrated_checkpoint_s_input_scale_reaches_the_module_s_slot(self):
        """A static checkpoint calibrates one `input_scale` per quantized module. Nothing routed
        those keys for plain FP8 — only NVFP4 read them, as its second-level global — so the two
        shipped static models (Ministral-3 dense, Mistral-4 MoE) loaded with the registered
        default of 1.0: every activation quantized against the wrong scale, silently.

        Each key must reach the slot its module holds, in that module's shape: one value on a
        dense linear, one per expert on the stacked experts, and the gate|up pair reduced to one
        per expert rather than flattened to 2E."""
        import re

        from transformers.utils.quantization_config import FineGrainedConfig

        quantizer = _quantizer_for(FineGrainedConfig(quant_method="fp8", activation_scheme="static"))
        quantizer.pre_quantized = True
        converters = quantizer.update_weight_conversions([])

        def claim(key):
            return [
                c
                for c in converters
                for pattern in (c.source_patterns if isinstance(c.source_patterns, list) else [c.source_patterns])
                if re.search(str(pattern), key)
            ]

        for key, expected in (
            ("model.layers.3.self_attn.q_proj.input_scale", "model.layers.3.self_attn.q_proj.activation_scale"),
            ("model.layers.3.mlp.experts.7.gate_proj.input_scale", "mlp.experts.gate_up_proj_activation_scale"),
            ("model.layers.3.mlp.experts.7.up_proj.input_scale", "mlp.experts.gate_up_proj_activation_scale"),
            ("model.layers.3.mlp.experts.7.down_proj.input_scale", "mlp.experts.down_proj_activation_scale"),
            ("model.layers.3.mlp.experts.gate_up_proj_input_scale", "experts.gate_up_proj_activation_scale"),
            ("model.layers.3.mlp.experts.down_proj_input_scale", "experts.down_proj_activation_scale"),
        ):
            with self.subTest(key=key.rsplit(".", 2)[-2]):
                claimed = claim(key)
                self.assertEqual(len(claimed), 1, f"{len(claimed)} converters claim {key}")
                # a renaming's target carries a backreference; resolve it against the key
                targets = [
                    re.sub(str(claimed[0].source_patterns[0]), str(t), key) if "\\1" in str(t) else str(t)
                    for t in claimed[0]._original_target_patterns
                ]
                self.assertTrue(any(expected in t for t in targets), f"{key} -> {targets}")
        # a dynamic checkpoint has no such key to route, and a weight-only one ignores it
        dynamic = _quantizer_for(FineGrainedConfig(quant_method="fp8"))
        dynamic.pre_quantized = True
        self.assertEqual(dynamic.get_weight_conversions(), [])

    def test_the_calibrated_expert_scales_reduce_per_expert_not_per_tensor(self):
        """`FineGrainedInputScales` collapses the gate_up to ONE value for the NVFP4 global — the
        global is a split of the block scale, so an inflated one is exact. A static scale IS the
        quantization scale, with no block level to absorb it, so collapsing it would quantize
        every expert against the largest one's range. It stays per expert."""
        import torch

        from transformers.integrations.finegrained.conversions import FineGrainedInputScales

        sources = {
            "mlp.experts.*.gate_proj.input_scale": torch.tensor([1.0, 3.0]),
            "mlp.experts.*.up_proj.input_scale": torch.tensor([2.0, 1.0]),
        }
        target = "mlp.experts.gate_up_proj_activation_scale"
        scale = FineGrainedInputScales().convert(dict(sources), full_layer_name=target)[target]
        torch.testing.assert_close(scale, torch.tensor([2.0, 3.0]))  # per expert, over the pair

    def test_the_gate_up_fold_survives_the_loader_s_list_wrapping(self):
        """The loader hands a converter its tensors in a list. Unwrapped, a fused `(E, 2)` pair
        reads as `(1, 2E)`, the per-half fold does not fire, and the module gets `2E` globals
        where the kernels assert one per expert."""
        import torch

        from transformers.integrations.finegrained.conversions import FineGrainedWeightGlobals

        targets = [
            "experts.gate_up_proj_weight_global_scale",
            "experts.down_proj_weight_global_scale",
            "experts.down_proj_input_global_scale",
        ]
        experts = 8
        for wrap in (False, True):
            pair = torch.rand(experts, 2) + 1
            src = {
                "experts.gate_up_proj_weight_scale_2": [pair] if wrap else pair,
                "experts.down_proj_weight_scale_2": torch.rand(experts) + 1,
                "experts.down_proj_input_scale": torch.rand(experts) + 1,
            }
            out = FineGrainedWeightGlobals(None).convert(src, target_patterns=targets, model=None)
            with self.subTest(list_wrapped=wrap):
                self.assertEqual(
                    out["experts.gate_up_proj_weight_global_scale"].shape,
                    torch.Size([experts]),
                    "the gate|up halves were not folded to one global per expert",
                )

    def test_one_converter_merges_a_layer_s_globals(self):
        """modelopt calibrates the gate|up halves separately. One converter owns every global of
        a layer, because merging them to the one-per-expert the kernels take moves the up half's
        onto the down projection: SwiGLU is linear in the up half, so the stack keeps the gate's
        global and the down's weight global scales the expert output back. The down's calibrated
        input scale moves the other way, keeping the requantized intermediate on the range the
        checkpoint calibrated."""
        from transformers.integrations.finegrained.conversions import FineGrainedWeightGlobals

        gate, up = torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])
        down, down_input = torch.tensor([5.0, 6.0]), torch.tensor([0.5, 2.0])
        targets = [
            "mlp.experts.gate_up_proj_weight_global_scale",
            "mlp.experts.down_proj_weight_global_scale",
            "mlp.experts.down_proj_input_global_scale",
        ]
        out = FineGrainedWeightGlobals().convert(
            {
                "mlp.experts.*.up_proj.weight_scale_2": up,
                "mlp.experts.*.gate_proj.weight_scale_2": gate,
                "mlp.experts.*.down_proj.weight_scale_2": down,
                "mlp.experts.*.down_proj.input_scale": down_input,
            },
            target_patterns=targets,
        )
        ratio = up / gate
        torch.testing.assert_close(out[targets[0]], gate)
        torch.testing.assert_close(out[targets[1]], down * ratio)
        torch.testing.assert_close(out[targets[2]], down_input / ratio)

    def test_globals_with_one_calibrated_projection_pass_through(self):
        """A checkpoint that calibrates the stack as one matrix (its halves agree, or it ships a
        single global) has nothing to merge, so every global reaches its module unchanged."""
        from transformers.integrations.finegrained.conversions import FineGrainedWeightGlobals

        targets = [
            "experts.gate_up_proj_weight_global_scale",
            "experts.down_proj_weight_global_scale",
            "experts.down_proj_input_global_scale",
        ]
        out = FineGrainedWeightGlobals().convert(
            {
                "experts.gate_up_proj_weight_scale_2": torch.tensor([[1.0], [2.0]]),
                "experts.down_proj_weight_scale_2": torch.tensor([3.0, 4.0]),
                "experts.down_proj_input_scale": torch.tensor([0.5, 0.25]),
            },
            target_patterns=targets,
        )
        torch.testing.assert_close(out[targets[0]], torch.tensor([1.0, 2.0]))
        torch.testing.assert_close(out[targets[1]], torch.tensor([3.0, 4.0]))
        torch.testing.assert_close(out[targets[2]], torch.tensor([0.5, 0.25]))

    def test_the_fused_layout_ships_both_halves_in_one_tensor(self):
        """A stacked `(E, 2)` `weight_scale_2` is the same pair as the per-expert layout's two
        keys, and merges the same way."""
        from transformers.integrations.finegrained.conversions import FineGrainedWeightGlobals

        targets = [
            "experts.gate_up_proj_weight_global_scale",
            "experts.down_proj_weight_global_scale",
        ]
        out = FineGrainedWeightGlobals().convert(
            {
                "experts.gate_up_proj_weight_scale_2": torch.tensor([[1.0, 3.0], [2.0, 4.0]]),
                "experts.down_proj_weight_scale_2": torch.tensor([5.0, 6.0]),
            },
            target_patterns=targets,
        )
        torch.testing.assert_close(out[targets[0]], torch.tensor([1.0, 2.0]))
        torch.testing.assert_close(out[targets[1]], torch.tensor([5.0 * 3.0, 6.0 * 2.0]))

    def test_activation_global_is_one_value_for_the_gate_up_and_per_expert_for_the_down(self):
        """The gate_up quantizes the hidden states once, BEFORE routing, so its calibrated
        `input_scale` reduces to one value; the down's rows are per expert, so its stays per
        expert — the requant epilogue normalizes each row by its own expert's value."""
        from transformers.integrations.finegrained.conversions import FineGrainedInputScales

        up = FineGrainedInputScales().convert(
            {
                "mlp.experts.*.gate_proj.input_scale": torch.tensor([1.0, 3.0]),
                "mlp.experts.*.up_proj.input_scale": torch.tensor([2.0, 1.0]),
            },
            full_layer_name="mlp.experts.gate_up_proj_input_global_scale",
        )["mlp.experts.gate_up_proj_input_global_scale"]
        torch.testing.assert_close(up, torch.tensor([3.0]))
        down = FineGrainedInputScales().convert(
            {"mlp.experts.*.down_proj.input_scale": torch.tensor([1.0, 4.0])},
            full_layer_name="mlp.experts.down_proj_input_global_scale",
        )["mlp.experts.down_proj_input_global_scale"]
        torch.testing.assert_close(down, torch.tensor([1.0, 4.0]))


class _InputScaledRMSNorm(torch.nn.Module):
    """A post-expert norm that scales by ``1 + weight`` BEFORE normalizing, so the row's
    mean square is taken on the scaled values (not the same function as normalizing first)."""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.zeros(dim, dtype=torch.bfloat16))

    def forward(self, x):
        out = x.float() * (1.0 + self.weight.float())
        return (out * torch.rsqrt(out.pow(2).mean(-1, keepdim=True) + self.eps)).type_as(x)


class _UnfusableNorm(torch.nn.Module):
    """A norm whose math is none of the fused forms (a bias the kernels have no slot for): the
    binder has to leave it a module call rather than pick the nearest form."""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(dim, dtype=torch.bfloat16))
        self.bias = torch.nn.Parameter(torch.full((dim,), 0.5, dtype=torch.bfloat16))

    def forward(self, x):
        xf = x.float()
        normed = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return (normed * self.weight.float() + self.bias.float()).type_as(x)


def _nvfp4_two_level(kernel, weight):
    """Canonical two-level NVFP4 quant of one matrix: the global is ``amax / (6 * 448)`` — the
    smallest one that keeps every e4m3 block scale in range — and the block quant runs on the
    normalized values. Returns ``(packed_e2m1, e4m3 block scales, fp32 global)``."""
    global_scale = (weight.abs().amax() / (6.0 * 448.0)).clamp(min=torch.finfo(torch.float32).tiny)
    packed, block = kernel.nvfp4_act_quant((weight.float() / global_scale).contiguous())
    return packed.view(torch.int8), block, global_scale


def _unpack_e2m1(packed):
    """Packed-E2M1 bytes (two codes per byte along the last axis) back to fp32 values."""
    lut = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
        device=packed.device,
    )
    b = packed.view(torch.uint8)
    pairs = torch.stack([lut[(b & 0x0F).long()], lut[(b >> 4).long()]], dim=-1)
    return pairs.reshape(*packed.shape[:-1], packed.shape[-1] * 2)


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
        if ue8m0:  # power-of-two scales: the tcgen05 dot_scaled format
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
        from transformers.integrations.finegrained.conversions import FineGrainedDequantize, FineGrainedQuantize

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
                    global_scale = out["experts.gate_up_proj_weight_global_scale"]
                    deq = deq * global_scale.reshape(-1, 1, 1)
                else:
                    self.assertNotIn("experts.gate_up_proj_weight_global_scale", out)
                rel = ((deq - original.float()).norm() / original.float().norm()).item()
                self.assertLess(rel, floor, f"{fmt}: {rel:.3f}")
                self.assertGreater(rel, 0.0)

    def _nvfp4_experts_modelopt_layout(self, cfg, activation_format=None, magnitudes=(8.0, 1.0)):
        """NVFP4 experts in the layout a modelopt checkpoint delivers: every (expert, half) of the
        gate|up stack quantized against its OWN global (``weight_scale_2`` is per projection, and
        the two can differ per expert), the down per expert, plus calibrated activation
        globals. Returns the module, the exact dequantized weights (so the reference is the
        quantization floor rather than the pre-quant weight), and the dequantization a reader that
        applied ONE of the two globals to both halves would get."""
        from transformers.integrations.finegrained.conversions import FineGrainedWeightGlobals

        kernel = load_finegrained_kernel()
        experts = FineGrainedExperts(cfg, weight_format="nvfp4", activation_format=activation_format).cuda()
        dequantized, single_global, checkpoint_globals = {}, {}, {}
        for proj, rows, cols in (
            ("gate_up_proj", 2 * cfg.intermediate_size, cfg.hidden_size),
            ("down_proj", cfg.hidden_size, cfg.intermediate_size),
        ):
            reference = torch.randn(cfg.num_local_experts, rows, cols, device="cuda")
            if proj == "gate_up_proj":  # the halves interleave by row; draw them far apart
                reference[:, 0::2] *= magnitudes[0]
                reference[:, 1::2] *= magnitudes[1]
            packed = torch.empty(cfg.num_local_experts, rows, cols // 2, dtype=torch.int8, device="cuda")
            block = torch.empty(cfg.num_local_experts, rows, cols // 16, dtype=torch.float8_e4m3fn, device="cuda")
            halves = 2 if proj == "gate_up_proj" else 1
            globals_ = torch.empty(cfg.num_local_experts, halves, device="cuda")
            deq, deq_one = torch.empty_like(reference), torch.empty_like(reference)
            for e in range(cfg.num_local_experts):
                for h in range(halves):
                    rows_h = slice(h, None, halves)
                    q, b, g = _nvfp4_two_level(kernel, reference[e, rows_h])
                    packed[e, rows_h], block[e, rows_h], globals_[e, h] = q, b, g
                    values = _unpack_e2m1(q) * b.float().repeat_interleave(16, dim=-1)
                    deq[e, rows_h] = values * g
                if halves == 2:
                    # the control: BOTH halves read the up half's global, which is what a kernel
                    # that ignored the layout does. It has to be the up one — putting the gate's
                    # global on both scales the expert's whole output uniformly, and the
                    # post-expert RMSNorm divides exactly that out; a wrong GATE scale goes
                    # through the SiLU and survives.
                    for h in (0, 1):
                        rows_h = slice(h, None, 2)
                        deq_one[e, rows_h] = deq[e, rows_h] * (globals_[e, 1] / globals_[e, h])
            scale = block if getattr(experts, f"{proj}_scale_inv").dim() != 5 else kernel.swizzle_mx_scales(block)
            setattr(experts, proj, torch.nn.Parameter(packed, requires_grad=False))
            setattr(experts, f"{proj}_scale_inv", torch.nn.Parameter(scale, requires_grad=False))
            checkpoint_globals[f"experts.{proj}_weight_scale_2"] = globals_
            dequantized[proj], single_global[proj] = deq, deq_one
        # calibrated input_scale: one value for the gate_up (quantized before routing), one per
        # expert for the down (its rows are per expert). A weight-only module holds neither —
        # nothing there quantizes activations — so the checkpoint's keys stay unused.
        if activation_format != "bf16":
            experts.gate_up_proj_input_global_scale = torch.nn.Parameter(
                torch.full((1,), 0.05, device="cuda"), requires_grad=False
            )
            checkpoint_globals["experts.down_proj_input_scale"] = torch.linspace(
                0.02, 0.08, cfg.num_local_experts, device="cuda"
            )
        single_global["down_proj"] = dequantized["down_proj"]  # the down has one global either way
        # The globals go in through the loader's own converter, so the module holds exactly what a
        # real checkpoint puts there: one per expert, with the gate|up pair merged onto the down.
        targets = [
            "experts.gate_up_proj_weight_global_scale",
            "experts.down_proj_weight_global_scale",
            "experts.down_proj_input_global_scale",
        ]
        for target, value in FineGrainedWeightGlobals().convert(checkpoint_globals, target_patterns=targets).items():
            setattr(experts, target.split(".")[-1], torch.nn.Parameter(value, requires_grad=False))
        return experts, dequantized, single_global

    def _moe_reference(self, dequantized, x, idx, wts, post_norm=None):
        """The experts chain in torch over the dequantized weights: gate|up (interleaved columns),
        SwiGLU, down, the optional per-expert output norm, then the routing-weighted sum."""
        out = torch.zeros_like(x, dtype=torch.float32)
        for token in range(x.shape[0]):
            for slot in range(idx.shape[1]):
                e = int(idx[token, slot])
                pre = x[token].float() @ dequantized["gate_up_proj"][e].t()
                inter = torch.nn.functional.silu(pre[0::2]) * pre[1::2]
                row = inter @ dequantized["down_proj"][e].t()
                if post_norm is not None:
                    row = post_norm(row.to(x.dtype)).float()
                out[token] += row * float(wts[token, slot])
        return out

    def test_modelopt_nvfp4_experts_merge_their_globals_and_fuse_the_post_norm(self):
        """The stacked checkpoint shape end to end through the integration: a gate|up stack
        calibrated per half (merged at load), calibrated activation globals, and a per-expert
        output norm the binder fuses, on all three forwards. Weight-only pins the numbers against the dequantized weights (nothing else
        rounds); the W4A4 chain rides the 4-bit activation floor, so it is pinned by being far
        closer to the right globals than to one global folded over both halves — what a kernel
        that ignored the layout computes."""
        torch.manual_seed(0)
        cfg = _Cfg()
        cfg.hidden_size, cfg.intermediate_size, cfg.num_local_experts = 512, 256, 4
        cfg._experts_implementation = "grouped_mm"
        x = torch.randn(8, cfg.hidden_size, device="cuda", dtype=torch.bfloat16) * 0.1
        idx = torch.randint(0, cfg.num_local_experts, (8, 2), device="cuda", dtype=torch.long)
        wts = torch.rand(8, 2, device="cuda", dtype=torch.bfloat16)
        forwards = (
            ("grouped", fg.finegrained_grouped_mm_experts_forward),
            ("batched", fg.finegrained_batched_mm_experts_forward),
            ("eager", lambda module, *args: module(*args)),
        )

        def relative(out, reference):
            return ((out.float() - reference).norm() / reference.norm()).item()

        for activation_format, floor in (("bf16", 0.02), (None, 0.35)):
            experts, dequantized, one_global = self._nvfp4_experts_modelopt_layout(cfg, activation_format)
            reference = self._moe_reference(dequantized, x, idx, wts)
            wrong = self._moe_reference(one_global, x, idx, wts)
            for name, forward in forwards:
                with self.subTest(forward=name, activation_format=activation_format):
                    out = forward(experts, x, idx, wts)
                    rel, rel_wrong = relative(out, reference), relative(out, wrong)
                    self.assertLess(rel, floor, f"{name} diverged from the dequantized reference: {rel:.3f}")
                    self.assertLess(
                        3 * rel,
                        rel_wrong,
                        f"{name} is as close to ONE global over both gate|up halves ({rel_wrong:.3f}) "
                        f"as to the per-half ones ({rel:.3f}) — the fold is not reading the halves",
                    )

            # the same chain with the model's per-expert output norm, which runs on the routed
            # rows: every forward has to apply it where the reference forwards do. A name the
            # kernels implement is folded into the chain's reduce and anything else is the module
            # itself, so both have to land on the same numbers as the reference.
            for norm, norm_name in (
                (torch.nn.RMSNorm(cfg.hidden_size, device="cuda", dtype=torch.bfloat16), "rms_norm"),
                (_InputScaledRMSNorm(cfg.hidden_size).cuda(), "input_scaled_rms_norm"),
                # a form the kernels do not implement: the module is called on the rows
                (_UnfusableNorm(cfg.hidden_size).cuda(), "a_models_own_norm"),
            ):
                norm.weight.data.uniform_(0.5, 1.5)
                experts.post_expert_norm, experts.post_expert_norm_name = norm, norm_name
                experts.has_post_expert_norm = True
                normed = self._moe_reference(dequantized, x, idx, wts, norm)
                self.assertGreater(relative(normed, reference), 0.05, "the norm has to change the output")
                for name, forward in forwards:
                    with self.subTest(forward=name, activation_format=activation_format, norm=norm_name):
                        rel = relative(forward(experts, x, idx, wts), normed)
                        self.assertLess(rel, floor, f"{name} diverged with the post-expert norm: {rel:.3f}")

    def test_experts_scale_layout_op_keeps_the_forward_correct(self):
        """End-to-end over the integration: real MXFP8 experts hold swizzled scales, the loader's op
        fills them from the affine grid, the fused forward matches the affine module's, and the
        reverse op hands the affine grid back bitwise."""
        from transformers.integrations.finegrained.conversions import FineGrainedSwizzleScales

        cfg = _Cfg()
        cfg.hidden_size, cfg.intermediate_size, cfg.num_local_experts = (
            512,
            256,
            4,
        )  # K >= 256: the swizzled arm has tiles
        cfg._experts_implementation = "grouped_mm"
        experts = FineGrainedExperts(cfg, weight_format="mxfp8").cuda()
        with mock.patch.object(fg, "_holds_swizzled_scales", return_value=False):
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


def _checkpoint_shapes(path):
    """`{key: shape}` of a saved checkpoint, for comparing one save against another."""
    import glob

    from safetensors import safe_open

    shapes = {}
    for shard in sorted(glob.glob(os.path.join(path, "*.safetensors"))):
        with safe_open(shard, framework="pt") as handle:
            for key in handle.keys():
                shapes[key] = tuple(handle.get_slice(key).get_shape())
    return shapes


_SHARDING_WORKER = """
import importlib, os, sys, torch
from transformers.distributed import DistributedConfig

model_dir, out_dir, modes, cls_path = sys.argv[1], sys.argv[2], sys.argv[3].split(","), sys.argv[4]
module_name, cls_name = cls_path.split(":")
model_cls = getattr(importlib.import_module(module_name), cls_name)
world = int(os.environ["WORLD_SIZE"])
ids = torch.arange(16, dtype=torch.long).unsqueeze(0)

# both modes in ONE launch: they share the mesh, so the process group is created once and only
# the plan differs -- a second torchrun would pay another interpreter + CUDA start
for mode in modes:
    model = model_cls.from_pretrained(
        model_dir,
        dtype="auto",
        attn_implementation="eager",
        distributed_config=DistributedConfig(tp_size=world, enable_expert_parallel=(mode == "ep")),
    ).eval()
    experts = next(m for n, m in model.named_modules() if n.endswith("mlp.experts"))
    weight = getattr(experts, "gate_up_proj", None)
    if weight is None:
        weight = experts.up_proj
    local = weight.to_local() if hasattr(weight, "to_local") else weight
    with torch.no_grad():
        logits = model(ids.to(model.device)).logits.float().cpu()
    if int(os.environ["RANK"]) == 0:
        torch.save({"logits": logits, "expert_local": tuple(local.shape)},
                   os.path.join(out_dir, mode + ".pt"))
    del model
    torch.cuda.empty_cache()
"""


def _checkpoint_expert_shape(model_dir):
    """The UNSHARDED `(experts, rows)` of the stacked gate|up projection the module holds.

    Read from the checkpoint, which ships the experts either already fused
    (`experts.gate_up_proj`) or one per expert (`experts.0.gate_proj.weight`, or DeepSeek's `w1`/`w3`,
    whose rows are gate and up separately). The baseline a sharded leg is measured against has to come from outside
    the sharded runs: taking it from whichever leg ran first compares one mode to another, and a
    mode that placed nothing then still looks sharded.
    """
    import glob
    import json
    import os

    from safetensors import safe_open

    shards = sorted(glob.glob(os.path.join(model_dir, "*.safetensors")))
    index = os.path.join(model_dir, "model.safetensors.index.json")
    if os.path.exists(index):
        with open(index, encoding="utf-8") as fh:
            weight_map = json.load(fh)["weight_map"]
        shards = sorted({os.path.join(model_dir, f) for f in weight_map.values()})

    experts: set[int] = set()
    rows = 0
    for shard in shards:
        with safe_open(shard, framework="pt") as handle:
            for key in handle.keys():
                shape = None
                if re.search(r"\.experts\.gate_up_proj$", key):
                    shape = handle.get_slice(key).get_shape()
                    if len(shape) == 3:
                        return (shape[0], shape[1])
                if re.search(r"\.experts\.(\d+)\.(gate_proj|up_proj|w1|w3)\.weight$", key):
                    experts.add(int(key.rsplit(".experts.", 1)[1].split(".", 1)[0]))
                    rows = max(rows, handle.get_slice(key).get_shape()[0])
    if not experts or not rows:
        raise AssertionError(f"no expert projection in {model_dir} to size the shard against")
    return (len(experts), 2 * rows)  # the module stacks gate and up into one row extent


@require_torch_multi_accelerator
class FineGrainedLoadPathEquivalenceTest(TestCasePlus):
    """Expert parallelism and intra-expert tensor parallelism must both reproduce what
    `device_map` gives, which is the baseline because it places modules across devices without
    splitting a single tensor — no process group, no DTensor, no collectives. So it computes the
    unsharded answer, on the multi-GPU path people actually deploy.

    Each model is built in the format it actually ships, because the conversion path differs by
    format and by checkpoint layout — a per-expert FP8 checkpoint and a fused NVFP4 one reach the
    experts through different converters, and bugs have hidden in exactly that gap.

    Three properties make this able to fail, all of which earlier versions lacked:
      * the fixture is PRE-QUANTIZED. Quantizing on the fly derives each rank's scales from the
        shard it already holds, so they come out correctly sized whatever the plan says and a
        plan that shards no scale at all still produces the right answer.
      * the model's plan must carry expert entries. A model whose `base_model_tp_plan` is empty
        (GPT-OSS, DeepSeek-V4) shards nothing under TP, so that leg is skipped EXPLICITLY rather
        than passing vacuously.
      * every leg reports the local expert shard, so a leg that placed nothing fails loudly.
    """

    @staticmethod
    def _model_table():
        """`{label: (config_cls, model_cls, config_kwargs, quantization_config)}` — each model
        in the format it actually ships, with the quantization config that format arrives under,
        not just its name: block-FP8 carries a `weight_block_size`, and an NVFP4 checkpoint comes
        from modelopt under `quant_algo`, which is remapped on construction. Dims are multiples
        of the 128 block so a 2-way split stays block-aligned, and the expert count divides the
        mesh."""
        from transformers import (
            DeepseekV3Config,
            DeepseekV3ForCausalLM,
            DeepseekV4Config,
            DeepseekV4ForCausalLM,
            Glm4vMoeConfig,
            Glm4vMoeForConditionalGeneration,
            Glm4vMoeTextConfig,
            Glm4vMoeVisionConfig,
            GptOssConfig,
            GptOssForCausalLM,
            MiniMaxM3SparseForConditionalGeneration,
            MiniMaxM3VLConfig,
            Mistral4Config,
            Mistral4ForCausalLM,
        )
        from transformers.utils.quantization_config import FineGrainedConfig

        return {
            "deepseek_v3-fp8": (
                DeepseekV3Config,
                DeepseekV3ForCausalLM,
                {
                    "vocab_size": 64,
                    "hidden_size": 256,
                    "intermediate_size": 256,
                    "moe_intermediate_size": 256,
                    "num_hidden_layers": 1,
                    "num_attention_heads": 4,
                    "num_key_value_heads": 4,
                    "n_routed_experts": 4,
                    "num_experts_per_tok": 2,
                    "n_shared_experts": 1,
                    "n_group": 1,
                    "topk_group": 1,
                    "first_k_dense_replace": 0,
                    "max_position_embeddings": 32,
                    "q_lora_rank": None,
                    "kv_lora_rank": 32,
                    "qk_nope_head_dim": 32,
                    "qk_rope_head_dim": 16,
                    "v_head_dim": 32,
                },
                # DeepSeek-V3 ships block-FP8: 128x128 weight blocks, activations quantized
                # per token at run time
                FineGrainedConfig(quant_method="fp8", weight_block_size=(128, 128)),
            ),
            # the only shipped STATIC scheme: per-TENSOR weights (no `weight_block_size`) and a
            # calibrated activation scale per quantized module, which for a MoE is one per expert.
            # Its conversion script asserts `qscheme_act == "TENSOR"`; Ministral-3 is the dense
            # counterpart of the same export.
            "mistral4-fp8_tensor_static": (
                Mistral4Config,
                Mistral4ForCausalLM,
                {
                    "vocab_size": 64,
                    "hidden_size": 256,
                    "intermediate_size": 256,
                    "moe_intermediate_size": 256,
                    "num_hidden_layers": 1,
                    "num_attention_heads": 4,
                    "num_key_value_heads": 4,
                    "n_routed_experts": 4,
                    "num_experts_per_tok": 2,
                    "first_k_dense_replace": 0,
                    "max_position_embeddings": 32,
                },
                FineGrainedConfig(quant_method="fp8", weight_block_size=None, activation_scheme="static"),
            ),
            # MULTIMODAL: the experts' plans live on `text_config`, not on the config the
            # quantizer is handed — whose `base_model_ep_plan` is None. Reading only the outer
            # one adds no companion while the weights still shard, which is how a real
            # multimodal MoE ends up with whole scales against sharded weights.
            "glm4v_moe-nvfp4": (
                Glm4vMoeConfig,
                Glm4vMoeForConditionalGeneration,
                {
                    "text_config": Glm4vMoeTextConfig(
                        vocab_size=64,
                        hidden_size=256,
                        intermediate_size=256,
                        moe_intermediate_size=256,
                        num_hidden_layers=2,
                        num_attention_heads=4,
                        num_key_value_heads=2,
                        max_position_embeddings=32,
                        rope_parameters={"type": "default", "mrope_section": [16, 8, 8], "partial_rotary_factor": 1.0},
                        rope_theta=10000,
                        tie_word_embeddings=True,
                        bos_token_id=0,
                        eos_token_id=0,
                        pad_token_id=0,
                        n_routed_experts=4,
                        n_shared_experts=1,
                        n_group=1,
                        topk_group=1,
                        num_experts_per_tok=2,
                        first_k_dense_replace=0,
                    ),
                    "vision_config": Glm4vMoeVisionConfig(
                        depth=2,
                        num_heads=4,
                        hidden_size=64,
                        out_hidden_size=256,
                        intermediate_size=64,
                        patch_size=14,
                        spatial_merge_size=1,
                        temporal_patch_size=2,
                    ),
                },
                # the GLM NVFP4 checkpoints are modelopt exports: `quant_algo` names the format
                # and `FineGrainedConfig` remaps it to nvfp4 at construction
                FineGrainedConfig(quant_method="modelopt", quant_algo="NVFP4"),
            ),
            # interleaved rows (`is_concatenated=False`), transposed, with expert biases — and
            # an EP plan that already names those biases, so the companion rules meet entries
            # the model wrote itself
            "gpt_oss-mxfp4": (
                GptOssConfig,
                GptOssForCausalLM,
                {
                    "vocab_size": 64,
                    "hidden_size": 256,
                    "intermediate_size": 256,
                    "num_hidden_layers": 1,
                    "num_attention_heads": 4,
                    "num_key_value_heads": 4,
                    "num_local_experts": 4,
                    "num_experts_per_tok": 2,
                    "max_position_embeddings": 32,
                },
                # GPT-OSS ships weight-only: raw bf16 activations against packed fp4 weights
                FineGrainedConfig(quant_method="mxfp4", activation_format="bf16"),
            ),
            # MIXED precision, and the only entry whose expert format is not the quantization
            # config's: `expert_dtype` is a model-config side-channel that makes the EXPERTS
            # mxfp4 while the dense and attention paths stay block-FP8 — with scales in UE8M0
            # containers rather than fp32, the other `scale_fmt`
            "deepseek_v4-fp4_experts+fp8_dense": (
                DeepseekV4Config,
                DeepseekV4ForCausalLM,
                {
                    "vocab_size": 64,
                    "hidden_size": 256,
                    "intermediate_size": 256,
                    "moe_intermediate_size": 256,
                    "num_hidden_layers": 1,
                    "num_attention_heads": 4,
                    "num_key_value_heads": 4,
                    "n_routed_experts": 4,
                    "num_experts_per_tok": 2,
                    "n_shared_experts": 1,
                    "first_k_dense_replace": 0,
                    "max_position_embeddings": 32,
                    "expert_dtype": "fp4",
                },
                FineGrainedConfig(quant_method="fp8", weight_block_size=(128, 128), scale_fmt="ue8m0"),
            ),
            # a second MULTIMODAL nesting, in the group-32 MX format. The sub-config shape
            # follows this model's own tester; only the MoE dims are raised to a multiple of
            # the 128 block so a 2-way split stays block-aligned.
            "minimax_m3_vl-mxfp8": (
                MiniMaxM3VLConfig,
                MiniMaxM3SparseForConditionalGeneration,
                {
                    "text_config": {
                        "hidden_size": 256,
                        # 512 so the 2-way split leaves 256: the experts' down projection
                        # contracts over this, and a sharded 128 has no 128-wide swizzled tile
                        "intermediate_size": 512,
                        "dense_intermediate_size": 256,
                        "shared_intermediate_size": 256,
                        "num_hidden_layers": 2,
                        "num_attention_heads": 4,
                        "num_key_value_heads": 4,
                        "head_dim": 64,
                        "rotary_dim": 32,
                        "vocab_size": 64,
                        "max_position_embeddings": 32,
                        "bos_token_id": 0,
                        "eos_token_id": 1,
                        "pad_token_id": 2,
                        "num_local_experts": 4,
                        "num_experts_per_tok": 2,
                        "n_shared_experts": 1,
                        "moe_layer_freq": [0, 1],
                        "layer_types": ["full_attention", "minimax_m3_sparse"],
                        "tie_word_embeddings": False,
                        "index_n_heads": 2,
                        "index_head_dim": 16,
                        "index_block_size": 8,
                        "index_topk_blocks": 4,
                        "index_local_blocks": 1,
                    },
                    "vision_config": {
                        # 256 so the 2-way SPLIT is still 128-aligned: an MXFP8 weight with
                        # pre-swizzled scales is read in 128-wide K tiles, and a sharded 128 dim
                        # leaves 64 — which has none to offer, and the tuner has no config at all
                        "hidden_size": 256,
                        "intermediate_size": 256,
                        "num_hidden_layers": 2,
                        "num_attention_heads": 4,
                        "num_channels": 3,
                        "image_size": 14,
                        "patch_size": 14,
                        "temporal_patch_size": 2,
                        "spatial_merge_size": 1,
                    },
                    "image_token_index": 4,
                    "video_token_index": 5,
                    "projector_hidden_size": 256,
                    "pad_token_id": 2,
                },
                FineGrainedConfig(quant_method="mxfp8"),
            ),
        }

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        # the kernels autotune per shape from a cold cache, which dwarfs everything else here
        cls._env = {
            "FINEGRAINED_AUTOTUNE_TRIALS": "1",
            "TRITON_CACHE_DIR": os.path.join(cls._tmp.name, "triton"),
        }
        os.environ.update(cls._env)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    @staticmethod
    def _free_port() -> int:
        with socket.socket() as sock:
            sock.bind(("", 0))
            return sock.getsockname()[1]

    def _fixture(self, label):
        """A PRE-QUANTIZED checkpoint of one tiny model, in the format that model ships."""
        import torch

        config_cls, model_cls, kwargs, quantization_config = self._model_table()[label]
        config = config_cls(**kwargs)

        bf16_dir = os.path.join(self._tmp.name, label, "bf16")
        quant_dir = os.path.join(self._tmp.name, label, "quantized")
        torch.manual_seed(0)
        # BF16 is what every model here ships, and the dtype decides which kernel arms run: a
        # float32 checkpoint (torch's default for a freshly built model, which `dtype="auto"`
        # then faithfully reloads) sends the weight-only formats down arms `tl.dot_scaled`
        # cannot serve at all, so the suite would exercise a dtype nobody deploys and leave the
        # real one uncovered. Saving in bf16 makes `"auto"` mean bf16 for every load below.
        model_cls(config).to(torch.bfloat16).save_pretrained(bf16_dir, safe_serialization=True)
        quantized = model_cls.from_pretrained(
            bf16_dir,
            dtype="auto",
            attn_implementation="eager",
            quantization_config=quantization_config,
            device_map="cuda:0",
        )
        quantized.save_pretrained(quant_dir, safe_serialization=True)
        del quantized
        torch.cuda.empty_cache()

        # SAVE must restore the checkpoint's own layout: the layout ops (gate|up interleave,
        # scale container, swizzle) each have a reverse, and a quantized model reloaded and
        # written again has to land on the same keys and shapes. A broken reverse writes a
        # corrupt checkpoint silently — the module still reads back whatever it wrote.
        reloaded = model_cls.from_pretrained(quant_dir, dtype="auto", device_map="cuda:0")
        round_trip = os.path.join(self._tmp.name, label, "round_trip")
        reloaded.save_pretrained(round_trip, safe_serialization=True)
        del reloaded
        torch.cuda.empty_cache()
        self.assertEqual(
            _checkpoint_shapes(quant_dir), _checkpoint_shapes(round_trip), f"{label}: save did not round-trip"
        )
        return quant_dir, config, model_cls

    def _logits(self, model_dir, model_cls, **load_kwargs):
        import torch

        model = model_cls.from_pretrained(model_dir, dtype="auto", attn_implementation="eager", **load_kwargs).eval()
        ids = torch.arange(16, dtype=torch.long, device=model.device).unsqueeze(0)
        with torch.no_grad():
            logits = model(ids).logits.float().cpu()
        del model
        torch.cuda.empty_cache()
        return logits

    def _rounding_floor(self, model_dir, model_cls, reference):
        """How far this model's logits move when every sharded block is perturbed by BF16 rounding.

        Sharding cannot be bit-exact: a rowwise all-reduce sums the same terms in a different
        order, so the first sharded op differs by an ULP. How far that travels is a property of
        the MODEL, not of the sharding — a chain of MoE layers can amplify it a hundredfold while
        a dense stack barely moves. Injecting the same magnitude on ONE device measures that
        amplification directly, giving each model a budget its own conditioning earns.
        """
        import torch

        model = model_cls.from_pretrained(
            model_dir, dtype="auto", attn_implementation="eager", device_map="cuda:0"
        ).eval()
        # Perturb EVERY block a sharded reduction passes through, not one: TP re-orders the sum
        # in each of them, so the rounding accumulates down the stack instead of cancelling.
        # Perturbing a single site with random noise measured LESS movement than sharding caused
        # even at 14x the magnitude, which is what accumulation looks like from the wrong model.
        blocks = [
            m
            for n, m in model.named_modules()
            if n.endswith((".self_attn", ".mlp", ".block_sparse_moe")) and n.count(".layers.") == 1
        ]
        generator = torch.Generator(device=model.device).manual_seed(0)

        def perturb(module, args, output):
            tensor = output[0] if isinstance(output, tuple) else output
            if not torch.is_tensor(tensor):
                return output
            ulp = torch.finfo(tensor.dtype).eps * tensor.abs().max()
            noise = torch.randn(tensor.shape, generator=generator, device=tensor.device, dtype=tensor.dtype) * ulp
            tensor = tensor + noise
            return (tensor,) + output[1:] if isinstance(output, tuple) else tensor

        handles = [b.register_forward_hook(perturb) for b in blocks]
        ids = torch.arange(16, dtype=torch.long, device=model.device).unsqueeze(0)
        with torch.no_grad():
            perturbed = model(ids).logits.float().cpu()
        for handle in handles:
            handle.remove()
        del model
        torch.cuda.empty_cache()
        return (perturbed - reference).abs().max().item()

    def _sharded(self, model_dir, model_cls, modes):
        """`{mode: payload}` from one 2-rank `torchrun`, through the real EP / TP load path."""
        import torch

        script = os.path.join(self._tmp.name, "sharding_worker.py")
        with open(script, "w", encoding="utf-8") as fh:
            fh.write(_SHARDING_WORKER)
        out = os.path.join(self._tmp.name, "out")
        os.makedirs(out, exist_ok=True)
        subprocess.run(
            [
                "torchrun",
                "--nproc_per_node=2",
                f"--master_port={self._free_port()}",
                script,
                model_dir,
                out,
                ",".join(modes),
                # the model's own class: `AutoModelForCausalLM` cannot resolve a multimodal
                # `ForConditionalGeneration` from its config
                f"{model_cls.__module__}:{model_cls.__name__}",
            ],
            check=True,
            env={**os.environ, **self._env, "TOKENIZERS_PARALLELISM": "false"},
        )
        return {m: torch.load(os.path.join(out, m + ".pt")) for m in modes}

    def test_every_load_path_agrees(self):
        import torch

        for label in self._model_table():
            with self.subTest(model=label):
                model_dir, config, model_cls = self._fixture(label)
                # the baseline, spread across both devices: `max_memory` forces a real split,
                # since `auto` alone fits this whole model on one GPU and would quietly compare
                # the sharded legs against a single-device load
                reference = self._logits(model_dir, model_cls, device_map="auto", max_memory={0: "120MiB", 1: "40GiB"})
                noise_floor = self._rounding_floor(model_dir, model_cls, reference)

                # Only the modes this model's OWN plans shard experts under. A mode whose plan
                # has no expert entry (GPT-OSS and DeepSeek-V4 ship no TP plan at all) places
                # nothing, so running it would pass without being evidence of anything. A
                # multimodal model keeps these on a sub-config.
                owners = [config] + [
                    c for name in getattr(type(config), "sub_configs", {}) if (c := getattr(config, name, None))
                ]
                modes = [
                    mode
                    for mode, attr in (("ep", "base_model_ep_plan"), ("tp", "base_model_tp_plan"))
                    if any(".experts." in key for owner in owners for key in (getattr(owner, attr, None) or {}))
                ]
                self.assertTrue(modes, f"{label}: no plan shards experts, so nothing is under test")
                whole = _checkpoint_expert_shape(model_dir)
                for mode, payload in self._sharded(model_dir, model_cls, modes).items():
                    with self.subTest(model=label, mode=mode):
                        local = payload["expert_local"]
                        # against the CHECKPOINT's own shape, never another mode's shard: seeding
                        # this from the first leg made the second leg check EP against TP, so a
                        # mode that placed nothing still looked sharded (glm4v's TP left its
                        # experts replicated and passed).
                        self.assertLess(
                            local[0] * local[1],
                            whole[0] * whole[1],
                            f"{label}/{mode}: experts were not sharded ({local} of {whole}) — "
                            f"this leg cannot catch a bad axis",
                        )
                        # Sharding reorders reductions (a rowwise all-reduce sums the same terms
                        # in another order), so the legs differ by BF16 rounding at the first
                        # sharded op whatever the axes. What that becomes at the logits is the
                        # MODEL's business: the NVFP4 fixture amplifies it ~100x across two MoE
                        # layers (measured: a 1e-3 perturbation moves its logits by 0.99 on ONE
                        # device), while the dense fixtures barely amplify at all. A fixed
                        # tolerance therefore asks the impossible of one model and nothing of
                        # another. Compare against the model's OWN floor instead: perturb the
                        # unsharded run by the same rounding and take the resulting logit shift as
                        # the budget. A wrong shard axis lands orders of magnitude above it.
                        budget = max(2e-2, noise_floor)
                        gap = (payload["logits"] - reference).abs().max().item()
                        self.assertLessEqual(
                            gap,
                            budget,
                            f"{label}/{mode}: sharded logits differ by {gap:.4g}, beyond this "
                            f"model's own rounding budget {budget:.4g} — that is a sharding bug, "
                            f"not reduction order",
                        )
                        if noise_floor <= 2e-2:
                            # only meaningful where rounding does NOT already move the argmax
                            self.assertTrue(
                                torch.equal(payload["logits"].argmax(-1), reference.argmax(-1)),
                                f"{label}/{mode}: argmax diverged from the unsharded model",
                            )
