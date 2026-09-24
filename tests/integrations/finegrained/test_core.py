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

import re
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
from transformers.testing_utils import require_torch


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


def _targets(conv):
    t = conv.target_patterns
    return t if isinstance(t, (list, tuple)) else [t]


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
        substitute for: a branch keyed on the config CLASS is invisible to a fake, and a name
        substring (`"Qwen3" in config.__class__.__name__`) also claims Qwen3-MoE.
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
