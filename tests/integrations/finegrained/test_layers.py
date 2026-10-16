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
"""The `finegrained` integration's modules, no kernel involved: what replaces what, what the modules hold and
hand the kernels, and the parallel plans a quantized model needs. The kernel call contract lives in
`tests/kernels/test_finegrained.py`, the conversion ops in `test_conversions.py`."""

import pickle
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

import transformers.integrations.finegrained.layers as fg
from transformers.integrations.finegrained import FineGrainedExperts, FineGrainedLinear
from transformers.testing_utils import require_torch
from transformers.utils.quantization_config import BlockFp8Config, NVFP4Config

from .utils import (
    _Cfg,
    _format_config,
    _quantizer_for,
    _replace,
    fake_finegrained_kernel,
    patch_finegrained_kernel,
)


def _block_scaled_matmul_2d(A, B, As, Bs, *, output_dtype=None, **kwargs):
    """`matmul_2d`'s contract on a block-scaled weight: the grid read row-major, its block inferred from the shapes."""
    block_n, block_k = B.shape[0] // Bs.shape[0], B.shape[1] // Bs.shape[1]
    weight = B.float() * Bs.float().repeat_interleave(block_n, 0).repeat_interleave(block_k, 1)
    return (A.float() @ weight.T).to(output_dtype)


@require_torch
class FineGrainedModulesTest(unittest.TestCase):
    """The finegrained modules themselves: what they hold and what replaces what."""

    def test_non_mx_formats_do_not_require_ue8m0(self):
        from transformers.integrations.finegrained import conversions

        with (
            mock.patch.object(fg, "_get_ue8m0_dtype", side_effect=RuntimeError("UE8M0 unavailable")),
            mock.patch.object(conversions, "_get_ue8m0_dtype", side_effect=RuntimeError("UE8M0 unavailable")),
        ):
            for config in (BlockFp8Config(), NVFP4Config()):
                with self.subTest(format=config.method):
                    linear = FineGrainedLinear(64, 64, quantization_config=config)
                    self.assertEqual(linear.weight.element_size(), 1)
            model = torch.nn.Module()
            config = BlockFp8Config()
            model.linear = FineGrainedLinear(64, 64, quantization_config=config)
            quantizer = _quantizer_for(config)
            quantizer.pre_quantized = False
            result = quantizer.get_quantize_ops().convert({"linear.weight": torch.randn(64, 64)}, model=model)
            self.assertEqual(result["linear.weight"].dtype, torch.float8_e4m3fn)
            self.assertEqual(result["linear.weight_scale_inv"].dtype, torch.float32)

    def test_ue8m0_falls_back_to_exponent_bytes(self):
        from transformers.integrations.finegrained.conversions import FineGrainedQuantize, as_container
        from transformers.utils.quantization_config import Mxfp8Config

        fg._get_ue8m0_dtype.cache_clear()
        try:
            with mock.patch.object(torch, "float8_e8m0fnu", create=True):
                del torch.float8_e8m0fnu
                self.assertEqual(fg._get_ue8m0_dtype(), torch.uint8)
                linear = FineGrainedLinear(64, 64, quantization_config=Mxfp8Config())
                self.assertEqual(linear.weight_scale_inv.dtype, torch.uint8)
                scales = torch.tensor([0.5, 1.0, 2.0, 4.0])
                encoded = as_container(scales, torch.uint8)
                torch.testing.assert_close(encoded, torch.tensor([126, 127, 128, 129], dtype=torch.uint8))
                torch.testing.assert_close(as_container(encoded, torch.float32), scales)
                _, scale = FineGrainedQuantize._quantize_block_fp8(torch.full((2, 2), 448.0), (2, 2), True)
                torch.testing.assert_close(scale, torch.tensor([[127]], dtype=torch.uint8))
        finally:
            fg._get_ue8m0_dtype.cache_clear()

    def test_replacement_respects_a_disabled_post_expert_norm(self):
        from transformers.integrations.moe import use_experts_implementation

        @use_experts_implementation(has_post_expert_norm=False)
        class Experts(torch.nn.Module):
            def __init__(self, config):
                super().__init__()
                self.config = config
                self.post_expert_norm = torch.nn.RMSNorm(config.hidden_size)

            def forward(self, hidden_states, top_k_index, top_k_weights):
                return hidden_states

        config = _Cfg()
        source = Experts(config)
        self.assertFalse(source.has_post_expert_norm)
        model = SimpleNamespace(config=SimpleNamespace(get_text_config=lambda: config))
        quantized = FineGrainedExperts.from_module(source, BlockFp8Config(), model)
        self.assertIs(quantized.post_expert_norm, source.post_expert_norm)
        self.assertFalse(quantized.has_post_expert_norm)

    def test_a_grouped_linear_is_recognised_by_its_groups_not_its_name(self):
        """`replace_with_quantized_modules` picks `FineGrainedGroupedLinear` for a block-diagonal
        linear — a plain one would collapse the groups into a single giant matmul and return the
        wrong output dim. The branch keys on `n_groups`, the attribute the swap consumes: a class
        NAME substring would miss a grouped linear named anything else, and would match (then
        `AttributeError` on) one that carries its group count under another name.
        """
        from transformers.integrations.finegrained import FineGrainedGroupedLinear

        class OddlyNamedBlockDiagonal(torch.nn.Linear):
            def __init__(self):
                super().__init__(8, 16, bias=False)
                self.n_groups = 2

        model = torch.nn.Module()
        model.config = SimpleNamespace(get_text_config=lambda: SimpleNamespace())
        model.grouped = OddlyNamedBlockDiagonal()
        model.plain = torch.nn.Linear(8, 16, bias=False)

        with torch.device("meta"):
            _replace(model, BlockFp8Config())
        self.assertIsInstance(model.grouped, FineGrainedGroupedLinear)
        self.assertNotIsInstance(model.plain, FineGrainedGroupedLinear)

    # A quantized embedding TABLE (Qwen4-Exp's n-gram table): FP8 rows with one per-tensor scale,
    # rescaled on the rows a lookup gathers; swapped in for the names in `modules_to_convert`.
    def test_replacing_experts_leaves_the_experts_class_as_defined(self):
        """Each layer's flags come with its own class, shared by the layers that agree on them; a load modifies
        neither `FineGrainedExperts` nor what a later direct construction gets."""
        from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM

        config = Qwen3MoeConfig(
            vocab_size=64,
            hidden_size=64,
            intermediate_size=64,
            moe_intermediate_size=32,
            num_hidden_layers=3,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=32,
            num_experts=2,
            num_experts_per_tok=1,
        )
        with torch.device("meta"):
            model = Qwen3MoeForCausalLM(config)
            _replace(model, BlockFp8Config())
            experts = {type(layer.mlp.experts) for layer in model.model.layers}
            self.assertEqual(len(experts), 1)
            self.assertTrue(issubclass(experts.pop(), FineGrainedExperts))
            self.assertFalse(hasattr(FineGrainedExperts.forward, "__wrapped__"))
            direct = FineGrainedExperts(_Cfg(), quantization_config=_format_config("fp8"), has_bias=True)
        self.assertTrue(direct.has_bias)
        # a replaced experts class is found by name, so a pickled model loads back
        experts = model.model.layers[0].mlp.experts
        self.assertIs(pickle.loads(pickle.dumps(type(experts))), type(experts))

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

    def test_a_skip_list_names_modules_by_pattern(self):
        """`should_convert_module`: everything converts by default, a skip-list entry names its modules."""
        from transformers.quantizers.quantizers_utils import should_convert_module

        self.assertTrue(should_convert_module("model", None))
        self.assertTrue(should_convert_module("model", []))
        patterns = ["model.layers.*.self_attn", "lm_head"]
        self.assertFalse(should_convert_module("lm_head", patterns))
        self.assertTrue(should_convert_module("experts", patterns))

    def test_an_empty_plan_replaces_nothing(self):
        """A dequantized load walks an empty plan; it must not fall back to the quantizer's own."""
        model = torch.nn.Sequential(torch.nn.Linear(256, 256))
        model.config = SimpleNamespace(get_text_config=lambda: SimpleNamespace())
        self.assertFalse(_quantizer_for(BlockFp8Config()).replace_modules(model, plan={}))
        self.assertIsInstance(model[0], torch.nn.Linear)

    def test_replacement_follows_the_patterns_and_the_skip_list(self):
        """A table is quantized where `modules_to_convert` names it, never through the plan's catch-all."""

        model = torch.nn.Module()
        model.ngram_embedding = torch.nn.Embedding(8, 4)
        model.embed_tokens = torch.nn.Embedding(8, 4)
        model.other = torch.nn.Embedding(8, 4)
        _replace(model, BlockFp8Config(modules_to_convert=["ngram_embedding", "other"]), ["other"])
        self.assertIsInstance(model.ngram_embedding, fg.FineGrainedEmbedding)
        self.assertTrue(model.ngram_embedding._hf_quantized_needs_local_tp)
        self.assertIs(type(model.embed_tokens), torch.nn.Embedding)
        self.assertIs(type(model.other), torch.nn.Embedding)

    # `from_pretrained` sets the default dtype to the checkpoint's during construction, so a scale
    # allocated as `torch.ones(n)` comes out bf16 and the kernels read it as fp32 (NaN logits, no
    # error). Building under both defaults catches it.
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
                        quantization_config=_format_config(
                            f, **({"weight_block_size": (4, 4), "activation_scheme": s} if f == "fp8" else {})
                        ),
                        in_features=64,
                        out_features=32,
                    )
                )

    def test_embedding_scale_ignores_the_default_dtype(self):
        self._assert_default_dtype_invariant(lambda: fg.FineGrainedEmbedding(64, 16))

    def test_experts_scales_ignore_the_default_dtype(self):
        for weight_format, scheme in (("fp8", "static"), ("fp8", "dynamic"), ("nvfp4", "dynamic")):
            with self.subTest(weight_format=weight_format, activation_scheme=scheme):
                self._assert_default_dtype_invariant(
                    lambda f=weight_format, s=scheme: FineGrainedExperts(
                        _Cfg(),
                        quantization_config=_format_config(
                            f, **({"weight_block_size": (4, 4), "activation_scheme": s} if f == "fp8" else {})
                        ),
                    )
                )

    def test_static_activation_scheme_reaches_the_fused_chain(self):
        """Mistral-3 ships `activation_scheme="static"`: a calibrated scale per activation, and a
        MoE calibrates each expert separately, so the module holds one per expert. Both fused
        chains take them per projection and the kernels apply each tile's own expert scale in
        register — a host pre-quant could not, since top-k routes one row to several experts
        whose scales differ. A dynamic module holds no such slot and passes `None`."""
        static, dynamic = (
            FineGrainedExperts(
                _Cfg(), quantization_config=BlockFp8Config(weight_block_size=(128, 128), activation_scheme=scheme)
            )
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

    def test_eager_gated_block_scales_pair_each_half_with_its_own_blocks(self):
        """The loader interleaves a gate|up block grid per BLOCK, so one scale covers rows of one half while
        the rows alternate between halves: the eager loop runs each half on its own rows and blocks rather
        than hand `matmul_2d` the stack as a row-major grid. Halves of very different magnitudes, as real
        gate and up projections have, so a scale read from the wrong half shows."""
        experts = FineGrainedExperts(_Cfg(), quantization_config=BlockFp8Config(weight_block_size=(32, 32)))
        experts._deepgemm_disabled = True
        num_experts, hidden, inter = _Cfg.num_local_experts, _Cfg.hidden_size, _Cfg.intermediate_size
        torch.manual_seed(0)
        codes = torch.randn(num_experts, 2, inter, hidden).to(torch.float8_e4m3fn)  # [gate, up]
        exponents = torch.tensor([-8.0, 0.0]).reshape(1, 2, 1, 1) + torch.randint(-2, 3, (num_experts, 2, 1, 2))
        grid = torch.pow(2.0, exponents)  # (experts, half, row blocks, column blocks)
        # interleaved: rows [g0, u0, g1, u1, ...], grid rows [gate block, up block]
        experts.gate_up_proj = torch.nn.Parameter(codes.transpose(1, 2).reshape(num_experts, 2 * inter, hidden))
        experts.gate_up_proj_scale_inv = torch.nn.Parameter(grid.transpose(1, 2).reshape(num_experts, 2, 2))

        kernel, _ = fake_finegrained_kernel()
        kernel.matmul_2d = _block_scaled_matmul_2d
        x = torch.randn(3, hidden)
        p1, p2, p3 = patch_finegrained_kernel(kernel)
        with p1, p2, p3:
            out = experts.expert_linear(x, "gate_up_proj", 1)
        halves = [
            x @ (codes[1, half].float() * grid[1, half].repeat_interleave(32, 0).repeat_interleave(32, 1)).T
            for half in (0, 1)
        ]
        torch.testing.assert_close(out, torch.stack(halves, dim=-1).flatten(-2))

    def test_stacked_gate_up_is_gated_by_halves_and_kept_from_the_triton_chains(self):
        """The DeepGEMM backends load gate|up stacked: the host GLU splits the halves, and the triton chains,
        which read the rows interleaved, refuse such a module rather than pair the wrong rows."""
        config = _Cfg()
        config._experts_implementation = "deepgemm"
        experts = FineGrainedExperts(config, quantization_config=BlockFp8Config(weight_block_size=(32, 32)))
        self.assertFalse(experts.holds_interleaved_gate_up)
        gate, up = torch.randn(3, _Cfg.intermediate_size), torch.randn(3, _Cfg.intermediate_size)
        torch.testing.assert_close(
            experts._apply_gate(torch.cat([gate, up], dim=-1)), torch.nn.functional.silu(gate) * up
        )

        hidden = torch.randn(2, _Cfg.hidden_size, dtype=torch.bfloat16)
        routing = torch.zeros(2, 1, dtype=torch.long), torch.ones(2, 1)
        with mock.patch.object(fg, "load_finegrained_kernel") as load:
            for forward in (fg.finegrained_grouped_mm_experts_forward, fg.finegrained_batched_mm_experts_forward):
                with (
                    self.subTest(forward=forward.__name__),
                    self.assertRaisesRegex(NotImplementedError, "interleaved"),
                ):
                    forward(experts, hidden, *routing)
        load.assert_not_called()

    def test_swiglu_alpha_without_a_limit_runs_unclamped(self):
        config = _Cfg()
        config.swiglu_alpha = 1.702
        experts = FineGrainedExperts(config, quantization_config=BlockFp8Config(weight_block_size=(32, 32)))
        gate_up = torch.randn(3, 2 * _Cfg.intermediate_size)
        gate, up = gate_up[..., 0::2], gate_up[..., 1::2]
        torch.testing.assert_close(experts._apply_gate(gate_up), (up + 1) * gate * torch.sigmoid(gate * 1.702))

    def test_grouped_linear_hands_over_its_stored_weight_and_activation_scales(self):
        """The groups split the weight as stored (packed FP4 halves K), and the calibrated activation scale goes
        where `FineGrainedLinear` hands it: NVFP4's as the activation global, a static FP8 one as `As`."""
        from transformers.integrations.finegrained import FineGrainedGroupedLinear

        for config, packed in (
            (NVFP4Config(), True),
            (BlockFp8Config(weight_block_size=(16, 16), activation_scheme="static"), False),
        ):
            with self.subTest(weight_format=config.method.value):
                linear = FineGrainedGroupedLinear(64, 32, 2, quantization_config=config)
                kernel, recorder = fake_finegrained_kernel()
                p1, p2, p3 = patch_finegrained_kernel(kernel)
                with p1, p2, p3:
                    linear(torch.randn(3, 2, 64, dtype=torch.bfloat16))
                (call,) = recorder.calls["matmul_grouped"]
                self.assertEqual(call.args[1].shape, (2, 16, 32 if packed else 64))
                two_level = linear.weight_global_scale is not None
                self.assertIs(call.args[2], None if two_level else linear.activation_scale)
                self.assertIs(call.kwargs["a_global_scale"], linear.activation_scale if two_level else None)


@require_torch
class FineGrainedFusedNormTest(unittest.TestCase):
    """A norm the kernels can fuse is not fused when a forward collective sits on its input: under intra-expert
    TP the rows are a partial sum over the sharded intermediate, which the wrapped module all-reduces in its own
    forward."""

    def _operands(self, *, input_reduce):
        experts = FineGrainedExperts(_Cfg(), quantization_config=_format_config("fp8", weight_block_size=(4, 4)))
        experts.post_expert_norm, experts.has_post_expert_norm = torch.nn.RMSNorm(_Cfg.hidden_size), True
        if input_reduce:
            experts.post_expert_norm._hf_tp_input_reduce = True  # what `ReplicatedWithInputAllReduce` marks
        kernel = mock.Mock()
        kernel.get_supported_act_fns.return_value = ("silu",)
        kernel.get_supported_norms.return_value = ("rms_norm",)
        with mock.patch.dict(fg.SUPPORTED_FUSED_NORMS, {torch.nn.RMSNorm: "rms_norm"}):
            return experts._moe_operands(kernel), experts

    def test_a_fusable_norm_fuses_when_nothing_reduces_its_input(self):
        operands, experts = self._operands(input_reduce=False)
        self.assertEqual(operands["post_expert_norm"], "rms_norm")
        self.assertIs(operands["post_expert_norm_weight"], experts.post_expert_norm.weight)

    def test_a_fusable_norm_falls_back_when_its_input_is_all_reduced(self):
        operands, experts = self._operands(input_reduce=True)
        self.assertIs(operands["post_expert_norm"], experts.post_expert_norm)
        self.assertIsNone(operands["post_expert_norm_weight"])


class FineGrainedParallelPlanTest(unittest.TestCase):
    """Update the built model's TP/EP plans without mutating config plans."""

    BASE_EP = {
        "layers.*.mlp.gate": "ep_router",
        "layers.*.mlp.experts.gate_up_proj": "grouped_gemm",
        "layers.*.mlp.experts.down_proj": "grouped_gemm",
        "layers.*.mlp.experts": "moe_tp_experts",
        "layers.*.self_attn.q_proj": "colwise",
    }

    def _planned(self, impl=None, raw=None):
        from transformers.quantizers.quantizer_blockfp8 import BlockFp8HfQuantizer

        quantizer = BlockFp8HfQuantizer(BlockFp8Config())
        model = torch.nn.Module()
        model.config = SimpleNamespace(_experts_implementation=impl)
        model.layers = torch.nn.ModuleList([torch.nn.Module() for _ in range(4)])
        with torch.device("meta"), mock.patch.object(fg, "is_sm100", return_value=False):
            for layer in model.layers:
                layer.mlp = torch.nn.Module()
                layer.mlp.experts = FineGrainedExperts(
                    _Cfg(), quantization_config=NVFP4Config(activation_format="nvfp4"), has_bias=True
                )
        plan = dict(raw or self.BASE_EP)
        quantizer.update_parallel_plan(model, plan)
        return plan

    def test_every_companion_shards_with_its_expert_weight(self):
        """The plan matcher keys on the exact parameter name and falls back only to the owning
        module, whose entry shards nothing — so a scale with no entry of its own stays whole while
        its weight is this rank's expert slice, and every rank but the first reads another rank's
        scales."""
        from transformers.distributed.tensor_parallel import _get_parameter_plan

        plan = self._planned()

        def style(name):
            return _get_parameter_plan(f"layers.3.mlp.experts.{name}", plan)

        for proj in ("gate_up_proj", "down_proj"):
            for name in (proj, f"{proj}_scale_inv", f"{proj}_weight_global_scale", f"{proj}_activation_scale"):
                self.assertEqual(style(name), "grouped_gemm", name)
            for name in (f"{proj}_bias",):
                self.assertEqual(style(name), "grouped_gemm", name)
        self.assertEqual(plan["layers.*.self_attn.q_proj"], "colwise")

    def test_intra_expert_entries_follow_the_projection_axis(self):
        """Under intra-expert TP the experts stay whole and the projection's own axis splits, so the scale grid
        follows its weight while the per-expert globals stay replicated."""
        from transformers.distributed.tensor_parallel import _get_parameter_plan

        base = {
            "layers.*.mlp.experts.gate_up_proj": "packed_colwise",
            "layers.*.mlp.experts.down_proj": "rowwise",
            "layers.*.mlp.experts": "moe_tp_experts",
        }
        plan = self._planned(raw=base)

        def style(name):
            return _get_parameter_plan(f"layers.3.mlp.experts.{name}", plan)

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
        self.assertEqual(_get_parameter_plan("layers.3.mlp.experts.gate_up_proj", plan_i), "colwise")
        self.assertEqual(
            _get_parameter_plan("layers.3.mlp.experts.gate_up_proj_scale_inv", plan_i),
            "moe_experts_colwise",
        )

    def test_dense_projections_that_share_a_name_are_left_alone(self):
        """`mlp.up_proj` and `mlp.shared_experts.down_proj` end in the same words as the stacked expert
        projections but are plain 2-D linears that colwise/rowwise already shard right; an expert-axis style
        would shard a dim they do not have."""
        from transformers.distributed.tensor_parallel import _get_parameter_plan

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
            _get_parameter_plan("layers.3.mlp.experts.gate_up_proj_scale_inv", plan),
            "moe_experts_packed_colwise",
        )

    def test_a_multimodal_models_expert_plans_are_reached(self):
        from transformers.quantizers.quantizer_blockfp8 import BlockFp8HfQuantizer

        from .utils import _conversion_model

        quantizer = BlockFp8HfQuantizer(BlockFp8Config())
        text_model = _conversion_model(quantizer)
        model = torch.nn.Module()
        model.config = SimpleNamespace(_experts_implementation=None)
        model.language_model = text_model
        plan = {"language_model.model.layers.*.mlp.experts.gate_up_proj": "grouped_gemm"}
        quantizer.update_parallel_plan(model, plan)
        self.assertEqual(plan["language_model.model.layers.*.mlp.experts.gate_up_proj_scale_inv"], "grouped_gemm")
        self.assertFalse(text_model.config.base_model_ep_plan)

    def test_the_impl_rewrites_the_layer_kinds(self):
        megamoe = self._planned("deepgemm_megamoe")
        self.assertEqual(megamoe["layers.*.mlp.gate"], "megamoe_router")
        self.assertEqual(megamoe["layers.*.mlp.experts"], "megamoe_experts")
        # Mega MoE dispatches across ranks itself, in place of the EP dispatch a plan may default to
        dispatched = self._planned("deepgemm_megamoe", raw={"layers.*.mlp.experts": "ep_dispatch_experts"})
        self.assertEqual(dispatched["layers.*.mlp.experts"], "megamoe_experts")

    def test_a_real_models_plan_keeps_its_entries_and_gains_the_expert_companions(self):
        from transformers import AutoModelForCausalLM, Qwen3Config, Qwen3MoeConfig
        from transformers.quantizers.quantizer_blockfp8 import BlockFp8HfQuantizer

        for config_cls in (Qwen3Config, Qwen3MoeConfig):
            with self.subTest(config=config_cls.__name__), torch.device("meta"):
                config = config_cls(
                    vocab_size=64,
                    hidden_size=64,
                    intermediate_size=64,
                    moe_intermediate_size=32,
                    num_hidden_layers=1,
                    num_attention_heads=2,
                    num_key_value_heads=1,
                    head_dim=32,
                    num_experts=2,
                    num_experts_per_tok=1,
                )
                shipped = dict(config.base_model_tp_plan)
                model = AutoModelForCausalLM.from_config(config)
                before = dict(model.tp_plan)
                quantizer = BlockFp8HfQuantizer(BlockFp8Config())
                quantizer.preprocess_model(model)
                for key, style in before.items():
                    self.assertEqual(model.tp_plan[key], style, key)
                self.assertEqual(config.base_model_tp_plan, shipped)
        for companion in ("gate_up_proj_scale_inv", "down_proj_scale_inv"):
            self.assertIn(f"model.layers.*.mlp.experts.{companion}", model.tp_plan)
