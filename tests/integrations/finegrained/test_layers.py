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
"""Tests for the `finegrained` modules and parallel plans, with no kernel involved.

The kernel call contract is tested in `tests/kernels/test_finegrained.py`.
The conversion ops are tested in `test_conversions.py`.
"""

import pickle
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

import transformers.integrations.finegrained.layers as fg
from transformers.integrations.finegrained import FineGrainedExperts, FineGrainedLinear
from transformers.testing_utils import require_torch
from transformers.utils.quantization_config import Fp8Config, NVFP4Config

from .utils import (
    _Cfg,
    _format_config,
    _quantizer_for,
    _replace,
    _tiny_config,
    fake_finegrained_kernel,
    patch_finegrained_kernel,
)


def _block_scaled_matmul_2d(A, B, As, Bs, *, output_dtype=None, **kwargs):
    """Reference `matmul_2d` that reads the scale grid row-major and infers the block from the shapes."""
    block_n, block_k = B.shape[0] // Bs.shape[0], B.shape[1] // Bs.shape[1]
    weight = B.float() * Bs.float().repeat_interleave(block_n, 0).repeat_interleave(block_k, 1)
    return (A.float() @ weight.T).to(output_dtype)


@require_torch
class FineGrainedModulesTest(unittest.TestCase):
    """Check what the finegrained modules hold and which modules they replace."""

    def test_storage_can_be_built_on_meta_without_a_full_precision_table(self):
        with torch.device("meta"):
            m = fg.FineGrainedEmbedding(320001536, 160, quantization_config=Fp8Config(weight_block_size=None))
        self.assertEqual(m.weight.dtype, torch.float8_e4m3fn)
        self.assertEqual(m.weight_scale.shape, (1,))
        self.assertTrue(all(p.is_meta for p in m.parameters()))

    def test_only_per_tensor_fp8_tables(self):
        for config in (NVFP4Config(), Fp8Config(weight_block_size=(128, 128))):
            with self.subTest(format=config.method), self.assertRaisesRegex(ValueError, "per-tensor FP8"):
                fg.FineGrainedEmbedding(8, 32, quantization_config=config)

    def test_non_mx_formats_do_not_require_ue8m0(self):
        from transformers.integrations.finegrained import conversions, formats

        with (
            mock.patch.object(formats, "ue8m0_dtype", side_effect=RuntimeError("UE8M0 unavailable")),
            mock.patch.object(conversions, "ue8m0_dtype", side_effect=RuntimeError("UE8M0 unavailable")),
        ):
            for config in (Fp8Config(), NVFP4Config()):
                with self.subTest(format=config.method):
                    linear = FineGrainedLinear(64, 64, quantization_config=config)
                    self.assertEqual(linear.weight.element_size(), 1)
            model = torch.nn.Module()
            config = Fp8Config()
            model.linear = FineGrainedLinear(64, 64, quantization_config=config)
            quantizer = _quantizer_for(config)
            quantizer.pre_quantized = False
            result = quantizer.get_quantize_ops().convert({"linear.weight": torch.randn(64, 64)}, model=model)
            self.assertEqual(result["linear.weight"].dtype, torch.float8_e4m3fn)
            self.assertEqual(result["linear.weight_scale_inv"].dtype, torch.float32)

    def test_ue8m0_falls_back_to_exponent_bytes(self):
        from transformers.integrations.finegrained.conversions import FineGrainedQuantize, as_container
        from transformers.integrations.finegrained.formats import ue8m0_dtype
        from transformers.utils.quantization_config import Mxfp8Config

        ue8m0_dtype.cache_clear()
        try:
            with mock.patch.object(torch, "float8_e8m0fnu", create=True):
                del torch.float8_e8m0fnu
                self.assertEqual(ue8m0_dtype(), torch.uint8)
                linear = FineGrainedLinear(64, 64, quantization_config=Mxfp8Config())
                self.assertEqual(linear.weight_scale_inv.dtype, torch.uint8)
                scales = torch.tensor([0.5, 1.0, 2.0, 4.0])
                encoded = as_container(scales, torch.uint8)
                torch.testing.assert_close(encoded, torch.tensor([126, 127, 128, 129], dtype=torch.uint8))
                _, scale = FineGrainedQuantize._quantize_block_fp8(torch.full((2, 2), 448.0), (2, 2), True)
                torch.testing.assert_close(scale, torch.tensor([[127]], dtype=torch.uint8))
        finally:
            ue8m0_dtype.cache_clear()

    def test_only_weights_need_on_the_fly_quantization(self):
        config = Fp8Config()
        quantizer = _quantizer_for(config)
        quantizer.pre_quantized = False
        model = torch.nn.Module()
        model.linear = FineGrainedLinear(64, 64, has_bias=True, quantization_config=config)
        model.experts = FineGrainedExperts(_Cfg(), quantization_config=config, has_bias=True)
        for module_name, module in model.named_children():
            weights = {"weight"} if module_name == "linear" else {"gate_up_proj", "down_proj"}
            for name, _ in module.named_parameters(recurse=False):
                with self.subTest(parameter=f"{module_name}.{name}"):
                    self.assertEqual(
                        quantizer.param_needs_quantization(model, f"{module_name}.{name}"), name in weights
                    )

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
        quantized = FineGrainedExperts.from_module(source, Fp8Config(), model)
        self.assertIs(quantized.post_expert_norm, source.post_expert_norm)
        self.assertFalse(quantized.has_post_expert_norm)

    def test_a_grouped_linear_is_recognised_by_its_groups_not_its_name(self):
        """Check that a grouped linear is detected by its `n_groups` attribute, whatever its class name."""
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
            _replace(model, Fp8Config())
        self.assertIsInstance(model.grouped, FineGrainedGroupedLinear)
        self.assertNotIsInstance(model.plain, FineGrainedGroupedLinear)

    def test_replaced_experts_carry_their_flags_per_instance(self):
        """Check that each replacement holds its flags on the instance and leaves the class unchanged.

        A replaced module must also pickle as the plain class.
        """
        from transformers import GptOssConfig, GptOssForCausalLM

        config = _tiny_config(GptOssConfig, intermediate_size=32, sliding_window=16)
        model = GptOssForCausalLM(config)
        _replace(model, Fp8Config())
        experts = model.model.layers[0].mlp.experts
        self.assertIs(type(experts), FineGrainedExperts)
        self.assertEqual(
            (experts.has_bias, experts.is_concatenated, experts.source_is_transposed), (True, False, True)
        )
        self.assertEqual((FineGrainedExperts.has_bias, FineGrainedExperts.is_concatenated), (False, True))
        direct = FineGrainedExperts(_Cfg(), quantization_config=_format_config("fp8"))
        self.assertFalse(direct.has_bias)
        restored = pickle.loads(pickle.dumps(experts))
        self.assertIs(type(restored), FineGrainedExperts)
        self.assertEqual((restored.has_bias, restored.is_concatenated), (True, False))

    def test_lookup_rescales_the_gathered_rows(self):
        torch.manual_seed(0)
        table = torch.randn(64, 16)
        scale = table.abs().max() / 448.0
        embedding = fg.FineGrainedEmbedding(64, 16, quantization_config=Fp8Config(weight_block_size=None))
        embedding.weight = torch.nn.Parameter((table / scale).to(torch.float8_e4m3fn), requires_grad=False)
        embedding.weight_scale = torch.nn.Parameter(scale.to(torch.bfloat16).reshape(1), requires_grad=False)
        ids = torch.tensor([[3, 7], [60, 0]])
        out = embedding(ids)
        self.assertEqual(out.dtype, torch.bfloat16)
        expected = (table / scale).to(torch.float8_e4m3fn)[ids].to(torch.bfloat16) * scale.to(torch.bfloat16)
        torch.testing.assert_close(out, expected)

    def test_an_empty_plan_replaces_nothing(self):
        """A dequantized load passes an empty plan. It must not fall back to the quantizer's own plan."""
        model = torch.nn.Sequential(torch.nn.Linear(256, 256))
        model.config = SimpleNamespace(get_text_config=lambda: SimpleNamespace())
        self.assertFalse(_quantizer_for(Fp8Config()).replace_modules(model, plan={}))
        self.assertIsInstance(model[0], torch.nn.Linear)

    def test_replacement_follows_the_patterns_and_the_skip_list(self):
        """Check that an embedding is quantized only when `modules_to_convert` names it."""

        model = torch.nn.Module()
        model.ngram_embedding = torch.nn.Embedding(8, 4)
        model.embed_tokens = torch.nn.Embedding(8, 4)
        model.other = torch.nn.Embedding(8, 4)
        _replace(model, Fp8Config(modules_to_convert=["ngram_embedding", "other"]), ["other"])
        self.assertIsInstance(model.ngram_embedding, fg.FineGrainedEmbedding)
        self.assertTrue(model.ngram_embedding.tp_uses_local_tensors)
        self.assertIs(type(model.embed_tokens), torch.nn.Embedding)
        self.assertIs(type(model.other), torch.nn.Embedding)

    # `from_pretrained` builds the model under the checkpoint dtype. A scale must still stay fp32.
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
        # The weight and bias follow the default dtype. The scales must not.
        scales = {name for name in under_fp32 if "scale" in name}
        self.assertTrue(scales)
        for name in sorted(scales):
            self.assertEqual(under_bf16[name], under_fp32[name], f"{name} followed the default dtype")

    def test_scales_ignore_the_default_dtype(self):
        def fp8(scheme):
            return _format_config("fp8", weight_block_size=(4, 4), activation_scheme=scheme)

        builders = {
            "embedding": lambda: fg.FineGrainedEmbedding(64, 16, quantization_config=Fp8Config(weight_block_size=None))
        }
        for name, config in (("fp8_static", fp8("static")), ("fp8_dynamic", fp8("dynamic")), ("nvfp4", NVFP4Config())):
            builders[f"linear_{name}"] = lambda c=config: FineGrainedLinear(64, 32, quantization_config=c)
            builders[f"experts_{name}"] = lambda c=config: FineGrainedExperts(_Cfg(), quantization_config=c)
        for name, build in builders.items():
            with self.subTest(name):
                self._assert_default_dtype_invariant(build)

    def test_static_activation_scheme_reaches_the_fused_chain(self):
        """Check that a static module passes one activation scale per expert to the fused chains.

        A row routed to several experts needs each expert's own scale. A dynamic module passes `None`.
        """
        static, dynamic = (
            FineGrainedExperts(
                _Cfg(), quantization_config=Fp8Config(weight_block_size=(128, 128), activation_scheme=scheme)
            )
            for scheme in ("static", "dynamic")
        )
        held = static.gate_up_proj_activation_scale
        self.assertIsNotNone(held, "the calibrated scale the chain needs")
        self.assertEqual(held.numel(), _Cfg.num_local_experts, "one per expert, which the kernels index")
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

    def test_moe_operands_gather_the_tensors_each_call(self):
        """Check that only the non-tensor operands are cached. A Parameter swapped in later reaches the kernels."""
        # block-scaled gate|up are interleaved on SM100 only
        with mock.patch.object(fg, "is_sm100", return_value=True):
            experts = FineGrainedExperts(_Cfg(), quantization_config=Fp8Config(weight_block_size=(16, 16)))
        kernel = mock.Mock()
        kernel.get_supported_act_fns.return_value = ("silu",)
        kernel.get_supported_norms.return_value = ()
        first = experts._moe_operands(kernel)
        self.assertEqual(first["act_fn"], "silu")
        experts.down_proj = torch.nn.Parameter(torch.empty_like(experts.down_proj), requires_grad=False)
        second = experts._moe_operands(kernel)
        self.assertIs(second["down_proj"], experts.down_proj)
        self.assertIsNot(second["down_proj"], first["down_proj"])
        kernel.get_supported_act_fns.assert_called_once()

    def test_eager_gated_block_scales_pair_each_half_with_its_own_blocks(self):
        """Check that the eager loop pairs each gate and up half with its own scale blocks.

        The loader interleaves the scale grid per block and the rows per row.
        The halves get very different magnitudes, so a scale read from the wrong half shows.
        """
        # block-scaled gate|up are interleaved on SM100 only
        with mock.patch.object(fg, "is_sm100", return_value=True):
            experts = FineGrainedExperts(_Cfg(), quantization_config=Fp8Config(weight_block_size=(32, 32)))
        experts._deepgemm_disabled = True
        num_experts, hidden, inter = _Cfg.num_local_experts, _Cfg.hidden_size, _Cfg.intermediate_size
        torch.manual_seed(0)
        codes = torch.randn(num_experts, 2, inter, hidden).to(torch.float8_e4m3fn)  # [gate, up]
        exponents = torch.tensor([-8.0, 0.0]).reshape(1, 2, 1, 1) + torch.randint(-2, 3, (num_experts, 2, 1, 2))
        grid = torch.pow(2.0, exponents)  # (experts, half, row blocks, column blocks)
        # Rows are [g0, u0, g1, u1, ...] and grid rows are [gate block, up block]
        experts.gate_up_proj = torch.nn.Parameter(codes.transpose(1, 2).reshape(num_experts, 2 * inter, hidden))
        experts.gate_up_proj_scale_inv = torch.nn.Parameter(grid.transpose(1, 2).reshape(num_experts, 2, 2))

        kernel, _ = fake_finegrained_kernel()
        kernel.matmul_2d = _block_scaled_matmul_2d
        x = torch.randn(3, hidden)
        with patch_finegrained_kernel(kernel):
            out = experts.expert_linear(x, "gate_up_proj", 1)
        halves = [
            x @ (codes[1, half].float() * grid[1, half].repeat_interleave(32, 0).repeat_interleave(32, 1)).T
            for half in (0, 1)
        ]
        torch.testing.assert_close(out, torch.stack(halves, dim=-1).flatten(-2))

    def test_eager_gated_per_tensor_scale_is_shared_by_gate_and_up(self):
        experts = FineGrainedExperts(_Cfg(), quantization_config=Fp8Config(weight_block_size=None))
        experts._deepgemm_disabled = True
        torch.manual_seed(0)
        codes = torch.randn_like(experts.gate_up_proj, dtype=torch.float32).to(torch.float8_e4m3fn)
        experts.gate_up_proj = torch.nn.Parameter(codes)
        # On-the-fly quantization of a fused projection gives one scale per expert
        scales = torch.tensor([0.125, 0.5, 2.0, 8.0]).view(-1, 1, 1)
        experts.gate_up_proj_scale_inv = torch.nn.Parameter(scales)
        kernel, _ = fake_finegrained_kernel()
        kernel.matmul_2d = _block_scaled_matmul_2d
        x = torch.randn(3, _Cfg.hidden_size)
        with patch_finegrained_kernel(kernel):
            for expert in range(_Cfg.num_local_experts):
                with self.subTest(expert=expert):
                    actual = experts.expert_linear(x, "gate_up_proj", expert)
                    expected = x @ (codes[expert].float() * scales[expert]).T
                    torch.testing.assert_close(actual, expected)

    def test_block_scaled_gate_up_stay_stacked_off_sm100(self):
        """Check that block-scaled experts off SM100 hold gate|up stacked and gate them by halves."""
        with mock.patch.object(fg, "is_sm100", return_value=False):
            experts = FineGrainedExperts(_Cfg(), quantization_config=Fp8Config(weight_block_size=(32, 32)))
        self.assertFalse(experts.holds_interleaved_gate_up)
        gate, up = torch.randn(3, _Cfg.intermediate_size), torch.randn(3, _Cfg.intermediate_size)
        torch.testing.assert_close(
            experts._apply_gate(torch.cat([gate, up], dim=-1)), torch.nn.functional.silu(gate) * up
        )

    def test_a_model_switches_experts_implementation_after_load(self):
        """Check that the layout does not depend on the experts implementation, so a switch after load works."""
        from transformers import AutoModelForCausalLM, Qwen2MoeConfig

        config = _tiny_config(Qwen2MoeConfig, intermediate_size=32, experts_implementation="deepgemm")
        model = AutoModelForCausalLM.from_config(config)
        mlp = model.model.layers[0].mlp
        mlp.experts = FineGrainedExperts.from_module(mlp.experts, Fp8Config(weight_block_size=(32, 32)), model)
        for implementation in ("grouped_mm", "batched_mm", "eager"):
            with self.subTest(implementation):
                model.set_experts_implementation(implementation)
                self.assertEqual(model.config._experts_implementation, implementation)

    def test_swiglu_alpha_without_a_limit_runs_unclamped(self):
        config = _Cfg()
        config.swiglu_alpha = 1.702
        with mock.patch.object(fg, "is_sm100", return_value=True):
            experts = FineGrainedExperts(config, quantization_config=Fp8Config(weight_block_size=(32, 32)))
        gate_up = torch.randn(3, 2 * _Cfg.intermediate_size)
        gate, up = gate_up[..., 0::2], gate_up[..., 1::2]
        torch.testing.assert_close(experts._apply_gate(gate_up), (up + 1) * gate * torch.sigmoid(gate * 1.702))

    def test_grouped_linear_broadcasts_a_shared_per_tensor_scale(self):
        from transformers.integrations.finegrained import FineGrainedGroupedLinear

        linear = FineGrainedGroupedLinear(
            8, 12, 3, quantization_config=Fp8Config(weight_block_size=None), has_bias=True
        )
        torch.manual_seed(0)
        linear.weight = torch.nn.Parameter(torch.randn(12, 8).to(torch.float8_e4m3fn))
        linear.weight_scale_inv.data.fill_(0.25)
        linear.bias.data.copy_(torch.randn(12))
        kernel, _ = fake_finegrained_kernel()

        def grouped_matmul(a, b, activation_scale, scales, *, expert_start, **kwargs):
            self.assertEqual(expert_start.tolist(), [0, 10, 20, 30, 30])
            return torch.cat(
                [
                    _block_scaled_matmul_2d(
                        a[expert_start[group] : expert_start[group + 1]], b[group], None, scales[group]
                    )
                    for group in range(3)
                ]
            )

        kernel.matmul_grouped = grouped_matmul
        x = torch.randn(2, 5, 3, 8)
        with patch_finegrained_kernel(kernel):
            actual = linear(x)
        weight = linear.weight.float().view(3, 4, 8) * 0.25
        expected = torch.einsum("...gk,gnk->...gn", x, weight) + linear.bias.view(3, 4)
        torch.testing.assert_close(actual, expected)

    def test_grouped_linear_hands_over_its_stored_weight_and_activation_scales(self):
        """Check that the groups split the stored weight and pass the activation scale like `FineGrainedLinear`.

        NVFP4 passes it as the activation global. Static FP8 passes it as the activation scale.
        """
        from transformers.integrations.finegrained import FineGrainedGroupedLinear

        for config, packed in (
            (NVFP4Config(), True),
            (Fp8Config(weight_block_size=(16, 16), activation_scheme="static"), False),
        ):
            with self.subTest(weight_format=config.method.value):
                linear = FineGrainedGroupedLinear(64, 32, 2, quantization_config=config)
                kernel, recorder = fake_finegrained_kernel()
                with patch_finegrained_kernel(kernel):
                    linear(torch.randn(3, 2, 64, dtype=torch.bfloat16))
                (call,) = recorder.calls["matmul_grouped"]
                self.assertEqual(call.args[1].shape, (2, 16, 32 if packed else 64))
                two_level = linear.weight_global_scale is not None
                self.assertIs(call.args[2], None if two_level else linear.activation_scale)
                self.assertIs(call.kwargs["a_global_scale"], linear.activation_scale if two_level else None)


@require_torch
class FineGrainedFusedNormTest(unittest.TestCase):
    """Check that a post-expert norm the kernels implement is passed by name, with its weight and epsilon."""

    def _operands(self, norm):
        experts = FineGrainedExperts(_Cfg(), quantization_config=_format_config("fp8", weight_block_size=(4, 4)))
        experts.post_expert_norm, experts.has_post_expert_norm = norm, True
        kernel = mock.Mock()
        kernel.get_supported_act_fns.return_value = ("silu",)
        kernel.get_supported_norms.return_value = ("rms_norm", "input_scaled_rms_norm")
        return experts._moe_operands(kernel)

    def test_muse_spark_norm_fuses_as_input_scaled_rms_norm(self):
        from transformers.models.muse_spark.modeling_muse_spark import MuseSparkTextInputScaledRMSNorm

        norm = MuseSparkTextInputScaledRMSNorm(_Cfg.hidden_size, eps=1e-5)
        operands = self._operands(norm)
        self.assertEqual(operands["post_expert_norm"], "input_scaled_rms_norm")
        self.assertIs(operands["post_expert_norm_weight"], norm.weight)
        self.assertEqual(operands["post_expert_norm_eps"], 1e-5)
        # A subclass may compute something else, so it is passed as a callable
        subclass = type("Subclass", (MuseSparkTextInputScaledRMSNorm,), {})(_Cfg.hidden_size)
        self.assertIs(self._operands(subclass)["post_expert_norm"], subclass)


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
        from transformers.quantizers.quantizer_nvfp4 import NVFP4HfQuantizer

        quantizer = NVFP4HfQuantizer(NVFP4Config(activation_format="nvfp4"))
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

    def test_an_embedding_tables_scale_is_replicated(self):
        config = Fp8Config(modules_to_convert=["table"])
        quantizer = _quantizer_for(config)
        model = torch.nn.Module()
        model.config = SimpleNamespace(_experts_implementation=None)
        model.table = fg.FineGrainedEmbedding.from_module(torch.nn.Embedding(8, 64), quantizer.plan()["table$"], model)
        for style in ("embedding_colwise", "replicate"):
            with self.subTest(style=style):
                plan = {"table": style}
                quantizer.update_parallel_plan(model, plan)
                self.assertEqual(plan["table.weight_scale"], "replicate")

    def test_every_companion_shards_with_its_expert_weight(self):
        """Check that every expert companion gets the same plan entry as its weight.

        Without its own entry a scale falls back to the module entry, which shards nothing.
        """
        from transformers.distributed.tensor_parallel import _get_parameter_plan

        plan = self._planned()

        def style(name):
            return _get_parameter_plan(f"layers.3.mlp.experts.{name}", plan)

        for proj in ("gate_up_proj", "down_proj"):
            for name in (proj, f"{proj}_scale_inv", f"{proj}_weight_global_scale", f"{proj}_activation_scale"):
                self.assertEqual(style(name), "grouped_gemm", name)
            self.assertEqual(style(f"{proj}_bias"), "grouped_gemm")
        self.assertEqual(plan["layers.*.self_attn.q_proj"], "colwise")

    def test_intra_expert_entries_follow_the_projection_axis(self):
        """Check that intra-expert TP splits the scale grid like its weight.

        The per-expert globals stay replicated.
        """
        from transformers.distributed.tensor_parallel import _get_parameter_plan

        base = {
            "layers.*.mlp.experts.gate_up_proj": "packed_colwise",
            "layers.*.mlp.experts.down_proj": "rowwise",
            "layers.*.mlp.experts": "moe_tp_experts",
        }
        plan = self._planned(raw=base)

        def style(name):
            return _get_parameter_plan(f"layers.3.mlp.experts.{name}", plan)

        # The weight keeps the model's style. The interleave runs on each shard after sharding.
        self.assertEqual(style("gate_up_proj"), "packed_colwise")
        for name in ("gate_up_proj_scale_inv", "gate_up_proj_bias"):
            self.assertEqual(style(name), "moe_experts_packed_colwise", name)
        # The down scale splits on dim 2, the reduce axis of both scale layouts
        self.assertEqual(style("down_proj"), "rowwise")
        self.assertEqual(style("down_proj_scale_inv"), "moe_experts_rowwise")
        # TP does not split experts, so per-expert tensors stay whole
        for name in ("gate_up_proj_weight_global_scale", "down_proj_weight_global_scale", "down_proj_bias"):
            self.assertEqual(style(name), "moe_tp_experts", name)

        # A natively interleaved layout declares `colwise`. Its companions split contiguously.
        inter = dict(base, **{"layers.*.mlp.experts.gate_up_proj": "colwise"})
        plan_i = self._planned(raw=inter)
        self.assertEqual(_get_parameter_plan("layers.3.mlp.experts.gate_up_proj", plan_i), "colwise")
        self.assertEqual(
            _get_parameter_plan("layers.3.mlp.experts.gate_up_proj_scale_inv", plan_i),
            "moe_experts_colwise",
        )

    def test_dense_projections_that_share_a_name_are_left_alone(self):
        """Check that dense projections named like the expert projections keep their plan entries."""
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
        # A dense scale already gets the module entry, so no entry is added for it
        strays = [k for k in plan if k.endswith("_scale_inv") and ".experts." not in k]
        self.assertEqual(strays, [], f"invented dense scale entries: {strays}")
        self.assertEqual(
            _get_parameter_plan("layers.3.mlp.experts.gate_up_proj_scale_inv", plan),
            "moe_experts_packed_colwise",
        )

    def test_a_multimodal_models_expert_plans_are_reached(self):
        from transformers.quantizers.quantizer_fp8 import Fp8HfQuantizer

        from .utils import _conversion_model

        quantizer = Fp8HfQuantizer(Fp8Config())
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
        # Mega MoE does its own dispatch across ranks
        dispatched = self._planned("deepgemm_megamoe", raw={"layers.*.mlp.experts": "ep_dispatch_experts"})
        self.assertEqual(dispatched["layers.*.mlp.experts"], "megamoe_experts")

    def test_a_real_models_plan_keeps_its_entries_and_gains_the_expert_companions(self):
        from transformers import AutoModelForCausalLM, Qwen3Config, Qwen3MoeConfig
        from transformers.quantizers.quantizer_fp8 import Fp8HfQuantizer

        for config_cls in (Qwen3Config, Qwen3MoeConfig):
            with self.subTest(config=config_cls.__name__), torch.device("meta"):
                config = _tiny_config(config_cls)
                shipped = dict(config.base_model_tp_plan)
                model = AutoModelForCausalLM.from_config(config)
                before = dict(model.tp_plan)
                quantizer = Fp8HfQuantizer(Fp8Config())
                quantizer.preprocess_model(model)
                for key, style in before.items():
                    self.assertEqual(model.tp_plan[key], style, key)
                self.assertEqual(config.base_model_tp_plan, shipped)
        for companion in ("gate_up_proj_scale_inv", "down_proj_scale_inv"):
            self.assertIn(f"model.layers.*.mlp.experts.{companion}", model.tp_plan)
