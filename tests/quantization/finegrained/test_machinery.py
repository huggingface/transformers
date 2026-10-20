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
"""The fine-grained machinery against two formats defined here, for the tests only: the modules, the conversion ops,
the shared quantizer and the host need no shipped format. Each shipped format (FP8, MXFP4, NVFP4, MXFP8) brings its
config, its quantizer and its own tests on top."""

import json
import unittest
from contextlib import ExitStack, contextmanager
from types import SimpleNamespace
from unittest import mock

import torch

import transformers.integrations.deepgemm as deepgemm
import transformers.integrations.finegrained.layers as fg
from transformers.core_model_loading import convert_and_load_state_dict_in_model
from transformers.distributed.tensor_parallel import _get_parameter_plan
from transformers.integrations.finegrained import FineGrainedExperts, FineGrainedLinear
from transformers.modeling_utils import LoadStateDictConfig
from transformers.quantizers import auto
from transformers.quantizers.block_scaled_utils import BlockScaledQuantizer
from transformers.quantizers.quantizer_finegrained import FineGrainedHfQuantizer
from transformers.testing_utils import require_torch
from transformers.utils import quantization_config as qc
from transformers.utils.quantization_config import BlockScaledConfig, FineGrainedConfig, QuantizationMethod

from ...integrations.finegrained.utils import (
    E,
    H,
    I,
    _Cfg,
    _ChainAssertions,
    _load_and_save,
    _model_mapping,
    _moe_config,
    _MoeModel,
    _quantizer_for,
    fake_finegrained_kernel,
    patch_finegrained_kernel,
)
from ..block_scaled_tester import BlockScaledFormatTesterMixin, FormatSpec


class BlockFp8TestConfig(BlockScaledConfig):
    """Test-only: `float8_e4m3fn` weights with an fp32 scale per 32x32 block, dynamic activations."""

    method = QuantizationMethod.FP8
    weight_block_size = (32, 32)


class GroupFp4TestConfig(BlockScaledConfig):
    """Test-only: E2M1 weights two per byte with a UE8M0 scale per 32, weight-only by default."""

    method = QuantizationMethod.MXFP4
    activation_formats = (None, "bf16")
    default_activation_format = "bf16"
    weight_block_size = (1, 32)
    scale_fmt = "ue8m0"


class BlockFp8TestQuantizer(BlockScaledQuantizer):
    pass


class GroupFp4TestQuantizer(BlockScaledQuantizer):
    pass


EXPERTS = r"\.experts($|\.)"


@contextmanager
def registered_test_formats():
    """The test formats under their `quant_method`, as a shipped format registers itself."""
    configs = {"fp8": BlockFp8TestConfig, "mxfp4": GroupFp4TestConfig}
    quantizers = {"fp8": BlockFp8TestQuantizer, "mxfp4": GroupFp4TestQuantizer}
    with ExitStack() as stack:
        stack.enter_context(mock.patch.dict(auto.AUTO_QUANTIZER_MAPPING, quantizers))
        stack.enter_context(mock.patch.dict(auto.AUTO_QUANTIZATION_CONFIG_MAPPING, configs))
        stack.enter_context(mock.patch.dict(qc.BLOCK_SCALED_CONFIGS, configs))
        yield


def _fp8_blocks(n, k):
    return {
        "weight": torch.randn(n, k).to(torch.float8_e4m3fn),
        "weight_scale_inv": torch.rand(n // 32, k // 32) + 0.5,
    }


def _fp4_groups(n, k):
    return {
        "weight": torch.randint(-128, 127, (n, k // 2), dtype=torch.int8),
        "weight_scale_inv": torch.randint(120, 130, (n, k // 32), dtype=torch.uint8).view(torch.float8_e8m0fnu),
    }


def _checkpoint(experts, dense):
    """One layer in the per-expert layout: the experts' projections from `experts`, `q_proj` from `dense`."""
    checkpoint = {}
    for expert in range(E):
        for proj, shape in (("gate_proj", (I, H)), ("up_proj", (I, H)), ("down_proj", (H, I))):
            for suffix, tensor in experts(*shape).items():
                checkpoint[f"model.layers.0.mlp.experts.{expert}.{proj}.{suffix}"] = tensor
    for suffix, tensor in dense(H, H).items():
        checkpoint[f"model.layers.0.self_attn.q_proj.{suffix}"] = tensor
    return checkpoint


@require_torch
class TestFormatContractTest(BlockScaledFormatTesterMixin, unittest.TestCase):
    """The contract every shipped format keeps, kept by a format defined in the tests."""

    spec = FormatSpec(
        name="fp8",
        config_cls=BlockFp8TestConfig,
        quantizer_cls=BlockFp8TestQuantizer,
        checkpoint_config=BlockFp8TestConfig,
        activation_formats=(None,),
        weight_bits=8,
        checkpoint_keys=(
            "model.layers.3.mlp.experts.7.gate_proj.weight_scale_inv",
            "model.layers.3.mlp.experts.gate_up_proj_scale_inv",
            "model.layers.3.self_attn.q_proj.weight_scale_inv",
        ),
        dequantize_keys=("model.layers.0.mlp.down_proj.weight", "model.layers.0.mlp.down_proj.weight_scale_inv"),
    )

    def setUp(self):
        super().setUp()
        self.enterContext(registered_test_formats())


@require_torch
class TestFormatModulesTest(_ChainAssertions, unittest.TestCase):
    """The modules a format's config builds, what they hold and what they hand the kernels."""

    def setUp(self):
        super().setUp()
        self.enterContext(registered_test_formats())
        self.enterContext(mock.patch.object(fg, "is_sm100", return_value=False))

    def test_the_base_config_is_not_a_format(self):
        with self.assertRaises(TypeError):
            BlockScaledConfig()

    def test_a_format_swaps_linears_and_experts_and_keeps_its_skip_list(self):
        for skip in (None, ["q_proj"]):
            with self.subTest(skip=skip):
                with torch.device("meta"):
                    model = _MoeModel(_moe_config())
                _quantizer_for(BlockFp8TestConfig()).replace_modules(model, modules_to_not_convert=skip)
                layer = model.model.layers[0]
                self.assertIsInstance(layer.mlp.experts, FineGrainedExperts)
                if skip:
                    self.assertIs(type(layer.self_attn.q_proj), torch.nn.Linear)
                    continue
                q_proj = layer.self_attn.q_proj
                self.assertIsInstance(q_proj, FineGrainedLinear)
                self.assertTrue(q_proj.tp_uses_local_tensors)
                self.assertEqual((q_proj.weight.dtype, q_proj.weight_scale_inv.shape), (torch.float8_e4m3fn, (2, 2)))

    def test_a_linear_hands_the_kernel_its_weight_and_scale(self):
        kernel, rec = fake_finegrained_kernel()
        linear = FineGrainedLinear(64, 32, quantization_config=BlockFp8TestConfig())
        linear.weight.data = torch.randn(32, 64).to(torch.float8_e4m3fn)
        linear.weight_scale_inv.data = torch.ones(1, 2)
        with patch_finegrained_kernel(kernel), mock.patch.object(deepgemm, "is_deepgemm_loadable", return_value=False):
            out = linear(torch.randn(3, 5, 64, dtype=torch.bfloat16))
        call = rec.calls["matmul_2d"][-1]
        self.assertIs(call.args[1], linear.weight)
        self.assertIs(call.args[3], linear.weight_scale_inv)
        self.assertIsNone(call.args[2])
        self.assertEqual(out.shape, (3, 5, 32))

    def test_experts_run_the_kernels_fused_chain(self):
        kernel, rec = fake_finegrained_kernel()
        cfg = _Cfg()
        cfg._experts_implementation = "grouped_mm"
        experts = FineGrainedExperts(cfg, quantization_config=GroupFp4TestConfig())
        self.assertTrue(experts.holds_interleaved_gate_up)
        hidden = torch.randn(6, 64, dtype=torch.bfloat16)
        with patch_finegrained_kernel(kernel):
            experts(hidden, torch.randint(0, 4, (6, 2)), torch.rand(6, 2))
        call = rec.calls["moe_fused_grouped"][-1]
        self.assertIs(call.kwargs["gate_up_proj"], experts.gate_up_proj)
        self.assertEqual(call.kwargs["activation_format"], "bf16")

    def test_a_checkpoint_loads_into_the_modules_and_saves_back(self):
        checkpoint = _checkpoint(_fp8_blocks, _fp8_blocks)
        model, info, saved = _load_and_save(checkpoint, BlockFp8TestConfig())
        self.assertLoadsCleanly(info)
        self.assertSavesBack(saved, checkpoint)
        experts = model.model.layers[0].mlp.experts
        self.assertHoldsInterleaved(experts, checkpoint, "weight", experts.gate_up_proj)

    def test_a_dequantized_load_keeps_the_full_precision_modules(self):
        checkpoint = _checkpoint(_fp8_blocks, _fp8_blocks)
        quantizer = _quantizer_for(BlockFp8TestConfig(dequantize=True))
        quantizer.pre_quantized = True
        model = _MoeModel(_moe_config())
        quantizer._process_model_before_weight_loading(model)
        load_config = LoadStateDictConfig(
            weight_mapping=quantizer.update_weight_conversions(_model_mapping(), model), hf_quantizer=quantizer
        )
        info, _ = convert_and_load_state_dict_in_model(model, dict(checkpoint), load_config)
        self.assertLoadsCleanly(info)
        q_proj = model.model.layers[0].self_attn.q_proj
        self.assertIs(type(q_proj), torch.nn.Linear)
        weight = checkpoint["model.layers.0.self_attn.q_proj.weight"].float()
        scale = checkpoint["model.layers.0.self_attn.q_proj.weight_scale_inv"]
        expected = weight * scale.repeat_interleave(32, 0).repeat_interleave(32, 1)
        torch.testing.assert_close(q_proj.weight.float(), expected, rtol=1e-2, atol=1e-2)


@require_torch
class TestFormatHostTest(_ChainAssertions, unittest.TestCase):
    """`FineGrainedConfig` over the test formats: one member quantizer per format, each module in its own entry's."""

    def setUp(self):
        super().setUp()
        self.enterContext(registered_test_formats())
        self.enterContext(mock.patch.object(fg, "is_sm100", return_value=False))

    def _plan(self, **options):
        return FineGrainedConfig(dtype_plan={EXPERTS: GroupFp4TestConfig(), ".*": BlockFp8TestConfig()}, **options)

    def test_each_module_is_built_in_its_entrys_format(self):
        quantizer = _quantizer_for(self._plan())
        self.assertIsInstance(quantizer, FineGrainedHfQuantizer)
        self.assertEqual(
            sorted(type(member).__name__ for member in quantizer.members()),
            ["BlockFp8TestQuantizer", "GroupFp4TestQuantizer"],
        )
        with torch.device("meta"):
            model = _MoeModel(_moe_config())
        quantizer.replace_modules(model)
        layer = model.model.layers[0]
        self.assertEqual((layer.mlp.experts.weight_format, layer.self_attn.q_proj.weight_format), ("mxfp4", "fp8"))
        self.assertIs(quantizer.member_of(layer.mlp.experts).quantization_config, quantizer.plan()[EXPERTS])

    def test_a_plan_round_trips_through_config_json(self):
        config = self._plan(modules_to_not_convert=["lm_head"])
        read = auto.AutoQuantizationConfig.from_dict(json.loads(config.to_json_string()))
        self.assertIsInstance(read, FineGrainedConfig)
        self.assertEqual(read.to_dict(), config.to_dict())
        self.assertIsInstance(read.dtype_plan[EXPERTS], GroupFp4TestConfig)

    def test_a_plan_entry_takes_no_loading_options(self):
        with self.assertRaisesRegex(ValueError, "hold for the whole model"):
            FineGrainedConfig(dtype_plan={".*": BlockFp8TestConfig(dequantize=True)})
        with self.assertRaises(TypeError):
            FineGrainedConfig(dtype_plan={".*": SimpleNamespace()})

    def test_a_format_config_lends_its_loading_options_to_the_plan(self):
        lender = self._plan().loading_attributes_lender(BlockFp8TestConfig(dequantize=True))
        self.assertIsInstance(lender, FineGrainedConfig)
        self.assertTrue(lender.dequantize)
        self.assertEqual(list(lender.dtype_plan), [".*"])

    def test_a_mixed_checkpoint_loads_and_saves_back(self):
        checkpoint = _checkpoint(_fp4_groups, _fp8_blocks)
        model, info, saved = _load_and_save(checkpoint, self._plan())
        self.assertLoadsCleanly(info)
        self.assertSavesBack(saved, checkpoint)
        layer = model.model.layers[0]
        self.assertEqual(layer.mlp.experts.gate_up_proj.dtype, torch.int8)
        self.assertEqual(layer.self_attn.q_proj.weight.dtype, torch.float8_e4m3fn)

    def test_every_expert_companion_shards_with_its_weight(self):
        quantizer = _quantizer_for(BlockFp8TestConfig())
        model = torch.nn.Module()
        model.config = SimpleNamespace(_experts_implementation=None)
        model.layers = torch.nn.ModuleList([torch.nn.Module() for _ in range(2)])
        with torch.device("meta"):
            for layer in model.layers:
                layer.mlp = torch.nn.Module()
                layer.mlp.experts = FineGrainedExperts(_Cfg(), quantization_config=BlockFp8TestConfig(), has_bias=True)
        plan = {
            "layers.*.mlp.gate": "ep_router",
            "layers.*.mlp.experts.gate_up_proj": "grouped_gemm",
            "layers.*.mlp.experts.down_proj": "grouped_gemm",
            "layers.*.mlp.experts": "moe_tp_experts",
        }
        quantizer.update_parallel_plan(model, plan)
        for proj in ("gate_up_proj", "down_proj"):
            for name in (proj, f"{proj}_scale_inv", f"{proj}_bias"):
                self.assertEqual(_get_parameter_plan(f"layers.1.mlp.experts.{name}", plan), "grouped_gemm", name)
