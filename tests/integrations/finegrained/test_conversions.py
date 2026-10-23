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
"""Tests for the format-independent `finegrained` conversion ops.

They cover the layout ops, the per-module expert ops and a core chain combined with on-the-fly quantization.
Each format's checkpoint layouts are tested in its own folder. No kernel or GPU is needed.
"""

import unittest
from unittest import mock

import torch

import transformers.integrations.finegrained.layers as fg
from transformers import PreTrainedConfig
from transformers.core_model_loading import (
    Concatenate,
    WeightConverter,
    revert_weight_conversion,
)
from transformers.quantizers import auto
from transformers.quantizers.block_scaled_utils import BlockScaledQuantizer
from transformers.testing_utils import require_torch
from transformers.utils import quantization_config as qc
from transformers.utils.quantization_config import (
    BlockScaledConfig,
    QuantizationMethod,
)

from ...quantization.finegrained.test_machinery import GroupFp4TestConfig, GroupFp4TestQuantizer
from .utils import (
    _Cfg,
    _format_config,
    _quantizer_for,
    fake_finegrained_kernel,
    patch_finegrained_kernel,
)


class Mxfp8LikeTestConfig(BlockScaledConfig):
    """Test-only stand-in for MXFP8, which ships in a later PR: FP8 values with a UE8M0 scale per 32."""

    method = QuantizationMethod.MXFP8
    activation_formats = (None, "mxfp8", "mxfp4", "bf16")
    weight_block_size = (1, 32)
    scale_fmt = "ue8m0"


class Mxfp8LikeTestQuantizer(BlockScaledQuantizer):
    pass


@require_torch
class FineGrainedLayoutOpsTest(unittest.TestCase):
    """Check the scale layout the experts hold and the ops that fill it.

    MX formats on SM100 under a Triton backend hold swizzled 5-D scales. Every other case holds affine scales.
    `FineGrainedSwizzleScales` fills the swizzled layout and its reverse restores the checkpoint grid.
    """

    E, HIDDEN, INTER = 4, 256, 256  # both projections have whole 128-row blocks and two K bands

    def setUp(self):
        super().setUp()
        # MXFP8 and MXFP4 ship in later PRs: test-only stand-ins under their `quant_method`
        self.enterContext(
            mock.patch.dict(qc.BLOCK_SCALED_CONFIGS, {"mxfp8": Mxfp8LikeTestConfig, "mxfp4": GroupFp4TestConfig})
        )
        self.enterContext(
            mock.patch.dict(
                auto.AUTO_QUANTIZER_MAPPING, {"mxfp8": Mxfp8LikeTestQuantizer, "mxfp4": GroupFp4TestQuantizer}
            )
        )

    def _experts(self, weight_format, activation_format=None, impl="grouped_mm", sm100=True):
        cfg = _Cfg()
        cfg.hidden_size, cfg.intermediate_size, cfg.num_local_experts = self.HIDDEN, self.INTER, self.E
        cfg._experts_implementation = impl
        with (
            mock.patch.object(fg, "is_sm100", return_value=sm100),
            mock.patch("torch.cuda.is_available", return_value=True),
        ):
            experts = fg.FineGrainedExperts(
                cfg,
                quantization_config=_format_config(
                    weight_format,
                    activation_format=activation_format,
                    # The group formats fix their own block
                    **({"weight_block_size": (128, 128)} if weight_format == "fp8" else {}),
                ),
            )
        model = torch.nn.Module()
        model.config = cfg
        model.experts = experts
        return model, experts

    def test_module_holds_swizzled_scales_for_the_scaled_mma_chain(self):
        _, experts = self._experts("mxfp8")
        self.assertEqual(experts.gate_up_proj_scale_inv.shape, (4, 4, 2, 2, 256))  # (E, 512/128, (256/32)/4, 2, 256)
        self.assertEqual(experts.down_proj_scale_inv.shape, (4, 2, 2, 2, 256))
        self.assertEqual(experts.gate_up_proj_scale_inv.dtype, torch.float8_e8m0fnu)
        self.assertTrue(experts.holds_interleaved_gate_up)
        # Every MX format holds UE8M0 scales, whatever `scale_fmt` says
        _, experts = self._experts("mxfp4", activation_format="bf16")
        self.assertEqual(experts.gate_up_proj_scale_inv.dtype, torch.float8_e8m0fnu)

    def test_affine_off_sm100_and_weight_only(self):
        for kw in (
            {"weight_format": "mxfp8", "sm100": False},
            {"weight_format": "mxfp4", "activation_format": "bf16"},  # W4A16 reads scales per group affinely
        ):
            with self.subTest(**kw):
                _, experts = self._experts(**kw)
                self.assertEqual(
                    experts.gate_up_proj_scale_inv.shape,
                    (4, 512, 256 // (16 if kw["weight_format"] == "nvfp4" else 32)),
                )

    def _op_kernel(self):
        kernel, _ = fake_finegrained_kernel()
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
        grid = torch.zeros(4, 512, 8, dtype=torch.float8_e8m0fnu)
        weight = torch.zeros(4, 256, 256, dtype=torch.float8_e4m3fn)
        with patch_finegrained_kernel(kernel):
            # The loader passes a single target under its pattern and the full name separately
            out = op.convert(
                {"mlp.experts.gate_up_proj": grid}, model=model, full_layer_name="experts.gate_up_proj_scale_inv"
            )
            self.assertEqual(out["mlp.experts.gate_up_proj"].shape, (4, 4, 2, 2, 256))
            # The weight of the same converter passes through
            out = op.convert({"mlp.experts.gate_up_proj": weight}, model=model, full_layer_name="experts.gate_up_proj")
            self.assertIs(out["mlp.experts.gate_up_proj"], weight)
            # A multi-tensor converter names its outputs in full
            out = op.convert(
                {
                    "experts.down_proj": weight,
                    "experts.down_proj_scale_inv": torch.zeros(4, 256, 8, dtype=torch.float8_e8m0fnu),
                },
                model=model,
                full_layer_name="experts.down_proj",
            )
            self.assertEqual(out["experts.down_proj_scale_inv"].shape, (4, 2, 2, 2, 256))
            self.assertIs(out["experts.down_proj"], weight)
            # A module that holds affine scales keeps them
            affine_model, _ = self._experts("mxfp4", activation_format="bf16")
            out = op.convert(
                {"mlp.experts.gate_up_proj": grid},
                model=affine_model,
                full_layer_name="experts.gate_up_proj_scale_inv",
            )
            self.assertIs(out["mlp.experts.gate_up_proj"], grid)

    def test_fused_dense_scales_save_without_changing_the_runtime_dtype(self):
        from transformers.integrations.finegrained.conversions import FineGrainedDeserialize

        config = Mxfp8LikeTestConfig()
        model = torch.nn.Module()
        model.config = PreTrainedConfig()
        model.mlp = torch.nn.Module()
        model.mlp.gate_up_proj = fg.FineGrainedLinear(64, 64, quantization_config=config)
        scales = torch.arange(120, 128, dtype=torch.uint8).repeat(16, 1).reshape(64, 2)
        model._weight_conversions = [
            WeightConverter(
                [r"\.gate_proj.weight_scale_inv", r"\.up_proj.weight_scale_inv"],
                r"\.gate_up_proj.weight_scale_inv",
                [Concatenate(dim=0), FineGrainedDeserialize(_quantizer_for(config))],
            )
        ]
        saved = revert_weight_conversion(
            model, {"mlp.gate_up_proj.weight_scale_inv": scales.view(torch.float8_e8m0fnu)}
        )
        for name, expected in zip(("gate_proj", "up_proj"), scales.chunk(2)):
            actual = saved[f"mlp.{name}.weight_scale_inv"]
            self.assertEqual(actual.dtype, torch.float8_e8m0fnu)
            self.assertTrue(torch.equal(actual.view(torch.uint8), expected))

    def test_reverse_op_restores_the_affine_grid_from_the_artifact_alone(self):
        from transformers.integrations.finegrained.conversions import FineGrainedSwizzleScales

        kernel = self._op_kernel()
        reverse = FineGrainedSwizzleScales(hf_quantizer=None).reverse_op
        artifact = torch.zeros(4, 2, 2, 2, 256, dtype=torch.float8_e8m0fnu)
        affine = torch.zeros(4, 256, 8, dtype=torch.float8_e8m0fnu)
        with patch_finegrained_kernel(kernel):
            out = reverse.convert({"experts.gate_up_proj_scale_inv": artifact}, model=None, full_layer_name="x")
            self.assertEqual(out["experts.gate_up_proj_scale_inv"].shape, (4, 256, 8))
            out = reverse.convert({"experts.gate_up_proj_scale_inv": affine}, model=None, full_layer_name="x")
            self.assertIs(out["experts.gate_up_proj_scale_inv"], affine)

    def test_a_key_under_its_fused_name_takes_its_modules_ops_by_its_full_name(self):
        """Check a key already under its fused name, like GPT-OSS's `gate_up_proj_bias`.

        The loader runs the module's ops in a one-to-one converter. The ops must return the tensor under the same key.
        """
        model, experts = self._experts("mxfp8")
        experts.quantization_config = Mxfp8LikeTestConfig()  # as set by the module replacement
        quantizer = _quantizer_for(Mxfp8LikeTestConfig())
        for key, shape in (("experts.gate_up_proj_bias", (4, 512)), ("experts.down_proj_scale_inv", (4, 256, 8))):
            converter = WeightConverter(key, key, operations=quantizer.get_param_ops(model, key))
            tensor = torch.zeros(shape, dtype=torch.float8_e8m0fnu if "scale" in key else torch.float32)
            converter.add_tensor(key, key, key, lambda t=tensor: t)
            kernel = self._op_kernel()
            with patch_finegrained_kernel(kernel):
                out = converter.convert(key, model=model)
            self.assertEqual(list(out), [key])

    def test_interleave_op_follows_the_layer_it_fills(self):
        """Check that the op interleaves gate and up only for experts that hold them interleaved.

        Quantized experts get the interleaved order and the reverse restores the stack.
        Already interleaved checkpoints, Mega MoE and full-precision experts keep the stack.
        """
        from transformers.integrations.finegrained.conversions import FineGrainedInterleaveGateUp

        stacked = torch.arange(2 * 6 * 4, dtype=torch.float32).reshape(2, 6, 4)  # rows [g0,g1,g2,u0,u1,u2]
        interleaved = stacked[:, [0, 3, 1, 4, 2, 5]]
        full_precision = torch.nn.Module()
        full_precision.gate_up_proj = torch.nn.Parameter(torch.empty(2, 6, 4))
        full_precision.down_proj = torch.nn.Parameter(torch.empty(2, 4, 3))
        model = torch.nn.Module()
        model.layers = torch.nn.ModuleList([torch.nn.Module(), torch.nn.Module()])
        model.layers[0].experts = full_precision
        model.layers[1].experts = quantized = self._experts("mxfp8")[1]
        op = FineGrainedInterleaveGateUp(hf_quantizer=None)

        for layer, expected in ((0, stacked), (1, interleaved)):
            key = f"layers.{layer}.experts.gate_up_proj"
            out = op.convert({key: stacked}, model=model, full_layer_name=key)[key]
            torch.testing.assert_close(out, expected)
            saved_key = f"layers.{layer}.experts.*.gate_proj.weight"
            back = op.reverse_op.convert({saved_key: out}, model=model, full_layer_name=saved_key)[saved_key]
            torch.testing.assert_close(back, stacked)

        key = "layers.1.experts.gate_up_proj"
        quantized.is_concatenated = False
        self.assertIs(op.convert({key: stacked}, model=model, full_layer_name=key)[key], stacked)
        quantized.is_concatenated, quantized.holds_interleaved_gate_up = True, False
        self.assertIs(op.convert({key: stacked}, model=model, full_layer_name=key)[key], stacked)

    def test_interleave_op_stacks_an_interleaved_checkpoint_for_stacked_experts(self):
        """Check that stacked experts stack an interleaved checkpoint on load and interleave it back on save."""
        from transformers.integrations.finegrained.conversions import FineGrainedInterleaveGateUp

        stacked = torch.arange(2 * 6 * 4, dtype=torch.float32).reshape(2, 6, 4)
        interleaved = stacked[:, [0, 3, 1, 4, 2, 5]]
        model, experts = self._experts("mxfp8")
        # block-FP8 experts off SM100 hold gate|up stacked
        experts.is_concatenated, experts.holds_interleaved_gate_up = False, False
        op = FineGrainedInterleaveGateUp(hf_quantizer=None)
        key = "experts.gate_up_proj"
        loaded = op.convert({key: interleaved}, model=model, full_layer_name=key)[key]
        torch.testing.assert_close(loaded, stacked)
        torch.testing.assert_close(
            op.reverse_op.convert({key: loaded}, model=model, full_layer_name=key)[key], interleaved
        )

    def test_ops_refuse_a_list_of_several_tensors(self):
        """Check that an op unwrapping a one-element list refuses a list of several tensors."""
        from transformers.integrations.finegrained.conversions import (
            FineGrainedQuantize,
        )

        tensor = torch.zeros(2, 4, 32, 16, dtype=torch.uint8)
        for op in (FineGrainedQuantize(None),):
            with self.subTest(op=type(op).__name__), self.assertRaises(ValueError):
                op.convert({"experts.down_proj": [tensor, tensor]})
