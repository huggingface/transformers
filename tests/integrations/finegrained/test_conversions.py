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
from types import SimpleNamespace
from unittest import mock

import torch

import transformers.integrations.finegrained.layers as fg
from transformers import PreTrainedConfig, PreTrainedModel
from transformers.core_model_loading import (
    Chunk,
    Concatenate,
    PermuteForRope,
    WeightConverter,
    convert_and_load_state_dict_in_model,
    revert_weight_conversion,
)
from transformers.integrations.moe import use_experts_implementation
from transformers.modeling_utils import LoadStateDictConfig
from transformers.quantizers import auto
from transformers.quantizers.block_scaled_utils import BlockScaledQuantizer
from transformers.testing_utils import require_torch
from transformers.utils import quantization_config as qc
from transformers.utils.quantization_config import (
    BlockScaledConfig,
    Fp8Config,
    QuantizationMethod,
)

from ...quantization.finegrained.test_machinery import GroupFp4TestConfig, GroupFp4TestQuantizer
from .utils import (
    E,
    H,
    I,
    _Cfg,
    _format_config,
    _load_and_save,
    _moe_config,
    _MoeModel,
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


@use_experts_implementation(is_transposed=True)
class _TransposedExperts(torch.nn.Module):
    """Stacked experts stored as `(E, in, out)`, like GPT-OSS."""

    def __init__(self, config):
        super().__init__()
        self.gate_up_proj = torch.nn.Parameter(torch.empty(E, H, 2 * I))
        self.down_proj = torch.nn.Parameter(torch.empty(E, I, H))


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
        # Block-FP8 has one scale per 128x128 block and nothing to swizzle
        _, experts = self._experts("fp8")
        self.assertEqual(experts.gate_up_proj_scale_inv.shape, (4, 4, 2))

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
            FineGrainedInputScalesSplit,
            FineGrainedQuantize,
        )

        tensor = torch.zeros(2, 4, 32, 16, dtype=torch.uint8)
        for op in (
            FineGrainedInputScalesSplit(None),
            FineGrainedQuantize(None),
        ):
            with self.subTest(op=type(op).__name__), self.assertRaises(ValueError):
                op.convert({"experts.down_proj": [tensor, tensor]})

    def test_deepgemm_backends_hold_gate_up_stacked(self):
        """DeepGEMM reads the scale grid row-major, so it needs gate and up stacked."""
        for impl in ("deepgemm", "deepgemm_megamoe"):
            _, experts = self._experts("fp8", impl=impl)
            self.assertFalse(experts.holds_interleaved_gate_up, impl)

    def test_on_the_fly_quantization_still_interleaves_expert_biases(self):
        from transformers.integrations.finegrained.conversions import FineGrainedDeserialize

        model, experts = self._experts("fp8", sm100=False)
        quantizer = _quantizer_for(experts.quantization_config)
        quantizer.pre_quantized = False
        bias = torch.arange(4 * 256, dtype=torch.float32).reshape(4, 256)
        key = "experts.gate_up_proj_bias"
        actual = FineGrainedDeserialize(quantizer).convert({key: bias}, model=model, full_layer_name=key)[key]
        gate, up = bias.chunk(2, dim=1)
        torch.testing.assert_close(actual, torch.stack((gate, up), dim=-1).flatten(1))


@require_torch
class FineGrainedLoaderChainTest(unittest.TestCase):
    """Check conversion chains that combine core ops with fine-grained ops in one load."""

    def test_on_the_fly_quantization_blocks_each_projection_whole(self):
        """Check that each quantization block holds rows of one projection only.

        The gate is much smaller than up, so a block shared by both would lose the gate values.
        Cover a per-expert checkpoint and a transposed `(E, H, 2I)` stack.
        """
        from transformers.integrations.finegrained.conversions import FineGrainedDequantize

        torch.manual_seed(0)
        gate, up, down = torch.randn(E, I, H) * 1e-5, torch.randn(E, I, H), torch.randn(E, H, I)
        dense = {"model.layers.0.self_attn.q_proj.weight": torch.randn(H, H)}
        per_expert = {
            f"model.layers.0.mlp.experts.{expert}.{proj}.weight": tensor[expert]
            for expert in range(E)
            for proj, tensor in (("gate_proj", gate), ("up_proj", up), ("down_proj", down))
        }
        transposed = {
            "model.layers.0.mlp.experts.gate_up_proj": torch.cat([gate, up], dim=1).transpose(1, 2).contiguous(),
            "model.layers.0.mlp.experts.down_proj": down.transpose(1, 2).contiguous(),
        }
        config = Fp8Config(weight_block_size=(32, 32))
        for layout, checkpoint, mapping in (("concatenated", per_expert, None), ("transposed", transposed, [])):
            with self.subTest(layout=layout):
                model = _MoeModel(_moe_config())
                if layout == "transposed":
                    model.model.layers[0].mlp.experts = _TransposedExperts(model.config)
                model, info, saved = _load_and_save(
                    {**dense, **checkpoint}, config, model, mapping, pre_quantized=False
                )
                self.assertEqual(info.conversion_errors, {})
                experts = model.model.layers[0].mlp.experts
                reloaded, info, _ = _load_and_save(saved, config)
                self.assertEqual((info.missing_keys, info.unexpected_keys, info.conversion_errors), (set(), set(), {}))
                restored = reloaded.model.layers[0].mlp.experts
                for name in ("gate_up_proj", "gate_up_proj_scale_inv", "down_proj", "down_proj_scale_inv"):
                    torch.testing.assert_close(
                        getattr(restored, name).float(), getattr(experts, name).float(), rtol=0, atol=0
                    )

                self.assertTrue(experts.holds_interleaved_gate_up)
                for half, expected in ((0, gate), (1, up)):
                    held = FineGrainedDequantize(None)._dequantize_one(
                        experts.gate_up_proj[:, half::2], experts.gate_up_proj_scale_inv[:, half::2], torch.float32
                    )
                    error = ((held - expected).norm() / expected.norm()).item()
                    self.assertLess(error, 0.1, f"half {half}")

    def test_a_chunked_rope_permute_quantizes_only_the_finegrained_target(self):
        from transformers.integrations.finegrained import FineGrainedLinear
        from transformers.integrations.finegrained.conversions import FineGrainedDequantize, FineGrainedQuantize

        n_heads = 2
        head_dim = 4
        in_dim = 4
        out_dim = n_heads * head_dim
        block_size = (4, 4)

        class RopeProjector(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.zeros(out_dim, in_dim))

        class RopeSelfAttn(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.q_proj = FineGrainedLinear(
                    in_dim, out_dim, quantization_config=Fp8Config(weight_block_size=block_size)
                )
                self.k_proj = RopeProjector()
                self.v_proj = RopeProjector()

        class RopeLayer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.self_attn = RopeSelfAttn()

        class RopeModel(PreTrainedModel):
            base_model_prefix = "model"

            def __init__(self, config):
                super().__init__(config)
                self.layers = torch.nn.ModuleList([RopeLayer()])
                self.post_init()

            def _init_weights(self, module):
                pass  # every weight comes from the state dict

        config = PreTrainedConfig()
        config.num_attention_heads = n_heads
        model = RopeModel(config)

        raw_q = torch.tensor(
            [
                [1.0, -1.0, 1.0, -1.0],
                [0.5, -0.5, 0.5, -0.5],
                [-1.0, 1.0, -1.0, 1.0],
                [-0.5, 0.5, -0.5, 0.5],
                [1.0, 1.0, -1.0, -1.0],
                [0.5, 0.5, -0.5, -0.5],
                [-1.0, -1.0, 1.0, 1.0],
                [-0.5, -0.5, 0.5, 0.5],
            ],
            dtype=torch.float32,
        )
        raw_k = torch.arange(out_dim * in_dim, dtype=torch.float32).reshape(out_dim, in_dim)
        raw_v = torch.arange(out_dim * in_dim, dtype=torch.float32).reshape(out_dim, in_dim) + 100.0
        raw_qkv = torch.cat([raw_q, raw_k, raw_v], dim=0)
        state_dict = {"layers.0.self_attn.qkv_proj.weight": raw_qkv.clone()}

        quantizer_cls = type(
            "FineGrainedHfQuantizer",
            (),
            {
                "__init__": lambda self, bs=block_size: setattr(
                    self, "quantization_config", SimpleNamespace(weight_block_size=bs, scale_fmt="float")
                ),
                "param_needs_quantization": lambda self, _model, param_name: param_name.endswith("q_proj.weight"),
                "get_quantize_ops": lambda self: FineGrainedQuantize(self),
                "get_param_ops": lambda self, _model, _name: [],
                "pre_quantized": False,
            },
        )
        quantizer = quantizer_cls()

        weight_mapping = [
            WeightConverter(
                "self_attn.qkv_proj.weight",
                [
                    "self_attn.q_proj.weight",
                    "self_attn.k_proj.weight",
                    "self_attn.v_proj.weight",
                ],
                operations=[Chunk(dim=0), PermuteForRope(permute_layer_names=["q_proj", "k_proj"])],
            )
        ]
        load_config = LoadStateDictConfig(weight_mapping=weight_mapping, hf_quantizer=quantizer)
        loading_info, _ = convert_and_load_state_dict_in_model(model, state_dict, load_config)

        self.assertEqual(loading_info.missing_keys, set())
        self.assertEqual(loading_info.unexpected_keys, set())
        self.assertEqual(loading_info.mismatched_keys, set())
        self.assertEqual(loading_info.conversion_errors, {})

        permute_op = PermuteForRope(permute_layer_names=["q_proj", "k_proj"])
        permute_op.config = model.config
        expected_q = permute_op._apply(raw_q)
        expected_k = permute_op._apply(raw_k)
        expected_v = raw_v  # V has no RoPE

        model_state = model.state_dict()
        self.assertFalse(torch.allclose(raw_k, expected_k))
        torch.testing.assert_close(model_state["layers.0.self_attn.k_proj.weight"], expected_k)
        torch.testing.assert_close(model_state["layers.0.self_attn.v_proj.weight"], expected_v)

        q_weight_key = "layers.0.self_attn.q_proj.weight"
        scale_key = "layers.0.self_attn.q_proj.weight_scale_inv"
        self.assertIn(scale_key, model_state)
        self.assertEqual(model_state[q_weight_key].dtype, torch.float8_e4m3fn)
        self.assertEqual(model_state[q_weight_key].shape, torch.Size((out_dim, in_dim)))
        self.assertEqual(model_state[scale_key].dtype, torch.float32)
        self.assertEqual(
            model_state[scale_key].shape,
            torch.Size((out_dim // block_size[0], in_dim // block_size[1])),
        )

        dequantized_q = FineGrainedDequantize(None)._dequantize_one(
            model_state[q_weight_key], model_state[scale_key], output_dtype=torch.float32
        )
        torch.testing.assert_close(dequantized_q, expected_q, rtol=1e-2, atol=1e-2)
