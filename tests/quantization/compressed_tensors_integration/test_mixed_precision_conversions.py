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

import os
import subprocess  # nosec B404
import sys
import tempfile
import unittest

from transformers.testing_utils import require_torch
from transformers.utils import is_compressed_tensors_available, is_torch_available


if is_torch_available():
    import torch
    import torch.nn as nn
    from safetensors.torch import save_file

    from transformers import PretrainedConfig, PreTrainedModel
    from transformers.core_model_loading import Concatenate, MergeModulelist, WeightConverter
    from transformers.integrations.compressed_tensors import CompressedTensorsDequantize
    from transformers.quantizers.quantizer_compressed_tensors import (
        CompressedTensorsHfQuantizer,
        CompressedTensorsWeightConverter,
    )
    from transformers.utils.quantization_config import CompressedTensorsConfig

    class TinyGlmConfig(PretrainedConfig):
        model_type = "tiny_glm"

        def __init__(self, hidden_size=8, in_dim=8, **kwargs):
            super().__init__(**kwargs)
            self.hidden_size = hidden_size
            self.in_dim = in_dim

    class TinyGlmAttention(nn.Module):
        def __init__(self, config):
            super().__init__()
            # Linear weight shape: (out_features, in_features) = (24, 8)
            self.conv1d = nn.Linear(config.in_dim, 3 * config.hidden_size, bias=False)

    class TinyGlmExperts(nn.Module):
        def __init__(self, config):
            super().__init__()
            # 2 experts, each expert has gate_proj (8, 8) and up_proj (8, 8) concatenated along dim 1 -> (16, 8)
            # Stacked across the 2 experts along dim 0 -> (2, 16, 8)
            self.gate_up_proj = nn.Parameter(torch.empty(2, 2 * config.hidden_size, config.in_dim))

    class TinyGlmMoE(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.experts = TinyGlmExperts(config)

    class TinyGlmLayer(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.self_attn = TinyGlmAttention(config)
            self.mlp = TinyGlmMoE(config)

    class TinyGlmBase(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.layers = nn.ModuleList([TinyGlmLayer(config)])

    class TinyGlmModel(PreTrainedModel):
        config_class = TinyGlmConfig
        base_model_prefix = "model.language_model"

        def __init__(self, config):
            super().__init__(config)
            self.model = nn.Module()
            self.model.language_model = TinyGlmBase(config)
            self.post_init()

        def get_weight_conversions(self):
            return [
                WeightConverter(
                    source_patterns=[
                        r"self_attn\.q_conv1d\.weight",
                        r"self_attn\.k_conv1d\.weight",
                        r"self_attn\.v_conv1d\.weight",
                    ],
                    target_patterns=r"self_attn\.conv1d\.weight",
                    operations=[Concatenate(dim=0)],
                ),
                WeightConverter(
                    source_patterns=[
                        "mlp.experts.*.gate_proj.weight",
                        "mlp.experts.*.up_proj.weight",
                    ],
                    target_patterns="mlp.experts.gate_up_proj",
                    operations=[MergeModulelist(dim=0), Concatenate(dim=1)],
                ),
            ]

else:
    TinyGlmConfig = None
    TinyGlmModel = None


@require_torch
class MixedPrecisionCompressedTensorsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from transformers.conversion_mapping import register_checkpoint_conversion_mapping

        if TinyGlmModel is not None:
            convs = [
                WeightConverter(
                    source_patterns=[
                        r"self_attn\.q_conv1d\.weight",
                        r"self_attn\.k_conv1d\.weight",
                        r"self_attn\.v_conv1d\.weight",
                    ],
                    target_patterns=r"self_attn\.conv1d\.weight",
                    operations=[Concatenate(dim=0)],
                ),
                WeightConverter(
                    source_patterns=[
                        "mlp.experts.*.gate_proj.weight",
                        "mlp.experts.*.up_proj.weight",
                    ],
                    target_patterns="mlp.experts.gate_up_proj",
                    operations=[MergeModulelist(dim=0), Concatenate(dim=1)],
                ),
            ]
            register_checkpoint_conversion_mapping("tiny_glm", convs, overwrite=True)

    def setUp(self):
        if not is_compressed_tensors_available():
            self.skipTest("compressed-tensors is not installed.")

    def _get_fp8_mixed_config(self, dequantize=False):
        return CompressedTensorsConfig(
            format="mixed-precision",
            quantization_status="compressed",
            dequantize=dequantize,
            config_groups={
                "group_0": {
                    "format": "float-quantized",
                    "targets": [r"re:.*\.mlp\.experts\..*"],
                    "weights": {
                        "num_bits": 8,
                        "type": "float",
                        "strategy": "tensor",
                        "symmetric": True,
                        "dynamic": False,
                    },
                }
            },
        )

    def test_01_dense_fusion_and_moe_quantization_in_mixed_checkpoint(self):
        """Req A, B, C: Dense attention Q/K/V weights are recognized, fuse properly,
        and avoid random initialization while quantized MoE weights decompress correctly.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TinyGlmConfig()
            config.quantization_config = self._get_fp8_mixed_config().to_dict()
            config.save_pretrained(tmpdir)

            # Attention weights: shape (8, 8) each, concatenating to (24, 8)
            q = torch.full((8, 8), 1.0)
            k = torch.full((8, 8), 2.0)
            v = torch.full((8, 8), 3.0)
            expected_conv = torch.cat([q, k, v], dim=0)

            # MoE weights: 2 experts (0 and 1)
            gate_w0 = torch.full((8, 8), 1.0, dtype=torch.float8_e4m3fn)
            gate_s0 = torch.tensor([2.0])
            up_w0 = torch.full((8, 8), 1.0, dtype=torch.float8_e4m3fn)
            up_s0 = torch.tensor([3.0])

            gate_w1 = torch.full((8, 8), 2.0, dtype=torch.float8_e4m3fn)
            gate_s1 = torch.tensor([2.0])
            up_w1 = torch.full((8, 8), 2.0, dtype=torch.float8_e4m3fn)
            up_s1 = torch.tensor([3.0])

            gate_dense = torch.stack([gate_w0.float() * 2.0, gate_w1.float() * 2.0], dim=0)
            up_dense = torch.stack([up_w0.float() * 3.0, up_w1.float() * 3.0], dim=0)
            expected_expert = torch.cat([gate_dense, up_dense], dim=1)

            tensors = {
                "model.language_model.layers.0.self_attn.q_conv1d.weight": q,
                "model.language_model.layers.0.self_attn.k_conv1d.weight": k,
                "model.language_model.layers.0.self_attn.v_conv1d.weight": v,
                "model.language_model.layers.0.mlp.experts.0.gate_proj.weight": gate_w0,
                "model.language_model.layers.0.mlp.experts.0.gate_proj.weight_scale": gate_s0,
                "model.language_model.layers.0.mlp.experts.0.up_proj.weight": up_w0,
                "model.language_model.layers.0.mlp.experts.0.up_proj.weight_scale": up_s0,
                "model.language_model.layers.0.mlp.experts.1.gate_proj.weight": gate_w1,
                "model.language_model.layers.0.mlp.experts.1.gate_proj.weight_scale": gate_s1,
                "model.language_model.layers.0.mlp.experts.1.up_proj.weight": up_w1,
                "model.language_model.layers.0.mlp.experts.1.up_proj.weight_scale": up_s1,
            }
            save_file(tensors, os.path.join(tmpdir, "model.safetensors"))

            model = TinyGlmModel.from_pretrained(tmpdir, dtype=torch.float32)
            torch.testing.assert_close(
                model.model.language_model.layers[0].self_attn.conv1d.weight.data, expected_conv
            )
            torch.testing.assert_close(
                model.model.language_model.layers[0].mlp.experts.gate_up_proj.data, expected_expert
            )

    def test_02_fp8_scale_applied_once_and_save_reload(self):
        """Req F, I: FP8 scales are applied exactly once (not twice) and save/reload preserves weights."""
        from safetensors import safe_open

        with tempfile.TemporaryDirectory() as tmpdir, tempfile.TemporaryDirectory() as savedir:
            config = TinyGlmConfig()
            config.quantization_config = self._get_fp8_mixed_config().to_dict()
            config.save_pretrained(tmpdir)

            q = torch.full((8, 8), 1.5)
            k = torch.full((8, 8), 2.5)
            v = torch.full((8, 8), 3.5)
            expected_conv = torch.cat([q, k, v], dim=0)

            gate_w0 = torch.full((8, 8), 1.0, dtype=torch.float8_e4m3fn)
            gate_s0 = torch.tensor([4.0])
            up_w0 = torch.full((8, 8), 1.0, dtype=torch.float8_e4m3fn)
            up_s0 = torch.tensor([5.0])

            gate_w1 = torch.full((8, 8), 2.0, dtype=torch.float8_e4m3fn)
            gate_s1 = torch.tensor([4.0])
            up_w1 = torch.full((8, 8), 2.0, dtype=torch.float8_e4m3fn)
            up_s1 = torch.tensor([5.0])

            gate_dense = torch.stack([gate_w0.float() * 4.0, gate_w1.float() * 4.0], dim=0)
            up_dense = torch.stack([up_w0.float() * 5.0, up_w1.float() * 5.0], dim=0)
            expected_expert = torch.cat([gate_dense, up_dense], dim=1)

            tensors = {
                "model.language_model.layers.0.self_attn.q_conv1d.weight": q,
                "model.language_model.layers.0.self_attn.k_conv1d.weight": k,
                "model.language_model.layers.0.self_attn.v_conv1d.weight": v,
                "model.language_model.layers.0.mlp.experts.0.gate_proj.weight": gate_w0,
                "model.language_model.layers.0.mlp.experts.0.gate_proj.weight_scale": gate_s0,
                "model.language_model.layers.0.mlp.experts.0.up_proj.weight": up_w0,
                "model.language_model.layers.0.mlp.experts.0.up_proj.weight_scale": up_s0,
                "model.language_model.layers.0.mlp.experts.1.gate_proj.weight": gate_w1,
                "model.language_model.layers.0.mlp.experts.1.gate_proj.weight_scale": gate_s1,
                "model.language_model.layers.0.mlp.experts.1.up_proj.weight": up_w1,
                "model.language_model.layers.0.mlp.experts.1.up_proj.weight_scale": up_s1,
            }
            save_file(tensors, os.path.join(tmpdir, "model.safetensors"))

            model = TinyGlmModel.from_pretrained(tmpdir, dtype=torch.float32)
            torch.testing.assert_close(
                model.model.language_model.layers[0].self_attn.conv1d.weight.data, expected_conv
            )
            torch.testing.assert_close(
                model.model.language_model.layers[0].mlp.experts.gate_up_proj.data, expected_expert
            )

            # Mutation check: scale applied twice (16.0, 25.0) would fail
            double_scaled = torch.cat(
                [
                    torch.stack([gate_w0.float() * 16.0, gate_w1.float() * 16.0], dim=0),
                    torch.stack([up_w0.float() * 25.0, up_w1.float() * 25.0], dim=0),
                ],
                dim=1,
            )
            self.assertFalse(
                torch.allclose(model.model.language_model.layers[0].mlp.experts.gate_up_proj.data, double_scaled)
            )

            # Save pretrained and verify serialized keys are base unquantized names
            model.save_pretrained(savedir)
            with safe_open(os.path.join(savedir, "model.safetensors"), framework="pt") as handle:
                saved_keys = set(handle.keys())
                self.assertIn("model.language_model.layers.0.self_attn.q_conv1d.weight", saved_keys)
                self.assertIn("model.language_model.layers.0.mlp.experts.0.gate_proj.weight", saved_keys)
                self.assertNotIn("model.language_model.layers.0.self_attn.conv1d.weight", saved_keys)
                self.assertNotIn("model.language_model.layers.0.mlp.experts.gate_up_proj", saved_keys)

            # Reload from saved directory
            reloaded = TinyGlmModel.from_pretrained(savedir, dtype=torch.float32)
            torch.testing.assert_close(
                reloaded.model.language_model.layers[0].self_attn.conv1d.weight.data, expected_conv
            )
            torch.testing.assert_close(
                reloaded.model.language_model.layers[0].mlp.experts.gate_up_proj.data, expected_expert
            )

    def test_03_pack_quantized_experts_with_dense_attention(self):
        """Req G: Pack-quantized INT4 expert weights decompress correctly while dense attention fuses."""
        q_cfg = CompressedTensorsConfig(
            format="mixed-precision",
            quantization_status="compressed",
            config_groups={
                "group_0": {
                    "format": "pack-quantized",
                    "targets": [r"re:.*\.mlp\.experts\..*"],
                    "weights": {
                        "num_bits": 4,
                        "type": "int",
                        "strategy": "tensor",
                        "symmetric": True,
                        "dynamic": False,
                    },
                }
            },
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            config = TinyGlmConfig()
            config.quantization_config = q_cfg.to_dict()
            config.save_pretrained(tmpdir)

            q = torch.full((8, 8), 1.0)
            k = torch.full((8, 8), 2.0)
            v = torch.full((8, 8), 3.0)
            expected_conv = torch.cat([q, k, v], dim=0)

            # 2 experts packed int4
            gate_packed = torch.ones(8, 1, dtype=torch.int32)
            gate_scale = torch.tensor([0.5])
            gate_shape = torch.tensor([8, 8], dtype=torch.int64)

            up_packed = torch.ones(8, 1, dtype=torch.int32)
            up_scale = torch.tensor([0.25])
            up_shape = torch.tensor([8, 8], dtype=torch.int64)

            tensors = {
                "model.language_model.layers.0.self_attn.q_conv1d.weight": q,
                "model.language_model.layers.0.self_attn.k_conv1d.weight": k,
                "model.language_model.layers.0.self_attn.v_conv1d.weight": v,
                "model.language_model.layers.0.mlp.experts.0.gate_proj.weight_packed": gate_packed,
                "model.language_model.layers.0.mlp.experts.0.gate_proj.weight_scale": gate_scale,
                "model.language_model.layers.0.mlp.experts.0.gate_proj.weight_shape": gate_shape,
                "model.language_model.layers.0.mlp.experts.0.up_proj.weight_packed": up_packed,
                "model.language_model.layers.0.mlp.experts.0.up_proj.weight_scale": up_scale,
                "model.language_model.layers.0.mlp.experts.0.up_proj.weight_shape": up_shape,
                "model.language_model.layers.0.mlp.experts.1.gate_proj.weight_packed": gate_packed.clone(),
                "model.language_model.layers.0.mlp.experts.1.gate_proj.weight_scale": gate_scale.clone(),
                "model.language_model.layers.0.mlp.experts.1.gate_proj.weight_shape": gate_shape.clone(),
                "model.language_model.layers.0.mlp.experts.1.up_proj.weight_packed": up_packed.clone(),
                "model.language_model.layers.0.mlp.experts.1.up_proj.weight_scale": up_scale.clone(),
                "model.language_model.layers.0.mlp.experts.1.up_proj.weight_shape": up_shape.clone(),
            }
            save_file(tensors, os.path.join(tmpdir, "model.safetensors"))

            model = TinyGlmModel.from_pretrained(tmpdir, dtype=torch.float32)
            torch.testing.assert_close(
                model.model.language_model.layers[0].self_attn.conv1d.weight.data, expected_conv
            )
            self.assertEqual(model.model.language_model.layers[0].mlp.experts.gate_up_proj.shape, (2, 16, 8))

    def test_04_malformed_missing_and_mismatched_companions(self):
        """Req H: Missing companion scale, mismatched list lengths, and empty inputs raise explicit errors."""
        # 1. Missing companion scale during checkpoint loading
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TinyGlmConfig()
            config.quantization_config = {
                "quant_method": "compressed-tensors",
                "format": "pack-quantized",
                "quantization_status": "compressed",
                "config_groups": {
                    "group_0": {
                        "format": "pack-quantized",
                        "targets": [r"re:.*\.mlp\.experts\..*"],
                        "weights": {"num_bits": 4, "type": "int", "strategy": "tensor"},
                    }
                },
            }
            config.save_pretrained(tmpdir)

            tensors = {
                "model.language_model.layers.0.self_attn.conv1d.weight": torch.randn(24, 8),
                "model.language_model.layers.0.mlp.experts.0.gate_proj.weight_packed": torch.ones(
                    8, 1, dtype=torch.int32
                ),
                "model.language_model.layers.0.mlp.experts.0.up_proj.weight_packed": torch.ones(
                    8, 1, dtype=torch.int32
                ),
            }
            save_file(tensors, os.path.join(tmpdir, "model.safetensors"))

            with self.assertRaises(RuntimeError) as cm:
                TinyGlmModel.from_pretrained(tmpdir, dtype=torch.float32)
            self.assertIn("automatic conversion of the weights", str(cm.exception))

        # 2. Direct operation validation: mismatched length and empty list
        class DummyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = nn.Parameter(torch.empty(8, 8))

            def get_parameter(self, name):
                return self.proj

        dummy_model = DummyModel()
        quantizer_fp8 = CompressedTensorsHfQuantizer(self._get_fp8_mixed_config(), pre_quantized=True)
        op_fp8 = CompressedTensorsDequantize(quantizer_fp8)

        with self.assertRaisesRegex(ValueError, "has 2 entries but companion scale .* has 1 entries"):
            op_fp8.convert(
                {
                    "layer.0.gate_proj.weight": [torch.ones(8, 8), torch.ones(8, 8)],
                    "layer.0.gate_proj.weight_scale": [torch.tensor(1.0)],
                },
                ["layer.0.gate_proj.weight"],
                ["proj"],
                full_layer_name="proj",
                model=dummy_model,
                dtype=torch.float32,
            )

        with self.assertRaisesRegex(ValueError, "empty quantized tensor list"):
            op_fp8.convert(
                {
                    "layer.0.gate_proj.weight": [],
                    "layer.0.gate_proj.weight_scale": [],
                },
                ["layer.0.gate_proj.weight"],
                ["proj"],
                full_layer_name="proj",
                model=dummy_model,
                dtype=torch.float32,
            )

    def test_05_scoped_submodels_isolation_and_reversal(self):
        """Req D: Nested submodels with scope_prefix remain isolated and reversal preserves scope."""
        quantizer = CompressedTensorsHfQuantizer(self._get_fp8_mixed_config(), pre_quantized=True)

        class SubModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.gate_up_proj = nn.Parameter(torch.empty(2, 16, 8))

        class CompositeModel(PreTrainedModel):
            config_class = TinyGlmConfig

            def __init__(self, config):
                super().__init__(config)
                self.sub_a = SubModel()
                self.sub_b = SubModel()

            def get_weight_conversions(self):
                conv_a = WeightConverter(
                    source_patterns=[r"mlp\.experts\..*\.gate_proj\.weight", r"mlp\.experts\..*\.up_proj\.weight"],
                    target_patterns=r"mlp\.experts\.gate_up_proj",
                    operations=[Concatenate(dim=1)],
                )
                conv_a.scope_prefix = "sub_a"

                conv_b = WeightConverter(
                    source_patterns=[r"mlp\.experts\..*\.gate_proj\.weight", r"mlp\.experts\..*\.up_proj\.weight"],
                    target_patterns=r"mlp\.experts\.gate_up_proj",
                    operations=[Concatenate(dim=1)],
                )
                conv_b.scope_prefix = "sub_b"
                return [conv_a, conv_b]

        model = CompositeModel(TinyGlmConfig())
        raw_convs = model.get_weight_conversions()

        checkpoint_keys = [
            "sub_a.mlp.experts.0.gate_proj.weight",
            "sub_a.mlp.experts.0.gate_proj.weight_scale",
            "sub_a.mlp.experts.0.up_proj.weight",
            "sub_a.mlp.experts.0.up_proj.weight_scale",
            "sub_b.mlp.experts.0.gate_proj.weight",
            "sub_b.mlp.experts.0.up_proj.weight",
        ]

        updated = quantizer.update_weight_conversions(raw_convs, model, checkpoint_keys=checkpoint_keys)
        self.assertEqual(len(updated), 2)

        # sub_a is quantized
        has_deq_a = any(isinstance(op, CompressedTensorsDequantize) for op in updated[0].operations)
        self.assertTrue(has_deq_a)
        self.assertEqual(updated[0].scope_prefix, "sub_a")

        # sub_b remains dense
        has_deq_b = any(isinstance(op, CompressedTensorsDequantize) for op in updated[1].operations)
        self.assertFalse(has_deq_b)
        self.assertEqual(updated[1].scope_prefix, "sub_b")

        # Reversal maintains scope_prefix and maps back to original source keys
        rev_a = updated[0].reverse_transform()
        self.assertEqual(rev_a.scope_prefix, "sub_a")
        self.assertEqual(rev_a.target_patterns, raw_convs[0].source_patterns)

    def test_06_regex_renamings_and_negative_matching(self):
        """Req E: Regex patterns with escaped dots, renamings, and negative matching cases."""
        from transformers.core_model_loading import WeightRenaming

        quantizer = CompressedTensorsHfQuantizer(self._get_fp8_mixed_config(), pre_quantized=True)

        conv = WeightConverter(
            source_patterns=[r"\.experts.*\.w1\.weight"],
            target_patterns=r"mlp\.experts\.w1",
            operations=[Concatenate(dim=0)],
        )

        # 1. Matching case with WeightRenaming
        checkpoint_keys = ["model.layers.0.experts_raw.0.w1.weight_scale"]
        renaming = WeightRenaming(source_patterns=r"_raw", target_patterns="")
        status = quantizer.is_quantized_converter(conv, checkpoint_keys=checkpoint_keys, renamings=[renaming])
        self.assertTrue(status)

        # 2. Negative matching case: keys do not match converter
        unrelated_keys = ["model.layers.0.self_attn.q_proj.weight", "model.layers.0.self_attn.k_proj.weight"]
        status_neg = quantizer.is_quantized_converter(conv, checkpoint_keys=unrelated_keys)
        self.assertFalse(status_neg)

    def test_07_unquantized_loading_regression(self):
        """Req J: Standard unquantized model loading and converters are completely unaffected."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TinyGlmConfig()
            config.save_pretrained(tmpdir)

            q = torch.full((8, 8), 1.0)
            k = torch.full((8, 8), 2.0)
            v = torch.full((8, 8), 3.0)
            expected_conv = torch.cat([q, k, v], dim=0)

            gate_0 = torch.full((8, 8), 1.0)
            up_0 = torch.full((8, 8), 2.0)
            gate_1 = torch.full((8, 8), 3.0)
            up_1 = torch.full((8, 8), 4.0)
            expected_expert = torch.cat(
                [torch.stack([gate_0, gate_1], dim=0), torch.stack([up_0, up_1], dim=0)], dim=1
            )

            tensors = {
                "model.language_model.layers.0.self_attn.q_conv1d.weight": q,
                "model.language_model.layers.0.self_attn.k_conv1d.weight": k,
                "model.language_model.layers.0.self_attn.v_conv1d.weight": v,
                "model.language_model.layers.0.mlp.experts.0.gate_proj.weight": gate_0,
                "model.language_model.layers.0.mlp.experts.0.up_proj.weight": up_0,
                "model.language_model.layers.0.mlp.experts.1.gate_proj.weight": gate_1,
                "model.language_model.layers.0.mlp.experts.1.up_proj.weight": up_1,
            }
            save_file(tensors, os.path.join(tmpdir, "model.safetensors"))

            model = TinyGlmModel.from_pretrained(tmpdir)
            torch.testing.assert_close(
                model.model.language_model.layers[0].self_attn.conv1d.weight.data, expected_conv
            )
            torch.testing.assert_close(
                model.model.language_model.layers[0].mlp.experts.gate_up_proj.data, expected_expert
            )

    def test_08_torch_unavailable_import(self):
        """Req K: Verify quantizer_compressed_tensors and test module import when PyTorch is absent."""
        code = (
            "import sys\n"
            "sys.modules['torch'] = None\n"
            "from transformers.utils import is_torch_available\n"
            "assert not is_torch_available()\n"
            "import transformers.quantizers.quantizer_compressed_tensors as m\n"
            "assert m.CompressedTensorsWeightConverter is None\n"
            "assert m.CompressedTensorsHfQuantizer is not None\n"
            "import tests.quantization.compressed_tensors_integration.test_mixed_precision_conversions as tm\n"
            "assert tm.MixedPrecisionCompressedTensorsTest is not None\n"
            "print('OK')\n"
        )
        cmd = [sys.executable, "-c", code]
        result = subprocess.run(cmd, capture_output=True, text=True)  # nosec B603
        self.assertEqual(result.returncode, 0, msg=f"Import failed:\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}")
        self.assertIn("OK", result.stdout)

    def test_09_dequantize_true_save_and_reload(self):
        """Req I: Loading with dequantize=True preserves _IdentityOp without crashing on operations=[]
        and allows save_pretrained followed by reload.
        """
        from transformers.core_model_loading import _IdentityOp

        quantizer = CompressedTensorsHfQuantizer(CompressedTensorsConfig())
        conv = CompressedTensorsWeightConverter(
            source_patterns=["down_proj.weight", "down_proj.weight_scale"],
            target_patterns="down_proj.weight",
            operations=[CompressedTensorsDequantize(quantizer)],
            base_source_patterns="down_proj.weight",
        )
        rev = conv.reverse_transform()
        self.assertEqual(len(rev.operations), 1)
        self.assertIsInstance(rev.operations[0], _IdentityOp)

        with tempfile.TemporaryDirectory() as tmpdir, tempfile.TemporaryDirectory() as savedir:
            config = TinyGlmConfig()
            config.quantization_config = self._get_fp8_mixed_config(dequantize=True).to_dict()
            config.save_pretrained(tmpdir)

            q = torch.full((8, 8), 1.0)
            k = torch.full((8, 8), 2.0)
            v = torch.full((8, 8), 3.0)
            expected_conv = torch.cat([q, k, v], dim=0)

            gate_w0 = torch.full((8, 8), 1.0, dtype=torch.float8_e4m3fn)
            gate_s0 = torch.tensor([2.0])
            up_w0 = torch.full((8, 8), 1.0, dtype=torch.float8_e4m3fn)
            up_s0 = torch.tensor([3.0])

            gate_w1 = torch.full((8, 8), 2.0, dtype=torch.float8_e4m3fn)
            gate_s1 = torch.tensor([2.0])
            up_w1 = torch.full((8, 8), 2.0, dtype=torch.float8_e4m3fn)
            up_s1 = torch.tensor([3.0])

            gate_dense = torch.stack([gate_w0.float() * 2.0, gate_w1.float() * 2.0], dim=0)
            up_dense = torch.stack([up_w0.float() * 3.0, up_w1.float() * 3.0], dim=0)
            expected_expert = torch.cat([gate_dense, up_dense], dim=1)

            tensors = {
                "model.language_model.layers.0.self_attn.q_conv1d.weight": q,
                "model.language_model.layers.0.self_attn.k_conv1d.weight": k,
                "model.language_model.layers.0.self_attn.v_conv1d.weight": v,
                "model.language_model.layers.0.mlp.experts.0.gate_proj.weight": gate_w0,
                "model.language_model.layers.0.mlp.experts.0.gate_proj.weight_scale": gate_s0,
                "model.language_model.layers.0.mlp.experts.0.up_proj.weight": up_w0,
                "model.language_model.layers.0.mlp.experts.0.up_proj.weight_scale": up_s0,
                "model.language_model.layers.0.mlp.experts.1.gate_proj.weight": gate_w1,
                "model.language_model.layers.0.mlp.experts.1.gate_proj.weight_scale": gate_s1,
                "model.language_model.layers.0.mlp.experts.1.up_proj.weight": up_w1,
                "model.language_model.layers.0.mlp.experts.1.up_proj.weight_scale": up_s1,
            }
            save_file(tensors, os.path.join(tmpdir, "model.safetensors"))

            model = TinyGlmModel.from_pretrained(tmpdir, dtype=torch.float32)
            torch.testing.assert_close(
                model.model.language_model.layers[0].self_attn.conv1d.weight.data, expected_conv
            )
            torch.testing.assert_close(
                model.model.language_model.layers[0].mlp.experts.gate_up_proj.data, expected_expert
            )

            model.save_pretrained(savedir)
            reloaded = TinyGlmModel.from_pretrained(savedir, dtype=torch.float32)
            torch.testing.assert_close(
                reloaded.model.language_model.layers[0].self_attn.conv1d.weight.data, expected_conv
            )
            torch.testing.assert_close(
                reloaded.model.language_model.layers[0].mlp.experts.gate_up_proj.data, expected_expert
            )
