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
"""Check that producer configs read without changing module targets or quantization formats."""

import unittest

import torch
from parameterized import parameterized

from transformers.quantizers.auto import AutoQuantizationConfig
from transformers.testing_utils import require_torch
from transformers.utils.quantization_config import NVFP4Config, read_compressed_tensors, read_modelopt

from ...integrations.finegrained.utils import _compressed_tensors, _modelopt, _quantizer_for


@require_torch
class BlockScaledCheckpointReaderTest(unittest.TestCase):
    """How modelopt and compressed-tensors exports read into block-scaled configs."""

    def test_a_producers_config_groups_are_normalized(self):
        """Check that modelopt config groups read as one named format.

        GLM-5.2-NVFP4 ships this config. Its `targets: ["Linear"]` means every linear.
        """
        config = _modelopt(
            config_groups={
                "group_0": {
                    "weights": {"num_bits": 4, "type": "float", "group_size": 16},
                    "input_activations": {"num_bits": 4, "type": "float", "group_size": 16, "dynamic": False},
                    "targets": ["Linear"],
                }
            },
        )
        self.assertIsInstance(config, NVFP4Config)
        self.assertIsNone(config.activation_format)
        # NVFP4's `dynamic: False` means a calibrated activation global, not a static activation scale
        self.assertEqual(config.activation_scheme, "dynamic")

    def test_an_unsupported_modelopt_group_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unsupported modelopt config_groups entry 'group_0'"):
            _modelopt(
                config_groups={
                    "group_0": {"weights": {"num_bits": 3, "type": "int", "group_size": 64}, "targets": ["Linear"]}
                },
            )

    def test_mixed_precision_maps_explicit_modules_and_leaves_others_unquantized(self):
        from transformers.quantizers.quantizers_utils import plan_entry_for
        from transformers.utils.quantization_config import FineGrainedConfig, Fp8Config

        spec = {
            "quant_algo": "MIXED_PRECISION",
            "quantized_layers": {
                "layers.8.attention.wq": {"quant_algo": "FP8"},
                "layers.9.experts": {"quant_algo": "NVFP4", "group_size": 16},
            },
        }
        for fields in (spec, {"quantization": spec}):
            with self.subTest(nested="quantization" in fields):
                config = AutoQuantizationConfig.from_dict({"quant_method": "modelopt", **fields})
                self.assertIsInstance(config, FineGrainedConfig)
                dense = plan_entry_for(config.dtype_plan, "model.layers.8.attention.wq")
                self.assertIsInstance(dense, Fp8Config)
                self.assertEqual(dense.activation_scheme, "static")
                self.assertIsNone(dense.weight_block_size)
                self.assertIsInstance(plan_entry_for(config.dtype_plan, "layers.9.experts.0.gate_proj"), NVFP4Config)
                self.assertIsNone(plan_entry_for(config.dtype_plan, "layers.90.experts"))
                self.assertIsNone(plan_entry_for(config.dtype_plan, "layers.8.attention.wk"))
                restored = AutoQuantizationConfig.from_dict(config.to_dict())
                self.assertEqual(restored.to_dict(), config.to_dict())

    def test_mixed_precision_rejects_incomplete_or_unsupported_assignments(self):
        for fields, message in (
            ({}, "non-empty quantized_layers"),
            ({"quantized_layers": {}}, "non-empty quantized_layers"),
            ({"quantized_layers": {"layers.0": {"quant_algo": "W4A16_AWQ"}}}, "layers.0"),
            ({"quantized_layers": {"layers.0": {"quant_algo": "NVFP4", "group_size": 32}}}, "group_size=16"),
        ):
            with self.subTest(fields=fields), self.assertRaisesRegex(ValueError, message):
                AutoQuantizationConfig.from_dict(
                    {"quant_method": "modelopt", "quant_algo": "MIXED_PRECISION", **fields}
                )

    def test_modelopt_runtime_transforms_are_not_dropped(self):
        nvfp4 = {"num_bits": 4, "type": "float", "group_size": 16}
        for field, value in (
            ("pre_quant_scale", True),
            ("lora_rank", 16),
            ("has_zero_point", True),
            ("weight_scale_encoding", "csf"),
        ):
            for placement in ("top", "group", "layer"):
                with self.subTest(field=field, placement=placement), self.assertRaisesRegex(ValueError, field):
                    if placement == "top":
                        _modelopt(**{field: value})
                    elif placement == "group":
                        _modelopt(config_groups={"g": {"weights": nvfp4, field: value}})
                    else:
                        AutoQuantizationConfig.from_dict(
                            {
                                "quant_method": "modelopt",
                                "quant_algo": "MIXED_PRECISION",
                                "quantized_layers": {"layers.0": {"quant_algo": "NVFP4", field: value}},
                            }
                        )
        # The compressed-tensors reader must also leave unsupported transforms to its own backend.
        self.assertIsNone(read_compressed_tensors(_compressed_tensors(nvfp4, nvfp4, pre_quant_scale=True)))

    def test_modelopt_groups_accept_exported_module_names(self):
        from transformers.quantizers.quantizers_utils import plan_entry_for

        config = _modelopt(
            config_groups={
                "g": {
                    "weights": {"num_bits": 4, "type": "float", "group_size": 16},
                    "targets": ["model.layers.0.mlp.experts"],
                }
            }
        )
        self.assertIsInstance(plan_entry_for(config.dtype_plan, "model.layers.0.mlp.experts"), NVFP4Config)
        self.assertIsNone(plan_entry_for(config.dtype_plan, "model.layers.1.mlp.experts"))

    def test_a_compressed_tensors_nvfp4_checkpoint_reads_as_nvfp4(self):
        nvfp4 = {"num_bits": 4, "type": "float", "group_size": 16, "strategy": "tensor_group"}
        config = AutoQuantizationConfig.from_dict(_compressed_tensors(nvfp4, nvfp4, ignore=["lm_head"]))
        self.assertIsInstance(config, NVFP4Config)
        self.assertEqual(config.checkpoint_format, "compressed-tensors")

    def test_compressed_tensors_checkpoints_we_do_not_serve_stay_with_it(self):
        """Check that integer, per-channel FP8 and sparse checkpoints stay with compressed-tensors."""

        int4 = {"num_bits": 4, "type": "int", "group_size": 128, "strategy": "group"}
        channel = {"num_bits": 8, "type": "float", "strategy": "channel"}
        nvfp4 = {"num_bits": 4, "type": "float", "group_size": 16}
        for checkpoint in (
            _compressed_tensors(int4, fmt="pack-quantized"),
            _compressed_tensors(channel, fmt="float-quantized"),
            _compressed_tensors(nvfp4, sparsity_config={"format": "sparse-24-bitmask"}),
        ):
            self.assertIsNone(read_compressed_tensors(checkpoint))

    def test_activations_the_weights_kernels_do_not_serve_stay_with_compressed_tensors(self):
        """An activation scheme is read only where the weights' kernels serve it: NVFP4 weights with FP8 per-token
        activations and FP8 weights with int8 activations stay compressed-tensors' own, as do
        group-32 FP8 weights whose scales are not E8M0 (MXFP8's)."""
        from transformers.utils.quantization_config import Mxfp8Config, read_compressed_tensors

        nvfp4 = {"num_bits": 4, "type": "float", "group_size": 16, "strategy": "tensor_group"}
        block = {"num_bits": 8, "type": "float", "strategy": "block", "block_structure": [128, 128]}
        fp8_token = {"num_bits": 8, "type": "float", "strategy": "token", "dynamic": True}
        int8_token = {"num_bits": 8, "type": "int", "strategy": "token", "dynamic": True}
        group_32 = {"num_bits": 8, "type": "float", "strategy": "group", "group_size": 32}
        for name, checkpoint in (
            ("nvfp4 x fp8", _compressed_tensors(nvfp4, fp8_token)),
            ("block-fp8 x int8", _compressed_tensors(block, int8_token, fmt="float-quantized")),
            (
                "group-32 fp32 scales",
                _compressed_tensors({**group_32, "scale_dtype": "torch.float32"}, group_32, fmt="float-quantized"),
            ),
        ):
            with self.subTest(name):
                self.assertIsNone(read_compressed_tensors(checkpoint))
        e8m0 = {**group_32, "scale_dtype": "torch.float8_e8m0fnu"}
        self.assertIsInstance(
            read_compressed_tensors(_compressed_tensors(e8m0, group_32, fmt="float-quantized")), Mxfp8Config
        )

    def test_unrepresentable_modelopt_targets_raise_before_loading(self):
        model = torch.nn.Module()
        model.linear = torch.nn.Linear(16, 16)
        for target in ("unknown", "experts.0.gate_proj"):
            config = AutoQuantizationConfig.from_dict(
                {
                    "quant_method": "modelopt",
                    "quant_algo": "MIXED_PRECISION",
                    "quantized_layers": {target: {"quant_algo": "NVFP4"}},
                }
            )
            with self.subTest(target=target), self.assertRaisesRegex(ValueError, "matches no supported"):
                _quantizer_for(config)._process_model_before_weight_loading(model)


class FineGrainedConfigGroupReaderTest(unittest.TestCase):
    def _groups(self, reverse=False, identical=False):
        nvfp4 = {"targets": ["Linear"], "weights": {"num_bits": 4, "type": "float", "group_size": 16}}
        fp8 = {
            "targets": ["Linear"],
            "weights": {"num_bits": 8, "type": "float", "strategy": "block", "block_structure": [128, 128]},
            "input_activations": {"num_bits": 8, "type": "float", "dynamic": True},
        }
        groups = [("nvfp4", nvfp4), ("other", nvfp4 if identical else fp8)]
        return dict(reversed(groups) if reverse else groups)

    def test_modelopt_rejects_conflicting_group_targets_in_either_order(self):
        for reverse in (False, True):
            with (
                self.subTest(reverse=reverse),
                self.assertRaisesRegex(ValueError, "Conflicting quantization configs for target"),
            ):
                read_modelopt({"quant_algo": "NVFP4", "config_groups": self._groups(reverse=reverse)})

    def test_compressed_tensors_keeps_its_backend_for_conflicting_group_targets(self):
        for reverse in (False, True):
            with self.subTest(reverse=reverse):
                self.assertIsNone(
                    read_compressed_tensors(
                        {"format": "float-quantized", "config_groups": self._groups(reverse=reverse)}
                    )
                )

    def test_identical_group_targets_can_share_a_config(self):
        for reverse in (False, True):
            for reader, fields, checkpoint_format in (
                (read_modelopt, {"quant_algo": "NVFP4"}, "modelopt"),
                (read_compressed_tensors, {"format": "float-quantized"}, "compressed-tensors"),
            ):
                with self.subTest(reverse=reverse, producer=checkpoint_format):
                    config = reader({**fields, "config_groups": self._groups(reverse=reverse, identical=True)})
                    self.assertIsInstance(config, NVFP4Config)
                    self.assertEqual(config.checkpoint_format, checkpoint_format)


NVFP4 = {"num_bits": 4, "type": "float", "group_size": 16}
MODELOPT = {"quant_method": "modelopt", "quant_algo": "NVFP4"}


@require_torch
class ProducerSkipListTest(unittest.TestCase):
    """Check that producer skip-lists skip exactly the modules they name.

    modelopt names modules as globs in `exclude_modules` or `ignore`. A trailing `.` means a subtree.
    compressed-tensors uses exact names and `re:` patterns.
    Read as plain regexes, "model.layers.1.*" would also match layers 10-19.
    """

    @parameterized.expand(
        [
            (
                "modelopt_globs",
                {**MODELOPT, "exclude_modules": ["model.layers.*.self_attn", "*.shared_experts.*"]},
                ["model.layers.0.self_attn.q_proj", "model.layers.12.self_attn", "model.layers.0.mlp.shared_experts.up_proj"],
                ["model.layers.0.mlp.experts", "model.layers.0.my_self_attn", "modelXlayers.0.self_attn"],
            ),
            (
                "modelopt_bounded_globs",
                {**MODELOPT, "ignore": ["model.layers.0*", "model.layers.1.*", "self_attn"]},
                ["model.layers.0.mlp", "model.layers.01.mlp", "model.layers.1.mlp.down_proj", "model.layers.2.self_attn.q_proj"],
                ["model.layers.16.mlp.down_proj", "model.layers.2.mlp.gate_proj", "model.my_self_attn.q_proj"],
            ),
            (
                "modelopt_trailing_separator",
                {**MODELOPT, "exclude_modules": ["layers.0.", "layers.61.", "self_attn", "lm_head"]},
                ["model.layers.0.mlp.experts", "model.layers.0", "model.layers.61.mlp.up_proj", "model.layers.61",
                 "model.layers.3.self_attn.q_proj", "lm_head"],
                ["model.layers.00.mlp.up_proj", "model.layers.610.mlp.up_proj", "model.layers.6.mlp.up_proj",
                 "model.layers.1.mlp.up_proj"],
            ),
            (
                "modelopt_merged_with_a_native_regex",
                {**MODELOPT, "modules_to_not_convert": [r"^model\.audio_tower(?:\.|$)"],
                 "ignore": ["model.visual_tower.*"], "exclude_modules": ["layers.0."]},
                ["model.audio_tower.layers.1.proj", "model.visual_tower.layers.1.proj", "model.layers.0.mlp.up_proj"],
                ["model.audio_tower_extra.proj", "model.visual_tower_extra.proj", "model.layers.00.mlp.up_proj"],
            ),
            (
                "compressed_tensors_names_and_regexes",
                _compressed_tensors(NVFP4, ignore=["model.layers.0.mlp.gate", "re:.*router$"]),
                ["model.layers.0.mlp.gate", "model.layers.3.mlp.router"],
                ["model.layers.0.mlp.gate_proj"],
            ),
        ]
    )  # fmt: skip
    def test_a_skip_list_names_exactly_its_modules(self, _, checkpoint, skipped, converted):
        from transformers.quantizers.quantizers_utils import should_convert_module

        config = AutoQuantizationConfig.from_dict(checkpoint)
        for name in skipped:
            self.assertFalse(should_convert_module(name, config.modules_to_not_convert), name)
        for name in converted:
            self.assertTrue(should_convert_module(name, config.modules_to_not_convert), name)
        restored = AutoQuantizationConfig.from_dict(config.to_dict())
        self.assertEqual(restored.modules_to_not_convert, config.modules_to_not_convert)
