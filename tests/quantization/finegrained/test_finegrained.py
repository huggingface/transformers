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
"""Tests for `FineGrainedConfig` plans that mix the format quantizers.

They cover the checkpoints that read into a plan and the conversions a plan builds.
Each format's own tests live in its folder.
"""

import re
import unittest
from types import SimpleNamespace
from unittest import mock

import torch
from torch import nn

import transformers.integrations.finegrained.layers as fg
from transformers.testing_utils import (
    require_torch,
)
from transformers.utils.quantization_config import (
    BlockScaledConfig,
    FineGrainedConfig,
    Fp8Config,
)

from ...integrations.finegrained.utils import (
    _Cfg,
    _conversion_model,
    _quantizer_for,
)


@require_torch
class FineGrainedPlanTest(unittest.TestCase):
    """`FineGrainedConfig` plans, how mixed-format checkpoints read into them, and what each entry loads."""

    def test_the_base_config_is_not_a_format(self):
        with self.assertRaises(TypeError):
            BlockScaledConfig()

    def test_a_plan_entry_is_a_format_config_without_loading_options(self):
        """Check that a plan entry with a loading option is refused. Loading options apply to the whole model."""
        for entry in (
            Fp8Config(dequantize=True),
            Fp8Config(modules_to_convert=["embed_tokens"]),
        ):
            with self.subTest(entry=entry):
                with self.assertRaises(ValueError):
                    FineGrainedConfig(dtype_plan={".*": entry})
        with self.assertRaises(ValueError):
            FineGrainedConfig(dtype_plan={".*": {"quant_method": "awq"}})
        with self.assertRaises(TypeError):
            FineGrainedConfig(dtype_plan={".*": "fp8"})

    def test_a_loaded_model_is_finished_once_whatever_its_formats(self):
        """Check that each format checks its modules and disables multi-device DeepGEMM once per load."""
        from transformers.quantizers.quantizer_finegrained import FineGrainedHfQuantizer

        for config in (
            FineGrainedConfig(dtype_plan={".*": Fp8Config()}),
            Fp8Config(),
        ):
            with self.subTest(config=type(config).__name__):
                quantizer = (
                    FineGrainedHfQuantizer(config) if isinstance(config, FineGrainedConfig) else _quantizer_for(config)
                )
                with (
                    mock.patch.object(fg, "assert_modules_are_quantized") as checked,
                    mock.patch.object(fg, "disable_deepgemm_on_multi_device") as deepgemm,
                ):
                    quantizer._process_model_after_weight_loading(nn.Module())
                self.assertEqual((checked.call_count, deepgemm.call_count), (1, 1))

    def test_the_tensor_parallel_plan_resolves_the_experts_entry_for_the_tp_glob(self):
        """Check that intra-expert TP reads the expert scale style from the plan entries under the TP glob.

        The glob `layers.*.mlp.experts` gives every layer one style. A plan that gives some layers a scale grid and
        others a per-tensor scale is refused.
        """
        from transformers.quantizers.quantizer_finegrained import FineGrainedHfQuantizer

        def tp_plan(dtype_plan):
            config = SimpleNamespace(
                base_model_tp_plan={
                    "layers.*.mlp.experts": "moe_tp_experts",
                    "layers.*.mlp.experts.gate_up_proj": "colwise",
                    "layers.*.mlp.experts.down_proj": "rowwise",
                },
                num_hidden_layers=2,
                _experts_implementation=None,
            )
            quantizer = FineGrainedHfQuantizer(FineGrainedConfig(dtype_plan=dtype_plan))
            model = nn.Module()
            model.config = config
            model.layers = nn.ModuleList([nn.Module(), nn.Module()])
            with torch.device("meta"), mock.patch.object(fg, "is_sm100", return_value=False):
                for i, layer in enumerate(model.layers):
                    layer.mlp = nn.Module()
                    entry = next(
                        value
                        for pattern, value in dtype_plan.items()
                        if re.fullmatch(pattern, f"layers.{i}.mlp.experts")
                    )
                    layer.mlp.experts = fg.FineGrainedExperts(_Cfg(), quantization_config=entry)
            plan = dict(config.base_model_tp_plan)
            quantizer.update_parallel_plan(model, plan)
            return plan

        per_tensor = Fp8Config(weight_block_size=None)
        plan = tp_plan({r"layers\.\d+\.mlp\.experts": Fp8Config(weight_block_size=(128, 128)), ".*": per_tensor})
        self.assertEqual(plan.get("layers.*.mlp.experts.gate_up_proj_scale_inv"), "moe_experts_colwise")
        self.assertEqual(plan.get("layers.*.mlp.experts.down_proj_scale_inv"), "moe_experts_rowwise")
        with self.assertRaises(ValueError):
            tp_plan({r"layers\.1\.mlp\.experts": Fp8Config(weight_block_size=(128, 128)), ".*": per_tensor})

    def test_a_format_configs_options_are_keyword_only_and_unknown_ones_reported(self):
        with self.assertRaises(TypeError):
            Fp8Config((64, 64))
        with self.assertLogs("transformers.utils.quantization_config", level="INFO") as logs:
            Fp8Config(activaton_format="bf16")
        self.assertIn("activaton_format", "".join(logs.output))


@require_torch
class FineGrainedPlanConversionTest(unittest.TestCase):
    """The conversions a plan of several formats builds."""

    def test_a_rewritten_model_converter_keeps_its_placement(self):
        """Check that a rewritten model converter keeps `force_cpu`.

        Qwen4-Exp concatenates its large n-gram table on CPU.
        Both ways the quantizer rewrites a converter must keep it there.
        """
        from transformers.core_model_loading import MergeModulelist, WeightConverter

        for dequantize in (False, True):
            with self.subTest(dequantize=dequantize):
                model_converter = WeightConverter(
                    "ngram_embeddings.*.weight", "ngram_embedding.weight", [MergeModulelist(dim=0)], force_cpu=True
                )
                quantizer = _quantizer_for(Fp8Config(dequantize=dequantize))
                (rewritten, *_) = quantizer.update_weight_conversions([model_converter], _conversion_model(quantizer))
                self.assertTrue(rewritten.force_cpu)
