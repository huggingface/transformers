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
"""Tests for `FineGrainedHfQuantizer`, run on two test-only quantizers defined here."""

import json
import unittest
from types import SimpleNamespace
from unittest import mock

from transformers import LlamaConfig, LlamaForCausalLM
from transformers.core_model_loading import WeightRenaming
from transformers.quantizers import auto
from transformers.quantizers.base import HfQuantizer
from transformers.quantizers.quantizer_finegrained import FineGrainedHfQuantizer, _dedupe_conversions
from transformers.quantizers.quantizers_utils import (
    get_module_from_name,
    plan_entry_for,
    should_convert_module,
    skip_pattern,
)
from transformers.testing_utils import require_torch
from transformers.utils import is_torch_available
from transformers.utils.quantization_config import FineGrainedConfig, QuantizationConfigMixin


if is_torch_available():
    import torch


class TagTestConfig(QuantizationConfigMixin):
    """Test-only method that tags the linears it covers."""

    quant_method = "test_tag"

    def __init__(self, tag="a", modules_to_not_convert=None, dequantize=False, **kwargs):
        self.quant_method = type(self).quant_method
        self.tag = tag
        self.modules_to_not_convert = modules_to_not_convert
        self.dequantize = dequantize


class OtherTagTestConfig(TagTestConfig):
    quant_method = "test_other_tag"


class TagTestQuantizer(HfQuantizer):
    def _process_model_before_weight_loading(self, model, **kwargs):
        skipped = self.get_modules_to_not_convert(
            model, self.quantization_config.modules_to_not_convert, model._keep_in_fp32_modules
        )
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear) and should_convert_module(name, skipped):
                module.tag = self.quantization_config.tag

    def param_needs_quantization(self, model, param_name, **kwargs):
        module, name = get_module_from_name(model, param_name)
        return name == "weight" and getattr(module, "tag", None) == self.quantization_config.tag

    def get_new_dtype(self, converter, param, dtype):
        return torch.float16 if self.quantization_config.tag == "half" else dtype

    def get_checkpoint_renamings(self):
        return [WeightRenaming(r"\.scale$", ".weight_scale")]

    def is_serializable(self):
        return True

    @property
    def is_trainable(self):
        return self.quantization_config.tag != "frozen"


def _tags(model):
    return {name: getattr(module, "tag", None) for name, module in model.named_modules() if name.endswith("proj")}


@require_torch
class FineGrainedHostTest(unittest.TestCase):
    def setUp(self):
        super().setUp()
        configs = {"test_tag": TagTestConfig, "test_other_tag": OtherTagTestConfig}
        quantizers = dict.fromkeys(configs, TagTestQuantizer)
        self.enterContext(mock.patch.dict(auto.AUTO_QUANTIZATION_CONFIG_MAPPING, configs))
        self.enterContext(mock.patch.dict(auto.AUTO_QUANTIZER_MAPPING, quantizers))
        config = LlamaConfig(
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            vocab_size=32,
        )
        with torch.device("meta"):
            self.model = LlamaForCausalLM(config)

    def _load(self, dtype_plan, **options):
        quantizer = FineGrainedHfQuantizer(FineGrainedConfig(dtype_plan=dtype_plan, **options))
        quantizer._process_model_before_weight_loading(self.model)
        return quantizer

    def test_each_module_loads_through_its_entrys_quantizer(self):
        quantizer = self._load({"self_attn": OtherTagTestConfig(tag="attn"), ".*": TagTestConfig(tag="rest")})
        tags = _tags(self.model)
        self.assertEqual(tags["model.layers.1.self_attn.q_proj"], "attn")
        self.assertEqual(tags["model.layers.1.mlp.down_proj"], "rest")
        self.assertIsNone(getattr(self.model.lm_head, "tag", None))
        self.assertTrue(quantizer.param_needs_quantization(self.model, "model.layers.0.self_attn.k_proj.weight"))
        self.assertTrue(quantizer.param_needs_quantization(self.model, "model.layers.0.mlp.up_proj.weight"))
        self.assertFalse(quantizer.param_needs_quantization(self.model, "model.layers.0.input_layernorm.weight"))

    def test_a_plan_without_a_catch_all_keeps_the_other_modules(self):
        self._load({r"layers\.1\.mlp": TagTestConfig(tag="mlp")})
        tags = _tags(self.model)
        self.assertEqual(tags["model.layers.1.mlp.gate_proj"], "mlp")
        self.assertIsNone(tags["model.layers.0.mlp.gate_proj"])
        self.assertIsNone(tags["model.layers.1.self_attn.q_proj"])

    def test_a_none_entry_holds_for_every_member(self):
        self._load({r"^model\.layers\.0\.": None, "self_attn": OtherTagTestConfig(tag="attn"), ".*": TagTestConfig()})
        tags = _tags(self.model)
        self.assertIsNone(tags["model.layers.0.self_attn.q_proj"])
        self.assertIsNone(tags["model.layers.0.mlp.up_proj"])
        self.assertEqual(tags["model.layers.1.self_attn.q_proj"], "attn")

    def test_the_saved_config_keeps_the_plan_as_passed(self):
        config = FineGrainedConfig(dtype_plan={"self_attn": OtherTagTestConfig(), ".*": TagTestConfig()})
        before = config.to_dict()
        quantizer = FineGrainedHfQuantizer(config)
        quantizer._process_model_before_weight_loading(self.model)
        self.assertEqual(config.to_dict(), before)
        read = auto.AutoQuantizationConfig.from_dict(json.loads(config.to_json_string()))
        self.assertIsInstance(read.dtype_plan["self_attn"], OtherTagTestConfig)

    def test_equal_entries_share_one_member(self):
        quantizer = self._load({"layers.0": TagTestConfig(), "layers.1": TagTestConfig()})
        self.assertEqual(len(quantizer.members), 1)

    def test_a_member_that_falls_back_dequantizes_the_whole_model(self):
        quantizer = FineGrainedHfQuantizer(
            FineGrainedConfig(dtype_plan={"self_attn": OtherTagTestConfig(), ".*": TagTestConfig()})
        )
        falling_back = quantizer.members[0]

        def fall_back(*args, **kwargs):
            falling_back.quantization_config.dequantize = True

        with mock.patch.object(falling_back, "validate_environment", side_effect=fall_back):
            quantizer.validate_environment()
        self.assertTrue(quantizer.quantization_config.dequantize)
        self.assertTrue(all(member.quantization_config.dequantize for member in quantizer.members))

    def test_a_hook_with_one_answer_needs_every_member(self):
        quantizer = FineGrainedHfQuantizer(
            FineGrainedConfig(dtype_plan={"self_attn": OtherTagTestConfig(tag="frozen"), ".*": TagTestConfig()})
        )
        self.assertTrue(quantizer.is_serializable())
        self.assertFalse(quantizer.is_trainable)

    def test_a_renaming_every_member_adds_loads_once(self):
        quantizer = self._load({"self_attn": OtherTagTestConfig(), ".*": TagTestConfig()})
        conversions = quantizer.update_weight_conversions([], self.model)
        self.assertEqual([conversion.source_patterns for conversion in conversions], [[r"\.scale$"]])

    def test_members_adding_the_same_conversion_keep_it_once(self):
        model_converter = WeightRenaming("a", "b")
        first, again = WeightRenaming("x", "y"), WeightRenaming("x", "y")
        merged, shared = _dedupe_conversions([model_converter, first, again], keep=[model_converter])
        self.assertEqual(merged, [model_converter, first])
        self.assertEqual(shared, [first])
        with self.assertRaisesRegex(ValueError, "differently"):
            _dedupe_conversions([first, WeightRenaming("x", "z")])

    def test_an_entrys_config_lends_its_loading_options_to_the_plan(self):
        plan = FineGrainedConfig(dtype_plan={"self_attn": OtherTagTestConfig(), ".*": TagTestConfig()})
        source = plan.loading_attributes_source(TagTestConfig(dequantize=True, modules_to_not_convert=["lm_head"]))
        self.assertEqual((source.dequantize, source.dtype_plan), (True, {skip_pattern("lm_head"): None}))
        tags_only = FineGrainedConfig(dtype_plan={".*": TagTestConfig()})
        self.assertIsNone(tags_only.loading_attributes_source(OtherTagTestConfig()))

    def test_a_converter_loads_in_the_dtype_of_the_member_that_added_it(self):
        quantizer = self._load({"self_attn": OtherTagTestConfig(tag="half"), ".*": TagTestConfig()})
        half = quantizer.member_for("model.layers.0.self_attn.q_proj")
        rest = quantizer.member_for("model.layers.0.mlp.up_proj")

        def converter(*owners):
            return SimpleNamespace(operations=[SimpleNamespace(hf_quantizer=o) for o in owners], target_patterns=["w"])

        self.assertEqual(quantizer.get_new_dtype(converter(half), None, torch.bfloat16), torch.float16)
        self.assertEqual(quantizer.get_new_dtype(converter(rest), None, torch.bfloat16), torch.bfloat16)
        with self.assertRaisesRegex(ValueError, "different dtypes"):
            quantizer.get_new_dtype(converter(quantizer), None, torch.bfloat16)

    def test_a_none_entry_keeps_its_modules_in_full_precision(self):
        quantizer = self._load(
            {r"layers\.1\.self_attn": None, "self_attn": OtherTagTestConfig(), ".*": TagTestConfig()}
        )
        tags = _tags(self.model)
        self.assertIsNone(tags["model.layers.1.self_attn.q_proj"])
        self.assertEqual(tags["model.layers.0.self_attn.q_proj"], "a")
        self.assertIsNone(quantizer.member_for("model.layers.1.self_attn.q_proj"))
        self.assertIsNone(quantizer.member_for("lm_head"))
        read = auto.AutoQuantizationConfig.from_dict(json.loads(quantizer.quantization_config.to_json_string()))
        self.assertIsNone(read.dtype_plan[r"layers\.1\.self_attn"])

    def test_a_skip_list_entry_reads_as_the_plan_entry_that_skips_the_same_modules(self):
        names = ["model.layers.1.mlp", "model.layers.11.mlp", "model.layers.1", "lm_head", "model.decoder.fc1", "fc10"]
        for entry in ("model.layers.1", "layers.*.mlp", "fc1", "lm_head"):
            plan = {skip_pattern(entry): None, ".*": "quantized"}
            for name in names:
                with self.subTest(entry=entry, name=name):
                    skipped = not should_convert_module(name, [entry])
                    self.assertEqual(plan_entry_for(plan, name) is None, skipped)

    def test_a_member_converts_its_modules_when_checkpoint_names_differ(self):
        """Check that a member keeps the parents of its modules when the checkpoint renames them, as in a VLM."""

        def renamer(model):
            return lambda name: f"language_model.{name}" if name else name

        with (
            mock.patch("transformers.quantizers.base.checkpoint_module_namer", renamer),
            mock.patch("transformers.quantizers.quantizer_finegrained.checkpoint_module_namer", renamer),
        ):
            self._load({r"layers\.1\.mlp": TagTestConfig(tag="mlp"), "self_attn": OtherTagTestConfig(tag="attn")})
        tags = _tags(self.model)
        self.assertEqual(tags["model.layers.1.mlp.gate_proj"], "mlp")
        self.assertEqual(tags["model.layers.0.self_attn.q_proj"], "attn")
        self.assertIsNone(tags["model.layers.0.mlp.gate_proj"])

    def test_a_default_skip_yields_to_an_entry_that_names_the_module(self):
        """Check that `lm_head`, skipped by default, quantizes when an entry names it, and that a `None` entry wins."""
        self._load({"lm_head": OtherTagTestConfig(tag="head"), ".*": TagTestConfig()})
        self.assertEqual(self.model.lm_head.tag, "head")
        del self.model.lm_head.tag
        self._load({"lm_head": None, ".*": TagTestConfig()})
        self.assertIsNone(getattr(self.model.lm_head, "tag", None))

    def test_a_passed_none_entry_joins_the_checkpoints_plan(self):
        """Check that a passed plan adds its `None` entries to a checkpoint's plan, and keeps the checkpoint's formats."""
        from transformers.quantizers.auto import AutoHfQuantizer

        checkpoint = FineGrainedConfig(dtype_plan={".*": TagTestConfig(tag="shipped")}).to_dict()
        passed = FineGrainedConfig(dtype_plan={"lm_head": None, ".*": TagTestConfig(tag="passed")})
        merged = AutoHfQuantizer.merge_quantization_configs(checkpoint, passed)
        self.assertIsNone(merged.dtype_plan["lm_head"])
        self.assertEqual(merged.dtype_plan[".*"].tag, "shipped")
