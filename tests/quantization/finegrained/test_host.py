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
import tempfile
import unittest
from unittest import mock

from transformers import AutoModelForCausalLM, BitsAndBytesConfig, LlamaConfig, LlamaForCausalLM
from transformers.core_model_loading import ConversionOps, WeightConverter, WeightRenaming
from transformers.quantizers import auto
from transformers.quantizers.base import HfQuantizer
from transformers.quantizers.quantizer_finegrained import FineGrainedHfQuantizer
from transformers.quantizers.quantizers_utils import (
    get_module_from_name,
    plan_entry_for,
    should_convert_module,
    skip_pattern,
)
from transformers.testing_utils import require_bitsandbytes, require_torch, require_torch_accelerator, torch_device
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


class GreedyTagTestConfig(TagTestConfig):
    quant_method = "test_greedy_tag"


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

    def get_quantize_ops(self):
        return TagQuantizeTestOp(self.quantization_config.tag)

    def get_new_dtype(self, converter, param, dtype):
        return torch.float16 if self.quantization_config.tag == "half" else dtype

    def get_checkpoint_renamings(self):
        return [WeightRenaming(r"\.scale$", ".weight_scale")]

    def is_serializable(self):
        return True

    @property
    def is_trainable(self):
        return self.quantization_config.tag != "frozen"


class TagQuantizeTestOp(ConversionOps):
    """Test-only op that replaces each tensor with its member's tag."""

    def __init__(self, tag):
        self.tag = tag

    def convert(self, input_dict, **kwargs):
        return dict.fromkeys(input_dict, self.tag)


class GreedyTagTestQuantizer(TagTestQuantizer):
    """Test-only quantizer that replaces every linear and ignores its skip-list."""

    def _process_model_before_weight_loading(self, model, **kwargs):
        for module in model.modules():
            for name, child in module.named_children():
                if type(child) is torch.nn.Linear:
                    setattr(module, name, TaggedLinearTest(child.in_features, child.out_features, device="meta"))


class TaggedLinearTest(torch.nn.Linear):
    pass


def _tags(model):
    return {name: getattr(module, "tag", None) for name, module in model.named_modules() if name.endswith("proj")}


@require_torch
class FineGrainedHostTest(unittest.TestCase):
    def setUp(self):
        super().setUp()
        configs = {
            "test_tag": TagTestConfig,
            "test_other_tag": OtherTagTestConfig,
            "test_greedy_tag": GreedyTagTestConfig,
        }
        quantizers = {**dict.fromkeys(configs, TagTestQuantizer), "test_greedy_tag": GreedyTagTestQuantizer}
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

    def test_a_tensor_quantizes_through_its_members_op(self):
        quantizer = self._load({"self_attn": OtherTagTestConfig(tag="attn"), ".*": TagTestConfig(tag="rest")})
        op = quantizer.get_quantize_ops()
        for name, tag in (
            ("model.layers.0.self_attn.q_proj.weight", "attn"),
            ("model.layers.0.mlp.up_proj.weight", "rest"),
        ):
            with self.subTest(name=name):
                self.assertEqual(op.convert({name: torch.zeros(1)}, full_layer_name=name), {name: tag})

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

    def test_a_conversion_no_member_changes_stays_the_models(self):
        quantizer = self._load({"self_attn": OtherTagTestConfig(), ".*": TagTestConfig()})
        model_renaming = WeightRenaming("a", "b")
        self.assertEqual(quantizer._merge([model_renaming], [[WeightRenaming("a", "b")]] * 2), [model_renaming])
        with self.assertRaisesRegex(ValueError, "rename"):
            quantizer._merge([], [[WeightRenaming("x", "y")], [WeightRenaming("x", "z")]])

    def test_members_that_read_one_key_differently_share_one_converter(self):
        quantizer = self._load({"self_attn": OtherTagTestConfig(tag="attn"), ".*": TagTestConfig(tag="rest")})
        versions = [
            [WeightConverter(["weight", f"{tag}_scale"], "weight", [TagQuantizeTestOp(tag)])]
            for tag in (member.quantization_config.tag for member in quantizer.members)
        ]
        (converter,) = quantizer._merge([], versions)
        self.assertEqual(sorted(converter.source_patterns), ["attn_scale", "rest_scale", "weight"])
        for name, tag in (
            ("model.layers.0.self_attn.q_proj.weight", "attn"),
            ("model.layers.0.mlp.up_proj.weight", "rest"),
        ):
            with self.subTest(name=name):
                tensors = converter.operations[0].convert({"weight": torch.zeros(1)}, full_layer_name=name)
                self.assertEqual(tensors, {"weight": tag})

    def test_a_member_that_ignores_its_scope_raises(self):
        with self.assertRaisesRegex(ValueError, "does not cover"):
            self._load({"self_attn": GreedyTagTestConfig(), ".*": TagTestConfig()})

    def test_an_entrys_config_lends_its_loading_options_to_the_plan(self):
        plan = FineGrainedConfig(dtype_plan={"self_attn": OtherTagTestConfig(), ".*": TagTestConfig()})
        source = plan.loading_attributes_source(TagTestConfig(dequantize=True, modules_to_not_convert=["lm_head"]))
        self.assertTrue(source.dequantize)
        self.assertIsNone(source.dtype_plan[skip_pattern("lm_head")])
        tags_only = FineGrainedConfig(dtype_plan={".*": TagTestConfig()})
        self.assertIsNone(tags_only.loading_attributes_source(OtherTagTestConfig()))

    def test_a_converter_loads_in_the_dtype_of_the_member_that_added_it(self):
        quantizer = self._load({"self_attn": OtherTagTestConfig(tag="half"), ".*": TagTestConfig()})

        def merged(*tags):
            versions = [
                [WeightConverter("weight", "weight", [TagQuantizeTestOp("x")])]
                if member.quantization_config.tag in tags
                else []
                for member in quantizer.members
            ]
            return quantizer._merge([], versions)[0]

        self.assertEqual(quantizer.get_new_dtype(merged("half"), None, torch.bfloat16), torch.float16)
        self.assertEqual(quantizer.get_new_dtype(merged("a"), None, torch.bfloat16), torch.bfloat16)
        with self.assertRaisesRegex(ValueError, "different dtypes"):
            quantizer.get_new_dtype(merged("half", "a"), None, torch.bfloat16)

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


@require_bitsandbytes
@require_torch_accelerator
class FineGrainedHostBitsAndBytesTest(unittest.TestCase):
    """Two real quantizers that read the same keys, as members of one plan."""

    def test_bitsandbytes_in_two_formats_loads_saves_and_reloads(self):
        import bitsandbytes as bnb

        config = LlamaConfig(
            hidden_size=256, intermediate_size=512, num_hidden_layers=2, num_attention_heads=4, vocab_size=128
        )
        torch.manual_seed(0)
        reference = LlamaForCausalLM(config).to(torch.bfloat16)
        plan = {"mlp": BitsAndBytesConfig(load_in_4bit=True), "self_attn": BitsAndBytesConfig(load_in_8bit=True)}
        ids = torch.tensor([[1, 2, 3, 4]], device=torch_device)
        with tempfile.TemporaryDirectory() as path:
            reference.save_pretrained(path)
            model = AutoModelForCausalLM.from_pretrained(
                path,
                dtype=torch.bfloat16,
                device_map=torch_device,
                quantization_config=FineGrainedConfig(dtype_plan=plan),
            )
            layer = model.model.layers[0]
            self.assertIsInstance(layer.mlp.up_proj, bnb.nn.Linear4bit)
            self.assertIsInstance(layer.self_attn.q_proj, bnb.nn.Linear8bitLt)
            self.assertIs(type(model.lm_head), torch.nn.Linear)
            logits = model(ids).logits
            # 4-bit weights move the logits by about a tenth of their scale
            torch.testing.assert_close(logits, reference.to(torch_device)(ids).logits, atol=0.2, rtol=0)
            model.save_pretrained(f"{path}/quantized")
            reloaded = AutoModelForCausalLM.from_pretrained(f"{path}/quantized", device_map=torch_device)
            self.assertIsInstance(reloaded.model.layers[0].self_attn.q_proj, bnb.nn.Linear8bitLt)
            torch.testing.assert_close(reloaded(ids).logits, logits)
