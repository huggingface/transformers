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

import os
import re
import tempfile
import unittest
import warnings
from types import SimpleNamespace
from unittest import mock

import torch
from parameterized import parameterized
from torch import nn

import transformers.integrations.finegrained.layers as fg
from transformers import AutoModelForCausalLM, DeepseekV4Config, PreTrainedConfig, PreTrainedModel
from transformers.quantizers.auto import AutoQuantizationConfig
from transformers.testing_utils import (
    TestCasePlus,
    cleanup,
    require_kernels,
    require_torch,
    require_torch_accelerator,
    slow,
    torch_device,
)
from transformers.utils.quantization_config import (
    BitsAndBytesConfig,
    BlockScaledConfig,
    FineGrainedConfig,
    Fp8Config,
    Mxfp4Config,
    Mxfp8Config,
    NVFP4Config,
)

from ...integrations.finegrained.utils import (
    _Cfg,
    _conversion_model,
    _quantizer_for,
    _targets,
)
from ..block_scaled_checkpoint_tester import TinyCheckpointTesterMixin
from ..block_scaled_parallel_tester import LoadPathEquivalenceTesterMixin
from ..block_scaled_tester import (
    FINEGRAINED,
    FORMATS,
    FP8,
    MXFP4,
    MXFP8,
    NVFP4,
    _cheap_tuning,
    _require_kernel_capability,
)


def _read(model_config):
    """Return the quantization config `from_pretrained` reads from `model_config`."""
    return AutoQuantizationConfig.from_dict(model_config.quantization_config)


@require_torch
class FineGrainedPlanTest(unittest.TestCase):
    """`FineGrainedConfig` plans, how mixed-format checkpoints read into them, and what each entry loads."""

    def test_a_legacy_expert_dtype_becomes_the_two_entries_it_meant(self):
        """Check that DeepSeek-V4's model-level `expert_dtype` becomes a two-entry plan.

        DeepSeek-V4 ships `quant_method="fp8"` and declares its MXFP4 experts on the model config.
        `shared_experts` is dense despite its name, so the pattern matches a path segment, not a substring.
        """
        from transformers.quantizers.quantizers_utils import plan_entry_for

        config = _read(
            DeepseekV4Config(expert_dtype="fp4", quantization_config={"quant_method": "fp8", "scale_fmt": "ue8m0"})
        )
        self.assertIsInstance(config, FineGrainedConfig)
        for module, expected in (
            ("model.layers.3.mlp.experts", "mxfp4"),
            ("model.layers.3.mlp.experts.gate_up_proj", "mxfp4"),
            ("model.layers.3.self_attn.q_proj", "fp8"),
            ("model.layers.3.mlp.shared_experts.up_proj", "fp8"),
        ):
            self.assertEqual(plan_entry_for(config.dtype_plan, module).method.value, expected, module)
        # Without `expert_dtype`, block-FP8 stays block-FP8 in DeepSeek's layout
        plain = _read(DeepseekV4Config(quantization_config={"quant_method": "fp8"}))
        self.assertEqual((type(plain), plain.checkpoint_format), (Fp8Config, "deepseek"))
        # A model quantized to FP8 and saved by transformers keeps its FP8 experts
        saved = _read(DeepseekV4Config(expert_dtype="fp4", quantization_config=Fp8Config(scale_fmt="ue8m0").to_dict()))
        self.assertEqual((type(saved), saved.checkpoint_format), (Fp8Config, None))

    def test_deepseek_block_scales_load_onto_their_slot(self):
        """Check that DeepSeek's block scale `scale` is renamed onto `weight_scale_inv`."""
        from transformers.core_model_loading import WeightRenaming, rename_source_key

        quantizer = _quantizer_for(_read(DeepseekV4Config(quantization_config={"quant_method": "fp8"})))
        quantizer.pre_quantized = True
        renamings = quantizer.get_checkpoint_renamings()
        self.assertTrue(all(isinstance(c, WeightRenaming) for c in renamings))
        renamed, _ = rename_source_key("layers.0.attn.wq_a.scale", renamings, [])
        self.assertEqual(renamed, "layers.0.attn.wq_a.weight_scale_inv")

    def test_deepseek_v4_takes_one_config_whether_its_checkpoint_is_from_the_hub_or_a_save(self):
        """Check that DeepSeek-V4 takes the same passed config for a Hub checkpoint and a saved one.

        The Hub checkpoint says `fp8` and a `save_pretrained` one says `finegrained`.
        A format config lends its loading attributes to the plan, so the user passes the same class either way.
        """
        from transformers.quantizers.auto import AutoHfQuantizer

        hub = {"quant_method": "fp8", "scale_fmt": "ue8m0", "weight_block_size": [128, 128]}
        saved = _read(DeepseekV4Config(expert_dtype="fp4", quantization_config=hub)).to_dict()
        for checkpoint in (hub, saved):
            shipped = DeepseekV4Config(expert_dtype="fp4", quantization_config=checkpoint).quantization_config
            merged = AutoHfQuantizer.merge_quantization_configs(shipped, Fp8Config(dequantize=True))
            self.assertIsInstance(merged, FineGrainedConfig)
            self.assertTrue(merged.dequantize)

    def test_related_config_classes_lend_their_loading_attributes(self):
        """Check that a format config lends its loading attributes to a plan.

        It lends its loading options to the plan, and its activation format to the entries in its format. A plan does
        not lend to a format checkpoint. Unrelated classes are refused.
        """
        from transformers.quantizers.auto import AutoHfQuantizer
        from transformers.quantizers.quantizers_utils import EXPERTS_PATTERN, plan_entry_for

        checkpoint = FineGrainedConfig(
            dtype_plan={EXPERTS_PATTERN: NVFP4Config(activation_format="bf16"), ".*": Fp8Config()}
        )
        merged = AutoHfQuantizer.merge_quantization_configs(
            checkpoint.to_dict(), Fp8Config(modules_to_not_convert=["lm_head"])
        )
        experts, rest = merged.dtype_plan[EXPERTS_PATTERN], merged.dtype_plan[".*"]
        self.assertEqual((type(experts), experts.activation_format), (NVFP4Config, "bf16"))
        self.assertIsNone(rest.activation_format)
        # the passed skip-list joins the plan as a `None` entry
        self.assertIsNone(plan_entry_for(merged.dtype_plan, "lm_head"))
        self.assertIs(plan_entry_for(merged.dtype_plan, "model.layers.0.mlp.experts"), experts)

        for checkpoint, passed in (
            ({"quant_method": "fp8"}, Mxfp4Config(dequantize=True)),
            ({"quant_method": "fp8"}, FineGrainedConfig(dtype_plan={".*": Fp8Config()}, dequantize=True)),
            (checkpoint.to_dict(), BitsAndBytesConfig(load_in_8bit=True)),
        ):
            with self.subTest(passed=type(passed).__name__), self.assertRaises(ValueError):
                AutoHfQuantizer.merge_quantization_configs(checkpoint, passed)

    def test_an_expert_dtype_in_the_quantization_config_reads_as_the_same_plan(self):
        """Check that an `expert_dtype` in the quantization config gives the same plan.

        DeepSeek-V4.1 declares its FP4 experts in the quantization config. The plan is block-FP8 at the checkpoint's
        block with W4A4 MXFP4 experts, whichever config the load gets.
        """
        from transformers.quantizers.quantizers_utils import EXPERTS_PATTERN

        shipped = {
            "quant_method": "fp8",
            "activation_scheme": "dynamic",
            "weight_block_size": [32, 32],
            "scale_fmt": "ue8m0",
            "expert_dtype": "fp4",
        }
        model_config = DeepseekV4Config()
        model_config.quantization_config = shipped
        for passed in (AutoQuantizationConfig.from_dict(shipped), Fp8Config(dequantize=True)):
            with self.subTest(passed=type(passed).__name__):
                config = model_config.adapt_quantization_config(passed)
                self.assertIsInstance(config, FineGrainedConfig)
                experts, rest = config.dtype_plan[EXPERTS_PATTERN], config.dtype_plan[".*"]
                # W4A4 uses the weights' own activation format
                self.assertEqual((type(experts), experts.activation_format), (Mxfp4Config, None))
                self.assertEqual(
                    (type(rest), rest.weight_block_size, rest.scale_fmt),
                    (Fp8Config, passed.weight_block_size, passed.scale_fmt),
                )
                # Loading options belong to the plan, not to an entry
                self.assertEqual((config.dequantize, rest.dequantize), (passed.dequantize, False))

    def test_resolution_survives_a_round_trip_through_config_json(self):
        """Check that plan resolution does not depend on key order.

        `to_json_string` sorts keys, so a config read back loses its order. The catch-all sorts first here.
        Targeted entries must win outright, or it would swallow every module.
        """
        import json

        from transformers.quantizers.quantizers_utils import plan_entry_for

        config = FineGrainedConfig(
            dtype_plan={r"\.experts($|\.)": NVFP4Config(), ".*": Fp8Config(weight_block_size=(128, 128))}
        )
        back = FineGrainedConfig.from_dict(json.loads(config.to_json_string()))
        self.assertIsInstance(plan_entry_for(back.dtype_plan, "model.layers.0.mlp.experts"), NVFP4Config)
        self.assertIsInstance(plan_entry_for(back.dtype_plan, "model.layers.0.self_attn.q_proj"), Fp8Config)
        # JSON turns the tuple into a list, and the config must read it back as a tuple
        self.assertEqual(back.dtype_plan[".*"].weight_block_size, (128, 128))

    def test_two_targeted_entries_claiming_one_module_is_an_error(self):
        from transformers.quantizers.quantizers_utils import plan_entry_for

        plan = {r"\.experts": Fp8Config(weight_block_size=None), r"mlp\.": NVFP4Config()}
        with self.assertRaises(ValueError):
            plan_entry_for(plan, "model.layers.0.mlp.experts")

    def test_an_export_mixing_formats_is_a_finegrained_config(self):
        config = _modelopt(
            config_groups={
                "group_0": {"weights": {"num_bits": 4, "type": "float", "group_size": 16}, "targets": ["re:.*mlp.*"]},
                "group_1": {"weights": {"num_bits": 8, "type": "float", "group_size": 32}, "targets": ["re:.*attn.*"]},
            },
        )
        self.assertIsInstance(config, FineGrainedConfig)
        self.assertEqual(
            {pattern: type(entry) for pattern, entry in config.dtype_plan.items()},
            {".*mlp.*": NVFP4Config, ".*attn.*": Mxfp8Config},
        )
        # no `input_activations` is weight-only, whatever the weights' own activation format
        self.assertEqual(config.dtype_plan[".*mlp.*"].activation_format, "bf16")

    def test_expert_dtype_reaches_a_config_quantizing_on_the_fly(self):
        """Check that on-the-fly quantization honors the model config's `expert_dtype`.

        The experts quantize to MXFP4, so the saved checkpoint loads as what it holds.
        """
        from transformers.quantizers.auto import get_hf_quantizer
        from transformers.quantizers.block_scaled_utils import BlockScaledQuantizer
        from transformers.quantizers.quantizers_utils import EXPERTS_PATTERN

        model_config = DeepseekV4Config(expert_dtype="fp4")
        with mock.patch.object(BlockScaledQuantizer, "validate_environment"):
            quantizer, model_config, _ = get_hf_quantizer(model_config, Fp8Config(), None, True, {})
        self.assertIsInstance(quantizer.quantization_config, FineGrainedConfig)
        self.assertIsInstance(quantizer.quantization_config.dtype_plan[EXPERTS_PATTERN], Mxfp4Config)
        self.assertFalse(quantizer.pre_quantized)

    def test_a_passed_configs_activation_format_reaches_a_prequantized_checkpoint(self):
        """The activation format picks the kernel arm the same weights run on, so a passed config sets it; the
        checkpoint's format options stay its own, and passing a config warns that they are ignored."""
        from transformers.quantizers.auto import AutoHfQuantizer
        from transformers.quantizers.quantizers_utils import EXPERTS_PATTERN

        merged = AutoHfQuantizer.merge_quantization_configs(
            {"quant_method": "mxfp4"}, Mxfp4Config(activation_format="mxfp8")
        )
        self.assertEqual((type(merged), merged.activation_format), (Mxfp4Config, "mxfp8"))

        # a plan's entries lend theirs to the checkpoint's own entries
        model_config = DeepseekV4Config(expert_dtype="fp4")
        hub = {"quant_method": "fp8", "scale_fmt": "ue8m0", "weight_block_size": [128, 128]}
        passed = FineGrainedConfig(
            dtype_plan={
                EXPERTS_PATTERN: Mxfp4Config(activation_format="bf16"),
                ".*": Fp8Config(weight_block_size=(64, 64)),
            }
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            merged = AutoHfQuantizer.merge_quantization_configs(
                model_config.adapt_quantization_config(AutoQuantizationConfig.from_dict(hub)), passed
            )
        self.assertTrue(any(issubclass(w.category, UserWarning) for w in caught))
        experts, rest = merged.dtype_plan[EXPERTS_PATTERN], merged.dtype_plan[".*"]
        self.assertEqual(experts.activation_format, "bf16")
        self.assertEqual((rest.weight_block_size, rest.scale_fmt), ((128, 128), "ue8m0"))

    def test_a_passed_config_lends_its_activation_format(self):
        """Check that a passed config lends its activation format like its other loading attributes.

        A weight-only NVFP4 checkpoint has no calibrated activation globals. It refuses NVFP4 activations unless the
        load dequantizes.
        """
        from transformers.quantizers.auto import AutoHfQuantizer

        weight_only = {"quant_method": "nvfp4", "activation_format": "bf16"}
        merged = AutoHfQuantizer.merge_quantization_configs(
            {"quant_method": "mxfp4", "activation_format": "mxfp8"}, Mxfp4Config(modules_to_not_convert=["lm_head"])
        )
        self.assertEqual((merged.activation_format, merged.modules_to_not_convert), ("bf16", ["lm_head"]))
        merged = AutoHfQuantizer.merge_quantization_configs(weight_only, NVFP4Config(activation_format="bf16"))
        self.assertEqual(merged.activation_format, "bf16")
        merged = AutoHfQuantizer.merge_quantization_configs(weight_only, NVFP4Config(dequantize=True))
        self.assertTrue(merged.dequantize)
        with self.assertRaises(ValueError):
            AutoHfQuantizer.merge_quantization_configs(weight_only, NVFP4Config(modules_to_not_convert=["lm_head"]))

        # The same holds for a weight-only NVFP4 entry of a plan
        for passed, refused in ((NVFP4Config(dequantize=True), False), (NVFP4Config(), True)):
            with self.subTest(dequantize=passed.dequantize):
                mixed = FineGrainedConfig(
                    dtype_plan={".*mlp.*": NVFP4Config(activation_format="bf16"), ".*attn.*": Mxfp4Config()}
                )
                if refused:
                    with self.assertRaises(ValueError):
                        AutoHfQuantizer.merge_quantization_configs(mixed, passed)
                else:
                    self.assertTrue(AutoHfQuantizer.merge_quantization_configs(mixed, passed).dequantize)

    def test_a_copied_model_keeps_its_param_ops(self):
        """Check that `copy.deepcopy(model)` modules, whose configs are equal copies, still get their layout ops."""
        import copy

        from transformers.quantizers.quantizer_finegrained import FineGrainedHfQuantizer

        quantizer = FineGrainedHfQuantizer(FineGrainedConfig(dtype_plan={".*": Fp8Config()}))
        model = _conversion_model(quantizer)
        copied = copy.deepcopy(model)
        name = "model.layers.0.mlp.experts.gate_up_proj"
        member = quantizer.member_for("model.layers.0.mlp.experts")
        ops = [type(op) for op in member.get_param_ops(model, name)]
        self.assertEqual([type(op) for op in member.get_param_ops(copied, name)], ops)

    def test_an_embedding_table_loads_through_the_entry_that_names_it(self):
        """Check that a targeted entry opts a table in, while the catch-all leaves tables alone."""
        from transformers import LlamaConfig, LlamaForCausalLM
        from transformers.integrations.finegrained import FineGrainedEmbedding
        from transformers.quantizers.quantizer_finegrained import FineGrainedHfQuantizer

        config = LlamaConfig(hidden_size=64, intermediate_size=64, num_hidden_layers=1, vocab_size=64)
        per_tensor = Fp8Config(weight_block_size=None)
        dense = Fp8Config()
        for plan, quantized in (({"embed_tokens": per_tensor, ".*": dense}, True), ({".*": dense}, False)):
            with self.subTest(quantized=quantized), torch.device("meta"):
                model = LlamaForCausalLM(config)
                quantizer = FineGrainedHfQuantizer(FineGrainedConfig(dtype_plan=plan))
                quantizer._process_model_before_weight_loading(model)
                self.assertEqual(isinstance(model.model.embed_tokens, FineGrainedEmbedding), quantized)
                self.assertNotIsInstance(model.lm_head, FineGrainedEmbedding)


@require_torch
class FineGrainedPlanConversionTest(unittest.TestCase):
    """The conversions a plan of several formats builds."""

    def test_input_scale_reaches_each_formats_own_slot(self):
        """Check that `input_scale` reaches each format's own slot.

        A static FP8 module reads it as its activation scale. An NVFP4 module reads it as its activation global.
        Only the format that holds the experts reads their keys. A plan where two formats would is refused.
        """
        from transformers.core_model_loading import WeightConverter, WeightRenaming, rename_source_key
        from transformers.quantizers.quantizer_finegrained import FineGrainedHfQuantizer
        from transformers.quantizers.quantizers_utils import EXPERTS_PATTERN

        config = FineGrainedConfig(
            dtype_plan={
                EXPERTS_PATTERN: NVFP4Config(activation_format="nvfp4"),
                ".*": Fp8Config(weight_block_size=None, activation_scheme="static"),
            },
        )
        quantizer = FineGrainedHfQuantizer(config)
        fp8 = quantizer.member_for("model.layers.0.self_attn.q_proj")
        model = _conversion_model(quantizer)
        self.assertEqual(fp8._up_projections(model), ())
        conversions = quantizer.update_weight_conversions([], model)
        renamings = [c for c in conversions if isinstance(c, WeightRenaming)]
        converters = [c for c in conversions if isinstance(c, WeightConverter)]

        renamed, _ = rename_source_key("model.layers.0.self_attn.q_proj.input_scale", renamings, converters)
        self.assertEqual(renamed, "model.layers.0.self_attn.q_proj.activation_scale")
        _, pattern = rename_source_key("model.layers.0.mlp.experts.3.down_proj.input_scale", renamings, converters)
        (converter,) = [c for c in converters if pattern in c.source_patterns]
        self.assertIn("experts.down_proj_activation_scale", _targets(converter))
        self.assertIn("experts.down_proj_weight_global_scale", _targets(converter))

        with mock.patch.object(fp8, "_up_projections", return_value=("gate_up_proj",)), self.assertRaises(ValueError):
            quantizer.update_weight_conversions([], model)

    @staticmethod
    def _split(conversions):
        from transformers.core_model_loading import WeightConverter, WeightRenaming

        renamings = [c for c in conversions if isinstance(c, WeightRenaming)]
        return renamings, [c for c in conversions if isinstance(c, WeightConverter)]

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

    def test_experts_under_any_parent_and_ungated_stacks_get_their_converters(self):
        """Check that ungated experts under any parent get their converters.

        Nemotron-H holds ungated experts (`up_proj`) under `mixer.experts`. Their per-expert scales must reach that
        stack. The scales take the swizzle op but not the gate/up interleave.
        """
        from transformers.core_model_loading import rename_source_key
        from transformers.integrations.finegrained.conversions import (
            FineGrainedInterleaveGateUp,
            FineGrainedSwizzleScales,
        )
        from transformers.integrations.moe import use_experts_implementation

        @use_experts_implementation(has_gate=False)
        class UngatedExperts(nn.Module):
            def __init__(self, config):
                super().__init__()
                self.up_proj = nn.Parameter(
                    torch.empty(_Cfg.num_local_experts, _Cfg.intermediate_size, _Cfg.hidden_size)
                )
                self.down_proj = nn.Parameter(
                    torch.empty(_Cfg.num_local_experts, _Cfg.hidden_size, _Cfg.intermediate_size)
                )

        class HybridModel(PreTrainedModel):
            config: PreTrainedConfig

            def __init__(self, config):
                super().__init__(config)
                self.layers = nn.ModuleList([nn.Module()])
                self.layers[0].mixer = nn.Module()
                self.layers[0].mixer.experts = UngatedExperts(config)
                self.post_init()

            def _init_weights(self, module):
                pass

        config = PreTrainedConfig()
        for name in ("hidden_size", "intermediate_size", "num_local_experts", "hidden_act"):
            setattr(config, name, getattr(_Cfg, name))
        model = HybridModel(config)
        quantizer = _quantizer_for(NVFP4Config())
        with mock.patch.object(fg, "is_sm100", return_value=False):
            quantizer._process_model_before_weight_loading(model)
        renamings, converters = self._split(quantizer.get_weight_conversions(model))
        for shipped, slot in (("weight_scale_2", "weight_global_scale"), ("input_scale", "activation_scale")):
            with self.subTest(key=shipped):
                renamed, _ = rename_source_key(f"layers.0.mixer.experts.3.up_proj.{shipped}", renamings, converters)
                self.assertEqual(renamed, f"layers.0.mixer.experts.up_proj_{slot}")

        ops = quantizer.get_param_ops(model, "layers.0.mixer.experts.up_proj_scale_inv")
        self.assertTrue(any(isinstance(op, FineGrainedSwizzleScales) for op in ops))
        self.assertFalse(any(isinstance(op, FineGrainedInterleaveGateUp) for op in ops))


MIXED_PAIRS = [
    (f"{experts.name}_{experts_activation}_x_{dense.name}_{dense_activation}", *pair)
    for pair in (
        (FP8, None, NVFP4, "bf16"),
        (MXFP8, "mxfp4", FP8, None),
        (MXFP4, "mxfp8", MXFP8, "bf16"),
        (NVFP4, None, MXFP4, None),
    )
    for experts, experts_activation, dense, dense_activation in [pair]
]


@slow
@require_kernels
@require_torch_accelerator
class FineGrainedMixedSchemesTest(TestCasePlus):
    """One model quantized with two schemes, one for the experts and one for every other linear.

    Each module must be built in its plan entry's scheme. The forward must stay close to the bf16 model.
    A save must reload to the same logits, which rebuilds both schemes from the saved plan.
    """

    @classmethod
    def setUpClass(cls):
        from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM

        super().setUpClass()
        cls.enterClassContext(_cheap_tuning())
        cls.bf16_dir = cls.enterClassContext(tempfile.TemporaryDirectory())
        config = Qwen3MoeConfig(
            vocab_size=1024,
            hidden_size=256,
            intermediate_size=256,
            moe_intermediate_size=256,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=64,
            num_experts=8,
            num_experts_per_tok=2,
        )
        torch.manual_seed(0)
        reference = Qwen3MoeForCausalLM(config).to(torch.bfloat16)
        reference.save_pretrained(cls.bf16_dir)
        cls.ids = torch.arange(3, 35, dtype=torch.long, device=torch_device).unsqueeze(0)
        with torch.no_grad():
            cls.want = reference.to(torch_device)(cls.ids).logits.float()

    def setUp(self):
        super().setUp()
        _require_kernel_capability(self)

    def tearDown(self):
        cleanup(torch_device, gc_collect=True)
        super().tearDown()

    @parameterized.expand(MIXED_PAIRS)
    def test_mixed_schemes_load_save_and_reload(self, _, experts, experts_activation, dense, dense_activation):
        from transformers.quantizers.quantizers_utils import EXPERTS_PATTERN

        experts_type = experts.config_cls(activation_format=experts_activation)
        dense_type = dense.config_cls(activation_format=dense_activation)
        quantization_config = FineGrainedConfig(dtype_plan={EXPERTS_PATTERN: experts_type, ".*": dense_type})
        model = AutoModelForCausalLM.from_pretrained(
            self.bf16_dir, quantization_config=quantization_config, device_map=torch_device
        )
        layer = model.model.layers[0]
        for module, expected in ((layer.mlp.experts, experts_type), (layer.self_attn.q_proj, dense_type)):
            self.assertEqual(
                (module.weight_format, module.activation_format), (expected.quant_method, expected.activation_format)
            )

        ids, want = self.ids, self.want
        with torch.no_grad():
            logits = model(ids).logits.float()
        error = ((logits - want).norm() / want.norm()).item()
        self.assertLess(error, 0.5, f"{error:.3f} off bf16")

        saved = os.path.join(self.get_auto_remove_tmp_dir(), "saved")
        model.save_pretrained(saved)
        reloaded, info = AutoModelForCausalLM.from_pretrained(saved, device_map=torch_device, output_loading_info=True)
        self.assertEqual(
            (set(info["missing_keys"]), set(info["unexpected_keys"]), set(info["mismatched_keys"])),
            (set(), set(), set()),
        )
        with torch.no_grad():
            torch.testing.assert_close(reloaded(ids).logits.float(), logits, rtol=1e-2, atol=1e-2)


class FineGrainedTinyCheckpointTest(TinyCheckpointTesterMixin, TestCasePlus):
    spec = FINEGRAINED


class FineGrainedLoadPathEquivalenceTest(LoadPathEquivalenceTesterMixin, TestCasePlus):
    spec = FINEGRAINED
