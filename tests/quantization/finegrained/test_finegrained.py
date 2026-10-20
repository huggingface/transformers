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
    _modelopt,
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


@require_torch
class FineGrainedPlanTest(unittest.TestCase):
    """`FineGrainedConfig` plans, how mixed-format checkpoints read into them, and what each entry loads."""

    def test_a_legacy_expert_dtype_becomes_the_two_entries_it_meant(self):
        """Check that DeepSeek-V4's model-level `expert_dtype` becomes a two-entry plan.

        DeepSeek-V4 ships `quant_method="fp8"` and declares its MXFP4 experts on the model config.
        `shared_experts` is dense despite its name, so the pattern matches a path segment, not a substring.
        """
        from transformers.quantizers.quantizers_utils import plan_entry_for

        config = DeepseekV4Config(expert_dtype="fp4").adapt_quantization_config(Fp8Config(scale_fmt="ue8m0"))
        self.assertIsInstance(config, FineGrainedConfig)
        for module, expected in (
            ("model.layers.3.mlp.experts", "mxfp4"),
            ("model.layers.3.mlp.experts.gate_up_proj", "mxfp4"),
            ("model.layers.3.self_attn.q_proj", "fp8"),
            ("model.layers.3.mlp.shared_experts.up_proj", "fp8"),
        ):
            self.assertEqual(plan_entry_for(config.dtype_plan, module).method.value, expected, module)
        # Without `expert_dtype`, block-FP8 stays block-FP8 in DeepSeek's layout
        plain = DeepseekV4Config().adapt_quantization_config(Fp8Config())
        self.assertEqual((type(plain), plain.checkpoint_format), (Fp8Config, "deepseek"))

    def test_deepseek_block_scales_load_onto_their_slot(self):
        """Check that DeepSeek's block scale `scale` is renamed onto `weight_scale_inv`."""
        from transformers.core_model_loading import WeightRenaming, rename_source_key

        quantizer = _quantizer_for(DeepseekV4Config().adapt_quantization_config(Fp8Config()))
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

        model_config = DeepseekV4Config(expert_dtype="fp4")
        hub = {"quant_method": "fp8", "scale_fmt": "ue8m0", "weight_block_size": [128, 128]}
        saved = model_config.adapt_quantization_config(AutoQuantizationConfig.from_dict(hub)).to_dict()
        for checkpoint in (hub, saved):
            resolved = model_config.adapt_quantization_config(AutoQuantizationConfig.from_dict(checkpoint))
            merged = AutoHfQuantizer.merge_quantization_configs(resolved, Fp8Config(dequantize=True))
            self.assertIsInstance(merged, FineGrainedConfig)
            self.assertTrue(merged.dequantize)

    def test_related_config_classes_lend_their_loading_attributes(self):
        """Check that format configs and plans lend each other their loading attributes.

        A format config lends its loading options to a plan, and its activation format to the entries in its format.
        A single-format plan lends them back to a checkpoint in that format. Unrelated classes are refused.
        """
        from transformers.quantizers.auto import AutoHfQuantizer
        from transformers.quantizers.quantizers_utils import EXPERTS_PATTERN

        checkpoint = FineGrainedConfig(
            dtype_plan={EXPERTS_PATTERN: NVFP4Config(activation_format="bf16"), ".*": Fp8Config()}
        )
        merged = AutoHfQuantizer.merge_quantization_configs(
            checkpoint.to_dict(), Fp8Config(modules_to_not_convert=["lm_head"])
        )
        experts, rest = merged.dtype_plan[EXPERTS_PATTERN], merged.dtype_plan[".*"]
        self.assertEqual((type(experts), experts.activation_format), (NVFP4Config, "bf16"))
        self.assertEqual((rest.activation_format, merged.modules_to_not_convert), (None, ["lm_head"]))

        merged = AutoHfQuantizer.merge_quantization_configs(
            {"quant_method": "fp8"},
            FineGrainedConfig(dtype_plan={".*": Fp8Config()}, dequantize=True),
        )
        self.assertEqual((type(merged), merged.activation_format, merged.dequantize), (Fp8Config, None, True))
        for checkpoint, passed in (
            ({"quant_method": "fp8"}, Mxfp4Config(dequantize=True)),
            ({"quant_method": "fp8"}, FineGrainedConfig(dtype_plan={".*": Mxfp4Config()})),
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
        # Without `input_activations` the entry is weight-only, whatever the weights' own activation format
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
        """Check that a passed config sets the activation format of a pre-quantized checkpoint.

        The activation format picks the kernel arm for the same weights. The checkpoint keeps its own format options.
        Passing a config warns that its other options are ignored.
        """
        from transformers.quantizers.auto import AutoHfQuantizer
        from transformers.quantizers.quantizers_utils import EXPERTS_PATTERN

        merged = AutoHfQuantizer.merge_quantization_configs(
            {"quant_method": "mxfp4"}, Mxfp4Config(activation_format="mxfp8")
        )
        self.assertEqual((type(merged), merged.activation_format), (Mxfp4Config, "mxfp8"))

        # A plan's entries lend their activation formats to the checkpoint's entries
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

    def test_the_formats_own_activation_name_is_native(self):
        """Check that each format's own activation name normalizes to `None`."""
        for config_class, own in (
            (Fp8Config, "fp8"),
            (Mxfp8Config, "mxfp8"),
            (Mxfp4Config, "mxfp4"),
            (NVFP4Config, "nvfp4"),
        ):
            with self.subTest(config_class=config_class.__name__):
                self.assertIsNone(config_class(activation_format=own).activation_format)
        self.assertIsNone(_modelopt().activation_format)

    def test_options_the_format_fixes_are_refused_and_not_written(self):
        """Check that MXFP8, MXFP4 and NVFP4 refuse the options they fix and do not write them.

        Only block-FP8 sets its block, scale dtype and activation scheme. A `config.json` with the format's own values
        or the old defaults still reads back. MiniMax-M3's `weight_block_size: [1, 32]` is one such case.
        """
        for config_class in (Mxfp8Config, Mxfp4Config, NVFP4Config):
            for option in ({"weight_block_size": (128, 128)}, {"activation_scheme": "static"}):
                with self.subTest(config_class=config_class.__name__, option=option):
                    with self.assertRaises(ValueError):
                        config_class(**option)
            for written in (
                {"weight_block_size": None, "scale_fmt": "float", "activation_scheme": "dynamic"},
                {"weight_block_size": list(config_class.weight_block_size)},
            ):
                config = config_class.from_dict({"quant_method": config_class.method.value, **written})
                self.assertFalse(set(written) & set(config.to_dict()))
        self.assertEqual(Fp8Config(scale_fmt="ue8m0").to_dict()["scale_fmt"], "ue8m0")

    def test_a_format_configs_options_are_keyword_only_and_unknown_ones_reported(self):
        with self.assertRaises(TypeError):
            Fp8Config((64, 64))
        with self.assertLogs("transformers.utils.quantization_config", level="INFO") as logs:
            Mxfp8Config(activaton_format="bf16")
        self.assertIn("activaton_format", "".join(logs.output))

    def test_the_base_config_is_not_a_format(self):
        with self.assertRaises(TypeError):
            BlockScaledConfig()

    def test_a_plan_entry_is_a_format_config_without_loading_options(self):
        """Check that a plan entry with a loading option is refused. Loading options apply to the whole model."""
        for entry in (
            Fp8Config(dequantize=True),
            NVFP4Config(modules_to_not_convert=["lm_head"]),
            Fp8Config(modules_to_convert=["embed_tokens"]),
            {"quant_method": "mxfp8", "dequantize": True},
        ):
            with self.subTest(entry=entry):
                with self.assertRaises(ValueError):
                    FineGrainedConfig(dtype_plan={".*": entry})
        with self.assertRaises(ValueError):
            FineGrainedConfig(dtype_plan={".*": {"quant_method": "awq"}})
        with self.assertRaises(TypeError):
            FineGrainedConfig(dtype_plan={".*": "fp8"})

    def test_the_plans_members_load_as_the_host_does(self):
        from transformers.quantizers.quantizer_finegrained import FineGrainedHfQuantizer

        config = FineGrainedConfig(dtype_plan={r"\.experts($|\.)": NVFP4Config(), ".*": Fp8Config()})
        quantizer = FineGrainedHfQuantizer(config, pre_quantized=False)
        self.assertEqual([member.pre_quantized for member in quantizer.members()], [False, False])

    def test_a_loaded_model_is_finished_once_whatever_its_formats(self):
        """Check that each format checks its modules and disables multi-device DeepGEMM once per load."""
        from transformers.quantizers.quantizer_finegrained import FineGrainedHfQuantizer

        for config in (
            FineGrainedConfig(dtype_plan={r"\.experts($|\.)": NVFP4Config(), ".*": Fp8Config()}),
            Mxfp8Config(),
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
        plan = tp_plan({r"layers\.\d+\.mlp\.experts": Mxfp8Config(), ".*": per_tensor})
        self.assertEqual(plan.get("layers.*.mlp.experts.gate_up_proj_scale_inv"), "moe_experts_colwise")
        self.assertEqual(plan.get("layers.*.mlp.experts.down_proj_scale_inv"), "moe_experts_rowwise")
        with self.assertRaises(ValueError):
            tp_plan({r"layers\.1\.mlp\.experts": Mxfp8Config(), ".*": per_tensor})

    def test_ep_plans_use_masked_routing_for_every_format(self):
        from transformers.quantizers.quantizers_utils import EXPERTS_PATTERN

        configs = [spec.config_cls() for spec in FORMATS]
        configs.append(FineGrainedConfig(dtype_plan={EXPERTS_PATTERN: Mxfp4Config(), ".*": Fp8Config()}))
        for config in configs:
            for router in ("gate", "router"):
                with self.subTest(config=type(config).__name__, router=router):
                    model = nn.Module()
                    model.config = SimpleNamespace(_experts_implementation=None)
                    model.layers = nn.ModuleList([nn.Module()])
                    model.layers[0].mlp = nn.Module()
                    setattr(model.layers[0].mlp, router, nn.Module())
                    model.layers[0].mlp.experts = nn.Module()
                    plan = {
                        "layers.*.mlp.experts": "ep_dispatch_experts",
                        "layers.*.mlp.experts.gate_up_proj": "grouped_gemm",
                        "layers.*.mlp.experts.down_proj": "grouped_gemm",
                    }
                    quantizer = _quantizer_for(config)
                    quantizer.update_parallel_plan(model, plan)
                    expected = {
                        "layers.*.mlp.experts": "moe_tp_experts",
                        f"layers.*.mlp.{router}": "ep_router",
                        "layers.*.mlp.experts.gate_up_proj": "grouped_gemm",
                        "layers.*.mlp.experts.down_proj": "grouped_gemm",
                    }
                    self.assertEqual(plan, expected)
                    quantizer.update_parallel_plan(model, plan)
                    self.assertEqual(plan, expected)

    def test_parallel_plans_replicate_the_per_tensor_scales(self):
        """Check that parallel plans replicate per-tensor scales.

        A per-tensor scale is 0-dim and has no axis to split. Each one in a sharded module gets its own `replicate`
        entry, which takes precedence over the module's style. A scale grid follows its module's style.
        """
        from transformers.distributed.tensor_parallel import ALL_PARALLEL_STYLES, resolve_parallel_plans
        from transformers.integrations.finegrained import FineGrainedLinear
        from transformers.quantizers.quantizer_nvfp4 import NVFP4HfQuantizer

        model = nn.Module()
        model.q_proj = FineGrainedLinear(64, 64, quantization_config=NVFP4Config())
        model.o_proj = FineGrainedLinear(64, 64, quantization_config=NVFP4Config())
        model.tp_plan, model.ep_plan = {"q_proj": "colwise"}, {}
        model.config = SimpleNamespace(_experts_implementation=None)
        NVFP4HfQuantizer(NVFP4Config()).update_parallel_plan(model, model.tp_plan)
        distributed_config = SimpleNamespace(tp_plan=None, ep_plan=None, tp_size=2, ep_size=1)
        tp_plan, _ = resolve_parallel_plans(model, distributed_config)
        self.assertEqual(
            tp_plan,
            {"q_proj": "colwise", "q_proj.weight_global_scale": "replicate", "q_proj.activation_scale": "replicate"},
        )
        scale = model.q_proj.weight_global_scale
        ALL_PARALLEL_STYLES["replicate"].shard_param(model.q_proj, "weight_global_scale", mesh=None)
        self.assertIs(model.q_proj.weight_global_scale, scale)

    def test_a_copied_model_finds_its_members_by_config_value(self):
        """Check that `copy.deepcopy(model)` modules, whose configs are equal copies, still find their member."""
        import copy

        from transformers.quantizers.quantizer_finegrained import FineGrainedHfQuantizer

        quantizer = FineGrainedHfQuantizer(FineGrainedConfig(dtype_plan={".*": Fp8Config()}))
        model = _conversion_model(quantizer)
        copied = copy.deepcopy(model)
        name = "model.layers.0.mlp.experts.gate_up_proj"
        ops = [type(op) for op in quantizer.get_param_ops(model, name)]
        self.assertEqual([type(op) for op in quantizer.get_param_ops(copied, name)], ops)
        self.assertIsNone(
            quantizer.member_of(SimpleNamespace(quantization_config=Fp8Config(weight_block_size=(2, 2))))
        )


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
        quantizer.pre_quantized = True
        fp8 = next(member for member in quantizer.members() if member.quantization_config is config.dtype_plan[".*"])
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
        renamings, converters = self._split(
            quantizer.format_conversions(dequantize=False, up_projections=quantizer._up_projections(model))
        )
        for shipped, slot in (("weight_scale_2", "weight_global_scale"), ("input_scale", "activation_scale")):
            with self.subTest(key=shipped):
                renamed, _ = rename_source_key(f"layers.0.mixer.experts.3.up_proj.{shipped}", renamings, converters)
                self.assertEqual(renamed, f"layers.0.mixer.experts.up_proj_{slot}")

        ops = quantizer.get_param_ops(model, "layers.0.mixer.experts.up_proj_scale_inv")
        self.assertTrue(any(isinstance(op, FineGrainedSwizzleScales) for op in ops))
        self.assertFalse(any(isinstance(op, FineGrainedInterleaveGateUp) for op in ops))

    def test_an_embedding_tables_own_scale_keeps_its_name(self):
        """Check that an embedding table's per-tensor `weight_scale` keeps its name.

        modelopt and compressed-tensors name a block scale `weight_scale`, like a quantized embedding's per-tensor one.
        The rename onto the block-scale slot must skip the tables in `modules_to_convert`.
        """
        from transformers.core_model_loading import rename_source_key

        for config in (
            FineGrainedConfig(dtype_plan={".*": NVFP4Config()}, modules_to_convert=["embed_tokens"]),
            Fp8Config(checkpoint_format="compressed-tensors", modules_to_convert=["embed_tokens"]),
        ):
            with self.subTest(config=type(config).__name__):
                renamings, converters = self._split(
                    _quantizer_for(config).update_weight_conversions([], _conversion_model(_quantizer_for(config)))
                )
                for key, expected in (
                    ("model.embed_tokens.weight_scale", "model.embed_tokens.weight_scale"),
                    (
                        "model.layers.0.self_attn.q_proj.weight_scale",
                        "model.layers.0.self_attn.q_proj.weight_scale_inv",
                    ),
                ):
                    self.assertEqual(rename_source_key(key, renamings, converters)[0], expected)

    def test_scalar_scale_ops_go_only_to_the_experts_of_their_format(self):
        """Check that a per-tensor ModelOpt entry leaves the merge converters of another format's experts alone."""
        from transformers.integrations.finegrained.conversions import FineGrainedScalarScales
        from transformers.quantizers.quantizer_finegrained import FineGrainedHfQuantizer

        from ...integrations.finegrained.utils import _model_mapping

        per_tensor = Fp8Config(weight_block_size=None, activation_scheme="static", checkpoint_format="modelopt")
        block = Fp8Config(weight_block_size=(32, 32), checkpoint_format="modelopt")
        for experts, expected in ((per_tensor, True), (block, False)):
            with self.subTest(experts=experts.weight_block_size):
                plan = {r"mlp\.experts$": experts, ".*": per_tensor}
                quantizer = FineGrainedHfQuantizer(FineGrainedConfig(dtype_plan=plan))
                quantizer.pre_quantized = True
                model = _conversion_model(quantizer)
                converters = quantizer.update_weight_conversions(_model_mapping(), model)
                scalar = any(
                    isinstance(op, FineGrainedScalarScales)
                    for converter in converters
                    for op in getattr(converter, "operations", [])
                )
                self.assertEqual(scalar, expected)


# Each pair is (experts, dense). Every format appears once on each side.
# The activation formats are picked so each pair hits distinct kernel arms.
MIXED_PAIRS = [
    (f"{experts.name}_{experts_activation}_x_{dense.name}_{dense_activation}", *pair)
    for pair in (
        (FP8, None, NVFP4, "bf16"),
        (MXFP8, None, FP8, None),
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
