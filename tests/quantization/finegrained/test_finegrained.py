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
import warnings
from types import SimpleNamespace
from unittest import mock

import torch
from torch import nn

import transformers.integrations.finegrained.layers as fg
from transformers import DeepseekV4Config
from transformers.quantizers.auto import AutoQuantizationConfig
from transformers.testing_utils import (
    require_torch,
)
from transformers.utils.quantization_config import (
    BlockScaledConfig,
    FineGrainedConfig,
    Fp8Config,
    Mxfp4Config,
)

from ...integrations.finegrained.utils import (
    _Cfg,
    _conversion_model,
    _quantizer_for,
)
from ..block_scaled_tester import (
    FORMATS,
)


def _read(model_config):
    """Return the quantization config `from_pretrained` reads from `model_config`."""
    return AutoQuantizationConfig.from_dict(model_config.quantization_config)


@require_torch
class FineGrainedPlanTest(unittest.TestCase):
    """`FineGrainedConfig` plans, how mixed-format checkpoints read into them, and what each entry loads."""

    def test_the_base_config_is_not_a_format(self):
        with self.assertRaises(TypeError):
            BlockScaledConfig()

    def test_a_plan_entry_is_a_format_config_without_loading_options(self):
        """Check that a plan entry with a loading option is refused. Loading options apply to the whole model."""
        with self.assertRaises(ValueError):
            FineGrainedConfig(dtype_plan={".*": Fp8Config(dequantize=True)})
        with self.assertRaises(ValueError):
            FineGrainedConfig(dtype_plan={".*": {"quant_method": "not_a_method"}})
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
        from transformers.quantizers.auto import AutoHfQuantizer

        model_config = DeepseekV4Config(quantization_config=shipped)
        for passed in (None, Fp8Config(dequantize=True)):
            with self.subTest(passed=type(passed).__name__):
                config = AutoHfQuantizer.merge_quantization_configs(model_config.quantization_config, passed)
                self.assertIsInstance(config, FineGrainedConfig)
                experts, rest = config.dtype_plan[EXPERTS_PATTERN], config.dtype_plan[".*"]
                # W4A4 uses the weights' own activation format
                self.assertEqual((type(experts), experts.activation_format), (Mxfp4Config, None))
                self.assertEqual(
                    (type(rest), rest.weight_block_size, rest.scale_fmt),
                    (Fp8Config, (32, 32), "ue8m0"),
                )
                # Loading options belong to the plan, not to an entry
                self.assertEqual((config.dequantize, rest.dequantize), (passed is not None, False))

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

    def test_options_the_format_fixes_are_refused_and_not_written(self):
        """Check that MXFP4 refuse the options they fix and do not write them.

        Only block-FP8 sets its block, scale dtype and activation scheme. A `config.json` with the format's own values
        or the old defaults still reads back.
        """
        for config_class in (Mxfp4Config,):
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

        # A format config lends its activation format to the plan entries of its format
        hub = {"quant_method": "fp8", "scale_fmt": "ue8m0", "weight_block_size": [128, 128]}
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            merged = AutoHfQuantizer.merge_quantization_configs(
                DeepseekV4Config(expert_dtype="fp4", quantization_config=hub).quantization_config,
                Mxfp4Config(activation_format="bf16"),
            )
        self.assertTrue(any(issubclass(w.category, UserWarning) for w in caught))
        experts, rest = merged.dtype_plan[EXPERTS_PATTERN], merged.dtype_plan[".*"]
        self.assertEqual(experts.activation_format, "bf16")
        self.assertEqual((rest.weight_block_size, rest.scale_fmt), ((128, 128), "ue8m0"))


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
