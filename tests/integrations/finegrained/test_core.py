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
"""The `finegrained` integration's own pieces, no kernel involved: the modules and what replaces what,
the quantization config and its groups, environment validation, the parallel plans, and the frozen
legacy FP8 shim. The kernel call contract lives in `tests/kernels/test_finegrained.py`, the conversion
ops and chains in `test_conversions.py`."""

import re
import unittest
from contextlib import ExitStack, contextmanager
from unittest import mock

import torch
from parameterized import parameterized

import transformers.integrations.finegrained.core as fg
from transformers.integrations.finegrained import FineGrainedExperts, FineGrainedLinear
from transformers.testing_utils import require_torch


class _Cfg:
    hidden_size = 64
    num_local_experts = 4
    intermediate_size = 32
    hidden_act = "silu"
    _experts_implementation = None


def _quantizer_for(config):
    """The quantizer a `from_pretrained` would pick for this config. Since the arms were split
    per scheme, constructing the base directly only exercises what every scheme shares."""
    from transformers.quantizers.auto import AUTO_QUANTIZER_MAPPING

    method = getattr(config.quant_method, "value", config.quant_method)
    return AUTO_QUANTIZER_MAPPING[method](config)


def _targets(conv):
    t = conv.target_patterns
    return t if isinstance(t, (list, tuple)) else [t]


@require_torch
class FineGrainedModulesTest(unittest.TestCase):
    """The finegrained modules themselves: what they hold and what replaces what."""

    def test_a_grouped_linear_is_recognised_by_its_groups_not_its_name(self):
        """`replace_with_finegrained_layer` picks `FineGrainedGroupedLinear` for a block-diagonal
        linear — a plain one would collapse the groups into a single giant matmul and return the
        wrong output dim. The branch keys on `n_groups`, the attribute the swap consumes: a class
        NAME substring would miss a grouped linear named anything else, and would match (then
        `AttributeError` on) one that carries its group count under another name.
        """
        from types import SimpleNamespace

        import torch.nn as nn

        from transformers.integrations.finegrained import FineGrainedGroupedLinear, replace_with_finegrained_layer
        from transformers.utils.quantization_config import FineGrainedConfig

        class OddlyNamedBlockDiagonal(nn.Linear):
            def __init__(self):
                super().__init__(8, 16, bias=False)
                self.n_groups = 2

        model = nn.Module()
        model.config = SimpleNamespace(get_text_config=lambda: SimpleNamespace())
        model.grouped = OddlyNamedBlockDiagonal()
        model.plain = nn.Linear(8, 16, bias=False)

        with torch.device("meta"):
            replace_with_finegrained_layer(model, None, FineGrainedConfig(quant_method="fp8"))
        self.assertIsInstance(model.grouped, FineGrainedGroupedLinear)
        self.assertNotIsInstance(model.plain, FineGrainedGroupedLinear)

    # A quantized embedding TABLE (Qwen4-Exp's n-gram table): FP8 rows with one per-tensor scale,
    # rescaled on the rows a lookup gathers; swapped in for the names in `modules_to_convert`.
    def test_lookup_rescales_the_gathered_rows(self):
        torch.manual_seed(0)
        table = torch.randn(64, 16)
        scale = table.abs().max() / 448.0
        embedding = fg.FineGrainedEmbedding(64, 16)
        embedding.weight = torch.nn.Parameter((table / scale).to(torch.float8_e4m3fn), requires_grad=False)
        embedding.weight_scale = torch.nn.Parameter(scale.to(torch.bfloat16).reshape(1), requires_grad=False)
        ids = torch.tensor([[3, 7], [60, 0]])
        out = embedding(ids)
        self.assertEqual(out.dtype, torch.bfloat16)
        expected = (table / scale).to(torch.float8_e4m3fn)[ids].to(torch.bfloat16) * scale.to(torch.bfloat16)
        torch.testing.assert_close(out, expected)

    def test_replacement_follows_the_patterns_and_the_skip_list(self):
        model = torch.nn.Module()
        model.ngram_embedding = torch.nn.Embedding(8, 4)
        model.embed_tokens = torch.nn.Embedding(8, 4)
        model.other = torch.nn.Embedding(8, 4)
        fg.replace_with_finegrained_embedding(model, ["ngram_embedding", "other"], modules_to_not_convert=["other"])
        self.assertIsInstance(model.ngram_embedding, fg.FineGrainedEmbedding)
        self.assertTrue(model.ngram_embedding._hf_quantized_needs_local_tp)
        self.assertIs(type(model.embed_tokens), torch.nn.Embedding)
        self.assertIs(type(model.other), torch.nn.Embedding)

    # A scale's dtype is the format's, never the ambient default.
    #
    # `from_pretrained` sets the default dtype to the checkpoint's for the duration of model
    # construction, so a scale allocated as `torch.ones(n)` comes out bf16 and the kernels read it
    # as fp32 — NaN logits, no error. Building under both defaults and comparing is what catches
    # that, whatever the format decides each scale should be.
    def _dtypes(self, build, default):
        previous = torch.get_default_dtype()
        torch.set_default_dtype(default)
        try:
            with torch.device("meta"):
                module = build()
            return {name: p.dtype for name, p in module.named_parameters() if p is not None}
        finally:
            torch.set_default_dtype(previous)

    def _assert_default_dtype_invariant(self, build):
        under_fp32 = self._dtypes(build, torch.float32)
        under_bf16 = self._dtypes(build, torch.bfloat16)
        self.assertTrue(under_fp32)
        # the weight and bias DO follow the model dtype; every scale beside them must not
        scales = {name for name in under_fp32 if "scale" in name}
        self.assertTrue(scales)
        for name in sorted(scales):
            self.assertEqual(under_bf16[name], under_fp32[name], f"{name} followed the default dtype")

    def test_linear_scales_ignore_the_default_dtype(self):
        for weight_format, scheme in (("fp8", "static"), ("fp8", "dynamic"), ("nvfp4", "dynamic")):
            with self.subTest(weight_format=weight_format, activation_scheme=scheme):
                self._assert_default_dtype_invariant(
                    lambda f=weight_format, s=scheme: FineGrainedLinear(
                        in_features=64, out_features=32, block_size=(4, 4), weight_format=f, activation_scheme=s
                    )
                )

    def test_experts_scales_ignore_the_default_dtype(self):
        for weight_format, scheme in (("fp8", "static"), ("fp8", "dynamic"), ("nvfp4", "dynamic")):
            with self.subTest(weight_format=weight_format, activation_scheme=scheme):
                self._assert_default_dtype_invariant(
                    lambda f=weight_format, s=scheme: FineGrainedExperts(
                        _Cfg(), block_size=(4, 4), weight_format=f, activation_scheme=s
                    )
                )


@require_torch
class FrozenFp8ShimTest(unittest.TestCase):
    def test_frozen_module_is_self_contained(self):
        import transformers.integrations.finegrained_fp8 as frozen
        from transformers.quantizers.quantizer_finegrained_fp8 import FineGrainedFP8HfQuantizer
        from transformers.utils.quantization_config import FineGrainedFP8Config

        # the quantizer is still constructible — frozen means no new recipes, not removed
        FineGrainedFP8HfQuantizer(FineGrainedFP8Config())
        # distinct machinery: the frozen classes are not the live ones
        self.assertIsNot(frozen.FP8Linear, FineGrainedLinear)


@require_torch
class FineGrainedValidateEnvironmentTest(unittest.TestCase):
    """`validate_environment` on the LIVE quantizer. The only tests that covered a
    `validate_environment` sat on `FineGrainedFP8HfQuantizer`, which no `from_pretrained` can
    reach any more (`AUTO_QUANTIZER_MAPPING["fp8"]` is `FineGrainedHfQuantizer`)."""

    @contextmanager
    def _no_accelerator(self):
        with ExitStack() as stack:
            stack.enter_context(mock.patch("torch.cuda.is_available", return_value=False))
            stack.enter_context(
                mock.patch("transformers.quantizers.finegrained.base.is_torch_xpu_available", return_value=False)
            )
            yield

    def _quantizer(self, pre_quantized, **cfg_kwargs):
        # through the registry, so the test sees the class a `from_pretrained` would pick
        from transformers.quantizers.auto import AUTO_QUANTIZER_MAPPING
        from transformers.utils.quantization_config import FineGrainedConfig

        config = FineGrainedConfig(**cfg_kwargs)
        method = getattr(config.quant_method, "value", config.quant_method)
        quantizer = AUTO_QUANTIZER_MAPPING[method](config)
        quantizer.pre_quantized = pre_quantized
        return quantizer

    def test_no_accelerator_refuses_to_quantize(self):
        # quantizing a bf16 checkpoint needs the hardware; there is nothing to fall back to
        quantizer = self._quantizer(pre_quantized=False)
        with self._no_accelerator(), self.assertRaises(RuntimeError):
            quantizer.validate_environment()

    def test_no_accelerator_dequantizes_a_quantized_checkpoint(self):
        quantizer = self._quantizer(pre_quantized=True)
        with self._no_accelerator():
            quantizer.validate_environment()
        self.assertTrue(quantizer.quantization_config.dequantize)

    def test_nvfp4_refuses_to_dequantize(self):
        # two-level scales have no slot in the dequantize chain, so the CPU fallback that rescues
        # block-FP8 must refuse here rather than hand back a weight scaled by `1 / global`
        for pre_quantized, kwargs in ((True, {}), (False, {"dequantize": True})):
            with self.subTest(pre_quantized=pre_quantized):
                quantizer = self._quantizer(pre_quantized=pre_quantized, quant_method="nvfp4", **kwargs)
                with self._no_accelerator(), self.assertRaises(NotImplementedError):
                    quantizer.validate_environment()


class FineGrainedParallelPlanTest(unittest.TestCase):
    """`update_tp_plan`: a model's plan as a quantized experts module needs it."""

    BASE_EP = {
        "layers.*.mlp.gate": "ep_router",
        "layers.*.mlp.experts.gate_up_proj": "grouped_gemm",
        "layers.*.mlp.experts.down_proj": "grouped_gemm",
        "layers.*.mlp.experts": "moe_tp_experts",
        "layers.*.self_attn.q_proj": "colwise",
    }

    def _planned(self, impl=None, raw=None):
        from types import SimpleNamespace

        from transformers.quantizers.finegrained.base import FineGrainedHfQuantizer
        from transformers.utils.quantization_config import FineGrainedConfig

        quantizer = FineGrainedHfQuantizer(FineGrainedConfig())
        config = SimpleNamespace(base_model_tp_plan=dict(raw or self.BASE_EP), _experts_implementation=impl)
        quantizer.update_tp_plan(config)
        return config.base_model_tp_plan

    def test_every_companion_shards_with_its_expert_weight(self):
        """The plan matcher keys on the exact parameter name and falls back only to the owning
        module, whose entry shards nothing — so a scale with no entry of its own stays whole while
        its weight is this rank's expert slice, and every rank but the first reads another rank's
        scales."""
        from transformers.distributed.tensor_parallel import _get_parameter_tp_plan

        plan = self._planned()

        def style(name):
            return _get_parameter_tp_plan(f"layers.3.mlp.experts.{name}", plan)

        for name in ("gate_up_proj", "gate_up_proj_scale_inv", "gate_up_proj_weight_global_scale"):
            self.assertEqual(style(name), "grouped_gemm", name)
        for name in ("down_proj_scale_inv", "down_proj_bias", "down_proj_input_global_scale"):
            self.assertEqual(style(name), "grouped_gemm", name)
        # one per-tensor value for the pre-routing hidden states: replicated, so no entry of its own
        self.assertEqual(style("gate_up_proj_input_global_scale"), "moe_tp_experts")
        self.assertEqual(plan["layers.*.self_attn.q_proj"], "colwise")

    def test_intra_expert_entries_follow_the_projection_axis(self):
        """Under intra-expert TP the experts stay whole and the projection's own axis splits, so
        the scale grid must follow its weight while the per-expert globals stay replicated. And
        `packed_colwise` splits the output axis as two packed halves — the `[gate; up]` STACK —
        where a quantized module holds interleaved rows, so the weight's own style changes too."""
        from transformers.distributed.tensor_parallel import _get_parameter_tp_plan

        base = {
            "layers.*.mlp.experts.gate_up_proj": "packed_colwise",
            "layers.*.mlp.experts.down_proj": "rowwise",
            "layers.*.mlp.experts": "moe_tp_experts",
        }
        plan = self._planned(raw=base)

        def style(name):
            return _get_parameter_tp_plan(f"layers.3.mlp.experts.{name}", plan)

        # the weight keeps the style the MODEL declared — `packed_colwise` is the strided split a
        # concatenated `[gate; up]` checkpoint needs, since the interleave into the kernels' row
        # order runs on each rank's shard, after sharding. The companions follow it.
        self.assertEqual(style("gate_up_proj"), "packed_colwise")
        for name in ("gate_up_proj_scale_inv", "gate_up_proj_bias"):
            self.assertEqual(style(name), "moe_experts_packed_colwise", name)
        # the down scale splits on dim 2 — the reduce axis affine and 5-D swizzled alike
        self.assertEqual(style("down_proj"), "rowwise")
        self.assertEqual(style("down_proj_scale_inv"), "moe_experts_rowwise")
        # TP does not split experts, so anything indexed per expert stays whole
        for name in ("gate_up_proj_weight_global_scale", "down_proj_weight_global_scale", "down_proj_bias"):
            self.assertEqual(style(name), "moe_tp_experts", name)

        # a natively INTERLEAVED layout declares `colwise`, and its companions split contiguously
        inter = dict(base, **{"layers.*.mlp.experts.gate_up_proj": "colwise"})
        plan_i = self._planned(raw=inter)
        self.assertEqual(_get_parameter_tp_plan("layers.3.mlp.experts.gate_up_proj", plan_i), "colwise")
        self.assertEqual(
            _get_parameter_tp_plan("layers.3.mlp.experts.gate_up_proj_scale_inv", plan_i),
            "moe_experts_colwise",
        )

    def test_dense_projections_that_share_a_name_are_left_alone(self):
        """`mlp.up_proj` and `mlp.shared_experts.down_proj` end in the same words as the stacked
        expert projections but are plain 2-D linears that colwise/rowwise already shard right.
        Rewriting them to an expert-axis style sharded a dim they do not have — a CUDA illegal
        memory access in the dense FP8 matmul, seen on DeepSeek-V3 under TP."""
        from transformers.distributed.tensor_parallel import _get_parameter_tp_plan

        plan = self._planned(
            raw={
                "layers.*.mlp.experts.gate_up_proj": "packed_colwise",
                "layers.*.mlp.experts.down_proj": "rowwise",
                "layers.*.mlp.experts": "moe_tp_experts",
                "layers.*.mlp.up_proj": "colwise",
                "layers.*.mlp.down_proj": "rowwise",
                "layers.*.mlp.shared_experts.up_proj": "colwise",
                "layers.*.mlp.shared_experts.down_proj": "rowwise",
            }
        )
        for dense, expected in (
            ("layers.*.mlp.up_proj", "colwise"),
            ("layers.*.mlp.down_proj", "rowwise"),
            ("layers.*.mlp.shared_experts.up_proj", "colwise"),
            ("layers.*.mlp.shared_experts.down_proj", "rowwise"),
        ):
            self.assertEqual(plan[dense], expected, dense)
        # and no scale entry was invented for a dense linear, whose scale `rowwise`/`colwise`
        # already reaches through the module fallback
        strays = [k for k in plan if k.endswith("_scale_inv") and ".experts." not in k]
        self.assertEqual(strays, [], f"invented dense scale entries: {strays}")
        # the stacked experts still get theirs
        self.assertEqual(
            _get_parameter_tp_plan("layers.3.mlp.experts.gate_up_proj_scale_inv", plan),
            "moe_experts_packed_colwise",
        )

    def test_a_multimodal_models_expert_plans_are_reached(self):
        """A multimodal model keeps its experts' plans on a SUB-config; the config the quantizer
        is handed carries a few projector entries and often no `base_model_ep_plan` at all.
        Rewriting only what we were handed adds no companion anywhere, while the weights still
        shard from the sub-config's own plan — so the scale stays whole against a sharded weight,
        which is a wrong answer rather than a crash."""
        from transformers import Glm4vMoeConfig
        from transformers.quantizers.finegrained.base import FineGrainedHfQuantizer
        from transformers.utils.quantization_config import FineGrainedConfig

        config = Glm4vMoeConfig()
        self.assertFalse(
            getattr(config, "base_model_ep_plan", None),
            "this model no longer nests its plans, so it cannot guard the sub-config walk",
        )
        before = dict(getattr(config.text_config, "base_model_ep_plan", None) or {})
        FineGrainedHfQuantizer(FineGrainedConfig()).update_tp_plan(config)
        after = getattr(config.text_config, "base_model_ep_plan", None) or {}

        companions = {
            k
            for k in after
            if k.endswith(("_scale_inv", "_bias", "_weight_global_scale", "_input_global_scale", "_activation_scale"))
        }
        self.assertTrue(companions, "no companion reached the sub-config's plan")
        self.assertEqual(set(before), set(after) - companions, "the model's own entries were disturbed")

    def test_dequantize_folds_every_quantized_key_to_full_precision(self):
        """`dequantize=True` is the escape hatch for hardware that cannot serve the format. It
        must claim BOTH checkpoint key layouts — the `{proj}_blocks` + `{proj}_scales` pair
        GPT-OSS ships and the `weight` + `weight_scale_inv` pair everything else does — and
        land on the plain weight, since a module that was never swapped has no scale slot to
        write into."""

        from transformers.utils.quantization_config import FineGrainedConfig

        for method, sources in (
            ("mxfp4", ("experts.gate_up_proj_blocks", "experts.gate_up_proj_scales")),
            ("fp8", ("mlp.down_proj.weight", "mlp.down_proj.weight_scale_inv")),
        ):
            quantizer = _quantizer_for(FineGrainedConfig(quant_method=method, dequantize=True))
            quantizer.pre_quantized = True
            converters = quantizer.get_weight_conversions()
            with self.subTest(quant_method=method):
                self.assertTrue(converters, f"{method}: dequantize produced no converter")
                for source in sources:
                    self.assertTrue(
                        any(
                            re.search(str(pattern), source)
                            for c in converters
                            for pattern in (
                                c.source_patterns if isinstance(c.source_patterns, list) else [c.source_patterns]
                            )
                        ),
                        f"{method}: nothing dequantizes {source}",
                    )

    def test_the_impl_rewrites_the_layer_kinds(self):
        megamoe = self._planned("deepgemm_megamoe")
        self.assertEqual(megamoe["layers.*.mlp.gate"], "megamoe_router")
        self.assertEqual(megamoe["layers.*.mlp.experts"], "megamoe_experts")

    def test_a_models_own_plan_survives_the_update(self):
        """`update_tp_plan` ADDS companions; it must never drop what the model ships.

        The config here is a REAL one, which the `SimpleNamespace` the tests above use cannot
        substitute for: a branch keyed on the config CLASS is invisible to a fake, and a name
        substring (`"Qwen3" in config.__class__.__name__`) also claims Qwen3-MoE.
        """
        from transformers import Qwen3Config, Qwen3MoeConfig, Qwen3VLMoeConfig
        from transformers.quantizers.finegrained.base import FineGrainedHfQuantizer
        from transformers.utils.quantization_config import FineGrainedConfig

        for config_cls in (Qwen3Config, Qwen3MoeConfig):
            with self.subTest(config=config_cls.__name__):
                config = config_cls()
                shipped = dict(config.base_model_tp_plan)
                self.assertTrue(shipped, f"{config_cls.__name__} ships no plan to preserve")
                FineGrainedHfQuantizer(FineGrainedConfig()).update_tp_plan(config)
                for key, style in shipped.items():
                    self.assertEqual(config.base_model_tp_plan.get(key), style, key)

        # and a multimodal one keeps its plan on the SUB-config: the outer carries none, so
        # nothing may be invented there. 25 config classes contain "Qwen3" — a name-substring
        # branch reaches every one of them, including these.
        multimodal = Qwen3VLMoeConfig()
        FineGrainedHfQuantizer(FineGrainedConfig()).update_tp_plan(multimodal)
        self.assertFalse(getattr(multimodal, "base_model_tp_plan", None) or {})
        self.assertTrue(multimodal.text_config.base_model_tp_plan)

    def test_a_real_moe_models_experts_gain_their_companions(self):
        """The other half of the same invariant: preserving the model's entries is what lets the
        companion pass find `moe_tp_experts` and give the expert scales an entry of their own."""
        from transformers import Qwen3MoeConfig
        from transformers.quantizers.finegrained.base import FineGrainedHfQuantizer
        from transformers.utils.quantization_config import FineGrainedConfig

        config = Qwen3MoeConfig()
        FineGrainedHfQuantizer(FineGrainedConfig()).update_tp_plan(config)
        plan = config.base_model_tp_plan
        self.assertEqual(plan["layers.*.mlp.experts"], "moe_tp_experts")
        for companion in ("gate_up_proj_scale_inv", "down_proj_scale_inv", "gate_up_proj_bias"):
            self.assertIn(f"layers.*.mlp.experts.{companion}", plan, companion)


@require_torch
class FineGrainedConfigTest(unittest.TestCase):
    """The quantization config: formats per module group, and the skip-lists checkpoints ship."""

    @parameterized.expand(
        [
            (weights, activations, activations in supported)
            for weights, supported in (
                ("fp8", (None, "fp8")),
                ("mxfp8", (None, "mxfp8", "mxfp4", "bf16")),
                ("mxfp4", (None, "mxfp8", "mxfp4", "bf16")),
                ("nvfp4", (None, "nvfp4", "bf16")),
            )
            for activations in (None, "bf16", "fp8", "mxfp8", "mxfp4", "nvfp4")
        ]
    )
    def test_an_activation_format_the_weights_cannot_take_fails_at_construction(self, weights, activations, ok):
        """The kernels quantize activations only to what each weight format's MMA takes, and refuse
        the rest at the first forward; the config refuses them up front."""
        from transformers.utils.quantization_config import FineGrainedConfig

        if ok:
            FineGrainedConfig(quant_method=weights, activation_format=activations)
        else:
            with self.assertRaises(ValueError):
                FineGrainedConfig(quant_method=weights, activation_format=activations)

    # A quant config names a format per module SUBSET, because a checkpoint can be more than one.
    def _config(self, **kwargs):
        from transformers.utils.quantization_config import FineGrainedConfig

        return FineGrainedConfig(**kwargs)

    def test_a_flat_config_is_one_group_over_everything(self):
        groups = self._config(quant_method="mxfp8").groups
        self.assertEqual({name: group.quant_method for name, group in groups.items()}, {"default": "mxfp8"})

    def test_a_legacy_expert_dtype_becomes_the_two_groups_it_meant(self):
        """DeepSeek-V4 ships `quant_method="fp8"` and declares its mxfp4 experts on the MODEL
        config, because a flat config cannot say "the experts differ". `shared_experts` is DENSE
        despite the name, which is why the pattern matches a path segment and not a substring."""
        from transformers.utils.quantization_config import groups_with_expert_dtype

        config = self._config(quant_method="fp8", weight_block_size=(128, 128), scale_fmt="ue8m0")
        config.groups = groups_with_expert_dtype(config.groups, "fp4")
        for module, expected in (
            ("model.layers.3.mlp.experts", "mxfp4"),
            ("model.layers.3.mlp.experts.gate_up_proj", "mxfp4"),
            ("model.layers.3.self_attn.q_proj", "fp8"),
            ("model.layers.3.mlp.shared_experts.up_proj", "fp8"),
        ):
            self.assertEqual(config.group_for(module).quant_method, expected, module)

    def test_resolution_survives_a_round_trip_through_config_json(self):
        """`to_json_string` SORTS keys, so a config read back has lost the order it was written
        in. Resolution must not depend on it: the catch-all sorts first here and would swallow
        every module if targeted groups were not preferred outright."""
        import json

        from transformers.utils.quantization_config import FineGrainedConfig, FineGrainedGroup

        config = FineGrainedConfig(
            groups={
                "experts": FineGrainedGroup(quant_method="nvfp4", targets=[r"\.experts($|\.)"]),
                "dense": FineGrainedGroup(quant_method="fp8", weight_block_size=(128, 128)),
            }
        )
        back = FineGrainedConfig.from_dict(json.loads(config.to_json_string()))
        self.assertEqual(back.group_for("model.layers.0.mlp.experts").quant_method, "nvfp4")
        self.assertEqual(back.group_for("model.layers.0.self_attn.q_proj").quant_method, "fp8")
        # a list survives JSON where a tuple does not, and the block size is compared as a tuple
        self.assertEqual(back.groups["dense"].weight_block_size, (128, 128))

    def test_two_targeted_groups_claiming_one_module_is_an_error(self):
        from transformers.utils.quantization_config import FineGrainedGroup

        config = self._config(
            groups={
                "a": FineGrainedGroup(quant_method="fp8", targets=[r"\.experts"]),
                "b": FineGrainedGroup(quant_method="nvfp4", targets=[r"mlp\."]),
            }
        )
        with self.assertRaises(ValueError):
            config.group_for("model.layers.0.mlp.experts")

    def test_a_producers_config_groups_are_normalized(self):
        """modelopt describes a format by its parameters; we name it. GLM-5.2-NVFP4 ships exactly
        this, and `targets: ["Linear"]` is its spelling of the catch-all."""
        config = self._config(
            quant_method="modelopt",
            quant_algo="NVFP4",
            config_groups={
                "group_0": {
                    "weights": {"num_bits": 4, "type": "float", "group_size": 16},
                    "input_activations": {"num_bits": 4, "type": "float", "group_size": 16, "dynamic": False},
                    "targets": ["Linear"],
                }
            },
        )
        group = config.group_for("model.layers.0.self_attn.q_proj")
        self.assertEqual((group.quant_method, group.activation_format), ("nvfp4", "nvfp4"))
        # NVFP4's `dynamic: False` is its calibrated activation global, not a static activation scale
        self.assertEqual(group.activation_scheme, "dynamic")

    def test_a_format_we_do_not_serve_falls_back_rather_than_guessing(self):
        """Half a translation would quantize modules by a rule the producer did not write."""
        config = self._config(
            quant_method="modelopt",
            quant_algo="NVFP4",
            config_groups={
                "group_0": {"weights": {"num_bits": 3, "type": "int", "group_size": 64}, "targets": ["Linear"]}
            },
        )
        self.assertEqual(list(config.groups), ["default"])

    def test_a_checkpoints_glob_skip_list_is_bounded(self):
        """modelopt names skipped subtrees with globs. Passed through as regexes the dots match any
        character and the star is greedy, so "model.layers.1.*" also takes layers 10-19."""
        from transformers.quantizers.quantizers_utils import should_convert_module
        from transformers.utils.quantization_config import FineGrainedConfig

        # as a modelopt checkpoint ships them, through the config that normalizes them
        skip = FineGrainedConfig(
            quant_method="modelopt",
            quant_algo="NVFP4",
            ignore=["model.layers.0*", "model.layers.1.*", "self_attn"],
        ).modules_to_not_convert
        for name in ("model.layers.0.mlp", "model.layers.1.mlp.down_proj", "model.layers.2.self_attn.q_proj"):
            self.assertFalse(should_convert_module(name, skip), name)
        for name in ("model.layers.16.mlp.down_proj", "model.layers.2.mlp.gate_proj", "model.my_self_attn.q_proj"):
            self.assertTrue(should_convert_module(name, skip), name)
