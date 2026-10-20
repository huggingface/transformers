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
"""Tests for NVFP4 (`NVFP4Config`).

They cover the two-level global-scale ops, the modelopt and compressed-tensors layouts it loads and saves back, and
how those exports read.
"""

import unittest

import torch
from parameterized import parameterized

from transformers.integrations.finegrained import FineGrainedExperts
from transformers.quantizers.auto import AutoQuantizationConfig
from transformers.testing_utils import (
    TestCasePlus,
    require_torch,
)
from transformers.utils.quantization_config import NVFP4Config

from ...integrations.finegrained.utils import (
    E,
    _Cfg,
    _ChainAssertions,
    _load_and_save,
    _per_expert,
    _quantizer_for,
    _targets,
)
from ..block_scaled_checkpoint_tester import TinyCheckpointTesterMixin
from ..block_scaled_parallel_tester import LoadPathEquivalenceTesterMixin
from ..block_scaled_tester import (
    NVFP4,
    BlockScaledFormatTesterMixin,
)


@require_torch
class Nvfp4GlobalScaleOpsTest(unittest.TestCase):
    """NVFP4 second-level globals and calibrated input scales, in the layout the kernels index.

    The kernels take one per expert per projection, with the gate/up pair folded.
    """

    @parameterized.expand(
        [
            (
                # modelopt calibrates the gate and up halves separately. The stack keeps the gate's global.
                # The up half's global moves onto the down projection, since SwiGLU is linear in the up half.
                # The down input scale moves the other way, so the intermediate keeps its calibrated range.
                "per_expert_halves",
                {
                    "experts.*.up_proj.weight_scale_2": [3.0, 4.0],
                    "experts.*.gate_proj.weight_scale_2": [1.0, 2.0],
                    "experts.*.down_proj.weight_scale_2": [5.0, 6.0],
                    "experts.*.down_proj.input_scale": [0.5, 2.0],
                },
                {"gate_up_proj_weight_global_scale": [1.0, 2.0], "down_proj_weight_global_scale": [15.0, 12.0],
                 "down_proj_activation_scale": [0.5 / 3, 2.0 / 2]},
            ),
            (
                # vLLM's stacked layout ships the pair as one `(E, 2)` tensor
                "fused_pair",
                {"experts.gate_up_proj_weight_scale_2": [[1.0, 3.0], [2.0, 4.0]],
                 "experts.down_proj_weight_scale_2": [5.0, 6.0]},
                {"gate_up_proj_weight_global_scale": [1.0, 2.0], "down_proj_weight_global_scale": [15.0, 12.0]},
            ),
            (
                # Calibrated as one matrix, so there is nothing to merge
                "one_calibrated_projection",
                {"experts.gate_up_proj_weight_scale_2": [[1.0], [2.0]], "experts.down_proj_weight_scale_2": [3.0, 4.0],
                 "experts.down_proj_input_scale": [0.5, 0.25]},
                {"gate_up_proj_weight_global_scale": [1.0, 2.0], "down_proj_weight_global_scale": [3.0, 4.0],
                 "down_proj_activation_scale": [0.5, 0.25]},
            ),
        ]
    )  # fmt: skip
    def test_one_converter_folds_a_layers_globals_to_one_per_expert(self, _, sources, expected):
        """Check that one converter folds a layer's globals to one per expert and projection."""
        from transformers.integrations.finegrained.conversions import FineGrainedWeightGlobals

        targets = [f"experts.{name}" for name in expected]
        for wrapped in (False, True):
            # The loader passes tensors in a list. Unwrapped, an `(E, 2)` pair would read as `(1, 2E)`.
            inputs = {key: [torch.tensor(v)] if wrapped else torch.tensor(v) for key, v in sources.items()}
            out = FineGrainedWeightGlobals().convert(inputs, target_patterns=targets)
            for target in targets:
                with self.subTest(target=target, wrapped=wrapped):
                    torch.testing.assert_close(out[target], torch.tensor(expected[target.removeprefix("experts.")]))

    def test_activation_global_is_per_expert_for_both_projections(self):
        """Check that each expert keeps its calibrated `input_scale`. The gate/up pair reduces to its max."""
        from transformers.integrations.finegrained.conversions import FineGrainedInputScales

        up = FineGrainedInputScales().convert(
            {
                "mlp.experts.*.gate_proj.input_scale": torch.tensor([1.0, 3.0]),
                "mlp.experts.*.up_proj.input_scale": torch.tensor([2.0, 1.0]),
            },
            full_layer_name="mlp.experts.gate_up_proj_activation_scale",
        )["mlp.experts.gate_up_proj_activation_scale"]
        torch.testing.assert_close(up, torch.tensor([2.0, 3.0]))
        down = FineGrainedInputScales().convert(
            {"mlp.experts.*.down_proj.input_scale": torch.tensor([1.0, 4.0])},
            full_layer_name="mlp.experts.down_proj_activation_scale",
        )["mlp.experts.down_proj_activation_scale"]
        torch.testing.assert_close(down, torch.tensor([1.0, 4.0]))


@require_torch
class Nvfp4ChainTest(_ChainAssertions, unittest.TestCase):
    """NVFP4 as modelopt ships it, per expert (GLM-5.2-NVFP4), stacked per layer (vLLM) and on dense linears.

    `MergeModulelist` matters beyond the merge. `core_model_loading` stamps a source with its expert index only
    when the chain has one. Expert parallelism selects experts by that index.
    """

    def _modelopt_checkpoint(self):
        """Return modelopt's per-expert layout, with a dense linear.

        It has packed E2M1 bytes, E4M3 block scales per 16, an fp32 `weight_scale_2` and a calibrated `input_scale`.
        Gate and up share their globals, as modelopt's linear fusion exports them.
        A save then writes each back exactly.
        """
        torch.manual_seed(0)
        checkpoint = _per_expert(
            lambda n, k: {
                "weight": torch.randint(0, 256, (n, k // 2), dtype=torch.uint8),
                "weight_scale": (torch.rand(n, k // 16) + 0.5).to(torch.float8_e4m3fn),
                "weight_scale_2": torch.rand(()) * 1e-3 + 1e-4,
                "input_scale": torch.rand(()) * 1e-2 + 1e-3,
            }
        )
        for expert in range(E):
            prefix = f"model.layers.0.mlp.experts.{expert}"
            for suffix in ("weight_scale_2", "input_scale"):
                checkpoint[f"{prefix}.up_proj.{suffix}"] = checkpoint[f"{prefix}.gate_proj.{suffix}"].clone()
        return checkpoint

    def test_modelopt_per_expert_and_dense(self):
        checkpoint = self._modelopt_checkpoint()
        model, info, saved = _load_and_save(
            checkpoint, AutoQuantizationConfig.from_dict({"quant_method": "modelopt", "quant_algo": "NVFP4"})
        )
        self.assertLoadsCleanly(info)
        experts, q_proj = model.model.layers[0].mlp.experts, model.model.layers[0].self_attn.q_proj
        self.assertEqual((experts.gate_up_proj.dtype, q_proj.weight.dtype), (torch.int8, torch.int8))
        self.assertHoldsInterleaved(experts, checkpoint, "weight", experts.gate_up_proj)
        torch.testing.assert_close(
            experts.gate_up_proj_weight_global_scale,
            torch.stack([checkpoint[f"model.layers.0.mlp.experts.{e}.gate_proj.weight_scale_2"] for e in range(E)]),
        )
        for slot, key in (
            ("weight_global_scale", "weight_scale_2"),
            ("activation_scale", "input_scale"),
        ):
            torch.testing.assert_close(getattr(q_proj, slot), checkpoint[f"model.layers.0.self_attn.q_proj.{key}"])
        self.assertSavesBack(saved, checkpoint)

    def test_compressed_tensors_per_expert_and_dense(self):
        """Check that the compressed-tensors layout loads into the modelopt slots and saves back as shipped.

        compressed-tensors names the packed values `weight_packed` and the activation global `input_global_scale`.
        It stores both globals as the reciprocal of modelopt's. Power-of-two globals keep the reciprocal exact.
        """
        modelopt = self._modelopt_checkpoint()
        checkpoint = {}
        for key, tensor in modelopt.items():
            stem, _, suffix = key.rpartition(".")
            if suffix in ("weight_scale_2", "input_scale"):
                modelopt[key] = tensor = torch.exp2(torch.round(torch.log2(tensor)))
                name = "weight_global_scale" if suffix == "weight_scale_2" else "input_global_scale"
                checkpoint[f"{stem}.{name}"] = 1 / tensor
            else:
                checkpoint[f"{stem}.weight_packed" if suffix == "weight" and tensor.dtype == torch.uint8 else key] = (
                    tensor
                )
        config = NVFP4Config(activation_format="nvfp4", checkpoint_format="compressed-tensors")
        model, info, saved = _load_and_save(checkpoint, config)
        self.assertLoadsCleanly(info)
        experts, q_proj = model.model.layers[0].mlp.experts, model.model.layers[0].self_attn.q_proj
        torch.testing.assert_close(
            experts.gate_up_proj_weight_global_scale,
            torch.stack([modelopt[f"model.layers.0.mlp.experts.{e}.gate_proj.weight_scale_2"] for e in range(E)]),
        )
        torch.testing.assert_close(
            q_proj.weight_global_scale, modelopt["model.layers.0.self_attn.q_proj.weight_scale_2"]
        )
        torch.testing.assert_close(q_proj.activation_scale, modelopt["model.layers.0.self_attn.q_proj.input_scale"])
        self.assertSavesBack(saved, checkpoint)

    def test_modelopt_fused_layout(self):
        """The vLLM layout, with the per-expert modelopt tensors stacked per layer."""
        per_expert = self._modelopt_checkpoint()
        checkpoint = {k: v for k, v in per_expert.items() if ".experts." not in k}
        for proj, halves in (("gate_up_proj", ("gate_proj", "up_proj")), ("down_proj", ("down_proj",))):
            for suffix, stacked in (("", "weight"), ("_weight_scale", "weight_scale")):
                checkpoint[f"model.layers.0.mlp.experts.{proj}{suffix}"] = torch.stack(
                    [
                        torch.cat([per_expert[f"model.layers.0.mlp.experts.{e}.{h}.{stacked}"] for h in halves])
                        for e in range(E)
                    ]
                )
            # Globals stack one per expert. Gate and up agree here, so one value covers both halves.
            # Pairs calibrated apart are covered by the global-scale op tests.
            for suffix in ("weight_scale_2", "input_scale"):
                checkpoint[f"model.layers.0.mlp.experts.{proj}_{suffix}"] = torch.stack(
                    [per_expert[f"model.layers.0.mlp.experts.{e}.{halves[0]}.{suffix}"] for e in range(E)]
                )
        model, info, saved = _load_and_save(
            checkpoint, AutoQuantizationConfig.from_dict({"quant_method": "modelopt", "quant_algo": "NVFP4"})
        )
        self.assertLoadsCleanly(info)
        self.assertEqual(model.model.layers[0].mlp.experts.gate_up_proj.dtype, torch.int8)
        self.assertSavesBack(saved, checkpoint)

    def _modelopt_conversions(self, **cfg_kwargs):
        # A plain modelopt NVFP4 export reads as this config
        config = NVFP4Config(**{"activation_format": "nvfp4", **cfg_kwargs})
        return _quantizer_for(config).format_conversions(dequantize=False)

    def test_every_declared_target_is_a_slot_the_module_holds(self):
        """Check that every converter target is a slot the module holds.

        A target that no module holds fails the load. In a weight-only run the experts hold no activation global.
        Both runs are checked against real modules' parameter names.
        """
        from transformers.integrations.finegrained import FineGrainedLinear

        cfg = _Cfg()
        for activation_format, holds_global in ((None, True), ("bf16", False)):
            experts = FineGrainedExperts(
                cfg,
                quantization_config=NVFP4Config(activation_format=activation_format),
            )
            linear = FineGrainedLinear(32, 32, quantization_config=NVFP4Config(activation_format=activation_format))
            for module in (experts, linear):
                slots = {name for name, _ in module.named_parameters()}
                self.assertEqual(
                    any("activation_scale" in slot for slot in slots),
                    holds_global,
                    f"{type(module).__name__} with activation_format={activation_format!r} disagrees about the slot",
                )
            for c in self._modelopt_conversions(activation_format=activation_format):
                for target in _targets(c):
                    # Expert targets name the experts' slots. Dense renames name the linear's.
                    module, slot = (
                        (experts, target.split("experts.")[-1])
                        if "experts." in target
                        else (linear, target.removeprefix("\\1."))
                    )
                    self.assertIn(
                        slot.rstrip("$"),
                        {name for name, _ in module.named_parameters()},
                        f"activation_format={activation_format!r} converts onto a missing slot",
                    )

    def test_global_scale_converters_carry_an_expert_index(self):
        # Only the per-expert layout (`experts.*.`) needs an index. A checkpoint with one stacked tensor per
        # layer is sharded like the fused expert weight next to it.
        from transformers.core_model_loading import MergeModulelist, WeightConverter, WeightRenaming, rename_source_key

        conversions = self._modelopt_conversions()
        renamings = [c for c in conversions if isinstance(c, WeightRenaming)]
        converters = [c for c in conversions if isinstance(c, WeightConverter)]
        _, pattern = rename_source_key("model.layers.0.mlp.experts.7.gate_proj.weight_scale_2", renamings, converters)
        globals_converters = [c for c in converters if pattern in c.source_patterns]
        self.assertTrue(globals_converters, "no global-scale converter found")
        for conv in globals_converters:
            self.assertTrue(
                any(isinstance(op, MergeModulelist) for op in conv.operations),
                f"{_targets(conv)} has no MergeModulelist, so expert parallelism cannot select "
                "experts and every rank would collect all of them",
            )


class Nvfp4Test(BlockScaledFormatTesterMixin, unittest.TestCase):
    spec = NVFP4

    def test_the_config_takes_its_main_arguments_positionally(self):
        config = NVFP4Config(["lm_head"])
        self.assertEqual(config.modules_to_not_convert, ["lm_head"])


class Nvfp4TinyCheckpointTest(TinyCheckpointTesterMixin, TestCasePlus):
    spec = NVFP4


class Nvfp4LoadPathEquivalenceTest(LoadPathEquivalenceTesterMixin, TestCasePlus):
    spec = NVFP4
