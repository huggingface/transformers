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
"""NVFP4: E2M1 weights two per byte with an E4M3 scale per 16 and an fp32 global, in the layouts NVIDIA modelopt
exports (per expert per projection, or stacked per layer); quantizes a full-precision checkpoint on the fly too."""

from typing import TYPE_CHECKING

from .block_scaled_utils import BlockScaledQuantizer
from .quantizers_utils import FUSED_STACKS, get_module_from_name


if TYPE_CHECKING:
    from ..utils.quantization_config import NVFP4Config


class NVFP4HfQuantizer(BlockScaledQuantizer):
    """NVFP4, in both key layouts modelopt ships: per expert per projection, or stacked per layer."""

    quantization_config: "NVFP4Config"
    # modelopt calibrates an FP8 KV cache beside the weights; the cache here stays in the model dtype
    unexpected_keys = (r"\.(k_proj\.k|v_proj\.v)_scale$",)

    def finish_modules(self, model):
        from ..integrations.finegrained import register_shared_gate_up_input_globals

        register_shared_gate_up_input_globals(model)

    def format_conversions(self, dequantize):
        # weight-only keeps activations bf16, so there is no activation global for `input_scale` to land on; nor
        # does a dequantized model, which leaves those keys to the unexpected-key filter
        calibrated = not dequantize and self.quantization_config.activation_format != "bf16"
        conversions = self._nvfp4_renames(calibrated)
        if calibrated:
            # the down projection's is read with its globals
            conversions += self._activation_scale_conversions(
                projections=self.up_projections, ops=self._reciprocal_ops
            )
        if dequantize:
            # a stacked layout's scales reach its stack's dequantize converter under the slot names it collects; the
            # experts are left as they are, so every stack a model may hold is covered
            conversions += self._nvfp4_fused_renames(
                FUSED_STACKS, (("_weight_scale", "_scale_inv"), ("_weight_scale_2", "_weight_global_scale"))
            )
        elif self.holds_experts:
            conversions += self._nvfp4_per_expert_conversions(calibrated) + self._nvfp4_fused_conversions(calibrated)
        return conversions

    def get_param_ops(self, model, param_name):
        """The packed weights first take the int8 view the modules hold: the loader keeps a pre-quantized
        non-float tensor in the checkpoint's dtype, modelopt's uint8, and the reverse saves it back as that."""
        from ..integrations.finegrained import FineGrainedLinear
        from ..integrations.finegrained.conversions import FineGrainedViewPackedInt8
        from ..integrations.finegrained.layers import _FineGrainedModule

        ops = super().get_param_ops(model, param_name)
        module, name = get_module_from_name(model, param_name)
        if isinstance(module, _FineGrainedModule) and name in ("weight", *FUSED_STACKS):
            ops = [FineGrainedViewPackedInt8(self), *ops]
        # an experts module's globals are inverted in their converters, before the per-expert reductions
        elif isinstance(module, FineGrainedLinear) and name in ("weight_global_scale", "activation_scale"):
            ops = [*self._reciprocal_ops, *ops]
        return ops

    @property
    def _reciprocal_ops(self) -> list:
        """compressed-tensors ships the globals as their reciprocal; modelopt as the multiplier the kernels take."""
        from ..integrations.finegrained.conversions import FineGrainedReciprocal

        return (
            [FineGrainedReciprocal(self)] if self.quantization_config.checkpoint_format == "compressed-tensors" else []
        )

    def _nvfp4_renames(self, calibrated: bool):
        """modelopt's scales onto the slots every module holds them in, experts included: renamed before converter
        collection, a block scale then rides its weight's converter as `weight_scale_inv` (the merge that stacks
        the experts' weights stacks their grids too), and the globals reach the converters below by slot name."""
        from ..core_model_loading import WeightRenaming

        renames = [
            self._block_scale_rename(),
            WeightRenaming(source_patterns=r"^(.+)\.weight_scale_2", target_patterns=r"\1.weight_global_scale"),
        ]
        if self.quantization_config.checkpoint_format == "compressed-tensors":
            # its packed values ship as `weight_packed` and its activation global as `input_global_scale`; its weight
            # global already carries the slot's name
            renames.append(WeightRenaming(source_patterns=r"^(.+)\.weight_packed", target_patterns=r"\1.weight"))
            if calibrated:
                renames.append(
                    WeightRenaming(
                        source_patterns=r"^(.+)\.input_global_scale", target_patterns=r"\1.activation_scale"
                    )
                )
        return renames

    def _nvfp4_per_expert_conversions(self, calibrated: bool):
        """One key per expert per projection, as GLM-5.2-NVFP4 ships them.

        `MergeModulelist` stamps each source with its expert index, which is what lets expert
        parallelism keep only this rank's experts (`tensor_idx` in `core_model_loading`); without
        it every rank collects all E values and the forward asserts on the per-expert count.
        """
        from ..core_model_loading import MergeModulelist, WeightConverter
        from ..integrations.finegrained.conversions import FineGrainedWeightGlobals

        return [
            # every second-level global of a layer in ONE converter: folding the gate|up stack's
            # two halves into one moves the up half's onto the down projection, so they decide
            # together (`FineGrainedWeightGlobals`)
            WeightConverter(
                source_patterns=[
                    *self._per_expert_sources(up, "weight_global_scale"),
                    *self._per_expert_sources("down_proj", "weight_global_scale"),
                    *(self._per_expert_sources("down_proj", "activation_scale") if calibrated else []),
                ],
                target_patterns=[
                    f"experts.{up}_weight_global_scale",
                    "experts.down_proj_weight_global_scale",
                    *(["experts.down_proj_activation_scale"] if calibrated else []),
                ],
                operations=[MergeModulelist(dim=0), *self._reciprocal_ops, FineGrainedWeightGlobals(self)],
            )
            for up in self.up_projections
        ]

    @staticmethod
    def _nvfp4_fused_renames(projections, slots) -> list:
        """The stacked layout's `{proj}{shipped}` scales onto the `{proj}{slot}` slots, for each `(shipped, slot)`."""
        from ..core_model_loading import WeightRenaming

        return [
            WeightRenaming(source_patterns=rf"experts\.{proj}{shipped}", target_patterns=f"experts.{proj}{slot}")
            for proj in projections
            for shipped, slot in slots
        ]

    def _nvfp4_fused_conversions(self, calibrated: bool):
        """Already stacked per layer, as the vLLM fused layout ships them — so no `MergeModulelist`:
        the scales only take their slot's name, and the module's ops give every tensor its layout."""
        from ..core_model_loading import WeightConverter
        from ..integrations.finegrained.conversions import FineGrainedWeightGlobals

        converters = self._nvfp4_fused_renames((*self.up_projections, "down_proj"), (("_weight_scale", "_scale_inv"),))
        converters += [
            # the same one-converter rule as the per-expert layout, over the fused keys
            WeightConverter(
                source_patterns=[
                    rf"experts\.{up}_weight_scale_2",
                    r"experts\.down_proj_weight_scale_2",
                    *([r"experts\.down_proj_input_scale"] if calibrated else []),
                ],
                target_patterns=[
                    f"experts.{up}_weight_global_scale",
                    "experts.down_proj_weight_global_scale",
                    *(["experts.down_proj_activation_scale"] if calibrated else []),
                ],
                operations=[*self._reciprocal_ops, FineGrainedWeightGlobals(self)],
            )
            for up in self.up_projections
        ]
        return converters
