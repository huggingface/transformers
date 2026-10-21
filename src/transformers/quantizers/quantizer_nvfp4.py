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
"""NVFP4 quantizer for E2M1 weights with one E4M3 scale per 16 values and an fp32 global scale.

It loads both layouts NVIDIA ModelOpt exports, per expert per projection or stacked per layer.
It also quantizes a full-precision checkpoint on the fly.
"""

from typing import TYPE_CHECKING

from .block_scaled_utils import BlockScaledQuantizer, ProducerLayout
from .quantizers_utils import FUSED_STACKS


if TYPE_CHECKING:
    from ..utils.quantization_config import NVFP4Config


class NVFP4HfQuantizer(BlockScaledQuantizer):
    """NVFP4 weights in both key layouts that ModelOpt ships."""

    quantization_config: "NVFP4Config"
    # ModelOpt also calibrates an FP8 KV cache. Here the cache stays in the model dtype.
    unexpected_keys = (r"\.(k_proj\.k|v_proj\.v)_scale$",)
    # NVFP4 checkpoints, transformers saves included, use the ModelOpt layout
    producer_layouts = {None: ProducerLayout(block_scale_name="weight_scale", packed_as_uint8=True)}

    def get_weight_conversions(self, model=None):
        from ..core_model_loading import WeightRenaming

        conversions = super().get_weight_conversions(model)
        if not self.pre_quantized:
            return conversions
        dequantize, up_projections = self.quantization_config.dequantize, self._up_projections(model)
        # Weight-only and dequantized models have no activation global to load `input_scale` into
        calibrated = not dequantize and self.quantization_config.activation_format != "bf16"
        # rename the weight globals to the slot every module holds them in, experts included
        conversions.append(
            WeightRenaming(source_patterns=r"^(.+)\.weight_scale_2", target_patterns=r"\1.weight_global_scale")
        )
        if calibrated:
            # The down projection's activation scale loads with its weight globals
            conversions += self._activation_scale_conversions(up_projections, projections=up_projections)
        if dequantize:
            # Rename stacked scales to the slot names each stack's dequantize converter collects.
            # Cover every stack a model may hold, without looking at its experts.
            conversions += self._fused_renames(
                FUSED_STACKS, (("_weight_scale", "_scale_inv"), ("_weight_scale_2", "_weight_global_scale"))
            )
        elif up_projections:
            conversions += self._per_expert_conversions(calibrated, up_projections)
            conversions += self._fused_conversions(calibrated, up_projections)
        return conversions

    def _per_expert_conversions(self, calibrated: bool, up_projections: tuple[str, ...]):
        """Return converters for one key per expert per projection, as GLM-5.2-NVFP4 ships them.

        `MergeModulelist` lets expert parallelism keep only this rank's experts.
        """
        from ..core_model_loading import MergeModulelist, WeightConverter
        from ..integrations.finegrained.conversions import FineGrainedWeightGlobals

        return [
            # One converter for all of a layer's globals.
            # `FineGrainedWeightGlobals` folds the gate and up halves. This moves the up global to the down projection.
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
                operations=[MergeModulelist(dim=0), FineGrainedWeightGlobals(self)],
            )
            for up in up_projections
        ]

    @staticmethod
    def _fused_renames(projections, slots) -> list:
        """Rename the stacked `{proj}{shipped}` scales to `{proj}{slot}`, for each `(shipped, slot)` pair."""
        from ..core_model_loading import WeightRenaming

        return [
            WeightRenaming(source_patterns=rf"experts\.{proj}{shipped}", target_patterns=f"experts.{proj}{slot}")
            for proj in projections
            for shipped, slot in slots
        ]

    def _fused_conversions(self, calibrated: bool, up_projections: tuple[str, ...]):
        """Return converters for experts stacked per layer, as vLLM's fused layout ships them.

        The scales only take their slot name.
        """
        from ..core_model_loading import WeightConverter
        from ..integrations.finegrained.conversions import FineGrainedWeightGlobals

        converters = self._fused_renames((*up_projections, "down_proj"), (("_weight_scale", "_scale_inv"),))
        converters += [
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
                operations=[FineGrainedWeightGlobals(self)],
            )
            for up in up_projections
        ]
        return converters
