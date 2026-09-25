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
"""The NVFP4 arm of the fine-grained quantizer: everything modelopt's two-level scales need
that the other formats do not. Everything else is inherited.
"""

from .base import FineGrainedHfQuantizer


class FineGrainedNvfp4HfQuantizer(FineGrainedHfQuantizer):
    """NVFP4, in both key layouts modelopt ships: per expert per projection, or stacked per layer.

    What is NVFP4's alone, and why the base does not carry it:

    * the two-level globals. `weight_scale_2` has no slot in a one-level fold, so `dequantize=True`
      is refused here rather than tested for in the shared path.
    * the packed E2M1 weights, which need a uint8 -> int8 BITCAST (`copy_` would convert and
      corrupt anything >= 128) and an anchored `.weight` source so it cannot swallow
      `weight_scale` / `weight_scale_2`.
    """

    @property
    def supports_dequantize(self) -> bool:
        """No: NVFP4's two-level scales do not fold into a single full-precision weight here."""
        return False

    def get_weight_conversions(self):
        return self._nvfp4_conversions() if self.pre_quantized else []

    def update_weight_conversions(self, weight_conversions):
        """The base chain, with every `.weight` source anchored and bitcast to the int8 view."""
        from ...core_model_loading import WeightConverter
        from ...integrations.finegrained.conversions import FineGrainedViewPackedInt8

        if self.pre_quantized:
            updated = []
            for conv in weight_conversions:
                if isinstance(conv, WeightConverter) and any(p.endswith(".weight") for p in conv.source_patterns):
                    conv = WeightConverter(
                        source_patterns=[p + "$" if p.endswith(".weight") else p for p in conv.source_patterns],
                        target_patterns=conv.target_patterns,
                        operations=[*conv.operations, FineGrainedViewPackedInt8(self)],
                    )
                updated.append(conv)
            weight_conversions = updated
        return super().update_weight_conversions(weight_conversions)

    def _nvfp4_conversions(self):
        """modelopt NVFP4 scales (`weight_scale`, `weight_scale_2`, `input_scale`), in both layouts
        a modelopt checkpoint ships them in: per expert per projection (GLM-5.2-NVFP4) or already
        stacked per layer (the fused vLLM layout). A checkpoint matches one set and
        the other never fires. Routed experts only — a generic `weight_scale*` rename would run
        before converter collection and mangle these keys."""
        # weight-only keeps activations bf16, so there is no activation global for a calibrated
        # `input_scale` to land on — leave those keys to the unexpected-key filter
        calibrated = self.quantization_config.activation_format != "bf16"
        return self._nvfp4_per_expert_conversions(calibrated) + self._nvfp4_fused_conversions(calibrated)

    def _nvfp4_per_expert_conversions(self, calibrated: bool):
        """One key per expert per projection, as GLM-5.2-NVFP4 ships them.

        `MergeModulelist` stamps each source with its expert index, which is what lets expert
        parallelism keep only this rank's experts (`tensor_idx` in `core_model_loading`); without
        it every rank collects all E values and the forward asserts on the per-expert count.
        """
        from ...core_model_loading import Concatenate, MergeModulelist, WeightConverter
        from ...integrations.finegrained.conversions import FineGrainedInputScales, FineGrainedWeightGlobals

        converters = [
            WeightConverter(
                source_patterns=[
                    r"mlp\.experts\..*\.gate_proj\.weight_scale$",
                    r"mlp\.experts\..*\.up_proj\.weight_scale$",
                ],
                target_patterns="mlp.experts.gate_up_proj_scale_inv",
                operations=[MergeModulelist(dim=0), Concatenate(dim=1)],
            ),
            WeightConverter(
                source_patterns=r"mlp\.experts\..*\.down_proj\.weight_scale$",
                target_patterns="mlp.experts.down_proj_scale_inv",
                operations=[MergeModulelist(dim=0)],
            ),
            # every second-level global of a layer in ONE converter: folding the gate|up stack's
            # two halves into one moves the up half's onto the down projection, so they decide
            # together (`FineGrainedWeightGlobals`)
            WeightConverter(
                source_patterns=[
                    r"mlp\.experts\..*\.gate_proj\.weight_scale_2",
                    r"mlp\.experts\..*\.up_proj\.weight_scale_2",
                    r"mlp\.experts\..*\.down_proj\.weight_scale_2",
                    *([r"mlp\.experts\..*\.down_proj\.input_scale"] if calibrated else []),
                ],
                target_patterns=[
                    "mlp.experts.gate_up_proj_weight_global_scale",
                    "mlp.experts.down_proj_weight_global_scale",
                    *(["mlp.experts.down_proj_input_global_scale"] if calibrated else []),
                ],
                operations=[MergeModulelist(dim=0), FineGrainedWeightGlobals(self)],
            ),
        ]
        if calibrated:
            converters.append(
                WeightConverter(
                    source_patterns=[
                        r"mlp\.experts\..*\.gate_proj\.input_scale",
                        r"mlp\.experts\..*\.up_proj\.input_scale",
                    ],
                    target_patterns="mlp.experts.gate_up_proj_input_global_scale",
                    operations=[MergeModulelist(dim=0), FineGrainedInputScales(self)],
                )
            )
        return converters

    def _nvfp4_fused_conversions(self, calibrated: bool):
        """Already stacked per layer, as the vLLM fused layout ships them — so no
        `MergeModulelist`, and the expert weights need the packed uint8 -> int8 view the
        `.weight`-anchored rule gives the per-expert layout."""
        from ...core_model_loading import WeightConverter
        from ...integrations.finegrained.conversions import (
            FineGrainedInputScales,
            FineGrainedScaleContainer,
            FineGrainedViewPackedInt8,
            FineGrainedWeightGlobals,
        )

        converters = [
            # the container cast the module's scale dtype needs; `_with_expert_layout_ops`
            # adds the interleave and the swizzle on top (it re-adds the cast, which is
            # idempotent)
            WeightConverter(
                source_patterns=rf"experts\.{proj}_weight_scale$",
                target_patterns=f"experts.{proj}_scale_inv",
                operations=[FineGrainedScaleContainer(self)],
            )
            for proj in ("gate_up_proj", "down_proj")
        ]
        converters.append(
            # the same one-converter rule as the per-expert layout, over the fused keys
            WeightConverter(
                source_patterns=[
                    r"experts\.gate_up_proj_weight_scale_2$",
                    r"experts\.down_proj_weight_scale_2$",
                    *([r"experts\.down_proj_input_scale$"] if calibrated else []),
                ],
                target_patterns=[
                    "experts.gate_up_proj_weight_global_scale",
                    "experts.down_proj_weight_global_scale",
                    *(["experts.down_proj_input_global_scale"] if calibrated else []),
                ],
                operations=[FineGrainedWeightGlobals(self)],
            )
        )
        if calibrated:
            converters.append(
                WeightConverter(
                    source_patterns=r"experts\.gate_up_proj_input_scale$",
                    target_patterns="experts.gate_up_proj_input_global_scale",
                    operations=[FineGrainedInputScales(self)],
                )
            )
        converters += [
            WeightConverter(
                source_patterns=rf"experts\.{proj}$",
                target_patterns=f"experts.{proj}",
                operations=[FineGrainedViewPackedInt8(self)],
            )
            for proj in ("gate_up_proj", "down_proj")
        ]
        return converters
