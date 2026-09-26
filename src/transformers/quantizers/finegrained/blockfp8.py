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
"""The BLOCKFP8 arm of the fine-grained quantizer."""

from .base import FineGrainedHfQuantizer


class FineGrainedBlockFp8HfQuantizer(FineGrainedHfQuantizer):
    """Block-FP8 and per-tensor FP8 — the only formats that take a CALIBRATED activation scale.

    The group formats hand the kernels a scale GRID instead, and NVFP4 spends `input_scale` as
    its activation global, so neither can consume one. Named `blockfp8` because
    `quantizer_finegrained_fp8` is the frozen legacy module.
    """

    def get_weight_conversions(self):
        if not self.pre_quantized:
            return []
        if self.quantization_config.dequantize:
            return self._dequantize_conversions()
        if self.quantization_config.activation_scheme == "static":
            return self._static_activation_conversions()
        return []

    def _static_activation_conversions(self):
        """A calibrated checkpoint's ``input_scale`` onto the slot its module holds it in: one
        value per quantized module, so a MoE brings one per expert. Both expert layouts are
        covered — per expert per projection, and already stacked per layer — and a checkpoint
        matches one while the other never fires."""
        from ...core_model_loading import MergeModulelist, WeightConverter, WeightRenaming
        from ...integrations.finegrained.conversions import FineGrainedInputScales

        per_expert = [
            WeightConverter(
                source_patterns=[
                    "mlp.experts.*.gate_proj.input_scale",
                    "mlp.experts.*.up_proj.input_scale",
                ],
                target_patterns="mlp.experts.gate_up_proj_activation_scale",
                operations=[MergeModulelist(dim=0), FineGrainedInputScales(self)],
            ),
            WeightConverter(
                source_patterns="mlp.experts.*.down_proj.input_scale",
                target_patterns="mlp.experts.down_proj_activation_scale",
                operations=[MergeModulelist(dim=0), FineGrainedInputScales(self)],
            ),
        ]
        fused = [
            WeightConverter(
                source_patterns=rf"experts\.{proj}_input_scale$",
                target_patterns=f"experts.{proj}_activation_scale",
                operations=[FineGrainedInputScales(self)],
            )
            # `up_proj` because an UNGATED experts module holds its scale under that name; the
            # converter whose source no checkpoint has simply never fires
            for proj in ("gate_up_proj", "up_proj", "down_proj")
        ]
        # a dense linear holds the one value itself; the lookahead keeps this off the expert keys
        dense = [
            WeightRenaming(
                source_patterns=r"^(?!.*\.experts\.)(.+)\.input_scale$",
                target_patterns=r"\1.activation_scale",
            )
        ]
        return per_expert + fused + dense
