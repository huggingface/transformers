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

from .quantizer_finegrained import FineGrainedHfQuantizer


class FineGrainedBlockFp8HfQuantizer(FineGrainedHfQuantizer):
    """Block-FP8 and per-tensor FP8 — the only formats that take a CALIBRATED activation scale.

    A static scale is one value standing in for the quantization the other formats do inline, so
    it only means anything where activations are quantized per tensor or per block. The group
    formats hand the kernels a scale GRID instead (`As` is per group-32 / group-16), and NVFP4
    spends the checkpoint's `input_scale` as its second-level activation global, so neither can
    consume one. Keeping the converters here is what makes that structural.

    Named `blockfp8` because `quantizer_finegrained_fp8` is the frozen legacy module.
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
        """A calibrated checkpoint's ``input_scale`` onto the slot its module holds it in. One
        value per quantized module: a dense linear brings one (Ministral-3), and a MoE brings one
        per expert, each expert being its own quantized module (Mistral-4) — the gate|up pair
        reduces to one per expert, both halves reading the same routed rows. Both expert layouts,
        per expert per projection and already stacked per layer; a checkpoint matches one and the
        other never fires. NVFP4 consumes ``input_scale`` as its activation GLOBAL instead
        (the NVFP4 arm's converters) — a second level over a block scale, not the scale itself."""
        from ..core_model_loading import MergeModulelist, WeightConverter, WeightRenaming
        from ..integrations.finegrained_conversions import FineGrainedInputScales

        per_expert = [
            WeightConverter(
                source_patterns=[
                    r"mlp\.experts\..*\.gate_proj\.input_scale",
                    r"mlp\.experts\..*\.up_proj\.input_scale",
                ],
                target_patterns="mlp.experts.gate_up_proj_activation_scale",
                operations=[MergeModulelist(dim=0), FineGrainedInputScales(self)],
            ),
            WeightConverter(
                source_patterns=r"mlp\.experts\..*\.down_proj\.input_scale",
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
            for proj in ("gate_up_proj", "down_proj")
        ]
        # every other calibrated module is a dense linear holding the one value itself — a plain
        # rename. The lookahead keeps it off the expert keys above, which stack per layer instead
        dense = [
            WeightRenaming(
                source_patterns=r"^(?!.*\.experts\.)(.+)\.input_scale$",
                target_patterns=r"\1.activation_scale",
            )
        ]
        return per_expert + fused + dense
