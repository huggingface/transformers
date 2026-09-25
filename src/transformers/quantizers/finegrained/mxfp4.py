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
"""The MXFP4 arm of the fine-grained quantizer: GPT-OSS's packed `_blocks` / `_scales` layout,
and the weight-only default that layout implies.
"""

from .base import FineGrainedHfQuantizer


class FineGrainedMxfp4HfQuantizer(FineGrainedHfQuantizer):
    """MXFP4 weights, W4A16 or W4A4.

    One thing is MXFP4's alone: GPT-OSS names each projection `{proj}_blocks` + `{proj}_scales`
    rather than a weight/scale pair.

    The weight-only default is NOT a property of the format — real W4A4 (mxfp4 activations) is
    supported and is what DeepSeek-V4's experts run. It only fills the gap when a config leaves
    `activation_format` open, because the checkpoint that ships this key layout (GPT-OSS) is
    W4A16; an explicit `activation_format="mxfp4"` still gets W4A4.
    """

    default_activation_format = "bf16"

    def get_weight_conversions(self):
        if not self.pre_quantized:
            return []
        return self._dequantize_conversions() if self.quantization_config.dequantize else self._mxfp4_conversions()

    def _dequantize_conversions(self):
        """The blocks regroup into the packed rows and the exponent-byte scales come back out of
        them as bf16, in the `(E, hidden, 2I)` orientation the unquantized experts hold."""
        from ...core_model_loading import Transpose, WeightConverter
        from ...integrations.finegrained.conversions import FineGrainedDequantize, FineGrainedPackedBlocks

        return [
            WeightConverter(
                source_patterns=[rf"{proj}_blocks$", rf"{proj}_scales$"],
                target_patterns=proj,
                operations=[FineGrainedPackedBlocks(self), FineGrainedDequantize(self), Transpose(1, 2)],
            )
            for proj in ("gate_up_proj", "down_proj")
        ]

    def _mxfp4_conversions(self):
        """{proj}_blocks + {proj}_scales checkpoints (GPT-OSS): the blocks regroup into the packed
        weight, the exponent bytes take the scale container op; then the usual layout ops. Bare
        patterns (no `experts.` scope), so they ride here rather than via `_with_expert_layout_ops`."""
        from ...core_model_loading import WeightConverter
        from ...integrations.finegrained.conversions import (
            FineGrainedInterleaveGateUp,
            FineGrainedPackedBlocks,
            FineGrainedScaleContainer,
            FineGrainedSwizzleScales,
        )

        converters = []
        for proj in ("gate_up_proj", "down_proj"):
            interleave = [FineGrainedInterleaveGateUp(self)] if proj == "gate_up_proj" else []
            converters += [
                WeightConverter(
                    source_patterns=rf"{proj}_blocks$",
                    target_patterns=proj,
                    operations=[FineGrainedPackedBlocks(self), *interleave, FineGrainedSwizzleScales(self)],
                ),
                WeightConverter(
                    source_patterns=rf"{proj}_scales$",
                    target_patterns=f"{proj}_scale_inv",
                    operations=[*interleave, FineGrainedScaleContainer(self), FineGrainedSwizzleScales(self)],
                ),
            ]
        return converters
