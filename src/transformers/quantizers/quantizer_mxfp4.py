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
"""MXFP4: E2M1 weights two per byte with a UE8M0 scale per 32 — GPT-OSS's experts, which ship each projection as
`{proj}_blocks` + `{proj}_scales`, and DeepSeek-V4's, which ship a plain weight/scale pair."""

from typing import TYPE_CHECKING

from .block_scaled_utils import GATED_UP_PROJECTIONS, BlockScaledQuantizer, ProducerLayout


if TYPE_CHECKING:
    from ..utils.quantization_config import Mxfp4Config


class Mxfp4HfQuantizer(BlockScaledQuantizer):
    """MXFP4 weights, W4A16 (the default, what GPT-OSS runs), W4A8 or W4A4 (DeepSeek-V4's experts)."""

    quantization_config: "Mxfp4Config"
    producer_layouts = {
        "modelopt": ProducerLayout(block_scale_name="weight_scale", packed_as_uint8=True),
        "deepseek": ProducerLayout(block_scale_name="scale"),
    }

    def format_conversions(self, dequantize, up_projections=GATED_UP_PROJECTIONS):
        return self._dequantize_conversions() if dequantize else self._mxfp4_conversions()

    def _dequantize_conversions(self):
        """GPT-OSS's blocks and scales folded to bf16, in the `(E, hidden, 2I)` orientation its experts hold."""
        from ..core_model_loading import Transpose, WeightConverter
        from ..integrations.finegrained.conversions import FineGrainedDequantize, FineGrainedPackedBlocks

        return [
            WeightConverter(
                source_patterns=[rf"{proj}_blocks", rf"{proj}_scales"],
                target_patterns=f"{proj}",
                operations=[FineGrainedPackedBlocks(self), FineGrainedDequantize(self), Transpose(1, 2)],
            )
            for proj in ("gate_up_proj", "down_proj")
        ]

    def _mxfp4_conversions(self):
        """GPT-OSS's `{proj}_blocks` regrouped into the packed weight, `{proj}_scales` as its scale grid."""
        from ..core_model_loading import WeightConverter, WeightRenaming
        from ..integrations.finegrained.conversions import FineGrainedPackedBlocks

        conversions = []
        for proj in ("gate_up_proj", "down_proj"):
            conversions += [
                WeightConverter(
                    source_patterns=rf"{proj}_blocks",
                    target_patterns=f"{proj}",
                    operations=[FineGrainedPackedBlocks(self)],
                ),
                WeightRenaming(source_patterns=rf"{proj}_scales", target_patterns=f"{proj}_scale_inv"),
            ]
        return conversions
