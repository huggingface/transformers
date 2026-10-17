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

from .block_scaled_utils import BlockScaledQuantizer
from .quantizers_utils import FUSED_STACKS, get_module_from_name


if TYPE_CHECKING:
    from ..utils.quantization_config import Mxfp4Config


class Mxfp4HfQuantizer(BlockScaledQuantizer):
    """MXFP4 weights, W4A16 (the config's default, what GPT-OSS runs), W4A8 or W4A4 (DeepSeek-V4's experts).

    One thing is MXFP4's alone: GPT-OSS names each projection `{proj}_blocks` + `{proj}_scales` rather than a
    weight/scale pair.
    """

    quantization_config: "Mxfp4Config"

    def format_conversions(self, dequantize):
        conversions = self._dequantize_conversions() if dequantize else self._mxfp4_conversions()
        if self.quantization_config.checkpoint_format == "modelopt":
            conversions = [self._block_scale_rename(), *conversions]
        return conversions

    def get_param_ops(self, model, param_name):
        from ..integrations.finegrained.conversions import FineGrainedViewPackedInt8
        from ..integrations.finegrained.layers import _FineGrainedModule

        ops = super().get_param_ops(model, param_name)
        module, name = get_module_from_name(model, param_name)
        if (
            self.quantization_config.checkpoint_format == "modelopt"
            and isinstance(module, _FineGrainedModule)
            and name in ("weight", *FUSED_STACKS)
        ):
            ops = [FineGrainedViewPackedInt8(self), *ops]
        return ops

    def _dequantize_conversions(self):
        """The blocks regroup into the packed rows and the exponent-byte scales come back out of
        them as bf16, in the `(E, hidden, 2I)` orientation the unquantized experts hold."""
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
        """{proj}_blocks + {proj}_scales checkpoints (GPT-OSS): the blocks regroup into the packed weight and
        the exponent bytes are the scale grid. Bare patterns, with no `experts.` scope; the layout each lands
        in is the module's, from `get_param_ops`."""
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
