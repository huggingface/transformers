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
"""MXFP8: `float8_e4m3fn` weights with a UE8M0 scale per 32. It ships the plain `weight` / `weight_scale_inv`
pair; ModelOpt exports name that scale `weight_scale` instead."""

from typing import TYPE_CHECKING

from .block_scaled_utils import BlockScaledQuantizer


if TYPE_CHECKING:
    from ..utils.quantization_config import Mxfp8Config


class Mxfp8HfQuantizer(BlockScaledQuantizer):
    """MXFP8 weights, W8A8, W8A4 or weight-only."""

    quantization_config: "Mxfp8Config"

    def format_conversions(self, dequantize):
        return (
            [self._block_scale_rename()]
            if self.quantization_config.checkpoint_format in ("modelopt", "compressed-tensors")
            else []
        )
