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
"""MXFP8 quantizer for `float8_e4m3fn` weights with one UE8M0 scale per 32 values.

The checkpoint ships `weight` and `weight_scale_inv`. ModelOpt and compressed-tensors exports name the scale
`weight_scale`.
"""

from typing import TYPE_CHECKING

from .block_scaled_utils import BlockScaledQuantizer, ProducerLayout


if TYPE_CHECKING:
    from ..utils.quantization_config import Mxfp8Config


class Mxfp8HfQuantizer(BlockScaledQuantizer):
    """MXFP8 weights, W8A8 or weight-only."""

    quantization_config: "Mxfp8Config"
    producer_layouts = dict.fromkeys(
        ("compressed-tensors", "modelopt"), ProducerLayout(block_scale_name="weight_scale")
    )
