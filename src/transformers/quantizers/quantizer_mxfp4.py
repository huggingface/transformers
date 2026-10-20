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
"""MXFP4 quantizer for E2M1 weights, packed two per byte, with one UE8M0 scale per 32 values."""

from typing import TYPE_CHECKING

from ..utils import logging
from .block_scaled_utils import GATED_UP_PROJECTIONS, BlockScaledQuantizer, ProducerLayout


if TYPE_CHECKING:
    from ..utils.quantization_config import Mxfp4Config

logger = logging.get_logger(__name__)


class Mxfp4HfQuantizer(BlockScaledQuantizer):
    """MXFP4 weights with bf16, MXFP8 or MXFP4 activations.

    GPT-OSS runs the default W4A16. DeepSeek-V4 experts run W4A4.
    GPT-OSS ships each expert projection as `{proj}_blocks` and `{proj}_scales`.
    DeepSeek-V4 ships a plain weight and scale pair.
    """

    quantization_config: "Mxfp4Config"
    producer_layouts = {
        "modelopt": ProducerLayout(block_scale_name="weight_scale", packed_as_uint8=True),
        "deepseek": ProducerLayout(block_scale_name="scale"),
    }

    def _process_model_before_weight_loading(self, model, use_kernels: bool = False, **kwargs):
        # Hub kernels replace GPT-OSS's experts forward with one that reads full-precision weights on every device
        if use_kernels and not self.quantization_config.dequantize:
            logger.warning_once(
                "You are using full precision kernels, we will dequantize the model to bf16. "
                "To use the quantized model with quantization kernels, please set use_kernels=False"
            )
            self.quantization_config.dequantize = True
        return super()._process_model_before_weight_loading(model, **kwargs)

    def format_conversions(self, dequantize, up_projections=GATED_UP_PROJECTIONS):
        from ..core_model_loading import Transpose, WeightConverter, WeightRenaming
        from ..integrations.finegrained.conversions import FineGrainedDequantize, FineGrainedPackedBlocks

        conversions = []
        for proj in ("gate_up_proj", "down_proj"):
            if dequantize:
                # Dequantize into the `(E, hidden, 2I)` layout that GPT-OSS uses
                conversions.append(
                    WeightConverter(
                        source_patterns=[rf"{proj}_blocks", rf"{proj}_scales"],
                        target_patterns=proj,
                        operations=[FineGrainedPackedBlocks(self), FineGrainedDequantize(self), Transpose(1, 2)],
                    )
                )
            else:
                conversions += [
                    WeightConverter(
                        source_patterns=rf"{proj}_blocks",
                        target_patterns=proj,
                        operations=[FineGrainedPackedBlocks(self)],
                    ),
                    WeightRenaming(source_patterns=rf"{proj}_scales", target_patterns=f"{proj}_scale_inv"),
                ]
        return conversions
