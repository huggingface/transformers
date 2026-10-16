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
"""Block-FP8: `float8_e4m3fn` weights with fp32 or UE8M0 scales over blocks (or one per tensor), W8A8."""

from typing import TYPE_CHECKING

from .block_scaled_utils import BlockScaledQuantizer
from .quantizers_utils import embedding_pattern


if TYPE_CHECKING:
    from ..utils.quantization_config import BlockFp8Config


class BlockFp8HfQuantizer(BlockScaledQuantizer):
    """Block-FP8 and per-tensor FP8, with static or dynamic activations and scalar-scaled embedding tables."""

    quantization_config: "BlockFp8Config"

    def plan(self):
        embeddings = self.quantization_config.modules_to_convert or []
        return {**dict.fromkeys(map(embedding_pattern, embeddings), self.quantization_config), **super().plan()}

    def format_parallel_plan(self, model, plan):
        from ..distributed.tensor_parallel import _get_parameter_plan, replace_layer_number_by_wildcard
        from ..integrations.finegrained import FineGrainedEmbedding

        for name, module in model.named_modules():
            if (
                isinstance(module, FineGrainedEmbedding)
                and self.member_of(module) is not None
                and _get_parameter_plan(name, plan, is_weight=False) is not None
            ):
                plan.setdefault(f"{replace_layer_number_by_wildcard(name)}.weight_scale", "replicate")

    def format_weight_conversions(self, converters, model):
        from ..core_model_loading import MergeModulelist
        from ..integrations.finegrained import FineGrainedExperts
        from ..integrations.finegrained.conversions import FineGrainedScalarScales

        if (
            self.quantization_config.checkpoint_format != "modelopt"
            or self.quantization_config.weight_block_size is not None
        ):
            return
        scalar_scales = [
            f"{name}.{proj}_scale_inv"
            for name, module in model.named_modules()
            if isinstance(module, FineGrainedExperts) and self.member_of(module) is not None
            for proj in (module.gate_up_name, "down_proj")
        ]
        for converter in converters:
            if any(isinstance(op, MergeModulelist) for op in converter.operations) and not any(
                isinstance(op, FineGrainedScalarScales) for op in converter.operations
            ):
                reverse = converter.reverse_transform()
                if any(reverse.rename_source_key(name)[1] is not None for name in scalar_scales):
                    converter.operations = [FineGrainedScalarScales(self), *converter.operations]

    def format_conversions(self, dequantize):
        conversions = []
        if self.quantization_config.checkpoint_format in ("compressed-tensors", "modelopt"):
            # its block scale ships as `weight_scale`, and rides its weight's converter once renamed
            conversions.append(self._block_scale_rename())
        if not dequantize and self.quantization_config.activation_scheme == "static":
            conversions += self._activation_scale_conversions()
        return conversions
