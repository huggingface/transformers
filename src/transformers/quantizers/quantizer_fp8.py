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
"""FP8 quantizer for `float8_e4m3fn` weights and activations.

The weight scales are fp32 or UE8M0, per block or per tensor.
"""

from typing import TYPE_CHECKING

from .block_scaled_utils import BlockScaledQuantizer, ProducerLayout


if TYPE_CHECKING:
    from ..utils.quantization_config import Fp8Config


class Fp8HfQuantizer(BlockScaledQuantizer):
    """Block and per-tensor FP8, with static or dynamic activations and scalar-scaled embedding tables."""

    quantization_config: "Fp8Config"
    producer_layouts = dict.fromkeys(
        ("compressed-tensors", "modelopt"), ProducerLayout(block_scale_name="weight_scale")
    )

    def update_parallel_plan(self, model, plan):
        from ..distributed.tensor_parallel import _get_parameter_plan, replace_layer_number_by_wildcard
        from ..integrations.finegrained import FineGrainedEmbedding

        super().update_parallel_plan(model, plan)
        for name, module in model.named_modules():
            if (
                isinstance(module, FineGrainedEmbedding)
                and self.holds(module)
                and _get_parameter_plan(name, plan, is_weight=False) is not None
            ):
                plan.setdefault(f"{replace_layer_number_by_wildcard(name)}.weight_scale", "replicate")

    def get_weight_conversions(self, model=None):
        conversions = super().get_weight_conversions(model)
        config = self.quantization_config
        if self.pre_quantized and not config.dequantize and config.activation_scheme == "static":
            conversions += self._activation_scale_conversions(self._up_projections(model))
        return conversions

    def update_weight_conversions(self, weight_conversions, model, **kwargs):
        from ..core_model_loading import MergeModulelist, WeightConverter
        from ..integrations.finegrained import FineGrainedExperts
        from ..integrations.finegrained.conversions import FineGrainedScalarScales

        conversions = super().update_weight_conversions(weight_conversions, model, **kwargs)
        config = self.quantization_config
        if config.checkpoint_format != "modelopt" or config.weight_block_size is not None:
            return conversions
        scalar_scales = [
            f"{name}.{proj}_scale_inv"
            for name, module in model.named_modules()
            if isinstance(module, FineGrainedExperts) and self.holds(module)
            for proj in (module.gate_up_name, "down_proj")
        ]
        for converter in conversions:
            if not isinstance(converter, WeightConverter) or not any(
                isinstance(op, MergeModulelist) for op in converter.operations
            ):
                continue
            if any(isinstance(op, FineGrainedScalarScales) for op in converter.operations):
                continue
            reverse = converter.reverse_transform()
            if any(reverse.rename_source_key(name)[1] is not None for name in scalar_scales):
                converter.operations = [FineGrainedScalarScales(self), *converter.operations]
        return conversions
