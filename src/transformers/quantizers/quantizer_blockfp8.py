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

    def get_param_ops(self, model, param_name):
        from ..integrations.finegrained.conversions import FineGrainedRowScales

        ops = super().get_param_ops(model, param_name)
        if (
            self.quantization_config.weight_block_size == "row"
            and self.quantization_config.checkpoint_format == "modelopt"
            and param_name.endswith("_scale_inv")
        ):
            ops = [FineGrainedRowScales(self), *ops]
        return ops

    def format_parallel_plan(self, model, plan):
        from ..distributed.tensor_parallel import _get_parameter_plan, replace_layer_number_by_wildcard
        from ..integrations.finegrained import FineGrainedEmbedding, FineGrainedExperts, FineGrainedLinear

        for name, module in model.named_modules():
            if (
                self.quantization_config.weight_block_size == "row"
                and isinstance(module, FineGrainedExperts)
                and self.member_of(module) is not None
                and _get_parameter_plan(f"{name}.down_proj", plan) in ("rowwise", "packed_rowwise")
            ):
                plan.pop(f"{replace_layer_number_by_wildcard(name)}.down_proj_scale_inv", None)
            if (
                self.quantization_config.weight_block_size == "row"
                and isinstance(module, FineGrainedLinear)
                and self.member_of(module) is not None
                and _get_parameter_plan(f"{name}.weight", plan)
                in ("rowwise", "rowwise_rep", "rowwise_split_input", "packed_rowwise")
            ):
                plan[f"{replace_layer_number_by_wildcard(name)}.weight_scale_inv"] = "replicate"
            if (
                isinstance(module, FineGrainedEmbedding)
                and self.member_of(module) is not None
                and _get_parameter_plan(name, plan, is_weight=False) is not None
            ):
                plan.setdefault(f"{replace_layer_number_by_wildcard(name)}.weight_scale", "replicate")

    def format_weight_conversions(self, converters, model):
        from ..core_model_loading import MergeModulelist, Transpose
        from ..integrations.finegrained import FineGrainedExperts
        from ..integrations.finegrained.conversions import FineGrainedDequantize, FineGrainedScalarScales

        producer = self.quantization_config.checkpoint_format
        if self.quantization_config.weight_block_size == "row" and producer in ("fbgemm", "modelopt"):
            for converter in converters:
                dequantize = next((op for op in converter.operations if isinstance(op, FineGrainedDequantize)), None)
                if dequantize is None or not any(
                    isinstance(op, Transpose) and not op.check_dims and {op.dim0, op.dim1} == {1, 2}
                    for op in converter.operations
                ):
                    continue
                reverse = converter.reverse_transform()
                if any(
                    reverse.rename_source_key(name)[1] is not None
                    and dequantize._config_for(name) is self.quantization_config
                    and (producer == "modelopt" or name.endswith(".down_proj"))
                    for name, _ in model.named_parameters()
                ):
                    # Dequantization precedes the BMM transpose; these producers' scales already name output rows.
                    converter.operations = [
                        FineGrainedDequantize(op.hf_quantizer, transpose_scales=True) if op is dequantize else op
                        for op in converter.operations
                    ]
        if producer == "fbgemm" or (producer == "modelopt" and self.quantization_config.weight_block_size == "row"):
            # ModelOpt leaves BMM channel vectors untransposed; FBGEMM's down grid is already rowwise.
            scales = [
                f"{name}.{param}"
                for name, module in model.named_modules()
                if self.member_of(module) is not None
                for param, _ in module.named_parameters(recurse=False)
                if param.endswith("_scale_inv") and (producer == "modelopt" or param == "down_proj_scale_inv")
            ]
            for converter in converters:
                reverse = converter.reverse_transform()
                if any(reverse.rename_source_key(name)[1] is not None for name in scales):
                    converter.operations = [op for op in converter.operations if not isinstance(op, Transpose)]

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
        if self.quantization_config.checkpoint_format in ("compressed-tensors", "modelopt", "fbgemm"):
            # its block scale ships as `weight_scale`, and rides its weight's converter once renamed
            conversions.append(self._block_scale_rename())
        if self.quantization_config.checkpoint_format in ("fbgemm", "modelopt"):
            from ..core_model_loading import WeightRenaming

            suffix = "weight_scale" if self.quantization_config.checkpoint_format == "modelopt" else "scale"
            conversions.append(WeightRenaming(rf"(gate_up_proj|up_proj|down_proj)_{suffix}(?=$)", r"\1_scale_inv"))
        if not dequantize and self.quantization_config.activation_scheme == "static":
            conversions += self._activation_scale_conversions()
        return conversions
