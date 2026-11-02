# Copyright 2024 The HuggingFace Team. All rights reserved.
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
"AQLM (Additive Quantization of Language Model) integration file"

from ..core_model_loading import ConversionOps
from ..quantizers.quantizers_utils import should_convert_module
from ..utils import is_torch_available, logging


if is_torch_available():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

logger = logging.get_logger(__name__)


class AqlmDequantize(ConversionOps):
    """
    Dense weights from the AQLM `codes` (one per codebook for each `(out_group_size, in_group_size)` block), summed
    `codebooks` entries, times the `scales` of each output group. Same as `aqlm.utils._dequantize_weight`.
    """

    def __init__(self, hf_quantizer):
        self.hf_quantizer = hf_quantizer

    def convert(self, input_dict, model=None, full_layer_name=None, **kwargs) -> dict[str, list[torch.Tensor]]:
        output = {}
        for key, all_codes in input_dict.items():
            if not key.endswith("codes"):
                continue
            prefix = key.removesuffix("codes")
            all_codes = all_codes if isinstance(all_codes, list) else [all_codes]
            others = [input_dict[prefix + name] for name in ("codebooks", "scales")]
            others = [other if isinstance(other, list) else [other] * len(all_codes) for other in others]
            weights = []
            for codes, codebooks, scales in zip(all_codes, *others):
                num_out_groups, num_in_groups, num_codebooks = codes.shape
                _, codebook_size, out_group_size, in_group_size = codebooks.shape
                # The codes are saved as signed ints
                codes = codes.long() % codebook_size
                offsets = torch.arange(0, num_codebooks * codebook_size, codebook_size, device=codes.device)
                weight = F.embedding_bag(
                    codes.flatten(0, 1) + offsets, codebooks.flatten(0, 1).flatten(-2, -1), mode="sum"
                )
                weight = weight.view(num_out_groups, num_in_groups, out_group_size, in_group_size) * scales
                weights.append(
                    weight.swapaxes(1, 2).reshape(num_out_groups * out_group_size, num_in_groups * in_group_size)
                )
            output[prefix + "weight"] = weights
        return output


def replace_with_aqlm_linear(model, modules_to_not_convert: list[str] | None = None, quantization_config=None):
    """
    Public method that recursively replaces the Linear layers of the given model with AQLM quantized layers.

    Args:
        model (`torch.nn.Module`):
            The model to convert, can be any `torch.nn.Module` instance.
        modules_to_not_convert (`list[str]`, *optional*, defaults to `None`):
            A list of nn.Linear weights to not convert. If a parameter path is in the list (e.g. `lm_head.weight`), the corresponding module will not be
            converted.
        quantization_config (`AqlmConfig`):
            The quantization config object that contains the quantization parameters.
    """
    from aqlm import QuantizedLinear

    has_been_replaced = False
    # we need this to correctly materialize the weights during quantization
    for module_name, module in model.named_modules():
        if not should_convert_module(module_name, modules_to_not_convert):
            continue
        with torch.device("meta"):
            if isinstance(module, nn.Linear):
                new_module = QuantizedLinear(
                    module.in_features,
                    module.out_features,
                    bias=module.bias is not None,
                    in_group_size=quantization_config.in_group_size,
                    out_group_size=quantization_config.out_group_size,
                    num_codebooks=quantization_config.num_codebooks,
                    nbits_per_codebook=quantization_config.nbits_per_codebook,
                )
                new_module.source_cls = type(module)
                new_module.requires_grad_(False)
                model.set_submodule(module_name, new_module)
                has_been_replaced = True

    if not has_been_replaced:
        logger.warning(
            "You are loading your model using eetq but no linear modules were found in your model."
            " Please double check your model architecture, or submit an issue on github if you think this is"
            " a bug."
        )

    return model
