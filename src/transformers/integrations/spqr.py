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
"SpQR (Sparse-Quantized Representation) integration file"

from ..quantizers.quantizers_utils import should_convert_module
from ..utils import is_accelerate_available, is_spqr_available, is_torch_available, logging


if is_accelerate_available():
    from accelerate import init_empty_weights

if is_torch_available():
    import torch.nn as nn

logger = logging.get_logger(__name__)


def replace_with_spqr_linear(model, modules_to_not_convert: list[str] | None = None, quantization_config=None):
    """
    Public method that replaces the Linear layers of the given model with SPQR quantized layers.

    Args:
        model (`torch.nn.Module`):
            The model to convert, can be any `torch.nn.Module` instance.
        modules_to_not_convert (`list[str]`, *optional*, defaults to `None`):
            A list of nn.Linear weights to not convert. If a parameter path is in the list (e.g. `lm_head.weight`), the corresponding module will not be
            converted.
        quantization_config (`SpQRConfig`):
            The quantization config object that contains the quantization parameters.
    """
    if is_spqr_available():
        from spqr_quant import QuantizedLinear

    has_been_replaced = False
    # we need this to correctly materialize the weights during quantization
    for module_name, module in model.named_modules():
        if not should_convert_module(module_name, modules_to_not_convert):
            continue
        with init_empty_weights():
            if isinstance(module, nn.Linear):
                shapes = quantization_config.shapes

                new_module = QuantizedLinear.create_placehodler(
                    rows=module.out_features,
                    cols=module.in_features,
                    bits=quantization_config.bits,
                    beta1=quantization_config.beta1,
                    beta2=quantization_config.beta2,
                    dense_weights_shape=shapes[f"{module_name}.dense_weights.shape"],
                    row_offsets_shape=shapes[f"{module_name}.row_offsets.shape"],
                    col_vals_shape=shapes[f"{module_name}.col_vals.shape"],
                    in_perm_shape=shapes[f"{module_name}.in_perm.shape"],
                )
                # Force requires grad to False to avoid unexpected errors
                model._modules[module_name].requires_grad_(False)
                model.set_submodule(module_name, new_module)
                has_been_replaced = True
    if not has_been_replaced:
        logger.warning(
            "You are loading your model using eetq but no linear modules were found in your model."
            " Please double check your model architecture, or submit an issue on github if you think this is"
            " a bug."
        )

    return model
