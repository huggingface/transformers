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
"VPTQ (Vector Post-Training Quantization) integration file"

from ..utils import ACCELERATE_MIN_VERSION, is_accelerate_available, is_torch_available, is_vptq_available


if is_torch_available():
    import torch.nn as nn


def replace_with_vptq_linear(
    model,
    quantization_config=None,
    current_key_name=None,
    has_been_replaced=False,
):
    """
    Public method that recursively replaces the Linear layers of the given model with VPTQ quantized layers.
    `accelerate` is needed to use this method. Returns the converted model and a boolean that indicates if the
    conversion has been successfull or not.

    Args:
        model (`torch.nn.Module`):
            The model to convert, can be any `torch.nn.Module` instance.
        quantization_config (`VptqConfig`):
            The quantization config object that contains the quantization parameters.
        current_key_name (`list`, *optional*):
            A list that contains the current key name. This is used for recursion and should not be passed by the user.
        has_been_replaced (`bool`, *optional*):
            A boolean that indicates if the conversion has been successful or not. This is used for recursion and
            should not be passed by the user.
    """
    if not is_vptq_available():
        raise ValueError("VPTQ is not available. Please install it with `pip install vptq`")

    if not is_accelerate_available():
        raise ValueError(
            f"VPTQ requires Accelerate to be installed: `pip install 'accelerate>={ACCELERATE_MIN_VERSION}'`"
        )

    from accelerate import init_empty_weights
    from vptq import VQuantLinear

    for name, module in model.named_children():
        if current_key_name is None:
            current_key_name = []
        current_key_name.append(name)
        layer_name = ".".join(current_key_name)

        if isinstance(module, nn.Linear) and layer_name in quantization_config.config_for_layers:
            layer_params = quantization_config.config_for_layers[layer_name]
            with init_empty_weights():
                in_features = module.in_features
                out_features = module.out_features

                model._modules[name] = VQuantLinear(
                    in_features,
                    out_features,
                    vector_lens=layer_params["vector_lens"],
                    num_centroids=layer_params["num_centroids"],
                    num_res_centroids=layer_params["num_res_centroids"],
                    group_num=layer_params["group_num"],
                    group_size=layer_params["group_size"],
                    outlier_size=layer_params["outlier_size"],
                    indices_as_float=layer_params["indices_as_float"],
                    enable_norm=layer_params["enable_norm"],
                    enable_perm=layer_params["enable_perm"],
                    is_indice_packed=layer_params["is_indice_packed"],
                    enable_proxy_error=False,
                    bias=module.bias is not None,
                )
                has_been_replaced = True

                # Force requires grad to False to avoid unexpected errors
                model._modules[name].requires_grad_(False)
        if len(list(module.children())) > 0:
            _, has_been_replaced = replace_with_vptq_linear(
                module,
                quantization_config=quantization_config,
                current_key_name=current_key_name,
                has_been_replaced=has_been_replaced,
            )
        # Remove the last key for recursion
        current_key_name.pop(-1)
    return model, has_been_replaced
