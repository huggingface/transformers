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

from ..quantizers.quantizers_utils import should_convert_module
from ..utils import is_accelerate_available, is_torch_available, logging


if is_accelerate_available():
    from accelerate import init_empty_weights

if is_torch_available():
    import torch.nn as nn

logger = logging.get_logger(__name__)


def replace_with_vptq_linear(model, modules_to_not_convert: list[str] | None = None, quantization_config=None):
    """
    Public method that replaces the Linear layers of the given model with SPQR quantized layers.

    Args:
        model (`torch.nn.Module`):
            The model to convert, can be any `torch.nn.Module` instance.
        modules_to_not_convert (`list[str]`, *optional*, defaults to `None`):
            A list of nn.Linear weights to not convert. If a parameter path is in the list (e.g. `lm_head.weight`), the corresponding module will not be
            converted.
        quantization_config (`VptqConfig`):
            The quantization config object that contains the quantization parameters.
    """
    from vptq import VQuantLinear

    has_been_replaced = False
    shared_layer_config = quantization_config.shared_layer_config
    config_for_layers = quantization_config.config_for_layers

    for module_name, module in model.named_modules():
        if not should_convert_module(module_name, modules_to_not_convert):
            continue
        with init_empty_weights():
            if isinstance(module, nn.Linear):
                layer_params = config_for_layers.get(module_name, None) or shared_layer_config.get(
                    module_name.rsplit(".")[1], None
                )
                new_module = VQuantLinear(
                    module.in_features,
                    module.out_features,
                    vector_lens=layer_params["vector_lens"],
                    num_centroids=layer_params["num_centroids"],
                    num_res_centroids=layer_params["num_res_centroids"],
                    group_num=layer_params["group_num"],
                    group_size=layer_params["group_size"],
                    outlier_size=layer_params["outlier_size"],
                    indices_as_float=layer_params["indices_as_float"],
                    enable_norm=layer_params["enable_norm"],
                    enable_perm=layer_params["enable_perm"],
                    is_indice_packed=True,
                    enable_proxy_error=False,
                    bias=module.bias is not None,
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
