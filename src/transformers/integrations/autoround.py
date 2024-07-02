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
from typing import Union

from ..modeling_utils import Conv1D
from ..utils import is_torch_available


if is_torch_available():
    import torch
    import torch.nn as nn


def get_module(module, key):
    """Get module from model by key name.

    Args:
        module (torch.nn.Module): original model
        key (str): module name to be replaced
    """
    name_list = key.split(".")
    for name in name_list:
        module = getattr(module, name, None)
    return module


def set_module(model, key, new_module):
    """Set new module into model by key name.

    Args:
        model (torch.nn.Module): original model
        key (str): module name to be replaced
        new_module (torch.nn.Module): new module to be inserted
    """
    module = model
    name_list = key.split(".")
    for name in name_list[:-1]:
        if hasattr(module, name):
            module = getattr(module, name)
    setattr(module, name_list[-1], new_module)


def convert_auto_round_model(model: nn.Module, quantization_config):
    """Convert the model to an AutoRound model by getting and replacing the layers.

    Args:
        model (`nn.Module`):
            Model to be converted
        quantization_config:
            Configuration settings for quantization.
    """
    from auto_round.utils import get_layer_names_in_block

    layer_names = get_layer_names_in_block(model)
    bits = quantization_config.bits
    group_size = quantization_config.group_size
    data_type = quantization_config.data_type
    sym = quantization_config.sym
    extra_config = {}
    if hasattr(quantization_config, "extra_config"):
        extra_config = quantization_config.extra_config
    layer_names += extra_config.keys()
    layer_names = list(set(layer_names))
    layer_configs = {}
    for layer_name in layer_names:
        layer_configs[layer_name] = {}
        if layer_name not in extra_config:
            layer_configs[layer_name]["bits"] = bits
            layer_configs[layer_name]["group_size"] = group_size
            layer_configs[layer_name]["data_type"] = data_type
            layer_configs[layer_name]["sym"] = sym
        else:
            layer_configs[layer_name]["bits"] = extra_config[layer_name].get("bits", bits)
            layer_configs[layer_name]["group_size"] = extra_config[layer_name].get("group_size", group_size)
            layer_configs[layer_name]["data_type"] = extra_config[layer_name].get("data_type", data_type)
            layer_configs[layer_name]["sym"] = extra_config[layer_name].get("sym", sym)
    backend = quantization_config.backend
    _replace_by_quant_layers(model, layer_configs, backend)
    return model


def get_device(obj: Union[torch.Tensor, nn.Module]):
    """Returns the device of a given tensor or module.

    Args:
        obj (Union[torch.Tensor, nn.Module]):
            The tensor or module for which to determine the device.

    Returns:
        torch.device:
            The device on which the tensor or module resides.

    Raises:
        TypeError:
            If the input is neither a torch.Tensor nor an nn.Module.
    """
    if isinstance(obj, torch.Tensor):
        return obj.device
    return next(obj.parameters()).device


def _replace_by_quant_layers(module: nn.Module, layer_configs, backend):
    """Replaces linear layers in `module` by `QuantLinear`

    Args: module (nn.Module): The module containing layers to be quantized. layer_configs (dict): A dictionary
    containing configuration parameters for the quantization of specific layers. Keys are the names of the layers,
    and values are configuration parameters. backend (str): The backend to be used for the quantization process.

    Returns:
        nn.Module:
            The modified module with linear layers replaced by `QuantLinear` layers.

    """
    from auto_round.utils import dynamic_import_inference_linear

    for layer_name in layer_configs.keys():
        config = layer_configs[layer_name]
        bits = config["bits"]
        group_size = config["group_size"]
        if not (bits <= 8):
            continue

        layer = get_module(module, layer_name)
        device = get_device(layer)
        QuantLinear = dynamic_import_inference_linear(bits, group_size, backend)
        if isinstance(layer, nn.Linear):
            in_features = layer.in_features
            out_features = layer.out_features
        elif isinstance(layer, Conv1D):
            in_features = layer.weight.shape[0]
            out_features = layer.weight.shape[1]
        bias = layer.bias is not None
        new_layer = QuantLinear(
            bits,
            group_size,
            in_features,
            out_features,
            bias,
            weight_dtype=layer.weight.dtype,
        )

        new_layer.device = device
        set_module(module, layer_name, new_layer)


def qbits_post_init(model):
    dep_check = True
    import auto_round_extension.qbits.qlinear_qbits as qlinear_qbits

    for layer in model.modules():
        if isinstance(layer, qlinear_qbits.QuantLinear):
            if dep_check:
                layer.req_check()
            layer.post_init()
            dep_check = False
    return model


def post_init_auto_round_model(model):
    """Post-initialization that require device information, for example buffers initialization on device.

    Args:
        model (`nn.Module`):
            The input model
    """
    from auto_round_extension.cuda.post_init import autoround_post_init

    model = autoround_post_init(model)
    # there are no side-effects after call qbits_post_init when model quant-type not equal to qbits.
    model = qbits_post_init(model)

    return model
