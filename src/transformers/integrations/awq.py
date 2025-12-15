# Copyright 2023 The HuggingFace Team. All rights reserved.
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
"AWQ (Activation aware Weight Quantization) integration file"

from typing import Optional, Union

from ..quantizers.quantizers_utils import should_convert_module
from ..utils import is_accelerate_available, is_torch_available, logging


if is_accelerate_available():
    from accelerate import init_empty_weights

if is_torch_available():
    import torch
    import torch.nn as nn

logger = logging.get_logger(__name__)


AWQ_SCALES_MAPPINGS = {
    "starcoder2": {"act": "act", "layer_before_act": "c_fc"},
    "RefinedWebModel": {"act": "act", "layer_before_act": "dense_h_to_4h"},
    "falcon": {"act": "act", "layer_before_act": "dense_h_to_4h"},
    "mpt": {"act": "act", "layer_before_act": "up_proj"},
    "gptj": {"act": "act", "layer_before_act": "fc_in"},
    "gpt_neox": {"act": "act", "layer_before_act": "dense_h_to_4h"},
    "gpt_bigcode": {"act": "act", "layer_before_act": "c_fc"},
    "bloom": {"act": "gelu_impl", "layer_before_act": "dense_h_to_4h"},
}


def replace_quantization_scales(model, model_type):
    from gptqmodel.quantization.awq.modules.act import ScaledActivation

    if model_type not in AWQ_SCALES_MAPPINGS:
        return model
    for name, module in model.named_children():
        act_name = AWQ_SCALES_MAPPINGS[model_type]["act"]
        layer_before_act_name = AWQ_SCALES_MAPPINGS[model_type]["layer_before_act"]
        if name == act_name and hasattr(model, layer_before_act_name):
            layer_before_act = getattr(model, AWQ_SCALES_MAPPINGS[model_type]["layer_before_act"])
            size = layer_before_act.out_features
            scale_like = torch.ones(size)
            model._modules[name] = ScaledActivation(module, scale_like)
        _ = replace_quantization_scales(module, model_type)
    return model


def replace_with_awq_linear(
    model,
    modules_to_not_convert=None,
    quantization_config=None,
    device_map: Optional[Union[str, dict]] = None,
) -> bool:
    """
    Public method that replaces the linear layers of the given model with awq quantized layers.

    Args:
        model (`torch.nn.Module`):
            The model to convert, can be any `torch.nn.Module` instance.
        quantization_config (`AwqConfig`):
            The quantization config object that contains the quantization parameters.
        modules_to_not_convert (`list[str]`, *optional*, defaults to `None`):
            A list of nn.Linear weights to not convert. If a parameter path is in the list (e.g. `lm_head.weight`), the corresponding module will not be
            converted.
        device_map (`Union[str, dict]`, *optional*, defaults to `None`):
            The device map that maps the parameters to the device
    """
    from gptqmodel.quantization import METHOD
    from gptqmodel.utils.importer import hf_select_quant_linear_v2

    target_cls = hf_select_quant_linear_v2(
        bits=quantization_config.bits,
        group_size=quantization_config.group_size,
        desc_act=False,
        sym=False,
        format=quantization_config.format,
        backend=quantization_config.backend,
        device_map=device_map,
        quant_method=METHOD.AWQ,
        zero_point=quantization_config.zero_point,
        pack=False,
    )

    for module_name, module in model.named_modules():
        if not should_convert_module(module_name, modules_to_not_convert):
            continue
        with init_empty_weights():
            if isinstance(module, nn.Linear):
                new_module = target_cls(
                    bits=quantization_config.bits,
                    sym=quantization_config.sym,
                    desc_act=quantization_config.desc_act,
                    group_size=quantization_config.group_size,
                    in_features=module.in_features,
                    out_features=module.out_features,
                    bias=module.bias is not None,
                    dev=module.weight.device,
                    register_buffers=True,
                )
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
