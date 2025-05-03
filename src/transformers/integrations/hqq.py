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
"HQQ (Half-Quadratic Quantization) integration file"

from ..utils import is_hqq_available, is_torch_available, logging


if is_torch_available():
    import torch

logger = logging.get_logger(__name__)


# Name all modules inside the model
def autoname_modules(model):
    for name, module in model.named_modules():
        module.name = name


# Get the linear_tag from a module name. For example: model.layers.31.self_attn.k_proj -> self_attn.k_proj
def name_to_linear_tag(name):
    return ".".join([n for n in name.split(".") if ((n not in ["model", "layers"]) and (not n.isnumeric()))])


# Get all linear tags available
def get_linear_tags(model):
    if is_hqq_available():
        from hqq.core.quantize import HQQLinear

    linear_tags = set()
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.Linear, HQQLinear)):
            linear_tags.add(name_to_linear_tag(name))
    return list(linear_tags)


def _prepare_for_hqq_linear(model, patch_params, has_been_replaced, current_key_name=None):
    for name, module in model.named_children():
        if current_key_name is None:
            current_key_name = []
        current_key_name.append(name)

        if isinstance(module, torch.nn.Linear):
            # Get linear tag
            linear_tag = name_to_linear_tag(module.name)

            # We put the module quant_config into the nn.Linear layer so we can access it later in quantizer_hqq.create_quantized_param()
            if linear_tag in patch_params:
                if patch_params[linear_tag] is not None:
                    model._modules[name].quant_config = patch_params[linear_tag]
                    # Store the module class in case we need to transpose the weight later
                    model._modules[name].source_cls = type(module)
                    # Force requires grad to False to avoid unexpected errors
                    model._modules[name].requires_grad_(False)

            has_been_replaced = True

            # Add these fake parameters to avoid loading fail
            for att in ["W_q", "meta"]:
                setattr(module, att, None)

        if len(list(module.children())) > 0:
            _, has_been_replaced = _prepare_for_hqq_linear(
                module,
                patch_params=patch_params,
                has_been_replaced=has_been_replaced,
            )
        # Remove the last key for recursion
        current_key_name.pop(-1)

    return model, has_been_replaced


def prepare_for_hqq_linear(model, quantization_config=None, modules_to_not_convert=None, has_been_replaced=False):
    """
    Prepares nn.Linear layers for HQQ quantization.
    Since each layer type can have separate quantization parameters, we need to do the following:
    1- tag each module with its name via autoname_modules()
    2- Extract linear_tags (e.g. ['self_attn.q_proj', ...])
    3- Map quantization parameters as a dictionary linear_tag -> quant_params as HQQLinear expects it, this is referred to as patch_params
    """

    modules_to_not_convert = [] if modules_to_not_convert is None else modules_to_not_convert

    # Add name to module
    autoname_modules(model)

    # Get linear tags. This allows us to use different quant params to different layer types
    linear_tags = get_linear_tags(model)

    # Convert quantization_config to layer-wise config
    skip_modules = quantization_config.skip_modules
    quant_config = quantization_config.quant_config
    linear_tags = list(set(linear_tags) - set(skip_modules) - set(modules_to_not_convert))

    if any(key in linear_tags for key in quant_config.keys()):
        # If the user doesn't specify a key from get_linear_tags, the layer is not quantized via (key, None)
        patch_params = dict.fromkeys(linear_tags)
        patch_params.update(quant_config)
    else:
        # Same quant_config for all layers
        patch_params = dict.fromkeys(linear_tags, quant_config)

    model, has_been_replaced = _prepare_for_hqq_linear(
        model, patch_params=patch_params, has_been_replaced=has_been_replaced
    )

    # We store quantization config as linear_tag -> hqq quant config
    model.config.quantization_config = {
        "quant_config": quant_config,
        "quant_method": quantization_config.quant_method,
        "skip_modules": skip_modules,
    }

    if not has_been_replaced:
        logger.warning("No linear modules were found in your model for quantization.")

    return model
