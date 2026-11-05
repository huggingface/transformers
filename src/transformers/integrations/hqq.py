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

from ..core_model_loading import ConversionOps
from ..quantizers.quantizer_hqq import HQQ_SERIALIZED_NAMES
from ..utils import is_hqq_available, is_torch_available, logging


if is_torch_available():
    import torch

if is_hqq_available():
    from hqq.core.quantize import HQQLinear
else:
    HQQLinear = object

logger = logging.get_logger(__name__)


class HqqLinear(HQQLinear):
    """
    An `HQQLinear` swapped in empty before loading, keeping the `weight` (on meta) of the `nn.Linear` it replaces:
    while empty, its `state_dict` lists that `weight`, so the loader routes the checkpoint tensors to it (the
    serialized `W_q` and meta data, or the unquantized weight to quantize). The load operations then fill it in place.
    """

    def state_dict(self, *args, destination=None, prefix="", keep_vars=False):
        if self.is_initialized():
            return super().state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
        # hqq's own empty `state_dict` doesn't write into `destination`, so the model's would not list this layer
        state = {"weight": self.weight}
        if self.bias is not None:
            state["bias"] = self.bias
        if destination is not None:
            destination.update({prefix + name: value for name, value in state.items()})
        return state


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

            if patch_params.get(linear_tag) is not None:
                model._modules[name] = _empty_hqq_linear(module, patch_params[linear_tag])

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


def _empty_hqq_linear(linear, quant_config):
    "The `HqqLinear` replacing `linear` before loading, keeping its `weight` and `bias` params"
    layer = HqqLinear(None, quant_config, del_orig=False, compute_dtype=linear.weight.dtype, initialize=False)
    weight_quant_params = layer.quant_config["weight_quant_params"]
    if weight_quant_params["group_size"] is None:
        weight_quant_params["group_size"] = (
            linear.in_features if weight_quant_params["axis"] == 1 else linear.out_features
        )
    layer.name, layer.source_cls = linear.name, type(linear)
    layer.weight, layer.bias = linear.weight, linear.bias
    layer.requires_grad_(False)
    return layer


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

    if any(key in linear_tags for key in quant_config):
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


def _hqq_linears(input_dict, prefix=""):
    "One `HQQLinear` per value of `{prefix}W_q` (one per expert when they are lists), loaded from its serialized state"
    from hqq.core.quantize import HQQLinear

    states = {}
    for name in HQQ_SERIALIZED_NAMES:
        value = input_dict.get(f"{prefix}{name}")
        if value is not None:
            states[name] = value if isinstance(value, list) else [value]
    layers = []
    for i, weight in enumerate(states["W_q"]):
        layer = HQQLinear(None, None, del_orig=False, compute_dtype=states["scale"][i].dtype, device=weight.device)
        layer.load_state_dict({name: values[i] for name, values in states.items()})
        layers.append(layer)
    return layers


class HqqDeserialize(ConversionOps):
    """The serialized state of an `HqqLinear` (`W_q` and its meta data) is loaded in it."""

    def __init__(self, hf_quantizer):
        self.hf_quantizer = hf_quantizer

    def convert(self, input_dict, model=None, full_layer_name=None, missing_keys=None, **kwargs):
        state = {name: input_dict[name] for name in HQQ_SERIALIZED_NAMES if name in input_dict}
        state = {name: value[0] if isinstance(value, list) else value for name, value in state.items()}
        module = model.get_submodule(full_layer_name.rpartition(".")[0])
        module.device, module.compute_dtype = state["W_q"].device, state["scale"].dtype
        bias, module.bias = module.bias, None  # hqq resets and moves it, loaded before or after this op
        module.load_state_dict(state)
        module.bias = bias
        return _loaded(module, full_layer_name, missing_keys)


class HqqQuantize(ConversionOps):
    """The unquantized weight of an `HqqLinear` is quantized in it, with its `quant_config`."""

    def __init__(self, hf_quantizer):
        self.hf_quantizer = hf_quantizer

    def convert(self, input_dict, model=None, full_layer_name=None, missing_keys=None, **kwargs):
        weight = next(iter(input_dict.values()))
        weight = weight[0] if isinstance(weight, list) else weight
        module = model.get_submodule(full_layer_name.rpartition(".")[0])
        module.device, module.compute_dtype = weight.device, weight.dtype
        bias, module.bias = module.bias, None  # hqq moves it, loaded before or after this op
        module.quantize(weight, **module.quant_config)
        module.bias = bias
        return _loaded(module, full_layer_name, missing_keys)


def _loaded(module, full_layer_name, missing_keys):
    "Drops the placeholder `weight` of the now loaded `module`: its state is `W_q` and the meta data, not params"
    del module.weight
    if missing_keys is not None:
        missing_keys.discard(full_layer_name)
    return {}


class HqqDequantize(ConversionOps):
    """Dequantized weights from the serialized `HQQLinear` states, dequantized by hqq itself."""

    def __init__(self, hf_quantizer):
        self.hf_quantizer = hf_quantizer

    def convert(self, input_dict, model=None, full_layer_name=None, **kwargs) -> dict[str, list[torch.Tensor]]:
        output = {}
        for key in input_dict:
            if key.endswith("W_q"):
                prefix = key.removesuffix("W_q")  # `W_q` -> `weight`, `experts.*.W_q` -> `experts.*.weight`
                output[f"{prefix}weight"] = [layer.dequantize() for layer in _hqq_linears(input_dict, prefix)]
        return output
