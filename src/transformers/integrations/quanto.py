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

from ..core_model_loading import ConversionOps
from ..quantizers.quantizers_utils import get_module_from_name, should_convert_module
from ..utils import is_torch_available, logging


if is_torch_available():
    import torch
    import torch.nn as nn

logger = logging.get_logger(__name__)


class QuantoQuantize(ConversionOps):
    def __init__(self, hf_quantizer):
        self.hf_quantizer = hf_quantizer

    def convert(
        self,
        input_dict: dict[str, list[torch.Tensor]],
        model: torch.nn.Module | None = None,
        full_layer_name: str | None = None,
        missing_keys: list[str] | None = None,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        _, value = tuple(input_dict.items())[0]
        value = value[0]

        from ..modeling_utils import _load_parameter_into_model

        _load_parameter_into_model(model, full_layer_name, value)
        module, _ = get_module_from_name(model, full_layer_name)
        module.freeze()
        module.weight.requires_grad = False
        module._is_hf_initialized = True

        # need to discard some missing keys we already updated the module in freeze.
        module_name = full_layer_name.rsplit(".", 1)[0]
        missing_keys.discard(f"{module_name}.weight")
        missing_keys.discard(f"{module_name}.input_scale")
        missing_keys.discard(f"{module_name}.output_scale")
        return {}


def replace_with_quanto_layers(
    model,
    quantization_config=None,
    modules_to_not_convert=None,
):
    """
    Public method that recursively replaces the Linear layers of the given model with Quanto quantized layers.
    Returns the converted model and a boolean that indicates if the conversion has been successful or not.

    Args:
        model (`torch.nn.Module`):
            The model to convert, can be any `torch.nn.Module` instance.
        quantization_config (`QuantoConfig`, defaults to `None`):
            The quantization config object that contains the quantization parameters.
        modules_to_not_convert (`list`, *optional*, defaults to `None`):
            A list of modules to not convert. If a module name is in the list (e.g. `lm_head`), it will not be
            converted.
    """
    from accelerate import init_empty_weights
    from optimum.quanto import QLayerNorm, QLinear, qfloat8, qint2, qint4, qint8

    w_mapping = {"float8": qfloat8, "int8": qint8, "int4": qint4, "int2": qint2}
    a_mapping = {None: None, "float8": qfloat8, "int8": qint8}

    has_been_replaced = False
    for module_name, module in model.named_modules():
        if not should_convert_module(module_name, modules_to_not_convert):
            continue
        with init_empty_weights():
            new_module = None
            if isinstance(module, nn.Linear):
                new_module = QLinear(
                    in_features=module.in_features,
                    out_features=module.out_features,
                    bias=module.bias is not None,
                    dtype=module.weight.dtype,
                    weights=w_mapping[quantization_config.weights],
                    activations=a_mapping[quantization_config.activations],
                )
            elif isinstance(module, torch.nn.LayerNorm) and quantization_config.activations is not None:
                new_module = QLayerNorm(
                    module.normalized_shape,
                    module.eps,
                    module.elementwise_affine,
                    module.bias is not None,
                    activations=a_mapping[quantization_config.activations],
                )
            if new_module is not None:
                has_been_replaced = True
                model.set_submodule(module_name, new_module)

    if not has_been_replaced:
        logger.warning(
            "You are loading your model using quanto but no linear modules were found in your model."
            " Please double check your model architecture, or submit an issue on github if you think this is"
            " a bug."
        )

    return model
