# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
from typing import Any

from .auto import AutoHfQuantizer


def get_module_from_name(module, tensor_name: str) -> tuple[Any, str]:
    if "." in tensor_name:
        module_name, tensor_name = tensor_name.rsplit(".", 1)
        module = module.get_submodule(module_name)
    return module, tensor_name


def get_hf_quantizer(
    config, quantization_config, torch_dtype, from_tf, from_flax, device_map, weights_only, user_agent
):
    pre_quantized = hasattr(config, "quantization_config")
    if pre_quantized and not AutoHfQuantizer.supports_quant_method(config.quantization_config):
        pre_quantized = False

    if pre_quantized or quantization_config is not None:
        if pre_quantized:
            config.quantization_config = AutoHfQuantizer.merge_quantization_configs(
                config.quantization_config, quantization_config
            )
        else:
            config.quantization_config = quantization_config

        hf_quantizer = AutoHfQuantizer.from_config(
            config.quantization_config,
            pre_quantized=pre_quantized,
        )
    else:
        hf_quantizer = None

    if hf_quantizer is not None:
        hf_quantizer.validate_environment(
            torch_dtype=torch_dtype,
            from_tf=from_tf,
            from_flax=from_flax,
            device_map=device_map,
            weights_only=weights_only,
        )
        torch_dtype = hf_quantizer.update_torch_dtype(torch_dtype)
        device_map = hf_quantizer.update_device_map(device_map)
        config = hf_quantizer.update_tp_plan(config)

        # In order to ensure popular quantization methods are supported. Can be disable with `disable_telemetry`
        if not getattr(hf_quantizer.quantization_config, "dequantize", False):
            quant_method = hf_quantizer.quantization_config.quant_method
            user_agent["quant"] = getattr(quant_method, "value", quant_method)
    return hf_quantizer, config, torch_dtype, device_map
