# Copyright 2025 The HuggingFace Team. All rights reserved.
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
"FP-Quant integration file"

from ..utils import (
    is_fp_quant_available,
)

import torch
from typing import Optional

if is_fp_quant_available():
    from fp_quant import FPQuantConfig as FPQuantLinearConfig
    from fp_quant import FPQuantDtype

from transformers.utils.quantization_config import FPQuantConfig
from ..quantizers.quantizers_utils import get_module_from_name
from ..core_model_loading import ConversionOps

class FpQuantQuantize(ConversionOps):
    def __init__(self, hf_quantizer):
        self.hf_quantizer = hf_quantizer

    def convert(self, input_dict: torch.Tensor, model: Optional[torch.nn.Module] = None, missing_keys: Optional[list[str]] = None, **kwargs) -> dict[str, torch.Tensor]:
        target_key, value = tuple(input_dict.items())[0]
        value = value[0] if isinstance(value, list) else value

        module, _ = get_module_from_name(model, target_key)

        # TODO: check if we need this or not, commented for now
        # if target_device == "cpu" and param_name.endswith("weight"):
        #     # Works agains hard-coded missing key dispatch to CPU
        #     return
        # The module holds either:
        #  * `weight` when `store_master_weights=True`
        #  * `qweight` and `scales` when `store_master_weights=False` and `pseudoquantization=False`
        #  * `dqweight` when `store_master_weights=False` and `pseudoquantization=True`
        if target_key.endswith(".qweight"):
            # Loading a real quantized checkpoint without master weights
            qweight = torch.nn.Parameter(
                value,
                requires_grad=False,
            )
            
            weight_key = target_key.rsplit(".", 1)[0] + ".weight"
            dqweight_key = target_key.rsplit(".", 1)[0] + ".dqweight"

            return {f"{target_key}": qweight}

        if target_key.endswith(".dqweight"):
            print(f"target_key: {target_key}")
            # Loading a pseudo-quantized checkpoint without master weights
            dqweight = torch.nn.Parameter(value)
            
            weight_key = target_key.rsplit(".", 1)[0] + ".weight"
            dqweight_key = target_key.rsplit(".", 1)[0] + ".dqweight"
            scales_key = target_key.rsplit(".", 1)[0] + ".scales"
            
            return {
                f"{target_key}": dqweight,
                f"{weight_key}": torch.nn.Parameter(torch.empty(0)),
                f"{dqweight_key}": torch.nn.Parameter(torch.empty(0)),
                f"{scales_key}": torch.nn.Parameter(torch.empty(0))
                }

        # Loading master weights or an unquantized checkpoint
        weight = torch.nn.Parameter(value)
        module.weight = weight

        # print(f"module.state_dict(): {module.state_dict()}")
        # Let pre-forward handle the quantization and set None where necessary
        module.pre_forward()

        prefix_target_key = target_key.rsplit(".", 1)[0]

        # keys are set inside the module.pre_forward() method, we don't need remove them from the missing keys list
        missing_keys.discard(target_key)
        missing_keys.discard(f"{prefix_target_key}.backward_hadamard_matrix")
        missing_keys.discard(f"{prefix_target_key}.forward_hadamard_matrix")
        missing_keys.discard(f"{prefix_target_key}.act_global_scale")
        missing_keys.discard(f"{prefix_target_key}.weight_global_scale")
        missing_keys.discard(f"{prefix_target_key}.qweight")
        missing_keys.discard(f"{prefix_target_key}.scales")
        missing_keys.discard(f"{prefix_target_key}.dqweight")
        return {}

def adapt_fp_quant_config(config: FPQuantConfig):
    if config.forward_dtype == "mxfp4":
        forward_dtype = FPQuantDtype.MXFP4
    elif config.forward_dtype == "nvfp4":
        forward_dtype = FPQuantDtype.NVFP4
    else:
        raise ValueError(f"Unsupported forward dtype: {config.forward_dtype}")

    if config.backward_dtype == "bf16":
        backward_dtype = FPQuantDtype.BF16
    elif config.backward_dtype == "mxfp8":
        backward_dtype = FPQuantDtype.MXFP8
    elif config.backward_dtype == "mxfp4":
        backward_dtype = FPQuantDtype.MXFP4
    else:
        raise ValueError(f"Unsupported backward dtype: {config.backward_dtype}")

    return FPQuantLinearConfig(
        forward_dtype=forward_dtype,
        forward_method=config.forward_method,
        backward_dtype=backward_dtype,
        store_master_weights=config.store_master_weights,
        hadamard_group_size=config.hadamard_group_size,
        pseudoquantization=config.pseudoquantization,
        transform_init=config.transform_init,
        modules_to_not_convert=config.modules_to_not_convert,
    )