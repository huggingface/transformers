# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
from ..quantizers.quantizers_utils import get_module_from_name
from typing import Optional

from ..utils import is_torch_available

if is_torch_available():
    import torch

class QuarkDeserialize(ConversionOps):
    def __init__(self, hf_quantizer):
        self.hf_quantizer = hf_quantizer

    def convert(self, input_dict: torch.Tensor, model: Optional[torch.nn.Module] = None, missing_keys: Optional[list[str]] = None, full_layer_name: str | None = None, **kwargs) -> dict[str, torch.Tensor]:

        target_key, value = tuple(input_dict.items())[0]
        value = value[0] if isinstance(value, list) else value
        # this will get the param name : weight, input, bias or output
        param = target_key.split("_", 1)[0]
        quant_state = target_key.split("_", 1)[-1]
        # print(f"full_layer_name: {full_layer_name}")
        print(f"value: {value.dtype}")

        sub_module_state = full_layer_name.rsplit(".", 1)[0] + "." + param + "_quantizer" + "." + quant_state

        missing_keys.discard(full_layer_name)

        return {sub_module_state: value}