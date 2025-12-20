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

from typing import Optional

import torch

from transformers.utils import logging
from transformers.utils.import_utils import is_torch_available


if is_torch_available():
    from ..core_model_loading import ConversionOps
from ..quantizers.quantizers_utils import get_module_from_name


logger = logging.get_logger(__name__)


class CompressedTensorsMarkInitialize(ConversionOps):
    def __init__(self, hf_quantizer):
        self.hf_quantizer = hf_quantizer

    def convert(
        self,
        input_dict: dict[str, torch.Tensor],
        model: Optional[torch.nn.Module] = None,
        full_layer_name: str | None = None,
        missing_keys=None,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        _, value = tuple(input_dict.items())[0]
        module, tensor_name = get_module_from_name(model, full_layer_name)
        module._is_hf_initialized = True

        return {full_layer_name: value}
