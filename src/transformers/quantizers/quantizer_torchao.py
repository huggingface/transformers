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
from typing import TYPE_CHECKING

from .base import HfQuantizer
from .quantizers_utils import get_module_from_name


if TYPE_CHECKING:
    from ..modeling_utils import PreTrainedModel

from typing import Any, Dict, List

from ..utils import is_torch_available, is_torchao_available, logging


if is_torch_available():
    import torch

if is_torchao_available():
    from torchao.quantization import (
        quantize_,
    )

logger = logging.get_logger(__name__)


# Finds the parent of a node module named "name"
def find_parent(model, name):
    module_tree = name.split(".")[:-1]
    parent = model
    for m in module_tree:
        parent = parent._modules[m]
    return parent


class TorchAoHfQuantizer(HfQuantizer):
    """
    Quantizer for torchao: https://github.com/pytorch/ao/
    """

    requires_parameters_quantization = True
    requires_calibration = False
    required_packages = ["torchao"]

    def __init__(self, quantization_config, **kwargs):
        super().__init__(quantization_config, **kwargs)

    def validate_environment(self, device_map, **kwargs):
        if not is_torchao_available():
            raise ImportError("Loading an torchao quantized model requires torchao library (`pip install torchao`)")

    def update_torch_dtype(self, torch_dtype):
        if self.quantization_config.quant_method == "int4_weight_only":
            if torch_dtype is not None and torch_dtype != torch.bfloat16:
                logger.warn(
                    f"Setting torch_dtype to {torch_dtype} for int4_weight_only quantization, but only bfloat16 is supported right now."
                )
        return torch_dtype

    def _process_model_before_weight_loading(self, model: "PreTrainedModel", **kwargs):
        from ..integrations import get_keys_to_not_convert
        self.modules_to_not_convert = get_keys_to_not_convert(model)

        if self.quantization_config.modules_to_not_convert is not None:
            self.modules_to_not_convert.extend(self.quantization_config.modules_to_not_convert)

        return

    def check_quantized_param(
        self,
        model: "PreTrainedModel",
        param_value: "torch.Tensor",
        param_name: str,
        state_dict: Dict[str, Any],
        **kwargs,
    ) -> bool:
        # check if the param_name is not in self.modules_to_not_convert
        if any((key + "." in param_name) or (key == param_name) for key in self.modules_to_not_convert):
            return False
        else:
            # we only quantize the weight of nn.Linear
            module, tensor_name = get_module_from_name(model, param_name)
            return isinstance(module, torch.nn.Linear) and (tensor_name == "weight")

    def create_quantized_param(
        self,
        model: "PreTrainedModel",
        param_value: "torch.Tensor",
        param_name: str,
        target_device: "torch.device",
        state_dict: Dict[str, Any],
        unexpected_keys: List[str],
    ):
        """
        Each nn.Linear layer that needs to be quantized is processsed here.
        First, we set the value the weight tensor, then we move it to the target device. Finally, we quantize the module.
        """
        module, tensor_name = get_module_from_name(model, param_name)
        module._parameters[tensor_name] = torch.nn.Parameter(param_value).to(device=target_device)
        quantize_(module, self.quantization_config.get_apply_tensor_subclass())

    def _process_model_after_weight_loading(self, model):
        """No process required for torchao quantized model"""
        return

    @property
    def is_serializable(self):
        return False

    @property
    def is_trainable(self):
        # torchao does not have official support for QAT (Quantization Aware Training)
        # but torchao support nf4/PEFT, but it is not integrated yet
        # TODO: if this is supported in the future, do a version check here.
        return False
