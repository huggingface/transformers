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
        self.torch_dtype = None

    def validate_environment(self, device_map, **kwargs):
        if not is_torchao_available():
            raise ImportError("Loading an torchao quantized model requires torchao library (`pip install torchao`)")

    def update_torch_dtype(self, torch_dtype):
        if self.quantization_config.quant_method == "int4_weight_only":
            if torch_dtype is not None and torch_dtype != torch.bfloat16:
                logger.warn(
                    f"Setting torch_dtype to {torch_dtype} for int4_weight_only quantization, but only bfloat16 is supported right now."
                )

        self.torch_dtype = torch_dtype
        return torch_dtype

    def check_quantized_param(
        self,
        model: "PreTrainedModel",
        param_value: "torch.Tensor",
        param_name: str,
        state_dict: Dict[str, Any],
        **kwargs,
    ) -> bool:
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
        Each nn.Linear layer is processsed here.
        We first check if the corresponding module state_dict contains already torchao quantized parameters.
        If not, we create a temp linear layer with the module state_dict params and use it for quantization
        """
        module, tensor_name = get_module_from_name(model, param_name)

        layer_name = param_name.replace(".weight", "").replace(".bias", "")
        parent_module = find_parent(model, layer_name)
        node = layer_name.split(".")[-1]

        # Step 0: set module state_dict
        module_state_dict = {key.split(".")[-1]: state_dict[key] for key in state_dict if layer_name in key}

        # Step 1: populate module with weight/bias from module state dict
        for key in module_state_dict:
            setattr(module, key, torch.nn.Parameter(module_state_dict[key]))

        # Step 2: Update the module using the `quantize_` API from TorchAO

        module = module.to(dtype=self.torch_dtype, device=target_device)
        quantize_(module, self.quantization_config.get_apply_tensor_subclass())
        setattr(parent_module, node, module)

    def _process_model_before_weight_loading(self, model: "PreTrainedModel", **kwargs):
        """No process required for torchao quantized model"""
        return

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
