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

from typing import TYPE_CHECKING, Any, Dict, List

from ..integrations import prepare_for_hqq_linear
from ..utils import is_accelerate_available, is_hqq_available, is_torch_available, logging
from .base import HfQuantizer
from .quantizers_utils import get_module_from_name


if TYPE_CHECKING:
    from ..modeling_utils import PreTrainedModel


if is_accelerate_available():
    from accelerate.hooks import remove_hook_from_module

if is_torch_available():
    import torch

logger = logging.get_logger(__name__)


# Finds the parent of a node module named "name"
def find_parent(model, name):
    module_tree = name.split(".")[:-1]
    parent = model
    for m in module_tree:
        parent = parent._modules[m]
    return parent


class HqqHfQuantizer(HfQuantizer):
    """
    HQQ quantizer base HF class.
    nn.Linear modules are first tagged with quant_config in _process_model_before_weight_loading().
    The actual quantization and offloading to the GPU is done in check_quantized_param().
    """

    use_keep_in_fp32_modules = False
    requires_parameters_quantization = True
    requires_calibration = False
    required_packages = ["hqq"]

    def __init__(self, quantization_config, **kwargs):
        super().__init__(quantization_config, **kwargs)
        self.torch_dtype = None
        self.using_multi_gpu = False

    def validate_environment(self, *args, **kwargs):
        if not (is_hqq_available()):
            raise ImportError(
                "HQQ is not available. Please follow the instructions to install it: `https://github.com/mobiusml/hqq/`"
            )

        if kwargs.get("from_tf", False) or kwargs.get("from_flax", False):
            raise ValueError(
                "Converting weights from tf/flax weights is currently not supported, please make"
                " sure the weights are in PyTorch format."
            )

        if not torch.cuda.is_available():
            raise RuntimeError("No GPU found. A GPU is needed for quantization.")

        if self.torch_dtype is None:
            if "torch_dtype" in kwargs:
                self.torch_dtype = kwargs["torch_dtype"]
            else:
                self.torch_dtype = torch.float32
                logger.info("Setting torch_dtype to torch.float32 as the default value since it was not specified.")

        device_map = kwargs.get("device_map", None)
        if isinstance(device_map, dict):
            if "cpu" in device_map.values() or "disk" in device_map.values():
                raise ValueError(
                    "You are attempting to use an HQQ model with a device_map that contains a CPU or disk device."
                    " This is not supported. Please remove the CPU or disk device from the device_map."
                )
            else:
                self.using_multi_gpu = len(set(device_map.values())) > 1

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
        We first check if the corresponding module state_dict contains already HQQ quantized parameters.
        If not, we create a temp linear layer with the module state_dict params and use it for quantization
        """

        if is_hqq_available():
            from hqq.core.quantize import HQQLinear

        module, tensor_name = get_module_from_name(model, param_name)

        layer_name = param_name.replace(".weight", "").replace(".bias", "")
        parent_module = find_parent(model, layer_name)
        node = layer_name.split(".")[-1]

        # Step 0: set module state_dict
        module_state_dict = {key.split(".")[-1]: state_dict[key] for key in state_dict if layer_name in key}

        # Step 1: populate module with weight/bias from module state dict
        for key in module_state_dict:
            setattr(module, key, torch.nn.Parameter(module_state_dict[key]))

        # Step 2: Replace module with either HQQLinear or move it to device. We do this via setattr on the parent as doing on it on the module
        # directly doesn't work.

        if hasattr(module, "quant_config"):
            hqq_layer = HQQLinear(
                module,
                module.quant_config,
                compute_dtype=self.torch_dtype,
                device=target_device,
                del_orig=True,
            )

            if hqq_layer.bias is not None and isinstance(hqq_layer.bias, torch.Tensor):
                hqq_layer.bias = torch.nn.Parameter(hqq_layer.bias)

            if self.using_multi_gpu:
                hqq_layer = self._patch_layer_for_multigpu(hqq_layer)

            setattr(parent_module, node, hqq_layer)

        else:
            module = module.to(dtype=self.torch_dtype, device=target_device)
            setattr(parent_module, node, module)

        torch.cuda.empty_cache()

    # Remove accelerate hook and uses a simpler forward pass. Otherwise, this breaks with multi-gpu
    def _patch_layer_for_multigpu(self, hqq_layer):
        hqq_layer = remove_hook_from_module(hqq_layer)

        def forward_with_device(self, x):
            out = torch.matmul(x.to(self.device), self.dequantize().t())
            if self.bias is not None:
                out += self.bias
            return out

        hqq_layer.forward = lambda x: forward_with_device(hqq_layer, x)
        return hqq_layer

    def _process_model_before_weight_loading(
        self,
        model: "PreTrainedModel",
        device_map,
        keep_in_fp32_modules: List[str] = None,
        **kwargs,
    ):
        keep_in_fp32_modules = keep_in_fp32_modules if keep_in_fp32_modules is not None else []

        # Add the corresponding quant_config to each valid module. This allows us to do the actual nn.Linear -> HQQLinear conversion in create_quantized_param().
        # prepare_for_hqq_linear() also sets the right quantization config inside the model (model.config.quantization_config) and the layers (hqq_layer.quant_config)
        model = prepare_for_hqq_linear(model, quantization_config=self.quantization_config)

    def _process_model_after_weight_loading(self, model: "PreTrainedModel", **kwargs):
        model.is_hqq_quantized = True
        model.is_hqq_serializable = self.is_serializable
        return model

    @property
    def is_serializable(self):
        return False

    @property
    def is_trainable(self) -> bool:
        return True
