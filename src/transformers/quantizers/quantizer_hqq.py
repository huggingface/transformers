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
                "A valid HQQ version (>=0.2.1) is not available. Please follow the instructions to install it: `https://github.com/mobiusml/hqq/`."
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

    def update_missing_keys(
        self, model: "PreTrainedModel", missing_keys: List[str], prefix: str, **kwargs
    ) -> List[str]:
        if self.pre_quantized:
            return [key for key in missing_keys if ("weight" not in key)]
        else:
            return missing_keys

    # Adds missing keys for HQQLinear modules that are loaded but the model with initialized with torch.nn.Linear
    def update_expected_keys(
        self, model: "PreTrainedModel", expected_keys: List[str], loaded_keys: List[str]
    ) -> List[str]:
        if not self.pre_quantized:
            return expected_keys

        # Collects all quantizable (linear) layers
        def _find_hqq_quantizable_layers(model, layers):
            for name, module in model.named_children():
                if isinstance(module, (torch.nn.Linear)):
                    layers.add(module.name)
                _find_hqq_quantizable_layers(module, layers)

        new_keys = set(expected_keys)
        if is_hqq_available():
            from hqq.core.quantize import HQQLinear

            # Name modules
            for name, module in model.named_modules():
                module.name = name

            # valid modules are Linear layers that have HQQLinear state_dict. We ignore skip_modules and any layers with Linear state_dict() params
            _valid_modules = set()
            _find_hqq_quantizable_layers(model, _valid_modules)

            # Remove skipped modules
            _skipped_modules = set()
            for _module in _valid_modules:
                for _skip_module in model.config.quantization_config["skip_modules"]:
                    if _skip_module in _module:
                        _skipped_modules.add(_module)
            _valid_modules -= _skipped_modules

            # Append new expected layers based on _ref_keys
            _ref_keys = HQQLinear(
                linear_layer=None, quant_config=None, compute_dtype=torch.float16, device="cpu"
            ).state_dict_keys() - {"bias"}

            # Clean-up
            _rm_keys = set()
            for key in new_keys:
                if any(_module in key for _module in _valid_modules):
                    _rm_keys.add(key)
            new_keys -= _rm_keys
            # At this point, new_keys contains all the keys of the layers that are NOT HQQLinear or torch.nn.Linear

            # Re-populate Linear/HQQLinear
            for _module in _valid_modules:
                if _module + ".weight" in loaded_keys:
                    new_keys.add(_module + ".weight")
                else:
                    new_keys.update({_module + "." + _ref_key for _ref_key in _ref_keys})
                if _module + ".bias" in loaded_keys:
                    new_keys.add(_module + ".bias")

        return list(new_keys)

    def check_quantized_param(
        self,
        model: "PreTrainedModel",
        param_value: "torch.Tensor",
        param_name: str,
        state_dict: Dict[str, Any],
        **kwargs,
    ) -> bool:
        if is_hqq_available():
            from hqq.core.quantize import HQQLinear
        module, tensor_name = get_module_from_name(model, param_name)

        if self.pre_quantized:
            return (isinstance(module, torch.nn.Linear) or isinstance(module, HQQLinear)) and tensor_name != "weight"
        else:
            return (
                isinstance(module, torch.nn.Linear)
                and tensor_name == "weight"
                # bias doesn't need to be quantized, we use this as a workaround to avoid loading bias into HQQLinear assuming it was loaded
                # in the state_dict directly with the weight because hqq overwrote load_state_dict for this layer
                or (isinstance(module, HQQLinear) and tensor_name == "bias")
            )

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
        Each nn.Linear layer is processed here.
        We first check if the corresponding module state_dict contains already HQQ quantized parameters.
        If not, we create a temp linear layer with the module state_dict params and use it for quantization
        """

        if is_hqq_available():
            from hqq.core.quantize import HQQLinear

        module, tensor_name = get_module_from_name(model, param_name)
        layer_name = ".".join(param_name.split(".")[:-1])
        parent_module = find_parent(model, layer_name)
        node = layer_name.split(".")[-1]

        if tensor_name == "bias":
            # this should already be set
            return

        # set module state_dict
        module_state_dict = {}
        for k, v in state_dict.items():
            if layer_name + "." in k:
                module_state_dict[k.split(".")[-1]] = v
                if unexpected_keys is not None and k in unexpected_keys:
                    unexpected_keys.remove(k)

        if self.pre_quantized:
            if isinstance(module, HQQLinear):
                return
            else:
                hqq_layer = HQQLinear(
                    linear_layer=None,
                    quant_config=None,
                    compute_dtype=self.torch_dtype,
                    device=target_device,
                )

            hqq_layer.load_state_dict(module_state_dict)

            if hqq_layer.bias is not None and isinstance(hqq_layer.bias, torch.Tensor):
                hqq_layer.bias = torch.nn.Parameter(hqq_layer.bias)

            if self.using_multi_gpu:
                hqq_layer = self._patch_layer_for_multigpu(hqq_layer)

            setattr(parent_module, node, hqq_layer)

            # cleanup
            del module.__dict__, module
            torch.cuda.empty_cache()
            return

        # Step 1: populate module with weight/bias from module state dict
        for key in module_state_dict:
            setattr(module, key, torch.nn.Parameter(module_state_dict[key]))

        # Step 2: Replace module with either HQQLinear or move it to device. We do this via setattr on the parent as doing on it on the module
        # directly doesn't work.
        quant_config = model.config.quantization_config["quant_config"]
        skip_modules = model.config.quantization_config["skip_modules"]
        module_tag = ".".join(module.name.split(".")[-2:])
        module_quant_config = None
        if "weight_quant_params" in quant_config:
            module_quant_config = quant_config
        elif module_tag in quant_config:
            module_quant_config = quant_config[module_tag]

        for skip_module in skip_modules:
            if skip_module in module.name:
                module_quant_config = None
                break

        if module_quant_config is not None:
            hqq_layer = HQQLinear(
                module,
                quant_config=module_quant_config,
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
        **kwargs,
    ):
        # Add the corresponding quant_config to each valid module. This allows us to do the actual nn.Linear -> HQQLinear conversion in create_quantized_param().
        # prepare_for_hqq_linear() also sets the right quantization config inside the model (model.config.quantization_config) and the layers (hqq_layer.quant_config)
        model = prepare_for_hqq_linear(model, quantization_config=self.quantization_config)

    def _process_model_after_weight_loading(self, model: "PreTrainedModel", **kwargs):
        model.is_hqq_quantized = True
        model.is_hqq_serializable = self.is_serializable()
        return model

    def is_serializable(self, safe_serialization=None):
        return True

    @property
    def is_trainable(self) -> bool:
        return True
