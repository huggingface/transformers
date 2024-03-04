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
import importlib
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from packaging import version

from .base import HfQuantizer
from .quantizers_utils import get_module_from_name


if TYPE_CHECKING:
    from ..modeling_utils import PreTrainedModel

from ..utils import is_quanto_available, is_torch_available, logging
from ..utils.quantization_config import QuantoConfig


if is_torch_available():
    import torch

logger = logging.get_logger(__name__)


class QuantoHfQuantizer(HfQuantizer):
    """
    Quantizer for the quanto library
    """

    required_packages = ["quanto", "accelerate"]
    requires_parameters_quantization = True
    requires_calibration = False

    def __init__(self, quantization_config: QuantoConfig, **kwargs):
        super().__init__(quantization_config, **kwargs)
        self.post_init()

    def post_init(self):
        r"""
        Safety checker
        """
        if self.quantization_config.activations is not None and not self.pre_quantized:
            raise ValueError(
                "We don't support quantizing the activations with transformers library."
                "Use quanto library for more complex use cases such as activations quantization, calibration and quantization aware training."
            )

    def validate_environment(self, *args, **kwargs):
        if not is_quanto_available():
            raise ImportError("Loading a quanto quantized model requires quanto library (`pip install quanto`)")
        device_map = kwargs.get("device_map", None)
        if device_map is not None and isinstance(device_map, dict):
            if "cpu" in device_map.values() or "disk" in device_map.values():
                if version.parse(importlib.metadata.version("accelerate")) <= version.parse("0.27.0"):
                    raise ValueError(
                        "You have a version of `accelerate` that is not compatible cpu/disk offload with quanto quantized model. "
                        "You need to install a version of accelerate > 0.27.0."
                    )

    def update_torch_dtype(self, torch_dtype: "torch.dtype") -> "torch.dtype":
        if torch_dtype is None:
            # TODO: Discuss if we should do that for quanto. I think that we should probably not do that and let the user cast the torch_dtype by themselves.
            # since in this case, quanto can also work on cpu.
            # If a user have both a cpu and cuda and he wants to play with quanto on cpu, he will have a specify manually torch_dtype to torch.float32.
            if torch.cuda.is_available():
                torch_dtype = torch.float16
                logger.info(
                    "CUDA available. Assuming Quanto inference on GPU and loading the model in `torch.float16`. To overwrite it, set `torch_dtype` manually."
                )
            else:
                torch_dtype = torch.float32
                logger.info(
                    "CUDA is unavailable. Assuming AQLM inference on CPU and loading the model in `torch.float32`. To overwrite it, set `torch_dtype` manually."
                )
        return torch_dtype

    def update_weights_only_kwarg(self, weights_only_kwarg: Dict[str, Any]) -> Dict[str, Any]:
        weights_only_kwarg["weights_only"] = False
        return weights_only_kwarg

    def update_missing_keys(self, model, missing_keys: List[str], prefix: str) -> List[str]:
        import quanto

        # if the model is prequantized, we don't need to remove any keys
        if self.pre_quantized:
            return missing_keys
        not_missing_keys = []
        model_state_dict_keys = model.state_dict().keys()
        for key in missing_keys:
            updated_key = None
            if key in list(model_state_dict_keys):
                updated_key = key
            elif f"{prefix}.{key}" in list(model_state_dict_keys):
                updated_key = f"{prefix}.{key}"
            elif key.startswith(prefix) and ".".join(key.split(".")[1:]) in list(model_state_dict_keys):
                updated_key = ".".join(key.split(".")[1:])
            module, tensor_name = get_module_from_name(model, updated_key)
            # we remove some of the missing keys since when we replaced the modules by QModuleMixin, we created a few buffers.
            if isinstance(module, quanto.QModuleMixin):
                if tensor_name != "weight" and tensor_name != "bias":
                    not_missing_keys.append(key)
        return [k for k in missing_keys if k not in not_missing_keys]

    def check_quantized_param(
        self, model: "PreTrainedModel", param_value: "torch.Tensor", param_name: str, state_dict: Dict[str, Any]
    ) -> bool:
        """
        Check if a parameter needs to be quantized.
        """
        import quanto

        if self.pre_quantized:
            return False

        module, tensor_name = get_module_from_name(model, param_name)
        # We only quantize the weights and the bias is not quantized.
        if isinstance(module, quanto.QModuleMixin) and tensor_name == "weight":
            # if the weights are quantized, don't need to recreate it again with `create_quantized_param`
            return not module.frozen
        else:
            return False

    def adjust_max_memory(self, max_memory: Dict[str, Union[int, str]]) -> Dict[str, Union[int, str]]:
        max_memory = {key: val * 0.90 for key, val in max_memory.items()}
        return max_memory

    def create_quantized_param(
        self,
        model: "PreTrainedModel",
        param_value: "torch.Tensor",
        param_name: str,
        target_device: "torch.device",
        *args,
        **kwargs,
    ):
        """
        Create the quantized parameter by calling .freeze() after setting it to the module.
        """
        from accelerate.utils import set_module_tensor_to_device

        set_module_tensor_to_device(model, param_name, target_device, param_value)
        module, _ = get_module_from_name(model, param_name)
        module.freeze()
        module.weight.requires_grad = False

    def adjust_target_dtype(self, target_dtype: "torch.dtype") -> "torch.dtype":
        if version.parse(importlib.metadata.version("accelerate")) > version.parse("0.27.0"):
            from accelerate.utils import CustomDtype

            mapping = {
                "int8": torch.int8,
                "float8": CustomDtype.FP8,
                "int4": CustomDtype.INT4,
                "int2": CustomDtype.INT2,
            }
            target_dtype = mapping[self.quantization_config.weights]
            return target_dtype
        else:
            raise ValueError(
                "You are using `device_map='auto'` on a quanto quantized model. To automatically compute"
                " the appropriate device map, you should upgrade your `accelerate` library,"
                "`pip install --upgrade accelerate` or install it from source."
            )

    def _process_model_before_weight_loading(
        self, model: "PreTrainedModel", keep_in_fp32_modules: List[str] = [], **kwargs
    ):
        from ..integrations import get_keys_to_not_convert, replace_with_quanto_layers

        # We keep some modules such as the lm_head in their original dtype for numerical stability reasons
        if self.quantization_config.modules_to_not_convert is None:
            self.modules_to_not_convert = get_keys_to_not_convert(model)
        else:
            self.modules_to_not_convert = self.quantization_config.modules_to_not_convert

        if not isinstance(self.modules_to_not_convert, list):
            self.modules_to_not_convert = [self.modules_to_not_convert]

        self.modules_to_not_convert.extend(keep_in_fp32_modules)

        model, _ = replace_with_quanto_layers(
            model, modules_to_not_convert=self.modules_to_not_convert, quantization_config=self.quantization_config
        )
        model.config.quantization_config = self.quantization_config

    def _process_model_after_weight_loading(self, model):
        return model

    @property
    def is_trainable(self, model: Optional["PreTrainedModel"] = None):
        return False

    @property
    def is_serializable(self):
        return True

    @property
    def is_safe_serializable(self):
        logger.warning(
            "Serialization with safetensors is not supported with models quantized with quanto. "
            "Please pass `safe_serialization=False` in `save_pretrained`. You will most likely face errors or"
            " unexpected behaviours."
        )
        return False
