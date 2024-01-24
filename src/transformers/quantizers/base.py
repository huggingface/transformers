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
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, Optional, Union

from ..utils import is_torch_available
from ..utils.import_utils import _is_package_available
from ..utils.quantization_config import QuantizationConfigMixin


if TYPE_CHECKING:
    from ..modeling_utils import PreTrainedModel

if is_torch_available():
    import torch


class HFQuantizer(ABC):
    """
    Abstract class of the HuggingFace quantizer. Supports for now quantizing HF transformers models for inference and/or quantization.
    This class is used only for transformers.PreTrainedModel.from_pretrained and cannot be easily used outside the scope of that method
    yet.

    Attributes
        quantization_config (`transformers.utils.quantization_config.QuantizationConfigMixin`):
            The quantization config that defines the quantization parameters of your model that you want to quantize.
        modules_to_not_convert (`List[str]`, *optional*):
            The list of module names to not convert when quantizing the model.
        required_packages (`List[str]`, *optional*):
            The list of required pip packages to install prior to using the quantizer
        requires_calibration (`bool`):
            Whether the quantization method requires to calibrate the model before using it.
        requires_parameters_quantization (`bool`):
            Whether the quantization method requires to create a new Parameter. For example, for bitsandbytes, it is
            required to create a new xxxParameter in order to properly quantize the model.
    """

    requires_calibration = False
    required_packages = None
    requires_parameters_quantization = False

    def __init__(self, quantization_config: QuantizationConfigMixin, **kwargs):
        self.quantization_config = quantization_config

        # -- Handle extra kwargs below --
        self.modules_to_not_convert = kwargs.pop("modules_to_not_convert", [])
        self.pre_quantized = kwargs.pop("pre_quantized", True)

        if not self.pre_quantized and self.requires_calibration:
            raise ValueError(
                f"The quantization method {quantization_config.quant_method} does require the model to be pre-quantized."
                f" You explicitly passed `pre_quantized=False` meaning your model weights are not quantized. Make sure to "
                f"pass `pre_quantized=True` while knowing what you are doing."
            )

        self.check_packages_compatibility()

    def set_torch_dtype(self, torch_dtype: torch.dtype) -> torch.dtype:
        return torch_dtype

    def set_device_map(self, device_map: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        return device_map

    def adjust_target_dtype(self, torch_dtype: torch.dtype) -> torch.dtype:
        return torch_dtype

    def get_special_dtypes_update(self, model, torch_dtype: torch.dtype) -> Dict[str, torch.dtype]:
        """returns dtypes for modules that are not quantized"""
        return {
            name: torch_dtype
            for name, _ in model.named_parameters()
            if any(m in name for m in self.modules_to_not_convert)
        }

    def adjust_max_memory(self, max_memory: Dict[str, Union[int, str]]) -> Dict[str, Union[int, str]]:
        """adjust max_memory argument for infer_auto_device_map() if extra memory is needed for quantization"""
        return max_memory

    def check_quantized_param(
        self, model: "PreTrainedModel", param_value: torch.Tensor, param_name: str, state_dict: Dict[str, Any]
    ) -> bool:
        """
        checks if a loaded state_dict component is part of quantized param + some validation; only defined if
        requires_parameters_quantization == True for quantization methods that require to create a new parameters
        for quantization.
        """
        return False

    def create_quantized_param(self, *args, **kwargs) -> torch.nn.Parameter:
        """
        takes needed components from state_dict and creates quantized param; only applicable if
        requires_parameters_quantization == True
        """
        if not self.requires_parameters_quantization:
            raise AttributeError(
                f"`.create_quantized_param()` method is not supported by quantizer class {self.__class__.__name__}."
            )

    def validate_environment(self, *args, **kwargs):
        """
        This method is used to potentially check for potential conflicts with arguments that are
        passed in `from_pretrained`. You need to define it for all future quantizers that are integrated with transformers.
        If no explicit check are needed, simply return nothing.
        """
        return

    def check_packages_compatibility(self):
        if self.required_packages is not None:
            non_available_packages = []
            for package_name in self.required_packages:
                is_package_available = _is_package_available(package_name)
                if not is_package_available:
                    non_available_packages.append(package_name)

            if len(non_available_packages) > 0:
                raise ValueError(
                    f"The packages {self.required_packages} are required to use {self.__class__.__name__}"
                    f" the following packages are missing in your environment: {non_available_packages}, please make sure"
                    f" to install them in order to use the quantizer."
                )

    def preprocess_model(self, model: "PreTrainedModel", **kwargs):
        """setting model attributes and/or converting model BEFORE weights loading"""
        model.is_quantized = True
        model.quantization_method = self.quantization_config.quant_method
        return self._process_model_before_weight_loading(model, **kwargs)

    def postprocess_model(self, model: "PreTrainedModel", **kwargs):
        model._is_quantized_training_enabled = self.is_trainable
        return self._process_model_after_weight_loading(model, **kwargs)

    @abstractmethod
    def _process_model_before_weight_loading(self, model, **kwargs):
        ...

    @abstractmethod
    def _process_model_after_weight_loading(self, model, **kwargs):
        ...

    @property
    @abstractmethod
    def is_serializable(self):
        ...

    @property
    @abstractmethod
    def is_trainable(self):
        ...
