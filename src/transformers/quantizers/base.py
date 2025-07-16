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
from typing import TYPE_CHECKING, Any, Optional, Union

from ..utils import is_torch_available
from ..utils.quantization_config import QuantizationConfigMixin, QuantizationMethod
from .quantizers_utils import get_module_from_name


if TYPE_CHECKING:
    from ..modeling_utils import PreTrainedModel

if is_torch_available():
    import torch
    from torch.nn import ModuleList
else:
    ModuleList = str


class HfQuantizer(ABC):
    """
    Abstract class of the HuggingFace quantizer. Supports for now quantizing HF transformers models for inference and/or quantization.
    This class is used only for transformers.PreTrainedModel.from_pretrained and cannot be easily used outside the scope of that method
    yet.

    Attributes
        quantization_config (`transformers.utils.quantization_config.QuantizationConfigMixin`):
            The quantization config that defines the quantization parameters of your model that you want to quantize.
        modules_to_not_convert (`list[str]`, *optional*):
            The list of module names to not convert when quantizing the model.
        required_packages (`list[str]`, *optional*):
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

    def update_torch_dtype(self, torch_dtype: "torch.dtype") -> "torch.dtype":
        """
        Some quantization methods require to explicitly set the dtype of the model to a
        target dtype. You need to override this method in case you want to make sure that behavior is
        preserved

        Args:
            torch_dtype (`torch.dtype`):
                The input dtype that is passed in `from_pretrained`
        """
        return torch_dtype

    def update_device_map(self, device_map: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
        """
        Override this method if you want to pass a override the existing device map with a new
        one. E.g. for bitsandbytes, since `accelerate` is a hard requirement, if no device_map is
        passed, the device_map is set to `"auto"``

        Args:
            device_map (`Union[dict, str]`, *optional*):
                The device_map that is passed through the `from_pretrained` method.
        """
        return device_map

    def adjust_target_dtype(self, torch_dtype: "torch.dtype") -> "torch.dtype":
        """
        Override this method if you want to adjust the `target_dtype` variable used in `from_pretrained`
        to compute the device_map in case the device_map is a `str`. E.g. for bitsandbytes we force-set `target_dtype`
        to `torch.int8` and for 4-bit we pass a custom enum `accelerate.CustomDtype.int4`.

        Args:
            torch_dtype (`torch.dtype`, *optional*):
                The torch_dtype that is used to compute the device_map.
        """
        return torch_dtype

    def update_missing_keys(self, model, missing_keys: list[str], prefix: str) -> list[str]:
        """
        Override this method if you want to adjust the `missing_keys`.

        Args:
            missing_keys (`list[str]`, *optional*):
                The list of missing keys in the checkpoint compared to the state dict of the model
        """
        return missing_keys

    def update_unexpected_keys(self, model, unexpected_keys: list[str], prefix: str) -> list[str]:
        """
        Override this method if you want to adjust the `unexpected_keys`.

        Args:
            unexpected_keys (`list[str]`, *optional*):
                The list of unexpected keys in the checkpoint compared to the state dict of the model
        """
        return unexpected_keys

    def update_missing_keys_after_loading(self, model, missing_keys: list[str], prefix: str) -> list[str]:
        """
        Override this method if you want to adjust the `missing_keys` after loading the model params,
        but before the model is post-processed.

        Args:
            missing_keys (`list[str]`, *optional*):
                The list of missing keys in the checkpoint compared to the state dict of the model
        """
        return missing_keys

    def update_expected_keys(self, model, expected_keys: list[str], loaded_keys: list[str]) -> list[str]:
        """
        Override this method if you want to adjust the `update_expected_keys`.

        Args:
            expected_keys (`list[str]`, *optional*):
                The list of the expected keys in the initialized model.
            loaded_keys (`list[str]`, *optional*):
                The list of the loaded keys in the checkpoint.
        """
        return expected_keys

    def get_special_dtypes_update(self, model, torch_dtype: "torch.dtype") -> dict[str, "torch.dtype"]:
        """
        returns dtypes for modules that are not quantized - used for the computation of the device_map in case
        one passes a str as a device_map. The method will use the `modules_to_not_convert` that is modified
        in `_process_model_before_weight_loading`.

        Args:
            model (`~transformers.PreTrainedModel`):
                The model to quantize
            torch_dtype (`torch.dtype`):
                The dtype passed in `from_pretrained` method.
        """

        return {
            name: torch_dtype
            for name, _ in model.named_parameters()
            if any(m in name for m in self.modules_to_not_convert)
        }

    def adjust_max_memory(self, max_memory: dict[str, Union[int, str]]) -> dict[str, Union[int, str]]:
        """adjust max_memory argument for infer_auto_device_map() if extra memory is needed for quantization"""
        return max_memory

    def check_quantized_param(
        self,
        model: "PreTrainedModel",
        param_value: "torch.Tensor",
        param_name: str,
        state_dict: dict[str, Any],
        **kwargs,
    ) -> bool:
        """
        checks if a loaded state_dict component is part of quantized param + some validation; only defined if
        requires_parameters_quantization == True for quantization methods that require to create a new parameters
        for quantization.
        """
        return False

    def create_quantized_param(self, *args, **kwargs) -> "torch.nn.Parameter":
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

    def update_tp_plan(self, config):
        "updates the tp plan for the scales"
        return config

    def preprocess_model(self, model: "PreTrainedModel", **kwargs):
        """
        Setting model attributes and/or converting model before weights loading. At this point
        the model should be initialized on the meta device so you can freely manipulate the skeleton
        of the model in order to replace modules in-place. Make sure to override the abstract method `_process_model_before_weight_loading`.

        Args:
            model (`~transformers.PreTrainedModel`):
                The model to quantize
            kwargs (`dict`, *optional*):
                The keyword arguments that are passed along `_process_model_before_weight_loading`.
        """
        model.is_quantized = True
        model.quantization_method = self.quantization_config.quant_method
        if self.pre_quantized:
            self._convert_model_for_quantization(model)
        return self._process_model_before_weight_loading(model, **kwargs)

    def postprocess_model(self, model: "PreTrainedModel", **kwargs):
        """
        Post-process the model post weights loading.
        Make sure to override the abstract method `_process_model_after_weight_loading`.

        Args:
            model (`~transformers.PreTrainedModel`):
                The model to quantize
            kwargs (`dict`, *optional*):
                The keyword arguments that are passed along `_process_model_after_weight_loading`.
        """
        return self._process_model_after_weight_loading(model, **kwargs)

    def dequantize(self, model):
        """
        Potentially dequantize the model to retrieve the original model, with some loss in accuracy / performance.
        Note not all quantization schemes support this.
        """
        model = self._dequantize(model)

        # Delete quantizer and quantization config
        del model.hf_quantizer
        del model.config.quantization_config
        del model.config._pre_quantization_dtype
        del model.quantization_method
        model.is_quantized = False

        return model

    def get_cuda_warm_up_factor(self):
        """
        The factor to be used in `caching_allocator_warmup` to get the number of bytes to pre-allocate to warm up cuda.
        A factor of 2 means we allocate all bytes in the empty model (since we allocate in fp16), a factor of 4 means
        we allocate half the memory of the weights residing in the empty model, etc...
        """
        # By default we return 4, i.e. half the model size (this corresponds to the case where the model is not
        # really pre-processed, i.e. we do not have the info that weights are going to be 8 bits before actual
        # weight loading)
        return 4

    def _dequantize(self, model):
        raise NotImplementedError(
            f"{self.quantization_config.quant_method} has no implementation of `dequantize`, please raise an issue on GitHub."
        )

    @staticmethod
    def get_modules_to_not_convert(
        model: "PreTrainedModel",
        skip_modules: Optional[list[str]] = None,
        keep_in_fp32_modules: Optional[list[str]] = None,
        add_default_skips: bool = False,
    ):
        from ..integrations import get_keys_to_not_convert

        if skip_modules is None or add_default_skips:
            modules_to_not_convert = get_keys_to_not_convert(model)
        else:
            modules_to_not_convert = []

        if skip_modules is not None:
            modules_to_not_convert.extend(skip_modules)

        if keep_in_fp32_modules is not None:
            modules_to_not_convert.extend(keep_in_fp32_modules)

        return modules_to_not_convert

    @property
    def is_qat_trainable(self) -> bool:
        """Flag indicating whether the quantized model can carry out quantization aware training"""
        return False

    @property
    def is_compileable(self) -> bool:
        """Flag indicating whether the quantized model can be compiled"""
        return False

    @abstractmethod
    def _process_model_before_weight_loading(self, model, **kwargs): ...

    @abstractmethod
    def _process_model_after_weight_loading(self, model, **kwargs): ...

    @abstractmethod
    def is_serializable(self, safe_serialization=None): ...

    @property
    @abstractmethod
    def is_trainable(self): ...

    def _convert_model_for_quantization(self, model):
        from accelerate import init_empty_weights

        for name, module in model.named_modules():
            module_class_name = module.__class__.__name__
            if module_class_name in MODULES_TO_PATCH_FOR_QUANTIZATION.keys() and (
                self.quantization_config.quant_method
                in MODULES_TO_PATCH_FOR_QUANTIZATION[module_class_name]["quantization_methods"]
            ):
                with init_empty_weights():
                    parent_module, name = get_module_from_name(model, name)
                    parent_module._modules[name] = MODULES_TO_PATCH_FOR_QUANTIZATION[module_class_name]["module_name"](
                        model.config.get_text_config()
                    )


class SequentialLlama4TextExperts(ModuleList):
    """
    A module that implements a compressed version of a list of expert modules.
    This is specifically designed to work with Llama4TextExperts in MoE layers.
    """

    def __init__(self, config):
        from transformers.models.llama4.modeling_llama4 import Llama4TextMLP

        super().__init__([Llama4TextMLP(config) for _ in range(config.num_local_experts)])
        self.num_experts = config.num_local_experts

    def forward(
        self,
        hidden_states: "torch.Tensor",
    ) -> "torch.Tensor":
        hidden_states = hidden_states.reshape(self.num_experts, -1, hidden_states.shape[-1])
        routed_out = torch.zeros_like(hidden_states)
        for expert_idx in range(self.num_experts):
            routed_out[expert_idx] = self[expert_idx](hidden_states[expert_idx])
        return routed_out


MODULES_TO_PATCH_FOR_QUANTIZATION = {
    "Llama4TextExperts": {
        "module_name": SequentialLlama4TextExperts,
        "quantization_methods": [
            QuantizationMethod.COMPRESSED_TENSORS,
            QuantizationMethod.BITS_AND_BYTES,
        ],
    }
}
