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
from typing import TYPE_CHECKING, Optional, Union

from packaging import version

from .base import HfQuantizer


if TYPE_CHECKING:
    from ..modeling_utils import PreTrainedModel

from ..utils import (
    ACCELERATE_MIN_VERSION,
    is_accelerate_available,
    is_bitsandbytes_available,
    is_torch_available,
    is_torch_xpu_available,
    logging,
)
from .quantizers_utils import get_module_from_name


if is_torch_available():
    import torch

    from ..pytorch_utils import Conv1D

logger = logging.get_logger(__name__)


class Bnb8BitHfQuantizer(HfQuantizer):
    """
    8-bit quantization from bitsandbytes quantization method:
        before loading: converts transformer layers into Linear8bitLt during loading: load 16bit weight and pass to the
        layer object after: quantizes individual weights in Linear8bitLt into 8bit at fitst .cuda() call
    saving:
        from state dict, as usual; saves weights and 'SCB' component
    loading:
        need to locate SCB component and pass to the Linear8bitLt object
    """

    use_keep_in_fp32_modules = True
    requires_parameters_quantization = True
    requires_calibration = False

    required_packages = ["bitsandbytes", "accelerate"]

    def __init__(self, quantization_config, **kwargs):
        super().__init__(quantization_config, **kwargs)

        if self.quantization_config.llm_int8_skip_modules is not None:
            self.modules_to_not_convert = self.quantization_config.llm_int8_skip_modules

    def validate_environment(self, *args, **kwargs):
        if not is_accelerate_available():
            raise ImportError(
                f"Using `bitsandbytes` 8-bit quantization requires Accelerate: `pip install 'accelerate>={ACCELERATE_MIN_VERSION}'`"
            )
        if not is_bitsandbytes_available(check_library_only=True):
            raise ImportError(
                "Using `bitsandbytes` 8-bit quantization requires the latest version of bitsandbytes: `pip install -U bitsandbytes`"
            )
        if not is_torch_available():
            raise ImportError(
                "The bitsandbytes library requires PyTorch but it was not found in your environment. "
                "You can install it with `pip install torch`."
            )
        # `bitsandbytes` versions older than 0.43.1 eagerly require CUDA at import time,
        # so those versions of the library are practically only available when CUDA is too.
        if version.parse(importlib.metadata.version("bitsandbytes")) < version.parse("0.43.1"):
            if not torch.cuda.is_available():
                raise ImportError(
                    "The installed version of bitsandbytes (<0.43.1) requires CUDA, but CUDA is not available. "
                    "You may need to install PyTorch with CUDA support or upgrade bitsandbytes to >=0.43.1."
                )

        from ..integrations import validate_bnb_backend_availability
        from ..utils import is_bitsandbytes_multi_backend_available

        bnb_multibackend_is_enabled = is_bitsandbytes_multi_backend_available()
        validate_bnb_backend_availability(raise_exception=True)

        if kwargs.get("from_tf", False) or kwargs.get("from_flax", False):
            raise ValueError(
                "Converting into 4-bit or 8-bit weights from tf/flax weights is currently not supported, please make"
                " sure the weights are in PyTorch format."
            )

        device_map = kwargs.get("device_map")
        if (
            device_map is not None
            and isinstance(device_map, dict)
            and not self.quantization_config.llm_int8_enable_fp32_cpu_offload
        ):
            device_map_without_lm_head = {
                key: device_map[key] for key in device_map if key not in self.modules_to_not_convert
            }
            if set(device_map.values()) == {"cpu"} and bnb_multibackend_is_enabled:
                pass
            elif "cpu" in device_map_without_lm_head.values() or "disk" in device_map_without_lm_head.values():
                raise ValueError(
                    "Some modules are dispatched on the CPU or the disk. Make sure you have enough GPU RAM to fit the "
                    "quantized model. If you want to dispatch the model on the CPU or the disk while keeping these modules "
                    "in 32-bit, you need to set `llm_int8_enable_fp32_cpu_offload=True` and pass a custom `device_map` to "
                    "`from_pretrained`. Check "
                    "https://huggingface.co/docs/transformers/main/en/main_classes/quantization#offload-between-cpu-and-gpu "
                    "for more details. "
                )

        if version.parse(importlib.metadata.version("bitsandbytes")) < version.parse("0.37.2"):
            raise ValueError(
                "You have a version of `bitsandbytes` that is not compatible with 8bit inference and training"
                " make sure you have the latest version of `bitsandbytes` installed"
            )

    def adjust_max_memory(self, max_memory: dict[str, Union[int, str]]) -> dict[str, Union[int, str]]:
        # need more space for buffers that are created during quantization
        max_memory = {key: val * 0.90 for key, val in max_memory.items()}
        return max_memory

    def update_dtype(self, dtype: "torch.dtype") -> "torch.dtype":
        if dtype is None:
            # We force the `dtype` to be float16, this is a requirement from `bitsandbytes`
            logger.info(
                "Overriding dtype=%s with `dtype=torch.float16` due to "
                "requirements of `bitsandbytes` to enable model loading in 8-bit or 4-bit. "
                "Pass your own dtype to specify the dtype of the remaining non-linear layers or pass"
                " dtype=torch.float16 to remove this warning.",
                dtype,
            )
            dtype = torch.float16
        return dtype

    def update_device_map(self, device_map):
        if device_map is None:
            if torch.cuda.is_available():
                device_map = {"": torch.cuda.current_device()}
            elif is_torch_xpu_available():
                device_map = {"": torch.xpu.current_device()}
            else:
                device_map = {"": "cpu"}
            logger.info(
                "The device_map was not initialized. "
                f"Setting device_map to {device_map}. "
                "If you want to use the model for inference, please set device_map ='auto' "
            )
        return device_map

    def adjust_target_dtype(self, target_dtype: "torch.dtype") -> "torch.dtype":
        if target_dtype != torch.int8:
            logger.info("target_dtype {target_dtype} is replaced by `torch.int8` for 8-bit BnB quantization")
        return torch.int8

    def update_unexpected_keys(self, model, unexpected_keys: list[str]) -> list[str]:
        bnb_keys = ["SCB", "weight_format"]
        return [k for k in unexpected_keys if not any(k.endswith(x) for x in bnb_keys)]

    def param_needs_quantization(self, model: "PreTrainedModel", param_name: str, **kwargs) -> bool:
        import bitsandbytes as bnb

        module, name = get_module_from_name(model, param_name)
        return isinstance(module, bnb.nn.Linear8bitLt) and name != "bias"

    def create_quantized_param(
        self,
        model: "PreTrainedModel",
        param_value: "torch.Tensor",
        param_name: str,
        target_device: "torch.device",
        **kwargs,
    ):
        import bitsandbytes as bnb

        module, tensor_name = get_module_from_name(model, param_name)

        if self.pre_quantized and not self.is_serializable():
            raise ValueError(
                "Detected int8 weights but the version of bitsandbytes is not compatible with int8 serialization. "
                "Make sure to download the latest `bitsandbytes` version. `pip install --upgrade bitsandbytes`."
            )
        # Those 2 can only happen when self.pre_quantized == True
        if tensor_name == "SCB":
            setattr(module.weight, "SCB", param_value.to(target_device))
            return
        # It's not used, but it's getting serialized for BC reason...
        elif tensor_name == "weight_format":
            return

        # Support models using `Conv1D` in place of `nn.Linear` (e.g. openai-community/gpt2) by transposing the weight matrix prior to quantization.
        # Since weights are saved in the correct "orientation", we skip transposing when loading.
        if issubclass(module.source_cls, Conv1D) and not self.pre_quantized:
            param_value = param_value.T

        old_value = getattr(module, tensor_name)
        kwargs = old_value.__dict__
        kwargs.pop("_is_hf_initialized", None)
        # Need to pop SCB and reset it because of bnb internals that modifies its value when switching devices ...
        SCB = kwargs.pop("SCB", None)
        new_value = bnb.nn.Int8Params(param_value.to("cpu"), requires_grad=False, **kwargs).to(target_device)
        if SCB is not None:
            setattr(new_value, "SCB", SCB)
        # Set it to the module
        module._parameters[tensor_name] = new_value

    def _process_model_after_weight_loading(self, model: "PreTrainedModel", **kwargs):
        model.is_loaded_in_8bit = True
        model.is_8bit_serializable = self.is_serializable()
        return model

    def _process_model_before_weight_loading(
        self,
        model: "PreTrainedModel",
        device_map,
        keep_in_fp32_modules: Optional[list[str]] = None,
        **kwargs,
    ):
        from ..integrations import replace_with_bnb_linear

        llm_int8_enable_fp32_cpu_offload = self.quantization_config.llm_int8_enable_fp32_cpu_offload

        self.modules_to_not_convert = self.get_modules_to_not_convert(
            model, self.quantization_config.llm_int8_skip_modules, keep_in_fp32_modules
        )

        # Extend `self.modules_to_not_convert` to keys that are supposed to be offloaded to `cpu` or `disk`
        if isinstance(device_map, dict) and len(device_map.keys()) > 1:
            keys_on_cpu = [key for key, value in device_map.items() if value in ["disk", "cpu"]]

            if len(keys_on_cpu) > 0 and not llm_int8_enable_fp32_cpu_offload:
                raise ValueError(
                    "If you want to offload some keys to `cpu` or `disk`, you need to set "
                    "`llm_int8_enable_fp32_cpu_offload=True`. Note that these modules will not be "
                    " converted to 8-bit but kept in 32-bit."
                )
            self.modules_to_not_convert.extend(keys_on_cpu)

        model = replace_with_bnb_linear(
            model, modules_to_not_convert=self.modules_to_not_convert, quantization_config=self.quantization_config
        )

        model.config.quantization_config = self.quantization_config

    def is_serializable(self, safe_serialization=None):
        _bnb_supports_8bit_serialization = version.parse(importlib.metadata.version("bitsandbytes")) > version.parse(
            "0.37.2"
        )

        if not _bnb_supports_8bit_serialization:
            logger.warning(
                "You are calling `save_pretrained` to a 8-bit converted model, but your `bitsandbytes` version doesn't support it. "
                "If you want to save 8-bit models, make sure to have `bitsandbytes>0.37.2` installed. You will most likely face errors or"
                " unexpected behaviours."
            )
            return False

        return True

    @property
    def is_trainable(self) -> bool:
        return version.parse(importlib.metadata.version("bitsandbytes")) >= version.parse("0.37.0")

    def _dequantize(self, model):
        from ..integrations import dequantize_and_replace

        model = dequantize_and_replace(
            model, self.modules_to_not_convert, quantization_config=self.quantization_config
        )
        return model
