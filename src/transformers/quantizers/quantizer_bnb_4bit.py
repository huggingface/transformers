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

from ..utils import (
    ACCELERATE_MIN_VERSION,
    BITSANDBYTES_MIN_VERSION,
    is_accelerate_available,
    is_bitsandbytes_available,
    is_torch_available,
    is_torch_hpu_available,
    is_torch_npu_available,
    is_torch_xpu_available,
    logging,
)


if is_torch_available():
    import torch

    from ..core_model_loading import WeightConverter

logger = logging.get_logger(__name__)


class Bnb4BitHfQuantizer(HfQuantizer):
    """
    4-bit quantization from bitsandbytes quantization method
    """

    requires_calibration = False

    def __init__(self, quantization_config, **kwargs):
        super().__init__(quantization_config, **kwargs)

        # This describes the additional items that are saved on the state dict (on the params themselves)
        self.bnb_keys = [
            f"quant_state.bitsandbytes__{self.quantization_config.bnb_4bit_quant_type}",
            "absmax",
            "quant_map",
        ]
        if self.quantization_config.bnb_4bit_use_double_quant:
            self.bnb_keys.extend(["nested_absmax", "nested_quant_map"])

    def validate_environment(self, *args, **kwargs):
        if not is_accelerate_available():
            raise ImportError(
                f"Using `bitsandbytes` 4-bit quantization requires accelerate: `pip install 'accelerate>={ACCELERATE_MIN_VERSION}'`"
            )
        if not is_bitsandbytes_available():
            raise ImportError(
                f"Using `bitsandbytes` 4-bit quantization requires bitsandbytes: `pip install -U bitsandbytes>={BITSANDBYTES_MIN_VERSION}`"
            )

        from ..integrations import validate_bnb_backend_availability

        validate_bnb_backend_availability(raise_exception=True)

        device_map = kwargs.get("device_map")
        if not self.quantization_config.llm_int8_enable_fp32_cpu_offload and isinstance(device_map, dict):
            values = set(device_map.values())
            if values != {"cpu"} and ("cpu" in values or "disk" in values):
                raise ValueError(
                    "Some modules are dispatched on the CPU or the disk. Make sure you have enough GPU RAM to fit the "
                    "quantized model. If you want to dispatch the model on the CPU or the disk while keeping these modules "
                    "in 32-bit, you need to set `llm_int8_enable_fp32_cpu_offload=True` and pass a custom `device_map` to "
                    "`from_pretrained`. Check "
                    "https://huggingface.co/docs/transformers/main/en/main_classes/quantization#offload-between-cpu-and-gpu "
                    "for more details. "
                )

    def adjust_target_dtype(self, target_dtype: "torch.dtype") -> "torch.dtype":
        from accelerate.utils import CustomDtype

        if target_dtype != torch.int8:
            logger.info("target_dtype {target_dtype} is replaced by `CustomDtype.INT4` for 4-bit BnB quantization")
        return CustomDtype.INT4

    def param_needs_quantization(self, model: "PreTrainedModel", param_name: str, **kwargs) -> bool:
        import bitsandbytes as bnb

        # TODO: maybe remove
        # # They are on the params themselves, so we cannot easily extract the module from the name
        if any(param_name.endswith(x) for x in self.bnb_keys):
            return True
        module, name = get_module_from_name(model, param_name)
        return isinstance(module, bnb.nn.Linear4bit) and name != "bias"

    def get_param_name(self, param_name: str) -> str:
        """
        Get the right param_name in order to get the module associated with the param.
        This is useful for quantized stats lile absmax or quant_map as we need to update the param_name to get the module as they are stored in ...weight.absmax.
        """
        if self.pre_quantized:
            # We need to get the param name of quantized weights and not its components. Otherwise, we won't be able to get the nn.Module associated.
            if any(param_name.endswith(x) for x in self.bnb_keys):
                param_name = (
                    param_name.rsplit(".", 1)[0] if "quant_state." not in param_name else param_name.rsplit(".", 2)[0]
                )
        return param_name

    def adjust_max_memory(self, max_memory: dict[str, int | str]) -> dict[str, int | str]:
        # need more space for buffers that are created during quantization
        max_memory = {key: val * 0.90 for key, val in max_memory.items()}
        return max_memory

    def update_dtype(self, dtype: "torch.dtype") -> "torch.dtype":
        # TODO: remove ? is it still true ? we will move to dtype = "auto" so it will likely be either fp16 or bf16
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
            elif is_torch_npu_available():
                device_map = {"": f"npu:{torch.npu.current_device()}"}
            elif is_torch_hpu_available():
                device_map = {"": f"hpu:{torch.hpu.current_device()}"}
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

    def _process_model_before_weight_loading(
        self,
        model: "PreTrainedModel",
        device_map,
        keep_in_fp32_modules: list[str] | None = None,
        **kwargs,
    ):
        from ..integrations import replace_with_bnb_linear

        self.modules_to_not_convert = self.get_modules_to_not_convert(
            model, self.quantization_config.llm_int8_skip_modules, keep_in_fp32_modules
        )

        if self.quantization_config.llm_int8_enable_fp32_cpu_offload:
            if isinstance(device_map, dict):
                keys_on_cpu = [key for key, value in device_map.items() if value in ["disk", "cpu"]]
                self.modules_to_not_convert.extend(keys_on_cpu)

        model = replace_with_bnb_linear(
            model,
            modules_to_not_convert=self.modules_to_not_convert,
            quantization_config=self.quantization_config,
            pre_quantized=self.pre_quantized,
        )

    def _process_model_after_weight_loading(self, model: "PreTrainedModel", **kwargs):
        model.is_loaded_in_4bit = True
        model.is_4bit_serializable = self.is_serializable()
        return model

    def is_serializable(self, **kwargs):
        return True

    @property
    def is_trainable(self) -> bool:
        return True

    def _dequantize(self, model):
        from ..integrations import dequantize_and_replace

        model = dequantize_and_replace(model, quantization_config=self.quantization_config)
        return model

    def get_quantize_ops(self):
        from ..integrations.bitsandbytes import Bnb4bitQuantize

        return Bnb4bitQuantize(self)

    def get_weight_conversions(self):
        from ..integrations.bitsandbytes import Bnb4bitDeserialize

        if self.pre_quantized:
            return [
                WeightConverter(
                    source_patterns=[
                        "weight.nested_absmax",
                        "weight.nested_quant_map",
                        "weight.quant_map",
                        "weight.absmax",
                        "weight.quant_state.bitsandbytes__nf4",
                        "weight.quant_state.bitsandbytes__fp4",
                        "weight",
                    ],
                    target_patterns="weight",
                    operations=[Bnb4bitDeserialize(self)],
                )
            ]
        return []
