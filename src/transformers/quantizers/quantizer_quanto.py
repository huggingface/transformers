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
    is_accelerate_available,
    is_optimum_quanto_available,
    is_torch_available,
    logging,
)
from ..utils.quantization_config import QuantoConfig


if is_torch_available():
    import torch

logger = logging.get_logger(__name__)


class QuantoHfQuantizer(HfQuantizer):
    """
    Quantizer for the quanto library
    """

    requires_calibration = False

    def __init__(self, quantization_config: QuantoConfig, **kwargs):
        super().__init__(quantization_config, **kwargs)

    def validate_environment(self, *args, **kwargs):
        if not is_optimum_quanto_available():
            raise ImportError(
                "Loading an optimum-quanto quantized model requires optimum-quanto library (`pip install optimum-quanto`)"
            )
        if not is_accelerate_available():
            raise ImportError(
                "Loading an optimum-quanto quantized model requires accelerate library (`pip install accelerate`)"
            )
        device_map = kwargs.get("device_map")
        if isinstance(device_map, dict):
            if len(device_map) > 1 and "cpu" in device_map.values() or "disk" in device_map.values():
                raise ValueError(
                    "You are attempting to load an model with a device_map that contains a CPU or disk device."
                    "This is not supported with quanto when the model is quantized on the fly. "
                    "Please remove the CPU or disk device from the device_map."
                )
        if self.quantization_config.activations is not None:
            raise ValueError(
                "We don't support quantizing the activations with transformers library."
                "Use quanto library for more complex use cases such as activations quantization, calibration and quantization aware training."
            )

    def param_needs_quantization(self, model: "PreTrainedModel", param_name: str, **kwargs) -> bool:
        from optimum.quanto import QModuleMixin

        module, tensor_name = get_module_from_name(model, param_name)
        # We only quantize the weights and the bias is not quantized.
        if isinstance(module, QModuleMixin) and "weight" in tensor_name:
            # if the weights are quantized, don't need to recreate it again with `create_quantized_param`
            return not module.frozen
        else:
            return False

    def adjust_max_memory(self, max_memory: dict[str, int | str]) -> dict[str, int | str]:
        max_memory = {key: val * 0.90 for key, val in max_memory.items()}
        return max_memory

    def adjust_target_dtype(self, target_dtype: "torch.dtype") -> "torch.dtype":
        from accelerate.utils import CustomDtype

        mapping = {
            "int8": torch.int8,
            "float8": CustomDtype.FP8,
            "int4": CustomDtype.INT4,
            "int2": CustomDtype.INT2,
        }
        target_dtype = mapping[self.quantization_config.weights]
        return target_dtype

    def _process_model_before_weight_loading(
        self, model: "PreTrainedModel", keep_in_fp32_modules: list[str] | None = None, **kwargs
    ):
        from ..integrations import replace_with_quanto_layers

        self.modules_to_not_convert = self.get_modules_to_not_convert(
            model, self.quantization_config.modules_to_not_convert, keep_in_fp32_modules
        )

        model = replace_with_quanto_layers(
            model, modules_to_not_convert=self.modules_to_not_convert, quantization_config=self.quantization_config
        )

    @property
    def is_trainable(self) -> bool:
        return True

    def is_serializable(self):
        return False

    def get_quantize_ops(self):
        from ..integrations.quanto import QuantoQuantize

        return QuantoQuantize(self)
