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
from typing import TYPE_CHECKING, Dict, List, Optional, Union

from .base import HfQuantizer


if TYPE_CHECKING:
    from ..modeling_utils import PreTrainedModel

from ..utils import is_accelerate_available, is_torch_available, logging


if is_torch_available():
    import torch


logger = logging.get_logger(__name__)


class BitNetHfQuantizer(HfQuantizer):
    """
    1.58-bit quantization from BitNet quantization method:
    Before loading: it converts the linear layers into BitLinear layers during loading.

    Checkout the paper introducing this method : https://arxiv.org/pdf/2402.17764
    """

    requires_parameters_quantization = False
    requires_calibration = True

    required_packages = ["accelerate"]

    def __init__(self, quantization_config, **kwargs):
        super().__init__(quantization_config, **kwargs)
        self.quantization_config = quantization_config

    def validate_environment(self, *args, **kwargs):
        if not is_accelerate_available():
            raise ImportError("Loading a BitNet quantized model requires accelerate (`pip install accelerate`)")

        if kwargs.get("from_tf", False) or kwargs.get("from_flax", False):
            raise ValueError(
                "Loading ternary weights from tf/flax is currently not supported, please make"
                " sure the weights are in PyTorch format."
            )

        if not torch.cuda.is_available():
            logger.warning_once(
                "You don't have a GPU available to load the model, the inference will be slow because of weight unpacking"
            )
            return

        device_map = kwargs.get("device_map", None)
        if device_map is None:
            logger.warning_once(
                "You have loaded a BitNet model on CPU and have a CUDA device available, make sure to set "
                "your model on a GPU device in order to run your model."
            )
        elif device_map is not None:
            if isinstance(device_map, dict) and ("cpu" in device_map.values() or "disk" in device_map.values()):
                raise ValueError(
                    "You are attempting to load a BitNet model with a device_map that contains a CPU or disk device."
                    "This is not supported. Please remove the CPU or disk device from the device_map."
                )

    def _process_model_after_weight_loading(self, model: "PreTrainedModel", **kwargs):
        return model

    def _process_model_before_weight_loading(
        self,
        model: "PreTrainedModel",
        keep_in_fp32_modules: Optional[List[str]] = None,
        **kwargs,
    ):
        from ..integrations import replace_with_bitnet_linear

        self.modules_to_not_convert = self.get_modules_to_not_convert(
            model, self.quantization_config.modules_to_not_convert, keep_in_fp32_modules
        )

        model = replace_with_bitnet_linear(
            model,
            modules_to_not_convert=self.modules_to_not_convert,
            quantization_config=self.quantization_config,
            pre_quantized=self.pre_quantized,
        )

    def adjust_max_memory(self, max_memory: Dict[str, Union[int, str]]) -> Dict[str, Union[int, str]]:
        max_memory = {key: val * 0.90 for key, val in max_memory.items()}
        return max_memory

    def adjust_target_dtype(self, target_dtype: "torch.dtype") -> "torch.dtype":
        target_dtype = torch.int8
        return target_dtype

    def is_serializable(self, safe_serialization=None):
        return True

    @property
    def is_trainable(self) -> bool:
        return (
            self.quantization_config.linear_class == "autobitlinear"
            and self.quantization_config.quantization_mode == "online"
        )

    @property
    def is_qat_trainable(self) -> bool:
        """Flag indicating whether the quantized model can carry out quantization aware training"""
        return (
            self.quantization_config.linear_class == "autobitlinear"
            and self.quantization_config.quantization_mode == "online"
        )
