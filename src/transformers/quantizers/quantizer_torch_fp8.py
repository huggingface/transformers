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


if TYPE_CHECKING:
    from ..modeling_utils import PreTrainedModel

from ..utils import is_torch_available, logging, is_torch_fp8_available


if is_torch_available():
    import torch

logger = logging.get_logger(__name__)


class TorchFp8Quantizer(HfQuantizer):
    """
    FP8 Quantization using torch-fp8 library: https://github.com/pytorch-labs/float8_experimental
    """
    requires_calibration = False

    required_packages = ["float8_experimental"]

    def __init__(self, quantization_config, **kwargs):
        super().__init__(quantization_config, **kwargs)

    def validate_environment(self, device_map, **kwargs):
        if not torch.cuda.is_available():
            raise RuntimeError("GPU is required to run AWQ quantized model.")

        if not is_torch_fp8_available():
            raise ImportError("Loading a FP8 quantized model requires float8_experimental library (`pip install float8_experimental`)")


        if device_map is None:
            logger.warning_once(
                "You have loaded an FP8 model on CPU and have a CUDA device available, make sure to set "
                "your model on a GPU device in order to run your model."
            )
        elif device_map is not None:
            if isinstance(device_map, dict) and ("cpu" in device_map.values() or "disk" in device_map.values()):
                raise ValueError(
                    "You are attempting to load a FP8 model with a device_map that contains a CPU or disk device."
                    " This is not supported. Please remove the CPU or disk device from the device_map."
                )

    def update_torch_dtype(self, torch_dtype):
        return torch_dtype

    def adjust_target_dtype(self, target_dtype: "torch.dtype") -> "torch.dtype":
        # FP8 has the same memory footprint as int8
        return torch.int8

    def _process_model_before_weight_loading(self, model: "PreTrainedModel", **kwargs):
        from float8_experimental.float8_linear_utils import (
            swap_linear_with_float8_linear,
        )
        from float8_experimental.float8_linear import Float8Linear
        from float8_experimental.float8_dynamic_linear import Float8DynamicLinear

        if self.quantization_config.modules_to_not_convert is not None:
            self.modules_to_not_convert.extend(self.quantization_config.modules_to_not_convert)

        target_cls = Float8DynamicLinear if self.quantization_config.linear_type == "dynamic" else Float8Linear

        swap_linear_with_float8_linear(model, target_cls, skip_fqn_list=self.modules_to_not_convert)

    def _process_model_after_weight_loading(self, model):
        return

    @property
    def is_serializable(self):
        return True

    @property
    def is_trainable(self):
        return True
