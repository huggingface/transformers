# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from packaging import version

from .base import HfQuantizer


if TYPE_CHECKING:
    from ..modeling_utils import PreTrainedModel

from ..utils import is_torch_available, is_torchao_available, logging
from .quantizers_utils import get_module_from_name


if is_torch_available():
    import torch


logger = logging.get_logger(__name__)


class SpinquantHfQuantizer(HfQuantizer):
    """
    Spinquant quantization
    """

    requires_parameters_quantization = False
    requires_calibration = True

    required_packages = ["accelerate"]

    def __init__(self, quantization_config, **kwargs):
        super().__init__(quantization_config, **kwargs)
        self.quantization_config = quantization_config

    def validate_environment(self, *args, **kwargs):
        if not is_torchao_available():
            raise ImportError(
                "Using spinquant quantization requires torchao library"
            )

        if not torch.cuda.is_available():
            raise RuntimeError("Using spinquant quantized models requires a GPU")

        # try:
        #     from fast_hadamard_transform import hadamard_transform
        # except ImportError:
        #     raise ImportError(
        #     "Please install fast-hadamard-transform: pip install fast-hadamard-transform"
        # )

        device_map = kwargs.get("device_map", None)
        if device_map is not None:
            if (
                not self.pre_quantized
                and isinstance(device_map, dict)
                and ("cpu" in device_map.values() or "disk" in device_map.values())
            ):
                raise ValueError(
                    "You are attempting to load an quantized model with a device_map that contains a CPU or disk device."
                    "This is not supported when the model is quantized on the fly. "
                    "Please use a quantized checkpoint or remove the CPU or disk device from the device_map."
                )
        if not self.pre_quantized:
            raise ValueError("The model needs to be prequantized with Spinquant quantization method.")

    def _process_model_before_weight_loading(
        self,
        model: "PreTrainedModel",
        device_map,
        keep_in_fp32_modules: List[str] = [],
        **kwargs,
    ):
        from ..integrations import replace_with_spinquant_linear

        if self.quantization_config.modules_to_not_convert is not None:
            self.modules_to_not_convert.extend(self.quantization_config.modules_to_not_convert)

        model = replace_with_spinquant_linear(
            model,
            modules_to_not_convert=self.modules_to_not_convert,
            quantization_config=self.quantization_config,
            pre_quantized=self.pre_quantized,
        )

        model.config.quantization_config = self.quantization_config

    def _process_model_after_weight_loading(self, model: "PreTrainedModel", **kwargs):
        return model

    def is_serializable(self, safe_serialization=None):
        return True

    @property
    def is_trainable(self) -> bool:
        return False
