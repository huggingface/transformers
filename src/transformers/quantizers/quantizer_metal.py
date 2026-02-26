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
from typing import TYPE_CHECKING, Any

from ..utils import is_kernels_available, is_torch_available, logging
from .base import HfQuantizer
from .quantizers_utils import get_module_from_name


if is_torch_available():
    import torch

if TYPE_CHECKING:
    from ..modeling_utils import PreTrainedModel

logger = logging.get_logger(__name__)


class MetalHfQuantizer(HfQuantizer):
    """
    Quantizer for Metal affine quantization on Apple Silicon (MPS) devices.

    Uses the ``quantization-mlx`` Metal kernels from the Hub to pack weights into
    low-bit (2/4/8) uint32 tensors with per-group scales and biases, and performs
    fused dequant + matmul in the forward pass.
    """

    requires_calibration = False

    def __init__(self, quantization_config, **kwargs):
        super().__init__(quantization_config, **kwargs)

    def validate_environment(self, *args, **kwargs):
        if self.quantization_config.dequantize:
            return

        if not torch.backends.mps.is_available():
            if self.pre_quantized:
                logger.warning_once(
                    "Metal quantization requires an Apple Silicon GPU (MPS), but none is available. "
                    "We will default to dequantizing the model to the original dtype."
                )
                self.quantization_config.dequantize = True
                return
            else:
                raise RuntimeError("Metal quantization requires an Apple Silicon GPU (MPS). No MPS device found.")

        if not is_kernels_available():
            raise ImportError("Metal quantization requires kernels: `pip install kernels`")

        device_map = kwargs.get("device_map")
        if device_map is None:
            logger.warning_once(
                "You have loaded a Metal quantized model on CPU and have an MPS device available. "
                "Set device_map='mps' to use the Metal kernels."
            )
        elif isinstance(device_map, dict):
            if not self.pre_quantized and ("cpu" in device_map.values() or "disk" in device_map.values()):
                raise ValueError(
                    "Metal quantization on the fly does not support CPU or disk in the device_map. "
                    "Please use a pre-quantized checkpoint or remove CPU/disk from device_map."
                )

    def update_device_map(self, device_map: dict[str, Any] | None) -> dict[str, Any] | None:
        if device_map is None:
            device_map = {"": "mps"}
        return device_map

    def param_needs_quantization(self, model: "PreTrainedModel", param_name: str, **kwargs) -> bool:
        from ..integrations.metal_quantization import MetalLinear

        module, tensor_name = get_module_from_name(model, param_name)
        if isinstance(module, MetalLinear):
            if self.pre_quantized or tensor_name != "weight":
                return False
            return True
        return False

    def _process_model_before_weight_loading(self, model: "PreTrainedModel", **kwargs):
        from ..integrations.metal_quantization import replace_with_metal_linear

        self.modules_to_not_convert = self.get_modules_to_not_convert(
            model, self.quantization_config.modules_to_not_convert, model._keep_in_fp32_modules
        )

        model = replace_with_metal_linear(
            model,
            modules_to_not_convert=self.modules_to_not_convert,
            quantization_config=self.quantization_config,
            pre_quantized=self.pre_quantized,
        )

    def is_serializable(self):
        return True

    @property
    def is_trainable(self) -> bool:
        return False

    def get_quantize_ops(self):
        from ..integrations.metal_quantization import MetalQuantize

        return MetalQuantize(self)

    def get_weight_conversions(self):
        from ..core_model_loading import WeightConverter
        from ..integrations.metal_quantization import MetalDequantize

        if self.pre_quantized and self.quantization_config.dequantize:
            return [
                WeightConverter(
                    source_patterns=["weight$", "scales", "qbiases"],
                    target_patterns="weight",
                    operations=[MetalDequantize(self)],
                )
            ]
        return []
