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
from typing import TYPE_CHECKING, Optional

from .base import HfQuantizer
from .quantizers_utils import get_module_from_name


if TYPE_CHECKING:
    from ..modeling_utils import PreTrainedModel

from ..utils import is_fp_quant_available, is_qutlass_available, is_torch_available, is_torch_xpu_available, logging
from ..utils.quantization_config import QuantizationConfigMixin


if is_torch_available():
    import torch

logger = logging.get_logger(__name__)


class FPQuantHfQuantizer(HfQuantizer):
    """
    Quantizer for the FP-Quant method. Enables the loading of prequantized models and in-flight quantization of full-precision models.
    """

    requires_calibration = False
    is_qat_trainable = True

    def __init__(self, quantization_config: QuantizationConfigMixin, **kwargs):
        super().__init__(quantization_config, **kwargs)

    def validate_environment(self, device_map, **kwargs):
        if not torch.cuda.is_available() and not is_torch_xpu_available():
            raise NotImplementedError(
                "FPQuant quantization is only supported on GPU or Intel XPU. Please use a different quantizer."
            )

        if not is_qutlass_available() and not self.quantization_config.pseudoquantization:
            raise ImportError(
                "Using `fp_quant` with real quantization requires a **Blackwell GPU** and qutlass: `git clone https://github.com/IST-DASLab/qutlass.git && cd qutlass && pip install --no-build-isolation .`. You can use `FPQuantConfig(pseudoquantization=True, ...)` to use Triton-based pseudo-quantization. It doesn't provide any speedups but emulates the quantization behavior of the real quantization."
            )

        if self.quantization_config.pseudoquantization:
            logger.warning(
                "Using pseudo-quantization for FP-Quant. This doesn't provide any speedups but emulates the quantization behavior of the real quantization."
            )

        if not is_fp_quant_available():
            raise ImportError("Using `fp_quant` quantization requires fp_quant: `pip install fp_quant`")

        if device_map is None and not self.quantization_config.pseudoquantization:
            raise ValueError(
                "You are attempting to load a FPQuant model without setting device_map."
                " Please set device_map comprised of 'cuda' devices."
            )
        elif isinstance(device_map, dict):
            if (
                not self.quantization_config.pseudoquantization
                and len(device_map) > 1
                and "cpu" in device_map.values()
                or "disk" in device_map.values()
            ):
                raise ValueError(
                    "You are attempting to load a FPQuant model with a device_map that contains a CPU or disk device."
                    " This is not supported. Please remove the CPU or disk device from the device_map."
                )

    def update_dtype(self, dtype: "torch.dtype") -> "torch.dtype":
        if dtype is None:
            logger.info("`dtype` is None. Setting `dtype=torch.bfloat16` for qutlass compatibility.")
            dtype = torch.bfloat16
        elif dtype != torch.bfloat16:
            raise ValueError(f"Invalid `dtype` {dtype}. fp_quant quantization only supports `dtype=torch.bfloat16`.")
        return dtype

    def param_needs_quantization(self, model: "PreTrainedModel", param_name: str, **kwargs) -> bool:
        from fp_quant import FPQuantLinear

        module, tensor_name = get_module_from_name(model, param_name)
        if isinstance(module, FPQuantLinear) and tensor_name in ["weight", "qweight", "dqweight"]:
            # Only quantize weights of FPQuantLinear modules that are not already quantized
            return True
        else:
            return False

    def _process_model_before_weight_loading(
        self,
        model: "PreTrainedModel",
        **kwargs,
    ):
        from fp_quant import replace_with_fp_quant_linear

        from ..integrations.fp_quant import adapt_fp_quant_config

        replace_with_fp_quant_linear(
            model,
            fp_quant_linear_config=adapt_fp_quant_config(self.quantization_config),
        )

    @property
    def is_trainable(self, model: Optional["PreTrainedModel"] = None):
        trainable = self.quantization_config.store_master_weights
        if not trainable:
            logger.warning(
                "You are attempting to train a model with FPQuant quantization. This is only supported when `store_master_weights=True`. Please set `store_master_weights=True` to train the model."
            )
        return trainable

    def is_serializable(self):
        return True
