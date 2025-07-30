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
from typing import TYPE_CHECKING, Any, Optional

from .base import HfQuantizer
from .quantizers_utils import get_module_from_name


if TYPE_CHECKING:
    from ..modeling_utils import PreTrainedModel

from ..utils import is_fp_quant_available, is_qutlass_available, is_torch_available, logging
from ..utils.quantization_config import QuantizationConfigMixin


if is_torch_available():
    import torch

logger = logging.get_logger(__name__)


class FPQuantHfQuantizer(HfQuantizer):
    """
    Quantizer for the FP-Quant method. Enables the loading of prequantized models and in-flight quantization of full-precision models.
    """

    requires_calibration = False
    requires_parameters_quantization = True
    is_qat_trainable = False
    required_packages = ["fp_quant"]

    def __init__(self, quantization_config: QuantizationConfigMixin, **kwargs):
        super().__init__(quantization_config, **kwargs)
        self.quantization_config = quantization_config

    def validate_environment(self, device_map, **kwargs):
        if not torch.cuda.is_available():
            raise NotImplementedError(
                "FPQuant quantization is only supported on GPU. Please use a different quantizer."
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

        if device_map is None:
            raise ValueError(
                "You are attempting to load a FPQuant model without setting device_map."
                " Please set device_map comprised of 'cuda' devices."
            )
        elif isinstance(device_map, dict) and ("cpu" in device_map.values() or "disk" in device_map.values()):
            raise ValueError(
                "You are attempting to load a FPQuant model with a device_map that contains a CPU or disk device."
                " This is not supported. Please remove the CPU or disk device from the device_map."
            )

    def update_dtype(self, dtype: "torch.dtype") -> "torch.dtype":
        if dtype is None:
            logger.info("`dtype` is None. Setting `dtype=torch.bfloat16` for qutlass compatibility.")
            dtype = torch.bfloat16
        elif dtype != torch.bfloat16:
            raise ValueError(
                f"Invalid `dtype` {dtype}. fp_quant quantization only supports `dtype=torch.bfloat16`."
            )

        return dtype

    def create_quantized_param(
        self,
        model: "PreTrainedModel",
        param_value: "torch.Tensor",
        param_name: str,
        target_device: "torch.device",
        state_dict: dict[str, Any],
        unexpected_keys: Optional[list[str]] = None,
    ):
        module, _ = get_module_from_name(model, param_name)

        # The module holds either:
        #  * `weight` when `store_master_weights=True`
        #  * `qweight` and `scales` when `store_master_weights=False` and `pseudoquantization=False`
        #  * `dqweight` when `store_master_weights=False` and `pseudoquantization=True`

        if param_name.endswith(".qweight"):
            # Loading a real quantized checkpoint without master weights
            module.qweight = torch.nn.Parameter(
                param_value.to(target_device),
                requires_grad=False,
            )
            module.weight = None
            module.dqweight = None
            return

        if param_name.endswith(".dqweight"):
            # Loading a pseudo-quantized checkpoint without master weights
            module.dqweight = torch.nn.Parameter(param_value.to(target_device))
            module.weight = None
            module.qweight = None
            module.scales = None
            return

        # Loading master weights or an unquantized checkpoint
        module.weight = torch.nn.Parameter(param_value.to(target_device))
        # Let pre-forward handle the quantization and set None where necessary
        module.pre_forward()

        if unexpected_keys is not None and param_name in unexpected_keys:
            unexpected_keys.remove(param_name)

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
        model.config.quantization_config = self.quantization_config

    def _process_model_after_weight_loading(self, model: "PreTrainedModel", **kwargs):
        return model

    def update_missing_keys(self, model, missing_keys: list[str], prefix: str) -> list[str]:
        from fp_quant import FPQuantLinear

        fp_quant_names = {name for name, module in model.named_modules() if isinstance(module, FPQuantLinear)}

        def should_exclude(key: str) -> bool:
            if key.endswith(".weight") or key.endswith(".bias"):
                return False
            full_key = f"{prefix}.{key}"
            return any(name in key or name in full_key for name in fp_quant_names)

        return [key for key in missing_keys if not should_exclude(key)]

    @property
    def is_trainable(self, model: Optional["PreTrainedModel"] = None):
        return False

    def is_serializable(self, safe_serialization=None):
        return True

    def check_quantized_param(
        self,
        model: "PreTrainedModel",
        param_value: "torch.Tensor",
        param_name: str,
        state_dict: dict[str, Any],
        **kwargs,
    ) -> bool:
        from fp_quant import FPQuantLinear

        module, tensor_name = get_module_from_name(model, param_name)
        if isinstance(module, FPQuantLinear) and tensor_name in ["weight", "qweight", "dqweight"]:
            # Only quantize weights of FPQuantLinear modules that are not already quantized
            return True
        else:
            return False
