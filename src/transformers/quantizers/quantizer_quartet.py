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
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from ..utils.logging import tqdm
from .base import HfQuantizer
from .quantizers_utils import get_module_from_name


if TYPE_CHECKING:
    from ..modeling_utils import PreTrainedModel

from ..utils import is_quartet_available, is_quartet_qat_available, is_qutlass_available, is_torch_available, logging
from ..utils.quantization_config import QuantizationConfigMixin


if is_torch_available():
    import torch

logger = logging.get_logger(__name__)


class QuartetHfQuantizer(HfQuantizer):
    """
    Quantizer of the HIGGS method. Enables the loading of prequantized models and in-flight quantization of full-precision models.
    """

    requires_calibration = False
    requires_parameters_quantization = True
    required_packages = ["qutlass", "quartet_qat", "quartet"]

    def __init__(self, quantization_config: QuantizationConfigMixin, **kwargs):
        super().__init__(quantization_config, **kwargs)
        self.quantization_config = quantization_config

    def validate_environment(self, device_map, **kwargs):
        if not torch.cuda.is_available():
            raise NotImplementedError(
                "Quartet quantization is only supported on GPU. Please use a different quantizer."
            )

        if not is_qutlass_available():
            raise ImportError("Using `quartet` quantization requires qutlass: `pip install qutlass`")

        if not is_quartet_qat_available():
            raise ImportError("Using `quartet` quantization requires quartet_qat: `pip install quartet_qat`")

        if not is_quartet_available():
            raise ImportError("Using `quartet` quantization requires quartet: `pip install quartet`")

        if device_map is None:
            raise ValueError(
                "You are attempting to load a Quartet model without setting device_map."
                " Please set device_map comprised of 'cuda' devices."
            )
        elif isinstance(device_map, dict) and ("cpu" in device_map.values() or "disk" in device_map.values()):
            raise ValueError(
                "You are attempting to load a Quartet model with a device_map that contains a CPU or disk device."
                " This is not supported. Please remove the CPU or disk device from the device_map."
            )

    def update_torch_dtype(self, torch_dtype: "torch.dtype") -> "torch.dtype":
        if torch_dtype is None:
            logger.info("`torch_dtype` is None. Setting `torch_dtype=torch.float16` for FLUTE compatibility.")
            torch_dtype = torch.float16
        elif torch_dtype != torch.float16 and torch_dtype != torch.bfloat16:
            raise ValueError(
                f"Invalid `torch_dtype` {torch_dtype}. HIGGS quantization only supports `torch_dtype=torch.float16` or `torch_dtype=torch.bfloat16`."
            )

        return torch_dtype

    def create_quantized_param(
        self,
        model: "PreTrainedModel",
        param_value: "torch.Tensor",
        param_name: str,
        target_device: "torch.device",
        state_dict: Dict[str, Any],
        unexpected_keys: Optional[List[str]] = None,
    ):
        from quartet_qat import QuartetLinear

        module, _ = get_module_from_name(model, param_name)
        assert isinstance(module, QuartetLinear), f"Module {param_name} is not a QuartetLinear somehow..."

        if param_name.endswith(".weight_q"):
            module.weight_q = torch.nn.Parameter(
                param_value.to(target_device), requires_grad=False,
            )
            if not self.quantization_config.store_master_weights:
                module.weight = None
            return

        module.weight = torch.nn.Parameter(param_value.to(target_device), requires_grad=module.weight.requires_grad)
        module.pre_forward()

        if unexpected_keys is not None and param_name in unexpected_keys:
            unexpected_keys.remove(param_name)

    def _process_model_before_weight_loading(
        self,
        model: "PreTrainedModel",
        **kwargs,
    ):
        from quartet_qat import replace_with_quartet_linear

        from ..integrations.quartet import adapt_quartet_config

        replace_with_quartet_linear(
            model,
            quartet_linear_config=adapt_quartet_config(self.quantization_config),
        )
        model.config.quantization_config = self.quantization_config

    def _process_model_after_weight_loading(self, model: "PreTrainedModel", **kwargs):
        from quartet_qat import QuartetLinear

        quartet_modules = {
            name: module for name, module in model.named_modules() if isinstance(module, QuartetLinear)
        }
        for name, module in tqdm(quartet_modules.items(), desc="Pre-processing Quartet modules", leave=False):
            if not self.quantization_config.store_master_weights and module.weight is not None:
                module.weight = None

    def update_missing_keys(self, model, missing_keys: List[str], prefix: str) -> List[str]:
        from quartet_qat import QuartetLinear

        quartet_names = {name for name, module in model.named_modules() if isinstance(module, QuartetLinear)}

        def should_exclude(key: str) -> bool:
            if key.endswith(".weight") or key.endswith(".bias"):
                return False
            full_key = f"{prefix}.{key}"
            return any(name in key or name in full_key for name in quartet_names)

        return [key for key in missing_keys if not should_exclude(key)]

    @property
    def is_trainable(self, model: Optional["PreTrainedModel"] = None):
        return True

    def is_serializable(self, safe_serialization=None):
        return True

    def check_quantized_param(
        self,
        model: "PreTrainedModel",
        param_value: "torch.Tensor",
        param_name: str,
        state_dict: Dict[str, Any],
        **kwargs,
    ) -> bool:
        from quartet_qat import QuartetLinear

        module, tensor_name = get_module_from_name(model, param_name)
        if isinstance(module, QuartetLinear) and tensor_name in ["weight", "weight_q"]:
            # Only quantize weights of QuartetLinear modules that are not already quantized
            return True
        else:
            return False
