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
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from packaging import version

from .base import HfQuantizer


if TYPE_CHECKING:
    from ..modeling_utils import PreTrainedModel

from ..utils import is_accelerate_available, is_fbgemm_gpu_available, is_torch_available, logging
from .quantizers_utils import get_module_from_name


if is_torch_available():
    import torch


logger = logging.get_logger(__name__)


class FbgemmFp8HfQuantizer(HfQuantizer):
    """
    FP8 quantization using fbgemm kernels
    """

    requires_parameters_quantization = True
    requires_calibration = False

    required_packages = ["fbgemm-gpu", "accelerate"]

    def __init__(self, quantization_config, **kwargs):
        super().__init__(quantization_config, **kwargs)
        self.quantization_config = quantization_config

    def validate_environment(self, *args, **kwargs):
        if not is_torch_available() or version.parse(importlib.metadata.version("torch")) < version.parse("2.1.0"):
            raise ImportError(
                "Using fbgemm fp8 quantization requires torch > 2.1.0"
                "Please install the latest version of torch ( pip install --upgrade torch )"
            )
        if not is_fbgemm_gpu_available():
            raise ImportError(
                "Using fbgemm fp8 quantization requires fbgemm-gpu library"
                "Please install the latest version of fbgemm-gpu library by following : https://pytorch.org/FBGEMM/fbgemm_gpu-development/InstallationInstructions.html#fbgemm-gpu-install-libraries"
            )

        if not is_accelerate_available("0.32.2"):
            raise ImportError(
                "Loading an FP8 quantized model requires accelerate > 0.32.1 (`pip install --upgrade accelerate`)"
            )

        if not torch.cuda.is_available():
            raise RuntimeError("Using FP8 quantized models with fbgemm kernels requires a GPU")

        compute_capability = torch.cuda.get_device_capability()
        major, minor = compute_capability
        if major < 9:
            raise ValueError(
                "FP8 quantized models is only supported on GPUs with compute capability >= 9.0 (e.g H100)"
            )

        device_map = kwargs.get("device_map", None)
        if device_map is None:
            logger.warning_once(
                "You have loaded an FP8 model on CPU and have a CUDA device available, make sure to set "
                "your model on a GPU device in order to run your model. To remove this warning, pass device_map = 'cuda'. "
            )
        elif device_map is not None:
            if (
                not self.pre_quantized
                and isinstance(device_map, dict)
                and ("cpu" in device_map.values() or "disk" in device_map.values())
            ):
                raise ValueError(
                    "You are attempting to load an FP8 model with a device_map that contains a CPU or disk device."
                    "This is not supported when the model is quantized on the fly. "
                    "Please use a quantized checkpoint or remove the CPU or disk device from the device_map."
                )

    def update_torch_dtype(self, torch_dtype: "torch.dtype") -> "torch.dtype":
        if torch_dtype is None:
            torch_dtype = torch.bfloat16
            logger.info(
                "Overriding torch_dtype=%s with `torch_dtype=torch.bloat16` due to "
                "requirements of `fbgemm-gpu` to enable model loading in fp8. "
                "Pass your own torch_dtype to specify the dtype of the remaining non-linear layers or pass"
                " torch_dtype=torch.bfloat16 to remove this warning.",
                torch_dtype,
            )
        elif torch_dtype == torch.float16:
            raise ValueError(
                "You cannot use FP8 with torch_dtype=torch.float16.We recommend you passing torch_dtype=torch.bfloat16"
            )
        return torch_dtype

    def check_quantized_param(
        self,
        model: "PreTrainedModel",
        param_value: "torch.Tensor",
        param_name: str,
        state_dict: Dict[str, Any],
        **kwargs,
    ):
        from ..integrations import FbgemmFp8Linear

        module, tensor_name = get_module_from_name(model, param_name)

        if isinstance(module, FbgemmFp8Linear):
            if self.pre_quantized or tensor_name == "bias":
                if tensor_name == "weight" and param_value.dtype != torch.float8_e4m3fn:
                    raise ValueError("Expect quantized weights but got an unquantized weight")
                return False
            else:
                if tensor_name == "weight_scale":
                    raise ValueError("Expect unquantized weights but got a quantized weight_scale")
                return True
        return False

    def create_quantized_param(
        self,
        model: "PreTrainedModel",
        param_value: "torch.Tensor",
        param_name: str,
        target_device: "torch.device",
        state_dict: Dict[str, Any],
        unexpected_keys: Optional[List[str]] = None,
    ):
        """
        Quantizes weights into weight and weight_scale
        """
        new_value, weight_scale = torch.ops.fbgemm.quantize_fp8_per_row(param_value)

        module, tensor_name = get_module_from_name(model, param_name)
        module._buffers[tensor_name] = new_value.to(target_device)
        # to have the right output shape -> (out_features, 1)
        module._buffers["weight_scale"] = weight_scale.view(weight_scale.shape[0], 1).to(target_device)

        if unexpected_keys is not None and param_name in unexpected_keys:
            unexpected_keys.remove(param_name)
        del param_name

    def _process_model_after_weight_loading(self, model: "PreTrainedModel", **kwargs):
        return model

    def _process_model_before_weight_loading(
        self,
        model: "PreTrainedModel",
        device_map,
        keep_in_fp32_modules: List[str] = [],
        **kwargs,
    ):
        from ..integrations import get_keys_to_not_convert, replace_with_fbgemm_fp8_linear

        self.modules_to_not_convert = get_keys_to_not_convert(model)

        if self.quantization_config.modules_to_not_convert is not None:
            self.modules_to_not_convert.extend(self.quantization_config.modules_to_not_convert)

        model = replace_with_fbgemm_fp8_linear(
            model,
            modules_to_not_convert=self.modules_to_not_convert,
            quantization_config=self.quantization_config,
            pre_quantized=self.pre_quantized,
        )

        model.config.quantization_config = self.quantization_config

    def update_missing_keys(self, model, missing_keys: List[str], prefix: str) -> List[str]:
        from ..integrations import FbgemmFp8Linear

        not_missing_keys = []
        for name, module in model.named_modules():
            if isinstance(module, FbgemmFp8Linear):
                for missing in missing_keys:
                    if (
                        (name in missing or name in f"{prefix}.{missing}")
                        and not missing.endswith(".weight")
                        and not missing.endswith(".bias")
                    ):
                        not_missing_keys.append(missing)
        return [k for k in missing_keys if k not in not_missing_keys]

    def is_serializable(self, safe_serialization=None):
        return True

    @property
    def is_trainable(self) -> bool:
        return False
