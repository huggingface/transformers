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
    from ..utils.quantization_config import FbgemmFp8Config

from ..utils import (
    is_accelerate_available,
    is_fbgemm_gpu_available,
    is_kernels_available,
    is_torch_available,
    is_torch_cuda_available,
    is_torch_xpu_available,
    logging,
)
from .quantizers_utils import get_module_from_name


if is_torch_available():
    import torch

logger = logging.get_logger(__name__)


class FbgemmFp8HfQuantizer(HfQuantizer):
    """
    FP8 quantization using fbgemm kernels
    """

    requires_calibration = False
    serialized_names = {"weight": ["weight", "weight_scale"]}
    quantization_config: "FbgemmFp8Config"

    @property
    def quantized_element_size(self) -> float:
        return 1

    def validate_environment(self, *args, **kwargs):
        if not is_torch_cuda_available() and not is_torch_xpu_available():
            raise ImportError("Using fbgemm fp8 quantization requires a GPU or XPU")
        if is_torch_xpu_available() and not is_kernels_available():
            raise ImportError("Using FP8 fbgemm on XPU requires kernels (`pip install kernels`)")
        if is_torch_cuda_available() and not is_fbgemm_gpu_available():
            raise ImportError(
                "Loading an FP8 fbgemm quantized model on CUDA requires fbgemm-gpu library"
                "Please install the latest version of fbgemm-gpu library by following : https://pytorch.org/FBGEMM/fbgemm_gpu-development/InstallationInstructions.html#fbgemm-gpu-install-libraries"
            )
        if not is_accelerate_available():
            raise ImportError(
                "Loading an FP8 quantized model requires accelerate (`pip install --upgrade accelerate`)"
            )
        if is_torch_cuda_available():
            compute_capability = torch.cuda.get_device_capability()
            major, _ = compute_capability
            if major < 9:
                raise ValueError(
                    "FP8 quantized models is only supported on GPUs with compute capability >= 9.0 (e.g H100)"
                )

        device_map = kwargs.get("device_map")
        if device_map is None:
            logger.warning_once(
                "You have loaded an FP8 model on CPU and have a CUDA/XPU device available, make sure to set "
                "your model on a GPU/XPU device in order to run your model. To remove this warning, pass device_map = 'cuda' or 'xpu' or 'auto'. "
            )
        elif isinstance(device_map, dict):
            if not self.pre_quantized and ("cpu" in device_map.values() or "disk" in device_map.values()):
                raise ValueError(
                    "You are attempting to load an FP8 model with a device_map that contains a CPU or disk device."
                    "This is not supported when the model is quantized on the fly. "
                    "Please use a quantized checkpoint or remove the CPU or disk device from the device_map."
                )

    def update_dtype(self, dtype: "torch.dtype") -> "torch.dtype":
        if dtype != torch.bfloat16:
            logger.warning_once(
                f"Setting dtype to {dtype}, but only bfloat16 is supported right now. Overwriting torch_dtype to bfloat16."
            )
            dtype = torch.bfloat16
        return dtype

    def param_needs_quantization(self, model: "PreTrainedModel", param_name: str, **kwargs) -> bool:
        from ..integrations import FbgemmFp8Linear, FbgemmFp8Llama4TextExperts

        module, tensor_name = get_module_from_name(model, param_name)

        if isinstance(module, FbgemmFp8Linear):
            if self.pre_quantized or tensor_name == "bias":
                return False
            else:
                return True
        if isinstance(module, FbgemmFp8Llama4TextExperts):
            if self.pre_quantized or tensor_name == "bias":
                return False
            else:
                return True
        return False

    def _process_model_before_weight_loading(
        self,
        model: "PreTrainedModel",
        **kwargs,
    ):
        from ..integrations import replace_with_fbgemm_fp8_linear

        self.modules_to_not_convert = self.get_modules_to_not_convert(
            model, self.quantization_config.modules_to_not_convert, model._keep_in_fp32_modules
        )

        model = replace_with_fbgemm_fp8_linear(
            model,
            modules_to_not_convert=self.modules_to_not_convert,
            quantization_config=self.quantization_config,
            pre_quantized=self.pre_quantized,
        )

    def _process_model_after_weight_loading(self, model, **kwargs):
        """
        Force update the input scale upper bound after weight loading and device dispatch are complete.
        This resolves issues where persistent buffers are zeroed out or overwritten during the loading process.
        """
        from ..integrations.fbgemm_fp8 import FbgemmFp8Linear, FbgemmFp8Llama4TextExperts

        for m in model.modules():
            if isinstance(m, (FbgemmFp8Linear, FbgemmFp8Llama4TextExperts)):
                if hasattr(m, "input_scale_ub"):
                    # The model is now on the target device, so we can use fill_ directly.
                    m.input_scale_ub.fill_(self.quantization_config.activation_scale_ub)
        return model

    def is_serializable(self):
        return True

    @property
    def is_trainable(self) -> bool:
        return False

    def get_quantize_ops(self):
        from ..integrations.fbgemm_fp8 import FbgemmFp8Quantize

        return FbgemmFp8Quantize(self)
