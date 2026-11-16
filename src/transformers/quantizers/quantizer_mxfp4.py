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
from typing import TYPE_CHECKING

from .base import HfQuantizer


if TYPE_CHECKING:
    from ..modeling_utils import PreTrainedModel
    from ..utils.quantization_config import Mxfp4Config

from ..utils import (
    get_current_accelerator,
    is_accelerate_available,
    is_kernels_available,
    is_torch_available,
    is_triton_available,
    logging,
)
from ..utils.import_utils import KERNELS_MAX_VERSION, KERNELS_MIN_VERSION
from .quantizers_utils import get_module_from_name


if is_torch_available():
    import torch


logger = logging.get_logger(__name__)


class Mxfp4HfQuantizer(HfQuantizer):
    """
    FP4 quantization using fbgemm kernels
    """

    requires_calibration = False
    quantization_config: "Mxfp4Config"

    def validate_environment(self, *args, **kwargs):
        if not is_torch_available():
            raise ImportError(
                "Using mxfp4 quantization requires torch"
                "Please install the latest version of torch ( pip install --upgrade torch )"
            )

        if self.quantization_config.dequantize:
            return

        if not is_accelerate_available():
            raise ImportError("Using mxfp4 requires Accelerate: `pip install accelerate`")

        device = get_current_accelerator() or torch.device("cpu")
        if device.type not in ["cuda", "xpu", "cpu"]:
            if self.pre_quantized:
                logger.warning_once(
                    f"Using MXFP4 quantized models requires model on cuda/xpu/cpu, but found {device}, we will default to dequantizing the model to bf16. To use mxfp4, please disable the current accelerator."
                )
                self.quantization_config.dequantize = True
                return
            else:
                raise RuntimeError(
                    f"Quantizing a model using MXFP4 requires model on cuda/xpu/cpu, but found {device}. To use mxfp4, please disable the current accelerator."
                )

        if torch.xpu.is_available():
            is_device_supported_mxfp4 = True
            triton_available = is_triton_available("3.5.0")
            kernels_installed = is_kernels_available()
        elif torch.cuda.is_available():
            compute_capability = torch.cuda.get_device_capability()
            is_device_supported_mxfp4 = compute_capability >= (7, 5)
            triton_available = is_triton_available("3.4.0")
            kernels_installed = is_kernels_available()
        elif device.type == "cpu":
            is_device_supported_mxfp4 = True
            triton_available = is_triton_available("3.5.0")
            kernels_installed = is_kernels_available()
        else:
            is_device_supported_mxfp4 = False
            triton_available = False
            kernels_installed = False

        if self.pre_quantized:
            if not is_device_supported_mxfp4:
                logger.warning_once(
                    "MXFP4 quantization is only supported on GPUs with compute capability >= 7.5 "
                    "(e.g T4, A100, L4, H100, or B200) or XPUs (e.g Intel® Data Center GPU Max Series). "
                    "We will default to dequantizing the model to bf16."
                )
                self.quantization_config.dequantize = True
                return

            if not triton_available:
                logger.warning_once(
                    "MXFP4 quantization requires Triton: CUDA requires Triton >= 3.4.0, "
                    "XPU/CPU requires Triton >= 3.5.0. Please install triton: `pip install triton`. "
                    "We will default to dequantizing the model to bf16."
                )
                self.quantization_config.dequantize = True
                return

            if not kernels_installed:
                logger.warning_once(
                    "MXFP4 quantization requires the `kernels` package: "
                    f"Please install a compatible version ({KERNELS_MIN_VERSION} <= version < {KERNELS_MAX_VERSION}), "
                    f"e.g. `pip install kernels=={KERNELS_MIN_VERSION}`"
                    "We will default to dequantizing the model to bf16."
                )
                self.quantization_config.dequantize = True
                return
        elif not is_device_supported_mxfp4:
            raise ValueError(
                "MXFP4 quantization is only supported on GPUs with compute capability >= 7.5 "
                "(e.g T4, A100, L4, H100, or B200) or XPUs (e.g Intel® Data Center GPU Max Series) or CPU"
            )
        elif not triton_available:
            raise ValueError(
                "MXFP4 quantization requires Triton: CUDA requires Triton >= 3.4.0, "
                "XPU/CPU requires Triton >= 3.5.0. Please install triton: `pip install triton`"
            )
        elif not kernels_installed:
            raise ValueError("MXFP4 quantization requires the `kernels` package: `pip install kernels>=0.12.0`")

        device_map = kwargs.get("device_map")
        if device_map is not None and isinstance(device_map, dict):
            if not self.pre_quantized and "disk" in device_map.values():
                raise ValueError(
                    "You are attempting to load an FP4 model with a device_map that contains a disk device."
                    "This is not supported when the model is quantized on the fly. "
                    "Please use a quantized checkpoint or remove the disk device from the device_map."
                )

    def param_needs_quantization(self, model: "PreTrainedModel", param_name: str, **kwargs) -> bool:
        from ..integrations import Mxfp4GptOssExperts

        module, tensor_name = get_module_from_name(model, param_name)
        if isinstance(module, Mxfp4GptOssExperts):
            if tensor_name in ["down_proj_bias", "gate_up_proj_bias"]:
                return False
            return True
        return False

    def _process_model_after_weight_loading(self, model: "PreTrainedModel", **kwargs):
        # clean cache due to triton ops
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.xpu.is_available():
            torch.xpu.empty_cache()

    def _process_model_before_weight_loading(
        self,
        model: "PreTrainedModel",
        use_kernels: bool = False,
        **kwargs,
    ):
        from ..integrations import replace_with_mxfp4_linear

        # if we are using kernels, we can't use the quantized model, since the forward pass is different and needs special handling
        # only CPU and XPU kernels can work with pre-quantized models
        device = get_current_accelerator() or torch.device("cpu")
        if use_kernels and device.type not in ["cpu", "xpu"]:
            logger.warning_once(
                "You are using full precision kernels, we will dequantize the model to bf16. "
                "To use the quantized model with quantization kernels, please set use_kernels=False"
            )
            self.quantization_config.dequantize = True

        if not use_kernels and device.type in ["cpu"]:
            logger.warning_once(
                "MXFP4 inference on CPU requires use_kernels=True, but use_kernels is disabled. "
                "We will dequantize the model to bf16. To run MXFP4 natively on CPU, please set use_kernels=True."
            )
            self.quantization_config.dequantize = True

        self.modules_to_not_convert = self.get_modules_to_not_convert(
            model, self.quantization_config.modules_to_not_convert, model._keep_in_fp32_modules
        )

        model = replace_with_mxfp4_linear(
            model, modules_to_not_convert=self.modules_to_not_convert, quantization_config=self.quantization_config
        )

    def get_state_dict_and_metadata(self, model):
        from ..integrations.mxfp4 import Mxfp4GptOssExperts, Mxfp4ReverseDeserialize

        state_dict = model.state_dict()
        reverse_op = Mxfp4ReverseDeserialize(self)
        for name, module in model.named_modules():
            if isinstance(module, Mxfp4GptOssExperts):
                for proj in ("gate_up_proj", "down_proj"):
                    state_dict.update(reverse_op.convert({}, model=model, full_layer_name=f"{name}.{proj}_blocks"))
        return state_dict, {}

    def is_serializable(self):
        return True

    @property
    def is_trainable(self) -> bool:
        logger.warning_once(
            "MXFP4 quantization don't support training, please consider dequantizing the model first by passing quantization_config=Mxfp4Config(dequantize=True) to .from_pretrained()"
        )
        return False

    def get_quantize_ops(self):
        from ..integrations.mxfp4 import Mxfp4Quantize

        return Mxfp4Quantize(self)

    serialized_names = {
        "gate_up_proj": ["gate_up_proj_blocks", "gate_up_proj_scales"],
        "down_proj": ["down_proj_blocks", "down_proj_scales"],
    }

    def get_load_operations(self):
        from ..integrations.mxfp4 import Mxfp4Dequantize, Mxfp4Deserialize

        return {"deserialize": Mxfp4Deserialize, "dequantize": Mxfp4Dequantize}
