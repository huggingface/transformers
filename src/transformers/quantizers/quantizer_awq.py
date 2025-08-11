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
import importlib.metadata
from typing import TYPE_CHECKING, Optional

from packaging import version

from .base import HfQuantizer


if TYPE_CHECKING:
    from ..modeling_utils import PreTrainedModel

from ..utils import is_accelerate_available, is_auto_awq_available, is_torch_available, logging
from ..utils.quantization_config import AWQLinearVersion


if is_torch_available():
    import torch

logger = logging.get_logger(__name__)


class AwqQuantizer(HfQuantizer):
    """
    4-bit quantization for Activation-aware Weight Quantization(AWQ) (https://huggingface.co/papers/2306.00978)
    """

    # AWQ requires data calibration - we support only inference
    requires_calibration = True

    required_packages = ["awq", "accelerate"]

    def __init__(self, quantization_config, **kwargs):
        super().__init__(quantization_config, **kwargs)

    def validate_environment(self, device_map, **kwargs):
        if not is_auto_awq_available():
            raise ImportError("Loading an AWQ quantized model requires auto-awq library (`pip install autoawq`)")

        if not is_accelerate_available():
            raise ImportError("Loading an AWQ quantized model requires accelerate (`pip install accelerate`)")

        if (
            self.quantization_config.version == AWQLinearVersion.GEMM
            and not torch.cuda.is_available()
            and not torch.xpu.is_available()
        ):
            logger.warning_once("No CUDA or XPU found, consider switching to the IPEX version for CPU-only execution.")
            self.quantization_config.version = AWQLinearVersion.IPEX

        if self.quantization_config.version == AWQLinearVersion.IPEX:
            if version.parse(importlib.metadata.version("autoawq")) < version.parse("0.2.6"):
                raise RuntimeError(
                    "To use IPEX backend, you need autoawq>0.2.6. Please install the latest version or from source."
                )
            if device_map is None:
                logger.warning_once(
                    "You have loaded an AWQ model without setting device_map, please set 'cpu' or 'xpu' or 'auto'"
                )
            elif isinstance(device_map, dict) and "disk" in device_map.values():
                raise ValueError(
                    "You are attempting to load an IPEX version AWQ model with a device_map that contains disk device."
                    " This is not supported. Please make sure only cpu and xpu in the device_map."
                )
        else:
            if not torch.cuda.is_available() and not torch.xpu.is_available():
                raise RuntimeError(
                    "GPU is required to run AWQ quantized model. You can use IPEX version AWQ if you have an Intel CPU"
                )

            if device_map is None:
                logger.warning_once(
                    "You have loaded an AWQ model on CPU and have a CUDA/XPU device available, make sure to set "
                    "your model on a GPU device in order to run your model."
                )
            elif device_map is not None:
                if isinstance(device_map, dict) and any(
                    forbidden in device_map.values() for forbidden in ("cpu", torch.device("cpu"), "disk")
                ):
                    raise ValueError(
                        "You are attempting to load an AWQ model with a device_map that contains a CPU or disk device."
                        " This is not supported. Please remove the CPU or disk device from the device_map."
                    )

    def update_torch_dtype(self, torch_dtype):
        if torch_dtype is None:
            torch_dtype = torch.float16
            logger.info("Loading the model in `torch.float16`. To overwrite it, set `torch_dtype` manually.")
        elif torch_dtype == torch.bfloat16 and (torch.cuda.is_available() or torch.xpu.is_available()):
            logger.warning(
                "`torch.bfloat16` is not supported for AWQ CUDA/XPU kernels yet. Casting to `torch.float16`."
            )
            torch_dtype = torch.float16
        elif torch_dtype != torch.float16 and (torch.cuda.is_available() or torch.xpu.is_available()):
            logger.warning(
                "We suggest you to set `torch_dtype=torch.float16` for better efficiency on CUDA/XPU with AWQ."
            )
        return torch_dtype

    def _process_model_before_weight_loading(
        self, model: "PreTrainedModel", keep_in_fp32_modules: Optional[list[str]] = None, **kwargs
    ):
        from ..integrations import replace_quantization_scales, replace_with_awq_linear

        self.modules_to_not_convert = self.get_modules_to_not_convert(
            model, self.quantization_config.modules_to_not_convert, keep_in_fp32_modules, add_default_skips=True
        )

        model, has_been_replaced = replace_with_awq_linear(
            model, quantization_config=self.quantization_config, modules_to_not_convert=self.modules_to_not_convert
        )

        model = replace_quantization_scales(model, model.config.model_type)

        if not has_been_replaced:
            logger.warning(
                "You are loading an AWQ model but no linear modules were found in your model."
                " Please double check your model architecture, or submit an issue on github if you think this is a bug."
            )

    def _process_model_after_weight_loading(self, model, **kwargs):
        if self.quantization_config.do_fuse:
            from ..integrations import fuse_awq_modules

            model = fuse_awq_modules(model, self.quantization_config)
            model._awq_is_fused = True  # TODO: consider storing this flag in model.config instead

        if self.quantization_config.version == AWQLinearVersion.EXLLAMA:
            from ..integrations import post_init_awq_exllama_modules

            model = post_init_awq_exllama_modules(model, self.quantization_config.exllama_config)

        if self.quantization_config.version == AWQLinearVersion.IPEX:
            from ..integrations import post_init_awq_ipex_modules

            model = post_init_awq_ipex_modules(model)

    def is_serializable(self, safe_serialization=None):
        # AWQ through auto-awq has been always serializable, except if the model is fused.
        if self.quantization_config.do_fuse:
            logger.warning("You cannot save an AWQ model that uses fused modules!")
            return False

        if self.quantization_config.version == AWQLinearVersion.EXLLAMA:
            logger.warning("You cannot save an AWQ model that uses Exllama backend!")
            return False

        return True

    @property
    def is_trainable(self):
        # AWQ supports PEFT fine-tuning from version 0.2.0
        MIN_AWQ_VERSION_FOR_PEFT = "0.2.0"
        return version.parse(importlib.metadata.version("autoawq")) >= version.parse(MIN_AWQ_VERSION_FOR_PEFT)
