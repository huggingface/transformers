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

from ..utils import is_accelerate_available, is_kernels_available, is_torch_available, logging
from .quantizers_utils import get_module_from_name


if is_torch_available():
    import torch


logger = logging.get_logger(__name__)


class EetqHfQuantizer(HfQuantizer):
    """
    8-bit quantization from EETQ quantization method:
        before loading: converts transformer layers into W8A16Linear during loading: load 16bit weight and pass to the
        layer object after: quantizes individual weights in Linear8bitLt into 8bit at first .cuda() call
    """

    requires_parameters_quantization = True
    requires_calibration = False

    required_packages = ["eetq", "accelerate"]

    def __init__(self, quantization_config, **kwargs):
        super().__init__(quantization_config, **kwargs)
        self.quantization_config = quantization_config

    def validate_environment(self, *args, **kwargs):
        if not is_kernels_available():
            raise ImportError("Loading an EETQ quantized model requires kernels (`pip install kernels`)")

        if not is_accelerate_available():
            raise ImportError("Loading an EETQ quantized model requires accelerate (`pip install accelerate`)")

        if not torch.cuda.is_available():
            raise RuntimeError("No GPU found. A GPU is needed for quantization.")

        device_map = kwargs.get("device_map")
        if device_map is None:
            logger.warning_once(
                "You have loaded an EETQ model on CPU and have a CUDA device available, make sure to set "
                "your model on a GPU device in order to run your model."
            )
        elif device_map is not None:
            if isinstance(device_map, dict) and ("cpu" in device_map.values() or "disk" in device_map.values()):
                raise ValueError(
                    "You are attempting to load an EETQ model with a device_map that contains a CPU or disk device."
                    " This is not supported. Please remove the CPU or disk device from the device_map."
                )

    def update_dtype(self, dtype: "torch.dtype") -> "torch.dtype":
        if dtype is None:
            dtype = torch.float16
            logger.info(
                "Overriding dtype=%s with `dtype=torch.float16` due to "
                "requirements of `eetq` to enable model loading in 8-bit. "
                "Pass your own dtype to specify the dtype of the remaining non-linear layers or pass"
                " dtype=torch.float16 to remove this warning.",
                dtype,
            )
        elif dtype != torch.float16:
            logger.info("We suggest you to set `dtype=torch.float16` for better efficiency with EETQ.")
        return dtype

    def param_needs_quantization(self, model: "PreTrainedModel", param_name: str, **kwargs) -> bool:
        from ..integrations.eetq import EetqLinear

        module, tensor_name = get_module_from_name(model, param_name)

        if isinstance(module, EetqLinear):
            if self.pre_quantized or tensor_name == "bias":
                return False
            else:
                return True
        return False

    def _process_model_before_weight_loading(
        self,
        model: "PreTrainedModel",
        keep_in_fp32_modules: list[str] | None = None,
        **kwargs,
    ):
        from ..integrations import replace_with_eetq_linear

        self.modules_to_not_convert = self.get_modules_to_not_convert(
            model, self.quantization_config.modules_to_not_convert, keep_in_fp32_modules
        )

        model = replace_with_eetq_linear(
            model, modules_to_not_convert=self.modules_to_not_convert, pre_quantized=self.pre_quantized
        )

        model.config.quantization_config = self.quantization_config

    def is_serializable(self, safe_serialization=None):
        return True

    @property
    def is_trainable(self) -> bool:
        return True

    def get_quantize_ops(self):
        from ..integrations.eetq import EetqQuantize

        return EetqQuantize(self)
