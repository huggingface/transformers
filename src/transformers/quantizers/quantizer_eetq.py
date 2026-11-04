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
    from ..utils.quantization_config import EetqConfig

from ..utils import is_accelerate_available, is_kernels_available, is_torch_available, logging
from .quantizers_utils import get_module_from_name


if is_torch_available():
    import torch
    import torch.nn as nn


logger = logging.get_logger(__name__)


class EetqHfQuantizer(HfQuantizer):
    """
    8-bit quantization from EETQ quantization method
    """

    requires_calibration = False
    quantization_config: "EetqConfig"

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
        elif isinstance(device_map, dict):
            if len(device_map) > 1 and "cpu" in device_map.values() or "disk" in device_map.values():
                raise ValueError(
                    "You are attempting to load an EETQ model with a device_map that contains a CPU or disk device."
                    " This is not supported. Please remove the CPU or disk device from the device_map."
                )

    def update_dtype(self, dtype: "torch.dtype") -> "torch.dtype":
        if dtype != torch.float16:
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
        **kwargs,
    ):
        from ..integrations import load_eetq_kernels

        load_eetq_kernels()
        self.modules_to_not_convert = self.get_modules_to_not_convert(
            model, self.quantization_config.modules_to_not_convert, model._keep_in_fp32_modules
        )

    def quantized_module(self, name, module):
        from ..integrations.eetq import EetqLinear

        if not isinstance(module, nn.Linear):
            return None
        # we need this to correctly materialize the weights during quantization
        module_kwargs = {} if self.pre_quantized else {"dtype": None}
        return EetqLinear(module.in_features, module.out_features, bias=module.bias is not None, **module_kwargs)

    def is_serializable(self):
        return True

    @property
    def is_trainable(self) -> bool:
        return True

    def get_quantize_ops(self):
        from ..integrations.eetq import EetqQuantize

        return EetqQuantize(self)
