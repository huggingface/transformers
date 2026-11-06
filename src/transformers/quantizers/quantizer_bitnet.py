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
    from ..utils.quantization_config import BitNetQuantConfig

from ..utils import is_accelerate_available, is_torch_available, logging


if is_torch_available():
    import torch


logger = logging.get_logger(__name__)


class BitNetHfQuantizer(HfQuantizer):
    """
    1.58-bit quantization from BitNet quantization method:
    Before loading: it converts the linear layers into BitLinear layers during loading.

    Check out the paper introducing this method: https://huggingface.co/papers/2402.17764
    """

    requires_calibration = True
    quantization_config: "BitNetQuantConfig"

    def validate_environment(self, *args, **kwargs):
        if not is_accelerate_available():
            raise ImportError("Loading a BitNet quantized model requires accelerate (`pip install accelerate`)")

        if not torch.cuda.is_available():
            logger.warning_once(
                "You don't have a GPU available to load the model, the inference will be slow because of weight unpacking"
            )
            return

        device_map = kwargs.get("device_map")
        if device_map is None:
            logger.warning_once(
                "You have loaded a BitNet model on CPU and have a CUDA device available, make sure to set "
                "your model on a GPU device in order to run your model."
            )
        elif isinstance(device_map, dict):
            if len(device_map) > 1 and "cpu" in device_map.values() or "disk" in device_map.values():
                raise ValueError(
                    "You are attempting to load a BitNet model with a device_map that contains a CPU or disk device."
                    "This is not supported. Please remove the CPU or disk device from the device_map."
                )

    def _process_model_before_weight_loading(
        self,
        model: "PreTrainedModel",
        **kwargs,
    ):
        self.modules_to_not_convert = self.get_modules_to_not_convert(
            model, self.quantization_config.modules_to_not_convert, model._keep_in_fp32_modules
        )

    def quantized_module(self, name, module):
        from ..integrations.bitnet import AutoBitLinear, BitLinear

        if not isinstance(module, torch.nn.Linear):
            return None
        config = self.quantization_config
        if config.linear_class == "autobitlinear":
            new_module = AutoBitLinear(
                in_features=module.in_features,
                out_features=module.out_features,
                bias=module.bias is not None,
                device=module.weight.device,
                dtype=module.weight.dtype,
                online_quant=(config.quantization_mode == "online"),
                use_rms_norm=config.use_rms_norm,
                rms_norm_eps=config.rms_norm_eps,
            )
            if config.quantization_mode == "offline":
                new_module.requires_grad_(False)
            return new_module
        new_module = BitLinear(
            in_features=module.in_features,
            out_features=module.out_features,
            bias=module.bias is not None,
            device=module.weight.device,
            dtype=module.weight.dtype,
            use_rms_norm=config.use_rms_norm,
            rms_norm_eps=config.rms_norm_eps,
        )
        return new_module.requires_grad_(False)

    def adjust_max_memory(self, max_memory: dict[str, int | str]) -> dict[str, int | str]:
        max_memory = {key: val * 0.90 for key, val in max_memory.items()}
        return max_memory

    def is_serializable(self):
        return True

    @property
    def is_trainable(self) -> bool:
        return (
            self.quantization_config.linear_class == "autobitlinear"
            and self.quantization_config.quantization_mode == "online"
        )

    @property
    def is_qat_trainable(self) -> bool:
        """Flag indicating whether the quantized model can carry out quantization aware training"""
        return (
            self.quantization_config.linear_class == "autobitlinear"
            and self.quantization_config.quantization_mode == "online"
        )

    serialized_names = {"weight": ["weight"]}

    def get_load_operations(self):
        from ..integrations.bitnet import BitNetDeserialize

        config = self.quantization_config
        if config.linear_class == "autobitlinear" and config.quantization_mode == "offline":
            return {"deserialize": BitNetDeserialize}
        return {}
