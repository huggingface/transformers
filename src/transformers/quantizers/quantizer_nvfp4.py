# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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
from __future__ import annotations

from typing import TYPE_CHECKING

from ..utils import is_kernels_available, is_torch_available
from ..utils.import_utils import KERNELS_MAX_VERSION, KERNELS_MIN_VERSION
from .base import HfQuantizer
from .quantizers_utils import get_module_from_name


if is_torch_available():
    import torch

if TYPE_CHECKING:
    from ..modeling_utils import PreTrainedModel
    from ..utils.quantization_config import NVFP4Config


def _as_cuda_device(device) -> torch.device | None:
    if isinstance(device, int):
        return torch.device("cuda", device)
    try:
        device = torch.device(device)
    except (TypeError, RuntimeError):
        return None
    return device if device.type == "cuda" else None


class NVFP4HfQuantizer(HfQuantizer):
    """Quantize eligible linear weights to NVFP4 while loading a full-precision checkpoint."""

    requires_calibration = False
    quantization_config: NVFP4Config

    def validate_environment(self, device_map, **kwargs):
        if self.pre_quantized:
            raise ValueError("Loading pre-quantized NVFP4 checkpoints is not supported yet.")
        if not is_kernels_available():
            raise ImportError(
                "NVFP4 quantization requires the `kernels` package. "
                f"Install a compatible version ({KERNELS_MIN_VERSION} <= version < {KERNELS_MAX_VERSION}), "
                f"for example with `pip install kernels=={KERNELS_MIN_VERSION}`."
            )
        if not torch.cuda.is_available():
            raise RuntimeError("NVFP4 quantization requires a CUDA GPU.")

        if device_map is None:
            raise ValueError("NVFP4 quantization requires a CUDA `device_map`.")
        devices = set(device_map.values()) if isinstance(device_map, dict) else {device_map}
        cuda_devices = {_as_cuda_device(device) for device in devices}
        if None in cuda_devices:
            raise ValueError("NVFP4 quantization does not support CPU or disk offload.")
        if len(cuda_devices) != 1:
            raise ValueError(
                "NVFP4 quantization currently supports one CUDA device. Tensor parallelism and multi-device "
                "`device_map` configurations are not supported yet."
            )
        self.quantization_device = cuda_devices.pop()

        major, minor = torch.cuda.get_device_capability(self.quantization_device)
        if major < 10:
            raise RuntimeError(f"NVFP4 requires Blackwell (sm100+) block-scaled tensor cores; found sm{major}{minor}.")

    def update_tp_plan(self, config):
        distributed_config = getattr(config, "distributed_config", None)
        tp_size = getattr(distributed_config, "tp_size", None)
        if tp_size is not None and tp_size > 1:
            raise ValueError(
                "NVFP4 quantization does not support tensor parallelism yet because its scale metadata does not have "
                "a sharding plan."
            )
        return config

    def param_needs_quantization(self, model: PreTrainedModel, param_name: str, **kwargs) -> bool:
        from ..integrations.nvfp4 import NVFP4Linear

        module, tensor_name = get_module_from_name(model, param_name)
        return isinstance(module, NVFP4Linear) and tensor_name == "weight"

    def param_element_size(self, model: PreTrainedModel, param_name: str, param: torch.Tensor) -> float:
        if self.param_needs_quantization(model, param_name):
            # NVFP4 packs two 4-bit weights per byte; this estimate excludes the smaller scale metadata.
            return 0.5
        return super().param_element_size(model, param_name, param)

    def _process_model_before_weight_loading(self, model: PreTrainedModel, **kwargs):
        from ..integrations.nvfp4 import replace_with_nvfp4_linear

        self.modules_to_not_convert = self.get_modules_to_not_convert(
            model, self.quantization_config.modules_to_not_convert, model._keep_in_fp32_modules
        )
        replace_with_nvfp4_linear(model, modules_to_not_convert=self.modules_to_not_convert)

    def get_quantize_ops(self):
        from ..integrations.nvfp4 import NVFP4Quantize

        return NVFP4Quantize(self.quantization_device)

    @property
    def is_serializable(self) -> bool:
        return False

    @property
    def is_trainable(self) -> bool:
        return False

    @property
    def is_compileable(self) -> bool:
        return True
