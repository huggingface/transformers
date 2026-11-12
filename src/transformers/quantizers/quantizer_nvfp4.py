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
    from torch import nn

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

    @property
    def quantized_element_size(self) -> float:
        return 0.5

    def validate_environment(self, device_map, **kwargs):
        if self.quantization_config.weight_only and not self.pre_quantized:
            raise ValueError("`NVFP4Config(weight_only=True)` loads a pre-quantized weight-only (W4A16) checkpoint.")
        if self.pre_quantized and (self.quantization_config.dequantize or self.quantization_config.weight_only):
            return  # the `dequantize` load operation and the weight-only forward are pure torch
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
        # Quantizing on the fly packs every weight on one device; a pre-quantized checkpoint can spread over several
        if len(cuda_devices) != 1 and not self.pre_quantized:
            raise ValueError(
                "Quantizing to NVFP4 while loading supports one CUDA device. Load a pre-quantized NVFP4 checkpoint to "
                "use a multi-device `device_map`."
            )
        for device in cuda_devices:
            major, minor = torch.cuda.get_device_capability(device)
            if major < 10:
                raise RuntimeError(
                    f"NVFP4 requires Blackwell (sm100+) block-scaled tensor cores; found sm{major}{minor} on {device}."
                )
        self.quantization_device = min(cuda_devices, key=lambda device: device.index or 0)

    def update_parallel_plan(self, model, parallel_plan):
        from ..distributed.parallel_utils import replace_layer_number_by_wildcard
        from ..integrations.nvfp4 import NVFP4Experts, NVFP4Linear

        # The experts' scales shard like their weight, along the expert axis, the swizzled ones too
        super().update_parallel_plan(model, parallel_plan)
        modules = {
            replace_layer_number_by_wildcard(name): module
            for name, module in model.named_modules()
            if isinstance(module, (NVFP4Linear, NVFP4Experts))
        }

        for key, style in list(parallel_plan.items()):
            module_name, _, param = key.rpartition(".")
            if isinstance(modules.get(module_name), NVFP4Experts) and hasattr(modules[module_name], f"{param}_sf"):
                parallel_plan.setdefault(f"{key}_sf", style)

        # Tensor parallelism would split the swizzled scales, which have no sharding plan yet. Expert parallelism takes
        # the experts' rules, and shards them whole
        distributed_config = getattr(model.config, "distributed_config", None)
        tp_size = getattr(distributed_config, "tp_size", None) or 1
        ep_size = getattr(distributed_config, "ep_size", None) or 1
        if tp_size == 1:
            return

        tp_sharded = (NVFP4Linear,) if ep_size > 1 else (NVFP4Linear, NVFP4Experts)
        if any(isinstance(module, tp_sharded) for module in modules.values()):
            raise ValueError(
                "NVFP4 quantization does not support tensor parallelism yet: its scale metadata does not have a "
                "sharding plan. NVFP4 experts shard with expert parallelism, so only the experts may be NVFP4."
            )

    def param_needs_quantization(self, model: PreTrainedModel, param_name: str, **kwargs) -> bool:
        from ..integrations.nvfp4 import NVFP4Linear

        module, tensor_name = get_module_from_name(model, param_name)
        return isinstance(module, NVFP4Linear) and tensor_name == "weight"

    def _process_model_before_weight_loading(self, model: PreTrainedModel, **kwargs):
        self.modules_to_not_convert = self.get_modules_to_not_convert(
            model, self.quantization_config.modules_to_not_convert, model._keep_in_fp32_modules
        )

    def quantized_module(self, name, module):
        from ..integrations.moe import use_experts_implementation
        from ..integrations.nvfp4 import ALL_NVFP4_EXPERTS_FUNCTIONS, NVFP4Experts, NVFP4Linear, is_nvfp4_compatible

        if self.pre_quantized and self.quantization_config.dequantize:
            return None
        weight_only = self.quantization_config.weight_only
        if name.endswith(".experts") and self.pre_quantized and is_nvfp4_compatible(module):
            # TODO: quantize experts from a full-precision checkpoint too
            has_gate = getattr(module, "has_gate", True)
            experts_class = use_experts_implementation(
                experts_class=NVFP4Experts, experts_interface=ALL_NVFP4_EXPERTS_FUNCTIONS, has_gate=has_gate
            )
            return experts_class(module.config, has_gate=has_gate, weight_only=weight_only)
        if isinstance(module, nn.Linear) and is_nvfp4_compatible(module):
            return NVFP4Linear(
                module.in_features, module.out_features, weight_only=weight_only, static_input_scale=self.pre_quantized
            )
        return None

    def get_quantize_ops(self):
        from ..integrations.nvfp4 import NVFP4Quantize

        return NVFP4Quantize(self.quantization_device)

    # As ModelOpt saves them, and the fused experts' params they merge into (`w1.weight_scale` -> `gate_up_proj_scale`)
    serialized_names = {
        "weight": ["weight", "weight_scale", "weight_scale_2", "input_scale"],
        "gate_up_proj": ["gate_up_proj", "gate_up_proj_scale", "gate_up_proj_scale_2", "gate_up_proj_input_scale"],
        "down_proj": ["down_proj", "down_proj_scale", "down_proj_scale_2", "down_proj_input_scale"],
    }

    # An expert's `input_scale`, named after its weight: the experts' converter takes it along
    checkpoint_renames = {"experts.*.input_scale": "weight_input_scale"}

    def get_load_operations(self):
        from ..integrations.nvfp4 import NVFP4Dequantize, NVFP4Deserialize

        return {"deserialize": NVFP4Deserialize, "dequantize": NVFP4Dequantize}

    def is_serializable(self) -> bool:
        return False

    @property
    def is_trainable(self) -> bool:
        return False

    @property
    def is_compileable(self) -> bool:
        return True
