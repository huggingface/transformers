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
from typing import TYPE_CHECKING, Optional

from .base import HfQuantizer


if TYPE_CHECKING:
    from ..modeling_utils import PreTrainedModel
    from ..utils.quantization_config import NVFP4Config

from ..utils import is_torch_available, logging
from ..utils.quantization_config import QuantizationConfigMixin


if is_torch_available():
    import torch


logger = logging.get_logger(__name__)


class NVFP4HfQuantizer(HfQuantizer):
    """
    Quantizer for NVIDIA's NVFP4 format (Blackwell 4-bit float with per-group
    scales + global scale). Loads pre-quantized checkpoints produced by
    NVIDIA's ModelOpt. Base weights remain frozen; LoRA adapters on top are
    trainable via PEFT.
    """

    requires_calibration = False
    is_qat_trainable = True  # LoRA on top of frozen NVFP4 base is trainable
    quantization_config: "NVFP4Config"

    def __init__(self, quantization_config: QuantizationConfigMixin, **kwargs):
        super().__init__(quantization_config, **kwargs)

    def validate_environment(self, device_map, **kwargs):
        if not torch.cuda.is_available():
            raise NotImplementedError(
                "NVFP4 quantization is only supported on CUDA GPUs. The Triton "
                "dequant kernel targets Blackwell (compute capability >= 10); "
                "the pure-torch fallback runs on any CUDA GPU."
            )

        if device_map is None:
            raise ValueError(
                "Loading an NVFP4 model requires `device_map` to be set "
                "(e.g. `device_map='cuda'` or an explicit layer-to-device dict)."
            )

        if isinstance(device_map, dict) and (
            "cpu" in device_map.values() or "disk" in device_map.values()
        ):
            raise ValueError(
                "NVFP4 quantization does not support offloading layers to CPU "
                "or disk. Remove any 'cpu' / 'disk' entries from device_map."
            )

    def update_dtype(self, dtype: "torch.dtype") -> "torch.dtype":
        # Don't force a global dtype — our buffers already declare their own
        # (uint8/fp8_e4m3/fp32/bf16). Forcing bf16 here may cause HF's loader
        # to cast intermediate tensors unnecessarily during state_dict apply.
        return dtype if dtype is not None else torch.bfloat16

    def param_needs_quantization(self, model: "PreTrainedModel", param_name: str, **kwargs) -> bool:
        # Pre-quantized checkpoints only — we never on-the-fly quantize.
        return False

    def param_element_size(self, model, param_name, param):
        """Report byte count per element. For NVFP4 buffers our element_size() is already
        correct (uint8 packed = 1 byte, fp8 scales = 1 byte). Issue: if accelerator_device_map
        references legacy '.weight' keys from pre-swap nn.Linear, get_parameter_or_buffer
        returns a meta-sized bf16 placeholder and inflates byte count. Print to see."""
        return param.element_size()

    def _process_model_before_weight_loading(self, model: "PreTrainedModel", **kwargs):
        from ..integrations.nvfp4 import (
            replace_with_nvfp4_linear,
            replace_fused_moe_experts_with_nvfp4,
        )

        # First: replace Qwen 3.5 MoE fused expert parameters with per-expert NVFP4 modules.
        # MUST run before replace_with_nvfp4_linear because these are not nn.Linear — they
        # are fused nn.Parameter tensors that Qwen's MoE module holds directly.
        replace_fused_moe_experts_with_nvfp4(
            model,
            modules_to_not_convert=self.quantization_config.modules_to_not_convert,
            quantization_config=self.quantization_config,
        )

        # Then: swap all remaining nn.Linear modules (attention q/k/v/o_proj, shared experts,
        # router gates) with NVFP4Linear.
        replace_with_nvfp4_linear(
            model,
            modules_to_not_convert=self.quantization_config.modules_to_not_convert,
            quantization_config=self.quantization_config,
        )

    def _process_model_after_weight_loading(self, model: "PreTrainedModel", **kwargs):
        # No-op for now. Hook reserved for optional cache_dequant() preload if
        # we later expose a config flag to trade memory for forward speed.
        pass

    @property
    def is_trainable(self) -> bool:
        # LoRA adapters (attached later via PEFT) are the trainable surface.
        # The NVFP4 base weights themselves stay frozen.
        return True

    def is_serializable(self):
        return True

    def get_weight_conversions(self):
        """Stream NVFP4 per-expert / per-linear tensors into pre-allocated GPU buffers
        without HF retention. See NVFP4PlaceOp for rationale.
        """
        from ..core_model_loading import WeightConverter
        from ..integrations.nvfp4 import NVFP4PlaceOp

        op = NVFP4PlaceOp()
        return [
            WeightConverter(
                source_patterns=[".weight_packed"],
                target_patterns=".weight_packed",
                operations=[op],
            ),
            WeightConverter(
                source_patterns=[".weight_scale"],
                target_patterns=".weight_scale",
                operations=[op],
            ),
            WeightConverter(
                source_patterns=[".weight_global_scale"],
                target_patterns=".weight_global_scale",
                operations=[op],
            ),
        ]
