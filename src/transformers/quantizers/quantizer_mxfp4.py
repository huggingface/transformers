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

from ..utils import (
    is_accelerate_available,
    is_kernels_available,
    is_torch_available,
    is_triton_available,
    logging,
)
from .quantizers_utils import get_module_from_name


if is_torch_available():
    import torch

    from ..core_model_loading import WeightConverter

logger = logging.get_logger(__name__)
triton_kernels_hub = None


class Mxfp4HfQuantizer(HfQuantizer):
    """
    FP4 quantization using fbgemm kernels
    """

    requires_calibration = False

    def __init__(self, quantization_config, **kwargs):
        super().__init__(quantization_config, **kwargs)
        self.triton_kernels_hub = None

    def _lazy_import_kernels(self):
        """Lazy import and initialize kernels only when needed"""
        if self.triton_kernels_hub is None:
            try:
                from ..integrations.hub_kernels import get_kernel

                self.triton_kernels_hub = get_kernel("kernels-community/triton_kernels")
            except ImportError:
                raise ImportError("kernels package is required for MXFP4 quantization")
        return self.triton_kernels_hub

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

        if torch.xpu.is_available():
            is_device_supported_mxfp4 = True
            kernels_available = is_triton_available("3.5.0") and is_kernels_available()
        elif torch.cuda.is_available():
            compute_capability = torch.cuda.get_device_capability()
            is_device_supported_mxfp4 = compute_capability >= (7, 5)
            kernels_available = is_triton_available("3.4.0") and is_kernels_available()
        else:
            # CPU support mxfp4 in kernels
            is_device_supported_mxfp4 = True
            kernels_available = is_triton_available("3.4.0") and is_kernels_available()

        if self.pre_quantized:
            # On unsupported GPUs or without kernels, we will dequantize the model to bf16
            if not is_device_supported_mxfp4:
                logger.warning_once(
                    "MXFP4 quantization is only supported on GPUs with compute capability >= 7.5 (e.g T4, A100, L4, H100, or B200) or XPUs (e.g Intel® Data Center GPU Max Series) "
                    "We will default to dequantizing the model to bf16."
                )
                self.quantization_config.dequantize = True
                return

            if not kernels_available:
                logger.warning_once(
                    "MXFP4 quantization requires Triton and kernels installed: CUDA requires Triton >= 3.4.0, XPU requires Triton >= 3.5.0, we will default to dequantizing the model to bf16"
                )
                self.quantization_config.dequantize = True
                return
        elif not is_device_supported_mxfp4:
            # we can't quantize the model in this case so we raise an error
            raise ValueError(
                "MXFP4 quantization is only supported on GPUs with compute capability >= 7.5 (e.g T4, A100, L4, H100, or B200) or XPUs (e.g Intel® Data Center GPU Max Series) or CPU"
            )
        elif not kernels_available:
            # we can't quantize the model in this case so we raise an error
            raise ValueError(
                "MXFP4 quantization requires Triton and kernels installed: CUDA requires Triton >= 3.4.0, XPU/CPU requires Triton >= 3.5.0"
            )

        if not self.pre_quantized:
            self._lazy_import_kernels()

        device_map = kwargs.get("device_map")
        if device_map is None:
            logger.warning_once(
                "You have loaded an FP4 model on CPU and have a CUDA/XPU device available, make sure to set "
                "your model on a GPU/XPU device in order to run your model. To remove this warning, pass device_map = 'cuda' or device_map = 'xpu'. "
            )
        elif isinstance(device_map, dict):
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
        if use_kernels:
            logger.warning_once(
                "You are using full precision kernels, we will dequantize the model to bf16. "
                "To use the quantized model with quantization kernels, please set use_kernels=False"
            )
            self.quantization_config.dequantize = True

        self.modules_to_not_convert = self.get_modules_to_not_convert(
            model, self.quantization_config.modules_to_not_convert, model._keep_in_fp32_modules
        )

        model = replace_with_mxfp4_linear(
            model, modules_to_not_convert=self.modules_to_not_convert, quantization_config=self.quantization_config
        )

    def update_tp_plan(self, config):
        if "GptOssConfig" in config.__class__.__name__:
            if getattr(config, "base_model_tp_plan", None) is not None:
                config.base_model_tp_plan.update(
                    {
                        "layers.*.mlp.experts.gate_up_proj_blocks": "grouped_gemm",
                        "layers.*.mlp.experts.gate_up_proj_scales": "grouped_gemm",
                        "layers.*.mlp.experts.down_proj_blocks": "grouped_gemm",
                        "layers.*.mlp.experts.down_proj_scales": "grouped_gemm",
                    }
                )
        return config

    def update_ep_plan(self, config):
        if "GptOssConfig" in config.__class__.__name__:
            if getattr(config, "base_model_ep_plan", None) is not None:
                config.base_model_ep_plan.update(
                    {
                        "layers.*.mlp.experts.gate_up_proj_blocks": "grouped_gemm",
                        "layers.*.mlp.experts.gate_up_proj_scales": "grouped_gemm",
                        "layers.*.mlp.experts.down_proj_blocks": "grouped_gemm",
                        "layers.*.mlp.experts.down_proj_scales": "grouped_gemm",
                    }
                )
        return config

    def get_state_dict_and_metadata(self, model):
        from ..integrations import Mxfp4GptOssExperts

        state_dict = model.state_dict()

        # Get num_local_experts from model config
        num_local_experts = getattr(model.config, "num_local_experts", 32)
        hidden_size = getattr(model.config, "hidden_size", 2880)

        for name, module in model.named_modules():
            if (
                isinstance(module, Mxfp4GptOssExperts)
                and hasattr(module, "gate_up_proj")
                and hasattr(module, "down_proj")
            ):
                state_dict[f"{name}.gate_up_proj_blocks"] = (
                    module.gate_up_proj.storage.layout.unswizzle_data(module.gate_up_proj.storage.data)
                    .transpose(-1, -2)
                    .reshape(num_local_experts, -1, 90, 16)
                )
                state_dict[f"{name}.gate_up_proj_scales"] = (
                    module.gate_up_proj_precision_config.weight_scale.storage.layout.unswizzle_data(
                        module.gate_up_proj_precision_config.weight_scale.storage.data
                    ).transpose(-1, -2)
                )
                state_dict[f"{name}.down_proj_blocks"] = (
                    module.down_proj.storage.layout.unswizzle_data(module.down_proj.storage.data)
                    .transpose(-1, -2)
                    .reshape(num_local_experts, hidden_size, 90, -1)
                )
                state_dict[f"{name}.down_proj_scales"] = (
                    module.down_proj_precision_config.weight_scale.storage.layout.unswizzle_data(
                        module.down_proj_precision_config.weight_scale.storage.data
                    ).transpose(-1, -2)
                )

        metadata = {}
        return state_dict, metadata

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

    def get_weight_conversions(self):
        from ..integrations.mxfp4 import Mxfp4Dequantize, Mxfp4Deserialize

        if self.pre_quantized:
            if self.quantization_config.dequantize:
                return [
                    WeightConverter(
                        source_patterns=["_blocks", "_scales"],
                        target_patterns="",
                        operations=[Mxfp4Dequantize(self)],
                    )
                ]
            else:
                return [
                    WeightConverter(
                        source_patterns=["_blocks", "_scales"],
                        target_patterns="",
                        operations=[Mxfp4Deserialize(self)],
                    )
                ]
        return []
