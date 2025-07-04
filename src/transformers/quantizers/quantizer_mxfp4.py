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
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from .base import HfQuantizer


if TYPE_CHECKING:
    from ..modeling_utils import PreTrainedModel

from ..utils import is_torch_available, logging, is_accelerate_available
from .quantizers_utils import get_module_from_name


if is_torch_available():
    import torch

logger = logging.get_logger(__name__)


class Mxfp4HfQuantizer(HfQuantizer):
    """
    FP4 quantization using fbgemm kernels
    """

    requires_parameters_quantization = True
    # to remove if we decide to allow quantizing weights with this method
    requires_calibration = True

    required_packages = ["accelerate"]

    def __init__(self, quantization_config, **kwargs):
        super().__init__(quantization_config, **kwargs)
        self.quantization_config = quantization_config

    def validate_environment(self, *args, **kwargs):
        if not is_torch_available():
            raise ImportError(
                "Using mxfp4 quantization requires torch"
                "Please install the latest version of torch ( pip install --upgrade torch )"
            )
        if not torch.cuda.is_available():
            raise RuntimeError("Using MXFP4 quantized models requires a GPU")

        compute_capability = torch.cuda.get_device_capability()
        major, minor = compute_capability
        # TODO: Fix that
        # if not is_triton_kernels_availalble():
        #     raise ValueError(
        #         "MXFP4 quantization requires triton_kernels library"
        #     )
        if major < 9:
            raise ValueError(
                "MXFP4 quantized models is only supported on GPUs with compute capability >= 9.0 (e.g H100)"
            )
        if not is_accelerate_available():
            raise ImportError(
                f"Using `bitsandbytes` 4-bit quantization requires Accelerate: `pip install 'accelerate>=1.8.0'`"
            )

        device_map = kwargs.get("device_map", None)
        if device_map is None:
            logger.warning_once(
                "You have loaded an FP4 model on CPU and have a CUDA device available, make sure to set "
                "your model on a GPU device in order to run your model. To remove this warning, pass device_map = 'cuda'. "
            )
        elif device_map is not None:
            if (
                not self.pre_quantized
                and isinstance(device_map, dict)
                and ("cpu" in device_map.values() or "disk" in device_map.values())
            ):
                raise ValueError(
                    "You are attempting to load an FP4 model with a device_map that contains a CPU or disk device."
                    "This is not supported when the model is quantized on the fly. "
                    "Please use a quantized checkpoint or remove the CPU or disk device from the device_map."
                )
        from triton_kernels.numerics_details.mxfp import SwizzlingType

        if major < 9:
            # NYI for Ampere
            swizzle_mx_value = None
            swizzle_mx_scale = None
        elif major < 10:
            swizzle_mx_value = SwizzlingType.HOPPER
            swizzle_mx_scale = SwizzlingType.HOPPER
        else:
            swizzle_mx_value = None
            swizzle_mx_scale = SwizzlingType.BLACKWELL

        self.swizzle_mx_value = swizzle_mx_value
        self.swizzle_mx_scale = swizzle_mx_scale

    def update_torch_dtype(self, torch_dtype: "torch.dtype") -> "torch.dtype":
        if torch_dtype is None:
            torch_dtype = torch.bfloat16
            logger.info(
                "Overriding torch_dtype=%s with `torch_dtype=torch.bfloat16` due to "
                "requirements of `fbgemm-gpu` to enable model loading in fp4. "
                "Pass your own torch_dtype to specify the dtype of the remaining non-linear layers or pass"
                " torch_dtype=torch.bfloat16 to remove this warning.",
                torch_dtype,
            )
        return torch_dtype

    def check_quantized_param(
        self,
        model: "PreTrainedModel",
        param_value: "torch.Tensor",
        param_name: str,
        state_dict: Dict[str, Any],
        **kwargs,
    ):
        from ..integrations import Mxfp4OpenAIMoeExperts
        module, tensor_name = get_module_from_name(model, param_name)

        if isinstance(module, Mxfp4OpenAIMoeExperts):
            if tensor_name in ["down_proj_bias", "gate_up_proj_bias"]:
                return False
            return True
        return False

    def create_quantized_param(
        self,
        model: "PreTrainedModel",
        param_value: "torch.Tensor",
        param_name: str,
        target_device: "torch.device",
        state_dict: Dict[str, Any],
        unexpected_keys: Optional[List[str]] = None,
    ):
        from ..integrations import quantize_to_mxfp4, Mxfp4OpenAIMoeExperts, shuffle_weight
        from triton_kernels.matmul_ogs import FlexCtx, PrecisionConfig
        
        module, _ = get_module_from_name(model, param_name)

        
        with torch.cuda.device(target_device):
            if isinstance(module, Mxfp4OpenAIMoeExperts):
                if "gate_up_proj" in param_name:
                    right_pad = module.gate_up_proj_right_pad
                    bottom_pad = module.gate_up_proj_bottom_pad
                    # we only shuffle gate_proj
                    loaded_weight_shuffled = shuffle_weight(param_value).to(target_device)
                    loaded_weight = torch.nn.functional.pad(loaded_weight_shuffled,
                                            (0, right_pad, 0, bottom_pad, 0, 0),
                                            mode="constant",
                                            value=0)
                    del loaded_weight_shuffled
                    torch.cuda.empty_cache()
                    loaded_weight, flex, mx = quantize_to_mxfp4(
                    loaded_weight, self.swizzle_mx_value, self.swizzle_mx_scale)
                    module.gate_up_proj_precision_config = PrecisionConfig(mx_ctx=mx, flex_ctx=FlexCtx(rhs_data=flex))
                    module.gate_up_proj = torch.nn.Parameter(loaded_weight, requires_grad=False)
                elif "down_proj" in param_name:
                    right_pad = module.down_proj_right_pad
                    bottom_pad = module.down_proj_bottom_pad
                    loaded_weight = torch.nn.functional.pad(param_value,
                                            (0, right_pad, 0, bottom_pad, 0, 0),
                                            mode="constant",
                                            value=0).to(target_device)
                    # delete intermediate tensor immediate to prevent OOM
                    loaded_weight, flex, mx = quantize_to_mxfp4(
                        loaded_weight, self.swizzle_mx_value, self.swizzle_mx_scale)
                    module.down_proj_precision_config = PrecisionConfig(mx_ctx=mx, flex_ctx=FlexCtx(rhs_data=flex))
                    module.down_proj = torch.nn.Parameter(loaded_weight, requires_grad=False)

    def _process_model_after_weight_loading(self, model: "PreTrainedModel", **kwargs):
        from ..integrations import shuffle_weight, Mxfp4OpenAIMoeExperts

        for module in model.modules():
            if isinstance(module, Mxfp4OpenAIMoeExperts):
                gate_up_proj_bias = shuffle_weight(module.gate_up_proj_bias)
                gate_up_proj_bias = gate_up_proj_bias.to(torch.float32)
                gate_up_proj_bias = torch.nn.functional.pad(gate_up_proj_bias, (0, module.gate_up_proj_right_pad, 0, 0),
                                mode="constant",
                                value=0)
                down_proj_bias = module.down_proj_bias.to(torch.float32)
                down_proj_bias = torch.nn.functional.pad(down_proj_bias, (0, module.down_proj_right_pad, 0, 0),
                                mode="constant",
                                value=0)
                module.gate_up_proj_bias = torch.nn.Parameter(gate_up_proj_bias, requires_grad=False)
                module.down_proj_bias = torch.nn.Parameter(down_proj_bias, requires_grad=False)
        return model

    def _process_model_before_weight_loading(
        self,
        model: "PreTrainedModel",
        keep_in_fp32_modules: Optional[List[str]] = None,
        **kwargs,
    ):
        from ..integrations import replace_with_mxfp4_linear

        tp_plan = model._tp_plan
        self.modules_to_not_convert = self.get_modules_to_not_convert(
            model, self.quantization_config.modules_to_not_convert, keep_in_fp32_modules
        )

        config = model.config
        model = replace_with_mxfp4_linear(
            model,
            modules_to_not_convert=self.modules_to_not_convert,
            quantization_config=self.quantization_config,
            pre_quantized=self.pre_quantized,
            config=config,
            tp_plan=tp_plan,
        )

        model.config.quantization_config = self.quantization_config

    def update_missing_keys(self, model, missing_keys: List[str], prefix: str) -> List[str]:
        from ..integrations import Mxfp4OpenAIMoeExperts

        not_missing_keys = []
        for name, module in model.named_modules():
            if isinstance(module, Mxfp4OpenAIMoeExperts):
                for missing in missing_keys:
                    if (
                        (name in missing or name in f"{prefix}.{missing}")
                        and not missing.endswith(".weight")
                        and not missing.endswith(".bias")
                    ):
                        not_missing_keys.append(missing)
        return [k for k in missing_keys if k not in not_missing_keys]

    def update_tp_plan(self, config):
        # TODO: for tp support
        # if "OpenAIMoeExperts" in config.__class__.__name__:
        #     return config
        return config

    def is_serializable(self, safe_serialization=None):
        return True

    @property
    def is_trainable(self) -> bool:
        return False
