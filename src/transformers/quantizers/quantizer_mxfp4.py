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
    requires_calibration = False

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
            swizzle_mx_scale = None
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
        from ..integrations import quantize_to_mxfp4, Mxfp4OpenAIMoeExperts, shuffle_weight, convert_moe_packed_tensors
        from triton_kernels.matmul_ogs import FlexCtx, PrecisionConfig
        from ..modeling_utils import _load_parameter_into_model

        if not self.pre_quantized:
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
        # we take this path if alredy quantized but not in a compatible way:
        else:
            module, _ = get_module_from_name(model, param_name)
            if isinstance(module, Mxfp4OpenAIMoeExperts):
                if "gate_up_proj" in param_name:
                    if module.gate_up_proj_blocks.device.type == "meta" and module.gate_up_proj_scales.device.type == "meta":
                        _load_parameter_into_model(model, param_name, param_value)
                        return
                    else:
                        # In this case the weights are already on the device, so param_value should be the scale value 
                        if (module.gate_up_proj_blocks.device != "meta" and "scales" in param_name) or (module.gate_up_proj_scales.device != "meta" and "blocks" in param_name):
                            _load_parameter_into_model(model, param_name, param_value)
                        else: 
                            raise ValueError(f"Something went horribly wrong mate in gate_up_proj")
                        
                        dequantized_gate_up_proj = convert_moe_packed_tensors(module.gate_up_proj_blocks, module.gate_up_proj_scales)

                        right_pad = module.gate_up_proj_right_pad
                        bottom_pad = module.gate_up_proj_bottom_pad
                        loaded_weight = torch.nn.functional.pad(dequantized_gate_up_proj,
                                                (0, right_pad, 0, bottom_pad, 0, 0),
                                                mode="constant",
                                                value=0)
                        # del dequantized_gate_up_proj
                        # torch.cuda.empty_cache()
                        loaded_weight, flex, mx = quantize_to_mxfp4(loaded_weight, self.swizzle_mx_value, self.swizzle_mx_scale)
                        module.gate_up_proj_precision_config = None #PrecisionConfig(mx_ctx=mx, flex_ctx=FlexCtx(rhs_data=flex))
                        module.gate_up_proj = dequantized_gate_up_proj #torch.nn.Parameter(loaded_weight, requires_grad=False)

                elif "down_proj" in param_name:
                    if module.down_proj_blocks.device.type == "meta" and module.down_proj_scales.device.type == "meta":
                        _load_parameter_into_model(model, param_name, param_value)
                        return
                    else:
                        if (module.down_proj_blocks.device != "meta" and "scales" in param_name) or (module.down_proj_scales.device != "meta" and "blocks" in param_name):
                            _load_parameter_into_model(model, param_name, param_value)
                        else: 
                            raise ValueError(f"Something went horribly wrong mate in down_proj")
                        
                        dequantized_down_proj = convert_moe_packed_tensors(module.down_proj_blocks, module.down_proj_scales)
                        
                        right_pad = module.down_proj_right_pad
                        bottom_pad = module.down_proj_bottom_pad

                        loaded_weight = torch.nn.functional.pad(dequantized_down_proj,
                                                (0, right_pad, 0, bottom_pad, 0, 0),
                                                mode="constant",
                                                value=0)
                        # del dequantized_down_proj

                        # torch.cuda.empty_cache()
                        loaded_weight, flex, mx = quantize_to_mxfp4(loaded_weight, self.swizzle_mx_value, self.swizzle_mx_scale)
                        module.down_proj_precision_config = None #PrecisionConfig(mx_ctx=mx, flex_ctx=FlexCtx(rhs_data=flex))
                        module.down_proj = dequantized_down_proj #torch.nn.Parameter(loaded_weight, requires_grad=False)

    def _process_model_after_weight_loading(self, model: "PreTrainedModel", **kwargs):
        from ..integrations import shuffle_weight, Mxfp4OpenAIMoeExperts
        # if not self.pre_quantized:
        #     for module in model.modules():
        #         if isinstance(module, Mxfp4OpenAIMoeExperts):
        #             # gate_up_proj_bias = shuffle_weight(module.gate_up_proj_bias)
        #             gate_up_proj_bias = module.gate_up_proj_bias.to(torch.float32)
        #             # gate_up_proj_bias = torch.nn.functional.pad(gate_up_proj_bias, (0, module.gate_up_proj_right_pad, 0, 0),
        #             #                 mode="constant",
        #             #                 value=0)
        #             down_proj_bias = module.down_proj_bias.to(torch.float32)
        #             # down_proj_bias = torch.nn.functional.pad(down_proj_bias, (0, module.down_proj_right_pad, 0, 0),
        #             #                 mode="constant",
        #             #                 value=0)
        #             module.gate_up_proj_bias = torch.nn.Parameter(gate_up_proj_bias, requires_grad=False)
        #             module.down_proj_bias = torch.nn.Parameter(down_proj_bias, requires_grad=False)
        reverse_replace_with_mxfp4_linear(model, modules_to_not_convert=self.modules_to_not_convert, quantization_config=self.quantization_config, config=model.config, tp_plan=model._tp_plan)
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
        return config

    def is_serializable(self, safe_serialization=None):
        return False

    @property
    def is_trainable(self) -> bool:
        return False


def _reverse_replace_with_mxfp4_linear(
    model,
    modules_to_not_convert=None,
    current_key_name=None,
    quantization_config=None,
    has_been_replaced=False,
    config=None,
    tp_plan=None,
):
    if current_key_name is None:
        current_key_name = []

    for name, module in model.named_children():
        current_key_name.append(name)
        # if isinstance(module, nn.Linear):
        #     raise NotImplementedError("Mxfp4 linear layer is not implemented yet")
        from ..models.openai_moe.modeling_openai_moe import OpenAIMoeExperts
        from accelerate import init_empty_weights
        if module.__class__.__name__ == "Mxfp4OpenAIMoeExperts":
            
                # tp_plan[re.sub(r"\d+", "*", current_key_name_str + ".down_proj_scale")] = None
            gate_up_proj = module.gate_up_proj
            down_proj = module.down_proj
            gate_up_proj_bias = module.gate_up_proj_bias
            down_proj_bias = module.down_proj_bias
            model._modules[name] = OpenAIMoeExperts(config)
            model._modules[name].gate_up_proj = torch.nn.Parameter(gate_up_proj.transpose(1,2), requires_grad=False)   
            model._modules[name].down_proj = torch.nn.Parameter(down_proj, requires_grad=False)
            model._modules[name].gate_up_proj_bias = torch.nn.Parameter(gate_up_proj_bias, requires_grad=False)
            model._modules[name].down_proj_bias = torch.nn.Parameter(down_proj_bias, requires_grad=False)
            has_been_replaced=True
        if len(list(module.children())) > 0:
            _, has_been_replaced = _reverse_replace_with_mxfp4_linear(
                module,
                modules_to_not_convert,
                current_key_name,
                quantization_config,
                has_been_replaced=has_been_replaced,
                config=config,
                tp_plan=tp_plan,
            )
        current_key_name.pop(-1)
    return model, has_been_replaced


def reverse_replace_with_mxfp4_linear(
    model,
    modules_to_not_convert=None,
    current_key_name=None,
    quantization_config=None,
    config=None,
    tp_plan=None,
):
    modules_to_not_convert = ["lm_head"] if modules_to_not_convert is None else modules_to_not_convert

    if quantization_config.modules_to_not_convert is not None:
        modules_to_not_convert.extend(quantization_config.modules_to_not_convert)
    modules_to_not_convert = list(set(modules_to_not_convert))
    model, has_been_replaced = _reverse_replace_with_mxfp4_linear(
        model,
        modules_to_not_convert,
        current_key_name,
        quantization_config,
        config=config,
        tp_plan=tp_plan,
    )
    if not has_been_replaced:
        logger.warning(
            "You are loading your model using mixed-precision FP4 quantization but no linear modules were found in your model."
            " Please double check your model architecture, or submit an issue on github if you think this is"
            " a bug."
        )

    return model
