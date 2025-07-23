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
from typing import TYPE_CHECKING, Any, Optional

from .base import HfQuantizer


if TYPE_CHECKING:
    from ..modeling_utils import PreTrainedModel

from ..utils import (
    is_accelerate_available,
    is_torch_available,
    is_triton_available,
    is_triton_kernels_availalble,
    logging,
)
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

        if not is_triton_available("3.4.0") or not is_triton_kernels_availalble():
            if self.pre_quantized:
                logger.warning_once(
                    "MXFP4 quantization requires triton >= 3.4.0 and triton_kernels installed, we will default to dequantizing the model to bf16"
                )
                self.quantization_config.dequantize = True
            else:
                # we can't quantize the model in this case so we raise an error
                raise ValueError(
                    "MXFP4 quantization requires triton >= 3.4.0 and triton_kernels installed"
                )

        if major < 9:
            raise ValueError(
                "MXFP4 quantized models is only supported on GPUs with compute capability >= 9.0 (e.g H100, or B100)"
            )
        if not is_accelerate_available():
            raise ImportError(
                "Using mxfp4 requires Accelerate: `pip install 'accelerate>=1.8.0'`"
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
        state_dict: dict[str, Any],
        **kwargs,
    ):
        from ..integrations import Mxfp4OpenAIMoeExperts
        from ..models.openai_moe.modeling_openai_moe import OpenAIMoeExperts
        if self.quantization_config.dequantize and ("blocks" in param_name or "scales" in param_name):
            module, tensor_name = get_module_from_name(model, param_name[:-len("_blocks")])
        else:
            module, tensor_name = get_module_from_name(model, param_name)

        if isinstance(module, Mxfp4OpenAIMoeExperts) or (isinstance(module, OpenAIMoeExperts) and self.quantization_config.dequantize):
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
        state_dict: dict[str, Any],
        unexpected_keys: Optional[list[str]] = None,
        **kwargs,
    ):
        from triton_kernels.matmul_ogs import FlexCtx, PrecisionConfig

        from ..integrations import Mxfp4OpenAIMoeExperts, convert_moe_packed_tensors, quantize_to_mxfp4, shuffle_weight
        from ..integrations.tensor_parallel import shard_and_distribute_module
        from ..modeling_utils import _load_parameter_into_model
        from ..models.openai_moe.modeling_openai_moe import OpenAIMoeExperts

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
        # we take this path if already quantized but not in a compatible way:
        else:
            if ("blocks" in param_name or "scales" in param_name) and self.quantization_config.dequantize:
                # blocks and scales have the same length that's why the below line works
                module, _ = get_module_from_name(model, param_name[:-len("_blocks")])
            else:
                module, _ = get_module_from_name(model, param_name)
            if isinstance(module, Mxfp4OpenAIMoeExperts) or (isinstance(module, OpenAIMoeExperts) and self.quantization_config.dequantize):
                tp_mode = kwargs.get("device_mesh", None) is not None
                if self.quantization_config.dequantize:
                    neutral_param_name = param_name[:-len("_blocks")] if "blocks" in param_name else param_name[:-len("_scales")]
                    if "gate_up_proj" in param_name:
                        if not hasattr(module, "gate_up_proj_blocks") and not hasattr(module, "gate_up_proj_scales"):
                            if tp_mode:
                                param_value = shard_and_distribute_module(model, param_value, kwargs.get("empty_param"), neutral_param_name, kwargs.get("casting_dtype"), kwargs.get("to_contiguous"), kwargs.get("rank"), kwargs.get("device_mesh"), set_param_inside=False)
                            setattr(module, param_name.rsplit(".", 1)[1], param_value)
                            return
                        else:
                            if tp_mode:
                                param_value = shard_and_distribute_module(model, param_value, kwargs.get("empty_param"), neutral_param_name, kwargs.get("casting_dtype"), kwargs.get("to_contiguous"), kwargs.get("rank"), kwargs.get("device_mesh"), set_param_inside=False)
                            setattr(module, param_name.rsplit(".", 1)[1], param_value)

                            dequantized_gate_up_proj = convert_moe_packed_tensors(module.gate_up_proj_blocks, module.gate_up_proj_scales)
                            dequantized_gate_up_proj = dequantized_gate_up_proj.transpose(1,2).to(target_device)
                            module.gate_up_proj = torch.nn.Parameter(dequantized_gate_up_proj, requires_grad=False)
                            del module.gate_up_proj_blocks
                            del module.gate_up_proj_scales
                            return
                    elif "down_proj" in param_name:
                        if not hasattr(module, "down_proj_blocks") and not hasattr(module, "down_proj_scales"):
                            if tp_mode:
                                param_value = shard_and_distribute_module(model, param_value, kwargs.get("empty_param"), neutral_param_name, kwargs.get("casting_dtype"), kwargs.get("to_contiguous"), kwargs.get("rank"), kwargs.get("device_mesh"), set_param_inside=False)
                            setattr(module, param_name.rsplit(".", 1)[1], param_value)
                            return
                        else:
                            if tp_mode:
                                param_value = shard_and_distribute_module(model, param_value, kwargs.get("empty_param"), neutral_param_name, kwargs.get("casting_dtype"), kwargs.get("to_contiguous"), kwargs.get("rank"), kwargs.get("device_mesh"), set_param_inside=False)
                            setattr(module, param_name.rsplit(".", 1)[1], param_value)

                            dequantized_down_proj = convert_moe_packed_tensors(module.down_proj_blocks, module.down_proj_scales)
                            dequantized_down_proj = dequantized_down_proj.transpose(1,2).to(target_device)
                            module.down_proj = torch.nn.Parameter(dequantized_down_proj, requires_grad=False)
                            del module.down_proj_blocks
                            del module.down_proj_scales
                            return
                else:
                    if "gate_up_proj" in param_name:
                        if module.gate_up_proj_blocks.device.type == "meta" and module.gate_up_proj_scales.device.type == "meta":
                            if tp_mode:
                                shard_and_distribute_module(model, param_value, kwargs.get("empty_param"), param_name, kwargs.get("casting_dtype"), kwargs.get("to_contiguous"), kwargs.get("rank"), kwargs.get("device_mesh"))
                            else:
                                _load_parameter_into_model(model, param_name, param_value)
                            return
                        else:
                            # In this case the weights or the scales are already on the correct device, so param_value should be the other missing param
                            if (module.gate_up_proj_blocks.device != "meta" and "scales" in param_name) or (module.gate_up_proj_scales.device != "meta" and "blocks" in param_name):
                                if tp_mode:
                                    shard_and_distribute_module(model, param_value, kwargs.get("empty_param"), param_name, kwargs.get("casting_dtype"), kwargs.get("to_contiguous"), kwargs.get("rank"), kwargs.get("device_mesh"))
                                else:
                                    _load_parameter_into_model(model, param_name, param_value)
                            else:
                                raise ValueError("Something went horribly wrong mate in gate_up_proj")

                            dequantized_gate_up_proj = convert_moe_packed_tensors(module.gate_up_proj_blocks, module.gate_up_proj_scales)
                            dequantized_gate_up_proj = dequantized_gate_up_proj.transpose(1,2).to(target_device)

                            module.device_mesh = kwargs.get("device_mesh")
                            module.rank = kwargs.get("rank")

                            right_pad = module.gate_up_proj_right_pad
                            bottom_pad = module.gate_up_proj_bottom_pad
                            loaded_weight = torch.nn.functional.pad(dequantized_gate_up_proj,
                                                    (0, right_pad, 0, bottom_pad, 0, 0),
                                                    mode="constant",
                                                    value=0)
                            del dequantized_gate_up_proj
                            torch.cuda.empty_cache()
                            with torch.cuda.device(target_device):
                                loaded_weight, flex, mx = quantize_to_mxfp4(loaded_weight, self.swizzle_mx_value, self.swizzle_mx_scale)
                            module.gate_up_proj_precision_config = PrecisionConfig(mx_ctx=mx, flex_ctx=FlexCtx(rhs_data=flex))
                            module.gate_up_proj = torch.nn.Parameter(loaded_weight, requires_grad=False)

                    elif "down_proj" in param_name:
                        if module.down_proj_blocks.device.type == "meta" and module.down_proj_scales.device.type == "meta":
                            if tp_mode:
                                shard_and_distribute_module(model, param_value, kwargs.get("empty_param"), param_name, kwargs.get("casting_dtype"), kwargs.get("to_contiguous"), kwargs.get("rank"), kwargs.get("device_mesh"))
                            else:
                                _load_parameter_into_model(model, param_name, param_value)
                            return
                        else:
                            if (module.down_proj_blocks.device != "meta" and "scales" in param_name) or (module.down_proj_scales.device != "meta" and "blocks" in param_name):
                                if tp_mode:
                                    shard_and_distribute_module(model, param_value, kwargs.get("empty_param"), param_name, kwargs.get("casting_dtype"), kwargs.get("to_contiguous"), kwargs.get("rank"), kwargs.get("device_mesh"))
                                else:
                                    _load_parameter_into_model(model, param_name, param_value)
                            else:
                                raise ValueError("Something went horribly wrong mate in down_proj")

                            dequantized_down_proj = convert_moe_packed_tensors(module.down_proj_blocks, module.down_proj_scales)
                            dequantized_down_proj = dequantized_down_proj.transpose(1,2).to(target_device)
                            module.device_mesh = kwargs.get("device_mesh")
                            module.rank = kwargs.get("rank")

                            right_pad = module.down_proj_right_pad
                            bottom_pad = module.down_proj_bottom_pad
                            loaded_weight = torch.nn.functional.pad(dequantized_down_proj,
                                                    (0, right_pad, 0, bottom_pad, 0, 0),
                                                    mode="constant",
                                                    value=0)
                            del dequantized_down_proj
                            torch.cuda.empty_cache()
                            with torch.cuda.device(target_device):
                                loaded_weight, flex, mx = quantize_to_mxfp4(loaded_weight, self.swizzle_mx_value, self.swizzle_mx_scale)

                            module.down_proj_precision_config = PrecisionConfig(mx_ctx=mx, flex_ctx=FlexCtx(rhs_data=flex))
                            module.down_proj = torch.nn.Parameter(loaded_weight, requires_grad=False)

    def _process_model_after_weight_loading(self, model: "PreTrainedModel", **kwargs):
        return model

    def update_expected_keys(self, model: "PreTrainedModel", expected_keys: list[str], checkpoint_keys: list[str]):
        # Replace expected_keys for experts' gate_up_proj and down_proj with their _blocks and _scales variants
        new_expected_keys = []
        for key in expected_keys:
            if key.endswith(".mlp.experts.gate_up_proj"):
                base = key[:-len("gate_up_proj")]
                new_expected_keys.append(base + "gate_up_proj_blocks")
                new_expected_keys.append(base + "gate_up_proj_scales")
            elif key.endswith(".mlp.experts.down_proj"):
                base = key[:-len("down_proj")]
                new_expected_keys.append(base + "down_proj_blocks")
                new_expected_keys.append(base + "down_proj_scales")
            else:
                new_expected_keys.append(key)
        return new_expected_keys

    def _process_model_before_weight_loading(
        self,
        model: "PreTrainedModel",
        keep_in_fp32_modules: Optional[list[str]] = None,
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

    def update_missing_keys(self, model, missing_keys: list[str], prefix: str) -> list[str]:
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
        config.base_model_tp_plan = {
        # "embed_tokens": "vocab_parallel_rowwise",
            "layers.*.self_attn.q_proj": "colwise",
            "layers.*.self_attn.k_proj": "colwise",
            "layers.*.self_attn.v_proj": "colwise",
            "layers.*.self_attn.o_proj": "rowwise",
            "layers.*.self_attn.sinks": "local_rowwise",
            "layers.*.mlp.experts.gate_up_proj": "local_packed_rowwise",
            "layers.*.mlp.experts.gate_up_proj_blocks": "local_packed_rowwise",
            "layers.*.mlp.experts.gate_up_proj_scales": "local_packed_rowwise",
            "layers.*.mlp.experts.gate_up_proj_bias": "local_packed_rowwise",
            "layers.*.mlp.experts.down_proj": "local_colwise",
            "layers.*.mlp.experts.down_proj_blocks": "local_colwise",
            "layers.*.mlp.experts.down_proj_scales": "local_colwise",
            "layers.*.mlp.experts.down_proj_bias": "local_colwise",
            # "layers.*.mlp": "gather",
        }
        config.base_model_ep_plan = {
            "layers.*.self_attn.q_proj": "colwise",
            "layers.*.self_attn.k_proj": "colwise",
            "layers.*.self_attn.v_proj": "colwise",
            "layers.*.self_attn.o_proj": "rowwise",
            "layers.*.self_attn.sinks": "local_rowwise",
            "layers.*.mlp.experts": "gather",
            "layers.*.mlp.router": "ep_router",
            "layers.*.mlp.experts.gate_up_proj": "grouped_gemm",
            "layers.*.mlp.experts.gate_up_proj_blocks": "grouped_gemm",
            "layers.*.mlp.experts.gate_up_proj_scales": "grouped_gemm",
            "layers.*.mlp.experts.gate_up_proj_bias": "grouped_gemm",
            "layers.*.mlp.experts.down_proj": "grouped_gemm",
            "layers.*.mlp.experts.down_proj_blocks": "grouped_gemm",
            "layers.*.mlp.experts.down_proj_scales": "grouped_gemm",
            "layers.*.mlp.experts.down_proj_bias": "grouped_gemm",
        }

        return config

    def is_serializable(self, safe_serialization=None):
        return False

    @property
    def is_trainable(self) -> bool:
        return False
