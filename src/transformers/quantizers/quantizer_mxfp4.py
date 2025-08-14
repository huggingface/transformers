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
    is_kernels_available,
    is_torch_available,
    is_triton_available,
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

        if self.quantization_config.dequantize:
            return

        if not torch.cuda.is_available():
            if self.pre_quantized:
                logger.warning_once(
                    "Using MXFP4 quantized models requires a GPU, we will default to dequantizing the model to bf16"
                )
                self.quantization_config.dequantize = True
                return
            else:
                raise RuntimeError("Quantizing a model using MXFP4 requires a GPU")

        if not is_accelerate_available():
            raise ImportError("Using mxfp4 requires Accelerate: `pip install accelerate`")

        compute_capability = torch.cuda.get_device_capability()
        gpu_is_supported = compute_capability >= (7, 5)
        kernels_available = is_triton_available("3.4.0") and is_kernels_available()

        if self.pre_quantized:
            # On unsupported GPUs or without kernels, we will dequantize the model to bf16
            if not gpu_is_supported:
                logger.warning_once(
                    "MXFP4 quantization is only supported on GPUs with compute capability >= 7.5 (e.g T4, A100, L4, H100, or B200). "
                    "We will default to dequantizing the model to bf16."
                )
                self.quantization_config.dequantize = True
                return

            if not kernels_available:
                logger.warning_once(
                    "MXFP4 quantization requires triton >= 3.4.0 and kernels installed, we will default to dequantizing the model to bf16"
                )
                self.quantization_config.dequantize = True
                return
        elif not gpu_is_supported:
            # we can't quantize the model in this case so we raise an error
            raise ValueError(
                "MXFP4 quantization is only supported on GPUs with compute capability >= 7.5 (e.g T4, A100, L4, H100, or B200)"
            )
        elif not kernels_available:
            # we can't quantize the model in this case so we raise an error
            raise ValueError("MXFP4 quantization requires triton >= 3.4.0 and triton_kernels installed")

        if not self.pre_quantized:
            from kernels import get_kernel

            global triton_kernels_hub
            triton_kernels_hub = get_kernel("kernels-community/triton_kernels")

        device_map = kwargs.get("device_map")
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

    def update_dtype(self, dtype: "torch.dtype") -> "torch.dtype":
        if dtype is None:
            dtype = torch.bfloat16
            logger.info(
                "Overriding dtype=%s with `dtype=torch.bfloat16` due to "
                "requirements of `fbgemm-gpu` to enable model loading in fp4. "
                "Pass your own dtype to specify the dtype of the remaining non-linear layers or pass"
                " dtype=torch.bfloat16 to remove this warning.",
                dtype,
            )
        return dtype

    def check_quantized_param(
        self,
        model: "PreTrainedModel",
        param_value: "torch.Tensor",
        param_name: str,
        state_dict: dict[str, Any],
        **kwargs,
    ):
        from ..integrations import Mxfp4GptOssExperts
        from ..models.gpt_oss.modeling_gpt_oss import GptOssExperts

        # if we are dequantizing, the model doesn't have scales, and blocks only params like gate_up_proj and down_proj so we need to handle this case differently
        if self.quantization_config.dequantize and ("blocks" in param_name or "scales" in param_name):
            module, tensor_name = get_module_from_name(model, param_name[: -len("_blocks")])
        else:
            module, tensor_name = get_module_from_name(model, param_name)

        if isinstance(module, Mxfp4GptOssExperts) or (
            isinstance(module, GptOssExperts) and self.quantization_config.dequantize
        ):
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
        from ..integrations import Mxfp4GptOssExperts, dequantize, load_and_swizzle_mxfp4, quantize_to_mxfp4
        from ..models.gpt_oss.modeling_gpt_oss import GptOssExperts

        if not self.pre_quantized:
            PrecisionConfig, FlexCtx, InFlexData = (
                triton_kernels_hub.matmul_ogs.PrecisionConfig,
                triton_kernels_hub.matmul_ogs.FlexCtx,
                triton_kernels_hub.matmul_ogs.InFlexData,
            )
            module, _ = get_module_from_name(model, param_name)
            with torch.cuda.device(target_device):
                if isinstance(module, Mxfp4GptOssExperts):
                    if "gate_up_proj" in param_name:
                        right_pad = module.gate_up_proj_right_pad
                        bottom_pad = module.gate_up_proj_bottom_pad
                        loaded_weight = torch.nn.functional.pad(
                            param_value, (0, right_pad, 0, bottom_pad, 0, 0), mode="constant", value=0
                        )
                        triton_weight_tensor, weight_scale = quantize_to_mxfp4(loaded_weight)
                        module.gate_up_proj_precision_config = PrecisionConfig(
                            weight_scale=weight_scale, flex_ctx=FlexCtx(rhs_data=InFlexData())
                        )
                        module.gate_up_proj = triton_weight_tensor
                        module.gate_up_proj_blocks = torch.nn.Parameter(
                            triton_weight_tensor.storage.data, requires_grad=False
                        )
                    elif "down_proj" in param_name:
                        right_pad = module.down_proj_right_pad
                        bottom_pad = module.down_proj_bottom_pad
                        loaded_weight = torch.nn.functional.pad(
                            param_value, (0, right_pad, 0, bottom_pad, 0, 0), mode="constant", value=0
                        ).to(target_device)
                        triton_weight_tensor, weight_scale = quantize_to_mxfp4(loaded_weight)
                        module.down_proj_precision_config = PrecisionConfig(
                            weight_scale=weight_scale, flex_ctx=FlexCtx(rhs_data=InFlexData())
                        )
                        module.down_proj = triton_weight_tensor
                        module.down_proj_blocks = torch.nn.Parameter(
                            triton_weight_tensor.storage.data, requires_grad=False
                        )

        # we take this path if already quantized but not in a compatible way
        # The params going here are either gate_up_proj_blocks, or down_proj_blocks, or gate_up_proj_scales, or down_proj_scales
        else:
            empty_param = kwargs.get("empty_param")
            casting_dtype = kwargs.get("casting_dtype")
            to_contiguous = kwargs.get("to_contiguous")
            rank = kwargs.get("rank")
            device_mesh = kwargs.get("device_mesh")
            if ("blocks" in param_name or "scales" in param_name) and self.quantization_config.dequantize:
                # blocks and scales have the same length that's this works for both
                module, _ = get_module_from_name(model, param_name[: -len("_blocks")])
            else:
                module, _ = get_module_from_name(model, param_name)

            shard_kwargs = {
                "empty_param": empty_param,
                "casting_dtype": casting_dtype,
                "to_contiguous": to_contiguous,
                "rank": rank,
                "device_mesh": device_mesh,
                "model": model,
            }

            if isinstance(module, Mxfp4GptOssExperts) or (
                isinstance(module, GptOssExperts) and self.quantization_config.dequantize
            ):
                if self.quantization_config.dequantize:
                    # dq_param_name is the name of the parameter without the blocks or scales suffix, it's used in this case since we don't switch linears
                    # so we only have the original param name
                    dq_param_name = param_name[: -len("_blocks")]
                    dequantize(module, param_name, param_value, target_device, dq_param_name, **shard_kwargs)
                else:
                    load_and_swizzle_mxfp4(
                        module,
                        param_name,
                        param_value,
                        target_device,
                        **shard_kwargs,
                    )

    def _process_model_after_weight_loading(self, model: "PreTrainedModel", **kwargs):
        # we are not really dequantizing, we are just removing everthing related to quantization here
        if self.quantization_config.dequantize:
            self.remove_quantization_config(model)
        # clean cache due to triton ops
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def update_expected_keys(self, model: "PreTrainedModel", expected_keys: list[str], checkpoint_keys: list[str]):
        # Replace expected_keys for experts' gate_up_proj and down_proj with their _blocks and _scales variants
        new_expected_keys = []
        for key in expected_keys:
            if key.endswith(".mlp.experts.gate_up_proj"):
                base = key[: -len("gate_up_proj")]
                new_expected_keys.append(base + "gate_up_proj_blocks")
                new_expected_keys.append(base + "gate_up_proj_scales")
            elif key.endswith(".mlp.experts.down_proj"):
                base = key[: -len("down_proj")]
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

        self.modules_to_not_convert = self.get_modules_to_not_convert(
            model, self.quantization_config.modules_to_not_convert, keep_in_fp32_modules
        )

        use_kernels = kwargs.get("use_kernels", False)
        # if we are using kernels, we can't use the quantized model, since the forward pass is different and needs special handling
        if use_kernels:
            logger.warning_once(
                "You are using full precision kernels, we will dequantize the model to bf16. "
                "To use the quantized model with quantization kernels, please set use_kernels=False"
            )
            self.quantization_config.dequantize = True

        config = model.config
        model = replace_with_mxfp4_linear(
            model,
            modules_to_not_convert=self.modules_to_not_convert,
            quantization_config=self.quantization_config,
            config=config,
        )

        model.config.quantization_config = self.quantization_config

    def update_missing_keys(self, model, missing_keys: list[str], prefix: str) -> list[str]:
        from ..integrations import Mxfp4GptOssExperts

        not_missing_keys = []
        for name, module in model.named_modules():
            if isinstance(module, Mxfp4GptOssExperts):
                for missing in missing_keys:
                    if (
                        (name in missing or name in f"{prefix}.{missing}")
                        and not missing.endswith(".weight")
                        and not missing.endswith(".bias")
                    ):
                        not_missing_keys.append(missing)
        return [k for k in missing_keys if k not in not_missing_keys]

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

    def update_param_name(self, param_name: str) -> str:
        if self.quantization_config.dequantize:
            if "_blocks" in param_name:
                return param_name.replace("_blocks", "")
            elif "_scales" in param_name:
                return param_name.replace("_scales", "")
        return param_name

    def is_serializable(self, safe_serialization=None):
        logger.warning_once("MXFP4 quantization is not serializable using safetensors for now")
        return False

    @property
    def is_trainable(self) -> bool:
        logger.warning_once(
            "MXFP4 quantization don't support training, please consider dequantizing the model first by passing quantization_config=Mxfp4Config(dequantize=True) to .from_pretrained()"
        )
        return False
