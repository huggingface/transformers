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
from typing import TYPE_CHECKING, Any, Optional

from .base import HfQuantizer


if TYPE_CHECKING:
    from ..modeling_utils import PreTrainedModel

from ..utils import is_accelerate_available, is_fbgemm_gpu_available, is_torch_available, logging
from .quantizers_utils import get_module_from_name


if is_torch_available():
    import torch


logger = logging.get_logger(__name__)


class FbgemmFp8HfQuantizer(HfQuantizer):
    """
    FP8 quantization using fbgemm kernels
    """

    requires_parameters_quantization = True
    requires_calibration = False

    required_packages = ["fbgemm-gpu", "accelerate"]

    def __init__(self, quantization_config, **kwargs):
        super().__init__(quantization_config, **kwargs)
        self.quantization_config = quantization_config

    def validate_environment(self, *args, **kwargs):
        if not is_torch_available():
            raise ImportError(
                "Using fbgemm fp8 quantization requires torch >= 2.1.0"
                "Please install the latest version of torch ( pip install --upgrade torch )"
            )
        if not is_fbgemm_gpu_available():
            raise ImportError(
                "Using fbgemm fp8 quantization requires fbgemm-gpu library"
                "Please install the latest version of fbgemm-gpu library by following : https://pytorch.org/FBGEMM/fbgemm_gpu-development/InstallationInstructions.html#fbgemm-gpu-install-libraries"
            )

        if not is_accelerate_available("0.32.2"):
            raise ImportError(
                "Loading an FP8 quantized model requires accelerate > 0.32.1 (`pip install --upgrade accelerate`)"
            )

        if not torch.cuda.is_available():
            raise RuntimeError("Using FP8 quantized models with fbgemm kernels requires a GPU")

        compute_capability = torch.cuda.get_device_capability()
        major, minor = compute_capability
        if major < 9:
            raise ValueError(
                "FP8 quantized models is only supported on GPUs with compute capability >= 9.0 (e.g H100)"
            )

        device_map = kwargs.get("device_map")
        if device_map is None:
            logger.warning_once(
                "You have loaded an FP8 model on CPU and have a CUDA device available, make sure to set "
                "your model on a GPU device in order to run your model. To remove this warning, pass device_map = 'cuda'. "
            )
        elif device_map is not None:
            if (
                not self.pre_quantized
                and isinstance(device_map, dict)
                and ("cpu" in device_map.values() or "disk" in device_map.values())
            ):
                raise ValueError(
                    "You are attempting to load an FP8 model with a device_map that contains a CPU or disk device."
                    "This is not supported when the model is quantized on the fly. "
                    "Please use a quantized checkpoint or remove the CPU or disk device from the device_map."
                )

    def update_dtype(self, dtype: "torch.dtype") -> "torch.dtype":
        if dtype is None:
            dtype = torch.bfloat16
            logger.info(
                "Overriding dtype=%s with `dtype=torch.bloat16` due to "
                "requirements of `fbgemm-gpu` to enable model loading in fp8. "
                "Pass your own dtype to specify the dtype of the remaining non-linear layers or pass"
                " dtype=torch.bfloat16 to remove this warning.",
                dtype,
            )
        elif dtype == torch.float16:
            raise ValueError(
                "You cannot use FP8 with dtype=torch.float16.We recommend you passing dtype=torch.bfloat16"
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
        from ..integrations import FbgemmFp8Linear, FbgemmFp8Llama4TextExperts

        module, tensor_name = get_module_from_name(model, param_name)

        if isinstance(module, FbgemmFp8Linear):
            if self.pre_quantized or tensor_name == "bias":
                if tensor_name == "weight" and param_value.dtype != torch.float8_e4m3fn:
                    raise ValueError("Expect quantized weights but got an unquantized weight")
                return False
            else:
                if tensor_name == "weight_scale":
                    raise ValueError("Expect unquantized weights but got a quantized weight_scale")
                return True
        if isinstance(module, FbgemmFp8Llama4TextExperts):
            if self.pre_quantized or tensor_name == "bias":
                return False
            else:
                if tensor_name == "gate_up_proj_scale" or tensor_name == "down_proj_scale":
                    raise ValueError("Expect unquantized weights but got a quantized weight_scale")
                return True
        return False

    def create_quantized_param(
        self,
        model: "PreTrainedModel",
        param_value: "torch.Tensor",
        param_name: str,
        target_device: "torch.device",
        state_dict: dict[str, Any],
    ):
        """
        Quantizes weights into weight and weight_scale
        """

        from ..integrations import FbgemmFp8Llama4TextExperts

        module, tensor_name = get_module_from_name(model, param_name)
        if isinstance(module, FbgemmFp8Llama4TextExperts):
            if tensor_name == "gate_up_proj":
                # Process each expert separately
                # Transpose the second and third dimension
                transposed_param = param_value.transpose(1, 2)

                # Reshape to 2D for quantization
                original_shape = transposed_param.shape
                flattened_param = transposed_param.reshape(-1, original_shape[-1])

                # Quantize using per row instead of per column
                new_value_flat, weight_scale_flat = torch.ops.fbgemm.quantize_fp8_per_row(flattened_param)

                # Reshape back to original dimensions
                new_value = new_value_flat.reshape(original_shape)
                new_value = new_value.transpose(1, 2)
                weight_scale = weight_scale_flat.reshape(original_shape[0], 1, original_shape[1])
            elif tensor_name == "down_proj":
                # Process each expert separately
                # Transpose the weights for proper quantization
                transposed_param = param_value.transpose(1, 2)

                # Reshape to 2D for quantization
                original_shape = transposed_param.shape
                flattened_param = transposed_param.reshape(-1, original_shape[-1])

                # Quantize using per column
                new_value_flat, weight_scale_flat = torch.ops.fbgemm.quantize_fp8_per_row(flattened_param)

                # Reshape back to original dimensions
                new_value = new_value_flat.reshape(original_shape)
                new_value = new_value.transpose(1, 2)
                weight_scale = weight_scale_flat.reshape(original_shape[0], original_shape[1], 1)

            module._parameters[f"{tensor_name}_scale"] = torch.nn.Parameter(weight_scale.to(target_device))
        else:
            new_value, weight_scale = torch.ops.fbgemm.quantize_fp8_per_row(param_value)
            module._parameters[f"{tensor_name}_scale"] = torch.nn.Parameter(
                weight_scale.view(weight_scale.shape[0], 1).to(target_device)
            )

        module._parameters[tensor_name] = torch.nn.Parameter(new_value.to(target_device))

        del param_name

    def _process_model_after_weight_loading(self, model: "PreTrainedModel", **kwargs):
        return model

    def _process_model_before_weight_loading(
        self,
        model: "PreTrainedModel",
        keep_in_fp32_modules: Optional[list[str]] = None,
        **kwargs,
    ):
        from ..integrations import replace_with_fbgemm_fp8_linear

        tp_plan = model._tp_plan
        self.modules_to_not_convert = self.get_modules_to_not_convert(
            model, self.quantization_config.modules_to_not_convert, keep_in_fp32_modules
        )

        config = model.config
        model = replace_with_fbgemm_fp8_linear(
            model,
            modules_to_not_convert=self.modules_to_not_convert,
            quantization_config=self.quantization_config,
            pre_quantized=self.pre_quantized,
            config=config,
            tp_plan=tp_plan,
        )

        model.config.quantization_config = self.quantization_config

    def update_missing_keys(self, model, missing_keys: list[str], prefix: str) -> list[str]:
        from ..integrations import FbgemmFp8Linear, FbgemmFp8Llama4TextExperts

        not_missing_keys = []
        for name, module in model.named_modules():
            if isinstance(module, (FbgemmFp8Linear, FbgemmFp8Llama4TextExperts)):
                for missing in missing_keys:
                    if (
                        (name in missing or name in f"{prefix}.{missing}")
                        and not missing.endswith(".weight")
                        and not missing.endswith(".bias")
                    ):
                        not_missing_keys.append(missing)
        return [k for k in missing_keys if k not in not_missing_keys]

    def update_tp_plan(self, config):
        if "Llama4" in config.__class__.__name__:
            text_plan = {
                # We are using a different tp plan with local_colwise and local_rowwise for the attention because fbgemm operations cannot be parallelized
                # With local_colwise and local_rowwise, all the operations are done locally, and we add a gather operation to gather the results instead of
                # using dtensors
                "layers.*.self_attn.q_proj.weight": "local_colwise",
                "layers.*.self_attn.q_proj.weight_scale": "local_colwise",
                "layers.*.self_attn.k_proj.weight": "local_colwise",
                "layers.*.self_attn.k_proj.weight_scale": "local_colwise",
                "layers.*.self_attn.v_proj.weight": "local_colwise",
                "layers.*.self_attn.v_proj.weight_scale": "local_colwise",
                "layers.*.self_attn.o_proj.weight": "local_rowwise",
                "layers.*.self_attn": "gather",
                # We keep the same sequence_parallel plan for layernorms
                "layers.*.input_layernorm.weight": "sequence_parallel",
                "layers.*.post_attention_layernorm.weight": "sequence_parallel",
                "norm.weight": "sequence_parallel",
                # We keep the same local_colwise and local_rowwise plan for the feed forward shared expert
                # We also add scales for the shared expert, for local_colwise the scale is also local_colwise
                # For local_rowwise the scale is replicated, so we don't need to add it
                "layers.*.feed_forward.shared_expert.gate_proj.weight": "local_colwise",
                "layers.*.feed_forward.shared_expert.gate_proj.weight_scale": "local_colwise",
                "layers.*.feed_forward.shared_expert.up_proj.weight": "local_colwise",
                "layers.*.feed_forward.shared_expert.up_proj.weight_scale": "local_colwise",
                "layers.*.feed_forward.shared_expert.down_proj.weight": "local_rowwise",
                "layers.*.feed_forward.experts": "local",
                "layers.*.feed_forward": "gather",
                "layers.*.feed_forward.experts.*.gate_proj.weight": "local_colwise",
                "layers.*.feed_forward.experts.*.gate_proj.weight_scale": "local_colwise",
                "layers.*.feed_forward.experts.*.up_proj.weight": "local_colwise",
                "layers.*.feed_forward.experts.*.up_proj.weight_scale": "local_colwise",
                "layers.*.feed_forward.experts.*.down_proj.weight": "local_rowwise",
                # For Fused implementation we use local_packed_rowwise for the gate_up_proj, and the same for the packed scales
                # We use local_colwise for the down_proj, and the scales are replicated so we don't add them
                "layers.*.feed_forward.experts.gate_up_proj": "local_packed_rowwise",
                "layers.*.feed_forward.experts.gate_up_proj_scale": "local_packed_rowwise",
                "layers.*.feed_forward.experts.down_proj": "local_colwise",
            }
            if config.get_text_config() is not None:
                config.get_text_config().base_model_tp_plan = text_plan
            else:
                config.base_model_tp_plan = text_plan
            return config

        return config

    def is_serializable(self, safe_serialization=None):
        return True

    @property
    def is_trainable(self) -> bool:
        return False
