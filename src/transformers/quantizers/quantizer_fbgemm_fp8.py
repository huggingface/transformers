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

from ..utils import is_accelerate_available, is_fbgemm_gpu_available, is_torch_available, logging
from .quantizers_utils import get_module_from_name


if is_torch_available():
    import torch


logger = logging.get_logger(__name__)


class FbgemmFp8HfQuantizer(HfQuantizer):
    """
    FP8 quantization using fbgemm kernels
    """

    requires_calibration = False

    def __init__(self, quantization_config, **kwargs):
        super().__init__(quantization_config, **kwargs)

    def validate_environment(self, *args, **kwargs):
        if not is_fbgemm_gpu_available():
            raise ImportError(
                "Using fbgemm fp8 quantization requires fbgemm-gpu library"
                "Please install the latest version of fbgemm-gpu library by following : https://pytorch.org/FBGEMM/fbgemm_gpu-development/InstallationInstructions.html#fbgemm-gpu-install-libraries"
            )
        if not is_accelerate_available():
            raise ImportError(
                "Loading an FP8 quantized model requires accelerate (`pip install --upgrade accelerate`)"
            )
        compute_capability = torch.cuda.get_device_capability()
        major, _ = compute_capability
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
        elif isinstance(device_map, dict):
            if not self.pre_quantized and ("cpu" in device_map.values() or "disk" in device_map.values()):
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
                "You cannot use FP8 with dtype=torch.float16. We recommend you passing dtype=torch.bfloat16"
            )
        return dtype

    def param_needs_quantization(self, model: "PreTrainedModel", param_name: str, **kwargs) -> bool:
        from ..integrations import FbgemmFp8Linear, FbgemmFp8Llama4TextExperts

        module, tensor_name = get_module_from_name(model, param_name)

        if isinstance(module, FbgemmFp8Linear):
            if self.pre_quantized or tensor_name == "bias":
                return False
            else:
                return True
        if isinstance(module, FbgemmFp8Llama4TextExperts):
            if self.pre_quantized or tensor_name == "bias":
                return False
            else:
                return True
        return False

    def _process_model_before_weight_loading(
        self,
        model: "PreTrainedModel",
        keep_in_fp32_modules: list[str] | None = None,
        **kwargs,
    ):
        from ..integrations import replace_with_fbgemm_fp8_linear

        self.modules_to_not_convert = self.get_modules_to_not_convert(
            model, self.quantization_config.modules_to_not_convert, keep_in_fp32_modules
        )

        model = replace_with_fbgemm_fp8_linear(
            model,
            modules_to_not_convert=self.modules_to_not_convert,
            quantization_config=self.quantization_config,
            pre_quantized=self.pre_quantized,
            config=model.config,
            tp_plan=model._tp_plan,
        )

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

    def is_serializable(self):
        return True

    @property
    def is_trainable(self) -> bool:
        return False

    def get_quantize_ops(self):
        from ..integrations.fbgemm_fp8 import FbgemmFp8Quantize

        return FbgemmFp8Quantize(self)
