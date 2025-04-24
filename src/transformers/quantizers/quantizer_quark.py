# coding=utf-8
# Copyright 2025 Advanced Micro Devices, Inc. and The HuggingFace Inc. team. All rights reserved.
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

from typing import TYPE_CHECKING, Any, Dict

from ..file_utils import is_torch_available
from .base import HfQuantizer


if TYPE_CHECKING:
    from ..modeling_utils import PreTrainedModel

    if is_torch_available():
        import torch

from ..utils import is_accelerate_available, is_quark_available, logging


if is_accelerate_available():
    from accelerate.utils import set_module_tensor_to_device

logger = logging.get_logger(__name__)


CHECKPOINT_KEYS = {
    "weight_scale": "weight_quantizer.scale",
    "bias_scale": "bias_quantizer.scale",
    "input_scale": "input_quantizer.scale",
    "output_scale": "output_quantizer.scale",
    "weight_zero_point": "weight_quantizer.zero_point",
    "bias_zero_point": "bias_quantizer.zero_point",
    "input_zero_point": "input_quantizer.zero_point",
    "output_zero_point": "output_quantizer.zero_point",
}


class QuarkHfQuantizer(HfQuantizer):
    """
    Quark quantizer (https://quark.docs.amd.com/latest/).
    """

    requires_calibration = True  # On-the-fly quantization with quark is not supported for now.
    required_packages = ["quark"]

    # Checkpoints are expected to be already quantized when loading a quark model. However, as some keys from
    # the checkpoint might mismatch the model parameters keys, we use the `create_quantized_param` method
    # to load the checkpoints, remapping the keys.
    requires_parameters_quantization = True

    def __init__(self, quantization_config, **kwargs):
        super().__init__(quantization_config, **kwargs)

        self.json_export_config = quantization_config.json_export_config

    def validate_environment(self, *args, **kwargs):
        if not is_quark_available():
            raise ImportError(
                "Loading a Quark quantized model requires the `quark` library but it was not found in the environment. Please refer to https://quark.docs.amd.com/latest/install.html."
            )

    def _process_model_before_weight_loading(self, model: "PreTrainedModel", **kwargs):
        from quark.torch.export.api import _map_to_quark

        _map_to_quark(
            model,
            self.quantization_config.quant_config,
            pack_method=self.json_export_config.pack_method,
            custom_mode=self.quantization_config.custom_mode,
        )

        return model

    def check_quantized_param(
        self,
        model: "PreTrainedModel",
        param_value: "torch.Tensor",
        param_name: str,
        state_dict: Dict[str, Any],
        **kwargs,
    ) -> bool:
        return True

    def create_quantized_param(
        self, model, param, param_name, param_device, state_dict, unexpected_keys
    ) -> "torch.nn.Parameter":
        postfix = param_name.split(".")[-1]

        if postfix in CHECKPOINT_KEYS:
            param_name = param_name.replace(postfix, CHECKPOINT_KEYS[postfix])

        set_module_tensor_to_device(model, param_name, param_device, value=param)

    def _process_model_after_weight_loading(self, model: "PreTrainedModel", **kwargs):
        return model

    def is_serializable(self, safe_serialization=None):
        return False

    @property
    def is_trainable(self):
        return False

    def update_tp_plan(self, config):
        if "Llama" in config.__class__.__name__:
            text_plan = config.base_model_tp_plan

            if config.get_text_config() is not None:
               text_plan = config.get_text_config().base_model_tp_plan

            # modify exsiting play
            text_plan_cp = {}
            for key in text_plan:
                text_plan_cp[key+".weight"] =  text_plan[key]
            text_plan = text_plan_cp

            # create a plan with missing values from base plan
            update_plan ={
                    "layers.*.self_attn.q_proj.weight_scale": "sequence_parallel",
                    "layers.*.self_attn.k_proj.weight_scale": "sequence_parallel",
                    "layers.*.self_attn.v_proj.weight_scale": "sequence_parallel",
                    "layers.*.self_attn.o_proj.weight_scale": "sequence_parallel",

                    "layers.*.self_attn.q_proj.input_scale": "sequence_parallel",
                    "layers.*.self_attn.k_proj.input_scale": "sequence_parallel",
                    "layers.*.self_attn.v_proj.input_scale": "sequence_parallel",
                    "layers.*.self_attn.o_proj.input_scale": "sequence_parallel",

                    "layers.*.self_attn.q_proj.weight_zero_point": "sequence_parallel",
                    "layers.*.self_attn.k_proj.weight_zero_point": "sequence_parallel",
                    "layers.*.self_attn.v_proj.weight_zero_point": "sequence_parallel",
                    "layers.*.self_attn.o_proj.weight_zero_point": "sequence_parallel",

                    "layers.*.self_attn.q_proj.input_zero_point": "sequence_parallel",
                    "layers.*.self_attn.k_proj.input_zero_point": "sequence_parallel",
                    "layers.*.self_attn.v_proj.input_zero_point": "sequence_parallel",
                    "layers.*.self_attn.o_proj.input_zero_point": "sequence_parallel",

                    "layers.*.mlp.gate_proj.weight_scale": "sequence_parallel",
                    "layers.*.mlp.up_proj.weight_scale": "sequence_parallel",
                    "layers.*.mlp.down_proj.weight_scale": "sequence_parallel",
                    "layers.*.mlp.gate_proj.input_scale": "sequence_parallel",
                    "layers.*.mlp.up_proj.input_scale": "sequence_parallel",
                    "layers.*.mlp.down_proj.input_scale": "sequence_parallel",

                    "layers.*.mlp.gate_proj.weight_zero_point": "sequence_parallel",
                    "layers.*.mlp.up_proj.weight_zero_point": "sequence_parallel",
                    "layers.*.mlp.down_proj.weight_zero_point": "sequence_parallel",
                    "layers.*.mlp.gate_proj.input_zero_point": "sequence_parallel",
                    "layers.*.mlp.up_proj.input_zero_point": "sequence_parallel",
                    "layers.*.mlp.down_proj.input_zero_point": "sequence_parallel",
                    }
            # combine the plans
            text_plan.update(update_plan)

        if text_plan is not None:
            if config.get_text_config() is not None:
                config.get_text_config().base_model_tp_plan = text_plan
            else:
                config.base_model_tp_plan = text_plan
            return config

        return config