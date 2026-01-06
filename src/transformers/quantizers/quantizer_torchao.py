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
import importlib
import re
from typing import TYPE_CHECKING

from packaging import version

from .base import HfQuantizer
from .quantizers_utils import get_module_from_name, should_convert_module


if TYPE_CHECKING:
    from ..modeling_utils import PreTrainedModel

from safetensors import safe_open

from ..utils import is_torch_available, is_torchao_available, logging


if is_torch_available():
    from ..core_model_loading import WeightConverter


if is_torch_available():
    import torch

if is_torchao_available():
    if version.parse(importlib.metadata.version("torchao")) >= version.parse("0.15.0"):
        from torchao.prototype.safetensors.safetensors_support import (
            flatten_tensor_state_dict,
        )


logger = logging.get_logger(__name__)


def fuzzy_match_size(config_name: str) -> str | None:
    """
    Extract the size digit from strings like "4weight", "8weight".
    Returns the digit as an integer if found, otherwise None.
    """
    config_name = config_name.lower()

    str_match = re.search(r"(\d)weight", config_name)

    if str_match:
        return str_match.group(1)

    return None


def _quantization_type(weight):
    from torchao.dtypes import AffineQuantizedTensor
    from torchao.quantization.linear_activation_quantized_tensor import LinearActivationQuantizedTensor

    if isinstance(weight, AffineQuantizedTensor):
        return f"{weight.__class__.__name__}({weight._quantization_type()})"

    if isinstance(weight, LinearActivationQuantizedTensor):
        return f"{weight.__class__.__name__}(activation={weight.input_quant_func}, weight={_quantization_type(weight.original_weight_tensor)})"


def _linear_extra_repr(self):
    weight = _quantization_type(self.weight)
    if weight is None:
        return f"in_features={self.weight.shape[1]}, out_features={self.weight.shape[0]}, weight=None"
    else:
        return f"in_features={self.weight.shape[1]}, out_features={self.weight.shape[0]}, weight={weight}"


if is_torchao_available():
    TORCHAO_VERSION = version.parse(importlib.metadata.version("torchao"))


class TorchAoHfQuantizer(HfQuantizer):
    """
    Quantizer for torchao: https://github.com/pytorch/ao/
    """

    requires_calibration = False

    def __init__(self, quantization_config, **kwargs):
        super().__init__(quantization_config, **kwargs)

        self.quantized_param_size = None
        quant_type = self.quantization_config.quant_type
        if isinstance(quant_type, str):
            map_to_param_size = {
                "int4_weight_only": 0.5,
                "int8_weight_only": 1,
                "int8_dynamic_activation_int8_weight": 1,
            }
            if quant_type in map_to_param_size:
                self.quantized_param_size = map_to_param_size[quant_type]
        else:
            size_digit = fuzzy_match_size(quant_type.__class__.__name__)
            self.quantized_param_size = 0.5 if size_digit == "4" else 1

    def validate_environment(self, *args, **kwargs):
        if not is_torchao_available():
            raise ImportError("Loading an torchao quantized model requires torchao library (`pip install torchao`)")

        self.offload = False
        device_map = kwargs.get("device_map")
        if isinstance(device_map, dict):
            if ("disk" in device_map.values() or "cpu" in device_map.values()) and len(device_map) > 1:
                self.offload = True
                if self.pre_quantized and "disk" in device_map.values():
                    raise ValueError(
                        "You are attempting to perform disk offload with a pre-quantized torchao model "
                        "This is not supported yet . Please remove the disk device from the device_map."
                    )
        if self.pre_quantized:
            weights_only = kwargs.get("weights_only")
            if weights_only:
                torch_version = version.parse(importlib.metadata.version("torch"))
                if torch_version < version.parse("2.5.0"):
                    raise RuntimeError(
                        f"In order to use torchao pre-quantized model, you need to have torch>=2.5.0. However, the current version is {torch_version}."
                        f" You can also set with `weights_only=False` in `from_pretrained` if you don't want to update torch"
                    )

    def update_dtype(self, dtype):
        if self.quantization_config.quant_type == "int4_weight_only":
            if dtype != torch.bfloat16:
                logger.warning_once(
                    f"Setting dtype to {dtype} for int4_weight_only quantization, but only bfloat16 is supported right now. Overwriting torch_dtype to bfloat16."
                )
                dtype = torch.bfloat16
        return dtype

    def get_state_dict_and_metadata(self, model):
        """
        We flatten the state dict of tensor subclasses so that it is compatible with the safetensors format.
        """
        if TORCHAO_VERSION >= version.parse("0.15.0"):
            return flatten_tensor_state_dict(model.state_dict())
        else:
            raise RuntimeError(
                f"In order to use safetensors with torchao, please use torchao version >= 0.15.0. Current version: {TORCHAO_VERSION}"
            )

    def param_element_size(self, model: "PreTrainedModel", param_name: str, param: "torch.Tensor") -> float:
        "Return the element size (in bytes) for `param_name`."
        if self.param_needs_quantization(model, param_name) and self.quantized_param_size is not None:
            return self.quantized_param_size

        return super().param_element_size(model, param_name, param)

    def adjust_max_memory(self, max_memory: dict[str, int | str]) -> dict[str, int | str]:
        # need more space for the quantization parameters (e.g. scale). Tested with int4 wo and group size = 128
        max_memory = {key: val * 0.9 for key, val in max_memory.items()}
        return max_memory

    def _process_model_before_weight_loading(self, model: "PreTrainedModel", checkpoint_files=None, **kwargs):
        self.modules_to_not_convert = self.get_modules_to_not_convert(
            model, self.quantization_config.modules_to_not_convert, model._keep_in_fp32_modules
        )
        if self.quantization_config.include_input_output_embeddings:
            input_emb = model.get_input_embeddings()
            input_emb_names = [name for name, module in model.named_modules() if id(module) == id(input_emb)]
            output_emb = model.get_output_embeddings()
            output_emb_names = [name for name, module in model.named_modules() if id(module) == id(output_emb)]
            self.modules_to_not_convert = [
                x for x in self.modules_to_not_convert if x not in input_emb_names + output_emb_names
            ]
        if checkpoint_files is not None:
            # Torchao needs access to all metadata later
            self.set_metadata(checkpoint_files)

    def param_needs_quantization(self, model: "PreTrainedModel", param_name: str, **kwargs) -> bool:
        if self.quantization_config.quant_type == "autoquant":
            return False

        # check if the param_name is not in self.modules_to_not_convert
        if not should_convert_module(param_name, self.modules_to_not_convert):
            return False

        # we only quantize the weight of nn.Linear and nn.Embedding
        module, tensor_name = get_module_from_name(model, param_name)
        _QUANTIZABLE = [torch.nn.Linear]
        if self.quantization_config.include_input_output_embeddings:
            _QUANTIZABLE.append(torch.nn.Embedding)

        # Handle FqnToConfig, introduced in torchao 0.15.0+
        if self.quantization_config._get_ao_version() >= version.parse("0.15.0"):
            from torchao.quantization import FqnToConfig, fqn_matches_fqn_config

            if isinstance(self.quantization_config.quant_type, FqnToConfig):
                module_fqn, param_name_fqn = param_name.rsplit(".", 1)
                if (
                    fqn_matches_fqn_config(module_fqn, self.quantization_config.quant_type)
                    or fqn_matches_fqn_config(param_name, self.quantization_config.quant_type)
                    or (
                        "_default" in self.quantization_config.quant_type.fqn_to_config
                        and isinstance(module, tuple(_QUANTIZABLE))
                    )
                ):
                    return True

        return isinstance(module, tuple(_QUANTIZABLE)) and tensor_name == "weight"

    def _process_model_after_weight_loading(self, model, **kwargs):
        """No process required for torchao quantized model"""
        if self.quantization_config.quant_type == "autoquant":
            from torchao import autoquant
            from torchao.quantization import ALL_AUTOQUANT_CLASS_LIST

            model = torch.compile(model, mode="max-autotune")
            model = autoquant(
                model,
                qtensor_class_list=ALL_AUTOQUANT_CLASS_LIST,
                set_inductor_config=False,
                **self.quantization_config.quant_type_kwargs,
            )
            return model
        return

    def is_serializable(self) -> bool:
        _is_torchao_serializable = TORCHAO_VERSION >= version.parse("0.15.0")
        if not TORCHAO_VERSION >= version.parse("0.15.0"):
            logger.warning(
                "torchao quantized model only supports serialization for torchao version >= 0.15.0, please upgrade "
                "your version to save the quantized model"
            )
        return _is_torchao_serializable

    @property
    def is_trainable(self) -> bool:
        supported_quant_types_for_training = [
            "int8_weight_only",
            "int8_dynamic_activation_int8_weight",
        ]
        return self.quantization_config.quant_type in supported_quant_types_for_training

    @property
    def is_compileable(self) -> bool:
        return True

    def set_metadata(self, checkpoint_files: list[str]):
        if checkpoint_files[0].endswith(".safetensors"):
            metadata = {}
            for checkpoint in checkpoint_files:
                with safe_open(checkpoint, framework="pt") as f:
                    metadata_ = f.metadata() or {}
                    metadata.update(metadata_)
            # Save it
            self.metadata = metadata

    def get_quantize_ops(self):
        from ..integrations.torchao import TorchAoQuantize

        return TorchAoQuantize(self)

    def get_weight_conversions(self):
        from ..integrations.torchao import TorchAoDeserialize

        if self.pre_quantized:
            return [
                WeightConverter(
                    # TODO: incr flexibility by generalizing the source patterns to match the format of "_weight_"
                    # note that the matching logic is greedy, so for ex, if _weight_scale is before _weight_scale_and_zero in this list, it will match _weight_scale always (this is incorrect)
                    # thus, the order of source_patterns is intentional
                    source_patterns=[
                        "_weight_qdata",
                        "_weight_scale_and_zero",
                        "_weight_scale",
                        "_weight_zero_point",
                        "_weight_act_pre_scale",
                    ],
                    target_patterns="weight",
                    operations=[TorchAoDeserialize(self)],
                ),
            ]
        return []
