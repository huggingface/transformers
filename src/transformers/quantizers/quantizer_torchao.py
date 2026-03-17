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
import re
from typing import TYPE_CHECKING

from .base import HfQuantizer
from .quantizers_utils import get_module_from_name, should_convert_module


if TYPE_CHECKING:
    from ..modeling_utils import PreTrainedModel
    from ..utils.quantization_config import TorchAoConfig

from safetensors import safe_open

from ..utils import is_torch_available, is_torchao_available, logging


MIN_TORCH_VERSION = "2.5.0"


if is_torch_available():
    from ..core_model_loading import WeightConverter


if is_torch_available():
    import torch

if is_torchao_available():
    from torchao.prototype.safetensors.safetensors_support import (
        flatten_tensor_state_dict,
    )


logger = logging.get_logger(__name__)


def _fuzzy_match_size(config_name: str) -> str | None:
    """
    Extract the size digit from torchao config class names like "Int4WeightOnlyConfig", "Int8WeightOnlyConfig".
    Returns the digit as a string if found, otherwise None.
    """
    match = re.search(r"(\d)weight", config_name.lower())
    return match.group(1) if match else None


class TorchAoHfQuantizer(HfQuantizer):
    """
    Quantizer for torchao: https://github.com/pytorch/ao/
    """

    requires_calibration = False
    quantization_config: "TorchAoConfig"

    def __init__(self, quantization_config, **kwargs):
        super().__init__(quantization_config, **kwargs)

        size_digit = _fuzzy_match_size(type(self.quantization_config.quant_type).__name__)
        self.quantized_param_size = 0.5 if size_digit == "4" else 1

    def validate_environment(self, *args, **kwargs):
        if not is_torchao_available():
            raise ImportError("Loading an torchao quantized model requires torchao library (`pip install torchao`)")

        device_map = kwargs.get("device_map")
        self.offload_to_cpu = False
        if isinstance(device_map, dict):
            if ("disk" in device_map.values() or "cpu" in device_map.values()) and len(device_map) > 1:
                self.offload_to_cpu = "cpu" in device_map.values()
                if self.pre_quantized and "disk" in device_map.values():
                    raise ValueError(
                        "You are attempting to perform disk offload with a pre-quantized torchao model "
                        "This is not supported yet . Please remove the disk device from the device_map."
                    )

    def get_state_dict_and_metadata(self, model):
        """
        We flatten the state dict of tensor subclasses so that it is compatible with the safetensors format.
        """
        return flatten_tensor_state_dict(model.state_dict())

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
        # check if the param_name is not in self.modules_to_not_convert
        if not should_convert_module(param_name, self.modules_to_not_convert):
            return False

        # we only quantize the weight of nn.Linear and nn.Embedding
        module, tensor_name = get_module_from_name(model, param_name)
        _QUANTIZABLE = [torch.nn.Linear]
        if self.quantization_config.include_input_output_embeddings:
            _QUANTIZABLE.append(torch.nn.Embedding)

        from torchao.quantization import FqnToConfig, fqn_matches_fqn_config

        if isinstance(self.quantization_config.quant_type, FqnToConfig):
            module_fqn, _ = param_name.rsplit(".", 1)
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

    def is_serializable(self) -> bool:
        return True

    @property
    def is_trainable(self) -> bool:
        # Only 8-bit quantization (e.g. Int8WeightOnly, Int8DynamicActivationInt8Weight) supports training
        return _fuzzy_match_size(type(self.quantization_config.quant_type).__name__) == "8"

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
