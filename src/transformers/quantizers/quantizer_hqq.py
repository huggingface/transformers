# Copyright 2024 The HuggingFace Team. All rights reserved.
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

from ..utils import is_hqq_available, is_torch_available, logging
from .base import HfQuantizer
from .quantizers_utils import get_module_from_name


if TYPE_CHECKING:
    from ..modeling_utils import PreTrainedModel
    from ..utils.quantization_config import HqqConfig


if is_torch_available():
    import torch

logger = logging.get_logger(__name__)


HQQ_SERIALIZED_NAMES = [
    "W_q",
    "nbits",
    "group_size",
    "shape",
    "scale",
    "zero",
    "axis",
    "packing",
    "unpack_view_dtype",
    "view_as_float",
    "quant_scale",
    "quant_zero",
    "compute_dtype",
    "offload_meta",
    "encoded_state_dict",
    "stores_quant_config",
    "channel_wise",
    "optimize",
    "round_zero",
]


class HqqHfQuantizer(HfQuantizer):
    """
    HQQ quantizer base HF class.
    nn.Linear modules are first tagged with quant_config in _process_model_before_weight_loading().
    """

    requires_calibration = False
    quantization_config: "HqqConfig"

    def __init__(self, quantization_config, **kwargs):
        if not is_hqq_available():
            raise ImportError(
                "A valid HQQ version (>=0.2.1) is not available. Please follow the instructions to install it: `https://github.com/mobiusml/hqq/`."
            )
        super().__init__(quantization_config, **kwargs)

    def validate_environment(self, *args, **kwargs):
        device_map = kwargs.get("device_map")
        if isinstance(device_map, dict):
            if "cpu" in device_map.values() or "disk" in device_map.values():
                raise ValueError(
                    "You are attempting to use an HQQ model with a device_map that contains a CPU or disk device."
                    " This is not supported. Please remove the CPU or disk device from the device_map."
                )

    def param_needs_quantization(self, model: "PreTrainedModel", param_name: str, **kwargs) -> bool:
        module, tensor_name = get_module_from_name(model, param_name)
        # `prepare_for_hqq_linear` tags the linears to quantize with their `quant_config`
        return (
            isinstance(module, torch.nn.Linear)
            and tensor_name == "weight"
            and getattr(module, "quant_config", None) is not None
        )

    def _process_model_before_weight_loading(
        self,
        model: "PreTrainedModel",
        **kwargs,
    ):
        if self.pre_quantized and self.quantization_config.dequantize:
            # The model stays dense: the `dequantize` load operation runs on every checkpoint weight
            return
        # Add the corresponding quant_config to each valid module. The `quantize` load operation then swaps each one for an `HQQLinear`.
        # prepare_for_hqq_linear() also sets the right quantization config inside the model (model.config.quantization_config) and the layers (hqq_layer.quant_config)
        from ..integrations import prepare_for_hqq_linear

        model = prepare_for_hqq_linear(model, quantization_config=self.quantization_config)

    def is_serializable(self):
        return True

    @property
    def is_trainable(self) -> bool:
        return True

    serialized_names = {"weight": HQQ_SERIALIZED_NAMES}

    def get_load_operations(self):
        from ..integrations.hqq import HqqDequantize, HqqDeserialize, HqqQuantize

        return {"deserialize": HqqDeserialize, "dequantize": HqqDequantize, "quantize": HqqQuantize}
