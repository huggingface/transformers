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

from ..integrations import prepare_for_hqq_linear
from ..utils import is_hqq_available, is_torch_available, logging
from .base import HfQuantizer
from .quantizers_utils import get_module_from_name


if TYPE_CHECKING:
    from ..modeling_utils import PreTrainedModel
    from ..utils.quantization_config import HqqConfig


if is_torch_available():
    import torch

logger = logging.get_logger(__name__)


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
        module, _ = get_module_from_name(model, param_name)
        # Since we do not prepare the modules in advance, we need every param of the Linear layer to go through
        # `create_quantized_param`, even when `self.is_quantized == True`
        return isinstance(module, torch.nn.Linear)

    def _process_model_before_weight_loading(
        self,
        model: "PreTrainedModel",
        **kwargs,
    ):
        # Add the corresponding quant_config to each valid module. This allows us to do the actual nn.Linear -> HQQLinear conversion in create_quantized_param().
        # prepare_for_hqq_linear() also sets the right quantization config inside the model (model.config.quantization_config) and the layers (hqq_layer.quant_config)
        model = prepare_for_hqq_linear(model, quantization_config=self.quantization_config)

    def is_serializable(self):
        return True

    @property
    def is_trainable(self) -> bool:
        return True
