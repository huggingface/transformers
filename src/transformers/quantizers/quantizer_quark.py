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

from typing import TYPE_CHECKING

from .base import HfQuantizer


if TYPE_CHECKING:
    from ..modeling_utils import PreTrainedModel
    from ..utils.quantization_config import QuarkConfig

from ..utils import is_quark_available, logging


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
    quantization_config: "QuarkConfig"

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

    def is_serializable(self):
        return False

    @property
    def is_trainable(self):
        return False

    def get_weight_conversions(self, model=None):
        from ..core_model_loading import WeightRenaming

        # The checkpoint stores `weight_scale`, ... while `QParamsLinear` holds them in its quantizers (`weight_quantizer.scale`)
        return [WeightRenaming(source, target) for source, target in CHECKPOINT_KEYS.items()]
