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

from typing import TYPE_CHECKING

from .base import HfQuantizer


if TYPE_CHECKING:
    from ..modeling_utils import PreTrainedModel

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

    def param_needs_quantization(self, model: "PreTrainedModel", param_name: str, **kwargs) -> bool:
        return True

    def is_serializable(self, **kwargs):
        return False

    @property
    def is_trainable(self):
        return False

    def get_weight_conversions(self):
        from ..core_model_loading import WeightConverter
        from ..integrations.quark import QuarkDeserialize
        # In Quark, quantization is managed through a QParamsLinear module, which holds
        # separate quantizers for the weights, inputs, and biases (e.g. weight_quantizer
        # input_quantizer, bias_quantizer, etc.).
        #
        # When you call `module.state_dict()`, Quark automatically renames the quantizer
        # parameters — for example, `input_quantizer.scale` becomes `input_scale` — and
        # saves them directly at the parent module level.
        #
        # This means we cannot simply rename keys like `weight_scale` back to
        # `weight_quantizer.scale` when loading the state_dict.
        # Otherwise, the `missing_keys` list would still expect keys such as
        # `weight_scale`, `bias_scale`, etc.
        #
        # To fix this, we keep the expected state_dict keys (like `weight_scale`,
        # `bias_scale`, etc.) unchanged, and during the conversion step, we explicitly
        # assign their values into the corresponding quantizer attributes
        # (`weight_quantizer.scale`, `input_quantizer.scale`, and so on).

        # You can notice here that in target_patterns we use the same key as the source_patterns,
        # this is because we just want to collect the tensors, and we will rename them later in the convert function.
        # We cannot rename directly or else the missing_keys list will not be able to find the tensors.
        converters = []
        for key in CHECKPOINT_KEYS.keys():
            converters.append(
                WeightConverter(
                    source_patterns=[key],
                    target_patterns=key,
                    operations=[QuarkDeserialize(self)],
                )
            )
        return converters
