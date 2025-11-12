# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
# Modifications Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
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

import copy
from typing import TYPE_CHECKING

from ..utils import logging
from ..utils.export_config import DynamoConfig
from ..utils.import_utils import is_torch_available, is_torch_greater_or_equal
from .base import HfExporter
from .utils import (
    get_auto_dynamic_shapes,
    patch_model_for_export,
    prepare_inputs_for_export,
    raise_on_unsupported_model,
    register_dynamic_cache_for_export,
    register_encoder_decoder_cache_for_export,
    warn_on_unsupported_cache_class,
)


if is_torch_available():
    import torch


if TYPE_CHECKING:
    from ..modeling_utils import PreTrainedModel

    if is_torch_greater_or_equal("2.6.0"):
        from torch.export import ExportedProgram

logger = logging.get_logger(__file__)


class DynamoExporter(HfExporter):
    export_config: DynamoConfig

    required_packages = ["torch"]

    def validate_environment(self, *args, **kwargs):
        super().validate_environment(*args, **kwargs)

        if not is_torch_greater_or_equal("2.6.0"):
            raise ImportError(f"{self.__class__.__name__} requires torch>=2.6.0 for stable Dynamo based export.")

    def export(self, model: "PreTrainedModel"):
        raise_on_unsupported_model(model)
        warn_on_unsupported_cache_class(model)

        if self.export_config.sample_inputs is None:
            raise NotImplementedError(
                f"{self.__class__.__name__} can't automatically generate export inptus. Please provide sample_inputs in the exporter_config as a dictionary. "
                "You can do so by using the tokenizer/processor to prepare a batch of inputs as you would do for a normal forward pass. "
                f"{self.__class__.__name__} can automatically generate past_key_values and its dynamic shapes if the model is "
                "an auto-regressive (CausalLM) and its model.config.use_cache is set to True."
            )

        sample_inputs = copy.deepcopy(self.export_config.sample_inputs)
        model, sample_inputs = prepare_inputs_for_export(model, sample_inputs)

        dynamic_shapes = self.export_config.dynamic_shapes
        if self.export_config.dynamic and dynamic_shapes is None:
            # assigns AUTO to all axes to let torch dynamo decide
            dynamic_shapes = get_auto_dynamic_shapes(sample_inputs)

        register_dynamic_cache_for_export()
        register_encoder_decoder_cache_for_export()
        with patch_model_for_export(model):
            exported_program: ExportedProgram = torch.export.export(
                model,
                args=(),
                kwargs=sample_inputs,
                dynamic_shapes=dynamic_shapes,
                strict=self.export_config.strict,
            )
        model.exported_model = exported_program
        return exported_program
