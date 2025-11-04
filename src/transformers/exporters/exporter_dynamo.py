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

from typing import TYPE_CHECKING

from ..generation.utils import GenerationMixin
from ..utils import logging
from ..utils.export_config import DynamoConfig
from ..utils.import_utils import is_torch_available, is_torch_greater_or_equal
from .base import HfExporter
from .utils import get_auto_dynamic_shapes, patch_masks_for_export, register_dynamic_cache_for_export


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
            raise ImportError("DynamoExporter requires torch>=2.6.0 for stable Dynamo based export.")

    def export(self, model: "PreTrainedModel"):
        if self.export_config.sample_inputs is None:
            raise NotImplementedError(
                "OnnxExporter can't automatically generate export inptus. Please provide sample_inputs in the exporter_config as a dictionary. "
                "You can do so by using the tokenizer/processor to prepare a batch of inputs as you would do for a normal forward pass. "
                "OnnxExporter can automatically generate past_key_values and its dynamic shapes if the model is "
                "auto-regressive and model.config.use_cache is set to True."
            )

        args = ()
        kwargs = self.export_config.sample_inputs
        if isinstance(model, GenerationMixin) and model.config.use_cache:
            register_dynamic_cache_for_export()
            if "past_key_values" not in kwargs:
                logger.info(
                    "OnnxExporter detected an auto-regressive model with use_cache=True but no past_key_values in sample_inputs. "
                    "Generating a dummy past_key_values for export requires running a forward pass which may be time-consuming. "
                    "You can also provide past_key_values in sample_inputs to avoid this step."
                )
                sample_outputs = model(**kwargs)
                kwargs["past_key_values"] = sample_outputs.past_key_values

        dynamic_shapes = self.export_config.dynamic_shapes
        if self.export_config.dynamic and dynamic_shapes is None:
            # assigns AUTO to all axes to let torch.onnx decide
            dynamic_shapes = get_auto_dynamic_shapes(kwargs)

        with patch_masks_for_export():
            exported_program: ExportedProgram = torch.export.export(
                model,
                args=args,
                kwargs=kwargs,
                dynamic_shapes=dynamic_shapes,
                strict=self.export_config.strict,
            )

        model.exported_model = exported_program
