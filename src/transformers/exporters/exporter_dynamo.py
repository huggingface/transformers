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
from .utils import patch_masks_for_export, register_dynamic_cache_export_support


if is_torch_available():
    import torch


if TYPE_CHECKING:
    from ..modeling_utils import PreTrainedModel

logger = logging.get_logger(__file__)


class DynamoExporter(HfExporter):
    export_config: DynamoConfig

    required_packages = ["torch"]

    def validate_environment(self, *args, **kwargs):
        super().validate_environment(*args, **kwargs)

        if not is_torch_greater_or_equal("2.6.0"):
            raise ImportError("DynamoExporter requires torch>=2.6.0 for stable Dynamo based export.")

    def export(self, model: "PreTrainedModel"):
        from torch.export import Dim, ExportedProgram

        if self.export_config.sample_inputs is None:
            raise NotImplementedError(
                "OnnxExporter can't automatically generate export inptus. Please provide sample_inputs in the exporter_config as a dictionary. "
                "You can do so by using the tokenizer/processor to prepare a batch of inputs as you would do for a normal forward pass. "
                "OnnxExporter can automatically generate past_key_values and its dynamic shapes if the model is "
                "auto-regressive and model.config.use_cache is set to True."
            )

        args = ()
        kwargs = self.export_config.sample_inputs
        dynamic_shapes = self.export_config.dynamic_shapes

        if isinstance(model, GenerationMixin) and model.config.use_cache:
            register_dynamic_cache_export_support()

            # NOTE: for now i'm creating it here reduces to reduce user burden
            kwargs["past_key_values"] = model(**kwargs).past_key_values

            if dynamic_shapes is not None:
                dynamic_shapes["past_key_values"] = [
                    [{0: Dim.DYNAMIC, 2: Dim.DYNAMIC} for _ in range(len(kwargs["past_key_values"].layers))],
                    [{0: Dim.DYNAMIC, 2: Dim.DYNAMIC} for _ in range(len(kwargs["past_key_values"].layers))],
                ]

        with patch_masks_for_export():
            exported_program: ExportedProgram = torch.export.export(
                model,
                args=args,
                kwargs=kwargs,
                dynamic_shapes=dynamic_shapes,
                strict=self.export_config.strict,
            )

        model.exported_model = exported_program
