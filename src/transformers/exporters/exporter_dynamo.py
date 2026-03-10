# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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
from typing import TYPE_CHECKING, Any

from ..utils import logging
from ..utils.export_config import DynamoConfig
from ..utils.import_utils import is_torch_available
from .base import HfExporter
from .utils import get_auto_dynamic_shapes, prepare_for_export, register_pytrees_for_model


if is_torch_available():
    import torch


if TYPE_CHECKING:
    from torch.export import ExportedProgram

    from ..modeling_utils import PreTrainedModel


logger = logging.get_logger(__file__)


class DynamoExporter(HfExporter):
    export_config: DynamoConfig

    required_packages = ["torch"]

    def validate_environment(self, *args, **kwargs):
        super().validate_environment(*args, **kwargs)

    def export(self, model: "PreTrainedModel", sample_inputs: dict[str, Any]) -> "ExportedProgram":
        """Exports a model to a TorchDynamo ExportedProgram.
        Args:
            model (`PreTrainedModel`):
                The model to export.
            sample_inputs (`Dict[str, Any]`):
                The sample inputs to use for the export.
        Returns:
            `ExportedProgram`: The exported model.
        """

        # we use a copy to avoid side effects
        inputs = copy.deepcopy(sample_inputs)
        model, inputs, _ = prepare_for_export(model, inputs)

        dynamic_shapes = self.export_config.dynamic_shapes
        if self.export_config.dynamic and dynamic_shapes is None:
            dynamic_shapes = get_auto_dynamic_shapes(inputs)

        register_pytrees_for_model(model)

        exported_program: ExportedProgram = torch.export.export(
            model,
            args=(),
            kwargs=inputs,
            dynamic_shapes=dynamic_shapes,
            strict=self.export_config.strict,
            prefer_deferred_runtime_asserts_over_guards=self.export_config.prefer_deferred_runtime_asserts_over_guards,
        )

        return exported_program
