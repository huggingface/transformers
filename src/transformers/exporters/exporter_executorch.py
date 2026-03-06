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

from typing import TYPE_CHECKING, Any

from ..utils import logging
from ..utils.export_config import ExecutorchConfig
from ..utils.import_utils import is_executorch_available, is_torch_available
from .exporter_dynamo import DynamoExporter


if is_torch_available():
    from torch.export import ExportedProgram

if is_executorch_available():
    from executorch.exir import to_edge_transform_and_lower

if TYPE_CHECKING:
    from ..modeling_utils import PreTrainedModel

logger = logging.get_logger(__file__)


class ExecutorchExporter(DynamoExporter):
    export_config: ExecutorchConfig

    required_packages = ["torch", "executorch"]

    def export(self, model: "PreTrainedModel", sample_inputs: dict[str, Any]) -> Any:
        """
        Exports a model for ExecuTorch using the full export and lowering workflow.
        Args:
            model (`PreTrainedModel`): The model to export.
            sample_inputs (`Dict[str, Any]`): The sample inputs to use for the export.
        Returns:
            The exported model in ExecuTorch format.
        """

        exported_program: ExportedProgram = super().export(model, sample_inputs)
        executorch_program = to_edge_transform_and_lower(exported_program).to_executorch()

        return executorch_program
