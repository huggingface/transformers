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
from ..utils import logging
from ..utils.export_config import DynamoConfig
from ..utils.import_utils import is_torch_available, is_torch_greater_or_equal
from .base import HfExporter


if is_torch_available():
    import torch

logger = logging.get_logger(__file__)


class DynamoExporter(HfExporter):
    export_config: DynamoConfig

    required_packages = ["torch"]

    def validate_environment(self, *args, **kwargs):
        super().validate_environment(*args, **kwargs)

        if not is_torch_greater_or_equal("2.6.0"):
            raise ImportError("DynamoExporter requires torch>=2.6.0")

    def export(self, model):
        from torch.export import ExportedProgram

        if self.export_config.sample_inputs is None:
            raise NotImplementedError(
                "DynamoExporter does not generate inptus for now. Please provide sample_inputs in the exporter_config."
            )

        args = ()
        kwargs = None
        dynamic_shapes = None

        if isinstance(self.export_config.sample_inputs, tuple):
            args = self.export_config.sample_inputs
        elif isinstance(self.export_config.sample_inputs, dict):
            kwargs = self.export_config.sample_inputs
        else:
            raise ValueError(
                "sample_inputs should be either a tuple of positional arguments or a dict of keyword arguments."
            )

        # some input validation can be done here (like using pytree)

        exported_program: ExportedProgram = torch.export.export(
            model,
            args=args,
            kwargs=kwargs,
            dynamic_shapes=dynamic_shapes,
            strict=self.export_config.strict,
        )
        model._exported_program = exported_program
        model._exported = True
