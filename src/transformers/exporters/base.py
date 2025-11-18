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
import importlib.util
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from ..utils.export_config import ExportConfigMixin


if TYPE_CHECKING:
    from ..modeling_utils import PreTrainedModel


class HfExporter(ABC):
    """
    Abstract class of the HuggingFace exporter. Supports exporting HF transformers models using various export formats.
    This class is used only for transformers.PreTrainedModel.from_pretrained and cannot be easily used outside the scope of that method
    yet.

    Attributes:
        export_config (`transformers.utils.export_config.ExportConfigMixin`):
            The export configuration used to export the model.
    """

    required_packages: list[str] | None = None

    def __init__(self, export_config: ExportConfigMixin):
        self.export_config = export_config

    def validate_environment(self, *args, **kwargs):
        """
        This method is used to potentially check for potential conflicts with arguments that are
        passed in `from_pretrained`. You need to define it for all future exporters that are integrated with transformers.
        If no explicit check are needed, simply return nothing.
        """

        if self.required_packages is not None:
            missing_dependencies = []
            for package in self.required_packages:
                if importlib.util.find_spec(package) is None:
                    missing_dependencies.append(package)

            if missing_dependencies:
                raise ImportError(
                    f"To use {self.__class__.__name__}, please install the following dependencies: {', '.join(missing_dependencies)}"
                )

    @abstractmethod
    def export(self, model: "PreTrainedModel"):
        """
        Exports the given model according to the export configuration provided during initialization.

        Args:
            model: The model to be exported.
        """
        pass
