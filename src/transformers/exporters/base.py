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
"""Abstract base class for all Transformers exporters."""

import importlib.util
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from ..utils.export_config import ExportConfigMixin


if TYPE_CHECKING:
    from ..modeling_utils import PreTrainedModel


class HfExporter(ABC):
    """
    Abstract base class for all Transformers exporters.

    Subclass and implement [`~HfExporter.export`] to add a new export backend.

    Args:
        export_config ([`~transformers.utils.export_config.ExportConfigMixin`]):
            Backend-specific configuration. The concrete subclass declares the
            expected type via a class-level annotation.
    """

    required_packages: list[str] | None = None

    def __init__(self, export_config: ExportConfigMixin):
        self.export_config = export_config

    def validate_environment(self, *args, **kwargs):
        """
        Check that all packages listed in `required_packages` are installed.
        Override to add exporter-specific environment checks.
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
    def export(self, model: "PreTrainedModel", sample_inputs: dict):
        """
        Export the model and return the backend-specific program object.

        Args:
            model ([`PreTrainedModel`]):
                The model to export.
            sample_inputs (`dict[str, Any]`):
                Forward kwargs used as concrete example inputs during tracing.

        Returns:
            Backend-specific export artifact.
        """
        pass
