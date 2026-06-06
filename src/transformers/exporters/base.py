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

from __future__ import annotations

import importlib.util
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from ..utils.export_config import ExportConfigMixin
from .utils import decompose_for_generation


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
    def export(self, model: PreTrainedModel, sample_inputs: dict):
        """
        Export the model and return the backend-specific program object.

        Args:
            model ([`PreTrainedModel`]):
                The model to export.
            sample_inputs (`dict[str, Any]`):
                **Forward** kwargs — what you'd pass to `model(**sample_inputs)`. These are used
                directly as the example inputs during tracing. For an autoregressive decode-step
                export, this means you need to include `past_key_values`, `cache_position`, etc.
                If you only have generation-style inputs, use [`~HfExporter.export_for_generation`]
                instead — it runs `model.generate` for you and exports each stage.

        Returns:
            Backend-specific export artifact.
        """
        pass

    def export_for_generation(self, model: PreTrainedModel, sample_inputs: dict) -> dict[str, object]:
        """
        Decompose a generative model and export each component independently.

        Thin wrapper around [`~exporters.utils.decompose_for_generation`] that calls
        [`~HfExporter.export`] on every returned `(submodel, forward_inputs)` pair. If you need
        the intermediate `(submodel, forward_inputs)` pairs (for verification, custom inputs,
        skipping a stage, …), call [`~exporters.utils.decompose_for_generation`] directly.

        Args:
            model ([`PreTrainedModel`]):
                The generative model to export. Must support `model.generate(**sample_inputs)`.
            sample_inputs (`dict[str, Any]`):
                **Generate** kwargs — what you'd pass to `model.generate(**sample_inputs)`
                (typically `input_ids` + `attention_mask`, plus any modality inputs like
                `pixel_values` / `input_features` for multi-modal models). Per-stage forward
                kwargs are captured internally.

        Returns:
            `dict[str, Any]`: `{component_name: backend_specific_artifact}` — same keys as
            [`~exporters.utils.decompose_for_generation`]. Values are whatever
            [`~HfExporter.export`] returns for the concrete backend (`ExportedProgram`,
            `ONNXProgram`, `ExecutorchProgramManager`).
        """
        components = decompose_for_generation(model, sample_inputs)
        exported: dict[str, object] = {}
        for name, (submodel, subinputs) in components.items():
            try:
                exported[name] = self.export(submodel, subinputs)
            except Exception as e:
                raise RuntimeError(
                    f"{type(self).__name__}.export failed on component '{name}' "
                    f"(submodel={type(submodel).__name__}, input keys={list(subinputs)})."
                ) from e
        return exported
