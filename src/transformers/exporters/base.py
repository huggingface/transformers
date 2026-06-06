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
from .utils import decompose_multimodal, decompose_prefill_decode, is_multimodal


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
        Decompose a generative model into independently exportable components and export each.

        Splits the forward into prefill and decode via [`~exporters.utils.decompose_prefill_decode`],
        and if the prefill is multi-modal (per [`~exporters.utils.is_multimodal`]) further splits it
        into one entry per submodule (vision/audio encoder, projector, language model, `lm_head`)
        via [`~exporters.utils.decompose_multimodal`]. Each component is then passed through
        [`~HfExporter.export`].

        Args:
            model ([`PreTrainedModel`]):
                The generative model to export. Must support `model.generate(**sample_inputs)`.
            sample_inputs (`dict[str, Any]`):
                **Generate** kwargs — what you'd pass to `model.generate(**sample_inputs)` (typically
                `input_ids` + `attention_mask`, plus any modality inputs like `pixel_values` /
                `input_features` for multi-modal models). Per-stage forward kwargs (with
                `past_key_values`, `cache_position`, etc.) are captured internally by running
                `model.generate(**sample_inputs, max_new_tokens=2)`. If you already have explicit
                forward kwargs for a single stage, use [`~HfExporter.export`] directly instead.

        Returns:
            `dict[str, Any]`: `{component_name: backend_specific_artifact}`. The keys are
            `"prefill"` / `"decode"` for plain generative models and
            `"<modality>_encoder"` / `"multi_modal_projector"` / `"language_model"` / `"lm_head"` / `"decode"`
            for multi-modal generative models. Values are whatever [`~HfExporter.export`] returns
            for the concrete backend (`ExportedProgram`, `ONNXProgram`, `ExecutorchProgramManager`).
        """
        stages = decompose_prefill_decode(model, sample_inputs)
        prefill_model, prefill_inputs = stages["prefill"]

        if is_multimodal(prefill_model):
            components = decompose_multimodal(prefill_model, prefill_inputs)
        else:
            components = {"prefill": stages["prefill"]}
        components["decode"] = stages["decode"]

        return {name: self.export(submodel, subinputs) for name, (submodel, subinputs) in components.items()}
