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
"""Abstract base class for all Transformers exporters."""

from __future__ import annotations

import dataclasses
import json
from abc import ABC, abstractmethod
from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..utils import logging
from .configs import ExportConfigMixin, ExportFormat, export_format_name
from .metadata import ExportMetadata
from .quantizers.utils import capture_calibration_inputs
from .utils import check_packages, decompose_for_generation


logger = logging.get_logger(__name__)


# Backend, file and recorded metadata per component. Kept outside the artifacts because not every format
# can carry the metadata (`torch.export` cannot).
EXPORT_MANIFEST_FILE = "export.json"


if TYPE_CHECKING:
    import torch

    from ..cache_utils import Cache
    from ..generation import GenerationConfig
    from ..modeling_utils import PreTrainedModel


@dataclass
class ExportedComponent:
    """One exported graph: the backend's program, and what the exporter recorded about it."""

    artifact: Any
    metadata: ExportMetadata


class ExportArtifacts(Mapping):
    """What an export produced: a `Mapping` over `{name: ExportedComponent}`, plus the configs they were traced
    against.

    [`~HfExporter.export`] produces a single `"model"` component, whose program is `.artifact`;
    [`~HfExporter.export_for_generation`] produces one component per generation stage.
    """

    def __init__(
        self,
        components: dict[str, ExportedComponent],
        export_format: ExportFormat | str,
        config: object | None = None,
        generation_config: GenerationConfig | None = None,
    ):
        self.components = dict(components)
        self.export_format = export_format
        self.config = config
        self.generation_config = generation_config

    def __getitem__(self, name: str) -> ExportedComponent:
        return self.components[name]

    def __iter__(self):
        return iter(self.components)

    def __len__(self) -> int:
        return len(self.components)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(format={export_format_name(self.export_format)!r}, components={list(self)})"

    @property
    def artifact(self):
        """The backend's own program, for a single-graph export."""
        if len(self.components) != 1:
            raise ValueError(
                f"This export has {len(self.components)} components ({list(self.components)}), so there is "
                "no single `.artifact`; index it by name and take that component's."
            )
        return next(iter(self.components.values())).artifact

    def save_pretrained(self, save_directory: str | Path) -> None:
        """Write the components (one file each), the configs, and a manifest naming each component's file and the
        metadata recorded when tracing it."""
        from .auto import export_backend

        exporter = export_backend(self.export_format, "exporter")
        directory = Path(save_directory)
        directory.mkdir(parents=True, exist_ok=True)

        components = {}
        for name, component in self.components.items():
            filename = f"{name}{exporter.artifact_suffix}"
            exporter.save_artifact(component.artifact, directory / filename)
            components[name] = {"file": filename, "metadata": dict(component.metadata.raw)}

        manifest = {
            "schema_version": 1,
            "export_format": export_format_name(self.export_format),
            "components": components,
        }
        (directory / EXPORT_MANIFEST_FILE).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        if self.config is not None:
            self.config.save_pretrained(directory)
        if self.generation_config is not None:
            self.generation_config.save_pretrained(directory)


class HfExporter(ABC):
    """
    Abstract base class for all Transformers exporters.

    To add a backend, subclass and implement its two halves: [`~HfExporter.export_artifact`] to trace one
    graph, and [`~HfExporter.save_artifact`] to write one out. The public [`~HfExporter.export`] and
    [`~HfExporter.export_for_generation`] build an [`ExportArtifacts`] on top of them.
    """

    export_format: ExportFormat | str
    config_class: type[ExportConfigMixin]
    # Extension of a saved component's file (`model.onnx`, `decode.pte`, ...).
    artifact_suffix: str

    required_packages: list[str] = []
    # Hard minimum versions — the exporter raises below these (features it relies on are absent).
    min_versions: dict[str, str] = {}
    # Versions the exporter is validated against — a mismatch only warns.
    tested_versions: dict[str, str] = {}

    def __init__(self):
        self.validate_environment()

    def validate_environment(self, *args, **kwargs):
        """Check `required_packages` are installed and warn on version drift from `tested_versions`."""
        check_packages(self.__class__.__name__, self.required_packages, self.min_versions, self.tested_versions)

    def _as_config(self, config: ExportConfigMixin | dict[str, Any]) -> ExportConfigMixin:
        """`config` as this exporter's `config_class`, built from a dict when handed one."""
        if isinstance(config, dict):
            return self.config_class(**config)
        if not isinstance(config, self.config_class):
            raise TypeError(f"Expected config to be a {self.config_class.__name__} or dict, got {type(config)}")
        return config

    @abstractmethod
    def export_artifact(
        self,
        model: PreTrainedModel,
        sample_inputs: MutableMapping[str, torch.Tensor | Cache],
        config: ExportConfigMixin,
    ) -> tuple[Any, dict[str, Any]]:
        """
        Trace one graph. The backend extension point, paired with [`~HfExporter.save_artifact`].

        Args:
            model ([`PreTrainedModel`]):
                The model to export.
            sample_inputs (`dict[str, torch.Tensor | Cache]`):
                **Forward** kwargs — what you'd pass to `model(**sample_inputs)`. These are used
                directly as the example inputs during tracing. For an autoregressive decode-step
                export, this means you need to include `past_key_values`, `cache_position`, etc.
                If you only have generation-style inputs, use [`~HfExporter.export_for_generation`]
                instead — it runs `model.generate` for you and exports each stage.
            config ([`~transformers.exporters.configs.ExportConfigMixin`]):
                Backend-specific configuration.

        Returns:
            `tuple[Any, dict]`: The backend's program object (`ExportedProgram`, `ONNXProgram`,
            `ExecutorchProgramManager`, `openvino.Model`) and the metadata recorded while tracing it
            ([`~exporters.metadata.build_export_metadata`]), returned separately since only some formats can
            carry it.
        """

    @classmethod
    @abstractmethod
    def save_artifact(cls, artifact: Any, path: str | Path) -> None:
        """Write one program returned by [`~HfExporter.export_artifact`] to `path`, in the backend's file format."""

    def export(
        self,
        model: PreTrainedModel,
        sample_inputs: MutableMapping[str, torch.Tensor | Cache],
        config: ExportConfigMixin | dict[str, Any],
    ) -> ExportArtifacts:
        """
        Export the model as one graph.

        Takes the same **forward** kwargs as [`~HfExporter.export_artifact`].

        Returns:
            [`ExportArtifacts`]: A single `"model"` component; the backend's program is `.artifact`, and
            `save_pretrained` writes it with its metadata and the model config.
        """
        artifact, metadata = self.export_artifact(model, sample_inputs, config)
        component = ExportedComponent(artifact, ExportMetadata.from_dict(metadata))
        # A decomposed component (e.g. `FSMTEncoder`) is a plain `nn.Module` with no config.
        return ExportArtifacts({"model": component}, self.export_format, config=getattr(model, "config", None))

    def export_for_generation(
        self,
        model: PreTrainedModel,
        sample_inputs: MutableMapping[str, torch.Tensor | Cache],
        config: ExportConfigMixin | dict[str, ExportConfigMixin],
        *,
        generation_config: GenerationConfig | None = None,
        multi_token_decode: bool = False,
    ) -> ExportArtifacts:
        """
        Decompose a generative model and export each component independently.

        Thin wrapper around [`~exporters.utils.decompose_for_generation`] that calls
        [`~HfExporter.export_artifact`] on every returned `(submodel, forward_inputs)` pair. If you need
        the intermediate `(submodel, forward_inputs)` pairs (for verification, custom inputs,
        skipping a stage, …), call [`~exporters.utils.decompose_for_generation`] directly.

        Args:
            model ([`PreTrainedModel`]):
                The generative model to export. Must support `model.generate(**sample_inputs)`.
            sample_inputs (`dict[str, torch.Tensor | Cache]`):
                **Generate** kwargs — what you'd pass to `model.generate(**sample_inputs)`
                (typically `input_ids` + `attention_mask`, plus any modality inputs like
                `pixel_values` / `input_features` for multi-modal models). Per-stage forward
                kwargs are captured internally.
            config ([`~transformers.exporters.configs.ExportConfigMixin`] or `dict[str, ExportConfigMixin]`):
                Backend-specific configuration. Pass a single config to apply to every
                component, or a `dict` keyed by component name (e.g. `"image_encoder"`,
                `"language_model"`, `"lm_head"`, `"decode"`) to override per-component —
                all component names must be present in the dict.
            generation_config ([`~generation.GenerationConfig`], *optional*):
                Forwarded to the `generate()` capture (defaults to the model's own). Pass one with
                `cache_implementation="static"` to export against a fixed-size `StaticCache`.
            multi_token_decode (`bool`, *optional*, defaults to `False`):
                Whether the `decode` component processes multiple query tokens at once — a dynamic
                query axis (prefill on an empty cache, continuation-from-past otherwise) — vs the
                classic single-token step (see [`~exporters.utils.decompose_for_generation`]). Only
                stays dynamic under a dynamic-shape export (`config.dynamic=True`).

        Returns:
            [`ExportArtifacts`]: One [`ExportedComponent`] per stage, with the same keys as
            [`~exporters.utils.decompose_for_generation`].
        """
        components = decompose_for_generation(
            model,
            sample_inputs,
            generation_config=generation_config,
            multi_token_decode=multi_token_decode,
        )

        if isinstance(config, dict):
            missing = set(components) - set(config)
            if missing:
                raise ValueError(
                    f"Per-component `config` dict is missing entries for: {sorted(missing)}. "
                    f"Expected one entry per component: {sorted(components)}."
                )
            configs = config
        else:
            configs = dict.fromkeys(components, config)
            quantizer = config.quantizer
            if quantizer is not None and quantizer.calibration_dataset:
                # Generate-level samples: capture each component's own forward inputs from them.
                calibration = capture_calibration_inputs(
                    model,
                    quantizer.calibration_dataset,
                    generation_config=generation_config,
                    multi_token_decode=multi_token_decode,
                )
                configs = {
                    name: dataclasses.replace(config, quantizer=quantizer.with_calibration(calibration[name]))
                    for name in components
                }

        exported: dict[str, ExportedComponent] = {}
        for name, (submodel, subinputs) in components.items():
            try:
                artifact, metadata = self.export_artifact(submodel, subinputs, config=configs[name])
            except Exception as e:
                raise RuntimeError(
                    f"{self.__class__.__name__}.export_artifact failed on component '{name}' "
                    f"(submodel={type(submodel).__name__}, input keys={list(subinputs)})."
                ) from e
            exported[name] = ExportedComponent(artifact, ExportMetadata.from_dict(metadata))

        return ExportArtifacts(
            exported, self.export_format, config=getattr(model, "config", None), generation_config=generation_config
        )
