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
"""Auto exporter factory for HuggingFace exporters."""

from __future__ import annotations

from dataclasses import dataclass

from ..utils import logging
from .base import HfExporter
from .configs import (
    DynamoConfig,
    ExecutorchConfig,
    ExportConfigMixin,
    ExportFormat,
    OnnxConfig,
    OpenVINOConfig,
)
from .runners.base import ModelRunner
from .utils import import_dotted_path


logger = logging.get_logger(__name__)


@dataclass
class ExportBackend:
    """One export format's config, exporter, and runner (`None` for a format that exports but can't run). The
    built-in exporters and runners are `"module.ClassName"` paths in this package, imported on first use so only the
    backend in use loads its dependencies."""

    config: type[ExportConfigMixin]
    exporter: type[HfExporter] | str
    runner: type[ModelRunner] | str | None = None


EXPORT_BACKENDS: dict[str, ExportBackend] = {
    "executorch": ExportBackend(
        ExecutorchConfig, "exporter_executorch.ExecutorchExporter", "runners.runner_executorch.ExecutorchModelRunner"
    ),
    "openvino": ExportBackend(
        OpenVINOConfig, "exporter_openvino.OpenVINOExporter", "runners.runner_openvino.OpenVINOModelRunner"
    ),
    "dynamo": ExportBackend(DynamoConfig, "exporter_dynamo.DynamoExporter", "runners.runner_dynamo.DynamoModelRunner"),
    "onnx": ExportBackend(OnnxConfig, "exporter_onnx.OnnxExporter", "runners.runner_onnx.OnnxModelRunner"),
}


def register_backend(exporter: type[HfExporter], runner: type[ModelRunner] | None = None) -> None:
    """Register an exporter, with its `config_class`, under its `export_format`, and the runner that runs its exports,
    so the auto classes find them."""
    if not (isinstance(exporter, type) and issubclass(exporter, HfExporter)):
        raise TypeError(f"The exporter must extend HfExporter, got {exporter!r}.")
    if not (isinstance(exporter.config_class, type) and issubclass(exporter.config_class, ExportConfigMixin)):
        raise TypeError(f"The exporter's `config_class` must extend ExportConfigMixin, got {exporter.config_class!r}.")
    if runner is not None and not (isinstance(runner, type) and issubclass(runner, ModelRunner)):
        raise TypeError(f"The runner must extend ModelRunner, got {runner!r}.")
    if exporter.export_format in EXPORT_BACKENDS:
        logger.warning(f"Export backend '{exporter.export_format}' is already registered and will be overwritten.")
    EXPORT_BACKENDS[exporter.export_format] = ExportBackend(exporter.config_class, exporter, runner)


def _backend_for(export_format: ExportFormat | str | None) -> ExportBackend:
    if export_format not in EXPORT_BACKENDS:
        raise ValueError(
            f"Unknown export format {export_format!r} — registered formats are {sorted(EXPORT_BACKENDS)}."
        )
    return EXPORT_BACKENDS[export_format]


def _resolve_class(value: type | str) -> type:
    """A registered exporter or runner: the class itself, or imported from its `"module.ClassName"` path."""
    return import_dotted_path(f"{__package__}.{value}") if isinstance(value, str) else value


def get_exporter_class(export_format: ExportFormat | str | None) -> type[HfExporter]:
    """The exporter class registered for a format, imported on first use."""
    return _resolve_class(_backend_for(export_format).exporter)


def get_runner_class(export_format: ExportFormat | str | None) -> type[ModelRunner]:
    """The runner class registered for a format, imported on first use."""
    backend = _backend_for(export_format)
    if backend.runner is None:
        raise ValueError(f"Export format '{export_format}' has no runner registered, so its exports can't run.")
    return _resolve_class(backend.runner)


class AutoExportConfig:
    """
    The Auto-HF export config class that takes care of automatically dispatching to the correct
    export config given an export config stored in a dictionary.
    """

    @classmethod
    def from_dict(cls, export_config_dict: dict):
        return _backend_for(export_config_dict.get("export_format")).config.from_dict(export_config_dict)


class AutoHfExporter:
    """
    The Auto-HF exporter class that takes care of automatically instantiating to the correct
    `HfExporter` given the `ExportConfig`.
    """

    @classmethod
    def from_config(cls, export_config: ExportConfigMixin | dict, **kwargs) -> HfExporter:
        if isinstance(export_config, ExportConfigMixin):
            return get_exporter_class(export_config.export_format)(**kwargs)
        return get_exporter_class(export_config.get("export_format"))(**kwargs)
