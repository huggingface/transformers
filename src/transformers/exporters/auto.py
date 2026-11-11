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

import importlib
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


logger = logging.get_logger(__name__)


@dataclass
class ExportBackend:
    """One export format's config and exporter. The built-in exporters are `"module.ClassName"` paths in this
    package, imported on first use so only the backend in use loads its dependencies."""

    config: type[ExportConfigMixin]
    exporter: type[HfExporter] | str


EXPORT_BACKENDS: dict[str, ExportBackend] = {
    "executorch": ExportBackend(ExecutorchConfig, "exporter_executorch.ExecutorchExporter"),
    "openvino": ExportBackend(OpenVINOConfig, "exporter_openvino.OpenVINOExporter"),
    "dynamo": ExportBackend(DynamoConfig, "exporter_dynamo.DynamoExporter"),
    "onnx": ExportBackend(OnnxConfig, "exporter_onnx.OnnxExporter"),
}


def register_backend(exporter: type[HfExporter]) -> None:
    """Register an exporter, with its `config_class`, under its `export_format`, so the auto classes find them."""
    if not (isinstance(exporter, type) and issubclass(exporter, HfExporter)):
        raise TypeError(f"The exporter must extend HfExporter, got {exporter!r}.")
    if not (isinstance(exporter.config_class, type) and issubclass(exporter.config_class, ExportConfigMixin)):
        raise TypeError(f"The exporter's `config_class` must extend ExportConfigMixin, got {exporter.config_class!r}.")
    if exporter.export_format in EXPORT_BACKENDS:
        logger.warning(f"Export backend '{exporter.export_format}' is already registered and will be overwritten.")
    EXPORT_BACKENDS[exporter.export_format] = ExportBackend(exporter.config_class, exporter)


def _registered_backend(export_format: ExportFormat | str | None) -> ExportBackend:
    if export_format not in EXPORT_BACKENDS:
        raise ValueError(
            f"Unknown export format {export_format!r} — registered formats are {sorted(EXPORT_BACKENDS)}."
        )
    return EXPORT_BACKENDS[export_format]


def get_export_config_class(export_format: ExportFormat | str | None) -> type[ExportConfigMixin]:
    """The config class registered for a format."""
    return _registered_backend(export_format).config


def get_exporter_class(export_format: ExportFormat | str | None) -> type[HfExporter]:
    """The exporter class registered for a format, imported on first use."""
    backend = _registered_backend(export_format)
    if isinstance(backend.exporter, str):
        module_name, class_name = backend.exporter.rsplit(".", 1)
        backend.exporter = getattr(importlib.import_module(f".{module_name}", __package__), class_name)
    return backend.exporter


class AutoExportConfig:
    """
    The Auto-HF export config class that takes care of automatically dispatching to the correct
    export config given an export config stored in a dictionary.
    """

    @classmethod
    def from_dict(cls, export_config_dict: dict):
        return get_export_config_class(export_config_dict.get("export_format")).from_dict(export_config_dict)


class AutoHfExporter:
    """
    The Auto-HF exporter class that takes care of automatically instantiating to the correct
    `HfExporter` given the `ExportConfig`.
    """

    @classmethod
    def from_config(cls, export_config: ExportConfigMixin | dict, **kwargs) -> HfExporter:
        export_config_dict = export_config.to_dict() if isinstance(export_config, ExportConfigMixin) else export_config
        return get_exporter_class(export_config_dict.get("export_format"))(**kwargs)
