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
from .base import ExportedModel, HfExporter, ModelRunner
from .configs import ExportConfigMixin, ExportFormat
from .exporter_dynamo import DynamoConfig, DynamoExporter
from .exporter_executorch import ExecutorchConfig, ExecutorchExporter
from .exporter_onnx import OnnxConfig, OnnxExporter
from .exporter_openvino import OpenVINOConfig, OpenVINOExporter
from .runner_dynamo import DynamoModelRunner
from .runner_executorch import ExecutorchModelRunner
from .runner_onnx import OnnxModelRunner
from .runner_openvino import OpenVINOModelRunner


logger = logging.get_logger(__name__)


@dataclass
class ExportBackend:
    """One export format's config, exporter, and runner (`None` for an export-only format)."""

    config: type[ExportConfigMixin]
    exporter: type[HfExporter]
    runner: type[ModelRunner] | None = None


EXPORT_BACKENDS: dict[str, ExportBackend] = {
    "executorch": ExportBackend(ExecutorchConfig, ExecutorchExporter, ExecutorchModelRunner),
    "dynamo": ExportBackend(DynamoConfig, DynamoExporter, DynamoModelRunner),
    "onnx": ExportBackend(OnnxConfig, OnnxExporter, OnnxModelRunner),
    "openvino": ExportBackend(OpenVINOConfig, OpenVINOExporter, OpenVINOModelRunner),
}


def register_backend(
    name: str,
    config: type[ExportConfigMixin],
    exporter: type[HfExporter],
    runner: type[ModelRunner] | None = None,
) -> None:
    """Register a format's config, exporter and runner, so every auto class and loader finds them. Without a
    runner the format can export and save, but not run or load its artifacts."""
    for part, cls, base in (
        ("config", config, ExportConfigMixin),
        ("exporter", exporter, HfExporter),
        ("runner", runner, ModelRunner),
    ):
        if part == "runner" and cls is None:
            continue
        if not (isinstance(cls, type) and issubclass(cls, base)):
            raise TypeError(f"The {part} must extend {base.__name__}, got {cls!r}.")
    if name in EXPORT_BACKENDS:
        logger.warning(f"Export backend '{name}' is already registered and will be overwritten.")
    EXPORT_BACKENDS[name] = ExportBackend(config, exporter, runner)


def export_backend(export_format, part: str):
    """One part (`"config"`, `"exporter"`, `"runner"`) of the backend registered for a format (an [`ExportFormat`]
    or its string value)."""
    if export_format is None:
        raise ValueError(f"No export format given — registered formats are {sorted(EXPORT_BACKENDS)}.")
    name = export_format.value if isinstance(export_format, ExportFormat) else export_format
    backend = EXPORT_BACKENDS.get(name)
    if backend is None:
        raise ValueError(f"Unknown export format '{name}' — registered formats are {sorted(EXPORT_BACKENDS)}.")
    registered = getattr(backend, part)
    if registered is None:
        raise ValueError(f"The '{name}' backend registers no {part}, so its artifacts can only be exported and saved.")
    return registered


class AutoExportConfig:
    """Dispatches an export config stored as a dict to the right config class."""

    @classmethod
    def from_dict(cls, export_config_dict: dict):
        return export_backend(export_config_dict.get("export_format"), "config").from_dict(export_config_dict)


class AutoHfExporter:
    """Instantiates the `HfExporter` matching an export config."""

    @classmethod
    def from_config(cls, export_config: ExportConfigMixin | dict, **kwargs) -> HfExporter:
        export_config_dict = export_config.to_dict() if isinstance(export_config, ExportConfigMixin) else export_config
        return export_backend(export_config_dict.get("export_format"), "exporter")(**kwargs)


class AutoExportedModel:
    """Load a saved export as an [`ExportedGenerator`] if it has a decode graph, else an [`ExportedModel`].

    Example:
        runtime = AutoExportedModel.from_pretrained("out/")
    """

    @classmethod
    def from_pretrained(cls, save_directory, **kwargs):
        """Load a saved export from a local directory or a Hub repo."""
        from .base import read_export_manifest, split_download_kwargs
        from .generator import ExportedGenerator

        download_kwargs, _ = split_download_kwargs(dict(kwargs))
        manifest = read_export_manifest(save_directory, **download_kwargs)
        can_generate = "decode" in manifest["components"]
        target = ExportedGenerator if can_generate else ExportedModel
        return target.from_pretrained(save_directory, **kwargs)
