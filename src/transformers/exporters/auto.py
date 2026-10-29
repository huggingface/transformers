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


def register_backend(name: str, config: type[ExportConfigMixin], exporter: type[HfExporter]) -> None:
    """Register a format's config and exporter, so the auto classes find them."""
    for part, cls, base in (("config", config, ExportConfigMixin), ("exporter", exporter, HfExporter)):
        if not (isinstance(cls, type) and issubclass(cls, base)):
            raise TypeError(f"The {part} must extend {base.__name__}, got {cls!r}.")
    if name in EXPORT_BACKENDS:
        logger.warning(f"Export backend '{name}' is already registered and will be overwritten.")
    EXPORT_BACKENDS[name] = ExportBackend(config, exporter)


def _registered_backend(export_format: ExportFormat | str | None) -> ExportBackend:
    if export_format is None:
        raise ValueError(f"No export format given — registered formats are {sorted(EXPORT_BACKENDS)}.")
    backend = EXPORT_BACKENDS.get(export_format)
    if backend is None:
        raise ValueError(
            f"Unknown export format '{export_format}' — registered formats are {sorted(EXPORT_BACKENDS)}."
        )
    return backend


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

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs) -> HfExporter | None:
        """
        Load an exporter instance from a pretrained model/checkpoint that ships an export config.

        **Not implemented yet** — placeholder for a first-class "export recipe" workflow.

        The idea: model owners publish an ``export_config.json`` (or an ``export_config`` field in
        ``config.json``) alongside their weights on the Hub. That file captures the settings the
        owner has already validated for their architecture — the target format (``dynamo`` /
        ``onnx`` / ``executorch``), exact dynamic-shape specs (e.g. ``text_ids`` dynamic to 4096,
        image tiles fixed at 448, ``batch=1`` for edge deployment), ``strict`` flag, ONNX opset,
        prefill vs. decode layout, ExecuTorch backend choice, and any other knob that today lives
        as tribal knowledge in a README or a private notebook.

        Consumers then get the owner-validated export in one call::

            exporter = AutoHfExporter.from_pretrained("org/model-name")
            program = exporter.export(model, inputs)

        Composes with the [`register_export_input_preparer`] registry: the owner supplies the
        shape spec via ``export_config.json``, transformers supplies the data-dependent
        precomputations (``cu_seqlens``, vision position ids, window indices, …) for that
        architecture. Together they cover the two hard parts of exporting new models — knowing
        the right shape contract and preparing the right inputs — so downstream users don't
        re-derive either from scratch (and don't break in production when they get it wrong).
        """
        raise NotImplementedError(
            "AutoHfExporter.from_pretrained is not implemented yet. "
            "Load/export configs explicitly and call AutoHfExporter.from_config(...) instead."
        )
