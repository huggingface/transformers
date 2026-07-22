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

from ..utils import logging
from .base import HfExporter
from .configs import ExportConfigMixin, ExportFormat
from .exporter_dynamo import DynamoConfig, DynamoExporter
from .exporter_executorch import ExecutorchConfig, ExecutorchExporter
from .exporter_onnx import OnnxConfig, OnnxExporter


AUTO_EXPORTER_MAPPING = {
    "executorch": ExecutorchExporter,
    "dynamo": DynamoExporter,
    "onnx": OnnxExporter,
}

AUTO_EXPORT_CONFIG_MAPPING = {
    "executorch": ExecutorchConfig,
    "dynamo": DynamoConfig,
    "onnx": OnnxConfig,
}

logger = logging.get_logger(__name__)


class AutoExportConfig:
    """
    The Auto-HF export config class that takes care of automatically dispatching to the correct
    export config given an export config stored in a dictionary.
    """

    @classmethod
    def from_dict(cls, export_config_dict: dict):
        export_format = export_config_dict.get("export_format")

        if export_format is None:
            raise ValueError("export_config_dict must contain key 'export_format' set to exporter name")

        # Allow passing an ExportFormat enum value or a plain string
        if isinstance(export_format, ExportFormat):
            name = export_format.value
        else:
            name = export_format

        if name not in AUTO_EXPORT_CONFIG_MAPPING:
            raise ValueError(
                f"Unknown exporter type, got {name} - supported exporters are: {list(AUTO_EXPORT_CONFIG_MAPPING.keys())}"
            )

        target_cls = AUTO_EXPORT_CONFIG_MAPPING[name]
        return target_cls.from_dict(export_config_dict)


class AutoHfExporter:
    """
    The Auto-HF expoerter class that takes care of automatically instantiating to the correct
    `HfExporter` given the `ExportConfig`.
    """

    @classmethod
    def from_config(cls, export_config: ExportConfigMixin | dict, **kwargs) -> HfExporter:
        # Normalize to a dict so ``supports_export_format`` can act as the single gate.
        export_config_dict = export_config.to_dict() if isinstance(export_config, ExportConfigMixin) else export_config
        if not cls.supports_export_format(export_config_dict):
            raise ValueError(
                f"Unsupported export config: {export_config_dict!r}. "
                f"Registered exporters: {sorted(AUTO_EXPORTER_MAPPING)}."
            )

        export_format = export_config_dict["export_format"]
        name = export_format.value if isinstance(export_format, ExportFormat) else export_format
        return AUTO_EXPORTER_MAPPING[name](**kwargs)

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

    @staticmethod
    def supports_export_format(export_config_dict: dict) -> bool:
        """Return True if the provided dict describes an ``export_format`` that has both a
        registered config class and a registered exporter class. Warns with an actionable message
        when the format is missing entirely, unknown, or only half-registered."""
        export_fmt = export_config_dict.get("export_format")
        if export_fmt is None:
            logger.warning(
                "No 'export_format' key in export config — supported values are: "
                f"{sorted(AUTO_EXPORTER_MAPPING)}. Skipping."
            )
            return False

        name = export_fmt.value if isinstance(export_fmt, ExportFormat) else export_fmt
        has_config = name in AUTO_EXPORT_CONFIG_MAPPING
        has_exporter = name in AUTO_EXPORTER_MAPPING

        if not has_config and not has_exporter:
            logger.warning(
                f"Unknown export format {export_fmt!r} — supported values are: "
                f"{sorted(set(AUTO_EXPORTER_MAPPING) & set(AUTO_EXPORT_CONFIG_MAPPING))}. Skipping."
            )
            return False
        if not has_config:
            logger.warning(
                f"Export format {name!r} has a registered exporter but no config class. "
                f"Register one via ``@register_export_config({name!r})``. Skipping."
            )
            return False
        if not has_exporter:
            logger.warning(
                f"Export format {name!r} has a registered config class but no exporter. "
                f"Register one via ``@register_exporter({name!r})``. Skipping."
            )
            return False
        return True


def register_exporter(name: str):
    def register_exporter_fn(cls):
        if name in AUTO_EXPORTER_MAPPING:
            logger.warning(f"Exporter '{name}' is already registered and will be overwritten.")
        if not issubclass(cls, HfExporter):
            raise TypeError("Exporter must extend HfExporter")
        AUTO_EXPORTER_MAPPING[name] = cls
        return cls

    return register_exporter_fn


def register_export_config(name: str):
    def register_export_config_fn(cls):
        if name in AUTO_EXPORT_CONFIG_MAPPING:
            logger.warning(f"Export config '{name}' is already registered and will be overwritten.")
        if not issubclass(cls, ExportConfigMixin):
            raise TypeError("Export config must extend ExportConfigMixin")
        AUTO_EXPORT_CONFIG_MAPPING[name] = cls
        return cls

    return register_export_config_fn


def get_hf_exporter(export_config) -> HfExporter:
    return AutoHfExporter.from_config(export_config)
