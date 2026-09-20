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

import json
from dataclasses import dataclass

from ..models.auto import AutoConfig
from ..utils import logging
from .base import ExportedModel, HfExporter, ModelRunner
from .configs import ExportConfigMixin, ExportFormat
from .exporter_dynamo import DynamoConfig, DynamoExporter
from .exporter_executorch import ExecutorchConfig, ExecutorchExporter
from .exporter_onnx import OnnxConfig, OnnxExporter
from .runner_dynamo import DynamoModelRunner
from .runner_executorch import ExecutorchModelRunner
from .runner_onnx import OnnxModelRunner


# The recipe a model owner publishes next to their weights, naming the export they validated.
EXPORT_CONFIG_NAME = "export_config.json"


@dataclass
class ExportBackend:
    """What one export format is made of: the config that parameterizes it, the exporter that writes it,
    and the runner that runs what was written.

    One entry per format rather than three parallel tables keyed by the same strings, so a backend cannot be
    half-registered without it being visible — which it was: there used to be no way to register a runner at
    all, so a third-party backend could export and save artifacts that nothing could load.
    """

    config: type[ExportConfigMixin] | None = None
    exporter: type[HfExporter] | None = None
    runner: type | None = None


EXPORT_BACKENDS: dict[str, ExportBackend] = {
    "executorch": ExportBackend(ExecutorchConfig, ExecutorchExporter, ExecutorchModelRunner),
    "dynamo": ExportBackend(DynamoConfig, DynamoExporter, DynamoModelRunner),
    "onnx": ExportBackend(OnnxConfig, OnnxExporter, OnnxModelRunner),
}


def export_backend(export_format, part: str | None = None):
    """The registered backend for a format, or one named part of it, with an error that says what is missing.

    `export_format` takes an [`ExportFormat`] or its string value, since a manifest carries the string and
    a config carries the enum.
    """
    if export_format is None:
        raise ValueError(f"No export format given — registered formats are {sorted(EXPORT_BACKENDS)}.")
    name = export_format.value if isinstance(export_format, ExportFormat) else export_format
    backend = EXPORT_BACKENDS.get(name)
    if backend is None:
        raise ValueError(f"Unknown export format '{name}' — registered formats are {sorted(EXPORT_BACKENDS)}.")
    if part is None:
        return backend
    registered = getattr(backend, part)
    if registered is None:
        raise ValueError(
            f"The '{name}' backend has no {part} registered, so it cannot be used for this. Register one "
            f"with `register_{part}('{name}')`."
        )
    return registered


logger = logging.get_logger(__name__)


class AutoExportConfig:
    """
    The Auto-HF export config class that takes care of automatically dispatching to the correct
    export config given an export config stored in a dictionary.
    """

    @classmethod
    def from_dict(cls, export_config_dict: dict):
        # `export_backend` takes the enum or its string value, and says what is missing if anything is --
        # including the key itself, so the absent case is not re-checked here.
        return export_backend(export_config_dict.get("export_format"), "config").from_dict(export_config_dict)


class AutoHfExporter:
    """
    The Auto-HF expoerter class that takes care of automatically instantiating to the correct
    `HfExporter` given the `ExportConfig`.
    """

    @classmethod
    def from_config(cls, export_config: ExportConfigMixin | dict, **kwargs) -> HfExporter:
        export_config_dict = export_config.to_dict() if isinstance(export_config, ExportConfigMixin) else export_config
        return export_backend(export_config_dict.get("export_format"), "exporter")(**kwargs)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs) -> HfExporter:
        """Build the exporter a checkpoint's own export recipe asks for.

        A model owner publishes an `export_config.json` next to their weights (or an `export_config` field
        in `config.json`) recording the settings they validated for their architecture — the target format,
        the dynamic-shape spec, opset, ExecuTorch backend, and the rest of what otherwise lives in a README.
        Consumers get that export in one call instead of re-deriving it:

            exporter = AutoHfExporter.from_pretrained("org/model-name")
            program = exporter.export(model, inputs)

        `kwargs` are split: anything naming an export-config field overrides the recipe, everything else is
        forwarded to the download and to the exporter's constructor.
        """
        config_dict = cls._load_export_config_dict(pretrained_model_name_or_path, **kwargs)
        overrides = {key: kwargs.pop(key) for key in list(kwargs) if key in config_dict}
        config_dict = {**config_dict, **overrides}
        return cls.from_config(AutoExportConfig.from_dict(config_dict), **kwargs)

    @staticmethod
    def _load_export_config_dict(pretrained_model_name_or_path, **kwargs) -> dict:
        """Find the export recipe: a standalone `export_config.json`, else an `export_config` field on the
        model config. Local directories and Hub repos both go through `cached_file`, which resolves either."""
        from .base import resolve_export_file, split_download_kwargs

        download_kwargs, _ = split_download_kwargs(dict(kwargs))
        resolved = resolve_export_file(pretrained_model_name_or_path, EXPORT_CONFIG_NAME, **download_kwargs)
        if resolved is not None:
            with open(resolved, encoding="utf-8") as file:
                return json.load(file)

        config = AutoConfig.from_pretrained(pretrained_model_name_or_path, **download_kwargs)
        export_config = getattr(config, "export_config", None)
        if export_config is None:
            raise OSError(
                f"{pretrained_model_name_or_path} ships no export recipe: no `{EXPORT_CONFIG_NAME}` and no "
                "`export_config` field in its `config.json`. Build the config yourself and call "
                "`AutoHfExporter.from_config(...)`."
            )
        return dict(export_config)


class AutoExportedModel:
    """Load a saved export as whatever it was exported as.

    The manifest records the `kind`, so this picks the same shape the export produced without the caller
    having to remember: an [`ExportedGenerator`] for a decomposed, cache-driven export, an
    [`ExportedModel`] for a single graph (a classifier, an encoder, a feature extractor).

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
        # Older manifests predate `kind`; a decomposed export is the one with more than one component.
        kind = manifest.get("kind") or ("generation" if len(manifest["components"]) > 1 else "model")
        target = ExportedGenerator if kind == "generation" else ExportedModel
        return target.from_pretrained(save_directory, **kwargs)


def _register(name: str, part: str, base: type):
    """Fill one slot of a format's [`ExportBackend`], creating the entry if this is its first part."""

    def register(cls):
        if not issubclass(cls, base):
            raise TypeError(f"{part.capitalize()} must extend {base.__name__}")
        backend = EXPORT_BACKENDS.setdefault(name, ExportBackend())
        if getattr(backend, part) is not None:
            logger.warning(f"{part.capitalize()} for '{name}' is already registered and will be overwritten.")
        setattr(backend, part, cls)
        return cls

    return register


def register_exporter(name: str):
    """Register the exporter that writes a format."""
    return _register(name, "exporter", HfExporter)


def register_export_config(name: str):
    """Register the config that parameterizes a format."""
    return _register(name, "config", ExportConfigMixin)


def register_runner(name: str):
    """Register the runner that runs a saved artifact of a format.

    Without one a backend can export and save, but nothing can load what it wrote.
    """
    return _register(name, "runner", ModelRunner)


def get_hf_exporter(export_config) -> HfExporter:
    return AutoHfExporter.from_config(export_config)
