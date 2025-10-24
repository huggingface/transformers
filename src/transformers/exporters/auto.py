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
"""Auto exporter factory for HuggingFace exporters."""

from typing import Optional, Union

from ..utils import logging
from ..utils.export_config import ExportConfigMixin, ExportFormat
from .base import HfExporter
from .exporter_dynamo import DynamoConfig, DynamoExporter
from .exporter_onnx import OnnxConfig, OnnxExporter


AUTO_EXPORTER_MAPPING = {
    # "executorch": ExecutorchExporter,
    "dynamo": DynamoExporter,
    "onnx": OnnxExporter,
}

AUTO_EXPORT_CONFIG_MAPPING = {
    # "executorch": ExecutorchConfig,
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
            raise ValueError(
                "export_config_dict must contain key 'export_format' (preferred) or 'exporter' set to exporter name"
            )

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
    def from_config(cls, export_config: Union[ExportConfigMixin, dict], **kwargs) -> Optional[HfExporter]:
        # Convert it to a ExportConfigMixin if the q_config is a dict
        if isinstance(export_config, dict):
            export_config = AutoExportConfig.from_dict(export_config)

        export_format = export_config.export_format

        target_cls = AUTO_EXPORTER_MAPPING[export_format]
        return target_cls(export_config, **kwargs)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs) -> Optional[HfExporter]:
        """
        Load an exporter instance from a pretrained model/config that contains an export config.
        This will look for common attributes on the model config (see `AutoExportConfig.from_pretrained`).
        """
        export_config = AutoExportConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)
        return cls.from_config(export_config.__dict__, **kwargs)

    @staticmethod
    def supports_export_format(export_config_dict: dict) -> bool:
        """Return True if the provided dict describes a supported export_format.

        Accepts both `export_format` and legacy `exporter` keys, and ExportFormat enum values.
        """
        export_fmt = export_config_dict.get("export_format", export_config_dict.get("exporter"))
        if export_fmt is None:
            return False

        if isinstance(export_fmt, ExportFormat):
            name = export_fmt.value
        else:
            name = export_fmt

        if name not in AUTO_EXPORT_CONFIG_MAPPING:
            logger.warning(
                f"Unknown export format, got {export_fmt} - supported types are: {list(AUTO_EXPORTER_MAPPING.keys())}. Skipping."
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
