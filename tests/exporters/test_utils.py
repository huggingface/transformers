# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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
"""Tests for the auto-factory and config serialisation utilities in `transformers.exporters`."""

import pytest

from transformers.exporters.auto import (
    AUTO_EXPORT_CONFIG_MAPPING,
    AUTO_EXPORTER_MAPPING,
    AutoExportConfig,
    AutoHfExporter,
    register_export_config,
    register_exporter,
)
from transformers.exporters.base import HfExporter
from transformers.exporters.configs import (
    DynamoConfig,
    ExecutorchConfig,
    ExportConfigMixin,
    ExportFormat,
    OnnxConfig,
)
from transformers.exporters.exporter_dynamo import DynamoExporter
from transformers.exporters.exporter_executorch import ExecutorchExporter
from transformers.exporters.exporter_onnx import OnnxExporter


CONCRETE_CONFIGS = [
    (DynamoConfig, ExportFormat.DYNAMO),
    (OnnxConfig, ExportFormat.ONNX),
    (ExecutorchConfig, ExportFormat.EXECUTORCH),
]


class TestExportConfigMixin:
    @pytest.mark.parametrize("config_cls,fmt", CONCRETE_CONFIGS)
    def test_to_dict_from_dict_roundtrip_enum(self, config_cls, fmt):
        original = config_cls(dynamic=True)
        restored = config_cls.from_dict(original.to_dict())
        assert restored == original
        assert restored.export_format is fmt

    def test_to_dict_is_a_copy(self):
        cfg = DynamoConfig(dynamic_shapes={"x": 0})
        d = cfg.to_dict()
        d["dynamic_shapes"]["y"] = 1
        assert cfg.dynamic_shapes == {"x": 0}


class TestAutoExportConfig:
    @pytest.mark.parametrize("config_cls,fmt", CONCRETE_CONFIGS)
    def test_from_dict_dispatches_to_concrete_config(self, config_cls, fmt):
        cfg = AutoExportConfig.from_dict({"export_format": fmt.value})
        assert isinstance(cfg, config_cls)

    @pytest.mark.parametrize("config_cls,fmt", CONCRETE_CONFIGS)
    def test_from_dict_accepts_enum_value(self, config_cls, fmt):
        cfg = AutoExportConfig.from_dict({"export_format": fmt})
        assert isinstance(cfg, config_cls)

    def test_from_dict_missing_export_format_raises(self):
        with pytest.raises(ValueError, match="export_format"):
            AutoExportConfig.from_dict({})

    def test_from_dict_unknown_format_raises(self):
        with pytest.raises(ValueError, match="Unknown exporter type"):
            AutoExportConfig.from_dict({"export_format": "openvino_ish"})


class TestAutoHfExporter:
    @pytest.mark.parametrize(
        "config,exporter_cls",
        [
            (DynamoConfig(), DynamoExporter),
            (OnnxConfig(), OnnxExporter),
            (ExecutorchConfig(), ExecutorchExporter),
        ],
    )
    def test_from_config_with_dataclass_instance(self, config, exporter_cls):
        exporter = AutoHfExporter.from_config(config)
        assert isinstance(exporter, exporter_cls)

    @pytest.mark.parametrize("config_cls,fmt", CONCRETE_CONFIGS)
    def test_from_config_with_dict_input(self, config_cls, fmt):
        exporter = AutoHfExporter.from_config({"export_format": fmt.value})
        expected = AUTO_EXPORTER_MAPPING[fmt.value]
        assert isinstance(exporter, expected)

    def test_from_config_raises_on_unknown_format(self):
        with pytest.raises(ValueError, match="Unsupported export config"):
            AutoHfExporter.from_config({"export_format": "not_a_real_backend"})
        with pytest.raises(ValueError, match="Unsupported export config"):
            AutoHfExporter.from_config({})

    def test_from_pretrained_is_not_implemented(self):
        with pytest.raises(NotImplementedError, match="not implemented"):
            AutoHfExporter.from_pretrained("dummy/path")

    def test_supports_export_format(self):
        assert AutoHfExporter.supports_export_format({"export_format": "dynamo"})
        assert AutoHfExporter.supports_export_format({"export_format": ExportFormat.DYNAMO})
        assert not AutoHfExporter.supports_export_format({"export_format": "not_a_real_backend"})
        assert not AutoHfExporter.supports_export_format({})


class TestRegistration:
    def _cleanup(self, name):
        AUTO_EXPORTER_MAPPING.pop(name, None)
        AUTO_EXPORT_CONFIG_MAPPING.pop(name, None)

    def test_register_exporter_adds_to_mapping(self):
        try:

            @register_exporter("stub_exporter")
            class _StubExporter(HfExporter):
                required_packages = []

                def export(self, model, sample_inputs, config):
                    return None

            assert AUTO_EXPORTER_MAPPING["stub_exporter"] is _StubExporter
        finally:
            self._cleanup("stub_exporter")

    def test_register_exporter_rejects_non_subclass(self):
        try:
            with pytest.raises(TypeError, match="HfExporter"):

                @register_exporter("bad")
                class _NotAnExporter:
                    pass

        finally:
            self._cleanup("bad")

    def test_register_export_config_adds_to_mapping(self):
        try:

            @register_export_config("stub_config")
            class _StubConfig(ExportConfigMixin):
                export_format: str = "stub_config"

            assert AUTO_EXPORT_CONFIG_MAPPING["stub_config"] is _StubConfig
        finally:
            self._cleanup("stub_config")

    def test_register_export_config_rejects_non_subclass(self):
        try:
            with pytest.raises(TypeError, match="ExportConfigMixin"):

                @register_export_config("bad_config")
                class _NotAConfig:
                    pass

        finally:
            self._cleanup("bad_config")
