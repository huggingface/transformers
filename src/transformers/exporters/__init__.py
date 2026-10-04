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
from typing import TYPE_CHECKING

from ..utils import OptionalDependencyNotAvailable, _LazyModule, is_torch_available


_import_structure = {
    "configs": [
        "DynamoConfig",
        "ExecutorchConfig",
        "ExportConfigMixin",
        "ExportFormat",
        "OnnxConfig",
        "OpenVINOConfig",
    ],
}

try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["auto"] = [
        "EXPORT_BACKENDS",
        "AutoExportConfig",
        "AutoExportedModel",
        "AutoHfExporter",
        "ExportBackend",
        "export_backend",
        "register_backend",
    ]
    _import_structure["base"] = [
        "ExportArtifacts",
        "ExportedModel",
        "HfExporter",
        "ModelRunner",
    ]
    _import_structure["components"] = [
        "Component",
        "ExportedComponent",
    ]
    _import_structure["exporter_dynamo"] = ["DynamoExporter"]
    _import_structure["exporter_executorch"] = ["ExecutorchExporter"]
    _import_structure["exporter_onnx"] = ["OnnxExporter"]
    _import_structure["exporter_openvino"] = ["OpenVINOExporter"]
    _import_structure["generator"] = [
        "ExportedGenerator",
        "Modality",
    ]
    _import_structure["runner_dynamo"] = ["DynamoModelRunner"]
    _import_structure["runner_executorch"] = ["ExecutorchModelRunner"]
    _import_structure["runner_onnx"] = ["OnnxModelRunner"]
    _import_structure["runner_openvino"] = ["OpenVINOModelRunner"]


if TYPE_CHECKING:
    from .configs import (
        DynamoConfig,
        ExecutorchConfig,
        ExportConfigMixin,
        ExportFormat,
        OnnxConfig,
        OpenVINOConfig,
    )

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .auto import (
            EXPORT_BACKENDS,
            AutoExportConfig,
            AutoExportedModel,
            AutoHfExporter,
            ExportBackend,
            export_backend,
            register_backend,
        )
        from .base import ExportArtifacts, ExportedModel, HfExporter, ModelRunner
        from .components import Component, ExportedComponent
        from .exporter_dynamo import DynamoExporter
        from .exporter_executorch import ExecutorchExporter
        from .exporter_onnx import OnnxExporter
        from .exporter_openvino import OpenVINOExporter
        from .generator import ExportedGenerator, Modality
        from .runner_dynamo import DynamoModelRunner
        from .runner_executorch import ExecutorchModelRunner
        from .runner_onnx import OnnxModelRunner
        from .runner_openvino import OpenVINOModelRunner
else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
