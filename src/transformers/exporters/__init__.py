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

from ..utils import _LazyModule


_import_structure = {
    "auto": ["AutoExportConfig", "AutoHfExporter", "get_hf_exporter", "register_export_config", "register_exporter"],
    "base": ["HfExporter"],
    "configs": [
        "DynamoConfig",
        "ExecutorchConfig",
        "ExportConfigMixin",
        "ExportFormat",
        "OnnxConfig",
        "OpenVINOConfig",
    ],
    "exporter_dynamo": ["DynamoExporter"],
    "exporter_executorch": ["ExecutorchExporter"],
    "exporter_onnx": ["OnnxExporter"],
    "exporter_openvino": ["OpenVINOExporter"],
    "quantizers": [
        "ExportQuantizer",
        "NNCFOnnxQuantizer",
        "NNCFOpenVINOQuantizer",
        "NNCFQuantizationConfig",
        "NNCFTorchFXQuantizer",
        "NNCFWeightCompressionConfig",
        "OnnxRuntimeDynamicConfig",
        "OnnxRuntimeQuantizer",
        "OnnxRuntimeStaticConfig",
        "PT2EQuantizer",
    ],
}

if TYPE_CHECKING:
    from .auto import AutoExportConfig, AutoHfExporter, get_hf_exporter, register_export_config, register_exporter
    from .base import HfExporter
    from .configs import DynamoConfig, ExecutorchConfig, ExportConfigMixin, ExportFormat, OnnxConfig, OpenVINOConfig
    from .exporter_dynamo import DynamoExporter
    from .exporter_executorch import ExecutorchExporter
    from .exporter_onnx import OnnxExporter
    from .exporter_openvino import OpenVINOExporter
    from .quantizers import (
        ExportQuantizer,
        NNCFOnnxQuantizer,
        NNCFOpenVINOQuantizer,
        NNCFQuantizationConfig,
        NNCFTorchFXQuantizer,
        NNCFWeightCompressionConfig,
        OnnxRuntimeDynamicConfig,
        OnnxRuntimeQuantizer,
        OnnxRuntimeStaticConfig,
        PT2EQuantizer,
    )
else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
