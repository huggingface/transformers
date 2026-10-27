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
from typing import TYPE_CHECKING

from ...utils import _LazyModule


_import_structure = {
    "base": ["ExportedModel", "ModelRunner"],
    "runner_dynamo": ["DynamoModelRunner"],
    "runner_executorch": ["ExecutorchModelRunner"],
    "runner_onnx": ["OnnxModelRunner"],
    "runner_openvino": ["OpenVINOModelRunner"],
}

if TYPE_CHECKING:
    from .base import ExportedModel, ModelRunner
    from .runner_dynamo import DynamoModelRunner
    from .runner_executorch import ExecutorchModelRunner
    from .runner_onnx import OnnxModelRunner
    from .runner_openvino import OpenVINOModelRunner
else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
