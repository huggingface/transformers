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
"""PyTorch 2 Export (PT2E) quantization of the FX graph, for every export backend."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from ...utils.import_utils import is_torch_available, is_torchao_available
from ..configs import ExportFormat
from .base import CalibrationSet, ExportQuantizer
from .utils import traced_inputs


if is_torch_available():
    import torch

if is_torchao_available():
    from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e


class PT2EQuantizer(ExportQuantizer):
    """
    Quantize the FX graph with [PyTorch 2 Export (PT2E) quantization](https://docs.pytorch.org/ao/main/pt2e_quantization/index.html):
    `prepare_pt2e` inserts observers, the calibration set drives them, and `convert_pt2e` replaces them with
    quantize/dequantize ops. Works with every export backend.

    Args:
        quantizer (`torchao.quantization.pt2e.quantizer.Quantizer`):
            The PT2E quantizer that annotates the graph. Its quantize/dequantize ops must be ones the target backend
            supports: `X86InductorQuantizer` for inductor, ONNX (QDQ) or OpenVINO, `XNNPACKQuantizer` for
            ExecuTorch.
        calibration_dataset (`Iterable[dict]`, *optional*):
            Input dicts the observers calibrate on (see [`ExportQuantizer`]), one forward each. To calibrate on
            batches, pass a `DataLoader` that collates the samples into batched input dicts instead of a list of single
            samples.

    Example:

    ```python
    >>> from torchao.quantization.pt2e.quantizer.x86_inductor_quantizer import (
    ...     X86InductorQuantizer,
    ...     get_default_x86_inductor_quantization_config,
    ... )
    >>> from transformers.exporters import DynamoConfig, PT2EQuantizer

    >>> x86 = X86InductorQuantizer().set_global(get_default_x86_inductor_quantization_config())
    >>> DynamoConfig(quantizer=PT2EQuantizer(x86, calibration_dataset=samples))
    ```
    """

    quantizes_fx_graph = True
    supported_formats = (ExportFormat.DYNAMO, ExportFormat.ONNX, ExportFormat.EXECUTORCH, ExportFormat.OPENVINO)

    required_packages = ("torch", "torchao")
    tested_versions = {"torch": "2.13.0", "torchao": "0.18.0"}
    min_versions = {"torchao": "0.11.0"}

    def __init__(self, quantizer: Any, calibration_dataset: Iterable[dict[str, Any]] | None = None):
        super().__init__(calibration_dataset)
        self.quantizer = quantizer

    def quantize(
        self, model: torch.fx.GraphModule, sample_inputs: dict[str, Any], export_format: ExportFormat
    ) -> torch.fx.GraphModule:
        traced_names = list(sample_inputs)
        calibration = CalibrationSet(
            self.calibration_dataset, sample_inputs, lambda sample: traced_inputs(sample, traced_names)
        )
        prepared = prepare_pt2e(model, self.quantizer)
        with torch.no_grad():
            for inputs in calibration:
                prepared(**inputs)
        # OpenVINO turns a weight's quantize/dequantize pair into a `FakeQuantize` and compresses it itself, but has
        # no conversion for the lone `dequantize` a folded int8 weight leaves behind.
        return convert_pt2e(prepared, fold_quantize=export_format is not ExportFormat.OPENVINO)
