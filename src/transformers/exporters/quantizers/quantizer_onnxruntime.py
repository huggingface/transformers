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
"""ONNX Runtime quantization of a converted ONNX model."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

from ..configs import ExportFormat
from .base import CalibrationSet, ExportQuantizer, QuantizationStage


if TYPE_CHECKING:
    import onnx


class OnnxRuntimeQuantizer(ExportQuantizer):
    """
    Quantize the converted ONNX model with ONNX Runtime's own tools (`onnxruntime.quantization`), which work on the
    ONNX graph rather than the PyTorch one.

    By default it runs `quantize_static` (int8 QDQ activations and weights, calibrated on the config's
    `calibration_dataset`); with `dynamic=True` it runs `quantize_dynamic` (int8 weights, activations quantized at
    runtime, no calibration). Other keyword arguments go to that function.

    Example:

    ```python
    >>> from transformers.exporters import OnnxConfig, OnnxRuntimeQuantizer

    >>> OnnxConfig(quantizer=OnnxRuntimeQuantizer(per_channel=True, calibration_dataset=samples))
    >>> OnnxConfig(quantizer=OnnxRuntimeQuantizer(dynamic=True))
    ```
    """

    stage = QuantizationStage.BACKEND
    supported_formats = (ExportFormat.ONNX,)
    required_packages = ("onnxruntime",)

    def __init__(self, dynamic: bool = False, calibration_dataset: Iterable[dict[str, Any]] | None = None, **kwargs):
        super().__init__(calibration_dataset)
        self.dynamic = dynamic
        self.kwargs = kwargs

    def _quantize(
        self, model: onnx.ModelProto, calibration: CalibrationSet, export_format: ExportFormat
    ) -> onnx.ModelProto:
        import onnx
        from onnxruntime.quantization import quantize_dynamic, quantize_static

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "model.onnx")
            if self.dynamic:
                _drop_initializer_shapes(model)
                quantize_dynamic(model, path, **self.kwargs)
            else:
                quantize_static(model, path, _FeedReader(calibration), **self.kwargs)
            return onnx.load(path)


class _FeedReader:
    """The `get_next` interface `quantize_static` reads its calibration feeds through."""

    def __init__(self, calibration: CalibrationSet):
        self.feeds = iter(calibration)

    def get_next(self):
        return next(self.feeds, None)


def _drop_initializer_shapes(model: onnx.ModelProto) -> None:
    """Remove the `value_info` of initializers: dynamic mode transposes `Gemm` weights in place, then re-infers shapes
    and trips over the stale ones."""
    initializers = {initializer.name for initializer in model.graph.initializer}
    value_info = [info for info in model.graph.value_info if info.name not in initializers]
    del model.graph.value_info[:]
    model.graph.value_info.extend(value_info)
