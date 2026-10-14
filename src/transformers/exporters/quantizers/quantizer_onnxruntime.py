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
from dataclasses import dataclass
from typing import Any

from ...utils.import_utils import is_onnx_available, is_onnxruntime_available, is_onnxscript_available
from ..configs import ExportFormat
from .base import CalibrationSet, ExportQuantizer, QuantizationConfigMixin
from .utils import onnx_feed


if is_onnx_available():
    import onnx

if is_onnxscript_available():
    import onnx_ir

if is_onnxruntime_available():
    from onnxruntime.quantization import quantize_dynamic, quantize_static


@dataclass
class OnnxRuntimeStaticConfig(QuantizationConfigMixin):
    """
    Arguments of `onnxruntime.quantization.quantize_static`: int8 activations and weights (QDQ by default), calibrated
    on the quantizer's `calibration_dataset`. A field left to `None` keeps ONNX Runtime's default. See
    [`quantize_static`](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html) for each one.
    """

    quant_format: Any = None
    op_types_to_quantize: list[str] | None = None
    per_channel: bool | None = None
    reduce_range: bool | None = None
    activation_type: Any = None
    weight_type: Any = None
    nodes_to_quantize: list[str] | None = None
    nodes_to_exclude: list[str] | None = None
    calibrate_method: Any = None
    calibration_providers: list[str] | None = None
    extra_options: dict[str, Any] | None = None


@dataclass
class OnnxRuntimeDynamicConfig(QuantizationConfigMixin):
    """
    Arguments of `onnxruntime.quantization.quantize_dynamic`: int8 weights, activations quantized at runtime, no
    calibration. A field left to `None` keeps ONNX Runtime's default. See
    [`quantize_dynamic`](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html) for each one.
    """

    op_types_to_quantize: list[str] | None = None
    per_channel: bool | None = None
    reduce_range: bool | None = None
    weight_type: Any = None
    nodes_to_quantize: list[str] | None = None
    nodes_to_exclude: list[str] | None = None
    extra_options: dict[str, Any] | None = None


class OnnxRuntimeQuantizer(ExportQuantizer):
    """
    Quantize the converted ONNX model with ONNX Runtime's own tools (`onnxruntime.quantization`): `quantize_static`
    with an [`OnnxRuntimeStaticConfig`] (the default), or `quantize_dynamic` with an [`OnnxRuntimeDynamicConfig`].

    Example:

    ```python
    >>> from transformers.exporters import (
    ...     OnnxConfig,
    ...     OnnxRuntimeDynamicConfig,
    ...     OnnxRuntimeQuantizer,
    ...     OnnxRuntimeStaticConfig,
    ... )

    >>> static = OnnxRuntimeStaticConfig(per_channel=True)
    >>> OnnxConfig(quantizer=OnnxRuntimeQuantizer(static, calibration_dataset=samples))
    >>> OnnxConfig(quantizer=OnnxRuntimeQuantizer(OnnxRuntimeDynamicConfig()))
    ```
    """

    supported_formats = (ExportFormat.ONNX,)

    required_packages = ("onnx", "onnx_ir", "onnxruntime")
    tested_versions = {"onnx": "1.23.0", "onnx_ir": "1.0.0", "onnxruntime": "1.23.2"}
    min_versions = {"onnxruntime": "1.21.0"}

    def __init__(
        self,
        config: OnnxRuntimeStaticConfig | OnnxRuntimeDynamicConfig | None = None,
        calibration_dataset: Iterable[dict[str, Any]] | None = None,
    ):
        super().__init__(calibration_dataset)
        self.config = config if config is not None else OnnxRuntimeStaticConfig()

    def quantize(
        self, model: onnx_ir.Model, sample_inputs: dict[str, Any], export_format: ExportFormat
    ) -> onnx_ir.Model:
        with tempfile.TemporaryDirectory() as tmp:
            # ONNX Runtime quantizes from and to files with external data, so models past protobuf's 2GB limit work
            path = os.path.join(tmp, "model.onnx")
            quantized_path = os.path.join(tmp, "quantized.onnx")
            onnx_ir.save(model, path, external_data="model.onnx.data")
            if isinstance(self.config, OnnxRuntimeDynamicConfig):
                # Dynamic mode transposes `Gemm` weights in place, then re-infers shapes and trips over the
                # initializers' stale `value_info`: drop it
                proto = onnx.load(path, load_external_data=False)
                initializers = {initializer.name for initializer in proto.graph.initializer}
                value_info = [info for info in proto.graph.value_info if info.name not in initializers]
                del proto.graph.value_info[:]
                proto.graph.value_info.extend(value_info)
                onnx.save(proto, path)
                quantize_dynamic(path, quantized_path, use_external_data_format=True, **self.config.to_kwargs())
            else:
                names = [graph_input.name for graph_input in model.graph.inputs]
                calibration = CalibrationSet(
                    self.calibration_dataset, sample_inputs, lambda sample: onnx_feed(names, sample)
                )
                quantize_static(
                    path,
                    quantized_path,
                    _FeedReader(calibration),
                    use_external_data_format=True,
                    **self.config.to_kwargs(),
                )
            # Load the weights into memory before the temporary directory goes away
            quantized = onnx_ir.load(quantized_path)
            onnx_ir.external_data.load_to_model(quantized)
            return quantized


class _FeedReader:
    """The `get_next` interface `quantize_static` reads its calibration feeds through."""

    def __init__(self, calibration: CalibrationSet):
        self.feeds = iter(calibration)

    def get_next(self):
        return next(self.feeds, None)
