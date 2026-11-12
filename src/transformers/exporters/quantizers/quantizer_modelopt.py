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
"""NVIDIA ModelOpt quantization of the converted ONNX model, for TensorRT."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Iterable
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

from ...utils.import_utils import is_modelopt_available, is_onnxscript_available
from ..configs import ExportFormat
from ..utils import onnx_feed
from .base import CalibrationSet, ExportQuantizer, QuantizationConfigMixin
from .quantizer_onnxruntime import _FeedReader


if is_onnxscript_available():
    import onnx_ir

if is_modelopt_available():
    from modelopt.onnx.quantization import quantize as modelopt_quantize


@contextmanager
def _restored_onnxruntime_quantization():
    """ModelOpt patches ONNX Runtime's calibrators and QDQ quantizer in place for its own calibration
    (`modelopt.onnx.quantization.ort_patches`) and leaves them patched, which breaks a later `OnnxRuntimeQuantizer` in
    the same process. Put back every attribute it adds or replaces on them."""
    from onnxruntime.quantization import base_quantizer, calibrate, qdq_quantizer

    owners = (
        calibrate,
        qdq_quantizer,
        calibrate.CalibraterBase,
        calibrate.MinMaxCalibrater,
        calibrate.HistogramCalibrater,
        calibrate.HistogramCollector,
        qdq_quantizer.QDQQuantizer,
        base_quantizer.BaseQuantizer,
    )
    snapshots = [(owner, dict(vars(owner))) for owner in owners]
    try:
        yield
    finally:
        for owner, attributes in snapshots:
            for name in set(vars(owner)) - set(attributes):
                delattr(owner, name)
            for name, value in attributes.items():
                if vars(owner).get(name) is not value:
                    setattr(owner, name, value)


@dataclass
class ModelOptOnnxConfig(QuantizationConfigMixin):
    """
    Arguments of `modelopt.onnx.quantization.quantize`: int8 or fp8 QDQ calibrated on the quantizer's
    `calibration_dataset`, or int4 weights. The Q/DQ nodes go where TensorRT fuses them, and the rest of an fp32 model
    is converted to `high_precision_dtype` (fp16 by default). A field left to `None` keeps ModelOpt's default. See
    [ModelOpt's ONNX quantization](https://nvidia.github.io/Model-Optimizer/guides/_onnx_quantization.html) for each
    one.
    """

    quantize_mode: str | None = None
    calibration_method: str | None = None
    calibration_eps: list[str] | None = None
    op_types_to_quantize: list[str] | None = None
    op_types_to_exclude: list[str] | None = None
    nodes_to_quantize: list[str] | None = None
    nodes_to_exclude: list[str] | None = None
    high_precision_dtype: str | None = None
    mha_accumulation_dtype: str | None = None
    disable_mha_qdq: bool | None = None
    block_size: int | None = None


class ModelOptOnnxQuantizer(ExportQuantizer):
    """
    Quantize the converted ONNX model with NVIDIA's [ModelOpt](https://github.com/NVIDIA/Model-Optimizer)
    (`modelopt.onnx.quantization.quantize`) for TensorRT, such as ONNX Runtime's TensorRT execution provider. Export
    the model in fp32: ModelOpt calibrates it in fp32, then converts what it leaves unquantized to fp16.

    Example:

    ```python
    >>> from transformers.exporters import ModelOptOnnxConfig, ModelOptOnnxQuantizer, OnnxConfig

    >>> OnnxConfig(quantizer=ModelOptOnnxQuantizer(calibration_dataset=samples))
    >>> OnnxConfig(quantizer=ModelOptOnnxQuantizer(ModelOptOnnxConfig(disable_mha_qdq=True), calibration_dataset=samples))
    ```
    """

    supported_formats = (ExportFormat.ONNX,)

    required_packages = ("modelopt", "onnx", "onnx_ir", "onnxruntime")
    tested_versions = {"modelopt": "0.48.0", "onnx": "1.21.0", "onnx_ir": "1.1.0", "onnxruntime": "1.29.0"}
    min_versions = {"modelopt": "0.48.0"}

    def __init__(
        self, config: ModelOptOnnxConfig | None = None, calibration_dataset: Iterable[dict[str, Any]] | None = None
    ):
        super().__init__(calibration_dataset)
        self.config = config if config is not None else ModelOptOnnxConfig()

    def quantize(
        self, model: onnx_ir.Model, sample_inputs: dict[str, Any], export_format: ExportFormat
    ) -> onnx_ir.Model:
        names = [graph_input.name for graph_input in model.graph.inputs]
        calibration = CalibrationSet(self.calibration_dataset, sample_inputs, lambda sample: onnx_feed(names, sample))
        with tempfile.TemporaryDirectory() as tmp:
            # ModelOpt quantizes from and to files with external data, so models past protobuf's 2GB limit work
            path = os.path.join(tmp, "model.onnx")
            quantized_path = os.path.join(tmp, "quantized.onnx")
            onnx_ir.save(model, path, external_data="model.onnx.data")
            with _restored_onnxruntime_quantization():
                modelopt_quantize(
                    path,
                    calibration_data_reader=_FeedReader(calibration),
                    output_path=quantized_path,
                    use_external_data_format=True,
                    **self.config.to_kwargs(),
                )
            # Load the weights into memory before the temporary directory goes away
            quantized = onnx_ir.load(quantized_path)
            onnx_ir.external_data.load_to_model(quantized)
            return quantized
