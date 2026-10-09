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
"""NNCF quantization and weight compression of the FX graph or of the converted OpenVINO or ONNX model."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

from ..configs import ExportFormat
from .base import CalibrationSet, ExportQuantizer, QuantizationStage


if TYPE_CHECKING:
    import onnx
    import openvino
    import torch


class _NNCFQuantizer(ExportQuantizer):
    """Runs `nncf.quantize` or `nncf.compress_weights`; NNCF picks its backend from the model type, so the subclasses
    only declare which model they quantize."""

    stage = QuantizationStage.BACKEND
    required_packages = ("nncf",)

    def __init__(
        self, weights_only: bool = False, calibration_dataset: Iterable[dict[str, Any]] | None = None, **kwargs
    ):
        super().__init__(calibration_dataset)
        self.weights_only = weights_only
        self.kwargs = kwargs

    def _quantize(
        self,
        model: openvino.Model | onnx.ModelProto | torch.fx.GraphModule,
        calibration: CalibrationSet,
        export_format: ExportFormat,
    ) -> openvino.Model | onnx.ModelProto | torch.fx.GraphModule:
        import nncf

        dataset = nncf.Dataset(calibration)
        if not self.weights_only:
            return nncf.quantize(model, dataset, **{"model_type": nncf.ModelType.TRANSFORMER, **self.kwargs})
        # The int8 modes are data-free and refuse a dataset; the others use it for their data-aware methods.
        mode = self.kwargs.get("mode", nncf.CompressWeightsMode.INT8_ASYM)
        data_free = mode in (nncf.CompressWeightsMode.INT8_SYM, nncf.CompressWeightsMode.INT8_ASYM)
        return nncf.compress_weights(model, dataset=None if data_free else dataset, **self.kwargs)


class NNCFOpenVINOQuantizer(_NNCFQuantizer):
    """
    Quantize the converted OpenVINO model with [NNCF](https://github.com/openvinotoolkit/nncf).

    By default it runs `nncf.quantize` (int8 activations and weights, calibrated on `calibration_dataset`, with
    `model_type=nncf.ModelType.TRANSFORMER` unless given); with `weights_only=True` it runs `nncf.compress_weights`
    (weight-only int8 or int4, with the data-aware methods such as AWQ and scale estimation). Other keyword arguments
    go to that NNCF function.

    Example:

    ```python
    >>> import nncf
    >>> from transformers.exporters import NNCFOpenVINOQuantizer, OpenVINOConfig

    >>> OpenVINOConfig(quantizer=NNCFOpenVINOQuantizer(calibration_dataset=samples))
    >>> OpenVINOConfig(quantizer=NNCFOpenVINOQuantizer(weights_only=True, mode=nncf.CompressWeightsMode.INT4_SYM))
    ```
    """

    supported_formats = (ExportFormat.OPENVINO,)


class NNCFOnnxQuantizer(_NNCFQuantizer):
    """
    Quantize the converted ONNX model with [NNCF](https://github.com/openvinotoolkit/nncf) through its ONNX backend.

    By default it runs `nncf.quantize` (int8 QDQ activations and weights, calibrated on `calibration_dataset`, with
    `model_type=nncf.ModelType.TRANSFORMER` unless given); with `weights_only=True` it runs `nncf.compress_weights`
    (weight-only int8, or int4 as ONNX Runtime's `MatMulNBits`). Other keyword arguments go to that NNCF function.

    Example:

    ```python
    >>> import nncf
    >>> from transformers.exporters import NNCFOnnxQuantizer, OnnxConfig

    >>> OnnxConfig(quantizer=NNCFOnnxQuantizer(calibration_dataset=samples))
    >>> OnnxConfig(quantizer=NNCFOnnxQuantizer(weights_only=True, mode=nncf.CompressWeightsMode.INT4_SYM))
    ```
    """

    supported_formats = (ExportFormat.ONNX,)


class NNCFTorchFXQuantizer(_NNCFQuantizer):
    """
    Quantize the `torch.export` FX graph with [NNCF](https://github.com/openvinotoolkit/nncf) through its TorchFX
    backend. Like [`PT2EQuantizer`], it runs before any conversion, so every backend takes its result.

    By default it runs `nncf.quantize` (int8 activations and weights as `quantize`/`dequantize` ops, calibrated on
    `calibration_dataset`, with `model_type=nncf.ModelType.TRANSFORMER` unless given); with `weights_only=True` it
    runs `nncf.compress_weights` (weight-only int8 or int4). Other keyword arguments go to that NNCF function.
    ExecuTorch lowers only the weight-only result: XNNPACK doesn't delegate the `quantize`/`dequantize` ops, and
    ExecuTorch has no kernels for them.

    Example:

    ```python
    >>> from transformers.exporters import DynamoConfig, NNCFTorchFXQuantizer

    >>> DynamoConfig(quantizer=NNCFTorchFXQuantizer(calibration_dataset=samples))
    ```
    """

    stage = QuantizationStage.FX
    supported_formats = (ExportFormat.DYNAMO, ExportFormat.ONNX, ExportFormat.OPENVINO, ExportFormat.EXECUTORCH)
