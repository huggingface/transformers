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

import copy
import dataclasses
import os
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np

from ...utils.import_utils import (
    is_nncf_available,
    is_onnx_available,
    is_onnxscript_available,
    is_openvino_available,
    is_torch_available,
)
from ..configs import ExportFormat
from .base import CalibrationSet, ExportQuantizer, QuantizationConfigMixin
from .utils import onnx_feed, openvino_feed, traced_inputs


if is_torch_available():
    import torch

if is_onnx_available():
    import onnx

if is_onnxscript_available():
    import onnx_ir

if is_openvino_available():
    import openvino

if is_nncf_available():
    import nncf
    from nncf.onnx.quantization.backend_parameters import BackendParameters


@dataclass
class NNCFQuantizationConfig(QuantizationConfigMixin):
    """
    Arguments of `nncf.quantize`: int8 activations and weights, calibrated on the quantizer's `calibration_dataset`.
    A field left to `None` keeps NNCF's default, except `model_type`, which defaults to `nncf.ModelType.TRANSFORMER`.
    See [`nncf.quantize`](https://openvinotoolkit.github.io/nncf/autoapi/nncf/index.html#nncf.quantize) for each one.
    """

    mode: Any = None
    preset: Any = None
    target_device: Any = None
    subset_size: int | None = None
    fast_bias_correction: bool | None = None
    model_type: Any = None
    ignored_scope: Any = None
    advanced_parameters: Any = None


@dataclass
class NNCFWeightCompressionConfig(QuantizationConfigMixin):
    """
    Arguments of `nncf.compress_weights`: weight-only int8 or int4, with data-aware methods (AWQ, scale estimation,
    GPTQ) for the int4 modes. The int8 modes are data-free; the others calibrate on the quantizer's
    `calibration_dataset`. `gptq` and `lora_correction` are OpenVINO-only. A field left to `None` keeps NNCF's
    default. See
    [`nncf.compress_weights`](https://openvinotoolkit.github.io/nncf/autoapi/nncf/index.html#nncf.compress_weights) for
    each one.
    """

    mode: Any = None
    ratio: float | None = None
    group_size: int | None = None
    ignored_scope: Any = None
    all_layers: bool | None = None
    sensitivity_metric: Any = None
    subset_size: int | None = None
    awq: bool | None = None
    scale_estimation: bool | None = None
    gptq: bool | None = None
    lora_correction: bool | None = None
    backup_mode: Any = None
    compression_format: Any = None
    advanced_parameters: Any = None


class _NNCFQuantizer(ExportQuantizer):
    """Runs `nncf.quantize` or `nncf.compress_weights`, depending on its config; NNCF picks its backend from the model
    type, so the subclasses only declare which model they quantize and map the calibration samples to its inputs."""

    tested_versions = {"nncf": "3.4.0"}
    min_versions = {"nncf": "2.18.0"}

    def __init__(
        self,
        config: NNCFQuantizationConfig | NNCFWeightCompressionConfig | None = None,
        calibration_dataset: Iterable[dict[str, Any]] | None = None,
    ):
        super().__init__(calibration_dataset)
        self.config = config if config is not None else NNCFQuantizationConfig()

    def _quantize_with_nncf(
        self,
        model: openvino.Model | onnx.ModelProto | torch.fx.GraphModule,
        calibration: CalibrationSet,
        backend_params: dict[str, Any] | None = None,
    ) -> openvino.Model | onnx.ModelProto | torch.fx.GraphModule:
        config = self.config
        quantization = isinstance(config, NNCFQuantizationConfig)
        if backend_params:
            advanced_parameters = copy.deepcopy(config.advanced_parameters) or (
                nncf.AdvancedQuantizationParameters() if quantization else nncf.AdvancedCompressionParameters()
            )
            advanced_parameters.backend_params.update(backend_params)
            config = dataclasses.replace(config, advanced_parameters=advanced_parameters)
        if quantization:
            nncf_kwargs = config.to_kwargs()
            nncf_kwargs.setdefault("model_type", nncf.ModelType.TRANSFORMER)
            return nncf.quantize(model, nncf.Dataset(calibration), **nncf_kwargs)
        # The int8 modes (NNCF's default is INT8_ASYM) are data-free and refuse a dataset, so they never read the
        # calibration set; the others use it for their data-aware methods.
        data_free = config.mode in (None, nncf.CompressWeightsMode.INT8_SYM, nncf.CompressWeightsMode.INT8_ASYM)
        return nncf.compress_weights(
            model, dataset=None if data_free else nncf.Dataset(calibration), **config.to_kwargs()
        )


class NNCFOpenVINOQuantizer(_NNCFQuantizer):
    """
    Quantize the converted OpenVINO model with [NNCF](https://github.com/openvinotoolkit/nncf): `nncf.quantize` with a
    [`NNCFQuantizationConfig`] (the default), or `nncf.compress_weights` with a [`NNCFWeightCompressionConfig`].

    Example:

    ```python
    >>> import nncf
    >>> from transformers.exporters import NNCFOpenVINOQuantizer, NNCFWeightCompressionConfig, OpenVINOConfig

    >>> OpenVINOConfig(quantizer=NNCFOpenVINOQuantizer(calibration_dataset=samples))
    >>> OpenVINOConfig(
    ...     quantizer=NNCFOpenVINOQuantizer(NNCFWeightCompressionConfig(mode=nncf.CompressWeightsMode.INT4_SYM))
    ... )
    ```
    """

    supported_formats = (ExportFormat.OPENVINO,)

    required_packages = ("nncf", "openvino")
    tested_versions = {**_NNCFQuantizer.tested_versions, "openvino": "2026.4.0"}

    def quantize(
        self, model: openvino.Model, sample_inputs: dict[str, Any], export_format: ExportFormat
    ) -> openvino.Model:
        calibration = CalibrationSet(
            self.calibration_dataset, sample_inputs, lambda sample: openvino_feed(model, sample)
        )
        return self._quantize_with_nncf(model, calibration)


class NNCFOnnxQuantizer(_NNCFQuantizer):
    """
    Quantize the converted ONNX model with [NNCF](https://github.com/openvinotoolkit/nncf) through its ONNX backend:
    `nncf.quantize` with a [`NNCFQuantizationConfig`] (the default, int8 QDQ), or `nncf.compress_weights` with a
    [`NNCFWeightCompressionConfig`] (int4 as ONNX Runtime's `MatMulNBits`).

    Example:

    ```python
    >>> import nncf
    >>> from transformers.exporters import NNCFOnnxQuantizer, NNCFWeightCompressionConfig, OnnxConfig

    >>> OnnxConfig(quantizer=NNCFOnnxQuantizer(calibration_dataset=samples))
    >>> OnnxConfig(quantizer=NNCFOnnxQuantizer(NNCFWeightCompressionConfig(mode=nncf.CompressWeightsMode.INT4_SYM)))
    ```
    """

    supported_formats = (ExportFormat.ONNX,)

    # NNCF's ONNX backend calibrates through ONNX Runtime
    required_packages = ("nncf", "onnx", "onnx_ir", "onnxruntime")
    tested_versions = {**_NNCFQuantizer.tested_versions, "onnx": "1.23.0", "onnx_ir": "1.0.0", "onnxruntime": "1.23.2"}

    def quantize(
        self, model: onnx_ir.Model, sample_inputs: dict[str, Any], export_format: ExportFormat
    ) -> onnx_ir.Model:
        # NNCF only quantizes a `MatMul` whose weight is an initializer, but the ONNX optimizer doesn't fold the
        # `Transpose` of a large one, which a tied `lm_head` gets: fold them here
        for node in list(model.graph):
            if node.op_type != "Transpose" or not node.inputs[0].is_initializer():
                continue
            perm = node.attributes["perm"].as_ints() if "perm" in node.attributes else None
            output = node.outputs[0]
            transposed = np.transpose(node.inputs[0].const_value.numpy(), perm)
            folded = onnx_ir.Value(
                name=output.name,
                shape=output.shape,
                type=output.type,
                const_value=onnx_ir.tensor(transposed, name=output.name),
            )
            onnx_ir.convenience.replace_all_uses_with(output, folded)
            model.graph.remove(node, safe=True)
            model.graph.register_initializer(folded)

        names = [graph_input.name for graph_input in model.graph.inputs]
        calibration = CalibrationSet(self.calibration_dataset, sample_inputs, lambda sample: onnx_feed(names, sample))
        with tempfile.TemporaryDirectory() as tmp:
            # NNCF reads the weights from `tmp` and returns the model with them loaded, so models past protobuf's 2GB
            # limit work
            path = os.path.join(tmp, "model.onnx")
            onnx_ir.save(model, path, external_data="model.onnx.data")
            proto = onnx.load(path, load_external_data=False)
            quantized = self._quantize_with_nncf(proto, calibration, {BackendParameters.EXTERNAL_DATA_DIR: tmp})
        return onnx_ir.from_proto(quantized)


class NNCFTorchFXQuantizer(_NNCFQuantizer):
    """
    Quantize the `torch.export` FX graph with [NNCF](https://github.com/openvinotoolkit/nncf) through its TorchFX
    backend: `nncf.quantize` with a [`NNCFQuantizationConfig`] (the default, `quantize`/`dequantize` ops), or
    `nncf.compress_weights` with a [`NNCFWeightCompressionConfig`]. Like [`PT2EQuantizer`], it runs before any
    conversion, so every backend takes its result. ExecuTorch lowers only the weight-only result: XNNPACK doesn't
    delegate the `quantize`/`dequantize` ops, and ExecuTorch has no kernels for them.

    Example:

    ```python
    >>> from transformers.exporters import DynamoConfig, NNCFTorchFXQuantizer

    >>> DynamoConfig(quantizer=NNCFTorchFXQuantizer(calibration_dataset=samples))
    ```
    """

    quantizes_fx_graph = True
    supported_formats = (ExportFormat.DYNAMO, ExportFormat.ONNX, ExportFormat.OPENVINO, ExportFormat.EXECUTORCH)

    required_packages = ("nncf", "torch")
    tested_versions = {**_NNCFQuantizer.tested_versions, "torch": "2.13.0"}

    def quantize(
        self, model: torch.fx.GraphModule, sample_inputs: dict[str, Any], export_format: ExportFormat
    ) -> torch.fx.GraphModule:
        traced_names = list(sample_inputs)
        calibration = CalibrationSet(
            self.calibration_dataset, sample_inputs, lambda sample: traced_inputs(sample, traced_names)
        )
        with torch.no_grad():
            return self._quantize_with_nncf(model, calibration)
