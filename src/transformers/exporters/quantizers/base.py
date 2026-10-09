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
"""Base class for the post-training quantizers an export config can carry, and the calibration set they read."""

from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Iterator
from enum import Enum
from typing import Any

from ...utils import logging
from ...utils.import_utils import _is_package_available
from ..configs import ExportFormat


logger = logging.get_logger(__name__)


class QuantizationStage(Enum):
    """Where in the export a quantizer runs."""

    # On the `torch.export` FX graph, before any backend lowering.
    FX = "fx"
    # On the backend's own model (`onnx.ModelProto`, `openvino.Model`); for Dynamo exports, the FX graph itself.
    BACKEND = "backend"


class ExportQuantizer(ABC):
    """
    Base class for post-training quantizers applied during export.

    A quantizer declares the export formats it supports, the packages it needs and the stage it runs at; the exporter
    validates it against the export up front, then calls [`~ExportQuantizer.quantize`] with the model at that stage.
    Subclass it and implement `_quantize`, which receives a [`CalibrationSet`] of the model's own inputs, to add a new
    quantization method.

    Args:
        calibration_dataset (`Iterable[dict]`, *optional*):
            Input dicts the quantizer calibrates on: a list, or a `DataLoader` whose batches collate to model inputs.
            For [`~HfExporter.export_for_generation`] they are generate kwargs, captured into a calibration set per
            component. When `None`, a quantizer that needs data calibrates on the export's own sample inputs, with a
            warning.
    """

    stage: QuantizationStage
    supported_formats: tuple[ExportFormat, ...] = ()
    required_packages: tuple[str, ...] = ()

    def __init__(self, calibration_dataset: Iterable[dict[str, Any]] | None = None):
        self.calibration_dataset = calibration_dataset

    def with_calibration(self, calibration_dataset: Iterable[dict[str, Any]]) -> ExportQuantizer:
        """A copy of this quantizer that calibrates on `calibration_dataset`."""
        quantizer = copy.copy(self)
        quantizer.calibration_dataset = calibration_dataset
        return quantizer

    def validate_environment(self, export_format: ExportFormat) -> None:
        """Raise if this quantizer can't run on an `export_format` export, or a package it needs is missing."""
        if export_format not in self.supported_formats:
            supported = ", ".join(sorted(f.value for f in self.supported_formats))
            raise ValueError(f"{type(self).__name__} quantizes {supported} exports, not {export_format.value} ones.")
        missing = [package for package in self.required_packages if not _is_package_available(package)]
        if missing:
            raise ImportError(f"{type(self).__name__} requires: {', '.join(missing)}")

    def quantize(
        self,
        model: Any,
        sample_inputs: dict[str, Any],
        transform: Callable[[dict[str, Any]], Any],
        export_format: ExportFormat,
    ) -> Any:
        """Quantize `model` (the FX graph or backend model for this quantizer's stage), calibrated on this quantizer's
        `calibration_dataset`, or on `sample_inputs` when it has none. `transform` maps each forward-kwarg sample to the
        model's own inputs."""
        calibration = CalibrationSet(self.calibration_dataset, sample_inputs, transform)
        return self._quantize(model, calibration, export_format)

    @abstractmethod
    def _quantize(self, model: Any, calibration: CalibrationSet, export_format: ExportFormat) -> Any:
        """Quantize `model` on the `calibration` set of its own inputs and return the result."""


class CalibrationSet:
    """
    Re-iterable view of calibration samples as model inputs.

    The samples are the quantizer's `calibration_dataset`, or the export's own `sample_inputs` (with a warning, once
    they are read) when it has none. `transform` maps each forward-kwarg sample to what the model at the quantizer's
    stage takes (traced kwargs, ONNX feeds, OpenVINO port feeds), lazily, one sample at a time, on every pass.
    """

    def __init__(
        self,
        calibration_dataset: Iterable[dict[str, Any]] | None,
        sample_inputs: dict[str, Any],
        transform: Callable[[dict[str, Any]], Any],
    ):
        self.calibration_dataset = calibration_dataset
        self.sample_inputs = sample_inputs
        self.transform = transform

    def __iter__(self) -> Iterator[Any]:
        samples = self.calibration_dataset
        if not samples:
            logger.warning_once(
                "Quantizing with no `calibration_dataset`; calibrating on the single sample input. Statistics from "
                "one sample can hurt accuracy — pass a representative `calibration_dataset` to the quantizer."
            )
            samples = [self.sample_inputs]
        return (self.transform(sample) for sample in samples)
