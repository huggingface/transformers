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
import dataclasses
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Iterator
from typing import Any

from ...utils import logging
from ..configs import ExportFormat
from ..utils import check_packages


logger = logging.get_logger(__name__)


class ExportQuantizer(ABC):
    """
    Base class for post-training quantizers applied during export.

    A quantizer declares the export formats it supports and the packages it needs (with the versions it is tested on,
    like the exporters) and whether it quantizes the `torch.export` FX graph, before any conversion (so every backend
    takes the result), or the model a backend converts it into (`onnx_ir.Model`, `openvino.Model`). The exporter
    validates it against the export up front, then calls [`~ExportQuantizer.quantize`] with that model and the export's
    sample inputs. Subclass it and implement `quantize` to add a new quantization method; a [`CalibrationSet`] maps the
    calibration samples to the model's own inputs.

    Args:
        calibration_dataset (`Iterable[dict]`, *optional*):
            Input dicts the quantizer calibrates on: a list, or a `DataLoader` whose batches collate to model inputs.
            For [`~HfExporter.export_for_generation`] they are generate kwargs, captured into a calibration set per
            component. When `None`, a quantizer that needs data calibrates on the export's own sample inputs, with a
            warning.
    """

    quantizes_fx_graph: bool = False
    supported_formats: tuple[ExportFormat, ...] = ()

    required_packages: tuple[str, ...] = ()
    tested_versions: dict[str, str] = {}
    min_versions: dict[str, str] = {}

    def __init__(self, calibration_dataset: Iterable[dict[str, Any]] | None = None):
        self.calibration_dataset = calibration_dataset

    def with_calibration(self, calibration_dataset: Iterable[dict[str, Any]]) -> ExportQuantizer:
        """A copy of this quantizer that calibrates on `calibration_dataset`."""
        quantizer = copy.copy(self)
        quantizer.calibration_dataset = calibration_dataset
        return quantizer

    def validate_environment(self, export_format: ExportFormat) -> None:
        """Raise if this quantizer can't run on an `export_format` export, or a package it needs is missing or too old;
        warn when one differs from the version it is tested on."""
        if export_format not in self.supported_formats:
            supported = ", ".join(sorted(f.value for f in self.supported_formats))
            raise ValueError(
                f"{self.__class__.__name__} quantizes {supported} exports, not {export_format.value} ones."
            )
        check_packages(self.__class__.__name__, self.required_packages, self.min_versions, self.tested_versions)

    @abstractmethod
    def quantize(self, model: Any, sample_inputs: dict[str, Any], export_format: ExportFormat) -> Any:
        """Quantize `model` (the FX graph if `quantizes_fx_graph`, else the backend model) and return the result,
        calibrated on `calibration_dataset`, or on the export's `sample_inputs` when it has none."""


class CalibrationSet:
    """
    Re-iterable view of calibration samples as model inputs.

    The samples are the quantizer's `calibration_dataset`, or the export's own `sample_inputs` (with a warning, once
    they are read) when it has none. They can be any mapping, such as a tokenizer's `BatchEncoding`. `transform` maps
    each forward-kwarg sample to what the quantized model takes (traced kwargs, ONNX feeds, OpenVINO
    port feeds), lazily, one sample at a time, on every pass.
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


class QuantizationConfigMixin:
    """Arguments of a quantization library's function, as a dataclass whose `None` fields keep the library's
    defaults."""

    def to_kwargs(self) -> dict[str, Any]:
        """The fields that are set, as keyword arguments for the library's function."""
        return {
            field.name: value for field in dataclasses.fields(self) if (value := getattr(self, field.name)) is not None
        }
