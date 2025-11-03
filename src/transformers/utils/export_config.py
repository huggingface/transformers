#!/usr/bin/env python
# coding=utf-8

# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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
import copy
from dataclasses import dataclass
from enum import Enum
from os import PathLike
from typing import Any

from ..utils import logging


logger = logging.get_logger(__name__)


class ExportFormat(Enum):
    # EXECUTORCH = "executorch"
    DYNAMO = "dynamo"
    ONNX = "onnx"
    # TORCHSCRIPT = "torchscript"


@dataclass
class ExportConfigMixin:
    """
    Mixin class for all export configuration classes. Provides methods for serialization and
    deserialization from/to dictionary and JSON file.
    """

    export_format: ExportFormat

    @classmethod
    def from_dict(cls, config_dict, return_unused_kwargs=False, **kwargs):
        """
        Instantiates a [`ExportConfigMixin`] from a Python dictionary of parameters.

        Args:
            config_dict (`dict[str, Any]`):
                Dictionary that will be used to instantiate the configuration object.
            return_unused_kwargs (`bool`,*optional*, defaults to `False`):
                Whether or not to return a list of unused keyword arguments. Used for `from_pretrained` method in
                `PreTrainedModel`.
            kwargs (`dict[str, Any]`):
                Additional parameters from which to initialize the configuration object.

        Returns:
            [`ExportConfigMixin`]: The configuration object instantiated from those parameters.
        """
        config = cls(**config_dict)

        to_remove = []
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
                to_remove.append(key)
        for key in to_remove:
            kwargs.pop(key, None)

        if return_unused_kwargs:
            return config, kwargs
        else:
            return config

    def to_dict(self) -> dict[str, Any]:
        """
        Serializes this instance to a Python dictionary. Returns:
            `dict[str, Any]`: Dictionary of all the attributes that make up this configuration instance.
        """
        return copy.deepcopy(self.__dict__)

    def __iter__(self):
        """allows `dict(obj)` for situations where obj may be a dict or an ExportConfigMixin instance."""
        for attr, value in copy.deepcopy(self.__dict__).items():
            yield attr, value


@dataclass
class DynamoConfig(ExportConfigMixin):
    """
    Configuration class for exporting models using TorchDynamo.

    Args:
        strict (`bool`, *optional*, defaults to `False`):
            Whether to enable strict mode during export.
        sample_inputs (`dict[str, Any]`, *optional*):
            Sample inputs to be used during export.
        dynamic_shapes (`dict[str, Any]`, *optional*):
            Dynamic shapes for the model inputs.
    """

    export_format: ExportFormat = ExportFormat.DYNAMO
    sample_inputs: dict[str, Any] | None = None

    strict: bool = False
    dynamic_shapes: dict[str, Any] | None = None


@dataclass
class OnnxConfig(ExportConfigMixin):
    """
    Configuration class for exporting models to ONNX format.

    Args:
        opset_version (`int`, *optional*, defaults to 23):
            The ONNX opset version to use for export.
        sample_inputs (`dict[str, Any]`, *optional*):
            Sample inputs to be used during export.
    """

    export_format: ExportFormat = ExportFormat.ONNX
    sample_inputs: dict[str, Any] | None = None

    f: str | PathLike | None = None
    dynamic_shapes: dict[str, Any] | None = None
    opset_version: int | None = None
    external_data: bool = True
    optimize: bool = True
    export_params: bool = True
    keep_initializers_as_inputs: bool = False
    do_constant_folding: bool = True
