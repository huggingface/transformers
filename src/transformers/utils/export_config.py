# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
    """Identifies the export backend. Stored in [`ExportConfigMixin`] for serialisation round-trips."""

    EXECUTORCH = "executorch"
    DYNAMO = "dynamo"
    ONNX = "onnx"


@dataclass
class ExportConfigMixin:
    """
    Base class for all export configuration dataclasses.

    Provides `to_dict` / `from_dict` serialisation so configs can be saved and round-tripped
    without knowing the concrete subclass. The `export_format` field identifies the subclass
    during deserialisation.
    """

    export_format: ExportFormat

    @classmethod
    def from_dict(cls, config_dict):
        """
        Instantiates a [`ExportConfigMixin`] from a Python dictionary of parameters.

        Args:
            config_dict (`dict[str, Any]`):
                Dictionary that will be used to instantiate the configuration object.

        Returns:
            [`ExportConfigMixin`]: The configuration object instantiated from those parameters.
        """
        config = cls(**config_dict)
        return config

    def to_dict(self) -> dict[str, Any]:
        """
        Serializes this instance to a Python dictionary.

        Returns:
            `dict[str, Any]`: Dictionary of all the attributes that make up this configuration instance.
        """
        return copy.deepcopy(self.__dict__)

    def __iter__(self):
        yield from self.__dict__.items()


@dataclass
class DynamoConfig(ExportConfigMixin):
    """
    Configuration class for exporting models via `torch.export`.

    Args:
        dynamic (`bool`, *optional*, defaults to `False`):
            Whether to export with dynamic (symbolic) shapes. When `True` and
            `dynamic_shapes` is not set, all tensor dimensions are set to
            `Dim.AUTO` automatically.
        strict (`bool`, *optional*, defaults to `False`):
            Whether to enable strict mode in `torch.export`. Runs the full
            symbolic trace and catches more errors, but is slower and more
            likely to fail on complex models.
        dynamic_shapes (`dict[str, Any]`, *optional*):
            Explicit per-input dynamic shape specifications passed to
            `torch.export`. Takes precedence over `dynamic`.
        prefer_deferred_runtime_asserts_over_guards (`bool`, *optional*, defaults to `False`):
            When `True`, shape guards are emitted as runtime assertions in the
            exported graph instead of being specialised at trace time. Useful
            for reducing retracing when shapes vary at runtime.
    """

    export_format: ExportFormat = ExportFormat.DYNAMO
    dynamic: bool = False

    strict: bool = False
    dynamic_shapes: dict[str, Any] | None = None
    prefer_deferred_runtime_asserts_over_guards: bool = False


@dataclass
class OnnxConfig(DynamoConfig):
    """
    Configuration class for exporting models to ONNX via `torch.onnx.export`.

    Inherits all fields from [`DynamoConfig`] (`dynamic`, `strict`,
    `dynamic_shapes`, `prefer_deferred_runtime_asserts_over_guards`).

    Args:
        f (`str` or `PathLike`, *optional*):
            Output path for the `.onnx` file. When `None` (default) the
            exported model is kept in memory as an `ONNXProgram` and not
            written to disk.
        opset_version (`int`, *optional*):
            ONNX opset version to target. Defaults to the latest opset
            supported by the installed `onnxscript` version.
        external_data (`bool`, *optional*, defaults to `True`):
            Store large weight tensors in a separate `.onnx_data` sidecar
            file instead of embedding them in the protobuf. Required for
            models whose weights exceed the 2 GB protobuf limit.
        optimize (`bool`, *optional*, defaults to `True`):
            Run `onnxscript` optimisation passes (constant folding, dead-code
            elimination, …) on the exported graph. Disable for models that
            hit upstream `onnxscript` optimiser bugs.
        export_params (`bool`, *optional*, defaults to `True`):
            Embed model weights in the ONNX graph. Set to `False` to export
            a weight-free graph (weights must be supplied at runtime).
        keep_initializers_as_inputs (`bool`, *optional*, defaults to `False`):
            Expose weight initializers as explicit graph inputs. Required by
            some older ONNX runtimes (opset < 9).
    """

    export_format: ExportFormat = ExportFormat.ONNX

    f: str | PathLike | None = None
    dynamic_shapes: dict[str, Any] | None = None
    opset_version: int | None = None
    external_data: bool = True
    optimize: bool = True
    export_params: bool = True
    keep_initializers_as_inputs: bool = False


@dataclass
class ExecutorchConfig(DynamoConfig):
    """
    Configuration class for exporting models to ExecuTorch format.

    Inherits all fields from [`DynamoConfig`] (`dynamic`, `strict`,
    `dynamic_shapes`, `prefer_deferred_runtime_asserts_over_guards`).

    Args:
        backend (`str`, *optional*):
            Target ExecuTorch backend. Supported values:

            - `"xnnpack"` — CPU inference via the XNNPACK library.
            - `"cuda"` — GPU inference via the ExecuTorch CUDA backend.
    """

    export_format: ExportFormat = ExportFormat.EXECUTORCH

    backend: str | None = None
