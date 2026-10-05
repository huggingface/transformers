# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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


class ExportFormat(Enum):
    """Identifies the export backend. Stored in [`ExportConfigMixin`] for serialisation round-trips."""

    EXECUTORCH = "executorch"
    OPENVINO = "openvino"
    DYNAMO = "dynamo"
    ONNX = "onnx"


@dataclass
class ExportConfigMixin:
    """Base class for export configs; `export_format` identifies the subclass when deserialising."""

    export_format: ExportFormat

    @classmethod
    def from_dict(cls, config_dict):
        """Instantiates a [`ExportConfigMixin`] from a dictionary of parameters."""
        config_dict = dict(config_dict)
        if isinstance(config_dict.get("export_format"), str):
            config_dict["export_format"] = ExportFormat(config_dict["export_format"])
        return cls(**config_dict)

    def to_dict(self) -> dict[str, Any]:
        """Serializes this instance to a JSON-compatible dictionary."""
        fields = copy.deepcopy(self.__dict__)
        fields["export_format"] = self.export_format.value
        return fields


@dataclass
class DynamoConfig(ExportConfigMixin):
    """
    Configuration class for exporting models via `torch.export`.

    Args:
        dynamic (`bool`, *optional*, defaults to `False`):
            Export with dynamic shapes; without `dynamic_shapes`, every dimension is `Dim.AUTO`.
        strict (`bool`, *optional*, defaults to `False`):
            Enable `torch.export` strict mode.
        dynamic_shapes (`dict[str, Any]`, *optional*):
            Explicit per-input dynamic shapes passed to `torch.export`. Takes precedence over `dynamic`.
        prefer_deferred_runtime_asserts_over_guards (`bool`, *optional*, defaults to `False`):
            Emit data-dependent shape guards as runtime asserts instead of failing the trace. Usually needed
            with explicit `Dim(min=, max=)` bounds, not with `Dim.AUTO`.
    """

    export_format: ExportFormat = ExportFormat.DYNAMO

    dynamic: bool = False
    strict: bool = False
    dynamic_shapes: dict[str, Any] | None = None
    prefer_deferred_runtime_asserts_over_guards: bool = False


@dataclass
class OnnxConfig(DynamoConfig):
    """
    Configuration class for exporting models to ONNX via `torch.onnx.export`. Inherits the [`DynamoConfig`] fields.

    Args:
        output_path (`str` or `PathLike`, *optional*):
            Output `.onnx` path. When `None`, the `ONNXProgram` is kept in memory.
        opset_version (`int`, *optional*):
            ONNX opset to target. Defaults to the latest one the installed `onnxscript` supports.
        external_data (`bool`, *optional*, defaults to `True`):
            Store weights in a `.onnx_data` sidecar; required past the 2 GB protobuf limit.
        optimize (`bool`, *optional*, defaults to `True`):
            Run `onnxscript` optimisation passes on the exported graph.
        export_params (`bool`, *optional*, defaults to `True`):
            Embed weights in the graph; `False` exports a weight-free graph.
    """

    export_format: ExportFormat = ExportFormat.ONNX

    output_path: str | PathLike | None = None
    opset_version: int | None = None
    external_data: bool = True
    optimize: bool = True
    export_params: bool = True


@dataclass
class ExecutorchConfig(DynamoConfig):
    """
    Configuration class for exporting models to ExecuTorch format. Inherits the [`DynamoConfig`] fields.

    Args:
        backend (`str`, *optional*, defaults to `"xnnpack"`):
            Target ExecuTorch backend: `"xnnpack"` (CPU), `"openvino"` (CPU), `"cuda"` (GPU) or `"mlx"`
            (Apple Silicon GPU).
        alloc_graph_input (`bool`, *optional*, defaults to `True`):
            Reserve arena memory for graph inputs. When `False` the caller's buffers are used directly, so an
            in-place input mutation (a `StaticCache` write) lands in the caller's tensor.
        alloc_graph_output (`bool`, *optional*, defaults to `True`):
            Reserve arena memory for graph outputs. When `False` the caller binds output buffers at runtime,
            which [`ExecutorchModelRunner`] uses to skip the cache copy-out.
        alloc_mutable_buffers (`bool`, *optional*, defaults to `True`):
            Reserve arena memory for mutable buffers; passed to the `MemoryPlanningPass`.
        partition (`bool`, *optional*, defaults to `True`):
            Delegate eligible subgraphs to the backend. When `False` everything lowers to the portable kernels,
            which gets past a backend refusing a partition it claimed (XNNPACK, at method load).
        partition_exclude (`tuple[str, ...]`, *optional*):
            XNNPACK partitioner configs to withhold (e.g. `("ViewCopyConfig",)`), from
            `executorch.backends.xnnpack.partition.config.ALL_PARTITIONER_CONFIGS`; those ops lower to the
            portable kernels.
    """

    export_format: ExportFormat = ExportFormat.EXECUTORCH

    backend: str = "xnnpack"
    partition: bool = True
    partition_exclude: tuple[str, ...] = ()
    alloc_graph_input: bool = True
    alloc_graph_output: bool = True
    alloc_mutable_buffers: bool = True


@dataclass
class OpenVINOConfig(DynamoConfig):
    """
    Configuration class for exporting models to OpenVINO IR via `openvino.convert_model`. Inherits the
    [`DynamoConfig`] fields.

    Args:
        output_path (`str` or `PathLike`, *optional*):
            Output `.xml` path (`.bin` alongside). When `None`, the `openvino.Model` is kept in memory.
        compress_to_fp16 (`bool`, *optional*, defaults to `False`):
            Compress `float32` weights to `float16` when saving (off by default for its narrower range).
        stateful (`bool`, *optional*, defaults to `True`):
            Fold round-tripped state (KV cache, SSM states) into internal variables carried across `infer()`
            calls, with a `beam_idx` input for beam search. Set `False` for targets without stateful support
            (NPU).
    """

    export_format: ExportFormat = ExportFormat.OPENVINO

    output_path: str | PathLike | None = None
    compress_to_fp16: bool = False
    stateful: bool = True
