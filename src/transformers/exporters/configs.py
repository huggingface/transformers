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

from ..utils import logging


logger = logging.get_logger(__name__)


class ExportFormat(Enum):
    """Identifies the export backend. Stored in [`ExportConfigMixin`] for serialisation round-trips."""

    EXECUTORCH = "executorch"
    TENSORRT = "tensorrt"
    DYNAMO = "dynamo"
    ONNX = "onnx"
    AOTI = "aoti"


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
        config_dict = dict(config_dict)
        # Back to the enum, so a config built from a dictionary is the one built directly.
        if isinstance(config_dict.get("export_format"), str):
            config_dict["export_format"] = ExportFormat(config_dict["export_format"])
        return cls(**config_dict)

    def to_dict(self) -> dict[str, Any]:
        """
        Serializes this instance to a Python dictionary.

        Returns:
            `dict[str, Any]`: Dictionary of all the attributes that make up this configuration instance.
        """
        # The format as its value, so the dictionary is JSON as it stands — a saved `export_config.json` is
        # read back by name (`AutoExportConfig.from_dict`), and an enum repr is not a name.
        fields = copy.deepcopy(self.__dict__)
        fields["export_format"] = self.export_format.value
        return fields


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
            When `True`, data-dependent shape guards are emitted as runtime asserts in the exported
            graph instead of failing the export at trace time when a guard wouldn't hold across the
            full symbolic shape range. Most transformer LLMs need this set to `True` when using
            fine-grained ``Dim(min=, max=)`` bounds. Not needed with ``dynamic=True`` / ``Dim.AUTO``,
            where ``torch.export`` infers shape relations instead of verifying them against the
            user-stated bounds.
    """

    export_format: ExportFormat = ExportFormat.DYNAMO
    dynamic: bool = False

    strict: bool = False
    dynamic_shapes: dict[str, Any] | None = None
    prefer_deferred_runtime_asserts_over_guards: bool = False


@dataclass
class AotiConfig(DynamoConfig):
    """
    Configuration class for compiling models ahead of time with AOTInductor.

    Takes everything [`DynamoConfig`] does — the trace is the same one — plus:

    Args:
        inductor_configs (`dict[str, Any]`, *optional*):
            Inductor settings for the compilation, as `torch._inductor.aoti_compile_and_package` takes
            them (`{"max_autotune": True}` and the rest). The exporter adds the package's metadata entry
            to whatever is passed here.
    """

    export_format: ExportFormat = ExportFormat.AOTI
    inductor_configs: dict[str, Any] | None = None


@dataclass
class TensorrtConfig(DynamoConfig):
    """
    Configuration class for compiling models with TensorRT, through Torch-TensorRT.

    Takes everything [`DynamoConfig`] does — the trace is the same one — plus:

    Args:
        min_block_size (`int`, *optional*, defaults to 5):
            The smallest run of convertible ops that becomes an engine. Below it the ops are left to
            torch, on the grounds that an engine that small costs more to enter than it saves.
        truncate_double (`bool`, *optional*, defaults to `True`):
            Whether to run float64 work as float32. TensorRT has no float64, so the alternative to
            truncating is leaving every subgraph that touches one to torch.
        torch_executed_ops (`set[str]`, *optional*):
            Ops to leave to torch instead of converting. Defaults to the ones Torch-TensorRT cannot take
            from these graphs — see `DEFAULT_TORCH_EXECUTED_OPS`. Pass an empty set to convert everything
            and see what breaks.
        compiler_options (`dict[str, Any]`, *optional*):
            The rest of `torch_tensorrt.dynamo.compile`'s settings, passed through as given.
    """

    export_format: ExportFormat = ExportFormat.TENSORRT
    min_block_size: int = 5
    truncate_double: bool = True
    torch_executed_ops: set[str] | None = None
    compiler_options: dict[str, Any] | None = None


@dataclass
class OnnxConfig(DynamoConfig):
    """
    Configuration class for exporting models to ONNX via `torch.onnx.export`.

    Inherits all fields from [`DynamoConfig`] (`dynamic`, `strict`,
    `dynamic_shapes`, `prefer_deferred_runtime_asserts_over_guards`).

    Args:
        output_path (`str` or `PathLike`, *optional*):
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
    Configuration class for exporting models to ExecuTorch format.

    Inherits all fields from [`DynamoConfig`] (`dynamic`, `strict`,
    `dynamic_shapes`, `prefer_deferred_runtime_asserts_over_guards`).

    Args:
        backend (`str`, *optional*, defaults to `"xnnpack"`):
            Target ExecuTorch backend. Supported values:

            - `"xnnpack"` — CPU inference via the XNNPACK library (default; runs anywhere).
            - `"cuda"` — GPU inference via the ExecuTorch CUDA backend.
        alloc_graph_input (`bool`, *optional*, defaults to `True`):
            Whether the memory-planning pass reserves arena memory for graph inputs. When `False`,
            the runtime uses the caller-provided input buffers directly instead of copying into the
            arena — so an in-place `USER_INPUT_MUTATION` (e.g. a `StaticCache` write) lands in the
            caller's tensor rather than an arena copy.
        alloc_graph_output (`bool`, *optional*, defaults to `True`):
            Whether the memory-planning pass reserves arena memory for graph outputs. When `False`,
            the caller must bind output buffers at runtime (`Method::set_output_data_ptr`); binding an
            output to its mutated input's buffer avoids the copy-out roundtrip.
        alloc_mutable_buffers (`bool`, *optional*, defaults to `True`):
            Whether the memory-planning pass reserves arena memory for mutable buffers (model-resident
            state). Passed through to the `MemoryPlanningPass`.
        partition (`bool`, *optional*, defaults to `True`):
            Whether to hand eligible subgraphs to the backend's partitioner. When `False` the whole graph
            lowers to the portable kernels — slower, and the way past a backend whose compiler refuses a
            partition its own partitioner claimed (XNNPACK does this at method load, so the refusal only
            shows up when the program is run).
        partition_exclude (`tuple[str, ...]`, *optional*):
            Names of the backend partitioner's per-op configs to withhold, e.g.
            `("ViewCopyConfig",)`. Those ops lower to the portable kernels while everything else stays
            delegated — the targeted form of `partition=False`, for a backend that refuses a partition
            because of one op pattern rather than the whole graph. XNNPACK only; names come from
            `executorch.backends.xnnpack.partition.config.ALL_PARTITIONER_CONFIGS`.
    """

    export_format: ExportFormat = ExportFormat.EXECUTORCH

    backend: str = "xnnpack"
    partition: bool = True
    partition_exclude: tuple[str, ...] = ()
    alloc_graph_input: bool = True
    alloc_graph_output: bool = True
    alloc_mutable_buffers: bool = True
