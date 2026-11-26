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
from __future__ import annotations

import copy
import json
from dataclasses import dataclass, is_dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Literal

from ..utils import logging


if TYPE_CHECKING:
    from .quantizers import ExportQuantizer


logger = logging.get_logger(__name__)


class ExportFormat(StrEnum):
    """Identifies the export backend. Stored in [`ExportConfigMixin`] for serialisation round-trips."""

    EXECUTORCH = "executorch"
    OPENVINO = "openvino"
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

    export_format: ExportFormat | str

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
        return cls(**config_dict)

    def to_dict(self) -> dict[str, Any]:
        """
        Serializes this instance to a Python dictionary.

        Returns:
            `dict[str, Any]`: Dictionary of all the attributes that make up this configuration instance.
        """
        return copy.deepcopy(self.__dict__)

    def to_json_string(self) -> str:
        """Serializes this instance to a JSON string, without its quantizer (the export records what it did). Values
        JSON can't hold (`Dim`s) are written as their `repr`.

        Returns:
            `str`: String containing all the attributes that make up this configuration instance in JSON format.
        """
        values = {name: value for name, value in self.__dict__.items() if name != "quantizer"}
        return json.dumps(values, indent=2, sort_keys=True, default=lambda v: vars(v) if is_dataclass(v) else repr(v))

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
            When `True`, data-dependent shape guards are emitted as runtime asserts in the exported
            graph instead of failing the export at trace time when a guard wouldn't hold across the
            full symbolic shape range. Most transformer LLMs need this set to `True` when using
            fine-grained ``Dim(min=, max=)`` bounds. Not needed with ``dynamic=True`` / ``Dim.AUTO``,
            where ``torch.export`` infers shape relations instead of verifying them against the
            user-stated bounds.
        quantizer ([`~exporters.quantizers.ExportQuantizer`], *optional*):
            Post-training quantization applied during the export: [`~exporters.quantizers.PT2EQuantizer`] or
            [`~exporters.quantizers.NNCFTorchFXQuantizer`] on the FX graph for every backend,
            [`~exporters.quantizers.OnnxRuntimeQuantizer`] on the converted ONNX model, or
            [`~exporters.quantizers.NNCFOpenVINOQuantizer`] / [`~exporters.quantizers.NNCFOnnxQuantizer`] on the
            converted OpenVINO / ONNX model.
    """

    export_format: ExportFormat = ExportFormat.DYNAMO

    dynamic: bool = False
    strict: bool = False
    dynamic_shapes: dict[str, Any] | None = None
    prefer_deferred_runtime_asserts_over_guards: bool = False

    quantizer: ExportQuantizer | None = None


ORT_PROVIDERS = ("cpu", "cuda", "webgpu", "dml")


@dataclass
class OrtFusionConfig:
    """
    Which of ONNX Runtime's fusions `OnnxConfig(ort_fusions=...)` applies. onnxscript's norm, attention and GELU
    fusions always do.

    Args:
        provider (`str`):
            The execution provider the fused graph runs on: `"cpu"`, `"cuda"`, `"webgpu"` or `"dml"`.
        rotary_embeddings (`bool`, *optional*, defaults to `True`):
            Fuse rotary embeddings into `RotaryEmbedding`, folded into `GroupQueryAttention` where it takes them.
        group_query_attention (`bool`, *optional*, defaults to `True`):
            Fuse attentions reading a key-value cache into `GroupQueryAttention` (DirectML keeps `MultiHeadAttention`).
        mixture_of_experts (`bool`, *optional*, defaults to `False`):
            Fuse mixture-of-experts blocks into `MoE` (CUDA only), whose kernels need a Hopper or newer GPU.
        packed_projections (`bool`, *optional*, defaults to `True`):
            Pack the projections reading one input (query, key and value; gate and up) into one `MatMul`.
    """

    provider: Literal["cpu", "cuda", "webgpu", "dml"]
    rotary_embeddings: bool = True
    group_query_attention: bool = True
    mixture_of_experts: bool = False
    packed_projections: bool = True

    def __post_init__(self):
        if self.provider not in ORT_PROVIDERS:
            raise ValueError(f"`provider` must be one of {ORT_PROVIDERS}, got {self.provider!r}")
        if self.mixture_of_experts and self.provider != "cuda":
            raise ValueError("`mixture_of_experts` fuses ONNX Runtime's CUDA `MoE`: it needs `provider='cuda'`.")


@dataclass
class OnnxConfig(DynamoConfig):
    """
    Configuration class for exporting models to ONNX via `torch.onnx.export`.

    Inherits all fields from [`DynamoConfig`] (`dynamic`, `strict`,
    `dynamic_shapes`, `prefer_deferred_runtime_asserts_over_guards`).

    Args:
        opset_version (`int`, *optional*):
            ONNX opset version to target. Defaults to 23, which exports SDPA, RMS norms and rotary embeddings as
            single ops (`Attention`, `RMSNormalization`, `RotaryEmbedding`); to 20 with `ort_fusions` or a
            `quantizer`, which work on their decompositions. ONNX Runtime's TensorRT provider takes 22 at most.
        optimize (`bool`, *optional*, defaults to `True`):
            Run `onnxscript` optimisation passes (constant folding, dead-code
            elimination, …) on the exported graph. Disable for models that
            hit upstream `onnxscript` optimiser bugs.
        ort_fusions (`str` or [`OrtFusionConfig`], *optional*):
            Fuse the graph into ONNX Runtime's ops for the execution provider it runs on: `"cpu"`, `"cuda"`,
            `"webgpu"` or `"dml"` for its default fusions, or an [`OrtFusionConfig`] picking them. The fused graph only
            runs on ONNX Runtime. Requires `optimize=True`.
    """

    export_format: ExportFormat = ExportFormat.ONNX

    opset_version: int | None = None
    optimize: bool = True
    ort_fusions: Literal["cpu", "cuda", "webgpu", "dml"] | OrtFusionConfig | None = None

    def __post_init__(self):
        # A provider for its default fusions; a dict as saved with an export.
        if isinstance(self.ort_fusions, str):
            self.ort_fusions = OrtFusionConfig(provider=self.ort_fusions)
        elif isinstance(self.ort_fusions, dict):
            self.ort_fusions = OrtFusionConfig(**self.ort_fusions)
        if self.ort_fusions is not None and not self.optimize:
            raise ValueError("`ort_fusions` requires `optimize=True`.")


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
            - `"mlx"` — GPU inference via the ExecuTorch MLX backend on Apple Silicon.
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
    """

    export_format: ExportFormat = ExportFormat.EXECUTORCH

    backend: str = "xnnpack"
    alloc_graph_input: bool = True
    alloc_graph_output: bool = True
    alloc_mutable_buffers: bool = True


@dataclass
class OpenVINOConfig(DynamoConfig):
    """
    Configuration class for exporting models to OpenVINO IR via ``openvino.convert_model``.

    Inherits all fields from [`DynamoConfig`] (`dynamic`, `strict`, `dynamic_shapes`,
    `prefer_deferred_runtime_asserts_over_guards`).

    Args:
        compress_to_fp16 (`bool`, *optional*, defaults to `False`):
            Compress the converted model's `float32` weights to `float16`, halving its size. Off by default
            for the narrower range.
        stateful (`bool`, *optional*, defaults to `True`):
            Fold round-tripped state tensors (KV cache, SSM states, …) into internal OV
            variables (``ReadValue``/``Assign``). The runtime then carries state across
            ``infer()`` calls instead of marshalling cache tensors through inputs/outputs on
            every step, and a fused ``beam_idx`` input reorders state in-graph for beam search.
            No-op for models without round-tripped state (encoders, prefill-only exports). Set it
            to `False` for targets that take no stateful model, such as the NPU plugin.
    """

    export_format: ExportFormat = ExportFormat.OPENVINO

    compress_to_fp16: bool = False
    stateful: bool = True
