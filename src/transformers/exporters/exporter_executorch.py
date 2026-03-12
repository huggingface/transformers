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
import math
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

from ..utils import logging
from ..utils.export_config import ExecutorchConfig
from ..utils.import_utils import is_executorch_available, is_torch_available
from .exporter_dynamo import DynamoExporter


if is_torch_available():
    import torch
    from torch.export import ExportedProgram

if is_executorch_available():
    from executorch.exir.program import EdgeProgramManager, ExecutorchProgramManager, to_edge_transform_and_lower

if TYPE_CHECKING:
    from ..modeling_utils import PreTrainedModel


logger = logging.get_logger(__file__)


class ExecutorchExporter(DynamoExporter):
    export_config: ExecutorchConfig

    required_packages = ["torch", "executorch"]

    def export(self, model: "PreTrainedModel", sample_inputs: dict[str, Any]) -> "ExecutorchProgramManager":
        """Export a model for ExecuTorch."""
        if self.export_config.backend == "xnnpack":
            from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner

            model = model.to(device="cpu")
            partitioner = [XnnpackPartitioner()]
        elif self.export_config.backend == "cuda":
            from executorch.backends.cuda.cuda_backend import CudaBackend
            from executorch.backends.cuda.cuda_partitioner import CudaPartitioner

            model = model.to(device="cuda")
            model_name = model.__class__.__name__
            partitioner = [CudaPartitioner([CudaBackend.generate_method_name_compile_spec(model_name)])]
            if (
                next(model.parameters()).dtype != torch.bfloat16
                and model._can_set_attn_implementation()
                and model._supports_sdpa
            ):
                model = model.to(dtype=torch.bfloat16)
        else:
            raise ValueError(f"Unsupported backend {self.export_config.backend} for ExecuTorch export")

        with patch_torch_ops(model):
            exported_program: ExportedProgram = super().export(model, sample_inputs)
            edge_program_manager: EdgeProgramManager = to_edge_transform_and_lower(
                exported_program, partitioner=partitioner
            )
            executorch_programs_manager: ExecutorchProgramManager = edge_program_manager.to_executorch()

        return executorch_programs_manager


# ── Torch patches ──────────────────────────────────────────────────────────────
# Same factory pattern as exporter_onnx.py: each _patch_* receives the original
# and returns the replacement. _TORCH_PATCH_TABLE lists (obj, attr, factory).


def _patch_split(original):
    """Narrow-based split (split_copy not supported by CUDA backend)."""

    def patch(input, split_size_or_sections, dim=0):
        if isinstance(split_size_or_sections, int):
            splits = []
            total = input.size(dim)
            for i in range(0, total, split_size_or_sections):
                splits.append(input.narrow(dim, i, min(split_size_or_sections, total - i)))
            return tuple(splits)
        else:
            splits = []
            start = 0
            for size in split_size_or_sections:
                splits.append(input.narrow(dim, start, size))
                start += size
            return tuple(splits)

    return patch


def _patch_chunk(original):
    """Narrow-based chunk (delegates to split patch)."""

    def patch(input, chunks, dim=0):
        total = input.size(dim)
        chunk_size = (total + chunks - 1) // chunks
        # Call through torch.split which is already patched
        return torch.split(input, chunk_size, dim)

    return patch


def _patch_topk(original):
    """Argsort-based topk fallback."""

    def patch(input, k, dim=None, largest=True, sorted=True):
        if dim is None:
            dim = -1
        indices = torch.argsort(input, dim=dim, descending=largest)
        topk_indices = indices.narrow(dim, 0, k)
        topk_values = torch.gather(input, dim, topk_indices)
        return topk_values, topk_indices

    return patch


def _patch_detach(_original):
    """No-op detach."""

    def patch(input):
        return input

    return patch


def _patch_avg_pool2d(original):
    """Decompose avg_pool2d as depthwise conv2d (no CUDA ExecuTorch kernel)."""

    def patch(
        input, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True, divisor_override=None
    ):
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if stride is None:
            stride = kernel_size
        elif isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding)
        kh, kw = kernel_size
        h, w = input.shape[-2:]
        channels = input.shape[1]
        actual_kh = min(kh, h + padding[0] * 2)
        actual_kw = min(kw, w + padding[1] * 2)
        divisor = divisor_override if divisor_override is not None else actual_kh * actual_kw
        weight = input.new_ones(channels, 1, actual_kh, actual_kw) / divisor
        return torch.nn.functional.conv2d(input, weight, bias=None, stride=stride, padding=padding, groups=channels)

    return patch


def _patch_scaled_dot_product_attention(original):
    """Manual matmul+softmax fallback for asymmetric head dims (D_q != D_v)."""

    def patch(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, **kwargs):
        if query.shape[-1] != value.shape[-1]:
            scale_factor = scale if scale is not None else math.sqrt(query.shape[-1]) ** -1
            attn_weight = torch.matmul(query, key.transpose(-2, -1)) * scale_factor
            if is_causal:
                L, S = query.shape[-2], key.shape[-2]
                causal_mask = torch.ones(L, S, dtype=torch.bool, device=query.device).tril()
                attn_weight = attn_weight.masked_fill(~causal_mask, float("-inf"))
            if attn_mask is not None:
                attn_weight = attn_weight + attn_mask
            attn_weight = torch.nn.functional.softmax(attn_weight, dim=-1)
            return torch.matmul(attn_weight, value)
        return original(
            query, key, value, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, scale=scale, **kwargs
        )

    return patch


# (object, attribute, factory) triples installed by patch_torch_ops.
_TORCH_PATCH_TABLE = [
    (torch, "split", _patch_split),
    (torch.Tensor, "split", _patch_split),
    (torch, "chunk", _patch_chunk),
    (torch.Tensor, "chunk", _patch_chunk),
    (torch, "topk", _patch_topk),
    (torch.Tensor, "topk", _patch_topk),
    (torch, "detach", _patch_detach),
    (torch.Tensor, "detach", _patch_detach),
    (torch.nn.functional, "avg_pool2d", _patch_avg_pool2d),
    (torch.nn.functional, "scaled_dot_product_attention", _patch_scaled_dot_product_attention),
]


@contextmanager
def patch_torch_ops(model: "PreTrainedModel"):
    """Context manager: install torch patches for ExecuTorch export."""
    originals = []
    for obj, attr, factory in _TORCH_PATCH_TABLE:
        original = getattr(obj, attr)
        originals.append((obj, attr, original))
        setattr(obj, attr, factory(original))

    try:
        yield
    finally:
        for obj, attr, original in originals:
            setattr(obj, attr, original)
