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
        """
        Exports a model for ExecuTorch using the full export and lowering workflow.
        Args:
            model (`PreTrainedModel`): The model to export.
            sample_inputs (`Dict[str, Any]`): The sample inputs to use for the export.
        Returns:
            The exported model in ExecuTorch format.
        """

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

        with patch_for_executorch_export(model):
            exported_program: ExportedProgram = super().export(model, sample_inputs)
            edge_program_manager: EdgeProgramManager = to_edge_transform_and_lower(
                exported_program, partitioner=partitioner
            )
            executorch_programs_manager: ExecutorchProgramManager = edge_program_manager.to_executorch()

        return executorch_programs_manager


@contextmanager
def patch_for_executorch_export(model: "PreTrainedModel"):
    # ExecuTorch export patcher context
    # This context manager monkey-patches PyTorch ops that are unsupported or buggy in ExecuTorch backends.
    # The following ops are patched with fallback implementations:
    #   - torch.split / torch.Tensor.split: replaced with narrow-based fallback
    #   - torch.chunk / torch.Tensor.chunk: replaced with narrow-based fallback
    #   - torch.topk / torch.Tensor.topk: replaced with argsort-based fallback
    #   - torch.detach / torch.Tensor.detach: replaced with no-op fallback
    #   - torch.nn.functional.avg_pool2d: decomposed as depthwise conv2d with uniform weights (kernel clamped to input size)
    #   - torch.nn.functional.scaled_dot_product_attention: manual matmul+softmax fallback when D_q != D_v
    # These patches are only active during export and are reverted afterwards.
    original_torch_split = torch.split
    original_tensor_split = torch.Tensor.split
    original_torch_chunk = torch.chunk
    original_tensor_chunk = torch.Tensor.chunk
    original_torch_topk = torch.topk
    original_tensor_topk = torch.Tensor.topk
    original_torch_detach = torch.detach
    original_tensor_detach = torch.Tensor.detach
    original_avg_pool2d = torch.nn.functional.avg_pool2d
    original_scaled_dot_product_attention = torch.nn.functional.scaled_dot_product_attention

    def _split(input, split_size_or_sections, dim=0):
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

    def _chunk(input, chunks, dim=0):
        total = input.size(dim)
        chunk_size = (total + chunks - 1) // chunks
        return _split(input, chunk_size, dim)

    def _topk(input, k, dim=None, largest=True, sorted=True):
        if dim is None:
            dim = -1
        values = input
        if largest:
            indices = torch.argsort(values, dim=dim, descending=True)
        else:
            indices = torch.argsort(values, dim=dim, descending=False)
        topk_indices = indices.narrow(dim, 0, k)
        topk_values = torch.gather(values, dim, topk_indices)
        return topk_values, topk_indices

    def _detach(input):
        return input

    def _avg_pool2d(
        input, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True, divisor_override=None
    ):
        # aten::avg_pool2d has no CUDA ExecuTorch kernel. Decompose as a depthwise conv2d
        # with uniform weights (1 / kernel_area), which IS supported by the CUDA backend.
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
        # Clamp kernel to padded input size (handles large-kernel global-pooling patterns where
        # kernel_size == feature_map_size, e.g. nn.AvgPool2d(hidden_dim, ceil_mode=True)).
        actual_kh = min(kh, h + padding[0] * 2)
        actual_kw = min(kw, w + padding[1] * 2)
        divisor = divisor_override if divisor_override is not None else actual_kh * actual_kw
        weight = input.new_ones(channels, 1, actual_kh, actual_kw) / divisor
        return torch.nn.functional.conv2d(input, weight, bias=None, stride=stride, padding=padding, groups=channels)

    def _scaled_dot_product_attention(
        query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, **kwargs
    ):
        # ExecuTorch CUDA SDPA requires D_q == D_k == D_v. For asymmetric head dims
        # (e.g. DiffLlama where D_v = 2*D_q), implement attention manually using matmul + softmax.
        if query.shape[-1] != value.shape[-1]:
            import math

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
        return original_scaled_dot_product_attention(
            query, key, value, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, scale=scale, **kwargs
        )

    torch.split = _split
    torch.Tensor.split = _split
    torch.chunk = _chunk
    torch.Tensor.chunk = _chunk
    torch.topk = _topk
    torch.Tensor.topk = _topk
    torch.detach = _detach
    torch.Tensor.detach = _detach
    torch.nn.functional.avg_pool2d = _avg_pool2d
    torch.nn.functional.scaled_dot_product_attention = _scaled_dot_product_attention

    try:
        yield
    finally:
        torch.split = original_torch_split
        torch.Tensor.split = original_tensor_split
        torch.chunk = original_torch_chunk
        torch.Tensor.chunk = original_tensor_chunk
        torch.topk = original_torch_topk
        torch.Tensor.topk = original_tensor_topk
        torch.detach = original_torch_detach
        torch.Tensor.detach = original_tensor_detach
        torch.nn.functional.avg_pool2d = original_avg_pool2d
        torch.nn.functional.scaled_dot_product_attention = original_scaled_dot_product_attention
