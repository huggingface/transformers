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
    from executorch.exir.program import ExecutorchProgramManager, to_edge_transform_and_lower

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

            partitioner = [XnnpackPartitioner()]
        elif self.export_config.backend == "cuda":
            from executorch.backends.cuda.cuda_backend import CudaBackend
            from executorch.backends.cuda.cuda_partitioner import CudaPartitioner

            model_name = model.__class__.__name__
            partitioner = [CudaPartitioner([CudaBackend.generate_method_name_compile_spec(model_name)])]
        else:
            raise ValueError(f"Unsupported backend {self.export_config.backend} for ExecuTorch export")

        with patch_torch_for_executorch_export():
            exported_program: ExportedProgram = super().export(model, sample_inputs)
            executorch_programs_manager: ExecutorchProgramManager = to_edge_transform_and_lower(
                exported_program, partitioner=partitioner
            ).to_executorch()

        return executorch_programs_manager


@contextmanager
def patch_torch_for_executorch_export():
    # ExecuTorch export patcher context
    # This context manager monkey-patches PyTorch ops that are unsupported or buggy in ExecuTorch backends.
    # The following ops are patched with fallback implementations:
    #   - torch.split / torch.Tensor.split: replaced with slicing-based fallback
    #   - torch.topk / torch.Tensor.topk: replaced with argsort-based fallback
    #   - torch.detach / torch.Tensor.detach: replaced with no-op fallback
    # These patches are only active during export and are reverted afterwards.
    original_torch_split = torch.split
    original_tensor_split = torch.Tensor.split
    original_torch_topk = torch.topk
    original_tensor_topk = torch.Tensor.topk
    original_torch_detach = torch.detach
    original_tensor_detach = torch.Tensor.detach

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

    torch.split = _split
    torch.Tensor.split = _split
    torch.topk = _topk
    torch.Tensor.topk = _topk

    def _detach(input):
        return input

    torch.detach = _detach
    torch.Tensor.detach = _detach

    torch.split = _split
    torch.Tensor.split = _split
    torch.topk = _topk
    torch.Tensor.topk = _topk
    torch.detach = _detach
    torch.Tensor.detach = _detach

    try:
        yield
    finally:
        torch.split = original_torch_split
        torch.Tensor.split = original_tensor_split
        torch.topk = original_torch_topk
        torch.Tensor.topk = original_tensor_topk
        torch.detach = original_torch_detach
        torch.Tensor.detach = original_tensor_detach
