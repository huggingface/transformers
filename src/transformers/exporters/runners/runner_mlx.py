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
"""Runs an MLX export: its graph rebuilt as MLX ops on every call, compiled once per input shape. Sizes are read from the
arrays the call gets, so one export runs on any shape its graph allows."""

from __future__ import annotations

from typing import Any

from ...utils.import_utils import is_mlx_available, is_torch_available
from ..metadata import ExportMetadata, load_treespec
from ..mlx_ops import MLXProgram, run_graph, to_mlx, to_torch
from .base import ModelRunner


if is_torch_available():
    import torch

if is_mlx_available():
    import mlx.core as mx


def _tensor_key(tensor: torch.Tensor) -> tuple:
    return tensor.data_ptr(), tensor.shape, tensor.stride(), tensor.dtype


class MLXModelRunner(ModelRunner):
    """Runs an [`MLXProgram`] on its tensor inputs flattened as `torch.export` traced them, compiled by `mx.compile`
    once per input shape.

    Outputs are torch views of MLX's memory. An input that is one of the last call's outputs (the cache a decode step
    returned) runs on that MLX array as is, so state that loops through `generate` is never copied back."""

    def __init__(self, program: MLXProgram, metadata: ExportMetadata, device=None):
        super().__init__(metadata)
        self.in_spec = load_treespec(metadata.in_spec)
        self.mlx_device = device
        graph, weights = program.graph, program.weights
        self.function = mx.compile(lambda *arrays: run_graph(graph, weights, arrays))
        self.last_outputs: dict[tuple, mx.array] = {}

    @property
    def device(self) -> torch.device:
        # Inputs and outputs cross through host memory, which Apple Silicon's GPU shares
        return torch.device("cpu")

    @classmethod
    def from_program(cls, program: MLXProgram, metadata: ExportMetadata, device=None) -> MLXModelRunner:
        """`device` is MLX's: `"gpu"`, `"cpu"`, or `None` for its default (the GPU where there is one)."""
        return cls(program, metadata, None if device is None else mx.Device(getattr(mx, str(device))))

    def __call__(self, share_inputs: bool = True, share_outputs: bool = False, **kwargs) -> Any:
        leaves = torch.fx._pytree.tree_flatten_spec(((), {**self.metadata.in_constants, **kwargs}), self.in_spec)
        tensors = [leaf for leaf in leaves if isinstance(leaf, torch.Tensor)]
        arrays = [self.last_outputs.get(_tensor_key(tensor)) for tensor in tensors]
        arrays = [to_mlx(tensor) if array is None else array for tensor, array in zip(tensors, arrays)]
        with mx.stream(self.mlx_device or mx.default_device()):
            outputs = self.function(*arrays)
            mx.eval(outputs)
        outputs_torch = [to_torch(output) for output in outputs]
        self.last_outputs = {_tensor_key(tensor): output for tensor, output in zip(outputs_torch, outputs)}
        return self.unflatten(outputs_torch)
