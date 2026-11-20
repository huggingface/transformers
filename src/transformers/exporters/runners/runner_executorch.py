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
"""Runs an ExecuTorch program in the ExecuTorch runtime."""

from __future__ import annotations

from typing import Any

from ...utils.import_utils import is_executorch_available, is_torch_available
from ..metadata import ExportMetadata, load_treespec
from .base import ModelRunner


if is_torch_available():
    import torch
    import torch.fx._pytree as fx_pytree

if is_executorch_available():
    from executorch.exir import ExecutorchProgramManager
    from executorch.runtime import Program, Runtime


class ExecutorchModelRunner(ModelRunner):
    """Runs an ExecuTorch program's `forward` method, its inputs flattened as `torch.export` traced them."""

    def __init__(self, program: Program, metadata: ExportMetadata):
        super().__init__(metadata)
        self.method = program.load_method("forward")
        self.in_spec = load_treespec(metadata.in_spec)

    @property
    def device(self) -> torch.device:
        return torch.device("cpu")

    @classmethod
    def from_program(cls, program, metadata: ExportMetadata, device=None) -> ExecutorchModelRunner:
        """`device` is not used: a program runs where it was lowered for (its delegate decides)."""
        if isinstance(program, ExecutorchProgramManager):
            program = Runtime.get().load_program(program.buffer)
        return cls(program, metadata)

    def __call__(self, share_inputs: bool = True, share_outputs: bool = False, **kwargs) -> Any:
        leaves = fx_pytree.tree_flatten_spec(((), {**self.metadata.in_constants, **kwargs}), self.in_spec)
        # A string is a constant of the program, whose slot takes `None`: the runtime has no string value.
        feed = [
            leaf.contiguous() if isinstance(leaf, torch.Tensor) else None if isinstance(leaf, str) else leaf
            for leaf in leaves
        ]
        return self.unflatten(list(self.method.execute(feed)))
