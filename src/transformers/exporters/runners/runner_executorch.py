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

import functools
import importlib.metadata
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


@functools.cache
def writes_inputs_in_place() -> bool:
    """Whether ExecuTorch writes an unplanned input a method mutates into the caller's tensor (pytorch/executorch#23183,
    #23508 for MLX): 1.7, or a nightly from 2026-11-17."""
    from packaging import version

    executorch_version = version.parse(importlib.metadata.version("executorch"))
    return executorch_version.release >= (1, 7) or (executorch_version.dev or 0) >= 20261117


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
        if not writes_inputs_in_place():
            return self.unflatten(list(self.method.execute(feed)))
        outputs = list(self.method.execute(feed, clone_outputs=False))
        if share_outputs:
            return self.unflatten(outputs)
        # Outputs sharing an input's storage are the caller's (a static cache written in place)
        given = {leaf.untyped_storage().data_ptr() for leaf in feed if isinstance(leaf, torch.Tensor)}
        return self.unflatten(
            [
                output.clone()
                if isinstance(output, torch.Tensor) and output.untyped_storage().data_ptr() not in given
                else output
                for output in outputs
            ]
        )
