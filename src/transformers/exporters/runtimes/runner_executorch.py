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

from ...utils.import_utils import is_executorch_available, is_torch_available
from ..metadata import ExportMetadata
from ..utils import get_leaf_tensors
from .base import ModelRunner


if is_torch_available():
    import torch

if is_executorch_available():
    from executorch.exir import ExecutorchProgramManager
    from executorch.runtime import Program, Runtime


class ExecutorchModelRunner(ModelRunner):
    """Runs an ExecuTorch program's `forward` method. Inputs bind by position, in the order the export recorded."""

    def __init__(self, program: Program, metadata: ExportMetadata):
        super().__init__(metadata)
        self.method = program.load_method("forward")

    @classmethod
    def from_program(cls, program, metadata: ExportMetadata, device=None) -> ExecutorchModelRunner:
        """`device` is not used: a program runs where it was lowered for (its delegate decides)."""
        if isinstance(program, ExecutorchProgramManager):
            program = Runtime.get().load_program(program.buffer)
        return cls(program, metadata)

    def __call__(self, **kwargs) -> dict[str, torch.Tensor]:
        # Traced placeholders join a nested input's path with underscores (`encoder_outputs_last_hidden_state`), or,
        # for a pytree like a `Cache`, number its flattened leaves (`past_key_values_0`).
        leaves = {name.replace(".", "_"): tensor for name, tensor in get_leaf_tensors(kwargs).items()}
        for name, nested in kwargs.items():
            if not isinstance(nested, torch.Tensor):
                for index, leaf in enumerate(torch.utils._pytree.tree_leaves(nested)):
                    leaves.setdefault(f"{name}_{index}", leaf)
        constants = self.metadata.constant_inputs

        def value(name):
            if name in constants:
                return constants[name]
            # A scalar kwarg the trace kept as an input of its own.
            return leaves[name].contiguous() if name in leaves else kwargs[name]

        feed = tuple(value(name) for name in self.metadata.input_names)
        outputs = self.method.execute(feed)
        # The model's own outputs come last, after the copies of mutated inputs a lowering adds.
        num_user_outputs = self.metadata.num_user_outputs or len(outputs)
        return dict(zip(self.metadata.output_names, outputs[len(outputs) - num_user_outputs :]))
