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
"""Running an exported program: `ModelRunner`, one per backend, which [`ExportedModel`] calls."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ...utils import is_torch_available
from ..metadata import ExportMetadata, load_treespec


if is_torch_available():
    import torch


class ModelRunner(ABC):
    """Runs one exported program on its backend: `runner(**kwargs)` returns what the source model returns."""

    def __init__(self, metadata: ExportMetadata):
        self.metadata = metadata
        self.out_spec = load_treespec(metadata.out_spec)
        self.out_constants = dict(metadata.out_constants)

    @property
    @abstractmethod
    def device(self) -> torch.device:
        """Where the program runs."""

    @property
    def dtype(self) -> torch.dtype:
        """The precision the program was exported at."""
        return getattr(torch, self.metadata.export_dtype)

    def unflatten(self, outputs: list) -> Any:
        """Rebuild the model's outputs from the program's flat ones (which come last) and the recorded constants."""
        tensors = iter(outputs[len(outputs) - (self.out_spec.num_leaves - len(self.out_constants)) :])
        leaves = [
            self.out_constants[index] if index in self.out_constants else next(tensors)
            for index in range(self.out_spec.num_leaves)
        ]
        return torch.utils._pytree.tree_unflatten(leaves, self.out_spec)

    @classmethod
    @abstractmethod
    def from_program(cls, program: Any, metadata: ExportMetadata, device=None) -> ModelRunner:
        """A runner for a program as an export returns it or [`~HfExporter.load_program`] reads it back, on
        `device` (the backend's default when `None`)."""

    @abstractmethod
    def __call__(self, share_inputs: bool = True, share_outputs: bool = False, **kwargs) -> Any:
        """Run the program on the traced forward's kwargs. `share_inputs` reads them in place rather than from copies;
        `share_outputs` returns views of the backend's own output buffers rather than copies, which its next call
        overwrites. A backend that always does one or the other ignores them (a torch program copies neither)."""
