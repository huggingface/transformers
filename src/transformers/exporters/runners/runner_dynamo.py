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
"""Runs a `torch.export` program through its unlifted module."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ...utils.import_utils import is_torch_available
from ..metadata import ExportMetadata, load_treespec
from .base import ModelRunner


if is_torch_available():
    import torch

if TYPE_CHECKING:
    from torch.export import ExportedProgram


class DynamoModelRunner(ModelRunner):
    """Runs an `ExportedProgram` through `ExportedProgram.module()`."""

    def __init__(self, module: torch.nn.Module, metadata: ExportMetadata):
        super().__init__(metadata)
        self.module = module
        # The program flattens its kwargs by position, in the order its input spec lists them.
        self.traced_kwarg_order = load_treespec(metadata.in_spec).children_specs[1].context

    @property
    def device(self) -> torch.device:
        return next(self.module.parameters(), torch.empty(0)).device

    @classmethod
    def from_program(cls, program: ExportedProgram, metadata: ExportMetadata, device=None) -> DynamoModelRunner:
        module = program.module()
        return cls(module if device is None else module.to(device), metadata)

    def __call__(self, share_inputs: bool = True, share_outputs: bool = False, **kwargs) -> Any:
        kwargs = {**self.metadata.in_constants, **kwargs}
        ordered = {name: kwargs[name] for name in self.traced_kwarg_order if name in kwargs}
        with torch.no_grad():
            return self.module(**ordered, **{name: value for name, value in kwargs.items() if name not in ordered})
