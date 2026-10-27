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

from typing import TYPE_CHECKING

from ...utils.import_utils import is_torch_available
from ..metadata import ExportMetadata
from ..utils import get_leaf_tensors
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

    @classmethod
    def from_program(cls, program: ExportedProgram, metadata: ExportMetadata, device=None) -> DynamoModelRunner:
        module = program.module()
        return cls(module if device is None else module.to(device), metadata)

    def __call__(self, **kwargs) -> dict[str, torch.Tensor]:
        with torch.no_grad():
            return get_leaf_tensors(self.module(**kwargs))
