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
"""Running an exported model: a runner per backend, and `ExportedModel`, which calls one like its source model."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ...utils import logging
from ...utils.generic import ModelOutput
from ..metadata import ExportMetadata


if TYPE_CHECKING:
    import torch

    from ...configuration_utils import PreTrainedConfig


logger = logging.get_logger(__name__)


class ModelRunner(ABC):
    """Runs one exported program the way its source module forwards: `runner(**kwargs)` returns the outputs as
    `{name: tensor}`, keyed by the model's own output names (`logits`, `last_hidden_state`, ...)."""

    def __init__(self, metadata: ExportMetadata):
        self.metadata = metadata

    @classmethod
    @abstractmethod
    def from_program(cls, program: Any, metadata: ExportMetadata, device=None) -> ModelRunner:
        """A runner for a program as an export returns it or [`~HfExporter.load_program`] reads it back, on
        `device` (the backend's default when `None`)."""

    @abstractmethod
    def __call__(self, **kwargs) -> dict[str, torch.Tensor]:
        """Run the program on the traced forward's kwargs."""


class ExportedModel:
    """Call an exported model the way you'd call the model it came from.

    Built by [`~ExportArtifacts.runtime`], or loaded with [`~ExportedModel.from_pretrained`]:

    ```python
    >>> exported = OnnxExporter().export(model, inputs, OnnxConfig(dynamic=True))
    >>> exported.save_pretrained("exported")
    >>> outputs = ExportedModel.from_pretrained("exported")(**inputs)
    >>> outputs.last_hidden_state
    ```
    """

    def __init__(self, runner: ModelRunner, config: PreTrainedConfig | None = None):
        self.runner = runner
        self.config = config

    def __repr__(self) -> str:
        return f"{type(self).__name__}(runner={type(self.runner).__name__})"

    def can_generate(self) -> bool:
        """Whether this export runs `generate`, as a model with `GenerationMixin` does: not a single graph."""
        return False

    def __call__(self, **kwargs) -> ModelOutput:
        """Run the export. Kwargs the export was not traced with are dropped, so a processor's whole output can be
        passed."""
        traced = self.runner.metadata.kwargs
        unused = [name for name in kwargs if name not in traced]
        if unused:
            logger.warning_once(f"Dropping inputs the export was not traced with: {unused}.")
        return ModelOutput(self.runner(**{name: value for name, value in kwargs.items() if name in traced}))

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str | Path, *, device=None, **kwargs) -> ExportedModel:
        """Load a single-graph export written by [`~ExportArtifacts.save_pretrained`], from a local directory or a
        Hub repo, to run on `device`. `kwargs` go to [`~ExportArtifacts.from_pretrained`] (`subfolder`, `revision`,
        `token`, `cache_dir`)."""
        # `exporters.base` imports this module (`ExportArtifacts.runtime` builds `ExportedModel`s).
        from ..base import ExportArtifacts

        return ExportArtifacts.from_pretrained(pretrained_model_name_or_path, **kwargs).runtime(device=device)
