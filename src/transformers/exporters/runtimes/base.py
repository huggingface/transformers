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
from typing import TYPE_CHECKING, Any, Unpack

from ...utils import is_torch_available, logging
from ...utils.hub import DownloadKwargs
from ..metadata import ExportMetadata, load_treespec


if is_torch_available():
    import torch

if TYPE_CHECKING:
    from ...configuration_utils import PreTrainedConfig


logger = logging.get_logger(__name__)


class ModelRunner(ABC):
    """Runs one exported program on its backend: `runner(**kwargs)` returns what the source model returns."""

    def __init__(self, metadata: ExportMetadata):
        self.metadata = metadata
        self.out_spec = load_treespec(metadata.out_spec)

    @property
    @abstractmethod
    def device(self) -> torch.device:
        """Where the program runs."""

    @property
    def dtype(self) -> torch.dtype:
        """The precision the program was exported at."""
        return getattr(torch, self.metadata.dtype)

    def unflatten(self, outputs: list) -> Any:
        """The model's own outputs, rebuilt from a program's flat ones, where they come last (after the copies of
        mutated inputs a lowering adds)."""
        return torch.utils._pytree.tree_unflatten(outputs[len(outputs) - self.out_spec.num_leaves :], self.out_spec)

    @classmethod
    @abstractmethod
    def from_program(cls, program: Any, metadata: ExportMetadata, device=None) -> ModelRunner:
        """A runner for a program as an export returns it or [`~HfExporter.load_program`] reads it back, on
        `device` (the backend's default when `None`)."""

    @abstractmethod
    def __call__(self, share_inputs: bool = True, share_outputs: bool = False, **kwargs) -> Any:
        """Run the program on the traced forward's kwargs. `share_inputs` reads them in place rather than from copies;
        `share_outputs` returns views of the backend's own output buffers rather than copies, which its next call
        overwrites. A backend that always does one or the other ignores them (ONNX Runtime binds inputs in place and
        hands its outputs over, a torch program copies neither)."""


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
        # What it runs, to save it back (set by [`~ExportArtifacts.runtime`]).
        self.artifacts = None

    def __repr__(self) -> str:
        return f"{type(self).__name__}(runner={type(self.runner).__name__})"

    @property
    def device(self) -> torch.device:
        return self.runner.device

    @property
    def dtype(self) -> torch.dtype:
        return self.runner.dtype

    def save_pretrained(self, save_directory: str | Path, **kwargs) -> None:
        """Save the export this runs, as [`~ExportArtifacts.save_pretrained`] does."""
        self.artifacts.save_pretrained(save_directory, **kwargs)

    def push_to_hub(self, repo_id: str, **kwargs) -> Any:
        """Upload the export this runs, as [`~ExportArtifacts.push_to_hub`] does."""
        return self.artifacts.push_to_hub(repo_id, **kwargs)

    def __call__(self, share_inputs: bool = True, share_outputs: bool = False, **kwargs) -> Any:
        """Run the export, returning what the source model returns. Kwargs the export was not traced with are
        dropped, so a processor's whole output can be passed. `share_inputs` and `share_outputs` go to the runner
        (see [`ModelRunner.__call__`]): with `share_outputs`, the outputs are valid until the next call."""
        traced = self.runner.metadata.kwargs
        unused = [name for name in kwargs if name not in traced]
        if unused:
            logger.warning_once(f"Dropping inputs the export was not traced with: {unused}.")
        inputs = {name: value for name, value in kwargs.items() if name in traced}
        return self.runner(share_inputs=share_inputs, share_outputs=share_outputs, **inputs)

    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path: str | Path, *, device=None, **kwargs: Unpack[DownloadKwargs]
    ) -> ExportedModel:
        """Load an export written by [`~ExportArtifacts.save_pretrained`], from a local directory or a Hub repo, to
        run on `device`, as the class it runs as ([`ExportArtifacts.runtime`] picks it): any kind from
        [`ExportedModel`], this class's own kind from a subclass. `kwargs` are download kwargs, as for
        [`~ExportArtifacts.from_pretrained`]."""
        from ..base import ExportArtifacts

        model = ExportArtifacts.from_pretrained(pretrained_model_name_or_path, **kwargs).runtime(device=device)
        if not isinstance(model, cls):
            raise TypeError(
                f"{pretrained_model_name_or_path} runs as a {type(model).__name__}, not a {cls.__name__}: load it with "
                f"`{type(model).__name__}.from_pretrained` or `ExportedModel.from_pretrained`."
            )
        return model
