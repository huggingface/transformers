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
"""What an exporter records about a graph, and how it is read back.

A loaded artifact says what its inputs are *called* and what shape they were traced at, but nothing about
what they mean. This is the trace's own account of itself — the precision it computes in, the kwargs as
traced, the flat input/output order — written at export and saved next to the artifact.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from .. import __version__
from ..utils.import_utils import is_torch_available
from .utils import get_leaf_tensors


if is_torch_available():
    import torch


def _traced_kwarg(value: Any) -> dict[str, Any]:
    """How one traced kwarg was shaped, as JSON: rank/shape/dtype for a tensor, per leaf for a mapping, and the
    kind for anything else."""
    if isinstance(value, torch.Tensor):
        return {
            "rank": value.dim(),
            "shape": [int(dim) for dim in value.shape],
            "dtype": str(value.dtype).removeprefix("torch."),
        }
    if isinstance(value, Mapping):
        return {"leaves": {str(key): _traced_kwarg(leaf) for key, leaf in value.items()}}
    return {"container": type(value).__name__}


def traced_output_names(exported_program) -> list[str]:
    """The model's output leaf names in order, read off the program's output pytree."""
    spec = exported_program.call_spec.out_spec
    placeholders = [torch.empty(0) for _ in range(spec.num_leaves)]
    return list(get_leaf_tensors(torch.utils._pytree.tree_unflatten(placeholders, spec)))


def _package_versions(packages: Iterable[str]) -> dict[str, str]:
    """`{package: version}` for transformers and the exporter's packages, as provenance."""
    from ..utils.import_utils import _is_package_available

    versions = {"transformers": __version__}
    for package in packages:
        exists, version = _is_package_available(package, return_version=True)
        if exists and version != "N/A":
            versions[package] = version
    return versions


@dataclass(frozen=True)
class ExportMetadata:
    """What the exporter recorded about one graph (`build_export_metadata`), parsed.

    Empty for an artifact without it; every accessor then answers `None`/empty.
    """

    raw: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, metadata: Any) -> ExportMetadata:
        return cls(metadata) if isinstance(metadata, Mapping) else cls()

    @property
    def input_names(self) -> tuple[str, ...]:
        """What the graph takes, in the flat order it takes them."""
        return tuple(self.raw.get("input_names", ()))

    @property
    def output_names(self) -> tuple[str, ...]:
        """What the graph returns, in the flat order it returns them."""
        return tuple(self.raw.get("output_names", ()))

    @property
    def num_user_outputs(self) -> int | None:
        """How many returned leaves are the model's own, before a backend's appended mutated inputs."""
        return self.raw.get("num_user_outputs")

    @property
    def dtype(self) -> torch.dtype | None:
        """The precision the graph was exported at."""
        dtype = getattr(torch, self.raw.get("dtype") or "", None)
        return dtype if isinstance(dtype, torch.dtype) else None

    @property
    def device(self) -> torch.device | None:
        """The device the graph was exported on."""
        device = self.raw.get("device")
        return torch.device(device) if device else None

    @property
    def constant_inputs(self) -> dict[str, Any]:
        """Declared inputs that carry no tensor (e.g. a `None` mask slot), with their values."""
        return self.raw.get("constant_inputs") or {}

    @property
    def kwargs(self) -> dict[str, dict]:
        """The traced kwargs under the model forward's own names."""
        return self.raw.get("kwargs") or {}


def build_export_metadata(
    model, inputs: Mapping[str, Any], exported_program, packages: Iterable[str] = ()
) -> dict[str, Any]:
    """Record a graph's precision, traced kwargs and flat input/output order."""
    metadata = {
        "schema_version": 1,
        "architecture": type(model).__name__,
        "packages": _package_versions(packages),
        # Off the parameters: a split-out component is a plain `nn.Module` with no `.dtype`.
        "dtype": str(
            next((p.dtype for p in model.parameters() if p.is_floating_point()), torch.get_default_dtype())
        ).removeprefix("torch."),
        "device": str(next((p.device for p in model.parameters()), torch.device("cpu"))),
        "kwargs": {name: _traced_kwarg(value) for name, value in inputs.items()},
    }
    graph_signature = exported_program.graph_signature
    # Includes `ConstantArgument` slots (`None` in `user_inputs`): a positional backend must still bind them.
    user_inputs = [spec.arg for spec in graph_signature.input_specs if spec.kind.name == "USER_INPUT"]
    metadata["input_names"] = [arg.name for arg in user_inputs]
    metadata["constant_inputs"] = {
        arg.name: arg.value for arg in user_inputs if type(arg).__name__ == "ConstantArgument"
    }
    metadata["output_names"] = traced_output_names(exported_program)
    metadata["num_user_outputs"] = sum(spec.kind.name == "USER_OUTPUT" for spec in graph_signature.output_specs)
    return metadata
