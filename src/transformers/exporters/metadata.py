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
"""What an exporter records about each graph it exports: precision, traced inputs, and input/output names."""

from __future__ import annotations

import importlib
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, fields
from typing import Any

from .. import __version__
from ..utils.import_utils import _is_package_available, is_torch_available
from .utils import module_dtype


if is_torch_available():
    import torch


def _describe_traced_kwarg(value: Any) -> dict[str, Any]:
    """How one traced kwarg was shaped, as JSON: rank/shape/dtype for a tensor, per leaf for a mapping, and the
    kind for anything else."""
    if isinstance(value, torch.Tensor):
        return {
            "rank": value.dim(),
            "shape": [int(dim) for dim in value.shape],
            "dtype": str(value.dtype).removeprefix("torch."),
        }
    if isinstance(value, Mapping):
        return {"leaves": {str(key): _describe_traced_kwarg(leaf) for key, leaf in value.items()}}
    return {"container": type(value).__name__}


def _package_versions(packages: Iterable[str]) -> dict[str, str]:
    """`{package: version}` for transformers and the exporter's packages, as provenance."""
    versions = {"transformers": __version__}
    for package in packages:
        exists, version = _is_package_available(package, return_version=True)
        if exists and version != "N/A":
            versions[package] = version
    return versions


@dataclass
class ExportMetadata:
    """What the exporter recorded about one graph ([`~exporters.metadata.build_export_metadata`])."""

    architecture: str | None = None
    packages: dict[str, str] = field(default_factory=dict)
    export_config: dict[str, Any] = field(default_factory=dict)
    export_dtype: str | None = None
    export_device: str | None = None
    kwargs: dict[str, dict] = field(default_factory=dict)
    # Pytree specs, from `treespec_dumps`.
    in_spec: str | None = None
    out_spec: str | None = None
    # Scalar kwargs traced as constants.
    in_constants: dict[str, Any] = field(default_factory=dict)
    # Non-tensor output leaves, as `[leaf index, value]`.
    out_constants: list = field(default_factory=list)

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> ExportMetadata:
        """Build the metadata from its saved dict, skipping fields this class doesn't have (added by a newer version)."""
        known = {f.name for f in fields(cls)}
        return cls(**{name: value for name, value in values.items() if name in known})


def build_export_metadata(
    model, inputs: Mapping[str, Any], exported_program, packages: Iterable[str] = ()
) -> ExportMetadata:
    """Record a graph's precision, traced kwargs, constants and pytree specs."""
    return ExportMetadata(
        architecture=type(model).__name__,
        packages=_package_versions(packages),
        export_dtype=str(module_dtype(model) or torch.get_default_dtype()).removeprefix("torch."),
        export_device=str(next((p.device for p in model.parameters()), torch.device("cpu"))),
        kwargs={name: _describe_traced_kwarg(value) for name, value in inputs.items()},
        in_constants={
            name: value
            for name, value in inputs.items()
            if value is None or isinstance(value, (bool, int, float, str))
        },
        in_spec=torch.utils._pytree.treespec_dumps(exported_program.call_spec.in_spec),
        out_spec=torch.utils._pytree.treespec_dumps(exported_program.call_spec.out_spec),
    )


def record_output_structure(metadata: ExportMetadata, outputs: Any) -> None:
    """Record what the model returned for a graph that returns its tensors only: the spec, and the other leaves."""
    leaves, spec = torch.utils._pytree.tree_flatten(outputs)
    constants = [[index, leaf] for index, leaf in enumerate(leaves) if not isinstance(leaf, torch.Tensor)]
    unsaveable = [leaf for _, leaf in constants if not (leaf is None or isinstance(leaf, (bool, int, float, str)))]
    if unsaveable:
        raise ValueError(f"The model returns {unsaveable}, which an export can't save: only tensors and scalars.")
    metadata.out_spec = torch.utils._pytree.treespec_dumps(spec)
    metadata.out_constants = constants


def register_treespec_types(dumped: str) -> None:
    """Register the transformers classes a spec written by `treespec_dumps` names as pytree nodes, so it can be read
    back: a `ModelOutput` registers itself when its module is imported, but a cache only once an export registers it."""
    from .exporter_dynamo import register_pytree_node

    for module, name in re.findall(r'"type": "(transformers(?:\.\w+)*)[.:](\w+)"', dumped):
        cls = getattr(importlib.import_module(module), name)
        if cls not in torch.utils._pytree.SUPPORTED_NODES:
            register_pytree_node(cls)


def load_treespec(dumped: str):
    """A spec written by `treespec_dumps`."""
    register_treespec_types(dumped)
    return torch.utils._pytree.treespec_loads(dumped)
