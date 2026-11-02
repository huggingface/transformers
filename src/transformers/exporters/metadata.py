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
import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, fields
from typing import Any

from .. import __version__
from ..utils.import_utils import _is_package_available, is_torch_available
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
    dtype: str | None = None
    device: str | None = None
    kwargs: dict[str, dict] = field(default_factory=dict)
    # Flat graph order.
    input_names: list[str] = field(default_factory=list)
    output_names: list[str] = field(default_factory=list)
    # `torch.export`'s input and output pytree specs, as `treespec_dumps` writes them.
    in_spec: str | None = None
    out_spec: str | None = None
    # The traced cache's per-layer key/value shapes, sparse-indexer slots and linear-attention state shapes, which
    # a runtime materializes the cache it feeds with.
    cache_geometry: dict[str, dict] = field(default_factory=dict)
    export_config: dict[str, Any] = field(default_factory=dict)


def build_export_metadata(
    model, inputs: Mapping[str, Any], exported_program, packages: Iterable[str] = ()
) -> dict[str, Any]:
    """Record a graph's precision, traced kwargs and flat input/output order. Precision and device are read off the
    parameters, since a split-out component is a plain `nn.Module` with no `.dtype`."""
    metadata = {
        "architecture": type(model).__name__,
        "packages": _package_versions(packages),
        "dtype": str(
            next((p.dtype for p in model.parameters() if p.is_floating_point()), torch.get_default_dtype())
        ).removeprefix("torch."),
        "device": str(next((p.device for p in model.parameters()), torch.device("cpu"))),
        "kwargs": {name: _traced_kwarg(value) for name, value in inputs.items()},
    }
    input_specs = exported_program.graph_signature.input_specs
    metadata["input_names"] = [spec.arg.name for spec in input_specs if spec.kind.name == "USER_INPUT"]
    metadata["output_names"] = traced_output_names(exported_program)
    cache = next((inputs[name] for name in ("past_key_values", "cache_params") if inputs.get(name) is not None), None)
    if cache is not None:
        from .cache import indexer_layers_of, kv_geometry_of, linear_states_of

        cache = getattr(cache, "self_attention_cache", cache)
        geometry = {
            "kv": kv_geometry_of(cache),
            "indexers": indexer_layers_of(cache),
            "linear_states": linear_states_of(cache),
        }
        # As it saves: JSON keys are strings and its sequences lists.
        metadata["cache_geometry"] = json.loads(json.dumps(geometry))
    metadata["in_spec"] = torch.utils._pytree.treespec_dumps(exported_program.call_spec.in_spec)
    metadata["out_spec"] = torch.utils._pytree.treespec_dumps(exported_program.call_spec.out_spec)
    return metadata


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


def export_config_metadata(config) -> dict[str, Any]:
    """`config` as JSON, without its quantizer. Values JSON can't hold (`Dim`s) are written as their `repr`."""
    values = {f.name: getattr(config, f.name) for f in fields(config) if f.name != "quantizer"}
    return json.loads(json.dumps(values, default=repr))
