# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Modifications Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
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
"""What an exporter records about a graph, and how a runner reads it back.

A loaded artifact says what its inputs are *called* and what shape they were traced at, but nothing about
what they mean. This is the trace's own account of itself — the precision it computes in, the kwargs as
traced, the cache's per-layer geometry — written at export and read by every runner.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from .. import __version__
from ..utils.import_utils import is_torch_available
from .cache import _self_attention_layers
from .utils import _class_to_path, _path_to_class, get_leaf_tensors


if is_torch_available():
    import torch

    from ..cache_utils import Cache


def _traced_kwarg(value: Any) -> dict[str, Any]:
    """How one traced kwarg was shaped, as JSON: rank/shape/dtype for a tensor, per leaf for a mapping, and the
    kind for anything else.

    Containers also record their class (`module:qualname`): the graph takes the kwarg as that type, so a
    runtime has to rebuild the same class.
    """
    if isinstance(value, torch.Tensor):
        return {
            "rank": value.dim(),
            "shape": [int(dim) for dim in value.shape],
            "dtype": str(value.dtype).removeprefix("torch."),
        }
    if isinstance(value, Mapping):
        recorded = {"leaves": {str(key): _traced_kwarg(leaf) for key, leaf in value.items()}}
        # A `ModelOutput` is a mapping too, but the graph takes it as its own class.
        if type(value) is not dict:
            recorded["class"] = _class_to_path(type(value))
        return recorded

    kind = "cache" if isinstance(value, Cache) else type(value).__name__
    return {"container": kind, "class": _class_to_path(type(value))}


def traced_output_names(exported_program) -> list[str]:
    """The model's output leaf names in order, read off the program's output pytree, matching the dynamo
    runner's keys."""
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
            # Local build suffixes (`+cu126`) are kept: they distinguish builds when a kernel misbehaves.
            versions[package] = version
    return versions


def _traced_cache_leaf_shapes(module) -> dict[int, tuple[int | None, ...]]:
    """`{leaf index: shape}` for the cache inputs, keyed by the placeholder name's index since a cache tensor
    folded into a constant has no placeholder."""
    shapes = {}
    for node in getattr(getattr(module, "graph", None), "nodes", []):
        if node.op != "placeholder" or not node.name.startswith("past_key_values"):
            continue
        index = node.name.rsplit("_", 1)[-1]
        value = node.meta.get("val")
        if index.isdigit() and hasattr(value, "shape"):
            shapes[int(index)] = tuple(int(dim) if isinstance(dim, int) else None for dim in value.shape)
    return shapes


def _traced_cache_layout(module) -> dict[int, dict[str, int]]:
    """`{layer index: {"heads", "key_dim", "value_dim", "length", "indexer"}}` from the traced cache, each key
    present only where the trace stated it.

    Shapes are matched via the pytree context's leaf indices, not position: a hybrid model's recurrent states
    are rank-4 too.
    """
    shapes = _traced_cache_leaf_shapes(module)
    kwargs_spec = module._in_spec.child(1)
    names = list(kwargs_spec.context or [])
    cache_name = next((name for name in ("past_key_values", "cache_params") if name in names), None)
    if cache_name is None:
        return {}
    context = kwargs_spec.children_specs[names.index(cache_name)].context
    state = context.get("s", {}) if isinstance(context, dict) else {}
    layout = {}
    for index, layer in enumerate(state.get("layers", []) or []):
        entries = layer.get("s", {}) if isinstance(layer, dict) else {}
        keys, values = entries.get("keys"), entries.get("values")
        key_shape = shapes.get(keys["i"]) if isinstance(keys, dict) and keys.get("_t") == "tensor" else None
        value_shape = shapes.get(values["i"]) if isinstance(values, dict) and values.get("_t") == "tensor" else None
        recorded = {}
        if key_shape is not None and value_shape is not None and len(key_shape) == 4:
            heads, key_dim, value_dim = key_shape[1], key_shape[3], value_shape[3]
            if None not in (heads, key_dim, value_dim):
                recorded = {"heads": heads, "key_dim": key_dim, "value_dim": value_dim}
        # Lengths can differ per layer (mllama's cross-attention layers are sized to the vision sequence).
        if isinstance(entries.get("max_cache_len"), int):
            recorded["length"] = entries["max_cache_len"]
        # Not every indexer layer writes one: hy_v4's "shared" layers leave the slot empty.
        if "indexer_keys" in entries:
            recorded["indexer"] = isinstance(entries["indexer_keys"], dict)
        if recorded:
            layout[index] = recorded
    return layout


@dataclass(frozen=True)
class ExportMetadata:
    """What the exporter recorded about one graph (`build_export_metadata`), parsed.

    Empty for an artifact without it; every accessor then answers `None`/empty.
    """

    raw: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, metadata: Any) -> ExportMetadata:
        return cls(metadata) if isinstance(metadata, Mapping) else cls()

    def __bool__(self) -> bool:
        return bool(self.raw)

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
        count = self.raw.get("num_user_outputs")
        return count if isinstance(count, int) else None

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
        constants = self.raw.get("constant_inputs")
        return constants if isinstance(constants, dict) else {}

    @property
    def kwargs(self) -> dict[str, dict]:
        """The traced kwargs under the model forward's own names."""
        kwargs = self.raw.get("kwargs")
        return kwargs if isinstance(kwargs, dict) else {}

    def kwarg_class(self, name: str) -> type | None:
        """The class a container kwarg was traced as, imported (which also registers it as a pytree node)."""
        path = self.kwargs.get(name, {}).get("class")
        return _path_to_class(path) if path else None

    @property
    def mask_rank(self) -> int | None:
        """The rank the graph's `attention_mask` was traced with, `None` when it takes none."""
        return self.kwargs.get("attention_mask", {}).get("rank")

    @property
    def position_axes(self) -> int | None:
        """Rows of the traced M-RoPE `position_ids`, `None` for 2-D positions; not derivable from the config."""
        shape = self.kwargs.get("position_ids", {}).get("shape")
        return shape[0] if isinstance(shape, list) and len(shape) == 3 else None

    @property
    def mask_dtype(self) -> torch.dtype | None:
        """The dtype the graph's `attention_mask` was traced with (bool keep-mask vs additive float bias)."""
        name = self.kwargs.get("attention_mask", {}).get("dtype")
        return getattr(torch, name, None) if name else None

    @property
    def mask_ranks(self) -> dict[str, int | None] | None:
        """`{attention type: rank}` when the graph was traced with a *dict* of masks, else `None`.

        Slots traced as `None` are kept with rank `None`: `None` is a pytree leaf, so dropping them shortens
        the input spec."""
        leaves = self.kwargs.get("attention_mask", {}).get("leaves")
        return {name: leaf.get("rank") for name, leaf in leaves.items()} if leaves else None

    @property
    def _cache_layers(self) -> list[dict]:
        """The recorded cache layers, in order."""
        return (self.raw.get("cache") or {}).get("layers") or []

    @property
    def cache_lengths(self) -> dict[int, int]:
        """`{layer index: length}` for the traced cache's fixed-size layers; empty for a growing cache."""
        layers = self._cache_layers
        return {index: layer["length"] for index, layer in enumerate(layers) if "length" in layer}

    @property
    def indexer_layers(self) -> dict[int, bool]:
        """`{layer index: whether it carried an indexer tensor}`, for layers whose class keeps one."""
        layers = self._cache_layers
        return {index: layer["indexer"] for index, layer in enumerate(layers) if "indexer" in layer}

    @property
    def kv_geometry(self) -> dict[int, tuple[int, int, int]]:
        """`{layer index: (num_kv_heads, key_head_dim, value_head_dim)}`; non-KV layers are absent."""
        layers = self._cache_layers
        return {
            index: (layer["heads"], layer["key_dim"], layer["value_dim"])
            for index, layer in enumerate(layers)
            if {"heads", "key_dim", "value_dim"} <= layer.keys()
        }


def build_export_metadata(
    model, inputs: Mapping[str, Any], exported_program, packages: Iterable[str] = ()
) -> dict[str, Any]:
    """Record a graph's precision, traced kwargs, flat input/output order and cache layout for its runner."""
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
    try:
        module = exported_program.module()
    except Exception:  # a program that cannot be unlifted describes neither
        module = None
    layout = _traced_cache_layout(module) if module is not None else {}
    cache = next((value for value in inputs.values() if isinstance(value, Cache)), None)
    layers = _self_attention_layers(cache)
    if cache is not None:
        metadata["cache"] = {
            "class": type(cache).__name__,
            "layers": [{"class": type(layer).__name__, **layout.get(index, {})} for index, layer in enumerate(layers)],
        }
    return metadata
