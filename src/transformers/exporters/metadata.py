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

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from .. import __version__
from ..utils import logging
from ..utils.import_utils import is_torch_available
from .utils import get_leaf_tensors


logger = logging.get_logger(__name__)

if is_torch_available():
    import torch

    from ..cache_utils import Cache

# The key the metadata travels under: an ONNX `metadata_props` entry, an ExecuTorch constant method.
EXPORT_METADATA_KEY = "transformers_export_metadata"


def _traced_kwarg(value: Any) -> dict[str, Any]:
    """How one traced kwarg was shaped, as JSON: rank and dtype for a tensor, the same per leaf for a
    mapping of them (a per-attention-type mask dict), and the container's kind for anything else — a cache
    is the one a runner has to recognise, since it feeds it as an object rather than a tensor."""
    if isinstance(value, torch.Tensor):
        return {"rank": value.dim(), "dtype": str(value.dtype).removeprefix("torch.")}
    if isinstance(value, Mapping):
        return {"leaves": {str(key): _traced_kwarg(leaf) for key, leaf in value.items()}}

    return {"container": "cache" if isinstance(value, Cache) else type(value).__name__}


def traced_output_names(exported_program) -> list[str]:
    """The model's own output leaf names, in output order, read off the program's output pytree.

    The exported call spec holds the output structure the trace returned; unflattening placeholders through
    it and walking them with `get_leaf_tensors` yields exactly the `{name: tensor}` keys the dynamo runner
    computes on the real outputs at run time — so an artifact can carry them for a runner that otherwise
    sees only positional tensors."""
    spec = exported_program.call_spec.out_spec
    placeholders = [torch.empty(0) for _ in range(spec.num_leaves)]
    return list(get_leaf_tensors(torch.utils._pytree.tree_unflatten(placeholders, spec)))


def _package_versions(packages: Iterable[str]) -> dict[str, str]:
    """`{package: version}` for what produced an artifact — transformers plus whatever the exporter needs.

    Provenance for a file that outlives the environment that wrote it: a `.pte` or `.onnx` that misbehaves
    is usually a version story (a lowering that changed, an op that moved), and the artifact is the only
    place that can still say which versions were involved."""
    from ..utils.import_utils import _is_package_available

    versions = {"transformers": __version__}
    for package in packages:
        exists, version = _is_package_available(package, return_version=True)
        if exists and version != "N/A":
            # Local build suffixes (`+cu126`, `+cpu`) name the wheel, not the API — keep them, they are
            # exactly what distinguishes two otherwise identical torch versions when a kernel misbehaves.
            versions[package] = version
    return versions


def _traced_input_shapes(module) -> dict[str, tuple[int | None, ...]]:
    """`{input name: shape}` from the traced graph's placeholders, `None` per symbolic axis."""
    return {
        node.name: tuple(int(dim) if isinstance(dim, int) else None for dim in node.meta["val"].shape)
        for node in getattr(getattr(module, "graph", None), "nodes", [])
        if node.op == "placeholder" and hasattr(node.meta.get("val"), "shape")
    }


def _traced_cache_leaf_shapes(module) -> dict[int, tuple[int | None, ...]]:
    """`{leaf index: shape}` for the graph's cache inputs, keyed by the index in the placeholder's own name.

    Keyed, not positional: a cache tensor the trace folded into a constant (a static sliding layer's
    `_sliding_window_tensor`) has no placeholder at all, so counting placeholders in order would shift
    every leaf after it — and the pytree context refers to leaves by index."""
    shapes = {}
    for node in getattr(getattr(module, "graph", None), "nodes", []):
        if node.op != "placeholder" or not node.name.startswith("past_key_values"):
            continue
        index = node.name.rsplit("_", 1)[-1]
        value = node.meta.get("val")
        if index.isdigit() and hasattr(value, "shape"):
            shapes[int(index)] = tuple(int(dim) if isinstance(dim, int) else None for dim in value.shape)
    return shapes


def _traced_kv_geometry(module) -> dict[int, tuple[int, int, int]]:
    """`{layer index: (num_kv_heads, key_head_dim, value_head_dim)}` from the traced cache.

    The serialized context records which leaf *each* layer's `keys` / `values` occupy, so the shapes are
    matched by name. Matching by position instead cannot work: a hybrid model's recurrent states are rank-4
    too, so any rank-based pairing shifts the geometry of every attention layer after the first
    linear-attention one.
    """
    shapes = _traced_cache_leaf_shapes(module)
    kwargs_spec = module._in_spec.child(1)
    names = list(kwargs_spec.context or [])
    cache_name = next((name for name in ("past_key_values", "cache_params") if name in names), None)
    if cache_name is None:
        return {}
    context = kwargs_spec.children_specs[names.index(cache_name)].context
    state = context.get("s", {}) if isinstance(context, dict) else {}
    geometry = {}
    for index, layer in enumerate(state.get("layers", []) or []):
        entries = layer.get("s", {}) if isinstance(layer, dict) else {}
        keys, values = entries.get("keys"), entries.get("values")
        if not (isinstance(keys, dict) and keys.get("_t") == "tensor" and isinstance(values, dict)):
            continue
        key_shape, value_shape = shapes.get(keys["i"]), shapes.get(values["i"])
        if key_shape is None or value_shape is None:
            continue
        if len(key_shape) == 4 and None not in (key_shape[1], key_shape[3], value_shape[3]):
            geometry[index] = (key_shape[1], key_shape[3], value_shape[3])
    return geometry


@dataclass(frozen=True)
class ExportMetadata:
    """What the exporter recorded about one graph (`build_export_metadata`), parsed.

    The artifact itself only says what its inputs are *called* and what shape they were traced at; this is
    the trace's own account of what they mean, and it is what every runner accessor reads. Build it from
    whatever the backend carries the payload as — JSON text for ONNX (a `metadata_props` entry) and
    ExecuTorch (a constant method), the dict itself for a dynamo program, which *is* the program.

    Empty for an artifact written before the metadata existed, or by another tool: every accessor then
    answers `None`/empty and each runner falls back to what its declared tensors say, which is what all of
    them used to do.
    """

    raw: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_json(cls, payload: str | None) -> ExportMetadata:
        """Parse a text payload, tolerating anything unreadable — a corrupt or foreign entry under our key
        is not worth failing an otherwise loadable artifact over."""
        if not payload:
            return cls()
        try:
            metadata = json.loads(payload)
        except json.JSONDecodeError:
            logger.warning_once(f"Ignoring an unreadable `{EXPORT_METADATA_KEY}`; falling back to inference.")
            return cls()
        return cls.from_dict(metadata)

    @classmethod
    def from_dict(cls, metadata: Any) -> ExportMetadata:
        return cls(metadata) if isinstance(metadata, Mapping) else cls()

    def __bool__(self) -> bool:
        return bool(self.raw)

    @property
    def schema_version(self) -> int | None:
        """Version of the payload's own schema, so a reader can tell what to expect of it."""
        version = self.raw.get("schema_version")
        return version if isinstance(version, int) else None

    @property
    def architecture(self) -> str | None:
        """What was exported. Provenance only — nothing reads it to decide behaviour."""
        return self.raw.get("architecture")

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
        """How many of the returned leaves are the model's own, before a backend appended its mutated
        inputs — `None` when unrecorded, which leaves the runner to ask its own handle."""
        count = self.raw.get("num_user_outputs")
        return count if isinstance(count, int) else None

    @property
    def shapes(self) -> dict[str, tuple[int | None, ...]]:
        """Shape per input, `None` per symbolic axis — the shapes the *trace* saw."""
        return {name: tuple(shape) for name, shape in self.raw.get("shapes", {}).items()}

    @property
    def dtype(self) -> torch.dtype | None:
        """The precision the graph was exported at — not sniffed off whichever tensor is float."""
        dtype = getattr(torch, self.raw.get("dtype") or "", None)
        return dtype if isinstance(dtype, torch.dtype) else None

    @property
    def constant_inputs(self) -> dict[str, Any]:
        """Declared inputs that carry no tensor, and the value each holds — a `None` mask slot the trace kept
        (see `input_names`). A positional backend fills these rather than skipping them."""
        constants = self.raw.get("constant_inputs")
        return constants if isinstance(constants, dict) else {}

    @property
    def kwargs(self) -> dict[str, dict]:
        """The kwargs the graph was traced with, under the names the model's forward uses — before each
        backend mangled them into its own input names."""
        kwargs = self.raw.get("kwargs")
        return kwargs if isinstance(kwargs, dict) else {}

    @property
    def mask_ranks(self) -> dict[str, int | None] | None:
        """`{attention type: rank}` when the graph was traced with a *dict* of masks, else `None`.

        A single plain mask is recorded as one tensor rather than a mapping, and the generation layer keys on
        the dict form only — so that case stays `None`, as does an artifact carrying no metadata.

        Every entry the trace declared is kept, with rank `None` for one it held no mask in (a
        linear-attention slot the model never built a mask for). Dropping those would feed the graph a
        *shorter* dict than it was traced with: `None` is a pytree leaf, so the slot counts toward the input
        spec either way."""
        leaves = self.kwargs.get("attention_mask", {}).get("leaves")
        return {name: leaf.get("rank") for name, leaf in leaves.items()} if leaves else None

    @property
    def kv_geometry(self) -> dict[int, tuple[int, int, int]]:
        """`{layer index: (num_kv_heads, key_head_dim, value_head_dim)}` from the recorded cache layout.

        A layer whose state is not keys-and-values (a recurrent layer's conv / SSM buffers) has no geometry
        and is absent, which is what the caller checks."""
        layers = (self.raw.get("cache") or {}).get("layers") or []
        return {
            index: (layer["heads"], layer["key_dim"], layer["value_dim"])
            for index, layer in enumerate(layers)
            if {"heads", "key_dim", "value_dim"} <= layer.keys()
        }


def build_export_metadata(
    model, inputs: Mapping[str, Any], exported_program, packages: Iterable[str] = ()
) -> dict[str, Any]:
    """The facts about a graph that its runner would otherwise have to infer from names and shapes.

    A loaded artifact says what its inputs are *called* — after each backend has mangled the names, ONNX
    prefixing mutated ones with `input.` and ExecuTorch flattening pytrees to `<kwarg>_<leaf>` — and what
    shape they were traced at, but nothing about what they mean. So every runner ended up guessing: the
    compute precision off whichever tensor happened to be floating point, the mask layout off name
    prefixes, the cache kwarg by trying `past_key_values` then `cache_params`. Each guess has been wrong at
    least once, and feeding an fp32 cache to a half-precision program was read as a backend limitation and
    hid a bug across every MoE model.

    What goes in is deliberately generic — the precision, and the kwargs as *traced*, with each one's rank,
    dtype and container. Nothing here names a model family or a component role: which of those kwargs is a
    mask, and which mask a causal one belongs in, stays with the generation layer that already knows, and it
    can now ask about un-mangled names. Anything a runner reads straight off its own handle (declared
    shapes, input order) stays there too.
    """
    metadata = {
        # Version of this payload's own schema, so a reader can tell what to expect of it.
        "schema_version": 1,
        "architecture": type(model).__name__,
        # And what wrote it, for the day the artifact outlives this environment. The architecture is
        # provenance too — it names what was exported, and nothing reads it to decide behaviour.
        "packages": _package_versions(packages),
        # Precision the graph computes in, from the model's own parameters rather than from whichever tensor
        # happens to be floating point: integer ids and masks say nothing about it, and a cache fed at the
        # wrong dtype is refused outright when the method binds its inputs. Read the parameters the way
        # `PreTrainedModel.dtype` does rather than asking for that property, because a component split out
        # of a multi-modal or encoder-decoder model is a plain `nn.Module` and does not carry it (`FSMTEncoder`).
        "dtype": str(
            next((p.dtype for p in model.parameters() if p.is_floating_point()), torch.get_default_dtype())
        ).removeprefix("torch."),
        "kwargs": {name: _traced_kwarg(value) for name, value in inputs.items()},
    }
    # What the *graph* is, as the trace saw it. Each backend used to rebuild these its own way — ExecuTorch
    # baking bespoke constant methods, ONNX matching cache input names with a regex and reading the geometry
    # back off their shapes — so they now come from one place: the flat input order (a `.pte` binds inputs
    # positionally and carries no names), the output leaf names and how many of the outputs are the model's
    # own (a lowering emits its mutated-input copies first), and the cache's per-layer geometry, which no
    # artifact states — shapes alone don't say which leaf is a layer's keys.
    graph_signature = exported_program.graph_signature
    # Every user input, in the order the program binds them, including the ones carrying no tensor: a `None`
    # kwarg the trace kept as a slot (a mask *dict* whose `full_attention` entry was `None` — `None` is a
    # pytree leaf, so the slot counts) is a `ConstantArgument`, which `graph_signature.user_inputs` reports as
    # `None` rather than a name. Read the specs instead, which name every argument kind: drop one and a
    # positional backend binds every later input a slot early.
    user_inputs = [spec.arg for spec in graph_signature.input_specs if spec.kind.name == "USER_INPUT"]
    metadata["input_names"] = [arg.name for arg in user_inputs]
    # What those valueless slots hold, so a positional backend can fill them rather than skip them.
    metadata["constant_inputs"] = {
        arg.name: arg.value for arg in user_inputs if type(arg).__name__ == "ConstantArgument"
    }
    metadata["output_names"] = traced_output_names(exported_program)
    metadata["num_user_outputs"] = sum(spec.kind.name == "USER_OUTPUT" for spec in graph_signature.output_specs)
    try:
        module = exported_program.module()
    except Exception:  # a program that cannot be unlifted describes neither
        module = None
    # The cache the graph was traced against, layer by layer: the class names say what *kind* of state each
    # layer keeps — growing or fixed-size, windowed, cross-attention — which is the question the runtime
    # otherwise has to put to whatever cache object it happens to hold, and a `DynamicSlidingWindowLayer`
    # answers it misleadingly (it grows, yet reports its window as a maximum length). The geometry comes
    # from the graph's own cache inputs, because a config cannot always give it.
    geometry = _traced_kv_geometry(module) if module is not None else {}
    cache = next((value for value in inputs.values() if isinstance(value, Cache)), None)
    layers = getattr(getattr(cache, "self_attention_cache", cache), "layers", []) if cache is not None else []
    if cache is not None:
        metadata["cache"] = {
            "class": type(cache).__name__,
            "layers": [
                {
                    "class": type(layer).__name__,
                    **dict(zip(("heads", "key_dim", "value_dim"), geometry.get(index, ()))),
                }
                for index, layer in enumerate(layers)
            ],
        }
    # The shapes the trace *saw*, `None` per symbolic axis. This is not the same fact as the shape an
    # artifact *declares*, which is why the runners still read that off their own handle: ONNX reports
    # symbolic dims however it spells them, and a `.pte` reports the capacity its memory planner reserved —
    # inflated by the exporter's own caps, so a dim traced at 4 can be declared 1024. Recording the trace's
    # account keeps "what was this graph built for" answerable in one place, and identically per backend.
    shapes = _traced_input_shapes(module) if module is not None else {}
    metadata["shapes"] = {name: list(shape) for name, shape in shapes.items()}
    return metadata
