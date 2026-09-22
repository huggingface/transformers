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
from .cache import _self_attention_layers
from .utils import _class_to_path, _path_to_class, get_leaf_tensors


logger = logging.get_logger(__name__)

if is_torch_available():
    import torch

    from ..cache_utils import Cache

# The key the metadata travels under: an ONNX `metadata_props` entry, an ExecuTorch constant method.
EXPORT_METADATA_KEY = "transformers_export_metadata"


def _traced_kwarg(value: Any) -> dict[str, Any]:
    """How one traced kwarg was shaped, as JSON: rank and dtype for a tensor, the same per leaf for a
    mapping of them (a per-attention-type mask dict), and the container's kind for anything else — a cache
    is the one a runner has to recognise, since it feeds it as an object rather than a tensor.

    A container also records the class it was, as `module:qualname`. The graph takes such a kwarg as that
    *type* (dynamo bakes it into the input spec, and the other backends name their leaves by its fields),
    so a runtime assembling one has to build the same class rather than something shaped like it — an
    encoder output carrying more than hidden states (parakeet's frame mask) only survives as its own type.
    """
    if isinstance(value, torch.Tensor):
        return {"rank": value.dim(), "dtype": str(value.dtype).removeprefix("torch.")}
    if isinstance(value, Mapping):
        recorded = {"leaves": {str(key): _traced_kwarg(leaf) for key, leaf in value.items()}}
        # A per-attention-type mask is a plain dict and is rebuilt as one. An encoder's output is a mapping
        # too — `ModelOutput` subclasses `OrderedDict` — but the graph takes it as its own class, so that is
        # recorded here rather than under `container`, which this branch already claimed.
        if type(value) is not dict:
            recorded["class"] = _class_to_path(type(value))
        return recorded

    kind = "cache" if isinstance(value, Cache) else type(value).__name__
    return {"container": kind, "class": _class_to_path(type(value))}


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


def _traced_cache_leaf_shapes(module) -> dict[int, tuple[int | None, ...]]:
    """`{leaf index: shape}` for the graph's cache inputs, keyed by the index in the placeholder's own name.

    Keyed, not positional: a cache tensor the trace folded into a constant (a static sliding layer's
    `sliding_window_tensor`) has no placeholder at all, so counting placeholders in order would shift
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


def _traced_cache_layout(module) -> dict[int, dict[str, int]]:
    """`{layer index: {"heads", "key_dim", "value_dim", "length", "indexer"}}` from the traced cache — what
    the metadata records about each layer, in the shape it records it.

    Each key is present only where the trace stated it: a recurrent layer keeps conv / SSM buffers rather
    than keys and values and has no geometry, and a length belongs to a layer that was *sized* rather than
    grown (the sizes need not be uniform across one cache).

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
        # A fixed-size layer carries the length it was built for in its own context, and layers of one
        # cache need not agree: mllama sizes its cross-attention layers to the vision sequence and its
        # self-attention ones to the generation length.
        if isinstance(entries.get("max_cache_len"), int):
            recorded["length"] = entries["max_cache_len"]
        # A sparse-indexer layer keeps a third tensor beside keys and values, but not every layer of such a
        # cache writes one: hy_v4's "shared" indexer layers reuse the last full layer's, so their slot stays
        # empty and the graph has one leaf fewer there. Only the trace says which is which.
        if "indexer_keys" in entries:
            recorded["indexer"] = isinstance(entries["indexer_keys"], dict)
        if recorded:
            layout[index] = recorded
    return layout


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
    def dtype(self) -> torch.dtype | None:
        """The precision the graph was exported at — not sniffed off whichever tensor is float."""
        dtype = getattr(torch, self.raw.get("dtype") or "", None)
        return dtype if isinstance(dtype, torch.dtype) else None

    @property
    def device(self) -> torch.device | None:
        """The device the graph was exported on — not read off weights a compiled artifact no longer has."""
        device = self.raw.get("device")
        return torch.device(device) if device else None

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

    def kwarg_class(self, name: str) -> type | None:
        """The class a container kwarg was traced as, imported — `None` when the trace recorded none (an
        artifact written before this, or a kwarg that was a plain tensor). Importing it is also what
        registers that class as a pytree node, which a graph loaded from disk needs before it is called."""
        path = self.kwargs.get(name, {}).get("class")
        return _path_to_class(path) if path else None

    @property
    def mask_rank(self) -> int | None:
        """The rank the graph's `attention_mask` was traced with, `None` when it takes none.

        `generate` upgrades a 2-D padding mask to the 4-D causal mask for any compileable cache, assuming
        the model's forward wants one — but an exported graph starts *after* whatever mask building its
        model does, so only the trace can say which it took. An alibi model (bloom) reads the 2-D padding
        mask directly and compares its width to the cache length, so a 4-D mask fails a guard rather than
        mismatching a shape. Recorded in kwarg space, not read off an artifact's declared shapes — those
        are a different fact and gave this a different answer per backend."""
        return self.kwargs.get("attention_mask", {}).get("rank")

    @property
    def mask_dtype(self) -> torch.dtype | None:
        """The dtype the graph's `attention_mask` was traced with, `None` when it takes none. A model reads
        a bool mask and a float one differently (a keep-mask vs an additive bias), so a mask the runtime
        builds is built as the one the graph took."""
        name = self.kwargs.get("attention_mask", {}).get("dtype")
        return getattr(torch, name, None) if name else None

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
    def cache_lengths(self) -> dict[int, int]:
        """`{layer index: length}` for the traced cache's fixed-size layers; empty for a growing cache."""
        layers = (self.raw.get("cache") or {}).get("layers") or []
        return {index: layer["length"] for index, layer in enumerate(layers) if "length" in layer}

    @property
    def indexer_layers(self) -> dict[int, bool]:
        """`{layer index: whether the traced layer carried an indexer tensor}`, for the layers whose class
        keeps one at all. A layer absent here was not traced with an indexer slot to speak of."""
        layers = (self.raw.get("cache") or {}).get("layers") or []
        return {index: layer["indexer"] for index, layer in enumerate(layers) if "indexer" in layer}

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
        # Where it was exported, for the backends whose artifact cannot say. A `torch.export` program keeps
        # its weights and a runner reads the device off them, but a *compiled* one has none left to read —
        # AOTInductor bakes them into the package and TensorRT folds them into its engines, and a runner
        # that then assumed CPU had the generation loop building a CPU cache for a CUDA graph.
        "device": str(next((p.device for p in model.parameters()), torch.device("cpu"))),
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
    layout = _traced_cache_layout(module) if module is not None else {}
    cache = next((value for value in inputs.values() if isinstance(value, Cache)), None)
    layers = _self_attention_layers(cache)
    if cache is not None:
        metadata["cache"] = {
            "class": type(cache).__name__,
            "layers": [
                # Plus what the trace says about the layer's own state: its geometry, and the length it
                # was sized for when it was sized at all. The runtime builds to that length rather than
                # re-deriving a size, because `generate` sizes a fixed cache from the prompt in front of it
                # and from per-model facts (mllama's vision length), so the same config gives a different
                # cache elsewhere -- and a graph carries its cache's sizes in the input spec it refuses to
                # be called against anything else.
                {"class": type(layer).__name__, **layout.get(index, {})}
                for index, layer in enumerate(layers)
            ],
        }
    return metadata
