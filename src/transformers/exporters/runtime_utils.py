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
"""Run exported generative models by orchestrating their component graphs through `GenerationMixin.generate`.

`HfExporter.export_for_generation` produces one graph per component; this module plugs the graphs back
together and drives the generation loop from artifacts + configs alone — no model instance, no checkpoint
weights. The model config and the generation config **used at export** are the contract: save them with
the artifacts and hand them back to `ExportedGenerator.from_runners` — the generation config declares the
cache the graphs were traced against (growing `DynamicCache` vs fixed-size `StaticCache` + length), so
nothing is introspected from the graphs. Two public pieces:

- `ModelRunner` — wraps a backend's runtime handle (an `onnxruntime.InferenceSession`, a `torch.export`
  unlifted `module()`, a loaded ExecuTorch program) so it forwards like the module it was exported from:
  `runner(**kwargs) -> {name: tensor}`, torch tensors in and out. One subclass per backend
  (`OnnxModelRunner`, `DynamoModelRunner`, `ExecutorchModelRunner`), each hiding how its backend carries
  tensors (ORT's numpy boundary, executorch's positional buffers) and the KV-cache (flat `input.<name>` /
  `output.<name>` tensors vs the `Cache` pytree).
- `ExportedGenerator` — a `GenerationMixin` over the component runners: a `decode` graph alone is
  decoder-only text generation; add a `text_embed` graph and `Modality` entries (one features graph per
  image / video / audio input) and prefill scatters each modality's features into `inputs_embeds` at its
  placeholder positions, injecting the grid-derived precompute the graphs expect from the config alone.
  `ExportedGenerator.from_runners` assembles either kind from `{component_name: runner}` + the saved
  configs.
"""

from __future__ import annotations

import copy
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
from torch.utils._pytree import tree_flatten, tree_leaves, tree_unflatten

from ..cache_utils import DynamicCache, EncoderDecoderCache, StaticCache
from ..generation import GenerationConfig, GenerationMixin
from ..modeling_outputs import BaseModelOutput, CausalLMOutputWithPast


class ModelRunner(ABC):
    """Wraps one exported artifact so it forwards like the module it was exported from:
    `runner(**kwargs) -> {name: tensor}`, torch tensors in and out.

    `kwargs` are the exported forward's kwargs — including `past_key_values` as a `Cache` object for a
    decode graph; each backend adapts it to however its graph carries the cache (flattened `input.<name>`
    tensors, positional buffers, or the pytree itself). The returned dict maps output leaf names
    (`logits`, `past_key_values.…`) to tensors, so callers read outputs the same way for every backend.

    `input_names` lists the graph's declared inputs — a caller feeds only what the graph takes;
    `text_input` and `mask_inputs` are derived from it so a generation loop can key its feed. `device` /
    `dtype` are where the backend lands its output tensors — the generator reads them off the decode
    runner, so nobody has to pass them in.
    """

    # Shape per declared input, `None` for a symbolic axis, when the backend's handle reports them. The
    # generic form of "what does this graph take": the generation layer derives what it needs from it (which
    # rank the mask was traced with, say) without the runner having to know what the inputs mean.
    input_shapes: dict[str, tuple[int | None, ...]] = {}

    # `{attention type: rank}` when the graph took a *dict* of masks instead (mixed full/sliding attention),
    # which a model builds inside its forward — so the runtime has to hand one in. Exactly one of the two is
    # set, and the graph is the only thing that can say which.
    mask_dict_ranks: dict[str, int] | None = None

    input_names: tuple[str, ...] = ()
    device: torch.device | str = "cpu"
    dtype: torch.dtype = torch.float32

    @property
    def cache_input(self) -> str | None:
        """The kwarg this graph takes its cache under — `"cache_params"` for a recurrent model (fixed-size
        conv / recurrent state), `"past_key_values"` otherwise, `None` for a graph with no cache.

        Matched across each backend's naming of the same thing: dynamo takes the whole pytree under the bare
        name, ONNX flattens it to `input.<kwarg>.<path>` inputs, ExecuTorch to `<kwarg>_<leaf index>`."""
        for kwarg in ("cache_params", "past_key_values"):
            if any(name.removeprefix("input.").startswith(kwarg) for name in self.input_names):
                return kwarg
        return None

    @abstractmethod
    def __call__(self, **kwargs) -> dict[str, torch.Tensor]:
        """Run the graph on `kwargs`; return its outputs as `{leaf_name: tensor}`."""


def _cache_tensors(past_key_values) -> list[torch.Tensor]:
    """The cache's tensor leaves, in the pytree order the exporter named them."""
    return [t for t in tree_leaves(past_key_values) if isinstance(t, torch.Tensor)]


def _assign_cache_entry(cache, path: list[str], value) -> None:
    """Write a decode step's cache output back at its named path (`layers.0.conv_states.0`).

    Keeps a fixed-size buffer in place (`copy_`, so the cache object and any graph that mutated it stay
    valid) and replaces the entry outright when it grew or did not exist yet — a growing `DynamicCache`
    returns longer tensors, and a recurrent layer's states start as `None`.
    """

    # A step is an index into a list, a key into a dict (a layer may keep its state dict-keyed by entry
    # name — deepseek_v4's `buffer_kv["compressor"]`), or an attribute.
    def read(container, part):
        if part.isdigit():
            return container[int(part)]
        if isinstance(container, dict):
            return container.get(part)
        return getattr(container, part, None)

    target = cache
    for part in path[:-1]:
        target = read(target, part)
    last = path[-1]
    current = read(target, last)
    if isinstance(current, torch.Tensor) and isinstance(value, torch.Tensor) and current.shape == value.shape:
        if current is not value:
            current.copy_(value)
        return
    if not isinstance(value, torch.Tensor) and isinstance(current, torch.Tensor):
        value = torch.tensor(value, dtype=current.dtype, device=current.device)
    if last.isdigit():
        target[int(last)] = value
    elif isinstance(target, dict):
        target[last] = value
    else:
        setattr(target, last, value)


def _self_attention_layers(cache) -> list:
    """A cache's self-attention layers. An `EncoderDecoderCache` keeps them in the two caches it pairs
    rather than on itself, so asking it for `.layers` finds nothing — every question here (how long, what
    geometry, does it keep keys, which states exist) is about the self-attention half."""
    if cache is None:
        return []
    return getattr(getattr(cache, "self_attention_cache", cache), "layers", [])


def _cache_length(cache) -> int:
    """`cache.get_seq_length()`, or 0 for a cache that has no attention layer to ask.

    A recurrent-only cache (mamba, rwkv, …) raises rather than answering: it keeps a fixed-size state
    instead of a growing sequence, so "how many tokens are in it" is only ever 0 or "already running",
    which its own `has_previous_state` flag records.
    """
    if cache is None:
        return 0
    try:
        return cache.get_seq_length()
    except (ValueError, StopIteration):
        started = any(
            all(getattr(layer, "has_previous_state", {}).values() or [False])
            for layer in _self_attention_layers(cache)
        )
        return 1 if started else 0


def _advance_cache(past_key_values, outputs: dict[str, torch.Tensor], num_new_tokens: int):
    """Advance the cache with a decode step's outputs (the `past_key_values.…` entries, in cache-leaf
    order). A fixed-size cache keeps its shapes, so `copy_` in place — preserving the cache object and its
    non-tensor state; a graph that already mutated the cache in place returns the same tensors, making the
    copy a no-op. A growing `DynamicCache` returns longer tensors (its seq axis grew), so rebuild the cache
    from the grown leaves through the registered cache pytree. The tensors themselves tell the two apart.

    Sliding layers additionally keep their running length in a plain python int — `cumulative_length_int`
    on static sliding layers, `cumulative_length` itself on growing ones. It's not a pytree tensor, so the
    decode graph never updates it (the static graph bakes it as a trace-time constant; the growing-cache
    rebuild resurrects the pre-step value from the pytree context). Advance it by the tokens just
    processed, the way the eager `update` does — deliberately NOT read from the static layer's
    `cumulative_length` tensor: `int(tensor)` is a device→host sync (which also blocks CUDA-graph
    capture), and once a sliding layer is full the tensor stops advancing while the int keeps counting."""
    # a recurrent model's graph names its cache outputs after its own kwarg (`cache_params.…`)
    cache_updates = [
        (name, value) for name, value in outputs.items() if name.startswith(("past_key_values", "cache_params"))
    ]
    if cache_updates:
        cache_leaves = _cache_tensors(past_key_values)
        # Align updates to cache leaves. ExecuTorch names its cache inputs by flat leaf index
        # (`past_key_values_<N>`) and may prune placeholders its lowering left unused, so index by the
        # suffix and keep the old leaf where no update came back; other backends' dotted names arrive in
        # leaf order. The `.pte` runtime also returns rank-0 updates as python scalars — re-wrap them.
        updated = list(cache_leaves)
        for position, (name, new) in enumerate(cache_updates):
            path = name.split(".")[1:]
            if path:
                # A dotted name is the leaf's path in the cache (`layers.0.conv_states.0`, `layers.1.keys`),
                # which is the only alignment that holds when the graph returns entries the cache has no
                # leaf for — a recurrent layer keeps its `conv_states` / `recurrent_states` as `None` until
                # a step produces them, so counting leaves would run off the end.
                _assign_cache_entry(past_key_values, path, new)
                continue
            # ExecuTorch names its cache inputs by flat leaf index and may prune the ones its lowering left
            # unused, so index by the suffix and keep the old leaf where no update came back.
            suffix = name.rsplit("_", 1)[-1]
            index = int(suffix) if suffix.isdigit() else position
            if not isinstance(new, torch.Tensor):
                new = torch.tensor(new, dtype=cache_leaves[index].dtype, device=cache_leaves[index].device)
            updated[index] = new
        if any(name.split(".")[1:] == [] for name, _ in cache_updates):
            if any(old.shape != new.shape for old, new in zip(cache_leaves, updated)):
                _, spec = tree_flatten(past_key_values)
                past_key_values = tree_unflatten(updated, spec)
            else:
                for old, new in zip(cache_leaves, updated):
                    if old is not new:
                        old.copy_(new)
    _mark_existing_states(past_key_values)
    for layer in _self_attention_layers(past_key_values):
        if hasattr(layer, "cumulative_length_int"):
            layer.cumulative_length_int += num_new_tokens
        elif isinstance(getattr(layer, "cumulative_length", None), int):
            layer.cumulative_length += num_new_tokens
    # An `EncoderDecoderCache` also keeps `is_updated` python flags (pytree context, so the graph never
    # flips them and the growing rebuild resurrects the pre-step values): every decoder step leaves the
    # cross cache written — the prefill graph writes it, decode graphs read it.
    if getattr(past_key_values, "is_updated", None):
        past_key_values.is_updated = dict.fromkeys(past_key_values.is_updated, True)
    return past_key_values


def _mark_existing_states(past_key_values) -> None:
    """Mark each layer's recurrent states as existing, exactly where they do.

    A recurrent layer records "these states exist now" as python bools in the pytree context, so the
    graph cannot flip them — whoever filled the states (a decode step's write-back, or the fresh-cache
    materialization) marks them the way the eager `update` would, or the cache no longer matches the
    traced spec."""
    for layer in _self_attention_layers(past_key_values):
        conv_states = getattr(layer, "conv_states", None)
        if isinstance(conv_states, dict):
            # ... and the scalars its `lazy_initialization` records alongside them
            for key, conv in conv_states.items():
                if isinstance(conv, torch.Tensor):
                    if isinstance(getattr(layer, "conv_kernel_size", None), dict):
                        layer.conv_kernel_size[key] = conv.shape[-1]
                    if getattr(layer, "dtype", None) is None:
                        layer.dtype, layer.device = conv.dtype, conv.device
        # ... and mark exactly the states that now exist: a conv-only layer (lfm2) never gets recurrent
        # states, so flipping its flag would describe a cache the graph was not traced with
        for flag, attr in (
            ("is_conv_states_initialized", "conv_states"),
            ("is_recurrent_states_initialized", "recurrent_states"),
            ("has_previous_state", None),
        ):
            marks = getattr(layer, flag, None)
            if not isinstance(marks, dict):
                continue
            for key in marks:
                if attr is None:
                    present = any(
                        isinstance(getattr(layer, name, {}).get(key), torch.Tensor)
                        for name in ("conv_states", "recurrent_states")
                        if isinstance(getattr(layer, name, None), dict)
                    )
                else:
                    states = getattr(layer, attr, None)
                    present = isinstance(states, dict) and isinstance(states.get(key), torch.Tensor)
                if present:
                    marks[key] = True


def _traced_input_shapes(module) -> dict[str, tuple[int | None, ...]]:
    """`{input name: shape}` from the traced graph's placeholders, `None` per symbolic axis."""
    return {
        node.name: tuple(int(dim) if isinstance(dim, int) else None for dim in node.meta["val"].shape)
        for node in getattr(getattr(module, "graph", None), "nodes", [])
        if node.op == "placeholder" and hasattr(node.meta.get("val"), "shape")
    }


def _traced_mask_dict_ranks(module) -> dict[str, int] | None:
    """`{attention type: rank}` when the graph took a *dict* of masks, else `None`.

    Not derivable from the input shapes: the graph takes one `attention_mask` kwarg and the per-type keys live
    in its pytree child spec, so this reads the spec. A mixed-attention model builds that dict inside its
    forward, which the graph starts after, so the runtime has to hand one in — and the ranks differ per type
    (a full-attention entry is the 4D causal mask, a linear-attention one the 2D padding mask).
    """
    kwargs_spec = module._in_spec.child(1)
    names = list(kwargs_spec.context or [])
    if "attention_mask" not in names:
        return None
    keys = getattr(kwargs_spec.children_specs[names.index("attention_mask")], "context", None)
    if not (isinstance(keys, list) and keys):
        return None
    ranks = [
        len(node.meta["val"].shape)
        for node in getattr(getattr(module, "graph", None), "nodes", [])
        if node.op == "placeholder"
        and node.name.startswith("attention_mask")
        and hasattr(node.meta.get("val"), "shape")
    ]
    return dict(zip(keys, ranks)) if len(ranks) == len(keys) else dict.fromkeys(keys, 4)


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


class DynamoModelRunner(ModelRunner):
    """`ModelRunner` backed by a `torch.export` unlifted module (`ExportedProgram.module()`) — the
    runnable, like ORT's session. Kwargs pass straight through (the KV-cache stays a `Cache` pytree the
    graph consumes natively); the output object is flattened to its named tensor leaves."""

    def __init__(self, module):
        self._module = module
        # The module requires the exact kwarg set it was traced with (including baked scalars like
        # `max_seqlen` that aren't graph placeholders) — its input pytree spec carries those kwarg names.
        self.input_names = tuple(module._in_spec.child(1).context)
        self.kv_geometry = _traced_kv_geometry(module)
        self.input_shapes = _traced_input_shapes(module)
        self.mask_dict_ranks = _traced_mask_dict_ranks(module)
        # Outputs land wherever the exported weights live.
        weight = next(self._module.parameters(), None)
        if weight is not None:
            self.device, self.dtype = weight.device, weight.dtype

    def __call__(self, **kwargs) -> dict[str, torch.Tensor]:
        from .utils import get_leaf_tensors

        # The module rejects kwargs it wasn't traced with — e.g. a prefill graph whose capture predates the
        # cache (models that create it inside the first forward); its cache still rides out on the outputs.
        if "past_key_values" not in self.input_names:
            kwargs.pop("past_key_values", None)
        output = self._module(**kwargs)
        if isinstance(output, torch.Tensor):
            return {"output": output}
        return get_leaf_tensors(output)


def _ort_to_torch_dtype(ort_type: str | None) -> torch.dtype | None:
    """torch dtype for an ORT input/output type string (`"tensor(float)"`), or `None` if it names no tensor.

    ORT spells the element type in ONNX's own vocabulary, so go through ONNX's enum and torch's mapping
    instead of keeping a table by hand: that covers every dtype ONNX has (fp8, int4, …) rather than the few
    someone happened to add, and a missing entry can't quietly become `None`. `onnx` / `onnxruntime` are
    optional dependencies, hence the local import — only an ONNX runner ever asks.
    """
    import onnx
    from torch.onnx import JitScalarType

    match = re.fullmatch(r"tensor\((\w+)\)", ort_type or "")
    if match is None:
        return None
    try:
        return JitScalarType.from_onnx_type(onnx.TensorProto.DataType.Value(match.group(1).upper())).dtype()
    except (ValueError, KeyError):
        return None


def _session_kv_geometry(shapes: dict[str, tuple]) -> dict[int, tuple[int, int, int]]:
    """`{layer index: (num_kv_heads, key_head_dim, value_head_dim)}` from an ORT session's input shapes.

    The ONNX graph names its cache inputs by dotted path, so this is a lookup by name — no leaf indices to
    keep in step, unlike the dynamo side."""
    geometry = {}
    for name, shape in shapes.items():
        match = re.fullmatch(r"input\.(?:past_key_values|cache_params)\.layers\.(\d+)\.keys", name)
        if match is None or len(shape) != 4:
            continue
        values = shapes.get(name.removesuffix("keys") + "values")
        if values is None or len(values) != 4:
            continue
        heads, key_dim, value_dim = shape[1], shape[3], values[3]
        if all(isinstance(dim, int) for dim in (heads, key_dim, value_dim)):
            geometry[int(match.group(1))] = (heads, key_dim, value_dim)
    return geometry


class OnnxModelRunner(ModelRunner):
    """`ModelRunner` backed by an `onnxruntime.InferenceSession`. Torch tensors in, torch tensors out —
    the numpy boundary ORT requires is confined here. The KV-cache rides as matched `input.<name>` /
    `output.<name>` graph inputs/outputs, so a `past_key_values` kwarg is flattened into the feed and the
    `output.` prefix stripped from the results (back to plain leaf names)."""

    def __init__(self, session):
        self._session = session
        self._output_names = [o.name for o in session.get_outputs()]

        # The exporter prefixes *mutated* inputs with `input.` — the cache leaves always, but also any plain
        # kwarg the graph writes to (a merged multi-token decode mutates its `attention_mask`). The cache
        # keeps the prefix (its dotted leaf paths are this runner's own convention), while a mutated plain
        # kwarg is exposed bare so the generation layer sees the name `generate` uses; `_session_name` maps
        # back when feeding.
        def exposed(name):
            if name.startswith(("input.past_key_values", "input.cache_params")):
                return name
            return name.removeprefix("input.")

        self._session_names = {exposed(i.name): i.name for i in session.get_inputs()}
        self.input_names = tuple(self._session_names)
        self._cache_names = [n for n in self.input_names if n.startswith("input.")]
        shapes = {exposed(i.name): tuple(i.shape) for i in session.get_inputs()}
        # ORT rejects a feed whose dtype differs from the declared one, and the masks the runtime builds are
        # not always the type the graph was traced with (a bool padding mask vs a float causal one).
        self._input_dtypes = {exposed(i.name): _ort_to_torch_dtype(i.type) for i in session.get_inputs()}
        self.input_shapes = shapes
        self.kv_geometry = _session_kv_geometry(shapes)
        # Same contract as the dynamo runner's: the rank the single `attention_mask` input was traced with,
        # so the generation loop feeds the mask kind this graph took rather than assuming 4D.
        self.mask_dict_ranks = _mask_dict_ranks_from_shapes(shapes)
        # Where the loop's tensors live / at what precision: the session's execution provider and the
        # graph's float (logits) output type. ORT still hands back host numpy, so `__call__` bridges —
        # feeds to CPU, outputs back to this device.
        self.device = "cuda" if any("CUDA" in p for p in session.get_providers()) else "cpu"
        # Precision from the first floating-point output, whatever it is called: a runner serves whichever
        # task its graph was exported for (`logits`, `last_hidden_state`, a classifier score), so it must not
        # go looking for one output by name. Integer-only outputs are skipped — they say nothing about the
        # graph's compute precision.
        self.dtype = next(
            (
                dtype
                for output in session.get_outputs()
                if (dtype := _ort_to_torch_dtype(output.type)) is not None and dtype.is_floating_point
            ),
            torch.float32,
        )

    def __call__(self, **kwargs) -> dict[str, torch.Tensor]:
        from .utils import get_leaf_tensors

        cache = kwargs.pop(self.cache_input, None) if self.cache_input else None
        if cache is not None:
            kwargs.update({name: t.detach() for name, t in zip(self._cache_names, _cache_tensors(cache))})
        # Non-tensor kwargs are pytrees (`encoder_outputs`, …) — the graph declares their leaves by dotted
        # path, the exporter's own naming.
        for name in [n for n, v in kwargs.items() if not isinstance(v, torch.Tensor)]:
            kwargs.update({f"{name}.{leaf}": t for leaf, t in get_leaf_tensors(kwargs.pop(name)).items()})
        feed = {
            self._session_names.get(name, name): tensor.detach()
            .to(self._input_dtypes.get(name) or tensor.dtype)
            .cpu()
            .numpy()
            for name, tensor in kwargs.items()
        }
        outputs = self._session.run(None, feed)
        return {
            name.removeprefix("output."): torch.from_numpy(value).to(self.device)
            for name, value in zip(self._output_names, outputs)
        }


def _baked_kv_geometry(program) -> dict[int, tuple[int, int, int]]:
    """The per-layer KV geometry the exporter baked into a `.pte`, or `{}` for a program without one."""
    try:
        flat = list(program.load_method("kv_geometry").execute(()))
    except Exception:
        return {}
    return {flat[i]: tuple(flat[i + 1 : i + 4]) for i in range(0, len(flat) - 3, 4)}


class ExecutorchModelRunner(ModelRunner):
    """`ModelRunner` backed by a loaded ExecuTorch runtime program
    (`Runtime.get().load_program(...)`). Unlike ONNX, a `.pte` carries no `input.`/`output.` convention:
    inputs are **positional** (in the source graph's flat order) and `execute` returns the lowering's
    mutated-input copies first, the model's own outputs last. A loaded method's metadata carries only counts
    and tensor shapes, so the exporter bakes the input names and the user-output count into the `.pte`
    itself as constant methods (`_signature_constant_methods`) — the program is self-describing, exactly
    like an ONNX session or a `torch.export` module, and this runner needs nothing else. Cache inputs are
    the `past_key_values*` ones, each named by its flat leaf index; the model's outputs are
    `[logits, *cache_updates]`, in cache-input order.

    Multi-token decode works: XNNPACK needs a *bounded* dynamic sequence dim, which `_fix_range_constraints`
    already supplies (it caps unbounded `Dim.AUTO` extents), so one graph serves prefill and decode.
    """

    # Cache inputs are matched by name; the model's own outputs come back under the names the trace recorded
    # (`logits`, `past_key_values.layers.0.keys`, …), the same mapping the other backends return.

    def __init__(self, program):
        self._method = program.load_method("forward")
        self.input_names = tuple(program.load_method("input_names").execute(()))
        self._output_names = tuple(program.load_method("output_names").execute(()))
        num_user_outputs = program.load_method("num_user_outputs").execute(())[0]
        total_outputs = self._method.metadata.num_outputs()
        self._user_output_indices = range(total_outputs - num_user_outputs, total_outputs)
        self._cache_names = [n for n in self.input_names if n.startswith(self.cache_input or "\0")]
        # Baked by the exporter (`_signature_constant_methods`): a `.pte` keeps shapes but nothing saying
        # which leaf is a layer's keys, so the geometry rides along as `layer, heads, key_dim, value_dim`.
        self.kv_geometry = _baked_kv_geometry(program)
        self.input_shapes = {
            name: tuple(self._method.metadata.input_tensor_meta(index).sizes())
            for index, name in enumerate(self.input_names)
        }
        # Same contract as the other runners': a graph that took a *dict* of masks declares one input per
        # attention type, so the generation loop has the ranks to build it rather than assuming a single mask.
        self.mask_dict_ranks = _mask_dict_ranks_from_shapes(self.input_shapes)

    def __call__(self, **kwargs) -> dict[str, torch.Tensor]:
        from .utils import get_leaf_tensors

        cache = kwargs.pop(self.cache_input, None) if self.cache_input else None
        if cache is not None:
            # Cache inputs are named by flat leaf index (`<kwarg>_<N>`) — index rather than zip,
            # since the lowering may have pruned leaves it left unused (sliding caches' scalars).
            leaves = _cache_tensors(cache)
            kwargs.update({name: leaves[int(name.rsplit("_", 1)[-1])] for name in self._cache_names})
        # Remaining non-tensor kwargs are pytrees (`encoder_outputs`, mask dicts) — the graph names their
        # leaves by underscore-joined path (`encoder_outputs_last_hidden_state`).
        for name in [n for n, v in kwargs.items() if not isinstance(v, torch.Tensor)]:
            value = kwargs.pop(name)
            kwargs.update({f"{name}_{leaf.replace('.', '_')}": t for leaf, t in get_leaf_tensors(value).items()})
        outputs = self._method.execute(tuple(kwargs[name].contiguous() for name in self.input_names))
        return dict(zip(self._output_names, (outputs[i] for i in self._user_output_indices)))


# The text-path kwargs `generate` always carries. A modality graph that declares one of these names means
# its own (an audio encoder's `attention_mask` covers mel frames, not prompt tokens), so it is never
# sourced from `generate`'s — those come from the capture or the precompute instead.
_TEXT_KWARGS = frozenset(
    {
        "input_ids",
        "inputs_embeds",
        "attention_mask",
        "position_ids",
        "token_type_ids",
        "cache_position",
        "past_key_values",
        "cache_params",
        "use_cache",
    }
)


def _grid_renamed(key: str) -> str:
    """`generate` names the grid per modality (`image_grid_thw` / `video_grid_thw`); the exported graph
    takes the one the getter itself declares, `grid_thw`."""
    return "grid_thw" if key.endswith("_grid_thw") else key


@dataclass
class Modality:
    """Routes one input modality (image / video / audio) of an `ExportedGenerator`:

    - `token_id`: the placeholder id in `input_ids` its features scatter into (`config.image_token_id`, …);
      `None` for a model that marks the rows with an explicit `*_position_mask` kwarg instead (kosmos2_5).
    - `runner`: the exported `get_<modality>_features` graph.
    - `input_keys`: the generate kwargs that belong to it (e.g. `("pixel_values", "image_grid_thw")`); the
      first is the presence key — the modality runs only when it's passed.
    """

    token_id: int | None
    runner: ModelRunner
    input_keys: tuple


def _pack_anyres_features(config, features, image_sizes, outputs) -> torch.Tensor:
    """The packing `PatchVisionEncoder` leaves out of the graph: per image, the base patch's tokens followed
    by its patch grid reshaped, unpadded to the image's aspect ratio and given a newline column per row.
    Mirrors `pack_image_features`, calling the modeling's own grid/unpad helpers so the geometry lives in one
    place — the split optimum-intel uses, and the reason the graph is dynamic in image count and resolution."""
    from ..models.llava_next.modeling_llava_next import get_anyres_image_grid_shape, unpad_image
    from .utils import _find_config_attr, anyres_patch_counts

    newline = next((t for name, t in outputs.items() if name.endswith("image_newline")), None)
    pinpoints = _find_config_attr(config, "image_grid_pinpoints")
    tile = _find_config_attr(config, "image_size")
    # Tokens per patch, not `image_size // patch_size`: a deepstack projector downsamples the token grid, so
    # the square side has to come off the projected tensor rather than the vision config.
    side = round(features.shape[1] ** 0.5)
    packed = []
    for index, feature in enumerate(torch.split(features, anyres_patch_counts(config, image_sizes), dim=0)):
        if feature.shape[0] > 1:
            base, patches = feature[0], feature[1:]
            num_patch_height, num_patch_width = get_anyres_image_grid_shape(image_sizes[index], pinpoints, tile)
            patches = patches.view(num_patch_height, num_patch_width, side, side, -1).permute(4, 0, 2, 1, 3)
            patches = unpad_image(patches.flatten(1, 2).flatten(2, 3), image_sizes[index])
            if newline is not None:
                column = newline[:, None, None].expand(*patches.shape[:-1], 1).to(patches)
                patches = torch.cat((patches, column), dim=-1)
            packed.append(torch.cat((base, patches.flatten(1, 2).transpose(0, 1)), dim=0))
        else:
            feature = feature[0]
            if newline is not None:
                feature = torch.cat((feature, newline[None].to(feature)), dim=0)
            packed.append(feature)
    return torch.cat(packed, dim=0)


class _ExportedEncoder:
    """`get_encoder()` stand-in over the exported encoder graph. `generate` filters its kwargs by
    `forward`'s signature (a wildcard here, so nothing is dropped), always adds the `output_*` /
    `return_dict` flags, and expects a `ModelOutput` back — the wrapper feeds the graph only what it
    declares and wraps its features as `BaseModelOutput.last_hidden_state`, the form the decoder graphs
    were traced with."""

    def __init__(self, runner: ModelRunner, merge=None):
        self._runner = runner
        self._merge = merge

    def forward(self, **kwargs):
        # A multi-modal encoder-decoder merges its features once, in front of the text encoder — the graph
        # takes `inputs_embeds`, and `merge` (the generator's own embed-and-scatter) builds them from the
        # prompt and whichever modality inputs this call carries.
        if self._merge is not None:
            merged = self._merge(kwargs.pop("input_ids"), kwargs)
            # The merge keys its primary entry by the embed graph's own output name; this graph declares it
            # under its text input, the way the decode feed maps it.
            primary, *extra = merged
            kwargs[_text_input(self._runner)] = merged[primary]
            kwargs.update({name: merged[name] for name in extra})
        feed = {k: v for k, v in kwargs.items() if k in set(self._runner.input_names)}
        return BaseModelOutput(last_hidden_state=next(iter(self._runner(**feed).values())))

    def __call__(self, **kwargs):
        return self.forward(**kwargs)


# ── How the generation loop reads a runner's declared inputs ────────────────
# Pure derivations from `ModelRunner.input_names`, and generation-specific: which input carries the prompt,
# which the cache, which the mask. They live here rather than on the runner so a runner stays what it is —
# a callable graph with named inputs, a device and a dtype — and can serve any task its graph was exported
# for (classification, embeddings, …), not only generation.


def _mask_dict_ranks_from_shapes(shapes: dict[str, tuple]) -> dict[str, int] | None:
    """`{attention type: rank}` for a graph that declares one mask input per type (mixed full/sliding
    attention), read off its declared input names — each backend spells the dict's leaves its own way
    (`attention_mask.full_attention` for ONNX, `attention_mask_full_attention` for ExecuTorch)."""
    ranks = {
        name.removeprefix(prefix): len(shape)
        for name, shape in shapes.items()
        for prefix in ("attention_mask.", "attention_mask_")
        if name.startswith(prefix)
    }
    return ranks or None


def _mask_type(name: str) -> str:
    """The attention type a per-type mask input names, under either backend's flattening."""
    return name.removeprefix("attention_mask.").removeprefix("attention_mask_")


def _declares(runner, name: str, value) -> bool:
    """Whether this graph takes the feed entry `name` — directly, or as the pytree whose leaves it names.

    A pytree kwarg (`encoder_outputs`, a mask dict, the cache) goes in under its *kwarg* name and each runner
    flattens it to whatever its backend calls the leaves (`encoder_outputs.last_hidden_state` for ONNX,
    `encoder_outputs_last_hidden_state` for ExecuTorch, the kwarg itself for dynamo). Only a non-tensor value
    is flattened, so a plain tensor must be named outright — `input_features` is not declared by a graph that
    only takes `input_features_mask`."""
    if name in runner.input_names or name == runner.cache_input:
        return True
    return not isinstance(value, torch.Tensor) and any(
        declared.removeprefix("input.").startswith((f"{name}.", f"{name}_")) for declared in runner.input_names
    )


def _mask_rank(runner) -> int | None:
    """Rank the graph's single `attention_mask` input was traced with, or `None` if it takes no mask.

    `generate` upgrades a 2D padding mask to the 4D causal mask for any compileable cache, assuming the
    model's forward wants one — but an exported graph starts *after* whatever mask building its model does,
    so only the trace can say which it took. An alibi model (bloom) reads the 2D padding mask directly and
    compares its width to the cache length, so a 4D mask fails a guard rather than mismatching a shape.
    """
    shape = runner.input_shapes.get("attention_mask")
    return len(shape) if shape is not None else None


def _text_input(runner) -> str:
    """The graph's text input: `"decoder_input_ids"` (encoder-decoder decode), `"inputs_embeds"`
    (multi-modal decode) or `"input_ids"`."""
    for name in ("decoder_input_ids", "inputs_embeds"):
        if name in runner.input_names:
            return name
    return "input_ids"


def _mask_inputs(runner) -> tuple[str, ...]:
    """The graph's attention-mask input name(s) — several for mixed full/sliding attention."""
    return tuple(
        n for n in runner.input_names if n == "attention_mask" or n.startswith(("attention_mask.", "attention_mask_"))
    )


def _decoder_mask_input(runner) -> str | None:
    """`"decoder_attention_mask"` when the graph declares one. An encoder-decoder splits the two masks:
    `attention_mask` covers the *encoder's* sequence (what cross-attention reads) while this one covers
    the decoder's own — so the causal mask belongs here, not in `attention_mask`, and `generate` does
    not hand it over (the eager model builds it inside the forward the graph starts after)."""
    return "decoder_attention_mask" if "decoder_attention_mask" in runner.input_names else None


class ExportedGenerator(GenerationMixin):
    """Drive exported component graphs through `generate`, from artifacts + configs alone.

    A `decode` runner alone is decoder-only text generation. Pass `text_embed` (the `input_ids -> inputs_embeds`
    graph) and `modalities` for a multi-modal model: prefill computes each modality's features and scatters
    them into `inputs_embeds` at its placeholder positions; decode steps are text-only. Pass `prefill=` to
    drive a fixed query=1 `decode` graph from a separate dynamic prefill graph (the shape CUDA-graph capture
    / io-binding want); when omitted, `decode` serves both (it must then be a multi-token graph, i.e.
    exported with `multi_token_decode=True`).

    `generation_config` must be the one the model was **exported with**: it declares the cache the decode
    graph was traced against (no `cache_implementation` → growing `DynamicCache`; `"static"` → fixed-size
    `StaticCache`, whose `max_cache_len` a static-cache export should pin explicitly).

    Example:
        programs = OnnxExporter().export_for_generation(model, inputs,
                                                        OnnxConfig(dynamic=True, external_data=False),
                                                        generation_config=generation_config,
                                                        multi_token_decode=True)
        session = ort.InferenceSession(programs["decode"].model_proto.SerializeToString())
        runtime = ExportedGenerator(model.config, generation_config, OnnxModelRunner(session))
        # Called like a normal model — the runtime builds the cache the exported graph needs.
        ids = runtime.generate(input_ids=prompt, max_new_tokens=32)
    """

    base_model_prefix = ""
    main_input_name = "input_ids"

    _supports_cache_class = True

    def __init__(
        self,
        config,
        generation_config,
        decode: ModelRunner,
        *,
        prefill: ModelRunner | None = None,
        encoder: ModelRunner | None = None,
    ):
        self.config = config
        self.generation_config = generation_config
        self._decode_runner = decode
        # (`_prefill`/`_decode` without the suffix would shadow `GenerationMixin` methods `generate` calls.)
        self._prefill_runner = prefill if prefill is not None else decode
        self._encoder = _ExportedEncoder(encoder) if encoder is not None else None
        # Everything the component graphs declare: `generate` kwargs matching these are fed straight through
        # (a model may take extra per-step tensors, e.g. `token_type_ids`), so they count as consumed.
        runners = [decode, self._prefill_runner, encoder]
        self._graph_inputs = {name for runner in runners if runner is not None for name in runner.input_names}
        # `GenerationMixin` bookkeeping (where it builds tensors, at what precision) — read off the decode
        # runner, whose backend already decided where its outputs land.
        self._device = torch.device(decode.device)
        self._dtype = decode.dtype

    @classmethod
    def from_runners(
        cls,
        runners: dict[str, ModelRunner],
        config,
        generation_config: GenerationConfig | None = None,
    ) -> ExportedGenerator:
        """Assemble the generator from `{component_name: runner}` (the names
        `HfExporter.export_for_generation` produces) + the configs — text-only from a `"decode"` runner,
        multi-modal when `"embed_tokens"` and `"<modality>_encoder"` runners are present; each modality's
        precompute is built from `config` alone. `generation_config` must be the one the model was
        **exported with** (it declares the cache the graphs were traced against — save it with the
        artifacts); when `None`, the model config's own generation defaults apply (a growing cache). To
        load from disk, build each `ModelRunner` from its saved artifact (e.g.
        `OnnxModelRunner(onnxruntime.InferenceSession(path))`,
        `DynamoModelRunner(torch.export.load(path).module())`) and pass a config from
        `AutoConfig.from_pretrained(...)`."""
        from .utils import _MODALITY_SPECS

        if generation_config is None:
            generation_config = GenerationConfig.from_model_config(config)
        # The scatter path applies when a graph takes embeddings where a plain model's takes token ids: the
        # decode graph (decoder-only VLMs) or the encoder graph (a multi-modal encoder-decoder like
        # florence2, whose decode then reads the merged features through `encoder_outputs`). Otherwise it
        # runs as a plain generator even when an embed graph was exported.
        takes_embeds = _text_input(runners["decode"]) == "inputs_embeds" or (
            "encoder" in runners and "inputs_embeds" in runners["encoder"].input_names
        )
        if "embed_tokens" not in runners or not takes_embeds:
            return ExportedGenerator(
                config,
                generation_config,
                runners["decode"],
                prefill=runners.get("prefill"),
                encoder=runners.get("encoder"),
            )

        modalities = []
        for name, _getter, spec_input_keys, grid_key, token_field in _MODALITY_SPECS:
            if name not in runners:
                continue
            token_id = getattr(config, token_field, None)
            if token_id is None:  # older configs name it `<modality>_token_index`
                token_id = getattr(config, f"{token_field[:-3]}_index", None)
            input_keys = tuple(spec_input_keys) + ((grid_key,) if grid_key is not None else ())
            modalities.append(Modality(token_id, runners[name], input_keys))
        return ExportedMultimodalGenerator(
            config,
            generation_config,
            runners["decode"],
            prefill=runners.get("prefill"),
            encoder=runners.get("encoder"),
            text_embed=runners["embed_tokens"],
            modalities=modalities,
        )

    # ── GenerationMixin plumbing (a real PreTrainedModel provides all of this) ──
    @property
    def device(self):
        return self._device

    @property
    def dtype(self):
        return self._dtype

    def can_generate(self):
        return True

    def is_remote_code(self):
        return False

    def get_experts_implementation(self):
        return {}

    def get_output_embeddings(self):
        return None

    def get_encoder(self):
        return self._encoder

    def get_compiled_call(self, compile_config):
        """`generate` swaps in a compiled forward for the decode loop when the cache is compileable. The
        graphs are already compiled (that is what exporting them was), so hand back the plain call."""
        return self.__call__

    def __call__(self, **kwargs):
        return self.forward(**kwargs)

    @property
    def _consumed_kwargs(self) -> set[str]:
        """What `generate` may carry that this runtime consumes without naming it on `forward`: whatever the
        component graphs declare as an input, which `forward` feeds through (a model may take extra per-step
        tensors, e.g. `token_type_ids`), plus the text-path kwargs the runtime routes itself — a model that
        derives one internally (ctrl and xlm build their own `token_type_ids`) exports a graph that never
        takes it, and dropping it here is exactly what its own forward did."""
        return self._graph_inputs | _TEXT_KWARGS

    def _validate_model_kwargs(self, model_kwargs):
        super()._validate_model_kwargs({k: v for k, v in model_kwargs.items() if k not in self._consumed_kwargs})

    def _supports_default_dynamic_cache(self) -> bool:  # noqa: D401 (instance form: reads the prototype)
        """Whether `generate` should build it a `DynamicCache`.

        On a real model this is a class-level fact; here it is read off the cache the graphs were traced
        against — a recurrent-only model (mamba, rwkv, …) keeps fixed-size states instead, and handing it a
        `DynamicCache` makes `generate` ask a question (`get_seq_length`) that cache refuses to answer; a
        model with no cache at all (openai-gpt recomputes the whole sequence every step) exports graphs that
        take none.
        """
        return self._decode_runner.cache_input == "past_key_values"

    @property
    def _is_recurrent(self) -> bool:
        """Whether the graphs carry fixed-size recurrent state instead of a growing KV cache — the decode
        graph says which by the kwarg it takes its cache under (`ModelRunner.cache_input`)."""
        return self._decode_runner.cache_input == "cache_params"

    def _prepare_cache_for_generation(
        self, generation_config, model_kwargs, generation_mode, batch_size, max_cache_length
    ):
        """Build the cache the exported decode graph expects, so `generate` is called like a normal model
        (`generate(input_ids=..., max_new_tokens=N)` — no hand-rolled cache).

        `generate`'s own builder decides everything (kind, size, `EncoderDecoderCache` pairing, a
        source-length static cross cache, …) from the generation config — the same config that built the
        cache at export capture, which is the caller's contract — so the traced and runtime caches match by
        construction. The one addition is materialization: `torch.export` bakes real tensors into the
        graph's input spec, so the lazily-uninitialized layers `generate` hands back have to be filled in
        (`materialize_cache_layers`, the helper the capture uses too)."""
        from .utils import materialize_cache_layers

        super()._prepare_cache_for_generation(
            generation_config, model_kwargs, generation_mode, batch_size, max_cache_length
        )
        # Nothing to build when the graphs take no cache at all: turn caching off so `generate`'s loop
        # re-feeds the whole sequence each step — what those graphs were traced on — instead of slicing to a
        # single-token step, and drop any cache handed in for graphs that cannot read one.
        if not self._is_recurrent and self._decode_runner.cache_input is None:
            model_kwargs.pop("past_key_values", None)
            generation_config.use_cache = False
            return
        # Beam search (and several returned sequences) expand the batch before the first decode call, so the
        # zero-length tensors materialized below have to be sized the way `generate` sizes a static cache.
        batch_size *= max(generation_config.num_beams, generation_config.num_return_sequences)
        # A recurrent model gets no cache from `generate` (it expects the model's own
        # `prepare_inputs_for_generation` to make one), so build the one its configs describe: `layer_types`
        # gives the same mix of attention and linear-attention layers the model would, and the generation
        # config says whether those were traced growing or at a fixed size.
        if self._is_recurrent:
            text_config = self.config.get_text_config()
            model_kwargs.setdefault(
                "cache_params",
                StaticCache(config=text_config, max_cache_len=max_cache_length)
                if generation_config.cache_implementation == "static"
                else DynamicCache(config=text_config),
            )
        # Cross-attention caches the encoder's states in full, so it is never sliding — but `generate` builds
        # both halves of an `EncoderDecoderCache` from the same decoder config, sliding `layer_types` and all,
        # and each model that cares corrects it in its own `_prepare_cache_for_generation` (t5gemma). The
        # runtime has no model to inherit that from, and the layer *classes* are part of the graph's input
        # spec, so rebuild the cross half the way the traced cache had it.
        cache = model_kwargs.get("past_key_values")
        if isinstance(cache, EncoderDecoderCache):
            cross = cache.cross_attention_cache
            if any(type(layer).__name__.endswith("SlidingWindowLayer") for layer in cross.layers):
                cross_config = copy.deepcopy(self.config.get_text_config(decoder=True))
                cross_config.sliding_window = None
                cross_config.layer_types = ["full_attention"] * cross_config.num_hidden_layers
                cross_kwargs = {"config": cross_config}
                if isinstance(cross, StaticCache):
                    # A static cross cache is sized to the encoder sequence, not the decode length.
                    cross_kwargs["max_cache_len"] = model_kwargs["encoder_outputs"][0].shape[1]
                cache.cross_attention_cache = type(cross)(**cross_kwargs)

        for cache_name in ("past_key_values", "cache_params"):
            if (cache := model_kwargs.get(cache_name)) is not None:
                # The traced prototype is a cache the model filled, so it carries the real per-layer
                # geometry — the config can't always say (see `materialize_cache_layers`).
                materialize_cache_layers(
                    cache,
                    batch_size,
                    self.config,
                    self._dtype,
                    self._device,
                    kv_geometry=getattr(self._decode_runner, "kv_geometry", None),
                )

    # ── decode orchestration ──
    def create_masks_for_generate(self, config, inputs_embeds, attention_mask, **kwargs):
        """Keep the 2D padding mask when that is what the decode graph took.

        `prepare_inputs_for_generation` upgrades a 2D mask to the 4D causal mask for any compileable cache,
        which is right for a model whose forward builds its own mask but wrong for a graph traced *on* the 2D
        mask — the 4D one then fails an internal guard rather than a shape check. The trace is the authority
        (`_mask_rank`), so defer to it and only fall back to the generic builder otherwise.
        """
        from ..masking_utils import create_masks_for_generate

        if _mask_rank(self._decode_runner) == 2 and attention_mask is not None:
            return attention_mask
        return create_masks_for_generate(
            config=config, inputs_embeds=inputs_embeds, attention_mask=attention_mask, **kwargs
        )

    def _mask_feed(self, runner, attention_mask, position_ids, cache_len):
        """Feed the decode graph's mask input(s). `generate` hands us either a dict of 4D bool masks (one
        per attention type, for mixed full/sliding models) or a single 4D mask; when it drops a mask as
        redundant (`None` — the whole kwarg or one dict slot) the traced graph still takes a tensor there,
        so rebuild the full causal mask the eager model would have built internally."""
        # A mixed-attention model (nemotron_h, jamba, …) builds its per-layer-type mask dict *inside* its
        # forward, which the graph starts after — so when the graph takes one mask per type and `generate`
        # handed us a single tensor, build the dict here, keyed the way the config declares its layers.
        mask_ranks = runner.mask_dict_ranks
        if mask_ranks and not isinstance(attention_mask, dict):
            padding_mask = attention_mask if getattr(attention_mask, "dim", lambda: 0)() == 2 else None
            attention_mask = {
                layer_type: padding_mask if rank == 2 else None for layer_type, rank in mask_ranks.items()
            }
        if isinstance(attention_mask, dict):
            attention_mask = {
                layer_type: mask if mask is not None else self._causal_mask(position_ids, cache_len)
                for layer_type, mask in attention_mask.items()
            }
            # Mixed full/sliding models: ONNX declares one input per attention type
            # (`attention_mask.<type>`, flattened by the exporter); dynamo takes the whole dict as a single
            # `attention_mask` pytree kwarg.
            # ONNX flattens the dict into one input per type. Feed exactly the names it declares: keying
            # off our own layer types instead offers ones the graph never took and omits ones it needs.
            if declared := [name for name in _mask_inputs(runner) if name != "attention_mask"]:
                fallback = None
                feed = {}
                for name in declared:
                    mask = attention_mask.get(_mask_type(name))
                    if mask is None:
                        fallback = fallback if fallback is not None else self._causal_mask(position_ids, cache_len)
                        mask = fallback
                    feed[name] = mask
                return feed
            return {"attention_mask": attention_mask}
        # A graph may take no explicit mask — e.g. a prefill graph builds the causal mask internally from
        # positions on an empty, unpadded cache. Nothing to feed then.
        if not _mask_inputs(runner):
            return {}
        if attention_mask is None:
            return {_mask_inputs(runner)[0]: self._causal_mask(position_ids, cache_len)}
        # A graph traced on the 2D padding mask was traced against a *cache-width* mask: `generate` pads the
        # mask out to a static cache's length, and the model compares the two (bloom's alibi). The runtime's
        # mask tracks the real sequence instead, so pad the tail back — zeros, i.e. the unfilled cache slots
        # masked out, which is what the padded mask meant at capture.
        # Only when this *is* the decoder's own mask. An encoder-decoder's `attention_mask` covers the
        # *encoder* sequence and its width is tied to the encoder output's, not to the decoder cache — and it
        # stays under that name whether or not the graph also takes a `decoder_attention_mask`, so the config
        # is what settles it rather than the presence of the decoder mask input.
        pads_to_cache = _mask_rank(runner) == 2 and not self.config.is_encoder_decoder
        if pads_to_cache and attention_mask.dim() == 2:
            if (padding_length := cache_len - attention_mask.shape[-1]) > 0:
                attention_mask = torch.nn.functional.pad(attention_mask, (0, padding_length))
        return {_mask_inputs(runner)[0]: attention_mask}

    def _causal_mask(self, position_ids, cache_len):
        """Full causal mask `[batch, 1, query, cache_len]` from the positions (per batch row) — what the
        eager model builds internally when `generate` drops the mask as redundant. M-RoPE positions carry
        extra axes in front; the text row (axis 0) is the sequence position. Assumes no left-padding — the
        common single-sequence case."""
        text_positions = position_ids if position_ids.dim() == 2 else position_ids[0]
        positions = torch.arange(cache_len, device=text_positions.device)
        return (positions <= text_positions[..., None])[:, None]

    def _text_feed(self, runner, text_ids, kwargs, image_sizes=None) -> dict:
        """What goes in under the decode graph's text input — the token ids themselves.

        The seam a multi-modal runtime replaces: there the ids go through an embedding graph first and each
        modality's features are scattered into the result (`ExportedMultimodalGenerator`).
        """
        return {_text_input(runner): text_ids}

    def forward(
        self,
        past_key_values=None,
        input_ids=None,
        decoder_input_ids=None,
        position_ids=None,
        attention_mask=None,
        encoder_outputs=None,
        cache_params=None,
        image_sizes=None,
        **kwargs,
    ):
        # First step (empty cache) is the prefill; subsequent steps are decode. With no dedicated prefill
        # graph the two runners are the same object, so this just always runs `decode`. For an
        # encoder-decoder model the split is load-bearing beyond shapes: the prefill graph *writes* the
        # cross cache from `encoder_outputs`, decode graphs read it (the modeling's `is_updated` python
        # branch bakes at trace time). With caching off (`use_cache=False`) there is no cache at all and
        # every step re-feeds the whole sequence — that is a prefill each time.
        # A recurrent model (mamba, rwkv, …) carries its fixed-size state under `cache_params` instead;
        # it is the same thing to the graphs, so the loop below treats them alike.
        cache_name = "cache_params" if cache_params is not None else "past_key_values"
        past_key_values = past_key_values if past_key_values is not None else cache_params
        if past_key_values is None:
            runner = self._prefill_runner
        else:
            runner = self._prefill_runner if _cache_length(past_key_values) == 0 else self._decode_runner
        text_ids = decoder_input_ids if decoder_input_ids is not None else input_ids
        feed = self._text_feed(runner, text_ids, kwargs, image_sizes)
        text = feed[_text_input(runner)]
        if position_ids is not None and "position_ids" in runner.input_names:
            feed["position_ids"] = position_ids
        if encoder_outputs is not None and any(n.startswith("encoder_outputs") for n in runner.input_names):
            feed["encoder_outputs"] = encoder_outputs
        # Extra per-step inputs the graph declares (`token_type_ids`, per-model aux masks, …) come from
        # `generate`'s kwargs. `generate` slices sequence kwargs to the current step only for the ones named
        # on the real model's `forward` — ours takes them generically, so trim them here the same way.
        for name, value in kwargs.items():
            if name in runner.input_names and name not in feed:
                # Rank 2 or 3 only — those have the sequence at axis 1 (higgs_audio_v2's audio ids carry a
                # codebook axis after it). A 4-D mask is `[batch, 1, query, key]`, where axis 1 is heads.
                # A modality's own tensors (`pixel_values`, packed patches, …) are features, not per-token
                # kwargs — their axis 1 is patches or channels, so they go through whole (a prefill graph
                # with the vision tower inline takes them directly).
                is_per_token = isinstance(value, torch.Tensor) and name not in getattr(self, "_modality_keys", ())
                if is_per_token and value.dim() in (2, 3) and value.shape[1] > text.shape[1]:
                    value = value[:, -text.shape[1] :]
                feed[name] = value
        cache_len = (
            past_key_values.get_max_length()
            if past_key_values is not None and hasattr(past_key_values, "get_max_length")
            else text.shape[1]
        )
        feed.update(self._mask_feed(runner, attention_mask, position_ids, cache_len))
        # A graph that names the decoder's mask separately gets the causal one here — `attention_mask` is
        # the *encoder's* on those models. `generate` supplies neither this nor decoder positions (the
        # eager forward derives both inside), so count the positions off the cache.
        if _decoder_mask_input(runner) is not None and _decoder_mask_input(runner) not in feed:
            decoded = _cache_length(past_key_values)
            decoder_positions = (
                torch.arange(text.shape[1], device=self._device).unsqueeze(0).expand(text.shape[0], -1) + decoded
            )
            feed[_decoder_mask_input(runner)] = self._causal_mask(decoder_positions, cache_len)
        # Under the name this graph declares, and only if it declares one: a model whose `generate` hands
        # back a cache the exported graph does not take (xlstm) would otherwise be fed an input it never had.
        if past_key_values is not None and runner.cache_input is not None:
            feed[runner.cache_input] = past_key_values
        # Inputs of the graph's own beyond the text / mask / cache set: csm's decode takes the audio
        # (`input_values`, `input_values_cutoffs`) directly rather than through a modality component. Dynamo
        # wants the exact kwarg set it was traced with, so pass through whatever `generate` still carries
        # under a name this graph declares — never overwriting what the feed already resolved.

        # Only what this graph declares: a model whose decode graph takes its text some other way
        # (higgs_audio_v2 embeds it upstream) would otherwise be handed an `input_ids` it never had. Pytree
        # kwargs are the exception — see `_declares`, which knows a graph naming only their leaves still takes
        # them (the cache, `encoder_outputs`, a mask dict).
        outputs = runner(**{name: value for name, value in feed.items() if _declares(runner, name, value)})
        if past_key_values is not None:
            past_key_values = _advance_cache(past_key_values, outputs, num_new_tokens=text.shape[1])
        if cache_name == "cache_params":
            return CausalLMOutputWithPast(logits=outputs["logits"], past_key_values=None)
        return CausalLMOutputWithPast(logits=outputs["logits"], past_key_values=past_key_values)


class ExportedMultimodalGenerator(ExportedGenerator):
    """`ExportedGenerator` for a model whose prompt carries more than text.

    Adds exactly what a text-generation runtime has no use for: an embedding graph, one features graph per
    modality, the scatter that places those features at their placeholder rows, and the M-RoPE positions a
    layout-declaring config implies. Everything else — cache construction, masks, the decode loop — is the
    base class's, unchanged.
    """

    def __init__(
        self,
        config,
        generation_config,
        decode: ModelRunner,
        *,
        prefill: ModelRunner | None = None,
        encoder: ModelRunner | None = None,
        text_embed: ModelRunner | None = None,
        modalities: list[Modality] = (),
    ):
        super().__init__(config, generation_config, decode, prefill=prefill, encoder=encoder)
        self._text_embed = text_embed
        # An encoder-decoder scatters the features in front of its *text encoder* (florence2), not per
        # decode step — the encoder graph says so by taking `inputs_embeds` where a plain encoder-decoder's
        # takes `input_ids`.
        if encoder is not None and "inputs_embeds" in encoder.input_names:
            self._encoder = _ExportedEncoder(encoder, merge=self._merge_modalities)
        self._modalities = list(modalities)
        self._modality_keys = {key for modality in self._modalities for key in modality.input_keys}
        extra = [text_embed, *(modality.runner for modality in self._modalities)]
        self._graph_inputs |= {name for runner in extra if runner is not None for name in runner.input_names}

    @property
    def _consumed_kwargs(self) -> set[str]:
        """Also consumed here: each modality's own inputs, and `mm_token_type_ids` (the placeholder map the
        scatter reads, never a graph input of its own)."""
        return super()._consumed_kwargs | self._modality_keys | {"mm_token_type_ids"}

    def _text_feed(self, runner, text_ids, kwargs, image_sizes=None) -> dict:
        """Embed the ids, scatter each present modality's features in, and hand the result to the graph.

        The embed graph's first output is the embeddings themselves, whatever it named them; they go in under
        the decode graph's own text input. Any further per-token outputs (`per_layer_inputs`) are fed by name,
        and only if this graph declares them. `image_sizes` arrives as an explicit kwarg — no graph declares it
        once the anyres packing moved here, so `generate` would otherwise reject it — and goes only to the
        merge, since putting it in `kwargs` would expose an `(images, 2)` tensor to the per-step slicing.
        """
        # A decode graph that takes token ids directly (an encoder-decoder's, reading the merged features
        # through `encoder_outputs`) gets them as-is — the merge already happened at the encoder.
        if _text_input(runner) != "inputs_embeds":
            return super()._text_feed(runner, text_ids, kwargs, image_sizes)
        merge_kwargs = kwargs if image_sizes is None else {**kwargs, "image_sizes": image_sizes}
        embedded = self._merge_modalities(text_ids, merge_kwargs)
        primary, *extra = embedded
        feed = {_text_input(runner): embedded[primary]}
        feed.update({name: embedded[name] for name in extra if name in runner.input_names})
        return feed

    def _merge_modalities(self, input_ids, kwargs) -> dict[str, torch.Tensor]:
        """Embed `input_ids` and scatter each present modality's features into its placeholder rows.

        Returns every per-token input the embed graph produces, keyed by the decode graph's input name —
        `inputs_embeds` plus, for a decoder with per-layer embeddings, `per_layer_inputs`. Only the
        embeddings take the scattered features; the rest pass through as embedded.
        Presence keys on the primary input (`input_keys[0]`: pixel_values / input_features); `generate`
        drops it after prefill but may keep a stale grid kwarg, so don't trigger on the aux keys. Which
        *other* inputs a modality takes varies per model (`image_position_ids`, `input_features_mask`, …),
        so they come from the names its graph declares rather than a fixed list per modality. The eager
        `get_<modality>_features` computes its grid-derived tensors (`cu_seqlens` / `window_index` / …)
        internally; the export moved them out of the graph, so they're injected here from `self.config`
        alone (`precompute_export_inputs`), after renaming generate's grid kwarg (`image_grid_thw` /
        `video_grid_thw`) to the graph's `grid_thw` input."""

        from .utils import _find_config_attr, flatten_anyres_patches

        extras: dict = {}

        embedded = self._text_embed(input_ids=input_ids)
        embeds_name = next(iter(embedded))
        inputs_embeds = embedded[embeds_name]
        for modality in self._modalities:
            # Presence keys on whichever of the modality's own input names this call carries — never the
            # aux keys, which `generate` may keep after dropping the features themselves.
            aux_suffixes = ("_grid_thw", "_position_mask", "_attention_mask", "_sizes")
            if all(kwargs.get(key) is None for key in modality.input_keys if not key.endswith(aux_suffixes)):
                continue
            if modality.token_id is not None:
                mask = (input_ids == modality.token_id).unsqueeze(-1)
            else:
                # A model with no placeholder token id (kosmos2_5) marks the feature rows with an explicit
                # mask kwarg instead — the same tensor its own forward scatters by. `generate` keeps the
                # full-prompt mask across steps, so align it to this step's ids (its tail) first.
                mask_key = next(key for key in modality.input_keys if key.endswith("_position_mask"))
                mask = (kwargs[mask_key][:, -input_ids.shape[1] :] == 1).unsqueeze(-1)
            # A step with no placeholder rows has nothing to scatter into — a decode step whose feature
            # kwarg `generate` kept around would otherwise re-encode for nothing.
            if not mask.any():
                continue
            feed = self._modality_feed(modality, kwargs, input_ids)
            # An anyres image graph stops at the projector (`PatchVisionEncoder`) because the packing's token
            # count per image is data; so the padding rows come off here and the packing happens here too.
            image_sizes = kwargs.get("image_sizes")
            packs_anyres = (
                image_sizes is not None
                and modality.input_keys[0] == "pixel_values"
                and "image_sizes" not in modality.runner.input_names
                and _find_config_attr(self.config, "image_grid_pinpoints") is not None
            )
            if packs_anyres:
                tower_input = modality.runner.input_names[0]
                feed[tower_input] = flatten_anyres_patches(self.config, feed[tower_input], image_sizes)
            outputs = modality.runner(**feed)
            # A deepstack tower emits one feature tensor per decoder layer it is injected at
            # (`image_features.<layer>`). Those are summed in inside the decoder rather than scattered, so the
            # placeholder rows are zeroed here and the packed tensors go on to the decode graph by name.
            per_layer = {
                int(name.rsplit(".", 1)[-1]): tensor
                for name, tensor in outputs.items()
                if re.fullmatch(r".*image_features\.\d+", name)
            }
            if packs_anyres and per_layer:
                extras["deepstack_features"] = {
                    layer: _pack_anyres_features(self.config, tensor, image_sizes, outputs).to(inputs_embeds.dtype)
                    for layer, tensor in sorted(per_layer.items())
                }
                extras["vision_mask"] = mask
                inputs_embeds = inputs_embeds.masked_fill(mask, 0.0)
                continue
            features = next(iter(outputs.values()))
            if packs_anyres:
                features = _pack_anyres_features(self.config, features, image_sizes, outputs)
            inputs_embeds = inputs_embeds.masked_scatter(mask, features.to(inputs_embeds.dtype))
        return {**embedded, embeds_name: inputs_embeds, **extras}

    def _modality_feed(self, modality, kwargs, input_ids) -> dict:
        """Everything one modality graph declares, sourced in order: `generate`'s kwargs, then the prompt's
        ids if it takes them, then the tensors the precompute derives from the config, then the config
        itself.

        Those three cover the three kinds of input such a graph takes — the modality's own data
        (`pixel_values`, and any aux the model names beside it), the grid-derived tensors the export moved
        out of the graph (`cu_seqlens` / `window_index` / …), and plain settings the eager forward defaults
        from the config (`vision_feature_layer`). Anything the graph does not declare is left out, and
        `generate`'s text kwargs are never a source: an encoder's `attention_mask` is its own. The prompt's
        `input_ids` are the exception, fed only to a graph that names them — some getters read them to place
        their features in time (musicflamingo's rotary timestamps) — and a modality only ever runs on the
        prefill, which is where the capture read them too.

        Sourcing by name alone would cross the modalities over. `generate` names the features per modality
        (`pixel_values_videos`), but a graph takes the name its own getter declared — and a video getter
        often declares the generic `pixel_values` (llava_next_video), which is the *image* kwarg. So route
        this modality's own key onto its graph's feature input, the way `_grid_renamed` routes the grid, and
        drop the keys that belong to another modality.
        """
        from .utils import _find_config_attr, cast_leaf_tensors, precompute_export_inputs

        declared = set(modality.runner.input_names)
        feature_input = modality.runner.input_names[0]
        present = next(
            (
                key
                for key in modality.input_keys
                if not key.endswith(("_grid_thw", "_position_mask", "_attention_mask", "_sizes"))
                and key not in declared
                and kwargs.get(key) is not None
            ),
            None,
        )
        foreign = {key for other in self._modalities if other is not modality for key in other.input_keys} - set(
            modality.input_keys
        )
        inputs = {}
        for key, value in kwargs.items():
            if value is None or key in _TEXT_KWARGS or key in foreign:
                continue
            if (name := _grid_renamed(key)) in declared:
                inputs[name] = value
        # Only when nothing named the graph's feature input: the modality's kwarg is that tensor under
        # another name (`pixel_values_videos` for a video getter that declares `pixel_values`). A getter
        # whose first input is a *different* quantity (inkling's `audio_input_ids`, vibevoice_asr's
        # `input_values`) is fed by name above, so this must not overwrite it.
        if present is not None and feature_input not in inputs:
            inputs[feature_input] = kwargs[present]
        # Cast BEFORE the precompute: the graph was traced with the model's dtype for the modality inputs
        # (`pixel_values` …) but with whatever dtype the precompute itself produces (fp32 interpolation
        # weights), so casting after would hand the graph bf16 weights it never saw.
        inputs = cast_leaf_tensors(inputs, dtype=modality.runner.dtype, device=self._device)
        inputs = precompute_export_inputs(self.config, inputs)
        # Device-only move for what the precompute added — it builds some index tensors on CPU
        # (minicpmv4_6's `window_index` comes out of python list ops). No dtype cast: the graph was traced
        # with whatever dtype the precompute produces (see above).
        inputs = {
            name: value.to(self._device) if isinstance(value, torch.Tensor) else value
            for name, value in inputs.items()
        }
        feed = {name: value for name, value in inputs.items() if name in declared}
        if "input_ids" in declared:
            feed["input_ids"] = input_ids
        for name in declared - feed.keys():
            if (value := _find_config_attr(self.config, name)) is not None:
                feed[name] = value
        return feed

    def _prepare_position_ids_for_generation(self, inputs_tensor, model_kwargs):
        """Multi-modal M-RoPE: build the `[text; 3 vision]` 4-axis `position_ids` the exported decode graph
        expects — what `generate` normally gets from a VLM's own override of this method. Mirrors that
        override from the config alone: the text row from `super()` (GenerationMixin), the 3 vision rows
        from `modeling_rope_utils.get_mrope_index` — the very call the model's own override makes — and the
        decode step advances the text row by the cached rope-delta. Models that don't declare M-RoPE (plain
        decoders, VLMs with 1D text positions like Llava) keep the standard positions."""
        from ..modeling_multimodal_utils import get_mrope_index, uses_mrope

        text_positions = super()._prepare_position_ids_for_generation(inputs_tensor, model_kwargs)
        # Both signals, not just `uses_mrope`: a text config carries `mrope_section` while only the outer
        # multimodal config declares the `mrope_layout` that says how to lay the spans out (minicpmv4_6 has
        # the first and not the second, and `get_mrope_index` refuses a `None` layout).
        if not uses_mrope(self.config) or getattr(self.config, "mrope_layout", None) is None:
            return text_positions

        cache = model_kwargs.get("past_key_values")
        past_length = _cache_length(cache)
        if past_length != 0 and getattr(self, "_rope_deltas", None) is not None:
            return text_positions[None, ...] + self._rope_deltas

        if model_kwargs.get("input_ids") is not None and model_kwargs["input_ids"].shape[1] > 0:
            inputs_tensor = model_kwargs["input_ids"]
        # No `attention_mask`: unpadded single sequence, and `generate` has already turned the mask into the
        # per-layer form the attention needs, not the 2D form `get_rope_index` wants. `None` = all valid.
        # An audio-carrying layout (the omni thinkers) places its spans from the mel lengths, which
        # `generate` carries as the padding mask — derive them the way the eager forward does.
        audio_seqlens = model_kwargs.get("audio_feature_lengths")
        if audio_seqlens is None and model_kwargs.get("feature_attention_mask") is not None:
            audio_seqlens = model_kwargs["feature_attention_mask"].sum(-1)
        vision_positions, self._rope_deltas = get_mrope_index(
            self.config,
            inputs_tensor,
            # Optional in `get_mrope_index`, and `generate` only carries it for the models whose layout
            # places spans by token type — minicpmv4_6's does not.
            model_kwargs.get("mm_token_type_ids"),
            image_grid_thw=model_kwargs.get("image_grid_thw"),
            video_grid_thw=model_kwargs.get("video_grid_thw"),
            second_per_grid_ts=model_kwargs.get("second_per_grid_ts"),
            audio_seqlens=audio_seqlens,
        )
        return torch.cat([text_positions[None, ...], vision_positions], dim=0)
