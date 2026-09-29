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
"""Reading and advancing the caches an exported graph takes and returns.

An exported decode graph takes its cache as flat tensors and hands back the updated ones, so driving it
means finding a `Cache`'s leaves, writing the graph's outputs back into them, and knowing how far it has
filled. Both the runners (which flatten a cache into a feed) and the generation loop above them need that,
so it lives here rather than in either.
"""

from __future__ import annotations

from typing import Any

from ..utils import logging
from ..utils.import_utils import is_torch_available
from .precompute import _resolve_modeling_module


logger = logging.get_logger(__name__)


if is_torch_available():
    import torch
    from torch.utils._pytree import tree_flatten, tree_leaves, tree_unflatten

    from ..cache_utils import DynamicCache, kv_cache_geometry


def _cache_tensors(past_key_values) -> list[torch.Tensor]:
    """The cache's tensor leaves, in the pytree order the exporter named them."""
    return [t for t in tree_leaves(past_key_values) if isinstance(t, torch.Tensor)]


def _read_cache_step(container, part: str):
    """One step of a cache leaf path: an index into a list, a key into a dict (a layer may keep its state
    dict-keyed by entry name — deepseek_v4's `buffer_kv["compressor"]`), or an attribute."""
    if part.isdigit():
        return container[int(part)]
    if isinstance(container, dict):
        return container.get(part)
    return getattr(container, part, None)


def _read_cache_entry(cache, path: list[str]):
    """The cache's entry at a named leaf path (`layers.0.conv_states.0`), or `None` where the path does
    not (yet) lead anywhere — a recurrent layer's states are `None` until a step produces them."""
    target = cache
    for part in path:
        if target is None:
            return None
        target = _read_cache_step(target, part)
    return target


def _assign_cache_entry(cache, path: list[str], value) -> None:
    """Write a decode step's cache output back at its named path (`layers.0.conv_states.0`).

    Keeps a fixed-size buffer in place (`copy_`, so the cache object and any graph that mutated it stay
    valid) and replaces the entry outright when it grew or did not exist yet — a growing `DynamicCache`
    returns longer tensors, and a recurrent layer's states start as `None`.
    """
    target = _read_cache_entry(cache, path[:-1])
    last = path[-1]
    current = _read_cache_step(target, last)
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


def _cache_halves(cache: Any) -> list[Any]:
    """The caches to walk. An `EncoderDecoderCache` keeps its layers in the two caches it pairs rather than
    on itself, so both halves are walked and the self-attention one comes first; anything else is one half,
    itself. The single place this structure is read — everything else asks here."""
    if cache is None:
        return []
    if (self_attention := getattr(cache, "self_attention_cache", None)) is not None:
        return [self_attention, cache.cross_attention_cache]
    return [cache]


def _self_attention_layers(cache) -> list:
    """A cache's self-attention layers — the half every question here is about (how long, what geometry,
    does it keep keys, which states exist)."""
    halves = _cache_halves(cache)
    return getattr(halves[0], "layers", []) if halves else []


def keeps_write_once_state(layer) -> bool:
    """Whether this layer keeps state besides its keys and values — a conv window, an SSM state, a sparse
    indexer's keys.

    The prompt's forward *creates* that state; later steps only advance it, so a graph traced on a decode
    step holds the update and not the creation. Asked by name because each family spells it differently
    (`conv_states`, `recurrent_states`, `idx_keys`, `indexer_keys`) with no common base to ask instead:
    anything ending in `_states`, or a `*_keys` that is not the layer's own `keys`. A counter
    (`cumulative_length`) is not state in this sense — every step writes it.
    """
    return any(name.endswith("_states") or (name.endswith("_keys") and name != "keys") for name in vars(layer))


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


def resize_to_traced_lengths(cache, lengths: dict[int, int]) -> None:
    """Size each fixed-size layer to the length the trace recorded for it, before its buffers are made.

    `generate` sizes a fixed cache from the prompt in front of it and from per-model facts — mllama's
    cross-attention layers hold the vision sequence, not the generation length — so re-deriving a size here
    gives a different cache for a different prompt. A graph carries its cache's sizes in the input spec and
    refuses to be called against any other, so the trace's own account is the one that fits.
    """
    if not lengths:
        return
    layers = _self_attention_layers(cache)
    for index, length in lengths.items():
        if index < len(layers) and getattr(layers[index], "max_cache_len", None) not in (None, length):
            layers[index].max_cache_len = length


def is_fixed_size(cache) -> bool:
    """Whether `cache` is allocated at its full length up front (`StaticCache`) rather than growing.

    Read off the first self-attention layer, the one `mask_width` asks: a fixed-size layer is the kind
    that compiles (`is_compileable`), whatever it currently holds.
    """
    layers = _self_attention_layers(cache) if cache is not None else []
    return bool(layers) and getattr(layers[0], "is_compileable", False)


def mask_width(cache, query_length: int) -> int:
    """How wide a causal mask over `cache` has to be — what the cache itself reports (`get_mask_sizes`),
    which is the same question `create_causal_mask` puts to it in an eager forward.

    Per *layer*, because one cache's layers need not agree on a length: mllama sizes its cross-attention
    layers to the vision sequence and its self-attention ones to the generation length, and the text mask
    belongs to the latter. Layer 0 is the one asked, the way the eager mask builder asks for the first layer
    of the type it is building for — a graph traced that way guards on exactly that layer's width.
    `get_max_length()` answers for the *longest* layer instead, which on such a model is the vision one.
    """
    if cache is None:
        return query_length
    try:
        return int(cache.get_mask_sizes(query_length, 0)[0])
    except (AttributeError, IndexError, TypeError, ValueError, StopIteration):
        # A cache that keeps no per-layer attention state to ask (recurrent-only) answers by what it holds.
        return _cache_length(cache) + query_length


def _advance_cache(past_key_values, outputs: dict[str, torch.Tensor], num_new_tokens: int):
    """Advance the cache with a decode step's outputs (the `past_key_values.…` entries, in cache-leaf
    order). A fixed-size cache keeps its shapes, so `copy_` in place — preserving the cache object and its
    non-tensor state; a graph that already mutated the cache in place returns the same tensors, making the
    copy a no-op. A growing `DynamicCache` returns longer tensors (its seq axis grew), so rebuild the cache
    from the grown leaves through the registered cache pytree. The tensors themselves tell the two apart.
    The layers' length counters are then advanced (`advance_cache_length`)."""
    # a recurrent model's graph names its cache outputs after its own kwarg (`cache_params.…`)
    cache_updates = [
        (name, value) for name, value in outputs.items() if name.startswith(("past_key_values", "cache_params"))
    ]
    # A dotted name is the leaf's path in the cache (`layers.0.conv_states.0`), the only alignment that
    # holds when the graph returns entries the cache has no leaf for: a recurrent layer keeps its states
    # `None` until a step produces them, so counting leaves would run off the end.
    # The tensors this step wrote, so a counter the graph already advanced is not advanced again.
    written = set()
    for name, new in cache_updates:
        if "." in name:
            path = name.split(".")[1:]
            _assign_cache_entry(past_key_values, path, new)
            written.add(id(_read_cache_entry(past_key_values, path)))
    by_index = [(name, new) for name, new in cache_updates if "." not in name]
    if by_index:
        # ExecuTorch names its cache inputs by flat leaf index (`past_key_values_<N>`) and may prune the
        # placeholders its lowering left unused, so index by the suffix and keep the old leaf where no
        # update came back. Its runtime also returns rank-0 updates as python scalars — re-wrap them.
        cache_leaves = _cache_tensors(past_key_values)
        updated = list(cache_leaves)
        for position, (name, new) in enumerate(by_index):
            suffix = name.rsplit("_", 1)[-1]
            index = int(suffix) if suffix.isdigit() else position
            if not isinstance(new, torch.Tensor):
                new = torch.tensor(new, dtype=cache_leaves[index].dtype, device=cache_leaves[index].device)
            updated[index] = new
            written.update((id(cache_leaves[index]), id(new)))
        # A growing cache came back longer than it went in, so it is rebuilt through the registered pytree;
        # a fixed-size one kept its shapes and is copied in place, preserving the object and its non-tensor
        # state (and a graph that mutated it in place hands back the same tensors, making the copy a no-op).
        if any(old.shape != new.shape for old, new in zip(cache_leaves, updated)):
            _, spec = tree_flatten(past_key_values)
            past_key_values = tree_unflatten(updated, spec)
        else:
            for old, new in zip(cache_leaves, updated):
                if old is not new:
                    old.copy_(new)
    _mark_existing_states(past_key_values)
    advance_cache_length(past_key_values, num_new_tokens, written)
    # An `EncoderDecoderCache` also keeps `is_updated` python flags (pytree context, so the graph never
    # flips them and the growing rebuild resurrects the pre-step values): every decoder step leaves the
    # cross cache written — the prefill graph writes it, decode graphs read it.
    if getattr(past_key_values, "is_updated", None):
        past_key_values.is_updated = dict.fromkeys(past_key_values.is_updated, True)
    return past_key_values


def advance_cache_length(past_key_values, num_new_tokens: int, written=()) -> None:
    """Advance each self-attention layer's length counter by `num_new_tokens`, the way the eager `update` does.

    Sliding layers count in a plain python int (`cumulative_length_int` on static ones, `cumulative_length`
    on growing ones) that no graph updates: the static graph bakes it and the growing rebuild resurrects the
    pre-step value from the pytree context. It is advanced here rather than read from the static layer's
    tensor, which is a device→host sync and stops advancing once the window is full.

    A fixed-size layer counts in a tensor, and a graph that folded its cache into runtime state hands no
    counter back — so the count stays where it started and `get_seq_length()` answers 0 for a cache that is
    plainly full, which is also where `generate` anchors its 4D mask. A tensor counter whose `id` is in
    `written` came back from the graph already advanced, and is left alone."""
    for layer in _self_attention_layers(past_key_values):
        counter = getattr(layer, "cumulative_length", None)
        if hasattr(layer, "cumulative_length_int"):
            layer.cumulative_length_int += num_new_tokens
        elif isinstance(counter, int):
            layer.cumulative_length += num_new_tokens
        elif torch.is_tensor(counter) and id(counter) not in written:
            # In place, so a graph reading this buffer keeps reading the same one.
            counter.add_(num_new_tokens)


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


def _empty_container(container: str, config, batch_size: int, dtype, device, encoder_config=None):
    """A container shaped the way the trace saw it, holding nothing yet.

    `"cache"` is one of the generic `Cache` classes: built from `encoder_config` so its layers are the kinds
    that sub-model caches with (the audio tower's windowed layers, `sliding_window` and all) and materialized
    to zero length, which is the state the trace recorded — lazily-uninitialized layers would flatten to a
    shorter pytree than the graph declares. Anything else is a model's own container class
    (voxtral_realtime's conv-state `VoxtralRealtimeConv1dPaddingCache`), reached by name in its `modeling_*`
    module the way the precompute reaches a model's own helpers — from the config alone, no model instance."""
    if container != "cache":
        module = _resolve_modeling_module(config)
        container_class = getattr(module, container, None) if module is not None else None
        return container_class() if container_class is not None else None
    if encoder_config is None:
        return DynamicCache()
    cache = DynamicCache(config=encoder_config)
    materialize_cache_layers(cache, batch_size, encoder_config, dtype, device)
    return cache


# ── Geometry and materialization ──────────────────────────────────────────────
# What a cache's layers are shaped like, and giving them real tensors before a trace: `torch.export` cannot
# trace lazy allocation, so the traced cache and the one the runtime builds must be materialized the same way.


def check_cache_geometry(config: Any, cache: Any) -> None:
    """Raise if the geometry `kv_cache_geometry` derives disagrees with what the model really cached.

    Called on a post-prefill cache, whose layers hold real tensors. What a model caches is not always what
    its config reads like — latent attention is the standing example — so this turns the mismatch into a
    message naming what was cached, instead of an `index_copy_()` shape error deep in a later forward.

    Raises only when the derivation matches *no* layer. A model may cache different geometries across
    layers (deepseek_v32's sparse-indexer layers next to its latent ones), and the exporter fills only the
    layers that reach it uninitialized — so a layer disagreeing is normal, and none agreeing is the
    failure: whatever the exporter would have materialized fits nothing the model actually writes.
    """
    cached, derived_any = [], None
    by_layer = kv_cache_geometry(config) or []
    for cache_half in _cache_halves(cache):
        for layer_idx, layer in enumerate(cache_half.layers):
            if getattr(layer, "keys", None) is None or layer_idx >= len(by_layer):
                continue
            derived = by_layer[layer_idx]
            actual = (layer.keys.shape[1], layer.keys.shape[3], layer.values.shape[3])
            if actual == derived:
                return
            cached.append(actual)
            derived_any = derived
    if cached:
        model_type = getattr(config.get_text_config(), "model_type", type(config).__name__)
        raise ValueError(
            f"`{model_type}` caches (heads, key_dim, value_dim)={sorted(set(cached))} but the exporter "
            f"derives {derived_any} for every layer, so the runtime would build a cache the exported "
            "graph rejects. `kv_cache_geometry` needs to learn this model's layout."
        )


def kv_geometry_of(cache: Any) -> dict[int, tuple[int, int, int]]:
    """`{layer index: (num_kv_heads, key_head_dim, value_head_dim)}` of a cache the model itself filled."""
    return {
        index: (layer.keys.shape[1], layer.keys.shape[3], layer.values.shape[3])
        for index, layer in enumerate(getattr(cache, "layers", []) or [])
        if getattr(layer, "keys", None) is not None and layer.keys.dim() == 4
    }


def indexer_layers_of(cache: Any) -> dict[int, bool]:
    """`{layer index: whether the layer holds an indexer tensor}` for a cache the model itself filled.

    The runtime's counterpart is `ExportMetadata.indexer_layers`, read off the graph. Both answer the same
    question — which sparse-indexer slots this model actually writes — so a prefill cache materialized from
    this matches the decode graph the model filled by hand.
    """
    return {
        index: bool(getattr(layer, "is_indexer_initialized", False))
        for index, layer in enumerate(getattr(cache, "layers", []) or [])
        if hasattr(layer, "is_indexer_initialized")
    }


def materialize_cache_layers(
    cache: Any,
    batch_size: int,
    config: Any,
    dtype: Any,
    device: Any,
    kv_geometry: dict[int, tuple[int, int, int]] | None = None,
    indexer_layers: dict[int, bool] | None = None,
) -> None:
    """Give every lazily-uninitialized cache layer real tensors — `torch.export` can't trace lazy
    allocation, so both the traced (prefill) cache and the cache the runtime builds must be materialized,
    and identically (dynamo bakes the cache pytree into the graph's input spec). Each layer does it through
    its own `lazy_initialization`, from a `[batch, kv_heads, 0, head_dim]` hint: a static layer allocates
    its full buffer from it, a growing one keeps it as the zero-length tensor to `cat` onto.

    `kv_geometry` is the per-layer geometry read off the graph (`ModelRunner.kv_geometry`) or off a cache the
    model filled (`kv_geometry_of`). It wins where present, because a config cannot always give it —
    mimo_v2_flash caches 2 KV heads on its sliding layers and 4 on its full ones — and the config derivation
    covers the layers it does not reach. `indexer_layers` is the same kind of fact for the sparse-indexer
    slot (`ExportMetadata.indexer_layers`): a layer recorded without one is left alone, because the graph
    took no leaf for it.
    """
    kv_geometry = kv_geometry or {}
    for cache_half in _cache_halves(cache):
        # A model-specific cache may keep its state in fields of its own rather than a `layers` list
        # (xLSTM's `rnn_state`); there is nothing layer-shaped to fill in that case.
        if not hasattr(cache_half, "layers"):
            continue
        # `Cache.early_initialization` is the API's own answer to "export needs everything in advance". Let
        # it do the layers, and let a cache with state of its own size that state by overriding it
        # (`MiniMaxCache.linear_cache`).
        by_layer = kv_cache_geometry(config) or []
        geometry = [
            kv_geometry.get(index) or (by_layer[index] if index < len(by_layer) else None)
            for index in range(len(cache_half.layers))
        ]
        if all(entry is not None for entry in geometry) and geometry:
            cache_half.early_initialization(
                batch_size,
                [entry[0] for entry in geometry],
                [entry[1] for entry in geometry],
                dtype,
                device,
                value_head_dim=[entry[2] for entry in geometry],
            )
        _materialize_layers(cache_half, batch_size, config, dtype, device, kv_geometry, indexer_layers)


def _materialize_layers(cache, batch_size, config, dtype, device, kv_geometry, indexer_layers) -> None:
    """`materialize_cache_layers` for one flat cache — see there."""
    by_layer = kv_cache_geometry(config) or []
    for layer_idx, layer in enumerate(cache.layers):
        # A sparse-indexer layer (deepseek_v32, axk2, glm_moe_dsa) caches a third tensor beside keys and
        # values; leave it lazy and every later cache leaf shifts by one. It cannot wait behind the
        # `is_initialized` skip below, which `early_initialization` sets while the indexer is untouched.
        # Unless the trace says this layer had none — hy_v4's shared indexer layers write no tensor of
        # their own, and filling one in would add a leaf the graph never took.
        traced_indexer = indexer_layers.get(layer_idx, True) if indexer_layers else True
        if traced_indexer and hasattr(layer, "is_indexer_initialized") and not layer.is_indexer_initialized:
            index_head_dim = config.get_text_config().index_head_dim
            empty_indexer_keys = torch.zeros(batch_size, 0, index_head_dim, dtype=dtype, device=device)
            layer.lazy_initialization_indexer(empty_indexer_keys)
        if getattr(layer, "is_initialized", True):
            continue
        geometry = kv_geometry.get(layer_idx) or (by_layer[layer_idx] if layer_idx < len(by_layer) else None)
        if geometry is None:
            # This layer keeps no keys and values to size — a recurrent layer's conv / SSM buffers are its
            # own business, and its neighbours still need theirs (`continue`, like every other skip here).
            continue
        num_kv_heads, key_dim, value_dim = geometry
        # A static layer also *records* its head count, and that record is compared as part of the graph's
        # input spec — so it has to come from the same place the buffers do, not from the config's single
        # value (mimo_v2_flash caches 2 heads on sliding layers and 4 on full ones).
        if hasattr(layer, "num_heads"):
            layer.num_heads = num_kv_heads
        empty_keys = torch.zeros(batch_size, num_kv_heads, 0, key_dim, dtype=dtype, device=device)
        empty_values = torch.zeros(batch_size, num_kv_heads, 0, value_dim, dtype=dtype, device=device)
        layer.lazy_initialization(empty_keys, empty_values)
