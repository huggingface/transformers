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

import torch
from torch.utils._pytree import tree_flatten, tree_leaves, tree_unflatten

from ..cache_utils import DynamicCache
from ..utils import logging
from .utils import _resolve_modeling_module, materialize_cache_layers


logger = logging.get_logger(__name__)


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
    target = cache
    for part in path[:-1]:
        target = _read_cache_step(target, part)
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
