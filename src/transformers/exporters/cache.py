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

    from ..cache_utils import DynamicCache
    from ..configuration_utils import get_head_shapes


def _cache_tensors(past_key_values) -> list[torch.Tensor]:
    """The cache's tensor leaves, in the pytree order the exporter named them."""
    return [t for t in tree_leaves(past_key_values) if isinstance(t, torch.Tensor)]


def _read_cache_step(container, part: str):
    """One step of a cache leaf path: a list index, a dict key (deepseek_v4's `buffer_kv`), or an attribute."""
    if part.isdigit():
        return container[int(part)]
    if isinstance(container, dict):
        return container.get(part)
    return getattr(container, part, None)


def _read_cache_entry(cache, path: list[str]):
    """The cache's entry at a named leaf path (`layers.0.conv_states.0`), or `None` where it leads nowhere yet."""
    target = cache
    for part in path:
        if target is None:
            return None
        target = _read_cache_step(target, part)
    return target


def _assign_cache_entry(cache, path: list[str], value) -> None:
    """Write a decode step's cache output back at its named path (`layers.0.conv_states.0`).

    A same-shape buffer is updated in place (`copy_`); a grown or missing entry is replaced outright.
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
    """The caches to walk: an `EncoderDecoderCache`'s two halves (self-attention first), else the cache itself."""
    if cache is None:
        return []
    if (self_attention := getattr(cache, "self_attention_cache", None)) is not None:
        return [self_attention, cache.cross_attention_cache]
    return [cache]


def _self_attention_layers(cache) -> list:
    """A cache's self-attention layers."""
    halves = _cache_halves(cache)
    return getattr(halves[0], "layers", []) if halves else []


def keeps_write_once_state(layer) -> bool:
    """Whether this layer keeps state besides its keys and values (a conv window, an SSM state, indexer keys).

    Only the prompt's forward creates that state, so a decode-step graph cannot. Matched by name since there
    is no common base: `*_states`, or a `*_keys` other than `keys`.
    """
    return any(name.endswith("_states") or (name.endswith("_keys") and name != "keys") for name in vars(layer))


def _cache_length(cache) -> int:
    """`cache.get_seq_length()`; a recurrent-only cache, which raises, answers 0 or 1 by `has_previous_state`."""
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

    The graph's input spec pins those sizes; re-deriving them from the prompt (as `generate` does) would not fit.
    """
    if not lengths:
        return
    layers = _self_attention_layers(cache)
    for index, length in lengths.items():
        if index < len(layers) and getattr(layers[index], "max_cache_len", None) not in (None, length):
            layers[index].max_cache_len = length


def is_fixed_size(cache) -> bool:
    """Whether `cache` is allocated at its full length up front (`StaticCache`), read off its first layer."""
    layers = _self_attention_layers(cache)
    return bool(layers) and getattr(layers[0], "is_compileable", False)


def mask_width(cache, query_length: int) -> int:
    """How wide a causal mask over `cache` has to be, per `get_mask_sizes` on layer 0 as the eager mask builder asks.

    Not `get_max_length()`, which answers for the longest layer (mllama's vision cross-attention).
    """
    if cache is None:
        return query_length
    try:
        return int(cache.get_mask_sizes(query_length, 0)[0])
    except (AttributeError, IndexError, TypeError, ValueError, StopIteration):
        return _cache_length(cache) + query_length


def _advance_cache(past_key_values, outputs: dict[str, torch.Tensor], num_new_tokens: int):
    """Advance the cache with a decode step's `past_key_values.…` outputs, then its length counters.

    Same-shape leaves are copied in place; a grown `DynamicCache` is rebuilt through the registered pytree.
    """
    cache_updates = [
        (name, value) for name, value in outputs.items() if name.startswith(("past_key_values", "cache_params"))
    ]
    # Aligned by path, not leaf count: `None` recurrent states have no leaf yet.
    written = set()
    for name, new in cache_updates:
        if "." in name:
            path = name.split(".")[1:]
            _assign_cache_entry(past_key_values, path, new)
            written.add(id(_read_cache_entry(past_key_values, path)))
    by_index = [(name, new) for name, new in cache_updates if "." not in name]
    if by_index:
        # ExecuTorch: by flat leaf index (`past_key_values_<N>`), possibly pruned; rank-0 updates come back as scalars.
        cache_leaves = _cache_tensors(past_key_values)
        updated = list(cache_leaves)
        for position, (name, new) in enumerate(by_index):
            suffix = name.rsplit("_", 1)[-1]
            index = int(suffix) if suffix.isdigit() else position
            if not isinstance(new, torch.Tensor):
                new = torch.tensor(new, dtype=cache_leaves[index].dtype, device=cache_leaves[index].device)
            updated[index] = new
            written.update((id(cache_leaves[index]), id(new)))
        if any(old.shape != new.shape for old, new in zip(cache_leaves, updated)):
            _, spec = tree_flatten(past_key_values)
            past_key_values = tree_unflatten(updated, spec)
        else:
            for old, new in zip(cache_leaves, updated):
                if old is not new:
                    old.copy_(new)
    _mark_existing_states(past_key_values)
    advance_cache_length(past_key_values, num_new_tokens, written)
    # `is_updated` is pytree context the graph never flips; after any step the cross cache is written.
    if getattr(past_key_values, "is_updated", None):
        past_key_values.is_updated = dict.fromkeys(past_key_values.is_updated, True)
    return past_key_values


def advance_cache_length(past_key_values, num_new_tokens: int, written=()) -> None:
    """Advance each self-attention layer's length counter by `num_new_tokens`, the way the eager `update` does.

    Python-int counters on sliding layers are never updated by a graph (baked, or pytree context). A tensor
    counter whose `id` is in `written` came back from the graph already advanced, and is left alone."""
    for layer in _self_attention_layers(past_key_values):
        counter = getattr(layer, "cumulative_length", None)
        if hasattr(layer, "cumulative_length_int"):
            layer.cumulative_length_int += num_new_tokens
        elif isinstance(counter, int):
            layer.cumulative_length += num_new_tokens
        elif torch.is_tensor(counter) and id(counter) not in written:
            counter.add_(num_new_tokens)


def _mark_existing_states(past_key_values) -> None:
    """Mark each layer's recurrent states as existing, exactly where they do.

    Those flags are pytree context the graph cannot flip; left unset, the cache no longer matches the traced spec."""
    for layer in _self_attention_layers(past_key_values):
        conv_states = getattr(layer, "conv_states", None)
        if isinstance(conv_states, dict):
            for key, conv in conv_states.items():
                if isinstance(conv, torch.Tensor):
                    if isinstance(getattr(layer, "conv_kernel_size", None), dict):
                        layer.conv_kernel_size[key] = conv.shape[-1]
                    if getattr(layer, "dtype", None) is None:
                        layer.dtype, layer.device = conv.dtype, conv.device
        # Only states that exist: a conv-only layer (lfm2) never gets recurrent states.
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

    `"cache"` is a `DynamicCache` built from `encoder_config` and materialized to zero length (lazy layers
    would flatten shorter); anything else names a model's own class in its `modeling_*` module."""
    if container != "cache":
        module = _resolve_modeling_module(config)
        container_class = getattr(module, container, None)
        return container_class() if container_class is not None else None
    if encoder_config is None:
        return DynamicCache()
    cache = DynamicCache(config=encoder_config)
    materialize_cache_layers(cache, batch_size, encoder_config, dtype, device)
    return cache


# ── Geometry and materialization ──────────────────────────────────────────────


def _per_layer_head_shapes(config: Any) -> list[tuple[int, int, int]]:
    """Per-layer `(kv_heads, head_dim, head_dim)` from the config, for the layers no filled cache describes; empty
    for a model without attention."""
    text_config = config.get_text_config()
    if getattr(text_config.per_layer_config[0], "num_attention_heads", None) is None:
        return []
    num_heads, head_dim = get_head_shapes(text_config)
    num_layers = text_config.num_hidden_layers - (getattr(text_config, "num_kv_shared_layers", 0) or 0)
    num_heads = num_heads if isinstance(num_heads, list) else [num_heads] * num_layers
    head_dim = head_dim if isinstance(head_dim, list) else [head_dim] * num_layers
    return [(heads, dim, dim) for heads, dim in zip(num_heads, head_dim)]


def kv_geometry_of(cache: Any) -> dict[int, tuple[int, int, int]]:
    """`{layer index: (num_kv_heads, key_head_dim, value_head_dim)}` of a cache the model itself filled."""
    return {
        index: (layer.keys.shape[1], layer.keys.shape[3], layer.values.shape[3])
        for index, layer in enumerate(getattr(cache, "layers", []) or [])
        if getattr(layer, "keys", None) is not None and layer.keys.dim() == 4
    }


def indexer_layers_of(cache: Any) -> dict[int, bool]:
    """`{layer index: whether the layer holds an indexer tensor}` for a cache the model itself filled.

    The runtime's counterpart is `ExportMetadata.indexer_layers`, read off the graph.
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
    """Give every lazily-uninitialized cache layer real tensors, from a `[batch, kv_heads, 0, head_dim]` hint.

    `torch.export` cannot trace lazy allocation, so the traced cache and the runtime's must be materialized
    identically (the cache pytree is part of the input spec).

    Args:
        kv_geometry: per-layer `(kv_heads, key_dim, value_dim)` off the graph or a filled cache; wins over the
            config, which cannot always give it (mimo_v2_flash: 2 KV heads on sliding layers, 4 on full).
        indexer_layers: per-layer sparse-indexer presence (`ExportMetadata.indexer_layers`); layers without one
            are left alone.
    """
    kv_geometry = kv_geometry or {}
    by_layer = _per_layer_head_shapes(config)
    for cache_half in _cache_halves(cache):
        # A model-specific cache may have no `layers` list (xLSTM's `rnn_state`).
        if not hasattr(cache_half, "layers"):
            continue
        geometry = [
            kv_geometry.get(index) or (by_layer[index] if index < len(by_layer) else None)
            for index in range(len(cache_half.layers))
        ]
        # A cache with state of its own sizes it by overriding `early_initialization` (`MiniMaxCache`).
        if geometry and all(entry is not None for entry in geometry):
            cache_half.early_initialization(
                batch_size,
                [entry[0] for entry in geometry],
                [entry[1] for entry in geometry],
                dtype,
                device,
                value_head_dim=[entry[2] for entry in geometry],
            )
        for layer_idx, (layer, layer_geometry) in enumerate(zip(cache_half.layers, geometry)):
            # Before the `is_initialized` skip, which `early_initialization` sets with the indexer untouched; a
            # lazy indexer shifts every later leaf. Skipped where the trace had none (hy_v4's shared indexers).
            traced_indexer = indexer_layers.get(layer_idx, True) if indexer_layers else True
            if traced_indexer and hasattr(layer, "is_indexer_initialized") and not layer.is_indexer_initialized:
                index_head_dim = config.get_text_config().index_head_dim
                empty_indexer_keys = torch.zeros(batch_size, 0, index_head_dim, dtype=dtype, device=device)
                layer.lazy_initialization_indexer(empty_indexer_keys)
            if getattr(layer, "is_initialized", True) or layer_geometry is None:
                continue
            num_kv_heads, key_dim, value_dim = layer_geometry
            # The recorded head count is part of the input spec, so it follows the buffers, not the config.
            if hasattr(layer, "num_heads"):
                layer.num_heads = num_kv_heads
            empty_keys = torch.zeros(batch_size, num_kv_heads, 0, key_dim, dtype=dtype, device=device)
            empty_values = torch.zeros(batch_size, num_kv_heads, 0, value_dim, dtype=dtype, device=device)
            layer.lazy_initialization(empty_keys, empty_values)
