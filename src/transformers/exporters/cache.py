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
"""The cache a decode graph takes, read off a captured call and rebuilt for another.

A decode graph that also serves the prefill step takes the same cache structure on both calls: the prefill call's
cache is materialized empty in the decode capture's geometry (key/value shapes, sparse-indexer slots, zeroed
linear-attention states), and its mask is built to match.
"""

from __future__ import annotations

from typing import Any

from ..utils.import_utils import is_torch_available
from .utils import module_device, module_dtype


if is_torch_available():
    import torch

    from ..configuration_utils import get_head_shapes


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


def causal_mask(position_ids, cache_len: int):
    """Full causal mask `[batch, 1, query, cache_len]` from the positions (M-RoPE: text row 0). Assumes no
    left-padding."""
    text_positions = position_ids if position_ids.dim() == 2 else position_ids[0]
    positions = torch.arange(cache_len, device=text_positions.device)
    return (positions <= text_positions[..., None])[:, None]


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


def linear_states_of(cache: Any) -> dict[int, dict[str, list[tuple[int, ...]]]]:
    """`{layer index: {"conv_states"/"recurrent_states": [per-state shape without batch]}}` of a cache the model itself
    filled."""
    layout = {}
    for index, layer in enumerate(getattr(cache, "layers", []) or []):
        states = {
            name: [tuple(state.shape[1:]) for _, state in sorted(getattr(layer, name).items())]
            for name in ("conv_states", "recurrent_states")
            if isinstance(getattr(layer, name, None), dict)
            and all(isinstance(state, torch.Tensor) for state in getattr(layer, name).values())
        }
        if states:
            layout[index] = states
    return layout


def indexer_layers_of(cache: Any) -> dict[int, bool]:
    """`{layer index: whether the layer holds an indexer tensor}` for a cache the model itself filled."""
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
    linear_states: dict[int, dict[str, list[tuple[int, ...]]]] | None = None,
) -> None:
    """Give every lazily-uninitialized cache layer empty tensors (`[batch, kv_heads, 0, head_dim]`).

    `torch.export` cannot trace lazy allocation, and the cache pytree is part of the input spec, so a cache fed to a
    graph must be materialized the way the traced one was.

    Args:
        kv_geometry: per-layer `(kv_heads, key_dim, value_dim)` off a filled cache; wins over the config, which cannot
            always give it (mimo_v2_flash: 2 KV heads on sliding layers, 4 on full).
        indexer_layers: per-layer sparse-indexer presence; layers without one are left alone.
        linear_states: per-layer linear-attention state shapes, allocated as zeros and marked existing.
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
            for name, shapes in (linear_states or {}).get(layer_idx, {}).items():
                for state_idx, shape in enumerate(shapes):
                    if getattr(layer, name)[state_idx] is None:
                        zeros = torch.zeros(batch_size, *shape, dtype=dtype, device=device)
                        layer.lazy_initialization(**{name: zeros}, state_idx=state_idx)
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
        if linear_states:
            _mark_existing_states(cache_half)


def materialize_cache_like(model, inputs: dict, reference_inputs: dict) -> None:
    """Materialize `inputs`' cache like `reference_inputs`' (the prefill call's like the decode capture's), so the
    graph traced on one takes the other.

    Geometry, indexer slots and linear-attention states come from the reference cache, which holds what the model
    really writes; layers left empty there (hy_v4's shared indexers) stay empty so both take the same leaves.
    """
    name = next((name for name in ("past_key_values", "cache_params") if inputs.get(name) is not None), None)
    if name is None:
        return
    cache, reference = inputs[name], reference_inputs.get(name)
    batch_size = next(t for t in inputs.values() if isinstance(t, torch.Tensor)).shape[0]
    materialize_cache_layers(
        cache,
        batch_size,
        model.config,
        module_dtype(model),
        module_device(model),
        kv_geometry=kv_geometry_of(reference),
        indexer_layers=indexer_layers_of(reference),
        linear_states=linear_states_of(reference),
    )
