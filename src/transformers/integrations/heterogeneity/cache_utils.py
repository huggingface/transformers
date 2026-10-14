# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

from __future__ import annotations

from typing import TYPE_CHECKING

from transformers.integrations.heterogeneity.heterogeneous_modeling_spec import KEY_VALUE_CACHE, LINEAR_ATTENTION_CACHE


if TYPE_CHECKING:
    from transformers import PreTrainedConfig
    from transformers.cache_utils import Cache


# Maps each hybrid layer type, which holds several caches, to the layer type to use when only some of those caches
# remain. For example, a "hybrid" layer that keeps only its key/value cache becomes "full_attention". This is written
# by hand because one cache class can serve several layer types that are built with different arguments, like
# "sliding_attention" and "chunked_attention".
_HYBRID_LAYER_TYPE_TO_REDUCED_LAYER_TYPES = {
    "hybrid": {
        frozenset({KEY_VALUE_CACHE}): "full_attention",
        frozenset({LINEAR_ATTENTION_CACHE}): "linear_attention",
    },
    "hybrid_sliding": {
        frozenset({KEY_VALUE_CACHE}): "sliding_attention",
        frozenset({LINEAR_ATTENTION_CACHE}): "linear_attention",
    },
}


def _get_cache_layer_caches(cache_layer_cls: type) -> set[str]:
    """Return the caches that a cache layer class holds."""
    # Imported here because the core `cache_utils` module imports this one
    from transformers.cache_utils import CacheLayerMixin, LinearAttentionCacheLayerMixin

    caches = set()
    if issubclass(cache_layer_cls, CacheLayerMixin):
        caches.add(KEY_VALUE_CACHE)
    if issubclass(cache_layer_cls, LinearAttentionCacheLayerMixin):
        caches.add(LINEAR_ATTENTION_CACHE)

    return caches


def _get_layer_type_caches(layer_type: str) -> set[str] | None:
    """Return the caches that the cache layer class of `layer_type` holds, or `None` if `layer_type` has no class."""
    # Imported here because the core `cache_utils` module imports this one
    from transformers.cache_utils import DYNAMIC_LAYER_TYPE_MAPPING, STATIC_LAYER_TYPE_MAPPING

    layer_cls = DYNAMIC_LAYER_TYPE_MAPPING.get(layer_type, STATIC_LAYER_TYPE_MAPPING.get(layer_type))
    if layer_cls is None:
        return None

    return _get_cache_layer_caches(layer_cls)


def _get_layer_type_without_caches(layer_idx: int, layer_type: str, dropped_caches: set[str]) -> str:
    """Return the layer type that holds only the caches of `layer_type` that remain after dropping `dropped_caches`."""
    # Nothing was dropped, so the layer keeps its layer type
    if not dropped_caches:
        return layer_type

    # Find the caches that this layer type holds
    layer_type_caches = _get_layer_type_caches(layer_type)
    if layer_type_caches is None:
        raise ValueError(
            f"The skips of layer {layer_idx} replaced every writer of its {sorted(dropped_caches)} caches, but layer "
            f"type '{layer_type}' has no cache layer class, so the caches it holds are unknown."
        )

    # The layer type holds none of the dropped caches, which happens when the spec and `layer_types` disagree about a
    # layer, so there's nothing to drop
    if layer_type_caches.isdisjoint(dropped_caches):
        return layer_type

    # Every cache of the layer type was dropped, so the layer gets a `NoCacheLayer`
    remaining_caches = layer_type_caches - dropped_caches
    if not remaining_caches:
        return "no_cache"

    # Only some caches remain, which happens only for hybrid layer types. Use the layer type that holds just those.
    reduced_layer_type = _HYBRID_LAYER_TYPE_TO_REDUCED_LAYER_TYPES.get(layer_type, {}).get(frozenset(remaining_caches))
    if reduced_layer_type is None:
        raise ValueError(
            f"The skips of layer {layer_idx} replaced every writer of its {sorted(dropped_caches)} caches, but no "
            f"layer type is known to hold only the {sorted(remaining_caches)} caches of layer type '{layer_type}'."
        )
    return reduced_layer_type


def get_generic_heterogeneous_modeling_cache_layer_types(
    config: PreTrainedConfig, layer_types: list[str]
) -> list[str]:
    """Return the cache layer types of a model built with generic heterogeneous modeling.

    When the skips of a layer replaced every writer of some of its caches, the layer gets the layer type that holds
    only the remaining caches, or a `NoCacheLayer` when no cache remains.

    Args:
        config: The model's config.
        layer_types: The layer types of the model's layers.

    Returns:
        The layer types to build the cache layers with.
    """
    per_layer_unwritten_caches = config._heterogeneity_spec.generic_modeling_state.per_layer_unwritten_caches
    return [
        _get_layer_type_without_caches(layer_idx, layer_type, per_layer_unwritten_caches.get(layer_idx, set()))
        for layer_idx, layer_type in enumerate(layer_types)
    ]


def validate_no_unwritten_caches(past_key_values: Cache, *, layer_idx: int, unwritten_caches: set[str]) -> None:
    """Raise if `past_key_values` has a cache for layer `layer_idx`, but the layer never writes to it.

    `unwritten_caches` are the caches the layer never writes. A cache built from the model config doesn't have them,
    but other caches, like `DynamicCache()`, have them and they stay empty, so methods like `get_seq_length` can return
    wrong values.
    """
    if layer_idx < len(past_key_values.layers):
        cache_layer_cls = type(past_key_values.layers[layer_idx])
    elif past_key_values.layer_class_to_replicate is not None:
        # The cache adds its missing layers when a later layer is written
        cache_layer_cls = past_key_values.layer_class_to_replicate
    else:
        return

    if held_unwritten_caches := unwritten_caches & _get_cache_layer_caches(cache_layer_cls):
        raise ValueError(
            f"The cache holds a {' and '.join(sorted(held_unwritten_caches))} cache for layer {layer_idx}, but the "
            "layer's skips replaced every module that writes it. Build the cache from the model config, for "
            "example with `DynamicCache(config=model.config)`."
        )
