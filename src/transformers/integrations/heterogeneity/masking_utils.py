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

from collections.abc import Callable, Hashable
from functools import partial, wraps
from inspect import signature, unwrap
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from transformers import PreTrainedConfig


# Defaults match the mask factories when an attribute is absent from the config.
_COMMON_MASK_AFFECTING_ATTRIBUTES = {"is_causal": True, "_attn_implementation": None}


class AttentionMasksByLayerIdx(dict[int, Any]):
    """Attention masks selected by layer index."""


def _unwrap_mask_function(fn: Callable) -> Callable:
    fn = unwrap(fn)
    while isinstance(fn, partial):
        fn = unwrap(fn.func)
    return fn


def _get_mask_layer_indices(config: PreTrainedConfig, create_mask_fn: Callable) -> range | list[int]:
    layer_patterns = getattr(config, "layer_types", None)
    if layer_patterns is None:
        return range(config.num_hidden_layers)

    # Lazy import to avoid circular imports
    from transformers.masking_utils import LAYER_PATTERN_TO_MASK_FUNCTION_MAPPING

    matching_patterns = set()
    mask_fn = _unwrap_mask_function(create_mask_fn)
    for pattern, entry in LAYER_PATTERN_TO_MASK_FUNCTION_MAPPING.items():
        mask_functions = entry.values() if isinstance(entry, dict) else (entry,)
        if any(_unwrap_mask_function(fn) is mask_fn for fn in mask_functions):
            matching_patterns.add(pattern)

    return [idx for idx, pattern in enumerate(layer_patterns) if pattern in matching_patterns]


def _get_cache_geometry(
    past_key_values: Any,
    query_length: Any,
    layer_idx: int,
) -> tuple[int, ...] | None:
    if past_key_values is None:
        return ()

    query_offset = past_key_values.get_query_offset(layer_idx)
    key_value_length, key_value_offset = past_key_values.get_mask_sizes(query_length, layer_idx)
    geometry = (query_offset, key_value_length, key_value_offset)
    return geometry if all(type(value) is int for value in geometry) else None


def _get_mask_reuse_key(
    mask_settings: tuple[Any, ...],
    past_key_values: Any,
    query_length: Any,
    layer_idx: int,
) -> tuple[Any, ...] | None:
    if not all(isinstance(value, Hashable) for value in mask_settings):
        return None

    cache_geometry = _get_cache_geometry(past_key_values, query_length, layer_idx)
    if cache_geometry is None:
        return None

    return mask_settings, cache_geometry


def _create_attention_masks_by_layer_idx(
    create_mask_fn: Callable,
    attribute_name: str | None,
    config: PreTrainedConfig,
    *args: Any,
    **kwargs: Any,
) -> AttentionMasksByLayerIdx:
    attention_masks = AttentionMasksByLayerIdx()
    masks_by_reuse_key: dict[tuple[Any, ...], Any] = {}
    past_key_values = kwargs.get("past_key_values")

    for layer_idx in _get_mask_layer_indices(config, create_mask_fn):
        # Resolving a layer config copies it, which currently prevents full-graph compilation of this path.
        layer_config = config.per_layer_config[layer_idx]

        if attribute_name is not None:
            attribute_value = getattr(layer_config, attribute_name)
            if attribute_value is None:
                continue

            mask_settings = (attribute_value,)
        else:
            mask_settings = ()

        mask_settings = (
            *mask_settings,
            *(getattr(layer_config, name, default) for name, default in _COMMON_MASK_AFFECTING_ATTRIBUTES.items()),
        )

        layer_kwargs = {**kwargs, "layer_idx": layer_idx}

        reuse_key = _get_mask_reuse_key(
            mask_settings,
            past_key_values,
            layer_kwargs["inputs_embeds"].shape[1],
            layer_idx,
        )

        if reuse_key is not None and reuse_key in masks_by_reuse_key:
            attention_masks[layer_idx] = masks_by_reuse_key[reuse_key]
            continue

        attention_masks[layer_idx] = create_mask_fn(layer_config, *args, **layer_kwargs)
        if reuse_key is not None:
            masks_by_reuse_key[reuse_key] = attention_masks[layer_idx]

    return attention_masks


def support_per_layer_mask_creation(attribute_name: str | None = None) -> Callable:
    """Decorate a mask factory to return layer-indexed masks for generic heterogeneous models."""

    def decorator(create_mask_fn: Callable) -> Callable:
        parameter_names = tuple(signature(create_mask_fn).parameters)

        @wraps(create_mask_fn)
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            mask_kwargs = dict(zip(parameter_names, args))
            mask_kwargs.update(kwargs)
            config = mask_kwargs["config"]
            layer_idx = mask_kwargs.get("layer_idx")

            if layer_idx is None and config.is_heterogeneous and config._heterogeneity_spec.generic_modeling_applied:
                attention_mask = mask_kwargs.get("attention_mask")
                if isinstance(attention_mask, AttentionMasksByLayerIdx):
                    return attention_mask

                return _create_attention_masks_by_layer_idx(
                    create_mask_fn,
                    attribute_name,
                    **mask_kwargs,
                )

            return create_mask_fn(*args, **kwargs)

        return wrapped

    return decorator
