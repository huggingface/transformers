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
from functools import wraps
from inspect import signature
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from transformers import PreTrainedConfig


class AttentionMasksByLayerIdx(dict[int, Any]):
    """Attention masks selected by layer index."""


def support_per_layer_mask_creation(attribute_name: str) -> Callable:
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
                return create_attention_masks_by_layer_idx(
                    create_mask_fn,
                    attribute_name,
                    **mask_kwargs,
                )

            return create_mask_fn(*args, **kwargs)

        return wrapped

    return decorator


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
    attribute_value: Any,
    past_key_values: Any,
    query_length: Any,
    layer_idx: int,
) -> tuple[Any, ...] | None:
    if not isinstance(attribute_value, Hashable):
        return None

    cache_geometry = _get_cache_geometry(past_key_values, query_length, layer_idx)
    if cache_geometry is None:
        return None

    return attribute_value, cache_geometry


def create_attention_masks_by_layer_idx(
    create_mask_fn: Callable,
    attribute_name: str,
    config: PreTrainedConfig,
    *args: Any,
    **kwargs: Any,
) -> AttentionMasksByLayerIdx:
    attention_masks = AttentionMasksByLayerIdx()
    # Reuse assumes that the per-layer value of `attribute_name` is the only `per_layer_config` override that affects
    # mask creation. `is_causal` and `_attn_implementation` also affect masks, but are assumed not to vary by layer.
    masks_by_reuse_key: dict[tuple[Any, ...], Any] = {}
    disabled_kv_layer_indices = set(config.get_disabled_kv_layer_indices())
    past_key_values = kwargs.get("past_key_values")

    for layer_idx in range(config.num_hidden_layers):
        if layer_idx in disabled_kv_layer_indices:
            attention_masks[layer_idx] = None
            continue

        layer_config = config.per_layer_config[layer_idx]
        attribute_value = getattr(layer_config, attribute_name)
        if attribute_value is None:
            continue

        layer_kwargs = {**kwargs, "layer_idx": layer_idx}
        reuse_key = _get_mask_reuse_key(
            attribute_value,
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
