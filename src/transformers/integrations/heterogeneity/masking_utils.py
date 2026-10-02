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

from collections.abc import Callable
from functools import partial, wraps
from inspect import signature, unwrap
from typing import TYPE_CHECKING, Any

from transformers.utils import is_torch_tensor


if TYPE_CHECKING:
    from transformers import PreTrainedConfig


# Defaults match the mask factories when an attribute is absent from the config. These attributes are global-only, so
# they're the same on every layer.
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
    from transformers.masking_utils import (
        LAYER_PATTERN_TO_MASK_FUNCTION_MAPPING,
        create_bidirectional_mask,
        create_bidirectional_sliding_window_mask,
        create_causal_mask,
        create_sliding_window_causal_mask,
    )

    matching_patterns = set()
    mask_fn = _unwrap_mask_function(create_mask_fn)

    # Reuse the registry's causal full/sliding attention entries to select layers
    # for bidirectional masks.
    if mask_fn is _unwrap_mask_function(create_bidirectional_mask):
        mask_fn = _unwrap_mask_function(create_causal_mask)
    elif mask_fn is _unwrap_mask_function(create_bidirectional_sliding_window_mask):
        mask_fn = _unwrap_mask_function(create_sliding_window_causal_mask)

    for pattern, entry in LAYER_PATTERN_TO_MASK_FUNCTION_MAPPING.items():
        mask_functions = entry.values() if isinstance(entry, dict) else (entry,)
        if any(_unwrap_mask_function(fn) is mask_fn for fn in mask_functions):
            matching_patterns.add(pattern)

    return [idx for idx, pattern in enumerate(layer_patterns) if pattern in matching_patterns]


def _get_cache_geometry(
    *,
    past_key_values: Any,
    query_length: Any,
    layer_idx: int,
    assume_layers_have_same_query_offset_when_tensor: bool,
) -> tuple[int, ...] | None:
    if past_key_values is None:
        return ()

    query_offset = past_key_values.get_query_offset(layer_idx)
    key_value_length, key_value_offset = past_key_values.get_mask_sizes(query_length, layer_idx)
    if not is_torch_tensor(query_offset):
        return query_offset, key_value_length, key_value_offset

    # An initialized `StaticLayer` returns its query offset as a device tensor. Comparing tensor values would force a
    # device sync, or a graph break under `torch.compile`, so, like upstream mask creation, assume these layers are all
    # at the same position and leave the query offset out.
    return (key_value_length, key_value_offset) if assume_layers_have_same_query_offset_when_tensor else None


def _get_mask_reuse_key(
    *,
    mask_settings: tuple[Any, ...],
    past_key_values: Any,
    query_length: Any,
    layer_idx: int,
    assume_layers_have_same_query_offset_when_tensor: bool,
) -> tuple[Any, ...] | None:
    cache_geometry = _get_cache_geometry(
        past_key_values=past_key_values,
        query_length=query_length,
        layer_idx=layer_idx,
        assume_layers_have_same_query_offset_when_tensor=assume_layers_have_same_query_offset_when_tensor,
    )
    if cache_geometry is None:
        return None

    return mask_settings, cache_geometry


def _create_attention_masks_by_layer_idx(
    *,
    create_mask_fn: Callable,
    attribute_name: str | None,
    config: PreTrainedConfig,
    **kwargs: Any,
) -> AttentionMasksByLayerIdx:
    attention_masks = AttentionMasksByLayerIdx()
    # Keys are compared with `==` instead of being stored in a dict. Under `torch.compile`, a dict lookup ties the
    # compiled graph to the exact cache length in the key, so it would recompile at every decoding step. `==` only
    # needs the lengths to be equal to each other.
    reuse_keys_and_layer_indices: list[tuple[tuple[Any, ...], int]] = []
    past_key_values = kwargs.get("past_key_values")
    common_mask_settings = tuple(
        getattr(config, name, default) for name, default in _COMMON_MASK_AFFECTING_ATTRIBUTES.items()
    )
    layer_configs = config._heterogeneity_spec.model_layer_configs
    # A layer that skips a module may never write to its cache, so its static position can fall behind the others'
    any_layer_skips = any(layer_config.skip for layer_config in layer_configs.values())

    for layer_idx in _get_mask_layer_indices(config, create_mask_fn):
        layer_config = layer_configs[layer_idx]

        if attribute_name is not None:
            attribute_value = getattr(layer_config, attribute_name)
            if attribute_value is None:
                continue

            mask_settings = (attribute_value,)
        else:
            mask_settings = ()

        mask_settings = (*mask_settings, *common_mask_settings)

        layer_kwargs = {**kwargs, "layer_idx": layer_idx}

        reuse_key = _get_mask_reuse_key(
            mask_settings=mask_settings,
            past_key_values=past_key_values,
            query_length=layer_kwargs["inputs_embeds"].shape[1],
            layer_idx=layer_idx,
            assume_layers_have_same_query_offset_when_tensor=not any_layer_skips,
        )

        if reuse_key is not None:
            reused_layer_idx = next(
                (idx for key, idx in reuse_keys_and_layer_indices if key == reuse_key),
                None,
            )
            if reused_layer_idx is not None:
                attention_masks[layer_idx] = attention_masks[reused_layer_idx]
                continue

        attention_masks[layer_idx] = create_mask_fn(config=layer_config, **layer_kwargs)
        if reuse_key is not None:
            reuse_keys_and_layer_indices.append((reuse_key, layer_idx))

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

            if config.generic_modeling_applied and layer_idx is None:
                attention_mask = mask_kwargs.get("attention_mask")
                if isinstance(attention_mask, AttentionMasksByLayerIdx):
                    return attention_mask

                return _create_attention_masks_by_layer_idx(
                    create_mask_fn=create_mask_fn,
                    attribute_name=attribute_name,
                    **mask_kwargs,
                )

            return create_mask_fn(*args, **kwargs)

        return wrapped

    return decorator
