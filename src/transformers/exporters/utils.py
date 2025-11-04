# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
from collections.abc import Callable
from contextlib import contextmanager

from ..cache_utils import Cache, DynamicCache, DynamicLayer, DynamicSlidingWindowLayer, EncoderDecoderCache
from ..masking_utils import (
    ALL_MASK_ATTENTION_FUNCTIONS,
    _ignore_causal_mask_sdpa,
    and_masks,
    causal_mask_function,
    eager_mask,
    padding_mask_function,
    prepare_padding_mask,
    sdpa_mask,
)
from ..utils.import_utils import is_torch_available


if is_torch_available():
    import torch


def _get_dynamic_cache_dict(cache: DynamicCache):
    """Converts DynamicCache to dictionary format for pytree operations."""
    if any(not isinstance(layer, DynamicLayer | DynamicSlidingWindowLayer) for layer in cache.layers):
        raise RuntimeError("This pytree flattening function should only be applied to DynamicCache")

    return {
        "key_cache": [layer.keys for layer in cache.layers if layer.keys is not None],
        "value_cache": [layer.values for layer in cache.layers if layer.values is not None],
    }


def get_encoder_decoder_cache_dict(cache: EncoderDecoderCache):
    """Converts EncoderDecoderCache to dictionary format for pytree operations."""
    return {
        "self_attention_cache": _get_dynamic_cache_dict(cache.self_attention_cache),
        "cross_attention_cache": _get_dynamic_cache_dict(cache.cross_attention_cache),
    }


def _unflatten_dynamic_cache(values, context: torch.utils._pytree.Context):
    dictionary = torch.utils._pytree._dict_unflatten(values, context)
    cache = DynamicCache()
    # Reconstruct layers from keys and values lists
    key_list = dictionary.get("key_cache", [])
    value_list = dictionary.get("value_cache", [])
    for idx in range(max(len(key_list), len(value_list))):
        key = key_list[idx] if idx < len(key_list) else None
        value = value_list[idx] if idx < len(value_list) else None
        cache_layer = DynamicLayer()
        cache_layer.keys = key
        cache_layer.values = value
        cache_layer.is_initialized = True
        cache.layers.append(cache_layer)
    return cache


def _unflatten_encoder_decoder_cache(values, context: torch.utils._pytree.Context):
    dictionary = torch.utils._pytree._dict_unflatten(values, context)
    self_attention_cache = _unflatten_dynamic_cache(
        [
            dictionary.get("self_attention_cache", {}).get("key_cache", []),
            dictionary.get("self_attention_cache", {}).get("value_cache", []),
        ],
        context,
    )
    cross_attention_cache = _unflatten_dynamic_cache(
        [
            dictionary.get("cross_attention_cache", {}).get("key_cache", []),
            dictionary.get("cross_attention_cache", {}).get("value_cache", []),
        ],
        context,
    )
    return EncoderDecoderCache(self_attention_cache, cross_attention_cache)


def register_dynamic_cache_for_export():
    try:
        torch.utils._pytree.register_pytree_node(
            DynamicCache,
            lambda dynamic_cache: torch.utils._pytree._dict_flatten(_get_dynamic_cache_dict(dynamic_cache)),
            _unflatten_dynamic_cache,
            serialized_type_name=f"{DynamicCache.__module__}.{DynamicCache.__name__}",
            flatten_with_keys_fn=lambda dynamic_cache: torch.utils._pytree._dict_flatten_with_keys(
                _get_dynamic_cache_dict(dynamic_cache)
            ),
        )
        # TODO (tmanlaibaatar) This won't be needed in torch 2.7.
        torch.fx._pytree.register_pytree_flatten_spec(
            DynamicCache,
            lambda cache, spec: torch.fx._pytree._dict_flatten_spec(_get_dynamic_cache_dict(cache), spec),
        )
    # Catching this in case there are multiple runs for some test runs
    except ValueError as e:
        if "already registered as pytree node" not in str(e):
            raise


def register_encoder_decoder_cache_for_export():
    try:
        torch.utils._pytree.register_pytree_node(
            EncoderDecoderCache,
            lambda cache: torch.utils._pytree._dict_flatten(get_encoder_decoder_cache_dict(cache)),
            _unflatten_encoder_decoder_cache,
            serialized_type_name=f"{EncoderDecoderCache.__module__}.{EncoderDecoderCache.__name__}",
            flatten_with_keys_fn=lambda cache: torch.utils._pytree._dict_flatten_with_keys(
                get_encoder_decoder_cache_dict(cache)
            ),
        )
        # TODO (tmanlaibaatar) This won't be needed in torch 2.7.
        torch.fx._pytree.register_pytree_flatten_spec(
            EncoderDecoderCache,
            lambda cache, spec: torch.fx._pytree._dict_flatten_spec(get_encoder_decoder_cache_dict(cache), spec),
        )
    # Catching this in case there are multiple runs for some test runs
    except ValueError as e:
        if "already registered as pytree node" not in str(e):
            raise


# TODO: won't be needed when it becomes the default in transformers
# Custom vectorized implementation of sdpa_mask without using vmap
def sdpa_mask_without_vmap(
    batch_size: int,
    cache_position: torch.Tensor,
    kv_length: int,
    kv_offset: int = 0,
    mask_function: Callable | None = None,
    attention_mask: torch.Tensor | None = None,
    local_size: int | None = None,
    allow_is_causal_skip: bool = True,
    **kwargs,
) -> torch.Tensor | None:
    if mask_function is None:
        mask_function = causal_mask_function

    q_length = cache_position.shape[0]
    # Potentially pad the 2D mask, and slice it correctly
    padding_mask = prepare_padding_mask(attention_mask, kv_length, kv_offset, _slice=False)

    # Under specific conditions, we can avoid materializing the mask, instead relying on the `is_causal` argument
    if allow_is_causal_skip and _ignore_causal_mask_sdpa(padding_mask, q_length, kv_length, kv_offset, local_size):
        return None

    # Potentially add the padding 2D mask
    if padding_mask is not None:
        mask_function = and_masks(mask_function, padding_mask_function(padding_mask))

    # Create broadcatable indices
    device = cache_position.device
    q_indices = cache_position[None, None, :, None]
    head_indices = torch.arange(1, dtype=torch.long, device=device)[None, :, None, None]
    batch_indices = torch.arange(batch_size, dtype=torch.long, device=device)[:, None, None, None]
    kv_indices = torch.arange(kv_length, dtype=torch.long, device=device)[None, None, None, :] + kv_offset
    # Apply mask function element-wise through broadcasting
    causal_mask = mask_function(batch_indices, head_indices, q_indices, kv_indices)
    # Expand the mask to match batch size and query length if they weren't used in the mask function
    causal_mask = causal_mask.expand(batch_size, -1, q_length, kv_length)

    return causal_mask


# Adapted from https://github.com/huggingface/transformers/blob/v4.53.0/src/transformers/masking_utils.py#L433
def eager_mask_without_vmap(*args, **kwargs) -> torch.Tensor:
    kwargs.pop("allow_is_causal_skip", None)
    dtype = kwargs.get("dtype", torch.float32)
    mask = sdpa_mask_without_vmap(*args, allow_is_causal_skip=False, **kwargs)
    mask = torch.where(mask, torch.tensor(0.0, device=mask.device, dtype=dtype), torch.finfo(dtype).min)
    return mask


@contextmanager
def patch_masks_for_export():
    """
    Patch masking functions to use the non-vmap versions during export.
    """
    ALL_MASK_ATTENTION_FUNCTIONS["sdpa"] = sdpa_mask_without_vmap
    ALL_MASK_ATTENTION_FUNCTIONS["eager"] = eager_mask_without_vmap

    try:
        yield
    finally:
        ALL_MASK_ATTENTION_FUNCTIONS["sdpa"] = sdpa_mask
        ALL_MASK_ATTENTION_FUNCTIONS["eager"] = eager_mask


def get_auto_dynamic_shapes(inputs: dict[str, torch.Tensor | Cache]) -> dict[str, dict[int, torch.export.Dim]]:
    """
    Utility function to automatically generate dynamic shapes for a dictionary of model inputs.

    Args:
        inputs (`dict[str, torch.Tensor | Cache]`):
            The inputs with which the model will be exported.
    Returns:
        `dict[str, dict[int, torch.export.Dim]]`: A dictionary mapping input names to their dynamic shapes.
    """
    from torch.export import Dim

    dynamic_shapes = {}
    for name, input in inputs.items():
        if isinstance(input, DynamicCache):
            dynamic_shapes[name] = [
                [dict.fromkeys(range(len(layer.keys.shape)), Dim.AUTO) for layer in input.layers],
                [dict.fromkeys(range(len(layer.values.shape)), Dim.AUTO) for layer in input.layers],
            ]
        elif isinstance(input, torch.Tensor):
            dynamic_shapes[name] = dict.fromkeys(range(len(input.shape)), Dim.AUTO)
        else:
            raise ValueError(
                f"Input '{name}' is of unsupported type '{type(input)}'. Only 'torch.Tensor' and 'DynamicCache' are supported."
            )

    return dynamic_shapes
