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
from typing import TYPE_CHECKING

from ..cache_utils import DynamicCache, DynamicLayer, DynamicSlidingWindowLayer
from ..generation.utils import GenerationMixin
from ..masking_utils import (
    ALL_MASK_ATTENTION_FUNCTIONS,
    _ignore_causal_mask_sdpa,
    and_masks,
    causal_mask_function,
    padding_mask_function,
    prepare_padding_mask,
)
from ..utils import logging
from ..utils.export_config import DynamoConfig
from ..utils.import_utils import is_torch_available, is_torch_greater_or_equal
from .base import HfExporter


if is_torch_available():
    import torch


if TYPE_CHECKING:
    from ..modeling_utils import PreTrainedModel

logger = logging.get_logger(__file__)


class DynamoExporter(HfExporter):
    export_config: DynamoConfig

    required_packages = ["torch"]

    def validate_environment(self, *args, **kwargs):
        super().validate_environment(*args, **kwargs)

        if not hasattr(torch, "export"):
            raise ImportError(
                "DynamoExporter requires torch.export which does not seem to be available in your torch installation. "
                "Please update your torch installation to a more recent version (torch 2.6.0+)."
            )

    def export(self, model: "PreTrainedModel"):
        from torch.export import Dim, ExportedProgram

        if self.export_config.sample_inputs is None:
            raise NotImplementedError(
                "OnnxExporter can't automatically generate export inptus. Please provide sample_inputs in the exporter_config as a dictionary. "
                "You can do so by using the tokenizer/processor to prepare a batch of inputs as you would do for a normal forward pass. "
                "OnnxExporter can automatically generate past_key_values and its dynamic shapes if the model is "
                "auto-regressive and model.config.use_cache is set to True."
            )

        args = ()
        kwargs = self.export_config.sample_inputs
        dynamic_shapes = self.export_config.dynamic_shapes

        if isinstance(model, GenerationMixin) and model.config.use_cache:
            register_dynamic_cache_export_support()

            if "past_key_values" not in kwargs:
                kwargs["past_key_values"] = model(**kwargs).past_key_values

                if dynamic_shapes is not None:
                    dynamic_shapes["past_key_values"] = [
                        [{0: Dim.DYNAMIC, 2: Dim.DYNAMIC} for _ in range(len(kwargs["past_key_values"].layers))],
                        [{0: Dim.DYNAMIC, 2: Dim.DYNAMIC} for _ in range(len(kwargs["past_key_values"].layers))],
                    ]

        ALL_MASK_ATTENTION_FUNCTIONS["sdpa"] = sdpa_mask_without_vmap
        ALL_MASK_ATTENTION_FUNCTIONS["eager"] = eager_mask_without_vmap

        exported_program: ExportedProgram = torch.export.export(
            model,
            args=args,
            kwargs=kwargs,
            dynamic_shapes=dynamic_shapes,
            strict=self.export_config.strict,
        )
        model.exported_model = exported_program


def register_dynamic_cache_export_support():
    """
    Utilities for `DynamicCache` <> torch.export support
    """

    try:
        torch.utils._pytree.register_pytree_node(
            DynamicCache,
            lambda dynamic_cache: torch.utils._pytree._dict_flatten(_get_cache_dict(dynamic_cache)),
            _unflatten_dynamic_cache,
            serialized_type_name=f"{DynamicCache.__module__}.{DynamicCache.__name__}",
            flatten_with_keys_fn=lambda dynamic_cache: torch.utils._pytree._dict_flatten_with_keys(
                _get_cache_dict(dynamic_cache)
            ),
        )
        # TODO (tmanlaibaatar) This won't be needed in torch 2.7.
        torch.fx._pytree.register_pytree_flatten_spec(
            DynamicCache,
            lambda cache, spec: torch.fx._pytree._dict_flatten_spec(_get_cache_dict(cache), spec),
        )
    # Catching this in case there are multiple runs for some test runs
    except ValueError as e:
        if "already registered as pytree node" not in str(e):
            raise


def _get_cache_dict(cache: DynamicCache):
    """Convert cache to dictionary format for pytree operations."""
    if any(not isinstance(layer, DynamicLayer | DynamicSlidingWindowLayer) for layer in cache.layers):
        raise RuntimeError("This pytree flattening function should only be applied to DynamicCache")

    if not is_torch_greater_or_equal("2.6.0"):
        logging.warning("DynamicCache + torch.export is tested on torch 2.6.0+ and may not work on earlier versions.")

    return {
        "key_cache": [layer.keys for layer in cache.layers if layer.keys is not None],
        "value_cache": [layer.values for layer in cache.layers if layer.values is not None],
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
        cache.layers.append(cache_layer)
    return cache


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
