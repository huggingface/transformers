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

from collections import defaultdict
from collections.abc import Callable, Hashable
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from transformers import PreTrainedConfig


class AttentionMasksByAttributeValue(dict[Hashable, Any]):
    """Attention masks selected by the value of a per-layer config attribute."""


class _SinglePatternAttentionMasks(AttentionMasksByAttributeValue):
    """Attribute-value masks that also support lookup by their shared attention pattern.
    This lets existing model code resolve masks without adding heterogeneity-specific branches.
    """

    def __init__(self, attention_pattern: str) -> None:
        super().__init__()
        self._attention_pattern = attention_pattern

    def __getitem__(self, key: Hashable) -> Any:
        if key == self._attention_pattern:
            return self
        return super().__getitem__(key)


def create_attention_masks_by_attribute_value(
    create_mask_fn: Callable,
    attribute_name: str,
    config: PreTrainedConfig,
    *args: Any,
    **kwargs: Any,
) -> AttentionMasksByAttributeValue:
    layer_patterns = set(getattr(config, "layer_types", ()))
    attention_masks = (
        _SinglePatternAttentionMasks(next(iter(layer_patterns)))
        if len(layer_patterns) == 1
        else AttentionMasksByAttributeValue()
    )
    attribute_value_to_layer_indices: dict[Hashable, list[int]] = defaultdict(list)
    for layer_idx in range(config.num_hidden_layers):
        layer_config = config.per_layer_config[layer_idx]
        attribute_value = getattr(layer_config, attribute_name)
        if attribute_value is None:
            continue

        attribute_value_to_layer_indices[attribute_value].append(layer_idx)

    past_key_values = kwargs.get("past_key_values")
    for attribute_value, layer_indices in attribute_value_to_layer_indices.items():
        layer_idx = layer_indices[0]
        if past_key_values is not None:
            updated_layer_idx = past_key_values.get_updated_kv_layer_idx(layer_indices)
            if updated_layer_idx is not None:
                layer_idx = updated_layer_idx

        layer_config = config.per_layer_config[layer_idx]
        attention_masks[attribute_value] = create_mask_fn(layer_config, *args, **kwargs, layer_idx=layer_idx)

    return attention_masks
