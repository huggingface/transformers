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


def create_attention_masks_by_attribute_value(
    create_mask_fn: Callable,
    attribute_name: str,
    config: PreTrainedConfig,
    *args: Any,
    **kwargs: Any,
) -> AttentionMasksByAttributeValue:
    attention_masks = AttentionMasksByAttributeValue()
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
            representative_layer_idx = past_key_values.get_representative_kv_layer_idx(layer_indices)
            if representative_layer_idx is not None:
                layer_idx = representative_layer_idx

        layer_config = config.per_layer_config[layer_idx]
        attention_masks[attribute_value] = create_mask_fn(layer_config, *args, **kwargs, layer_idx=layer_idx)

    return attention_masks
