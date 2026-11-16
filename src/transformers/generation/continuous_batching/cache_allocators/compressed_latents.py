# Copyright 2026 The HuggingFace Inc. team.
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

import torch

from ....configuration_utils import PreTrainedConfig
from .full_attention import PackedAttentionCacheAllocator


class MLACacheAllocator(PackedAttentionCacheAllocator):
    """Cache allocator for a group of MLA layers, which cache two single-head latents per token instead of regular
    keys and values: the compressed KV latent, then the rotated key."""

    supports_block_sharing = True
    supports_block_table = False  # TODO: feed the pages to an MLA decode kernel to support the block table

    def __init__(
        self,
        index: int,
        config: PreTrainedConfig,
        num_key_value_heads: int,
        cache_dtype: torch.dtype,
        page_size: int,
        layer_indices: list[int],
        allow_block_sharing: bool,
    ) -> None:
        """Initializes the cache allocator for a group of MLA layers. The arguments are the ones of the parent class,
        `num_key_value_heads` is unused since the latents are cached before they are expanded to heads."""
        super().__init__(
            index=index,
            field_shapes=[(1, config.kv_lora_rank), (1, config.qk_rope_head_dim)],
            cache_dtype=cache_dtype,
            page_size=page_size,
            layer_indices=layer_indices,
            allow_block_sharing=allow_block_sharing,
        )

    def update(
        self,
        key_states: torch.Tensor,  # shape [seqlen_q, 1, kv_lora_rank]
        value_states: torch.Tensor,  # shape [seqlen_q, 1, qk_rope_head_dim]
        layer_idx: int,
        read_index: torch.Tensor,  # shape [seqlen_q + past_length]
        write_index: torch.Tensor,  # shape [seqlen_q]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Updates the cache with the new latents of a layer, and returns the complete latents (cached + new) for
        expansion to heads."""
        return self.update_fields((key_states, value_states), layer_idx, read_index, write_index)
