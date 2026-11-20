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
import itertools

from ....configuration_utils import PreTrainedConfig
from .cache_pool import CachePool
from .full_attention import FullAttentionCacheAllocator


class MLACacheAllocator(FullAttentionCacheAllocator):
    """Cache allocator for a group of MLA layers, which cache two single-head latents per token instead of regular
    keys and values: the compressed KV latent, then the rotated key. Blocks, pages and indices work like they do in
    the parent class, only the content of a page differs: each token takes a single row holding its two latents.

        [  (KV LATENT 0 | ROTATED KEY 0)  |  (KV LATENT 1 | ROTATED KEY 1)  |  ...  ]
    """

    supports_block_table = False
    rows_per_token = 1  # A single row for the two latents of each token

    def __init__(
        self,
        index: int,
        config: PreTrainedConfig,
        cache_tensor_meta: dict[int, list[tuple[torch.Size, torch.dtype, torch.device]]],
        cache_dtype: torch.dtype,
        page_size: int,
        layer_indices: list[int],
        allow_block_sharing: bool,
    ) -> None:
        """Initializes the cache allocator for a group of MLA layers."""
        # Retrieve the kv_lora_rank and qk_rope_head_dim from the cache tensor metadata
        kv_and_qk = {
            (kv_pass_shape[-1], k_rot_shape[-1])
            for (kv_pass_shape, _, _), (k_rot_shape, _, _)
            in (cache_tensor_meta[layer_idx] for layer_idx in layer_indices)
        }
        if len(kv_and_qk) != 1:
            raise ValueError(f"kv_lora_rank and qk_rope_head_dim differ between layers: found {kv_and_qk}")
        self.kv_lora_rank, self.qk_rope_head_dim = kv_and_qk.pop()

        self.row_numel = self.kv_lora_rank + self.qk_rope_head_dim
        self.cache_dtype = cache_dtype
        self._before_cache_tensor_init(
            index=index,
            layer_indices=layer_indices,
            tokens_per_page=page_size,
            bytes_per_page=self.row_numel * cache_dtype.itemsize * page_size,
            allow_block_sharing=allow_block_sharing,
        )

    def register_cache_tensor(
        self, bytes_per_sector: int, non_trash_bytes: int, cache_tensor: torch.Tensor, pool: CachePool
    ) -> None:
        """Registers the cache tensor so the allocator can use it for updates. Pages, blocks and sectors are arranged
        like they are in the parent class, but the cache is viewed as one row per token and each latent is a column
        slice of the rows."""
        self.bytes_per_sector = bytes_per_sector
        # Byte view of the whole tensor as one row per block, used to copy blocks when forking
        self._copy_view = cache_tensor.view(-1, self.bytes_per_block)

        # Reshape the cache as one row per token. The view spans the entire tensor, including the trash sectors
        total_bytes = non_trash_bytes + 2 * bytes_per_sector
        total_tokens = (total_bytes // self.bytes_per_page) * self.tokens_per_page
        cache_tensor = cache_tensor.view(self.cache_dtype)
        cache_tensor = cache_tensor[: total_tokens * self.row_numel].view(total_tokens, self.row_numel)
        self._after_cache_tensor_init(non_trash_bytes, bytes_per_sector, cache_tensor, pool)

        # Precompute the per-layer shifted views used by update(): the rows of a layer, split into the two latents
        self._latent_token_views: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        for i, layer_idx in enumerate(self.layer_indices):
            rows = self.cache_tensor[i * self.tokens_per_page :]
            self._latent_token_views[layer_idx] = (rows[:, : self.kv_lora_rank], rows[:, self.kv_lora_rank :])

    def update(
        self,
        key_states: torch.Tensor,  # the KV latents, shape [seqlen_q, 1, kv_lora_rank]
        value_states: torch.Tensor,  # the rotated keys, shape [seqlen_q, 1, qk_rope_head_dim]
        layer_idx: int,
        read_index: torch.Tensor,  # shape [seqlen_q + past_length]
        write_index: torch.Tensor,  # shape [seqlen_q]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Updates the cache with the new latents of a layer, and returns the complete latents (cached + new) for
        expansion to heads. Like in the parent class, the read-back is skipped when the layer's read index is empty."""
        kv_latent_cache, rotated_key_cache = self._latent_token_views[layer_idx]
        kv_latent_cache.index_copy_(0, write_index, key_states.flatten(1))
        rotated_key_cache.index_copy_(0, write_index, value_states.flatten(1))
        if read_index.numel() == 0:
            return key_states, value_states
        # Each latent is read on its own, so both come back contiguous: some linear layers (fp8) require it
        kv_latents = torch.index_select(kv_latent_cache, 0, read_index)
        rotated_keys = torch.index_select(rotated_key_cache, 0, read_index)
        return kv_latents.unsqueeze(1), rotated_keys.unsqueeze(1)

    def get_cache_for_block_table(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns the K and V cache views for a block table update."""
        raise NotImplementedError("Not implemented for the MLA cache allocator")
