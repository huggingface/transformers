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

from math import gcd

import torch

from ....configuration_utils import PreTrainedConfig
from .cache_allocator import CacheAllocator
from .cache_pool import CachePool


# The shape and dtype of one per-request state of a linear attention layer
StateSpec = tuple[tuple[int, ...], torch.dtype]


class LinearAttentionCacheAllocator(CacheAllocator):
    """Cache allocator for a group of linear attention layers.

    A linear attention layer does not cache the tokens it has seen: it keeps states of fixed size per request, which
    the layer declares through `linear_attention_states` (for a gated delta net, the last inputs of its causal
    convolution and its recurrent state). The allocator stores them in one page per layer, so a request owns exactly
    one block, allocated with its first tokens and overwritten in place by every forward pass. A page holds the states
    one after the other, each starting on a 128-byte boundary:

        [ STATE 0 | padding | STATE 1 | padding | ... ]

    Blocks are never shared: a state cannot be matched by content like a page of tokens can. Forks copy them.
    """

    supports_block_sharing = False
    supports_block_table = True
    layer_type = "linear_attention"
    # A page holds one state, not tokens, so it counts as one row of one token
    rows_per_token = 1
    # The indices point at the block of each sequence, not at the slot of each token
    indexes_by_sequence = True

    def __init__(
        self,
        index: int,
        config: PreTrainedConfig,
        num_key_value_heads: int,
        cache_dtype: torch.dtype,
        page_size: int,
        layer_indices: list[int],
        allow_block_sharing: bool,
        states: list[StateSpec],
        block_alignment: int = 1,
    ) -> None:
        """Initializes the cache allocator for a group of linear attention layers.

        Args:
            - config: unused, the states are declared by the layers
            - num_key_value_heads: unused, the linear attention heads are not split across tensor parallel ranks
            - cache_dtype: unused, each state declares its dtype
            - page_size: unused, a page holds one set of states rather than tokens
            - layer_indices: the indices of the layers which cache is handled by this allocator
            - allow_block_sharing: unused, states are never shared
            - states: the shape and dtype of each per-request state of a layer, in the order the layer reads and
              writes them
            - block_alignment: the block is sized to a multiple of this, so the sectors shared with the other
              allocators stay small (the sector is the LCM of all block sizes)
        """
        self.states = states
        # Each state starts on a 128-byte boundary, so every dtype can be viewed from the page bytes
        self.state_offsets = []
        bytes_per_page = 0
        for shape, dtype in states:
            self.state_offsets.append(bytes_per_page)
            bytes_per_page += (torch.Size(shape).numel() * dtype.itemsize + 127) // 128 * 128
        # Pages are padded so a block (one page per layer) is a multiple of the alignment
        page_alignment = block_alignment // gcd(block_alignment, len(layer_indices))
        bytes_per_page = (bytes_per_page + page_alignment - 1) // page_alignment * page_alignment
        self._before_cache_tensor_init(
            index=index,
            layer_indices=layer_indices,
            tokens_per_page=1,
            bytes_per_page=bytes_per_page,
            allow_block_sharing=allow_block_sharing,
        )

    def register_cache_tensor(
        self, bytes_per_sector: int, non_trash_bytes: int, cache_tensor: torch.Tensor, pool: CachePool
    ) -> None:
        """Registers the cache tensor so the allocator can use it for updates. For a linear attention cache allocator
        with 2 layers and 2 states per layer (a conv state C and a recurrent state R), the cache is arranged this way
        (L0 = Layer 0):

        [ L0 C | L0 R | L1 C | L1 R | L0 C | L0 R | L1 C | L1 R ]
        [ --- PAGE 0 -- | --- PAGE 1 -- | --- PAGE 2 -- | --- PAGE 3 -- ]
        [ ---------- BLOCK 0 ---------- | ---------- BLOCK 1 ---------- ]
        [ -------------------------- SECTOR -------------------------- ]

        Here, a sector is dimensionned to hold 2 linear attention blocks.
        """
        self.bytes_per_sector = bytes_per_sector
        # Byte view of the whole tensor as one row per block, used to copy blocks when forking
        self._copy_view = cache_tensor.view(-1, self.bytes_per_block)

        # The views span the entire tensor, including the two leading trash sectors
        total_bytes = non_trash_bytes + 2 * bytes_per_sector
        num_blocks = total_bytes // self.bytes_per_block
        pages = cache_tensor[: num_blocks * self.bytes_per_block].view(num_blocks, self.pages_per_block, -1)
        self._state_views: list[torch.Tensor] = []
        for (shape, dtype), offset in zip(self.states, self.state_offsets):
            state_bytes = pages[:, :, offset : offset + torch.Size(shape).numel() * dtype.itemsize]
            self._state_views.append(state_bytes.view(dtype).view(num_blocks, self.pages_per_block, *shape))
        self._after_cache_tensor_init(non_trash_bytes, bytes_per_sector, cache_tensor, pool)

        # States are indexed by block: the read trash is the first block of sector 0 (zeroed, never written) and the
        # write trash is the first block of sector 1 (written, never read)
        self.read_trash_index = 0
        self.write_trash_index = self.blocks_per_sector

    # _________________________________________________ BLOCK LEVEL __________________________________________________ #

    def needs_new_blocks(self, request_id: str, past_length: int, query_length: int) -> int:
        """Returns the number of new blocks needed for a given request: one block on its first tokens, none after."""
        return 0 if request_id in self.block_table else 1

    def allocate_cache_to_request(self, request_id: str, past_length: int, query_length: int) -> None:
        """Allocates the block holding the request's states, if it does not have one yet."""
        if request_id not in self.block_table:
            self.block_table[request_id] = self.pool.get_free_blocks(self.index, 1)

    # ______________________________________________ INPUT PREPARATION _______________________________________________ #

    def get_seqlen_k(self, past_length: int, query_length: int) -> int:
        """A linear attention layer reads no keys from the cache: it only attends to the new tokens."""
        return query_length

    def get_read_indices(self, request_id: str, past_length: int, query_length: int) -> list[int]:
        """Returns the block to read the request's states from: the read trash, which holds zeros, when the request
        has no past, its own block otherwise. The list holds one entry per request rather than one per token."""
        if past_length == 0:
            return [self.read_trash_index]
        return self._get_block(request_id)

    def get_write_indices(self, request_id: str, past_length: int, query_length: int) -> list[int]:
        """Returns the block to write the request's states to. The list holds one entry per request."""
        return self._get_block(request_id)

    def fill_block_table(
        self, request_id: str, past_length: int, query_length: int, block_table: torch.Tensor
    ) -> None:
        """Fills the request's row of the kernel block table with its single block."""
        block_table[:1].copy_(torch.tensor(self._get_block(request_id), dtype=block_table.dtype), non_blocking=True)

    def _get_block(self, request_id: str) -> list[int]:
        block = self.block_table.get(request_id)
        if block is None:
            raise ValueError(f"No block table found for request {request_id}")
        return block

    # ________________________________________________ RUNTIME UPDATE ________________________________________________ #

    def state_views(self, layer_idx: int) -> list[torch.Tensor]:
        """The states of all blocks for the given layer, one view of shape [num_blocks, *shape] per declared state."""
        page = self.layer_indices.index(layer_idx)
        return [view[:, page] for view in self._state_views]

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        read_index: torch.Tensor,
        write_index: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError("Linear attention layers hold states, not key and value states")

    def get_cache_for_block_table(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError("Linear attention layers hold states, not key and value states")
