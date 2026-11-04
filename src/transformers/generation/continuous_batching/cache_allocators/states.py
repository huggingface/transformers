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

from typing import NamedTuple

import torch

from .cache_allocator import CacheAllocator
from .cache_pool import CachePool


class StateSpec(NamedTuple):
    """A state a layer keeps per request, as it declares it through `cache_states`: its shape, its dtype (None for the
    cache dtype) and the value it holds before the request's first forward pass."""

    shape: tuple[int, ...]
    dtype: torch.dtype | None
    fill: int | float = 0


class StateCacheAllocator(CacheAllocator):
    """Cache allocator for the states the layers keep per request.

    Some layers keep no tokens but states of fixed size per request, which they declare by name through
    `cache_states`: a linear attention layer keeps the last inputs of its causal convolution and its recurrent state.
    Any layer can declare states, whatever its attention type, and the layers need not declare the same ones. The
    allocator stores the states of all those layers in one block per request, allocated with its first tokens and
    overwritten in place by every forward pass. Inside a block, each layer's states follow each other, each starting on
    a 128-byte boundary:

        [ L0 STATE A | L0 STATE B | L1 STATE A | L1 STATE B | L1 STATE C | ... ]

    Blocks are never shared: a state cannot be matched by content like a page of tokens can. Forks copy them.
    """

    supports_block_sharing = False
    supports_block_table = True
    layer_type = "linear_attention"
    # The indices point at the block of each sequence, not at the slot of each token
    indexes_by_sequence = True

    def __init__(
        self,
        index: int,
        states: dict[int, dict[str, tuple[tuple[int, ...], torch.dtype, int | float]]],
        block_alignment: int = 1,
    ) -> None:
        """Initializes the cache allocator for the states of the given layers.

        Args:
            - states: the states each layer keeps per request, by layer index then by name, as `(shape, dtype, fill)`
            - block_alignment: the block is sized to a multiple of this, so the sectors shared with the other
              allocators stay small (the sector is the LCM of all block sizes)
        """
        self.states = states
        # The offset of each state in the block, by layer index then by name
        self.state_offsets: dict[int, dict[str, int]] = {}
        bytes_per_block = 0
        for layer_idx, layer_states in states.items():
            self.state_offsets[layer_idx] = {}
            for name, (shape, dtype, _) in layer_states.items():
                self.state_offsets[layer_idx][name] = bytes_per_block
                bytes_per_block += (torch.Size(shape).numel() * dtype.itemsize + 127) // 128 * 128
        bytes_per_block = (bytes_per_block + block_alignment - 1) // block_alignment * block_alignment
        self._before_cache_tensor_init(
            index=index,
            layer_indices=list(states),
            tokens_per_page=1,
            bytes_per_block=bytes_per_block,
            allow_block_sharing=False,
        )

    def register_cache_tensor(
        self, bytes_per_sector: int, non_trash_bytes: int, cache_tensor: torch.Tensor, pool: CachePool
    ) -> None:
        """Registers the cache tensor so the allocator can use it for updates: one view per state of each layer, of
        shape [num_blocks, *shape], over the state's bytes in every block."""
        self.bytes_per_sector = bytes_per_sector
        # Byte view of the whole tensor as one row per block, used to copy blocks when forking
        self._copy_view = cache_tensor.view(-1, self.bytes_per_block)

        # The views span the entire tensor, including the two leading trash sectors
        total_bytes = non_trash_bytes + 2 * bytes_per_sector
        num_blocks = total_bytes // self.bytes_per_block
        blocks = cache_tensor[: num_blocks * self.bytes_per_block].view(num_blocks, self.bytes_per_block)
        self._state_views: dict[int, dict[str, torch.Tensor]] = {}
        for layer_idx, layer_states in self.states.items():
            self._state_views[layer_idx] = {}
            for name, (shape, dtype, _) in layer_states.items():
                offset = self.state_offsets[layer_idx][name]
                state_bytes = blocks[:, offset : offset + torch.Size(shape).numel() * dtype.itemsize]
                self._state_views[layer_idx][name] = state_bytes.view(dtype).view(num_blocks, *shape)
        # With one token per page, the indices address blocks: the read trash is the first block of sector 0 (zeroed,
        # never written) and the write trash is the first block of sector 1 (written, never read)
        self._after_cache_tensor_init(non_trash_bytes, bytes_per_sector, cache_tensor, pool)

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
        """A layer keeping states reads no keys from the cache: it only attends to the new tokens."""
        return query_length

    def get_read_indices(self, request_id: str, past_length: int, query_length: int) -> list[int]:
        """Returns the block to read the request's states from: the read trash when the request has no past, its own
        block otherwise. The list holds one entry per request rather than one per token."""
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

    def state_views(self, layer_idx: int) -> dict[str, torch.Tensor]:
        """The states of all blocks for the given layer, by name, each of shape [num_blocks, *shape]."""
        return self._state_views[layer_idx]

    def update(
        self,
        rows: dict[str, torch.Tensor],
        layer_idx: int,
        read_index: torch.Tensor,
        write_index: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        raise NotImplementedError("States are kept per request, not per token")

    def get_cache_for_block_table(self, layer_idx: int) -> dict[str, torch.Tensor]:
        raise NotImplementedError("States are kept per request, not per token")
