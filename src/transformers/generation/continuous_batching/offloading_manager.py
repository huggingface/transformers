# Copyright 2026 The HuggingFace Inc. team
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
"""Centralized offloading logic for continuous batching.

Handles two offloading strategies when the GPU KV cache is full:
  1. CPU offloading: copy the KV cache to a pre-allocated pinned CPU buffer, preserving exact request state.
  2. Soft reset: discard the KV cache and re-prefill from scratch when the request is re-scheduled. This incurs no data
    transfer overhead, but we need to re-run prefill over all intial + generated tokens (so more compute overhead).

The CPU swap pool is a static set of pinned tensors allocated once at init (like vLLM/SGLang). Blocks are tracked
with a simple free set — no dynamic allocation or deallocation of tensors ever happens at runtime.
"""

from collections import deque
from contextlib import nullcontext

import torch

from ...utils import is_psutil_available
from .cache import PagedAttentionCache
from .requests import FutureRequestState, RequestState, RequestStatus, logger
from .scheduler import Scheduler


class OffloadingManager:
    """Manages request offloading and restoration for continuous batching.

    Owns a static CPU swap pool (pre-allocated pinned tensors mirroring the GPU cache layout), performs GPU↔CPU block
    copies, decides between CPU offloading and soft reset, and ensures cleanup on cancellation/failure/reset.
    """

    def __init__(
        self,
        cache: PagedAttentionCache,
        scheduler: Scheduler,
        cpu_offload_space_gib: float | None,
        safety_threshold: float,
        compute_stream: torch.cuda.Stream | None,
    ) -> None:
        self.cache = cache
        self.scheduler = scheduler
        # All offloading transfers run on the compute stream (stream-ordered, like the fork copy path)
        self._compute_stream = compute_stream

        # Bookkeeping defaults, valid whether or not the pool is allocated
        self._cpu_key_cache: list[torch.Tensor] = []
        self._cpu_value_cache: list[torch.Tensor] = []
        self._gpu_key_views: list[torch.Tensor] = []
        self._gpu_value_views: list[torch.Tensor] = []
        self._free_cpu_blocks: deque[int] = deque()
        self._request_id_to_cpu_blocks: dict[str, list[int]] = {}
        self._request_id_to_group_block_counts: dict[str, list[int]] = {}

        # Compute the size of the CPU swap pool in blocks
        self._num_cpu_blocks = self._compute_num_cpu_blocks(cpu_offload_space_gib, safety_threshold)
        offloading_enabled = cpu_offload_space_gib is not None and cpu_offload_space_gib > 0
        if self._num_cpu_blocks == 0:
            if offloading_enabled:
                logger.warning(
                    f"cpu_offload_space={cpu_offload_space_gib:.1f} GiB is too small for even one block. "
                    "No CPU offloading."
                )
            return None

        # Allocate the CPU swap pool
        cpu_cache_shape = (self._num_cpu_blocks, cache.block_size, cache.num_key_value_heads, cache.head_dim)
        for _ in cache.key_cache:
            self._cpu_key_cache.append(torch.empty(cpu_cache_shape, dtype=cache.dtype, pin_memory=True))
            self._cpu_value_cache.append(torch.empty(cpu_cache_shape, dtype=cache.dtype, pin_memory=True))

        # Pre-view the GPU cache tensors as block-shaped so the hot copy paths avoid per-op .view() calls
        block_shape = (-1, cache.block_size, cache.num_key_value_heads, cache.head_dim)
        for k_cache, v_cache in zip(cache.key_cache, cache.value_cache):
            self._gpu_key_views.append(k_cache.view(*block_shape))
            self._gpu_value_views.append(v_cache.view(*block_shape))

        # FIFO order favors contiguity when blocks are returned in bulk
        self._free_cpu_blocks = deque(range(self._num_cpu_blocks))

        # Reusable int64 scratch for cpu_ids / gpu_ids (bounded by _num_cpu_blocks on both paths).
        # int64 is required by index_copy_ / index_select.
        self._cpu_ids_scratch = torch.empty(self._num_cpu_blocks, dtype=torch.long, pin_memory=True)
        self._gpu_ids_scratch = torch.empty(self._num_cpu_blocks, dtype=torch.long, device=cache.device)

        # Log the size of the CPU swap pool
        cache_tensor = self._cpu_key_cache[0]
        size_in_bytes = 2 * cache_tensor.numel() * cache_tensor.element_size() * len(cache.key_cache)
        logger.info(
            f"CPU swap pool initialized: {self._num_cpu_blocks} blocks ({size_in_bytes / (1024**3):.2f} GiB pinned)"
        )

    def _compute_num_cpu_blocks(self, cpu_offload_space_gib: float | None, safety_threshold: float) -> int:
        """Returns the number of blocks that can fit in the CPU swap pool."""
        # Compute the CPU pool size in bytes
        offload_bytes = int(cpu_offload_space_gib * (1024**3)) if cpu_offload_space_gib is not None else None

        # Determine the maximum number of bytes that can be offloaded based on the safety threshold
        if is_psutil_available():
            import psutil

            total_ram = psutil.virtual_memory().available
            max_bytes = int(total_ram * safety_threshold)
        else:
            max_bytes = None

        # If both the request number of bytes and its limit are not None, we just clamp one to the other
        if offload_bytes is not None and max_bytes is not None:
            if offload_bytes > max_bytes:
                clamped_gib = max_bytes / (1024**3)
                logger.warning(
                    f"cpu_offload_space={cpu_offload_space_gib:.1f} GiB exceeds {safety_threshold:.0%} of total RAM "
                    f"({total_ram / (1024**3):.1f} GiB). Clamping to {clamped_gib:.1f} GiB."
                )
                offload_bytes = max_bytes
        # Else if the max is None, throw a warning and accept the requested number of bytes as is
        elif offload_bytes is not None:
            logger.warning(
                "psutil is not available — cpu_offload_space_safety_threshold cannot be enforced. "
                "Install psutil to enable the safety cap."
            )
        # Else if the requested number of bytes is None, we use the max number of bytes as the requested number of bytes
        elif max_bytes is not None:
            offload_bytes = max_bytes
            logger.warning(f"Auto-sizing CPU swap pool from safety threshold: {max_bytes / (1024**3):.2f} GiB.")
        # Otherwise, it means the pool was supposed to be sized using psutil but it is not available
        else:
            raise ImportError(
                "cpu_offload_space=None requires psutil to auto-size the CPU swap pool. Install psutil or pass an "
                "explicit GiB value."
            )

        # Compute how many blocks fit in CPU pool
        bytes_per_block = (
            2                                 # one for key, one for value
            * len(self.cache.key_cache)       # number of layers in a layer group
            * self.cache.block_size           # block size
            * self.cache.num_key_value_heads  # number of key value heads
            * self.cache.head_dim             # head dimension
            * self.cache.dtype.itemsize       # data type size in bytes
        )  # fmt: skip
        if bytes_per_block == 0:
            raise ValueError("The number of bytes per block is 0. This is not possible.")
        return offload_bytes // bytes_per_block

    def _stream_ctx(self):
        """Returns a context manager that runs enclosed ops on the compute stream, or a no-op when none is set."""
        return torch.cuda.stream(self._compute_stream) if self._compute_stream is not None else nullcontext()

    def offload_one_request(self) -> None:
        """Offload one active request to make room in the GPU cache. Tries CPU offloading first; if the pool is full,
        falls back to the legacy soft reset."""
        scheduler = self.scheduler
        request_id, state = scheduler.pop_request_to_evict()
        logger.info(
            f"Offloading request {request_id} with {len(state.initial_tokens)} initial tokens and "
            f"{len(state.generated_tokens)} generated tokens."
        )

        # Try CPU offloading first, if it fails, we soft reset the request
        offloaded_to_cpu = self._offload_to_cpu(request_id, state)
        if offloaded_to_cpu:
            # We set the allocated blocks to 0 so the scheduler re-allocates all blocks using position_offset.
            state.allocated_blocks = 0
            # DECODING requests have empty remaining_prefill_tokens, so we use tokens_to_process as a placeholder
            # so the scheduler has at least 1 token to schedule and enters the allocation path.
            if state._status == RequestStatus.DECODING:
                state.remaining_prefill_tokens = state.tokens_to_process[:]
            # Here, the new state is the same as the old one, but with the status set to PENDING. We bypass the setter
            # to avoid the lifespan bookeeping and the associated warning
            state._status = RequestStatus.PENDING
            new_state = state
            logger.debug(f"Offloaded request {request_id} to CPU: {len(self._free_cpu_blocks)} free blocks remaining.")
        else:
            new_state = state.create_equivalent_initial_request()
            state._status = RequestStatus.FINISHED
            logger.debug(f"Soft reset request {request_id}.")

        scheduler.finish_request(request_id)
        scheduler.add_waiting_request(new_state)
        scheduler.block_new_requests = True

    def restore_scheduled_requests(self, requests_in_batch: list[FutureRequestState]) -> None:
        """Restore KV caches from CPU for any CPU-offloaded requests in the scheduled batch. Indices are accumulated
        per group across all requests, then copied in one batched operation per layer."""
        cache = self.cache
        all_cpu_indices: list[int] = []
        all_gpu_indices: list[int] = []

        for future_state in requests_in_batch:
            # Skip state that are not CPU-offloaded
            state = future_state.state
            if not state.is_cpu_offloaded:
                continue
            # TODO: if the H2D copy below raises, already-popped entries leak (never returned to _free_cpu_blocks)
            # Accumulate CPU indices for this request
            cpu_indices = self._request_id_to_cpu_blocks.pop(state.request_id)
            group_counts = self._request_id_to_group_block_counts.pop(state.request_id)
            all_cpu_indices.extend(cpu_indices)
            # Accumulate GPU indices for this request, but since there may be extra block due to re-allocation, slice to
            # match the number of blocks offloaded.
            max_allocated_blocks = 0
            for group_idx, n in enumerate(group_counts):
                gpu_blocks = cache.group_cache_managers[group_idx].block_table.get(state.request_id, [])
                all_gpu_indices.extend(gpu_blocks[:n])
                max_allocated_blocks = max(max_allocated_blocks, n)
            # Restore the state to non-offloaded state
            state.is_cpu_offloaded = False
            state.allocated_blocks = max_allocated_blocks  # ensures re-allocation is accounted for
            # Prefix sharing: restored blocks will be re-hashed during the next update
            if cache.allow_block_sharing:
                future_state.complete_blocks += state.position_offset // cache.block_size
            logger.debug(
                f"Restored CPU-offloaded request {state.request_id} with {len(state.initial_tokens)} prefill tokens "
                f"and {len(state.generated_tokens)} generated tokens."
            )

        # Early return if there are no copy to perform
        if not all_cpu_indices:
            return None

        # Single batched copy for all requests (still, one copy per layer)
        n = len(all_cpu_indices)
        cpu_ids = self._cpu_ids_scratch[:n]
        gpu_ids = self._gpu_ids_scratch[:n]
        cpu_ids.copy_(torch.as_tensor(all_cpu_indices, dtype=torch.long))  # cpu op, not in the stream
        with self._stream_ctx():
            gpu_ids.copy_(torch.as_tensor(all_gpu_indices, dtype=torch.long))
            for cpu_k, gpu_k in zip(self._cpu_key_cache, self._gpu_key_views):
                device_side_cpu_blocks = cpu_k.index_select(0, cpu_ids).to(gpu_k.device)
                gpu_k.index_copy_(0, gpu_ids, device_side_cpu_blocks)
            for cpu_v, gpu_v in zip(self._cpu_value_cache, self._gpu_value_views):
                device_side_cpu_blocks = cpu_v.index_select(0, cpu_ids).to(gpu_v.device)
                gpu_v.index_copy_(0, gpu_ids, device_side_cpu_blocks)
        self._free_cpu_blocks.extend(all_cpu_indices)

    def free_request_cpu_cache(self, state: RequestState) -> None:
        """Free CPU blocks for a single request (e.g., on cancellation)."""
        if state.is_cpu_offloaded:
            self._return_cpu_blocks(state.request_id)
            state.is_cpu_offloaded = False

    def free_all_waiting_cpu_caches(self) -> None:
        """Free all CPU-offloaded caches in the waiting queue (e.g., on fail_all or reset)."""
        for state in self.scheduler.waiting_requests.values():
            self.free_request_cpu_cache(state)

    def reset(self) -> None:
        """Reset CPU offloading state for a new generation session."""
        self.free_all_waiting_cpu_caches()
        self._request_id_to_cpu_blocks.clear()
        self._request_id_to_group_block_counts.clear()
        self._free_cpu_blocks = deque(range(self._num_cpu_blocks))

    def _offload_to_cpu(self, request_id: str, state: RequestState) -> bool:
        """Copy a request's KV cache blocks from GPU to the static CPU swap pool. Returns True on success, False if
        the pool is full."""

        # Get the indices to offload from
        gpu_indices = []
        group_block_counts = []
        for cm in self.cache.group_cache_managers:
            blocks = cm.block_table.get(request_id, [])
            gpu_indices.extend(blocks)
            group_block_counts.append(len(blocks))

        # No CPU offloading if there are no blocks to offload or not enough free blocks in the CPU swap pool
        total_gpu_blocks = len(gpu_indices)
        if total_gpu_blocks == 0 or len(self._free_cpu_blocks) < total_gpu_blocks:
            return False

        # Reserve CPU blocks from the free pool
        cpu_indices = [self._free_cpu_blocks.popleft() for _ in range(total_gpu_blocks)]

        # Offload using the compute stream so it does not interfere with current generation
        cpu_ids = self._cpu_ids_scratch[:total_gpu_blocks]
        gpu_ids = self._gpu_ids_scratch[:total_gpu_blocks]
        cpu_ids.copy_(torch.as_tensor(cpu_indices, dtype=torch.long))  # cpu op, not in the stream
        with self._stream_ctx():
            gpu_ids.copy_(torch.as_tensor(gpu_indices, dtype=torch.long))
            for cpu_key_cache, gpu_key_view in zip(self._cpu_key_cache, self._gpu_key_views):
                host_side_gpu_blocks = gpu_key_view.index_select(0, gpu_ids).to(cpu_key_cache.device)
                cpu_key_cache.index_copy_(0, cpu_ids, host_side_gpu_blocks)
            for cpu_value_cache, gpu_value_view in zip(self._cpu_value_cache, self._gpu_value_views):
                host_side_gpu_blocks = gpu_value_view.index_select(0, gpu_ids).to(cpu_value_cache.device)
                cpu_value_cache.index_copy_(0, cpu_ids, host_side_gpu_blocks)
            # TODO: async path with a preallocated pinned scratch + non_blocking=True; current .to() materializes
            # an unpinned intermediate per layer, so _stream_ctx() does not actually overlap.

        # No explicit sync needed: finish_request is logical, and the next forward pass serializes on the same stream.
        self._request_id_to_cpu_blocks[request_id] = cpu_indices
        self._request_id_to_group_block_counts[request_id] = group_block_counts
        state.is_cpu_offloaded = True
        return True

    def _return_cpu_blocks(self, request_id: str) -> tuple[list[int], list[int]]:
        """Return CPU blocks to the free pool without copying anything."""
        cpu_ids = self._request_id_to_cpu_blocks.pop(request_id)
        group_counts = self._request_id_to_group_block_counts.pop(request_id)
        self._free_cpu_blocks.extend(cpu_ids)
        return cpu_ids, group_counts
