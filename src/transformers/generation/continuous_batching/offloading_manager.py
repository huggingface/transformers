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
    transfer overhead, but we need to re-run prefill over all initial + generated tokens (so more compute overhead).

The CPU swap pool is a single pinned tensor allocated once at init (like vLLM/SGLang), mirroring the GPU cache
geometry: it is tracked by its own CachePool with the same sector size and per-allocator blocks, but no trash
sectors.
"""

import logging
from contextlib import nullcontext
from math import ceil

import torch

from ...utils import is_psutil_available
from .cache import PagedAttentionCache
from .cache_allocators import CachePool
from .distributed import DistributedHelper
from .requests import FutureRequestState, RequestState, RequestStatus, logger
from .scheduler import Scheduler


def contiguous_runs(indices: list[int]) -> list[tuple[int, int, int]]:
    """Groups an index list into (start_index, offset, length) runs of consecutive values, so scattered block copies
    can be performed as a few slice copies."""
    runs = []
    if not indices:
        return runs
    start = prev = indices[0]
    offset = 0
    for i in range(1, len(indices)):
        if indices[i] != prev + 1:
            runs.append((start, offset, i - offset))
            start, offset = indices[i], i
        prev = indices[i]
    runs.append((start, offset, len(indices) - offset))
    return runs


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
        distributed_helper: DistributedHelper,
    ) -> None:
        self.cache = cache
        self.scheduler = scheduler
        # All offloading transfers run on the compute stream (stream-ordered, like the fork copy path)
        self._compute_stream = compute_stream

        # Bookkeeping defaults, valid whether or not the pool is allocated
        self._cpu_pool: CachePool | None = None
        self._cpu_views: dict[str, torch.Tensor] = {}
        self._request_id_to_cpu_blocks: dict[str, dict[str, list[int]]] = {}

        # Compute the size of the CPU swap pool in sectors
        num_cpu_sectors = self._compute_num_cpu_sectors(cpu_offload_space_gib, safety_threshold)
        num_cpu_sectors = torch.tensor(num_cpu_sectors, dtype=torch.int32, device="cpu")
        self._num_cpu_sectors = int(distributed_helper.tp_all_reduce_min(num_cpu_sectors, on_cpu=True).item())

        offloading_enabled = cpu_offload_space_gib is not None and cpu_offload_space_gib > 0
        if self._num_cpu_sectors == 0:
            if offloading_enabled:
                logger.warning(
                    f"cpu_offload_space={cpu_offload_space_gib:.1f} GiB is too small for even one sector. "
                    "No CPU offloading."
                )
            return None

        # Allocate the pinned CPU tensor. It uses the same sector-based system as the GPU cache, but no trash sectors.
        cpu_cache_size = self._num_cpu_sectors * cache.bytes_per_sector
        self._cpu_tensor = torch.empty(cpu_cache_size, dtype=torch.uint8, pin_memory=True)
        self._cpu_pool = CachePool(self._num_cpu_sectors, len(cache.cache_allocators), num_reserved_sectors=0)
        for name, allocator in cache.cache_allocators.items():
            self._cpu_pool.set_blocks_per_sector(allocator.index, allocator.blocks_per_sector)
            self._cpu_views[name] = self._cpu_tensor.view(-1, allocator.bytes_per_block)
        logger.info(
            f"CPU swap pool initialized: {self._num_cpu_sectors} sectors "
            f"({self._cpu_tensor.numel() / 1024**3:.2f} GiB pinned)"
        )

    def _compute_num_cpu_sectors(self, cpu_offload_space_gib: float | None, safety_threshold: float) -> int:
        """Returns the number of sectors that can fit in the CPU swap pool."""
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

        # The CPU pool mirrors the GPU sector geometry, so its capacity is expressed in whole sectors
        return offload_bytes // self.cache.bytes_per_sector

    def _stream_ctx(self):
        """Returns a context manager that runs enclosed ops on the compute stream, or a no-op when none is set."""
        return torch.cuda.stream(self._compute_stream) if self._compute_stream is not None else nullcontext()

    def offload_requests(self) -> bool:
        """Evict enough active requests that, at the next batch, every remaining starved request can allocate the
        cache it needs. Offloaded requests are taken from the starved requests reported by the scheduler, newest first,
        so the batch that was just scheduled is never touched. Offloaded requests are copied to the CPU swap pool when
        they fit; the others are soft reset: their cache is freed and an equivalent request is requeued. Returns a
        boolean indicating if offloading was successful."""
        scheduler = self.scheduler
        starved = scheduler.starved_requests

        # Before offloading anything, we evict all cached blocks, and try re-scheduling if there was actual eviction.
        # It's better to un-cache block that may be used in the future than offloading requests that are active now.
        cached_blocks_evicted = self.cache.evict_cached_blocks()
        if not starved:
            return cached_blocks_evicted
        # Then, we count the blocks needed for each request (computed once but used multiple times)
        list_blocks_needed = [
            {
                name: allocator.needs_new_blocks(state.request_id, state.current_len(), request_len)
                for name, allocator in self.cache.cache_allocators.items()
            }
            for state, request_len in scheduler.starved_requests
        ]

        # Offload request until all the remaining starved request can be scheduled
        num_active = len(scheduler.active_requests)
        offloaded: list[RequestState] = []
        offloaded_block_tables: dict[str, dict[str, list[int]]] = {}
        while starved and num_active - len(offloaded) > 1:
            # Stop when all the remaining starved requests can be scheduled
            num_storable = self.cache.count_storable_requests(list_blocks_needed)
            if num_storable == len(starved):
                break

            # Otherwise, we offload the oldest starved request
            state, _ = starved.pop()
            list_blocks_needed.pop()
            offloaded.append(state)
            # Copy the offloaded block tables before freeing the blocks (which destroys the block tables)
            offloaded_block_tables[state.request_id] = {
                name: allocator.block_table.get(state.request_id, [])[:]
                for name, allocator in self.cache.cache_allocators.items()
            }
            self.cache.free_blocks(state.request_id)

        # Sometimes, no request is offloaded because evicting cached blocks has made enough room on its own
        if not offloaded:
            return True

        # Copy as many victims as fit in the CPU pool, in one batched copy per allocator
        cpu_offloaded = self._offload_to_cpu(offloaded, offloaded_block_tables)

        # Requeue victims oldest-first so they will become active again in (roughly) their original order
        offloaded.reverse()
        for state in offloaded:
            request_id = state.request_id
            if request_id in cpu_offloaded:
                # We set the allocated blocks to 0 so the scheduler re-allocates all blocks using position_offset.
                state.allocated_blocks = 0
                if state._status == RequestStatus.DECODING:
                    # In async mode, a request can be offloaded for preparation of batch N+1 while still in flight in
                    # batch N. Since the token generated by batch N will be discarded at the update, we roll back one
                    # token to avoid restoring with fake information (placeholder token or partial KV).
                    if state.position_offset == len(state.initial_tokens) + len(state.generated_tokens):
                        state.position_offset -= 1
                        last_true_token = (state.generated_tokens or state.initial_tokens)[-1]
                        state.remaining_prefill_tokens = [last_true_token]
                    # Otherwise the next token is known: re-processing it on restore continues the request exactly.
                    else:
                        state.remaining_prefill_tokens = state.tokens_to_process[:]
                # The new state is the same as the old one, but with the status set to PENDING. We bypass the setter
                # to avoid the lifespan bookkeeping and the associated warning
                state._status = RequestStatus.PENDING
                new_state = state
            else:
                new_state = state.create_equivalent_initial_request()
                state._status = RequestStatus.FINISHED
            scheduler.finish_request(request_id)
            scheduler.add_waiting_request(new_state)

        scheduler.block_new_requests = True
        if logger.isEnabledFor(logging.INFO):
            logger.info(
                f"Offloaded {len(offloaded)} requests ({len(cpu_offloaded)} to CPU, {len(offloaded) - len(cpu_offloaded)} "
                f"soft reset): {len(starved)} starved requests remain."
            )
        return True

    def restore_scheduled_requests(self, requests_in_batch: list[FutureRequestState]) -> None:
        """Restore KV caches from CPU for any CPU-offloaded requests in the scheduled batch. The scheduler has already
        re-allocated GPU blocks for them, since they came back as pending requests with an empty block table and a
        preserved position_offset."""
        if self._cpu_pool is None:
            return None
        all_cpu_ids: dict[str, list[int]] = {name: [] for name in self.cache.cache_allocators}
        all_gpu_ids: dict[str, list[int]] = {name: [] for name in self.cache.cache_allocators}

        for future_state in requests_in_batch:
            state = future_state.state
            if not state.is_cpu_offloaded:
                continue
            # TODO: if the H2D copy below raises, already-popped entries leak (never returned to the CPU pool)
            cpu_blocks = self._request_id_to_cpu_blocks.pop(state.request_id)
            for name, cpu_ids in cpu_blocks.items():
                allocator = self.cache.cache_allocators[name]
                # The request may have been re-allocated more blocks than were offloaded: slice to match
                gpu_blocks = allocator.block_table.get(state.request_id, [])
                num_blocks = min(len(cpu_ids), len(gpu_blocks))
                all_gpu_ids[name].extend(gpu_blocks[:num_blocks])
                all_cpu_ids[name].extend(cpu_ids[:num_blocks])
                self._cpu_pool.free_blocks(allocator.index, cpu_ids)
            state.is_cpu_offloaded = False
            # Prefix sharing: the restored blocks are fresh and unhashed, so the next update re-hashes them, which
            # also de-duplicates them against any identical block still living in the cache
            if self.cache.use_prefix_sharing:
                for name, allocator in self.cache.cache_allocators.items():
                    restored_blocks = state.position_offset // allocator.tokens_per_page
                    if restored_blocks:
                        new_count = future_state.complete_blocks.get(name, 0) + restored_blocks
                        future_state.complete_blocks[name] = new_count
            logger.debug(
                f"Restored CPU-offloaded request {state.request_id} with {len(state.initial_tokens)} prefill tokens "
                f"and {len(state.generated_tokens)} generated tokens."
            )

        # Early return if there is no copy to perform
        if not any(all_cpu_ids.values()):
            return None

        # Single batched copy per allocator: a few non-blocking slice copies into a staging tensor, then one scatter
        # into the cache. All stream-ordered, so the host never waits and the next forward sees the data.
        with self._stream_ctx():
            for name, allocator in self.cache.cache_allocators.items():
                cpu_ids = all_cpu_ids[name]
                if not cpu_ids:
                    continue
                cpu_view = self._cpu_views[name]
                staging = torch.empty(
                    (len(cpu_ids), allocator.bytes_per_block), dtype=torch.uint8, device=self.cache.device
                )
                for start, offset, length in contiguous_runs(cpu_ids):
                    staging[offset : offset + length].copy_(cpu_view[start : start + length], non_blocking=True)
                gpu_ids = torch.as_tensor(all_gpu_ids[name], dtype=torch.long).to(self.cache.device, non_blocking=True)
                allocator._copy_view.index_copy_(0, gpu_ids, staging)

    def free_request_cpu_cache(self, state: RequestState) -> None:
        """Free CPU blocks for a single request (e.g., on cancellation)."""
        if state.is_cpu_offloaded and self._cpu_pool is not None:
            cpu_blocks = self._request_id_to_cpu_blocks.pop(state.request_id)
            for name, cpu_ids in cpu_blocks.items():
                self._cpu_pool.free_blocks(self.cache.cache_allocators[name].index, cpu_ids)
            state.is_cpu_offloaded = False

    def free_all_waiting_cpu_caches(self) -> None:
        """Free all CPU-offloaded caches in the waiting queue (e.g., on fail_all or reset)."""
        for state in self.scheduler.waiting_requests.values():
            self.free_request_cpu_cache(state)

    def reset(self) -> None:
        """Reset CPU offloading state for a new generation session."""
        self.free_all_waiting_cpu_caches()
        self._request_id_to_cpu_blocks.clear()
        if self._cpu_pool is not None:
            self._cpu_pool.reset()

    def _reserve_cpu_blocks(self, block_counts: dict[str, int]) -> dict[str, list[int]] | None:
        """Reserves CPU pool blocks to hold the given per-allocator block counts, allocating new pool sectors as
        needed. Returns None when the pool cannot fit them."""
        if self._cpu_pool is None:
            return None
        sectors_needed = {}
        for name, num_blocks in block_counts.items():
            allocator = self.cache.cache_allocators[name]
            missing_blocks = num_blocks - self._cpu_pool.count_free_blocks(allocator.index)
            if missing_blocks > 0:
                sectors_needed[name] = ceil(missing_blocks / allocator.blocks_per_sector)
        if self._cpu_pool.num_free_sectors < sum(sectors_needed.values()):
            self._cpu_pool.try_to_free_sectors()
            if self._cpu_pool.num_free_sectors < sum(sectors_needed.values()):
                return None
        for name, num_sectors in sectors_needed.items():
            for _ in range(num_sectors):
                self._cpu_pool.allocate_sector(self.cache.cache_allocators[name].index)
        # Sorted CPU blocks make the CPU-side copies land in more contiguously
        return {
            name: sorted(self._cpu_pool.get_free_blocks(self.cache.cache_allocators[name].index, num_blocks))
            for name, num_blocks in block_counts.items()
        }

    def _offload_to_cpu(
        self, victims: list[RequestState], victim_block_tables: dict[str, dict[str, list[int]]]
    ) -> set[str]:
        """Copy the KV cache blocks of as many victims as fit in the CPU swap pool from GPU to the pool, in one
        batched, non-blocking copy per allocator. Returns the request ids that were offloaded.

        All transfers are enqueued on the compute stream with pinned destinations, so the host never waits on them:
        correctness is guaranteed by stream ordering, since restores and cache writes go through the same stream.
        """
        if self._cpu_pool is None:
            return set()

        # Select the victims that fit in the pool, reserving their CPU blocks as we go
        cpu_offloaded: list[RequestState] = []
        all_gpu_ids: dict[str, list[int]] = {name: [] for name in self.cache.cache_allocators}
        all_cpu_ids: dict[str, list[int]] = {name: [] for name in self.cache.cache_allocators}
        for state in victims:
            gpu_tables = victim_block_tables[state.request_id]
            block_counts = {name: len(blocks) for name, blocks in gpu_tables.items() if blocks}
            if not block_counts:
                continue
            cpu_blocks = self._reserve_cpu_blocks(block_counts)
            if cpu_blocks is None:
                continue
            # The pairing is positional: the k-th CPU block holds the content of the k-th block of the GPU table
            self._request_id_to_cpu_blocks[state.request_id] = cpu_blocks
            for name, cpu_ids in cpu_blocks.items():
                all_gpu_ids[name].extend(gpu_tables[name])
                all_cpu_ids[name].extend(cpu_ids)
            state.is_cpu_offloaded = True
            cpu_offloaded.append(state)
        if not cpu_offloaded:
            return set()

        # One gather and a few pinned slice copies per allocator, all stream-ordered and non-blocking for the host
        with self._stream_ctx():
            for name, allocator in self.cache.cache_allocators.items():
                if not all_gpu_ids[name]:
                    continue
                gpu_ids = torch.as_tensor(all_gpu_ids[name], dtype=torch.long).to(self.cache.device, non_blocking=True)
                gathered_blocks = allocator._copy_view.index_select(0, gpu_ids)
                cpu_view = self._cpu_views[name]
                for start, offset, length in contiguous_runs(all_cpu_ids[name]):
                    cpu_view[start : start + length].copy_(
                        gathered_blocks[offset : offset + length], non_blocking=True
                    )

        # No explicit sync needed: finish_request is a CPU op, and the next forward pass serializes on the same stream
        return {state.request_id for state in cpu_offloaded}
