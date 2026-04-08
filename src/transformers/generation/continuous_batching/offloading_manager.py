# Copyright 2025 The HuggingFace Inc. team
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
  2. Soft reset (legacy): discard the KV cache and re-prefill from scratch when the request is re-scheduled.

The CPU swap pool is a static set of pinned tensors allocated once at init (like vLLM/SGLang). Blocks are tracked
with a simple free set — no dynamic allocation or deallocation of tensors ever happens at runtime.
"""

from contextlib import nullcontext

import torch

from ...utils import is_psutil_available
from .cache import PagedAttentionCache
from .input_outputs import ContinuousBatchingAsyncIOs, ContinuousBatchingIOs
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
        cpu_offload_space_gib: float,
        safety_threshold: float,
        inputs_and_outputs: ContinuousBatchingIOs | ContinuousBatchingAsyncIOs,
    ) -> None:
        self.cache = cache
        self.scheduler = scheduler

        # All offloading transfers run on the compute stream (stream-ordered, like the fork copy path)
        self._compute_stream: torch.cuda.Stream | None = inputs_and_outputs.compute_stream

        # Compute the size of the CPU swap pool in blocks
        self._num_cpu_blocks = self._compute_num_cpu_blocks(cpu_offload_space_gib, safety_threshold)
        if self._num_cpu_blocks == 0 and cpu_offload_space_gib > 0:
            logger.warning(
                f"cpu_offload_space={cpu_offload_space_gib:.1f} GiB is too small for even one block. No CPU offloading."
            )
        # Allocate the CPU swap pool
        layer_group_size = len(cache.key_cache)
        cpu_cache_shape = (self._num_cpu_blocks, cache.block_size, cache.num_key_value_heads, cache.head_dim)

        self._cpu_key_cache: list[torch.Tensor] = [
            torch.empty(cpu_cache_shape, dtype=cache.dtype, pin_memory=True) for _ in range(layer_group_size)
        ]
        self._cpu_value_cache: list[torch.Tensor] = [
            torch.empty(cpu_cache_shape, dtype=cache.dtype, pin_memory=True) for _ in range(layer_group_size)
        ]

        # Record-keeping attributes
        self._free_cpu_blocks = set(range(self._num_cpu_blocks))
        self._request_id_to_cpu_blocks: dict[str, list[int]] = {}
        self._request_id_to_group_block_counts: dict[str, list[int]] = {}
        self._gpu_cache_block_shape = (-1, self.cache.block_size, self.cache.num_key_value_heads, self.cache.head_dim)

        # Log the size of the CPU swap pool
        size_in_bytes = 2 * self._cpu_key_cache[0].numel() * self._cpu_key_cache[0].element_size() * layer_group_size
        logger.info(
            f"CPU swap pool initialized: {self._num_cpu_blocks} blocks ({size_in_bytes / (1024**3):.2f} GiB pinned)"
        )

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def _compute_num_cpu_blocks(self, cpu_offload_space_gib: float, safety_threshold: float) -> int:
        """Returns the number of blocks that can fit in the CPU swap pool."""
        # Compute the CPU pool size in bytes
        offload_bytes = int(cpu_offload_space_gib * (1024**3))

        if is_psutil_available():
            import psutil

            total_ram = psutil.virtual_memory().total
            max_bytes = int(total_ram * safety_threshold)
            if offload_bytes > max_bytes:
                clamped_gib = max_bytes / (1024**3)
                logger.warning(
                    f"cpu_offload_space={cpu_offload_space_gib:.1f} GiB exceeds {safety_threshold:.0%} of total RAM "
                    f"({total_ram / (1024**3):.1f} GiB). Clamping to {clamped_gib:.1f} GiB."
                )
                offload_bytes = max_bytes
        else:
            logger.warning(
                "psutil is not available — cpu_offload_space_safety_threshold cannot be enforced. "
                "Install psutil to enable the safety cap."
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

    # ------------------------------------------------------------------
    # Tiny helpers
    # ------------------------------------------------------------------

    def can_cpu_offload(self, num_blocks: int) -> bool:
        """Returns True if there are enough free CPU blocks to offload `num_blocks`."""
        return len(self._free_cpu_blocks) >= num_blocks

    # ------------------------------------------------------------------
    # Offloading: pick a victim, copy GPU→CPU or soft-reset
    # ------------------------------------------------------------------

    def offload_one_request(self) -> None:
        """Offload one active request to make room in the GPU cache. Tries CPU offloading first; if the pool is full,
        falls back to the legacy soft reset."""
        # The offloaded request is the newest (resp. oldest) if block_new_requests is True (resp. False)
        scheduler = self.scheduler
        if scheduler.block_new_requests:
            request_id, state = scheduler.active_requests.popitem()
        else:
            request_id, state = next(iter(scheduler.active_requests.items()))
        logger.info(
            f"Soft resetting request {request_id} with {len(state.initial_tokens)} initial tokens and "
            f"{len(state.generated_tokens)} generated tokens"
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
            # Here, the new state is the same as the old one, but with the status set to PENDING
            state._status = RequestStatus.PENDING
            new_state = state
        else:
            new_state = state.create_equivalent_initial_request()
            state._status = RequestStatus.FINISHED

        scheduler.finish_request(request_id)
        scheduler.add_waiting_request(new_state)
        logger.info(
            f"{'Offloaded' if offloaded_to_cpu else 'Soft reset'} request {request_id} with {len(state.initial_tokens)}"
            f" initial tokens and {len(state.generated_tokens)} generated tokens."
        )
        scheduler.block_new_requests = True

    # ------------------------------------------------------------------
    # Restoration: copy CPU→GPU, fix up request state
    # ------------------------------------------------------------------

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
            state._is_cpu_offloaded = False
            state.allocated_blocks = max_allocated_blocks
            # If prefix sharing is on, these freshly allocated blocks will be deduplicated at the next update
            if cache.allow_block_sharing:
                future_state.complete_blocks += state.position_offset // cache.block_size
            logger.info(f"Restored CPU-offloaded request {state.request_id}")

        # Early return if there are no copy to perform
        if not all_cpu_indices:
            return None

        # Single batched copy for all requests (still, one copy per layer)
        cpu_ids = torch.tensor(all_cpu_indices, device="cpu", dtype=torch.int32)
        gpu_ids = torch.tensor(all_gpu_indices, device=cache.device, dtype=torch.int32)
        maybe_stream = torch.cuda.stream(self._compute_stream) if self._compute_stream is not None else nullcontext()
        with maybe_stream:
            for cpu_k, gpu_k in zip(self._cpu_key_cache, cache.key_cache):
                gpu_k.view(*self._gpu_cache_block_shape)[gpu_ids] = cpu_k[cpu_ids].to(cache.device)
            for cpu_v, gpu_v in zip(self._cpu_value_cache, cache.value_cache):
                gpu_v.view(*self._gpu_cache_block_shape)[gpu_ids] = cpu_v[cpu_ids].to(cache.device)
        self._free_cpu_blocks.update(all_cpu_indices)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def free_request_cpu_cache(self, state: RequestState) -> None:
        """Free CPU blocks for a single request (e.g., on cancellation)."""
        if state.is_cpu_offloaded:
            self._return_cpu_blocks(state.request_id)
            state._is_cpu_offloaded = False

    def free_all_waiting_cpu_caches(self) -> None:
        """Free all CPU-offloaded caches in the waiting queue (e.g., on fail_all or reset)."""
        for state in self.scheduler.waiting_requests.values():
            self.free_request_cpu_cache(state)

    def reset(self) -> None:
        """Reset CPU offloading state for a new generation session."""
        self.free_all_waiting_cpu_caches()
        self._request_id_to_cpu_blocks.clear()
        self._request_id_to_group_block_counts.clear()
        self._free_cpu_blocks = set(range(self._num_cpu_blocks))

    # ------------------------------------------------------------------
    # Private: GPU↔CPU block copy using the static pool
    # ------------------------------------------------------------------

    def _offload_to_cpu(self, request_id: str, state: RequestState) -> bool:
        """Copy a request's KV cache blocks from GPU to the static CPU swap pool. Returns True on success, False if
        the pool is full. Must be called BEFORE free_blocks() for this request."""

        # Get the indices to offload from
        gpu_indices = []
        group_block_counts = []
        for cm in self.cache.group_cache_managers:
            blocks = cm.block_table.get(request_id, [])
            gpu_indices.extend(blocks)
            group_block_counts.append(len(blocks))

        # No CPU offloading if there are no blocks to offload or not enough free blocks in the CPU swap pool
        total_gpu_blocks = len(gpu_indices)
        if total_gpu_blocks == 0 or not self.can_cpu_offload(total_gpu_blocks):
            return False

        # Reserve CPU blocks from the free set
        cpu_indices = [self._free_cpu_blocks.pop() for _ in range(total_gpu_blocks)]

        # Offload using the compute stream so it does not interfere with current generation
        maybe_stream = torch.cuda.stream(self._compute_stream) if self._compute_stream is not None else nullcontext()
        with maybe_stream:
            cpu_ids = torch.tensor(cpu_indices, device="cpu", dtype=torch.int32)
            gpu_ids = torch.tensor(gpu_indices, device=self.cache.device, dtype=torch.int32)
            # Keys
            for cpu_key_cache, gpu_key_cache in zip(self._cpu_key_cache, self.cache.key_cache):
                cpu_key_cache[cpu_ids] = gpu_key_cache.view(*self._gpu_cache_block_shape)[gpu_ids].to("cpu")
            # Values
            for cpu_value_cache, gpu_value_cache in zip(self._cpu_value_cache, self.cache.value_cache):
                cpu_value_cache[cpu_ids] = gpu_value_cache.view(*self._gpu_cache_block_shape)[gpu_ids].to("cpu")

        # No explicit sync needed: finish_request is logical, and the next forward pass serializes on the same stream.
        self._request_id_to_cpu_blocks[request_id] = cpu_indices
        self._request_id_to_group_block_counts[request_id] = group_block_counts
        state._is_cpu_offloaded = True
        return True

    def _return_cpu_blocks(self, request_id: str) -> tuple[list[int], list[int]]:
        """Return CPU blocks to the free set without copying anything."""
        cpu_ids = self._request_id_to_cpu_blocks.pop(request_id)
        group_counts = self._request_id_to_group_block_counts.pop(request_id)
        self._free_cpu_blocks.update(cpu_ids)
        return cpu_ids, group_counts
