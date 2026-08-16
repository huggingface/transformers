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
import threading
from abc import ABC, abstractmethod
from collections import deque

from .cache import PagedAttentionCache
from .requests import FutureRequestState, RequestState, RequestStatus, logger


class Scheduler(ABC):
    """
    Abstract base class for scheduling requests in the continuous batch processor. Schedulers manage the lifecycle of
    requests from when they are added to the waiting queue to when they are scheduled for processing. Different
    schedulers implement different strategies for prioritizing and batching requests.
    """

    def __init__(self, cache: PagedAttentionCache, safety_margin: float, max_requests_per_batch: int):
        """Initializes the scheduler. The safety margin is the percentage of free blocks under which we stop
        scheduling new prefill requests, so safety_margin = 0.1 means that when there is less than 10% of free blocks,
        or equivalently when more than 90% of blocks are already allocated, we stop scheduling new prefill requests.
        Setting safety_margin to 0.0 means no safety margin is applied."""
        self.cache = cache
        self.safety_margin = safety_margin
        self.max_requests_per_batch = max_requests_per_batch
        self._cancellation_lock = threading.Lock()
        # Check args
        if safety_margin < 0 or safety_margin > 1:
            raise ValueError(f"Got {safety_margin = } but expected a value in [0, 1]")
        if max_requests_per_batch < 1:
            raise ValueError(f"Got {max_requests_per_batch = } but expected a value >= 1")
        # Initialize mutable states via reset()
        self.reset()

    def reset(self) -> None:
        """Reset scheduler state for a new generation loop."""
        self.active_requests: dict[str, RequestState] = {}
        self.waiting_requests: dict[str, RequestState] = {}
        self.waiting_requests_order: deque[str] = deque()
        self._requests_to_cancel: set[str] = set()
        self._requests_to_fork: list[RequestState] = []
        self.block_new_requests = False
        # Active requests that failed cache allocation in the last scheduled batch w/ the length that was denied
        self.starved_requests: list[tuple[RequestState, int]] = []

    def add_waiting_request(self, state: RequestState):
        """Adds a request to the waiting list."""
        self.waiting_requests[state.request_id] = state
        self.waiting_requests_order.append(state.request_id)

    @abstractmethod
    def schedule_batch(
        self, token_budget: int, cache_budget: int
    ) -> tuple[list[FutureRequestState] | None, bool, int, int]:
        """Schedules requests for the next batch based on available token and cache budgets. This method selects which
        requests should be processed in the current batch, considering the budgets and the scheduler's prioritization
        rules. The token_budget is the maximum number of tokens that can be processed in a batch, and the cache_budget
        is the maximum number of KV cache entries that can be read in a batch.
        Returns the list of scheduled requests in their "FutureRequestState" form, a boolean indicating if the decode
        fast path can be used, the total number of query tokens and the maximum number of kv tokens read."""

    def has_pending_requests(self) -> bool:
        """Checks if there are requests ready to be processed."""
        return bool(len(self.active_requests) or len(self.waiting_requests))

    def finish_request(self, request_id: str) -> None:
        """Completes processing of a request and frees its allocated cache blocks. This method is called
        when a request has finished generation or encountered an error.
        """
        self.cache.free_blocks(request_id)
        self.active_requests.pop(request_id, None)

    def get_active_request_static_outputs(self, request_id: str) -> list[int]:
        """Gets generated tokens for an active request."""
        if request_id in self.active_requests:
            return self.active_requests[request_id].generated_tokens
        return []

    def set_request_cancellation(self, request_id: str):
        """Marks a request for cancellation."""
        with self._cancellation_lock:
            self._requests_to_cancel.add(request_id)

    def clear_cancelled_requests(self) -> list[RequestState]:
        """Remove all cancelled requests from active and waiting queues."""
        cancelled_states = []
        with self._cancellation_lock:
            for request_id in self._requests_to_cancel:
                state_a = self.active_requests.pop(request_id, None)
                state_w = self.waiting_requests.pop(request_id, None)
                # Invariant: a request is never in both queues; state_a or state_w picks the one it was in
                state = state_a or state_w
                if state is not None:
                    cancelled_states.append(state)
                if request_id in self.waiting_requests_order:
                    self.waiting_requests_order.remove(request_id)
                self.cache.free_blocks(request_id)
            self._requests_to_cancel = set()
        return cancelled_states

    def request_is_cancelled(self, request_id: str) -> bool:
        """Checks if a request has been cancelled or removed."""
        return request_id in self._requests_to_cancel or (
            request_id not in self.active_requests and request_id not in self.waiting_requests
        )

    def _infer_request_tokens(self, state: RequestState, request_ids_to_remove_from_waiting: set[str]) -> list[int]:
        """Prepares a request for processing in the current batch. If prefix sharing is enabled, and the request was
        pending, this is where we look for a prefix match and split the request if found."""
        # If prefix sharing is enabled, we look for a prefix match and split the request if found
        if self.cache.use_prefix_sharing and state.status == RequestStatus.PENDING and not state.is_cpu_offloaded:
            prefill_length = self.cache.search_prefix_match(state.request_id, state.remaining_prefill_tokens)
            if prefill_length > 0:
                self.active_requests[state.request_id] = state
                self.waiting_requests.pop(state.request_id, None)  # takes effect even if scheduling fails later
                request_ids_to_remove_from_waiting.add(state.request_id)
                state.status = RequestStatus.PREFILLING
                # The match never covers the whole prompt, so there is always at least 1 token left to prefill
                state.remaining_prefill_tokens = state.remaining_prefill_tokens[prefill_length:]
                state.position_offset += prefill_length

        # If the request is decoding, the tokens to process are already set
        if state.status == RequestStatus.DECODING:
            request_tokens = state.tokens_to_process
        # Otherwise, the tokens to process are the remaining prefill tokens
        else:
            request_tokens = state.remaining_prefill_tokens
        return request_tokens

    def _schedule_request(
        self,
        state: RequestState,
        request_tokens: list[int],
        token_budget: int,
        request_ids_to_remove_from_waiting: set[str],
    ) -> None:
        """Schedules a request for the current batch, updating the request's status according to the token budget left.
        After a request is scheduled, it is part of the next batch unless there is an error.
        If the request has children (for parallel decoding), it ensures at least one token remains before the request is
        forked."""
        # If the request has one or more children we make sure not to prefill it entirely
        # This does not check the request state, but DECODING request already have children set to 0.
        if state.num_children > 0 and token_budget >= len(request_tokens) - 1:
            token_budget = len(request_tokens) - 1
            self._requests_to_fork.append(state)

        # Case: we can process the entire prompt/remainder
        if len(request_tokens) <= token_budget:
            if state.status == RequestStatus.PENDING:
                self.active_requests[state.request_id] = state
                request_ids_to_remove_from_waiting.add(state.request_id)
            if state.status <= RequestStatus.PREFILLING:
                state.tokens_to_process = state.remaining_prefill_tokens
                state.remaining_prefill_tokens = []
                # Although prefill will only be done after the batch being scheduled now, we set the status to DECODING
                # to stay coherent when using asynchronous batching
                state.status = RequestStatus.DECODING

        # Otherwise: we need to split the request
        else:
            if state.status == RequestStatus.PENDING:
                self.active_requests[state.request_id] = state
                state.status = RequestStatus.PREFILLING
                request_ids_to_remove_from_waiting.add(state.request_id)
            state.remaining_prefill_tokens = request_tokens[token_budget:]
            state.tokens_to_process = request_tokens[:token_budget]

    def _try_to_meet_safety_margin(self) -> bool:
        """Tries to meet the safety_margin by freeing unused sectors and (if needed) evicting cached blocks. Returns a
        boolean indicating if the safety margin is met."""
        # Once before loop starts, does not affect cache but non-referenced blocks
        self.cache.pool.try_to_free_sectors()
        cache_free_percent = self.cache.compute_free_capacity()
        in_safety_margin = cache_free_percent >= self.safety_margin
        # When under the margin, evicting the blocks cached for de-duplication may free enough memory
        if not in_safety_margin and self.cache.evict_cached_blocks():
            self.cache.pool.try_to_free_sectors()
            cache_free_percent = self.cache.compute_free_capacity()
            in_safety_margin = cache_free_percent >= self.safety_margin
        # Log and return
        if not in_safety_margin:
            logger.debug(f"{cache_free_percent = } < {self.safety_margin = }: limiting the requests scheduled.")
        return in_safety_margin

    def _process_candidates(
        self,
        candidates: list[RequestState],
        token_budget: int,
        cache_budget: int,
        request_ids_to_remove_from_waiting: set[str],
    ) -> tuple[list[FutureRequestState], bool, bool, int, int]:
        """Schedules candidate requests for the current batch.

        This method contains the common logic shared by all schedulers: it checks token and cache budgets, allocates
        cache blocks if needed, updates request states, and tracks which waiting requests should be removed from the
        waiting queue.
        """
        scheduled_requests = []
        one_allocation_failed = False
        self.starved_requests = []
        decode_fast_path = self.cache.max_blocks_per_request > 0  # zeroed at resolution when the path is unavailable
        original_token_budget, original_cache_budget = token_budget, cache_budget
        request_budget = self.max_requests_per_batch

        # Check safety margin
        in_safety_margin = self._try_to_meet_safety_margin()

        for state in candidates:

            # If we are outside the safety margin, we only accept decoding requests or the first prefill request
            if not in_safety_margin and scheduled_requests and state.status != RequestStatus.DECODING:
                break

            # Infer the tokens that will be present in the batch if token budget is enough
            request_tokens = self._infer_request_tokens(state, request_ids_to_remove_from_waiting)
            # Account for token budget
            request_len = min(len(request_tokens), token_budget)

            # This block checks cache budget: decode batches have infinite budget, but varlen batches don't, because KV
            # cache is read through a fixed-sized index tensor. We keep track of the current budget in case the batch
            # goes from decode to varlen
            is_decode_eligible = request_len == 1 and state.position_offset < self.cache.max_decode_fast_path_length
            read_cache_needed = state.current_len()
            if self.cache.read_cache_limit is not None:
                read_cache_needed = min(read_cache_needed, self.cache.read_cache_limit)
            # A request that would change the batch from decode to varlen is rejected if the cache budget is too low
            if not (decode_fast_path and is_decode_eligible) and cache_budget < read_cache_needed:
                continue

            # Final check before we schedule the request: can the cache handle it, or is the request starved?
            request_fits = self.cache.can_store_request_tokens(state, request_len)
            if not request_fits:
                one_allocation_failed = True
                # If the request is active and cannot be scheduled, we mark it as starved
                if state.request_id in self.active_requests:
                    self.starved_requests.append((state, request_len))
                continue

            # If this point is reached, it means we can safely schedule the request
            self._schedule_request(state, request_tokens, token_budget, request_ids_to_remove_from_waiting)
            request_len = len(state.tokens_to_process)  # it may change after scheduling

            # The decode fast path is only used if the request is a single token and its length is less than the max blocks per request
            decode_fast_path &= request_len == 1 and state.position_offset < self.cache.max_decode_fast_path_length

            # Update the token and cache budgets
            token_budget -= request_len
            cache_budget -= read_cache_needed
            request_budget -= 1

            # If using prefix sharing, we make note of the blocks that will be completed by the forward pass
            complete_blocks = self.cache.count_new_complete_blocks(state, request_len)

            # Store the future request state
            has_new_token = not state.remaining_prefill_tokens
            scheduled_requests.append(FutureRequestState(state, has_new_token, complete_blocks, request_len))

            # Remove the request from the waiting queue and mark it as removed
            req_id = state.request_id
            was_waiting = self.waiting_requests.pop(req_id, None) is not None
            if was_waiting:
                request_ids_to_remove_from_waiting.add(req_id)

            # Early exit of the loop if we have no budget left
            if token_budget == 0 or (cache_budget <= 0 and not decode_fast_path) or request_budget <= 0:
                break

        num_q_tokens = original_token_budget - token_budget
        max_kv_read = original_cache_budget - cache_budget
        return scheduled_requests, one_allocation_failed, decode_fast_path, num_q_tokens, max_kv_read

    def _get_waiting_candidates(self) -> list[RequestState]:
        """Returns waiting requests in priority order. Since CPU-offloaded requests are cheaper to restore than fresh
        requests, they get priority, but we interleave them with fresh request to not saturate new batches with only
        offloaded requests."""
        offloaded: deque[RequestState] = deque()
        fresh: deque[RequestState] = deque()
        for req_id in self.waiting_requests_order:
            state = self.waiting_requests[req_id]
            (offloaded if state.is_cpu_offloaded else fresh).append(state)
        ordered: list[RequestState] = []
        while offloaded or fresh:
            if offloaded:
                ordered.append(offloaded.popleft())
            if fresh:
                ordered.append(fresh.popleft())
        return ordered

    def _cleanup_waiting_queue(self, request_ids_to_remove_from_waiting: set[str]) -> None:
        """Removes processed requests from the waiting queue order."""
        self.waiting_requests_order = deque(
            [req_id for req_id in self.waiting_requests_order if req_id not in request_ids_to_remove_from_waiting]
        )


# TODO: further common-ize the two classes
class FIFOScheduler(Scheduler):
    """This scheduler processes requests in the order they arrive, meaning decoding requests has priority over
    prefilling requests."""

    def __init__(self, cache: PagedAttentionCache, safety_margin: float | None, max_requests_per_batch: int):
        """Initializes the FIFO scheduler, with a default safety margin of 0.15 (ie. 15% of free blocks)."""
        if safety_margin is None:
            safety_margin = 0.15
        super().__init__(cache, safety_margin, max_requests_per_batch)

    def schedule_batch(
        self, token_budget: int, cache_budget: int
    ) -> tuple[list[FutureRequestState] | None, bool, int, int]:
        priority_states: list[RequestState] = []
        second_priority_states: list[RequestState] = []

        for state in self.active_requests.values():
            if state.status == RequestStatus.DECODING:
                priority_states.append(state)
            elif state.status == RequestStatus.PREFILLING:
                second_priority_states.append(state)

        # Add waiting requests to second priority, with CPU-offloaded requests first
        if not self.block_new_requests:
            second_priority_states.extend(self._get_waiting_candidates())

        candidates = priority_states + second_priority_states
        request_ids_to_remove_from_waiting = set()
        scheduled_requests, one_allocation_failed, decode_fast_path, num_q_tokens, max_kv_read = (
            self._process_candidates(
                candidates,
                token_budget,
                cache_budget,
                request_ids_to_remove_from_waiting,
            )
        )

        # We remove waiting requests before checking requests were scheduled, because there might have been prefill matches
        self._cleanup_waiting_queue(request_ids_to_remove_from_waiting)

        # If no requests were scheduled and the cache is full, we signal it by returning None
        if not scheduled_requests and one_allocation_failed:
            return None, decode_fast_path, 0, 0

        return scheduled_requests, decode_fast_path, num_q_tokens, max_kv_read


# FIXME: prioritize adding from waiting reqs before scheduling `RequestStatus.DECODING` when cache space allows it
# TODO: further consolidate the code by making more of it common. The reference Scheduler is FIFO, not this one.
class PrefillFirstScheduler(Scheduler):
    """Scheduler that prioritizes split prefill requests over decoding requests. This scheduler ensures that split
    prefill requests (which are continuations of partially processed prompts) are completed before processing new
    decoding requests."""

    def __init__(self, cache: PagedAttentionCache, safety_margin: float | None, max_requests_per_batch: int):
        """Initializes the prefill first scheduler, with a default safety margin of 0.0 (no safety margin)."""
        if safety_margin is None:
            safety_margin = 0.0
        super().__init__(cache, safety_margin, max_requests_per_batch)

    def schedule_batch(
        self, token_budget: int, cache_budget: int
    ) -> tuple[list[FutureRequestState] | None, bool, int, int]:
        priority_states: list[RequestState] = []
        second_priority_states: list[RequestState] = []

        for state in self.active_requests.values():
            # XXX: when cache is full, state can stay on `PREFILLING_SPLIT` so we need to take those into account
            if state.status == RequestStatus.PREFILLING:
                priority_states.append(state)
            elif state.status == RequestStatus.DECODING:
                second_priority_states.append(state)

        # Add waiting requests to second priority, with CPU-offloaded requests first
        if not self.block_new_requests:
            second_priority_states.extend(self._get_waiting_candidates())

        candidates = priority_states + second_priority_states
        request_ids_to_remove_from_waiting = set()
        scheduled_requests, one_allocation_failed, decode_fast_path, num_q_tokens, max_kv_read = (
            self._process_candidates(
                candidates,
                token_budget,
                cache_budget,
                request_ids_to_remove_from_waiting,
            )
        )

        # We remove waiting requests before checking requests were scheduled, because there might have been prefill matches
        self._cleanup_waiting_queue(request_ids_to_remove_from_waiting)

        # If no requests were scheduled and the cache is full, we signal it by returning None
        if not scheduled_requests and one_allocation_failed:
            return None, decode_fast_path, 0, 0

        return scheduled_requests, decode_fast_path, num_q_tokens, max_kv_read


SCHEDULER_MAPPING = {
    "fifo": FIFOScheduler,
    "prefill_first": PrefillFirstScheduler,
}
