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

from ...utils.metrics import attach_tracer, traced
from .cache import PagedAttentionCache
from .requests import RequestState, RequestStatus, logger


class Scheduler(ABC):
    """
    Abstract base class for scheduling requests in the continuous batch processor. Schedulers manage the lifecycle of
    requests from when they are added to the waiting queue to when they are scheduled for processing. Different
    schedulers implement different strategies for prioritizing and batching requests.
    """

    def __init__(self, cache: PagedAttentionCache, retain_cache_on_finish: bool = False):
        self.active_requests: dict[str, RequestState] = {}
        self.waiting_requests: dict[str, RequestState] = {}
        self.waiting_requests_order: deque[str] = deque()
        self.cache = cache
        self.retain_cache_on_finish = retain_cache_on_finish
        self._cancellation_lock = threading.Lock()
        self._requests_to_cancel: set[str] = set()
        self._requests_to_fork: list[RequestState] = []
        # This state is used to avoid infinite loops when offloading requests
        self.block_new_requests = False
        # This is to compute the cache used by a new request being scheduled
        self.cache_budget_module = None if cache.num_full_attention_groups else cache.config.sliding_window

    @traced
    def add_waiting_request(self, state: RequestState):
        """Adds a request to the waiting list."""
        if self.retain_cache_on_finish and state.request_id in self.active_requests:
            old_state = self.active_requests.pop(state.request_id)
            state.tokens_to_process = state.tokens_to_process[
                len(old_state.initial_tokens) :
            ]  # XXX: check for indexing error?
            state.allocated_blocks = old_state.allocated_blocks
            state.position_offset = old_state.position_offset
        self.waiting_requests[state.request_id] = state
        self.waiting_requests_order.append(state.request_id)

    @abstractmethod
    def schedule_batch(self, token_budget: int, cache_budget: int) -> list[RequestState] | None:
        """Schedules requests for the next batch based on available token and cache budgets. This method selects which
        requests should be processed in the current batch, considering the budgets and the scheduler's prioritization
        rules. The token_budget is the maximum number of tokens that can be processed in a batch, and the cache_budget
        is the maximum number of KV cache entries that can be read in a batch."""

    @traced
    def has_pending_requests(self) -> bool:
        """Checks if there are requests ready to be processed."""
        return len(self.active_requests) or len(self.waiting_requests)

    @traced
    def finish_request(self, request_id: str, evict_from_cache: bool = True) -> None:
        """Completes processing of a request and optionally frees its allocated cache blocks. This method is called
        when a request has finished generation or encountered an error.
        """
        if evict_from_cache:
            self.cache.free_blocks(request_id)
            self.active_requests.pop(request_id, None)

    @traced
    def get_active_request_static_outputs(self, request_id: str) -> list[int]:
        """Gets generated tokens for an active request."""
        if request_id in self.active_requests:
            return self.active_requests[request_id].generated_tokens
        return []

    @traced
    def set_request_cancellation(self, request_id: str):
        """Marks a request for cancellation."""
        with self._cancellation_lock:
            self._requests_to_cancel.add(request_id)

    @traced
    def clear_cancelled_requests(self):
        """Remove all cancelled requests from active and waiting queues."""
        with self._cancellation_lock:
            for request_id in self._requests_to_cancel:
                self.active_requests.pop(request_id, None)
                self.waiting_requests.pop(request_id, None)
                if request_id in self.waiting_requests_order:
                    self.waiting_requests_order.remove(request_id)
                self.cache.free_blocks(request_id)
            self._requests_to_cancel = set()

    @traced
    def request_is_cancelled(self, request_id: str) -> bool:
        """Checks if a request has been cancelled or removed."""
        return request_id in self._requests_to_cancel or (
            request_id not in self.active_requests and request_id not in self.waiting_requests
        )

    @traced
    def _allocate_blocks_if_needed(self, state: RequestState, len_next_tokens: int) -> bool:
        """Allocate additional cache blocks for a request if the currently allocated blocks are insufficient to
        accommodate the next tokens. It calculates how many blocks are needed based on the request's current
        cache occupancy and the number of tokens to be processed. The allocation itself is done by the CacheAllocator
        objects. Returns a boolean indicating if the allocation was successful or not.
        """
        # 1. we check that the occupancy is less than the requested length
        # 2. we allocate enough blocks to cover the requested length
        current_len = state.current_len()
        occupancy = state.allocated_blocks * self.cache.block_size - current_len
        if occupancy < len_next_tokens or state.allocated_blocks == 0:
            blocks_needed = ((len_next_tokens - occupancy + 1) // self.cache.block_size) + 1
            allocated = self.cache.allocate_blocks(blocks_needed, state.request_id, state.allocated_blocks)
            if allocated is None:
                return False
            state.allocated_blocks += allocated
        return True

    def _infer_request_tokens(self, state: RequestState, request_ids_to_remove_from_waiting: set[str]) -> list[int]:
        """Prepares a request for processing in the current batch. If prefix sharing is enabled, and the request was
        pending, this is where we look for a prefix match and split the request if found."""
        # If prefix sharing is enabled, we look for a prefix match and split the request if found
        if self.cache.use_prefix_sharing and state.status == RequestStatus.PENDING:
            prefill_length = self.cache.search_prefix_match(state.request_id, state.tokens_to_process)
            if prefill_length > 0:
                self.active_requests[state.request_id] = state
                request_ids_to_remove_from_waiting.add(state.request_id)
                state.status = RequestStatus.SPLIT_PENDING_REMAINDER
                # We keep track of the number of allocated blocks to avoid double allocation
                state.allocated_blocks += prefill_length // self.cache.block_size
                # Even if we match the whole request, we keep at least 1 token to start decoding
                prefill_length = min(prefill_length, len(state.tokens_to_process) - 1)
                state.remaining_prefill_tokens = state.tokens_to_process[prefill_length:]
                state.tokens_to_process = state.tokens_to_process[prefill_length:]
                state.position_offset += prefill_length

        # If the request has a split prefill, the tokens to process are the remaining prompt ids
        if state.status == RequestStatus.SPLIT_PENDING_REMAINDER:
            request_tokens = state.remaining_prefill_tokens
        # Otherwise, the tokens to process are the prompt ids, which are the full prompt or the last predicted tokens
        else:
            request_tokens = state.tokens_to_process
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
        if len(request_tokens) < token_budget:
            if state.status == RequestStatus.PENDING:
                self.active_requests[state.request_id] = state
                state.status = RequestStatus.PREFILLING
                request_ids_to_remove_from_waiting.add(state.request_id)
            elif state.status == RequestStatus.SPLIT_PENDING_REMAINDER:
                state.status = RequestStatus.PREFILLING
                state.tokens_to_process = state.remaining_prefill_tokens
                state.remaining_prefill_tokens = []

        # Otherwise: we need to split the request
        else:
            if state.status == RequestStatus.PENDING:
                self.active_requests[state.request_id] = state
                state.status = RequestStatus.PREFILLING_SPLIT
                request_ids_to_remove_from_waiting.add(state.request_id)
            elif state.status == RequestStatus.SPLIT_PENDING_REMAINDER:
                state.status = RequestStatus.PREFILLING_SPLIT
            state.remaining_prefill_tokens = request_tokens[token_budget:]
            state.tokens_to_process = request_tokens[:token_budget]

    def _process_candidates(
        self,
        candidates: list[RequestState],
        token_budget: int,
        cache_budget: int,
        request_ids_to_remove_from_waiting: set[str],
        safety_margin: float = 0.0,
    ) -> tuple[list[RequestState], bool]:
        """Schedules candidate requests for the current batch.

        This method contains the common logic shared by all schedulers: it checks token and cache budgets, allocates
        cache blocks if needed, updates request states, and tracks which waiting requests should be removed from the
        waiting queue.
        """
        scheduled_requests = []
        one_allocation_failed = False
        safety_margins = safety_margin * self.cache.num_blocks

        for state in candidates:
            num_free_blocks = self.cache.get_num_free_blocks()
            # If we are out the safety margin, we only accept decoding requests or the first prefill request
            outside_safety_margin = num_free_blocks < safety_margins
            if outside_safety_margin and scheduled_requests and state.status != RequestStatus.DECODING:
                logger.info(
                    f"Outside safety margin, breaking out of scheduling loop. {num_free_blocks = } {safety_margins = }"
                )
                break

            # Check cache budget
            cache_needed = state.current_len()
            cache_needed = (
                cache_needed if self.cache_budget_module is None else cache_needed % self.cache_budget_module
            )
            if cache_budget < cache_needed:
                continue

            # Infer the tokens that will be present in the batch if token budget is enough
            request_tokens = self._infer_request_tokens(state, request_ids_to_remove_from_waiting)
            # Account for token budget
            request_len = min(len(request_tokens), token_budget)
            # Check there will be enough cache for the new tokens
            allocation_successful = self._allocate_blocks_if_needed(state, request_len)

            # If the allocation would not be successful, we move on to the next request
            if not allocation_successful:
                one_allocation_failed = True
                # If we reached a waiting request and the cache is full, all subsequent waiting requests will need
                # allocation as well, so we can safely break out of the scheduling loop.
                if num_free_blocks == 0 and state.request_id in self.waiting_requests:
                    logger.info(f"Breaking mid-loop for request {state.request_id} because the cache is full")
                    break
                continue

            # If this point is reached, it means we can safely schedule the request
            self._schedule_request(state, request_tokens, token_budget, request_ids_to_remove_from_waiting)
            request_len = len(state.tokens_to_process)  # it may change after scheduling
            scheduled_requests.append(state)

            # Update the token and cache budgets
            token_budget -= request_len
            cache_budget -= cache_needed

            # If using prefix sharing, we make note of the blocks that will be computed in the forward pass
            if self.cache.allow_block_sharing:
                tokens_in_current_block = state.current_len() % self.cache.block_size
                tokens_after_forward = tokens_in_current_block + request_len
                complete_blocks = tokens_after_forward // self.cache.block_size
                self.cache.blocks_to_complete[state.request_id] = complete_blocks

            # Remove the request from the waiting queue and mark it as removed
            req_id = state.request_id
            was_waiting = self.waiting_requests.pop(req_id, None) is not None
            if was_waiting:
                request_ids_to_remove_from_waiting.add(req_id)

            # Early exit of the loop if we have no budget left
            if token_budget == 0 or cache_budget == 0:
                break

        return scheduled_requests, one_allocation_failed

    def _cleanup_waiting_queue(self, request_ids_to_remove_from_waiting: set[str]) -> None:
        """Removes processed requests from the waiting queue order."""
        self.waiting_requests_order = deque(
            [req_id for req_id in self.waiting_requests_order if req_id not in request_ids_to_remove_from_waiting]
        )


# TODO: further common-ize the two classes
@attach_tracer()
class FIFOScheduler(Scheduler):
    """This scheduler processes requests in the order they arrive, meaning decoding requests has priority over
    prefilling requests. Additionally, it includes a safety margin mechanism to prevent cache exhaustion. By default,
    when 80% of the cache is full, new requests will not be scheduled to prioritize decoding active requests."""

    def __init__(self, cache: PagedAttentionCache, retain_cache_on_finish: bool = False, safety_margin: float = 0.2):
        """Initializes the FIFO scheduler. The safety margin is the percentage of free blocks under which we stop
        scheduling new prefill requests, so safety_margin = 0.1 means that when there is less than 10% of free blocks,
        or equivalently when more than 90% of blocks are already allocated, we stop scheduling new prefill requests.
        """
        super().__init__(cache, retain_cache_on_finish)
        self.safety_margin = safety_margin

    @traced
    def schedule_batch(self, token_budget: int, cache_budget: int) -> list[RequestState] | None:
        priority_states: list[RequestState] = []
        second_priority_states: list[RequestState] = []

        for state in self.active_requests.values():
            if state.status == RequestStatus.DECODING:
                priority_states.append(state)
            if state.status in [RequestStatus.SPLIT_PENDING_REMAINDER, RequestStatus.PREFILLING_SPLIT]:
                second_priority_states.append(state)

        # Add waiting requests to second priority
        if not self.block_new_requests:
            for req_id in self.waiting_requests_order:
                second_priority_states.append(self.waiting_requests[req_id])

        candidates = priority_states + second_priority_states
        request_ids_to_remove_from_waiting = set()
        scheduled_requests, one_allocation_failed = self._process_candidates(
            candidates,
            token_budget,
            cache_budget,
            request_ids_to_remove_from_waiting,
            safety_margin=self.safety_margin,
        )

        # We remove waiting requests before checking requests were scheduled, because there might have been prefill matches
        self._cleanup_waiting_queue(request_ids_to_remove_from_waiting)

        # If no requests were scheduled and the cache is full, we signal it by returning None
        if not scheduled_requests and one_allocation_failed:
            return None

        return scheduled_requests


# FIXME: prioritize adding from waiting reqs before scheduling `RequestStatus.DECODING` when cache space allows it
# TODO: further consolidate the code by making more of it common. The reference Scheduler is FIFO, not this one.
@attach_tracer()
class PrefillFirstScheduler(Scheduler):
    """Scheduler that prioritizes split prefill requests over decoding requests. This scheduler ensures that split
    prefill requests (which are continuations of partially processed prompts) are completed before processing new
    decoding requests."""

    @traced
    def schedule_batch(self, token_budget: int, cache_budget: int) -> list[RequestState] | None:
        priority_states: list[RequestState] = []
        second_priority_states: list[RequestState] = []

        for state in self.active_requests.values():
            # XXX: when cache is full, state can stay on `PREFILLING_SPLIT` so we need to take those into account
            if state.status in [RequestStatus.PREFILLING_SPLIT, RequestStatus.SPLIT_PENDING_REMAINDER]:
                priority_states.append(state)
            elif state.status == RequestStatus.DECODING:
                second_priority_states.append(state)

        # Add waiting requests to second priority
        if not self.block_new_requests:
            for req_id in self.waiting_requests_order:
                second_priority_states.append(self.waiting_requests[req_id])

        candidates = priority_states + second_priority_states
        request_ids_to_remove_from_waiting = set()
        scheduled_requests, one_allocation_failed = self._process_candidates(
            candidates,
            token_budget,
            cache_budget,
            request_ids_to_remove_from_waiting,
            safety_margin=0.0,
        )

        # We remove waiting requests before checking requests were scheduled, because there might have been prefill matches
        self._cleanup_waiting_queue(request_ids_to_remove_from_waiting)

        # If no requests were scheduled and the cache is full, we signal it by returning None
        if not scheduled_requests and one_allocation_failed:
            return None

        return scheduled_requests


SCHEDULER_MAPPING = {
    "fifo": FIFOScheduler,
    "prefill_first": PrefillFirstScheduler,
}
