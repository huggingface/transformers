# coding=utf-8
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
from abc import ABC, abstractmethod
from collections import deque

from ...utils.metrics import attach_tracer, traced
from .cache import PagedAttentionCache
from .classes import RequestState, RequestStatus


class Scheduler(ABC):
    """
    Abstract base class for scheduling requests in the continuous batch processor.
    It is expected that cache allocation and scheduling logic will be implemented in subclasses.
    """

    def __init__(self, cache: PagedAttentionCache, retain_cache_on_finish: bool = False):
        self.active_requests: dict[str, RequestState] = {}
        self.waiting_requests: dict[str, RequestState] = {}
        self.waiting_requests_order: deque[str] = deque()
        self.cache = cache
        self.retain_cache_on_finish = retain_cache_on_finish
        self.requests_to_cancel: set[str] = set()

    @abstractmethod
    def add_waiting_request(self, state: RequestState):
        """Add a request to the waiting list."""
        pass

    @abstractmethod
    def schedule_batch(self, token_budget: int) -> list[RequestState]:
        pass

    @traced
    def has_pending_requests(self) -> bool:
        """Check if there are requests ready to be processed."""
        return len(self.active_requests) or len(self.waiting_requests)

    @abstractmethod
    def finish_request(self, request_id: str, evict_from_cache: bool = True):
        """Finish processing a request and free its allocated blocks."""
        pass

    @traced
    def get_active_request_static_outputs(self, request_id: str) -> list[int]:
        if request_id in self.active_requests:
            return self.active_requests[request_id].static_outputs
        return []

    @traced
    def set_request_cancellation(self, request_id: str):
        self.requests_to_cancel.add(request_id)

    @traced
    def clear_cancelled_requests(self):
        for request_id in self.requests_to_cancel:
            if request_id in self.active_requests:
                del self.active_requests[request_id]
            if request_id in self.waiting_requests:
                del self.waiting_requests[request_id]
            if request_id in self.waiting_requests_order:
                self.waiting_requests_order.remove(request_id)
            self.cache.free_blocks(request_id)
        self.requests_to_cancel = set()

    @traced
    def request_is_cancelled(self, request_id: str) -> bool:
        return request_id in self.requests_to_cancel or (
            request_id not in self.active_requests and request_id not in self.waiting_requests
        )


@attach_tracer()
class FIFOScheduler(Scheduler):
    def __init__(self, cache: PagedAttentionCache, retain_cache_on_finish: bool = False, safety_margin: float = 0.0):
        super().__init__(cache, retain_cache_on_finish)
        self.safety_margin = safety_margin

    @traced
    def _allocate_blocks_if_needed(self, state: RequestState, len_next_tokens: int):
        # 1. we check that the occupancy is less than the requested length
        # 2. we allocate enough blocks to cover the requested length
        current_len = state.current_len()
        occupancy = len(state.allocated_blocks) * self.cache.block_size - current_len
        if occupancy < len_next_tokens or (len(state.allocated_blocks) == 0):
            blocks_needed = ((len_next_tokens - occupancy + 1) // self.cache.block_size) + 1
            allocated = self.cache.allocate_blocks(blocks_needed, state.request_id)
            if not allocated:
                return False
            state.allocated_blocks.extend(allocated)
        return True

    @traced(span_name="prepare_request")
    def _prepare_request_for_processing(
        self, state: RequestState, token_budget: int, request_ids_to_remove_from_waiting: set[str]
    ):
        """Prepare a request for processing in the current batch."""
        request_tokens = (
            state.remaining_prompt_ids if state.status == RequestStatus.SPLIT_PENDING_REMAINDER else state.prompt_ids
        )
        if len(request_tokens) < token_budget:
            # Can process the entire prompt/remainder
            if state.status == RequestStatus.PENDING:
                self.active_requests[state.request_id] = state
                state.status = RequestStatus.PREFILLING
                request_ids_to_remove_from_waiting.add(state.request_id)
            elif state.status == RequestStatus.SPLIT_PENDING_REMAINDER:
                state.status = RequestStatus.PREFILLING
                state.prompt_ids = state.remaining_prompt_ids
                state.remaining_prompt_ids = []
        else:
            # Need to split the request
            if state.status == RequestStatus.PENDING:
                self.active_requests[state.request_id] = state
                state.status = RequestStatus.PREFILLING_SPLIT
                request_ids_to_remove_from_waiting.add(state.request_id)
            elif state.status == RequestStatus.SPLIT_PENDING_REMAINDER:
                state.status = RequestStatus.PREFILLING_SPLIT
            state.remaining_prompt_ids = request_tokens[token_budget:]
            state.prompt_ids = request_tokens[:token_budget]

    @traced
    def add_waiting_request(self, state: RequestState):
        """Add a request to the waiting list."""
        if self.retain_cache_on_finish and state.request_id in self.active_requests:
            old_state = self.active_requests.pop(state.request_id)
            state.prompt_ids = state.prompt_ids[len(old_state.full_prompt_ids) :]
            state.allocated_blocks = old_state.allocated_blocks
            state.position_offset = old_state.position_offset
        self.waiting_requests[state.request_id] = state
        self.waiting_requests_order.append(state.request_id)

    @traced
    def schedule_batch(self, token_budget: int) -> list[RequestState]:
        priority_states: list[RequestState] = []
        second_priority_states: list[RequestState] = []
        scheduled_requests = []

        for state in self.active_requests.values():
            if state.status == RequestStatus.DECODING:
                priority_states.append(state)
            if state.status == RequestStatus.SPLIT_PENDING_REMAINDER:
                second_priority_states.append(state)

        # Add waiting requests to second priority
        for req_id in self.waiting_requests_order:
            second_priority_states.append(self.waiting_requests[req_id])

        candidates = priority_states + second_priority_states
        request_ids_to_remove_from_waiting = set()
        safety_margins = self.safety_margin * self.cache.num_blocks

        for state in candidates:
            # If we are out the safety margin, we only accept decoding requests or the first prefill request
            num_free_blocks = self.cache.get_num_free_blocks()
            outside_safety_margin = num_free_blocks < safety_margins
            if outside_safety_margin and scheduled_requests and state.status != RequestStatus.DECODING:
                break

            self._prepare_request_for_processing(state, token_budget, request_ids_to_remove_from_waiting)
            request_len = len(state.prompt_ids)
            if not self._allocate_blocks_if_needed(
                state, len(state.prompt_ids)
            ):  # don't schedule if we can't allocate blocks
                if len(self.cache._free_blocks) == 0:
                    break
                continue

            @traced
            def _add_to_scheduled_requests(state: RequestState):
                scheduled_requests.append(state)

            _add_to_scheduled_requests(state)

            token_budget -= request_len

            @traced
            def _remove_from_waiting_requests(state: RequestState):
                req_id = state.request_id
                if req_id in self.waiting_requests:
                    del self.waiting_requests[req_id]
                    request_ids_to_remove_from_waiting.add(req_id)

            _remove_from_waiting_requests(state)

            if token_budget == 0:
                break

        self.waiting_requests_order = deque(
            [req_id for req_id in self.waiting_requests_order if req_id not in request_ids_to_remove_from_waiting]
        )

        return scheduled_requests

    @traced
    def finish_request(self, request_id: str, evict_from_cache: bool = True):
        if evict_from_cache:
            self.cache.free_blocks(request_id)
            if request_id in self.active_requests:
                del self.active_requests[request_id]


@attach_tracer()
class PrefillFirstScheduler(Scheduler):
    @traced
    def _allocate_blocks_if_needed(self, state: RequestState, len_next_tokens: int):
        # 1. we check that the occupancy is less than the requested length
        # 2. we allocate enough blocks to cover the requested length
        current_len = state.current_len()
        occupancy = len(state.allocated_blocks) * self.cache.block_size - current_len
        if occupancy < len_next_tokens or (len(state.allocated_blocks) == 0):
            blocks_needed = ((len_next_tokens - occupancy + 1) // self.cache.block_size) + 1
            allocated = self.cache.allocate_blocks(blocks_needed, state.request_id)
            if not allocated:
                return False
            state.allocated_blocks.extend(allocated)
        return True

    @traced(span_name="prepare_request")
    def _prepare_request_for_processing(
        self, state: RequestState, token_budget: int, request_ids_to_remove_from_waiting: set[str]
    ):
        """Prepare a request for processing in the current batch."""
        request_tokens = (
            state.remaining_prompt_ids if state.status == RequestStatus.SPLIT_PENDING_REMAINDER else state.prompt_ids
        )
        if len(request_tokens) < token_budget:
            # Can process the entire prompt/remainder
            if state.status == RequestStatus.PENDING:
                self.active_requests[state.request_id] = state
                state.status = RequestStatus.PREFILLING
                request_ids_to_remove_from_waiting.add(state.request_id)
            elif state.status == RequestStatus.SPLIT_PENDING_REMAINDER:
                state.status = RequestStatus.PREFILLING
                state.prompt_ids = state.remaining_prompt_ids
                state.remaining_prompt_ids = []
        else:
            # Need to split the request
            if state.status == RequestStatus.PENDING:
                self.active_requests[state.request_id] = state
                state.status = RequestStatus.PREFILLING_SPLIT
                request_ids_to_remove_from_waiting.add(state.request_id)
            elif state.status == RequestStatus.SPLIT_PENDING_REMAINDER:
                state.status = RequestStatus.PREFILLING_SPLIT
            state.remaining_prompt_ids = request_tokens[token_budget:]
            state.prompt_ids = request_tokens[:token_budget]

    @traced
    def add_waiting_request(self, state: RequestState):
        """Add a request to the waiting list."""
        if self.retain_cache_on_finish and state.request_id in self.active_requests:
            old_state = self.active_requests.pop(state.request_id)
            state.prompt_ids = state.prompt_ids[len(old_state.full_prompt_ids) :]  # XXX: check for indexing error?
            state.allocated_blocks = old_state.allocated_blocks
            state.position_offset = old_state.position_offset
        self.waiting_requests[state.request_id] = state
        self.waiting_requests_order.append(state.request_id)

    @traced
    def schedule_batch(self, token_budget: int) -> list[RequestState]:
        priority_states: list[RequestState] = []
        second_priority_states: list[RequestState] = []
        scheduled_requests = []

        for state in self.active_requests.values():
            if state.status == RequestStatus.SPLIT_PENDING_REMAINDER:
                priority_states.append(state)
            elif state.status == RequestStatus.DECODING:
                second_priority_states.append(state)

        for req_id in self.waiting_requests_order:
            second_priority_states.append(self.waiting_requests[req_id])

        candidates = priority_states + second_priority_states

        request_ids_to_remove_from_waiting = set()

        for state in candidates:
            self._prepare_request_for_processing(state, token_budget, request_ids_to_remove_from_waiting)
            request_len = len(state.prompt_ids)
            if not self._allocate_blocks_if_needed(
                state, len(state.prompt_ids)
            ):  # don't schedule if we can't allocate blocks
                if len(self.cache._free_blocks) == 0:
                    break
                continue

            @traced
            def _add_to_scheduled_requests(state: RequestState):
                scheduled_requests.append(state)

            _add_to_scheduled_requests(state)

            token_budget -= request_len

            @traced
            def _remove_from_waiting_requests(state: RequestState):
                req_id = state.request_id
                if req_id in self.waiting_requests:
                    del self.waiting_requests[req_id]
                    request_ids_to_remove_from_waiting.add(req_id)

            _remove_from_waiting_requests(state)

            if token_budget == 0:
                break

        self.waiting_requests_order = deque(
            [req_id for req_id in self.waiting_requests_order if req_id not in request_ids_to_remove_from_waiting]
        )

        return scheduled_requests

    @traced
    def finish_request(self, request_id: str, evict_from_cache: bool = True):
        if evict_from_cache:
            self.cache.free_blocks(request_id)
            if request_id in self.active_requests:
                del self.active_requests[request_id]


SCHEDULER_MAPPING = {
    "fifo": FIFOScheduler,
    "prefill_first": PrefillFirstScheduler,
}
