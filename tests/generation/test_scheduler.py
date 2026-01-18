# Copyright 2025 The HuggingFace Team Inc.
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

import unittest

from transformers.generation.continuous_batching.requests import RequestState, RequestStatus
from transformers.generation.continuous_batching.scheduler import PrefillFirstScheduler


class MockCache:
    """Mock cache for scheduler unit tests that tracks block allocations."""

    def __init__(self, num_blocks: int, block_size: int):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.allocated = {}  # request_id -> num_blocks
        self.allow_block_sharing = False
        self.blocks_to_complete = {}
        self.use_prefix_sharing = False
        self.num_full_attention_groups = 1

    def get_num_free_blocks(self):
        return self.num_blocks - sum(self.allocated.values())

    def allocate_blocks(self, num_requested_blocks, request_id, allocated_blocks):
        if self.get_num_free_blocks() < num_requested_blocks:
            return None
        if request_id not in self.allocated:
            self.allocated[request_id] = 0
        self.allocated[request_id] += num_requested_blocks
        return num_requested_blocks

    def free_blocks(self, request_id):
        self.allocated.pop(request_id, None)


class SchedulerTestCase(unittest.TestCase):
    """Unit tests for Scheduler implementations."""

    def _create_mock_cache(self, num_blocks: int = 128, block_size: int = 16):
        return MockCache(num_blocks, block_size)

    def _create_request(
        self, request_id: str, prompt_length: int, max_new_tokens: int, status: RequestStatus = RequestStatus.PENDING
    ):
        """Helper to create a RequestState for testing."""
        tokens = list(range(prompt_length))
        request = RequestState(
            request_id=request_id,
            initial_tokens=tokens,
            max_new_tokens=max_new_tokens,
        )
        request.tokens_to_process = tokens.copy()
        request._status = status
        return request

    def test_prefill_first_scheduler_waiting_added_before_decoding_when_cache_allows(self):
        cache = self._create_mock_cache(num_blocks=128, block_size=16)
        scheduler = PrefillFirstScheduler(cache, retain_cache_on_finish=False)

        decoding_req = self._create_request(
            "decoding", prompt_length=8, max_new_tokens=10, status=RequestStatus.DECODING
        )
        pending_req = self._create_request("pending", prompt_length=4, max_new_tokens=10, status=RequestStatus.PENDING)

        scheduler.active_requests[decoding_req.request_id] = decoding_req
        scheduler.add_waiting_request(pending_req)

        scheduled = scheduler.schedule_batch(token_budget=64, cache_budget=10_000)

        self.assertIsNotNone(scheduled)
        self.assertGreaterEqual(len(scheduled), 2)
        self.assertEqual(scheduled[0].request_id, "pending")
        self.assertEqual(scheduled[1].request_id, "decoding")

    def test_prefill_first_scheduler_waiting_blocked_when_block_new_requests_is_true(self):
        cache = self._create_mock_cache(num_blocks=128, block_size=16)
        scheduler = PrefillFirstScheduler(cache, retain_cache_on_finish=False)
        scheduler.block_new_requests = True

        decoding_req = self._create_request(
            "decoding", prompt_length=8, max_new_tokens=10, status=RequestStatus.DECODING
        )
        pending_req = self._create_request("pending", prompt_length=4, max_new_tokens=10, status=RequestStatus.PENDING)

        scheduler.active_requests[decoding_req.request_id] = decoding_req
        scheduler.add_waiting_request(pending_req)

        scheduled = scheduler.schedule_batch(token_budget=64, cache_budget=10_000)

        self.assertIsNotNone(scheduled)
        self.assertEqual(len(scheduled), 1)
        self.assertEqual(scheduled[0].request_id, "decoding")


if __name__ == "__main__":
    unittest.main()
