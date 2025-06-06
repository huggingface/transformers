# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team.
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

import torch
from parameterized import parameterized

from transformers.generation.continuous_batching import PagedAttentionCache, RequestState, RequestStatus
from transformers.testing_utils import require_torch


class MockConfig:
    """Mock configuration for testing sliding window cache."""

    def __init__(
        self, sliding_window=None, num_attention_heads=4, num_key_value_heads=4, hidden_size=128, num_hidden_layers=2
    ):
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_size = hidden_size
        self.head_dim = hidden_size // num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.sliding_window = sliding_window
        self._attn_implementation = "paged_attention"


class MockGenerationConfig:
    """Mock generation configuration for testing."""

    def __init__(self, sliding_window=None, num_blocks=16, block_size=8, max_new_tokens=10, eos_token_id=2):
        self.sliding_window = sliding_window
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.max_new_tokens = max_new_tokens
        self.eos_token_id = eos_token_id


@require_torch
class SlidingWindowCacheTest(unittest.TestCase):
    """Unit tests for PagedAttentionCache sliding window functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device("cpu")  # Use CPU for faster testing
        self.dtype = torch.float32
        self.sliding_window_size = 16

        # Create configs
        self.config = MockConfig(sliding_window=self.sliding_window_size)
        self.generation_config = MockGenerationConfig(sliding_window=self.sliding_window_size)

    def test_sliding_window_cache_initialization(self):
        """Test that sliding window cache initializes correctly."""
        cache = PagedAttentionCache(
            config=self.config, generation_config=self.generation_config, device=self.device, dtype=self.dtype
        )

        # Test sliding window properties
        self.assertTrue(cache.is_sliding_window)
        self.assertEqual(cache.sliding_window, self.sliding_window_size)
        self.assertEqual(cache.get_sliding_window_size(), self.sliding_window_size)

        # Test cache structure
        self.assertEqual(len(cache.key_cache), self.config.num_hidden_layers)
        self.assertEqual(len(cache.value_cache), self.config.num_hidden_layers)

        # Test cache shapes
        expected_shape = (
            self.config.num_key_value_heads,
            self.generation_config.num_blocks,
            self.generation_config.block_size,
            self.config.head_dim,
        )
        for layer_cache in cache.key_cache:
            self.assertEqual(layer_cache.shape, expected_shape)
        for layer_cache in cache.value_cache:
            self.assertEqual(layer_cache.shape, expected_shape)

    def test_sliding_window_disabled(self):
        """Test cache behavior when sliding window is disabled."""
        config = MockConfig(sliding_window=None)
        generation_config = MockGenerationConfig(sliding_window=None)

        cache = PagedAttentionCache(
            config=config, generation_config=generation_config, device=self.device, dtype=self.dtype
        )

        self.assertFalse(cache.is_sliding_window)
        self.assertIsNone(cache.sliding_window)
        self.assertIsNone(cache.get_sliding_window_size())

    @parameterized.expand(
        [
            (0, "must be positive"),
            (-1, "must be positive"),
            (-10, "must be positive"),
        ]
    )
    def test_invalid_sliding_window_sizes(self, window_size, expected_error):
        """Test that invalid sliding window sizes raise appropriate errors."""
        config = MockConfig(sliding_window=window_size)
        generation_config = MockGenerationConfig(sliding_window=window_size)

        with self.assertRaises(ValueError) as cm:
            PagedAttentionCache(
                config=config, generation_config=generation_config, device=self.device, dtype=self.dtype
            )

        self.assertIn(expected_error, str(cm.exception))

    def test_block_allocation_and_deallocation(self):
        """Test block allocation and deallocation with sliding window."""
        cache = PagedAttentionCache(
            config=self.config, generation_config=self.generation_config, device=self.device, dtype=self.dtype
        )

        # Test initial state
        self.assertEqual(cache.get_num_free_blocks(), self.generation_config.num_blocks)

        # Test block allocation
        request_id = "test_request"
        allocated_blocks = cache.allocate_blocks(4, request_id)
        self.assertEqual(len(allocated_blocks), 4)
        self.assertEqual(cache.get_num_free_blocks(), self.generation_config.num_blocks - 4)

        # Test block table
        block_table = cache.get_block_table(request_id)
        self.assertEqual(block_table, allocated_blocks)

        # Test block deallocation
        cache.free_blocks(request_id)
        self.assertEqual(cache.get_num_free_blocks(), self.generation_config.num_blocks)
        self.assertEqual(cache.get_block_table(request_id), [])

    def test_get_sliding_window_indices(self):
        """Test sliding window indices calculation."""
        cache = PagedAttentionCache(
            config=self.config, generation_config=self.generation_config, device=self.device, dtype=self.dtype
        )

        # Test case 1: Sequence within sliding window
        logical_indices = [0, 1, 2, 3, 4, 5]
        current_seq_len = 10  # Less than sliding_window_size (16)
        result = cache._get_sliding_window_indices(logical_indices, current_seq_len)
        self.assertEqual(result, logical_indices)

        # Test case 2: Sequence exactly at sliding window size
        logical_indices = list(range(16))
        current_seq_len = 16
        result = cache._get_sliding_window_indices(logical_indices, current_seq_len)
        self.assertEqual(result, logical_indices)

        # Test case 3: Sequence exceeds sliding window
        logical_indices = list(range(24))  # 0 to 23
        current_seq_len = 24
        result = cache._get_sliding_window_indices(logical_indices, current_seq_len)
        # Should return only last 16 positions (24-16=8 to 23)
        expected = list(range(8, 24))
        self.assertEqual(result, expected)
        self.assertEqual(len(result), self.sliding_window_size)

        # Test case 4: Empty logical indices
        result = cache._get_sliding_window_indices([], current_seq_len)
        self.assertEqual(result, [])

    def test_sliding_window_update_error_handling(self):
        """Test error handling in sliding window update."""
        cache = PagedAttentionCache(
            config=self.config, generation_config=self.generation_config, device=self.device, dtype=self.dtype
        )

        # Create dummy tensors
        key_states = torch.randn(1, 4, 2, 32, dtype=self.dtype, device=self.device)
        value_states = torch.randn(1, 4, 2, 32, dtype=self.dtype, device=self.device)
        write_index = torch.tensor([0, 1], device=self.device)
        read_index = torch.tensor([0, 1], device=self.device)

        # Test missing cumulative_seqlens_k
        with self.assertRaises(ValueError) as cm:
            cache.update(
                key_states=key_states,
                value_states=value_states,
                layer_idx=0,
                read_index=read_index,
                write_index=write_index,
            )
        self.assertIn("cumulative_seqlens_k", str(cm.exception))

        # Test empty cumulative_seqlens_k
        with self.assertRaises(ValueError) as cm:
            cache.update(
                key_states=key_states,
                value_states=value_states,
                layer_idx=0,
                read_index=read_index,
                write_index=write_index,
                cumulative_seqlens_k=torch.tensor([], device=self.device),
            )
        self.assertIn("cumulative_seqlens_k", str(cm.exception))

    def test_sliding_window_update_valid_case(self):
        """Test valid sliding window update operation."""
        cache = PagedAttentionCache(
            config=self.config, generation_config=self.generation_config, device=self.device, dtype=self.dtype
        )

        # Allocate blocks
        request_id = "test_request"
        allocated_blocks = cache.allocate_blocks(2, request_id)
        self.assertTrue(allocated_blocks)

        # Create valid tensors
        seq_len = 4
        key_states = torch.randn(1, 4, seq_len, 32, dtype=self.dtype, device=self.device)
        value_states = torch.randn(1, 4, seq_len, 32, dtype=self.dtype, device=self.device)
        write_index = torch.arange(seq_len, device=self.device)
        read_index = torch.arange(seq_len, device=self.device)
        cumulative_seqlens_k = torch.tensor([0, seq_len], device=self.device)

        # Perform update
        k_out, v_out = cache.update(
            key_states=key_states,
            value_states=value_states,
            layer_idx=0,
            read_index=read_index,
            write_index=write_index,
            cumulative_seqlens_k=cumulative_seqlens_k,
        )

        # Verify output shapes
        self.assertEqual(k_out.shape[0], 1)  # batch size
        self.assertEqual(k_out.shape[1], 4)  # num_heads
        self.assertEqual(k_out.shape[2], seq_len)  # sequence length
        self.assertEqual(k_out.shape[3], 32)  # head_dim
        self.assertEqual(v_out.shape, k_out.shape)

    def test_non_sliding_window_update(self):
        """Test update operation when sliding window is disabled."""
        config = MockConfig(sliding_window=None)
        generation_config = MockGenerationConfig(sliding_window=None)

        cache = PagedAttentionCache(
            config=config, generation_config=generation_config, device=self.device, dtype=self.dtype
        )

        # Allocate blocks
        request_id = "test_request"
        allocated_blocks = cache.allocate_blocks(2, request_id)
        self.assertTrue(allocated_blocks)

        # Create tensors
        seq_len = 4
        key_states = torch.randn(1, 4, seq_len, 32, dtype=self.dtype, device=self.device)
        value_states = torch.randn(1, 4, seq_len, 32, dtype=self.dtype, device=self.device)
        write_index = torch.arange(seq_len, device=self.device)
        read_index = torch.arange(seq_len, device=self.device)

        # Perform update (should not require cumulative_seqlens_k)
        k_out, v_out = cache.update(
            key_states=key_states,
            value_states=value_states,
            layer_idx=0,
            read_index=read_index,
            write_index=write_index,
        )

        # Verify output shapes
        self.assertEqual(k_out.shape[2], seq_len)
        self.assertEqual(v_out.shape[2], seq_len)


@require_torch
class RequestStateTest(unittest.TestCase):
    """Test RequestState behavior with sliding window scenarios."""

    def test_request_state_basic_functionality(self):
        """Test basic RequestState functionality."""
        state = RequestState(
            request_id="test_1",
            prompt_ids=[1, 2, 3, 4],
            full_prompt_ids=[1, 2, 3, 4],
            max_new_tokens=5,
            eos_token_id=2,
        )

        # Test initial state
        self.assertEqual(state.current_len(), 0)
        self.assertEqual(state.generated_len(), 0)
        self.assertEqual(state.status, RequestStatus.PENDING)

        # Test position tracking
        state.position_offset = 10
        self.assertEqual(state.current_len(), 10)

    def test_request_state_token_generation(self):
        """Test token generation and completion detection."""
        state = RequestState(
            request_id="test_2", prompt_ids=[1, 2, 3], full_prompt_ids=[1, 2, 3], max_new_tokens=3, eos_token_id=5
        )

        # Move to decoding state
        state.status = RequestStatus.DECODING

        # Generate some tokens
        state.static_outputs = [10, 11]
        self.assertEqual(state.generated_len(), 2)

        # Test non-completion cases
        is_complete = state.update_with_token(12)
        self.assertFalse(is_complete)
        self.assertEqual(state.status, RequestStatus.DECODING)

        # Test EOS completion
        is_complete = state.update_with_token(5)  # EOS token
        self.assertTrue(is_complete)
        self.assertEqual(state.status, RequestStatus.FINISHED)

    def test_request_state_max_length_completion(self):
        """Test completion by reaching max_new_tokens."""
        state = RequestState(
            request_id="test_3", prompt_ids=[1, 2], full_prompt_ids=[1, 2], max_new_tokens=2, eos_token_id=5
        )

        state.status = RequestStatus.DECODING
        state.static_outputs = [10]  # 1 token generated

        # Add second token to static_outputs (simulating the actual system behavior)
        state.static_outputs.append(11)

        # Check for completion (should reach max_new_tokens)
        is_complete = state.update_with_token(11)
        self.assertTrue(is_complete)
        self.assertEqual(state.status, RequestStatus.FINISHED)
        self.assertEqual(state.generated_len(), 2)

    def test_request_state_to_generation_output(self):
        """Test conversion to GenerationOutput."""
        state = RequestState(
            request_id="test_4",
            prompt_ids=[1, 2, 3],
            full_prompt_ids=[1, 2, 3, 4],  # Different from prompt_ids for testing
            max_new_tokens=5,
            eos_token_id=2,
        )

        state.status = RequestStatus.FINISHED
        state.static_outputs = [10, 11, 12]

        output = state.to_generation_output()

        self.assertEqual(output.request_id, "test_4")
        self.assertEqual(output.prompt_ids, [1, 2, 3, 4])  # Should use full_prompt_ids
        self.assertEqual(output.generated_tokens, [10, 11, 12])
        self.assertEqual(output.status, RequestStatus.FINISHED)
        self.assertIsNone(output.error)


if __name__ == "__main__":
    unittest.main()
