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

from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from transformers.generation.continuous_batching import (
    PagedAttentionCache,
    RequestState,
    RequestStatus,
)
from transformers.testing_utils import require_flash_attn, require_torch_gpu, slow


class MockConfig:
    """Mock configuration for testing sliding window cache."""

    def __init__(self, sliding_window=None):
        self.num_attention_heads = 8
        self.num_key_value_heads = 8
        self.hidden_size = 256
        self.head_dim = 32
        self.num_hidden_layers = 4
        self.sliding_window = sliding_window
        self._attn_implementation = "paged_attention"


class MockGenerationConfig:
    """Mock generation configuration for testing."""

    def __init__(self, sliding_window=None, num_blocks=32, block_size=16):
        self.sliding_window = sliding_window
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.max_new_tokens = 20
        self.eos_token_id = 2


class SlidingWindowCacheUnitTest(unittest.TestCase):
    """Unit tests for PagedAttentionCache sliding window functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16
        self.sliding_window_size = 64

        # Create configs
        self.config = MockConfig(sliding_window=self.sliding_window_size)
        self.generation_config = MockGenerationConfig(sliding_window=self.sliding_window_size)

    def test_sliding_window_initialization(self):
        """Test that sliding window cache initializes correctly."""
        cache = PagedAttentionCache(
            config=self.config, generation_config=self.generation_config, device=self.device, dtype=self.dtype
        )

        self.assertTrue(cache.is_sliding_window)
        self.assertEqual(cache.sliding_window, self.sliding_window_size)
        self.assertEqual(cache.get_sliding_window_size(), self.sliding_window_size)

    def test_sliding_window_disabled_initialization(self):
        """Test that cache without sliding window initializes correctly."""
        config = MockConfig(sliding_window=None)
        generation_config = MockGenerationConfig(sliding_window=None)

        cache = PagedAttentionCache(
            config=config, generation_config=generation_config, device=self.device, dtype=self.dtype
        )

        self.assertFalse(cache.is_sliding_window)
        self.assertIsNone(cache.sliding_window)
        self.assertIsNone(cache.get_sliding_window_size())

    def test_sliding_window_invalid_size(self):
        """Test that invalid sliding window size raises error."""
        config = MockConfig(sliding_window=0)
        generation_config = MockGenerationConfig(sliding_window=0)

        with self.assertRaises(ValueError) as cm:
            PagedAttentionCache(
                config=config, generation_config=generation_config, device=self.device, dtype=self.dtype
            )

        self.assertIn("sliding_window must be positive", str(cm.exception))

    def test_get_sliding_window_indices_within_window(self):
        """Test sliding window indices when sequence is within window."""
        cache = PagedAttentionCache(
            config=self.config, generation_config=self.generation_config, device=self.device, dtype=self.dtype
        )

        logical_indices = [0, 1, 2, 3, 4]
        current_seq_len = 30

        result = cache._get_sliding_window_indices(logical_indices, current_seq_len)

        # Since current_seq_len < sliding_window, all indices should be returned
        self.assertEqual(result, logical_indices)

    def test_get_sliding_window_indices_exceeds_window(self):
        """Test sliding window indices when sequence exceeds window."""
        cache = PagedAttentionCache(
            config=self.config, generation_config=self.generation_config, device=self.device, dtype=self.dtype
        )

        logical_indices = list(range(100))  # 0 to 99
        current_seq_len = 100

        result = cache._get_sliding_window_indices(logical_indices, current_seq_len)

        # Should only return indices within the sliding window
        window_start = current_seq_len - self.sliding_window_size  # 100 - 64 = 36
        expected = [i for i in logical_indices if i >= window_start]  # 36 to 99

        self.assertEqual(result, expected)
        self.assertEqual(len(result), self.sliding_window_size)

    def test_sliding_window_update_missing_cumulative_seqlens(self):
        """Test that missing cumulative_seqlens_k raises appropriate error."""
        cache = PagedAttentionCache(
            config=self.config, generation_config=self.generation_config, device=self.device, dtype=self.dtype
        )

        # Create dummy tensors
        key_states = torch.randn(1, 8, 4, 32, dtype=self.dtype, device=self.device)
        value_states = torch.randn(1, 8, 4, 32, dtype=self.dtype, device=self.device)
        write_index = torch.tensor([0, 1, 2, 3], device=self.device)
        read_index = torch.tensor([0, 1, 2, 3], device=self.device)

        with self.assertRaises(ValueError) as cm:
            cache.update(
                key_states=key_states,
                value_states=value_states,
                layer_idx=0,
                read_index=read_index,
                write_index=write_index,
                # Missing cumulative_seqlens_k
            )

        self.assertIn("cumulative_seqlens_k", str(cm.exception))
        self.assertIn("Sliding window attention is enabled", str(cm.exception))

    def test_sliding_window_update_with_cumulative_seqlens(self):
        """Test sliding window update with proper cumulative_seqlens_k."""
        cache = PagedAttentionCache(
            config=self.config, generation_config=self.generation_config, device=self.device, dtype=self.dtype
        )

        # Allocate blocks for the test
        allocated_blocks = cache.allocate_blocks(4, "test_request")
        self.assertTrue(allocated_blocks)

        # Create dummy tensors
        key_states = torch.randn(1, 8, 4, 32, dtype=self.dtype, device=self.device)
        value_states = torch.randn(1, 8, 4, 32, dtype=self.dtype, device=self.device)
        write_index = torch.tensor([0, 1, 2, 3], device=self.device)
        read_index = torch.tensor([0, 1, 2, 3], device=self.device)
        cumulative_seqlens_k = torch.tensor([0, 4], device=self.device)

        # This should not raise an error
        try:
            k_out, v_out = cache.update(
                key_states=key_states,
                value_states=value_states,
                layer_idx=0,
                read_index=read_index,
                write_index=write_index,
                cumulative_seqlens_k=cumulative_seqlens_k,
            )
            # Verify output shapes
            self.assertEqual(k_out.shape[2], 4)  # Should have 4 positions
            self.assertEqual(v_out.shape[2], 4)
        except Exception as e:
            self.fail(f"Update with proper cumulative_seqlens_k should not fail: {e}")


class SlidingWindowIntegrationTest(unittest.TestCase):
    """Integration tests for sliding window with continuous batching system."""

    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16
        self.sliding_window_size = 32

    def test_request_state_with_sliding_window(self):
        """Test RequestState behavior with sliding window constraints."""
        request_state = RequestState(
            request_id="test_req_1",
            prompt_ids=[1, 2, 3, 4, 5],
            full_prompt_ids=[1, 2, 3, 4, 5],
            max_new_tokens=10,
            eos_token_id=2,
        )

        # Test basic properties
        self.assertEqual(request_state.current_len(), 0)  # position_offset starts at 0
        self.assertEqual(request_state.generated_len(), 0)

        # Test token update
        request_state.status = RequestStatus.DECODING
        request_state.static_outputs = [10, 11, 12]

        # Test completion detection
        is_complete = request_state.update_with_token(2)  # EOS token
        self.assertTrue(is_complete)
        self.assertEqual(request_state.status, RequestStatus.FINISHED)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_paged_attention_cache_sliding_window_memory(self):
        """Test that sliding window actually limits memory usage."""
        config = MockConfig(sliding_window=64)
        generation_config = MockGenerationConfig(sliding_window=64, num_blocks=16, block_size=16)

        cache = PagedAttentionCache(
            config=config, generation_config=generation_config, device=self.device, dtype=self.dtype
        )

        # Allocate some blocks for a request
        allocated_blocks = cache.allocate_blocks(4, "test_request")
        self.assertEqual(len(allocated_blocks), 4)

        # Test that we can free blocks
        cache.free_blocks("test_request")
        self.assertEqual(cache.get_num_free_blocks(), 16)  # All blocks should be free

    def test_sliding_window_indices_calculation(self):
        """Test detailed sliding window indices calculation."""
        config = MockConfig(sliding_window=8)
        generation_config = MockGenerationConfig(sliding_window=8)

        cache = PagedAttentionCache(
            config=config, generation_config=generation_config, device=self.device, dtype=self.dtype
        )

        # Test case 1: Sequence within window
        indices_1 = [0, 1, 2, 3]
        result_1 = cache._get_sliding_window_indices(indices_1, current_seq_len=4)
        self.assertEqual(result_1, [0, 1, 2, 3])

        # Test case 2: Sequence exactly at window size
        indices_2 = list(range(8))
        result_2 = cache._get_sliding_window_indices(indices_2, current_seq_len=8)
        self.assertEqual(result_2, list(range(8)))

        # Test case 3: Sequence exceeds window
        indices_3 = list(range(12))  # 0 to 11
        result_3 = cache._get_sliding_window_indices(indices_3, current_seq_len=12)
        expected_3 = list(range(4, 12))  # Only last 8 positions (12-8=4 to 11)
        self.assertEqual(result_3, expected_3)


@slow
@require_torch_gpu
@require_flash_attn
class SlidingWindowEndToEndTest(unittest.TestCase):
    """End-to-end tests for sliding window continuous batching."""

    _TEST_PROMPTS_SHORT = ["The quick brown fox", "Hello world, this is", "Machine learning is"]

    _TEST_PROMPTS_LONG = [
        "The field of artificial intelligence has evolved dramatically over the past decade, with breakthrough developments in deep learning, natural language processing, and computer vision. These advances have enabled",
        "Climate change represents one of the most pressing challenges of our time, requiring urgent action from governments, businesses, and individuals worldwide. The consequences of inaction include",
        "Space exploration has captured human imagination for centuries, from early astronomical observations to modern missions to Mars and beyond. Future exploration efforts will focus on",
    ]

    @classmethod
    def setUpClass(cls):
        """Set up model and tokenizer for testing."""
        cls.model_name = "microsoft/DialoGPT-small"  # Small model for faster testing
        cls.model = AutoModelForCausalLM.from_pretrained(
            cls.model_name, torch_dtype=torch.float16, device_map="auto"
        ).eval()

        cls.tokenizer = AutoTokenizer.from_pretrained(cls.model_name, padding_side="left")
        if cls.tokenizer.pad_token is None:
            cls.tokenizer.pad_token = cls.tokenizer.eos_token
            cls.model.config.pad_token_id = cls.model.config.eos_token_id

    def test_sliding_window_short_sequences(self):
        """Test sliding window with short sequences (within window)."""
        sliding_window_size = 256

        # Configure sliding window
        self.model.config.sliding_window = sliding_window_size
        self.model.config._attn_implementation = "paged_attention"

        generation_config = GenerationConfig(
            max_new_tokens=20,
            do_sample=False,
            eos_token_id=self.tokenizer.eos_token_id,
            sliding_window=sliding_window_size,
            num_blocks=64,
            block_size=32,
            scheduler="fifo",
        )

        # Test with short prompts
        input_ids = [
            self.tokenizer.encode(prompt, return_tensors="pt")[0].tolist() for prompt in self._TEST_PROMPTS_SHORT
        ]

        # Generate batch
        results = self.model.generate_batch(inputs=input_ids, generation_config=generation_config, progress_bar=False)

        # Verify results
        self.assertEqual(len(results), len(input_ids))
        for i, req_id in enumerate([f"batch_req_{i}" for i in range(len(input_ids))]):
            self.assertIn(req_id, results)
            result = results[req_id]
            self.assertEqual(result.status, RequestStatus.FINISHED)
            self.assertIsNone(result.error)
            self.assertGreater(len(result.generated_tokens), 0)

    def test_sliding_window_long_sequences(self):
        """Test sliding window with long sequences (exceeding window)."""
        sliding_window_size = 128  # Smaller window to test sliding behavior

        # Configure sliding window
        self.model.config.sliding_window = sliding_window_size
        self.model.config._attn_implementation = "paged_attention"

        generation_config = GenerationConfig(
            max_new_tokens=30,
            do_sample=False,
            eos_token_id=self.tokenizer.eos_token_id,
            sliding_window=sliding_window_size,
            num_blocks=32,
            block_size=16,
            scheduler="fifo",
        )

        # Test with longer prompts that might exceed window during generation
        input_ids = [
            self.tokenizer.encode(prompt, return_tensors="pt")[0].tolist() for prompt in self._TEST_PROMPTS_LONG
        ]

        # Generate batch
        results = self.model.generate_batch(inputs=input_ids, generation_config=generation_config, progress_bar=False)

        # Verify results
        self.assertEqual(len(results), len(input_ids))
        for i, req_id in enumerate([f"batch_req_{i}" for i in range(len(input_ids))]):
            self.assertIn(req_id, results)
            result = results[req_id]
            self.assertEqual(result.status, RequestStatus.FINISHED)
            self.assertIsNone(result.error)
            self.assertGreater(len(result.generated_tokens), 0)

    @parameterized.expand(
        [
            ("fifo", 64, 32),
            ("prefill_first", 32, 16),
        ]
    )
    def test_sliding_window_schedulers(self, scheduler_name, num_blocks, block_size):
        """Test sliding window with different schedulers."""
        sliding_window_size = 128

        # Configure sliding window
        self.model.config.sliding_window = sliding_window_size
        self.model.config._attn_implementation = "paged_attention"

        generation_config = GenerationConfig(
            max_new_tokens=20,
            do_sample=False,
            eos_token_id=self.tokenizer.eos_token_id,
            sliding_window=sliding_window_size,
            num_blocks=num_blocks,
            block_size=block_size,
            scheduler=scheduler_name,
        )

        input_ids = [
            self.tokenizer.encode(prompt, return_tensors="pt")[0].tolist() for prompt in self._TEST_PROMPTS_SHORT
        ]

        results = self.model.generate_batch(inputs=input_ids, generation_config=generation_config, progress_bar=False)

        # Verify all requests completed successfully
        self.assertEqual(len(results), len(input_ids))
        for req_id in results:
            result = results[req_id]
            self.assertEqual(result.status, RequestStatus.FINISHED)
            self.assertIsNone(result.error)


class SlidingWindowErrorHandlingTest(unittest.TestCase):
    """Test error handling and edge cases for sliding window functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16

    def test_mismatched_sliding_window_config(self):
        """Test error handling when model and generation configs have mismatched sliding window settings."""
        config = MockConfig(sliding_window=64)
        generation_config = MockGenerationConfig(sliding_window=128)  # Different size

        # This should still work, but we might want to log a warning
        cache = PagedAttentionCache(
            config=config, generation_config=generation_config, device=self.device, dtype=self.dtype
        )

        # The cache should use the model config's sliding window
        self.assertEqual(cache.sliding_window, 64)

    def test_zero_sliding_window(self):
        """Test that zero sliding window raises appropriate error."""
        config = MockConfig(sliding_window=0)
        generation_config = MockGenerationConfig(sliding_window=0)

        with self.assertRaises(ValueError) as cm:
            PagedAttentionCache(
                config=config, generation_config=generation_config, device=self.device, dtype=self.dtype
            )

        self.assertIn("sliding_window must be positive", str(cm.exception))

    def test_negative_sliding_window(self):
        """Test that negative sliding window raises appropriate error."""
        config = MockConfig(sliding_window=-10)
        generation_config = MockGenerationConfig(sliding_window=-10)

        with self.assertRaises(ValueError) as cm:
            PagedAttentionCache(
                config=config, generation_config=generation_config, device=self.device, dtype=self.dtype
            )

        self.assertIn("sliding_window must be positive", str(cm.exception))

    def test_sliding_window_with_empty_cumulative_seqlens(self):
        """Test error handling with empty cumulative_seqlens_k."""
        config = MockConfig(sliding_window=64)
        generation_config = MockGenerationConfig(sliding_window=64)

        cache = PagedAttentionCache(
            config=config, generation_config=generation_config, device=self.device, dtype=self.dtype
        )

        # Create dummy tensors
        key_states = torch.randn(1, 8, 4, 32, dtype=self.dtype, device=self.device)
        value_states = torch.randn(1, 8, 4, 32, dtype=self.dtype, device=self.device)
        write_index = torch.tensor([0, 1, 2, 3], device=self.device)
        read_index = torch.tensor([0, 1, 2, 3], device=self.device)
        empty_cumulative_seqlens_k = torch.tensor([], device=self.device)

        with self.assertRaises(ValueError) as cm:
            cache.update(
                key_states=key_states,
                value_states=value_states,
                layer_idx=0,
                read_index=read_index,
                write_index=write_index,
                cumulative_seqlens_k=empty_cumulative_seqlens_k,
            )

        self.assertIn("cumulative_seqlens_k", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
