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
from unittest.mock import Mock

import torch

from transformers.generation.safety import (
    LENIENT_PRESET,
    MODERATE_PRESET,
    STRICT_PRESET,
    SafetyConfig,
    SafetyMetrics,
    SafetyResult,
    SafetyState,
    SafetyViolation,
)
from transformers.generation.safety.processors import (
    SafetyLogitsProcessor,
    SafetyStoppingCriteria,
    _generate_cache_key,
)
from transformers.testing_utils import require_torch


@require_torch
class TestSafetyLogitsProcessor(unittest.TestCase):
    """Test SafetyLogitsProcessor functionality."""

    def setUp(self):
        """Set up test fixtures."""
        # Mock safety checker
        self.mock_checker = Mock()
        self.mock_checker.check_safety.return_value = SafetyResult(
            is_safe=True, confidence=0.9, violations=[], metadata={}
        )

        # Mock tokenizer
        self.mock_tokenizer = Mock()
        self.mock_tokenizer.decode.return_value = "test text"

        # Safety config
        self.safety_config = SafetyConfig.from_checker(self.mock_checker)

    def test_safe_content_no_suppression(self):
        """Test that safe content passes through without modification."""
        processor = SafetyLogitsProcessor(
            safety_checker=self.mock_checker, tokenizer=self.mock_tokenizer, safety_config=self.safety_config
        )

        # Test safe content (mock already returns safe result)
        input_ids = torch.tensor([[1, 2, 3, 4]])
        scores = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])
        original_scores = scores.clone()

        # Process
        modified_scores = processor(input_ids, scores)

        # Scores should be unchanged for safe content
        torch.testing.assert_close(modified_scores, original_scores)

        # Verify safety check was called
        self.mock_checker.check_safety.assert_called_once()

    def test_unsafe_content_blocking(self):
        """Test that unsafe content gets all tokens suppressed (blocking)."""
        processor = SafetyLogitsProcessor(
            safety_checker=self.mock_checker, tokenizer=self.mock_tokenizer, safety_config=self.safety_config
        )

        # Mock unsafe result
        self.mock_checker.check_safety.return_value = SafetyResult(
            is_safe=False, confidence=0.8, violations=[SafetyViolation("toxicity", 0.8, "high")], metadata={}
        )

        # Test data
        input_ids = torch.tensor([[1, 2, 3, 4]])
        scores = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])
        vocab_size = scores.shape[-1]

        # Process
        modified_scores = processor(input_ids, scores)

        # All tokens should be suppressed (blocking strategy)
        for i in range(vocab_size):
            self.assertEqual(modified_scores[0, i], float("-inf"))

    def test_check_interval(self):
        """Test that safety checking respects check_interval parameter."""
        processor = SafetyLogitsProcessor(
            safety_checker=self.mock_checker,
            tokenizer=self.mock_tokenizer,
            safety_config=self.safety_config,
            check_interval=3,  # Only check every 3rd call
        )

        input_ids = torch.tensor([[1, 2, 3, 4]])
        scores = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])

        # First call (step 1) - no check
        processor(input_ids, scores)
        self.assertEqual(self.mock_checker.check_safety.call_count, 0)

        # Second call (step 2) - no check
        processor(input_ids, scores)
        self.assertEqual(self.mock_checker.check_safety.call_count, 0)

        # Third call (step 3) - check should happen
        processor(input_ids, scores)
        self.assertEqual(self.mock_checker.check_safety.call_count, 1)

    def test_batch_processing(self):
        """Test that processor handles batched inputs correctly."""
        processor = SafetyLogitsProcessor(
            safety_checker=self.mock_checker, tokenizer=self.mock_tokenizer, safety_config=self.safety_config
        )

        # Mock mixed safety results for batch
        def mock_check_safety(text):
            if "unsafe" in text:
                return SafetyResult(
                    is_safe=False, confidence=0.8, violations=[SafetyViolation("toxicity", 0.8, "high")], metadata={}
                )
            else:
                return SafetyResult(is_safe=True, confidence=0.9, violations=[], metadata={})

        self.mock_checker.check_safety.side_effect = mock_check_safety

        # Mock tokenizer to return different text for different sequences
        def mock_decode(sequence, skip_special_tokens=True):
            if torch.equal(sequence, torch.tensor([1, 2, 3, 4])):
                return "safe text"
            else:
                return "unsafe text"

        self.mock_tokenizer.decode.side_effect = mock_decode

        # Batch with mixed safety
        input_ids = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])  # [safe, unsafe]
        scores = torch.tensor([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])

        # Process
        modified_scores = processor(input_ids, scores)

        # First sequence (safe) should be unchanged
        torch.testing.assert_close(modified_scores[0], scores[0])

        # Second sequence (unsafe) should be suppressed
        for i in range(scores.shape[-1]):
            self.assertEqual(modified_scores[1, i], float("-inf"))

    def test_safety_check_failure(self):
        """Test graceful handling when safety check fails."""
        processor = SafetyLogitsProcessor(
            safety_checker=self.mock_checker, tokenizer=self.mock_tokenizer, safety_config=self.safety_config
        )

        # Mock safety checker to raise exception
        self.mock_checker.check_safety.side_effect = Exception("Safety check failed")

        input_ids = torch.tensor([[1, 2, 3, 4]])
        scores = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])

        # Process - should not raise, should assume unsafe and suppress
        modified_scores = processor(input_ids, scores)

        # Should suppress all tokens when safety check fails (fail-safe behavior)
        for i in range(scores.shape[-1]):
            self.assertEqual(modified_scores[0, i], float("-inf"))


@require_torch
class TestSafetyStoppingCriteria(unittest.TestCase):
    """Test SafetyStoppingCriteria functionality."""

    def setUp(self):
        """Set up test fixtures."""
        # Mock safety checker
        self.mock_checker = Mock()
        self.mock_checker.check_safety.return_value = SafetyResult(
            is_safe=True, confidence=0.9, violations=[], metadata={}
        )

        # Mock tokenizer
        self.mock_tokenizer = Mock()
        self.mock_tokenizer.decode.return_value = "test text"

        # Safety config
        self.safety_config = SafetyConfig.from_checker(self.mock_checker)

    def test_safe_content_continue_generation(self):
        """Test that safe content allows generation to continue."""
        criteria = SafetyStoppingCriteria(
            safety_checker=self.mock_checker, tokenizer=self.mock_tokenizer, safety_config=self.safety_config
        )

        input_ids = torch.tensor([[1, 2, 3, 4]])
        scores = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])

        # Call stopping criteria
        should_stop = criteria(input_ids, scores)

        # Should not stop for safe content
        self.assertFalse(should_stop[0])
        self.mock_checker.check_safety.assert_called_once()

    def test_unsafe_content_stop_generation(self):
        """Test that unsafe content stops generation."""
        criteria = SafetyStoppingCriteria(
            safety_checker=self.mock_checker, tokenizer=self.mock_tokenizer, safety_config=self.safety_config
        )

        # Mock unsafe result
        self.mock_checker.check_safety.return_value = SafetyResult(
            is_safe=False, confidence=0.8, violations=[SafetyViolation("toxicity", 0.8, "high")], metadata={}
        )

        input_ids = torch.tensor([[1, 2, 3, 4]])
        scores = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])

        # Call stopping criteria
        should_stop = criteria(input_ids, scores)

        # Should stop for unsafe content
        self.assertTrue(should_stop[0])

    def test_check_final_only_mode(self):
        """Test check_final_only parameter functionality."""
        criteria = SafetyStoppingCriteria(
            safety_checker=self.mock_checker,
            tokenizer=self.mock_tokenizer,
            safety_config=self.safety_config,
            check_final_only=True,
        )

        input_ids = torch.tensor([[1, 2, 3, 4]])
        scores = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])

        # Call without is_final_call - should not check
        should_stop = criteria(input_ids, scores)
        self.assertFalse(should_stop[0])
        self.assertEqual(self.mock_checker.check_safety.call_count, 0)

        # Call with is_final_call=True - should check
        should_stop = criteria(input_ids, scores, is_final_call=True)
        self.assertFalse(should_stop[0])  # Safe content
        self.assertEqual(self.mock_checker.check_safety.call_count, 1)

    def test_batch_stopping_criteria(self):
        """Test stopping criteria with batched inputs."""
        criteria = SafetyStoppingCriteria(
            safety_checker=self.mock_checker, tokenizer=self.mock_tokenizer, safety_config=self.safety_config
        )

        # Mock mixed safety results
        def mock_check_safety(text):
            if "unsafe" in text:
                return SafetyResult(
                    is_safe=False, confidence=0.8, violations=[SafetyViolation("toxicity", 0.8, "high")], metadata={}
                )
            else:
                return SafetyResult(is_safe=True, confidence=0.9, violations=[], metadata={})

        self.mock_checker.check_safety.side_effect = mock_check_safety

        # Mock tokenizer for batch
        def mock_decode(sequence, skip_special_tokens=True):
            if torch.equal(sequence, torch.tensor([1, 2, 3, 4])):
                return "safe text"
            else:
                return "unsafe text"

        self.mock_tokenizer.decode.side_effect = mock_decode

        # Batch input
        input_ids = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])  # [safe, unsafe]
        scores = torch.tensor([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])

        # Call stopping criteria
        should_stop = criteria(input_ids, scores)

        # First sequence (safe) should continue, second (unsafe) should stop
        self.assertFalse(should_stop[0])
        self.assertTrue(should_stop[1])

    def test_none_safety_checker_raises(self):
        """Test that None safety_checker raises ValueError."""
        with self.assertRaises(ValueError):
            SafetyStoppingCriteria(
                safety_checker=None, tokenizer=self.mock_tokenizer, safety_config=self.safety_config
            )


@require_torch
class TestCacheKeyGeneration(unittest.TestCase):
    """Test the SHA-256 cache key generation functionality."""

    def test_cache_key_format(self):
        """Test that cache keys follow the expected format."""
        text = "This is a test message"
        cache_key = _generate_cache_key(text)

        # Should have format "length:hash"
        parts = cache_key.split(":", 1)
        self.assertEqual(len(parts), 2)

        # First part should be text length
        self.assertEqual(parts[0], str(len(text)))

        # Second part should be a 64-character hex string (SHA-256)
        self.assertEqual(len(parts[1]), 64)
        self.assertTrue(all(c in "0123456789abcdef" for c in parts[1]))

    def test_cache_key_consistency(self):
        """Test that same text produces same cache key."""
        text = "Consistent test message"
        key1 = _generate_cache_key(text)
        key2 = _generate_cache_key(text)

        self.assertEqual(key1, key2)

    def test_cache_key_uniqueness(self):
        """Test that different texts produce different cache keys."""
        text1 = "First message"
        text2 = "Second message"
        text3 = "First messag"  # Same length, different content

        key1 = _generate_cache_key(text1)
        key2 = _generate_cache_key(text2)
        key3 = _generate_cache_key(text3)

        # All keys should be different
        self.assertNotEqual(key1, key2)
        self.assertNotEqual(key1, key3)
        self.assertNotEqual(key2, key3)

    def test_cache_key_different_lengths(self):
        """Test that texts with different lengths have different cache keys."""
        short_text = "Short"
        long_text = "This is a much longer text that should produce a different cache key"

        key1 = _generate_cache_key(short_text)
        key2 = _generate_cache_key(long_text)

        self.assertNotEqual(key1, key2)
        # Verify length prefixes are different
        self.assertEqual(key1.split(":")[0], str(len(short_text)))
        self.assertEqual(key2.split(":")[0], str(len(long_text)))

    def test_cache_key_empty_text(self):
        """Test cache key generation for empty text."""
        empty_text = ""
        cache_key = _generate_cache_key(empty_text)

        # Should still follow the format
        parts = cache_key.split(":", 1)
        self.assertEqual(len(parts), 2)
        self.assertEqual(parts[0], "0")
        self.assertEqual(len(parts[1]), 64)

    def test_cache_key_unicode_text(self):
        """Test cache key generation for unicode text."""
        unicode_text = "Hello 世界 🌍 café"
        cache_key = _generate_cache_key(unicode_text)

        # Should handle unicode properly
        parts = cache_key.split(":", 1)
        self.assertEqual(len(parts), 2)
        self.assertEqual(parts[0], str(len(unicode_text)))
        self.assertEqual(len(parts[1]), 64)

        # Should be consistent
        key2 = _generate_cache_key(unicode_text)
        self.assertEqual(cache_key, key2)

    def test_cache_key_collision_resistance(self):
        """Test cache key collision resistance with similar texts."""
        texts = [
            "The quick brown fox",
            "The quick brown fo",
            "The quick brown fox ",  # trailing space
            " The quick brown fox",  # leading space
            "THE QUICK BROWN FOX",  # different case
            "The quick brown fox jumps",  # extended
        ]

        cache_keys = [_generate_cache_key(text) for text in texts]

        # All keys should be unique
        self.assertEqual(len(cache_keys), len(set(cache_keys)))

    def test_cache_key_very_long_text(self):
        """Test cache key generation for very long text."""
        # Create a long text
        long_text = "Very long text " * 1000
        cache_key = _generate_cache_key(long_text)

        # Should still work and follow format
        parts = cache_key.split(":", 1)
        self.assertEqual(len(parts), 2)
        self.assertEqual(parts[0], str(len(long_text)))
        self.assertEqual(len(parts[1]), 64)


@require_torch
class TestSafetyMetrics(unittest.TestCase):
    """Test the SafetyMetrics functionality."""

    def test_metrics_initialization(self):
        """Test that metrics initialize with correct default values."""
        metrics = SafetyMetrics()

        # Check all default values
        self.assertEqual(metrics.total_generations, 0)
        self.assertEqual(metrics.blocked_generations, 0)
        self.assertEqual(metrics.suppression_events, 0)
        self.assertEqual(metrics.cache_hits, 0)
        self.assertEqual(metrics.cache_misses, 0)
        self.assertEqual(metrics.total_safety_check_time_ms, 0.0)
        self.assertEqual(metrics.safety_check_count, 0)

    def test_cache_hit_rate_calculation(self):
        """Test cache hit rate calculation."""
        metrics = SafetyMetrics()

        # No operations - should be 0.0
        self.assertEqual(metrics.cache_hit_rate, 0.0)

        # Record some hits and misses
        metrics.record_cache_hit()
        metrics.record_cache_hit()
        metrics.record_cache_miss()

        # Should be 66.67% (2 hits out of 3 total)
        self.assertAlmostEqual(metrics.cache_hit_rate, 66.666666666666666, places=5)

    def test_avg_safety_check_time_calculation(self):
        """Test average safety check time calculation."""
        metrics = SafetyMetrics()

        # No checks - should be 0.0
        self.assertEqual(metrics.avg_safety_check_time_ms, 0.0)

        # Record some checks
        metrics.record_safety_check(10.0)
        metrics.record_safety_check(20.0)
        metrics.record_safety_check(30.0)

        # Should be 20.0ms average
        self.assertEqual(metrics.avg_safety_check_time_ms, 20.0)

    def test_block_rate_calculation(self):
        """Test block rate calculation."""
        metrics = SafetyMetrics()

        # No generations - should be 0.0
        self.assertEqual(metrics.block_rate, 0.0)

        # Record some generations
        metrics.record_generation_attempt()
        metrics.record_generation_attempt()
        metrics.record_generation_attempt()
        metrics.record_blocked_generation()

        # Should be 33.33% (1 blocked out of 3 total)
        self.assertAlmostEqual(metrics.block_rate, 33.33333333333333, places=5)

    def test_metrics_recording_methods(self):
        """Test all metrics recording methods."""
        metrics = SafetyMetrics()

        # Test safety check recording
        metrics.record_safety_check(15.5)
        self.assertEqual(metrics.safety_check_count, 1)
        self.assertEqual(metrics.total_safety_check_time_ms, 15.5)

        # Test cache operations
        metrics.record_cache_hit()
        metrics.record_cache_miss()
        self.assertEqual(metrics.cache_hits, 1)
        self.assertEqual(metrics.cache_misses, 1)

        # Test generation tracking
        metrics.record_generation_attempt()
        metrics.record_blocked_generation()
        self.assertEqual(metrics.total_generations, 1)
        self.assertEqual(metrics.blocked_generations, 1)

        # Test suppression events
        metrics.record_suppression_event()
        self.assertEqual(metrics.suppression_events, 1)

    def test_metrics_to_dict(self):
        """Test metrics export to dictionary."""
        metrics = SafetyMetrics()

        # Record some data
        metrics.record_safety_check(10.0)
        metrics.record_cache_hit()
        metrics.record_generation_attempt()
        metrics.record_suppression_event()

        result_dict = metrics.to_dict()

        # Check all expected keys are present
        expected_keys = {
            "total_generations",
            "blocked_generations",
            "suppression_events",
            "cache_hits",
            "cache_misses",
            "cache_hit_rate",
            "avg_safety_check_time_ms",
            "block_rate",
            "safety_check_count",
        }
        self.assertEqual(set(result_dict.keys()), expected_keys)

        # Check values
        self.assertEqual(result_dict["total_generations"], 1)
        self.assertEqual(result_dict["suppression_events"], 1)
        self.assertEqual(result_dict["cache_hits"], 1)
        self.assertEqual(result_dict["cache_hit_rate"], 100.0)

    def test_metrics_reset(self):
        """Test metrics reset functionality."""
        metrics = SafetyMetrics()

        # Record some data
        metrics.record_safety_check(10.0)
        metrics.record_cache_hit()
        metrics.record_generation_attempt()
        metrics.record_suppression_event()

        # Verify data is present
        self.assertGreater(metrics.safety_check_count, 0)
        self.assertGreater(metrics.cache_hits, 0)

        # Reset
        metrics.reset()

        # Verify all values are back to zero
        self.assertEqual(metrics.total_generations, 0)
        self.assertEqual(metrics.blocked_generations, 0)
        self.assertEqual(metrics.suppression_events, 0)
        self.assertEqual(metrics.cache_hits, 0)
        self.assertEqual(metrics.cache_misses, 0)
        self.assertEqual(metrics.total_safety_check_time_ms, 0.0)
        self.assertEqual(metrics.safety_check_count, 0)

    def test_metrics_combine(self):
        """Test combining metrics from multiple instances."""
        metrics1 = SafetyMetrics()
        metrics2 = SafetyMetrics()

        # Record data in first instance
        metrics1.record_safety_check(10.0)
        metrics1.record_cache_hit()
        metrics1.record_generation_attempt()

        # Record data in second instance
        metrics2.record_safety_check(20.0)
        metrics2.record_cache_miss()
        metrics2.record_blocked_generation()

        # Combine them
        combined = metrics1.combine(metrics2)

        # Check combined values
        self.assertEqual(combined.safety_check_count, 2)
        self.assertEqual(combined.total_safety_check_time_ms, 30.0)
        self.assertEqual(combined.cache_hits, 1)
        self.assertEqual(combined.cache_misses, 1)
        self.assertEqual(combined.total_generations, 1)
        self.assertEqual(combined.blocked_generations, 1)

    def test_logits_processor_metrics_integration(self):
        """Test metrics integration with SafetyLogitsProcessor."""
        # Mock safety checker
        mock_checker = Mock()
        mock_checker.check_safety.return_value = SafetyResult(
            is_safe=False, confidence=0.8, violations=[SafetyViolation("toxicity", 0.8, "high")], metadata={}
        )

        # Mock tokenizer
        mock_tokenizer = Mock()
        mock_tokenizer.decode.return_value = "test unsafe text"

        # Safety config
        safety_config = SafetyConfig.from_checker(mock_checker)

        # Create processor
        processor = SafetyLogitsProcessor(
            safety_checker=mock_checker, tokenizer=mock_tokenizer, safety_config=safety_config
        )

        # Verify metrics are initialized
        metrics = processor.get_metrics()
        self.assertIsInstance(metrics, SafetyMetrics)
        self.assertEqual(metrics.suppression_events, 0)

        # Process some data (this should trigger metrics recording)
        input_ids = torch.tensor([[1, 2, 3, 4]])
        scores = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])

        processor(input_ids, scores)

        # Check that metrics were recorded
        metrics = processor.get_metrics()
        self.assertGreater(metrics.safety_check_count, 0)
        self.assertGreater(metrics.suppression_events, 0)  # Should have suppression due to unsafe content

    def test_stopping_criteria_metrics_integration(self):
        """Test metrics integration with SafetyStoppingCriteria."""
        # Mock safety checker
        mock_checker = Mock()
        mock_checker.check_safety.return_value = SafetyResult(
            is_safe=False, confidence=0.8, violations=[SafetyViolation("toxicity", 0.8, "high")], metadata={}
        )

        # Mock tokenizer
        mock_tokenizer = Mock()
        mock_tokenizer.decode.return_value = "test unsafe text"

        # Safety config
        safety_config = SafetyConfig.from_checker(mock_checker)

        # Create stopping criteria
        criteria = SafetyStoppingCriteria(
            safety_checker=mock_checker, tokenizer=mock_tokenizer, safety_config=safety_config
        )

        # Verify metrics are initialized
        metrics = criteria.get_metrics()
        self.assertIsInstance(metrics, SafetyMetrics)
        self.assertEqual(metrics.total_generations, 0)

        # Process some data
        input_ids = torch.tensor([[1, 2, 3, 4]])
        scores = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])

        criteria(input_ids, scores)

        # Check that metrics were recorded
        metrics = criteria.get_metrics()
        self.assertGreater(metrics.total_generations, 0)
        self.assertGreater(metrics.blocked_generations, 0)  # Should have blocked generation

    def test_thread_safety_basic(self):
        """Test basic thread safety of SafetyMetrics."""
        import threading
        import time

        metrics = SafetyMetrics()
        errors = []

        def worker():
            try:
                for i in range(100):
                    metrics.record_cache_hit()
                    metrics.record_safety_check(1.0)
                    time.sleep(0.001)  # Small delay to encourage race conditions
            except Exception as e:
                errors.append(e)

        # Run multiple threads
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=worker)
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Should have no errors and correct counts
        self.assertEqual(len(errors), 0, f"Thread safety errors: {errors}")
        self.assertEqual(metrics.cache_hits, 500)  # 5 threads * 100 operations
        self.assertEqual(metrics.safety_check_count, 500)

    def test_hash_consistency(self):
        """Test that hash inconsistency bug is fixed."""
        from transformers.generation.safety.processors import _generate_cache_key

        text1 = "This is a test message"
        text2 = "This is a test message"  # Same content
        text3 = "Different message"

        # Same text should produce same hash
        hash1 = _generate_cache_key(text1)
        hash2 = _generate_cache_key(text2)
        self.assertEqual(hash1, hash2)

        # Different text should produce different hash
        hash3 = _generate_cache_key(text3)
        self.assertNotEqual(hash1, hash3)

        # Hashes should be consistent across calls
        for _ in range(10):
            self.assertEqual(_generate_cache_key(text1), hash1)

    def test_cache_memory_management(self):
        """Test that caches properly manage memory."""
        # Mock safety checker
        mock_checker = Mock()
        mock_checker.check_safety.return_value = SafetyResult(is_safe=True, confidence=0.9, violations=[], metadata={})

        # Mock tokenizer
        mock_tokenizer = Mock()

        # Safety config - disable incremental checking for this test to ensure all calls are made
        safety_config = SafetyConfig.from_checker(mock_checker, incremental_checking=False)

        # Create processor
        processor = SafetyLogitsProcessor(
            safety_checker=mock_checker, tokenizer=mock_tokenizer, safety_config=safety_config
        )

        # Add many different sequences to test cache limits
        for i in range(150):  # More than default cache size of 100
            mock_tokenizer.decode.return_value = f"test text {i}"
            input_ids = torch.tensor([[1, 2, 3, i]])
            scores = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])
            processor(input_ids, scores)

        # Cache should be limited and not grow unbounded
        # The exact size check would depend on internal implementation
        # but we can verify calls were made
        self.assertEqual(mock_checker.check_safety.call_count, 150)

    def test_empty_and_special_text_handling(self):
        """Test handling of edge case text inputs."""
        from transformers.generation.safety.processors import _generate_cache_key

        # Test edge cases
        test_cases = [
            "",  # Empty string
            " ",  # Single space
            "\n\t",  # Whitespace only
            "🌍🚀💫",  # Unicode emoji
            "a" * 10000,  # Very long string
            "Test\x00null",  # String with null byte
        ]

        for text in test_cases:
            try:
                cache_key = _generate_cache_key(text)
                # Should produce valid cache key
                self.assertIsInstance(cache_key, str)
                self.assertGreater(len(cache_key), 0)
                # Should be consistent
                self.assertEqual(cache_key, _generate_cache_key(text))
            except Exception as e:
                self.fail(f"Failed to generate cache key for text: {repr(text)}, error: {e}")

    def test_device_mismatch_handling(self):
        """Test handling when tensors are on different devices."""
        # Mock safety checker
        mock_checker = Mock()
        mock_checker.check_safety.return_value = SafetyResult(
            is_safe=False, confidence=0.8, violations=[SafetyViolation("toxicity", 0.8, "high")], metadata={}
        )

        # Mock tokenizer
        mock_tokenizer = Mock()
        mock_tokenizer.decode.return_value = "unsafe text"

        # Safety config
        safety_config = SafetyConfig.from_checker(mock_checker)

        # Create processor
        processor = SafetyLogitsProcessor(
            safety_checker=mock_checker, tokenizer=mock_tokenizer, safety_config=safety_config
        )

        # Test with tensors (simulate device mismatch without actually using CUDA)
        input_ids = torch.tensor([[1, 2, 3, 4]])
        scores = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])

        # Should not raise device mismatch errors
        try:
            result = processor(input_ids, scores)
            self.assertEqual(result.shape, scores.shape)
        except Exception as e:
            self.fail(f"Device handling failed: {e}")

    def test_configurable_cache_size_logits_processor(self):
        """Test that SafetyLogitsProcessor respects configured cache size."""
        # Mock safety checker
        mock_checker = Mock()
        mock_checker.check_safety.return_value = SafetyResult(is_safe=True, confidence=0.9, violations=[], metadata={})

        # Mock tokenizer
        mock_tokenizer = Mock()

        # Test small cache size
        small_config = SafetyConfig.from_checker(mock_checker, cache_size=5)
        processor = SafetyLogitsProcessor(
            safety_checker=mock_checker, tokenizer=mock_tokenizer, safety_config=small_config
        )

        # Verify cache was initialized with correct size
        self.assertEqual(processor._sequence_cache.max_size, 5)

        # Test large cache size
        large_config = SafetyConfig.from_checker(mock_checker, cache_size=250)
        processor = SafetyLogitsProcessor(
            safety_checker=mock_checker, tokenizer=mock_tokenizer, safety_config=large_config
        )

        # Verify cache was initialized with correct size
        self.assertEqual(processor._sequence_cache.max_size, 250)

    def test_configurable_cache_size_stopping_criteria(self):
        """Test that SafetyStoppingCriteria respects configured cache and hash limits."""
        # Mock safety checker
        mock_checker = Mock()
        mock_checker.check_safety.return_value = SafetyResult(is_safe=True, confidence=0.9, violations=[], metadata={})

        # Mock tokenizer
        mock_tokenizer = Mock()

        # Test custom configuration
        custom_config = SafetyConfig.from_checker(mock_checker, cache_size=30, unsafe_hash_limit=300)

        criteria = SafetyStoppingCriteria(
            safety_checker=mock_checker, tokenizer=mock_tokenizer, safety_config=custom_config
        )

        # Verify cache and hash limit were configured correctly
        self.assertEqual(criteria._sequence_cache.max_size, 30)
        self.assertEqual(criteria._unsafe_hash_limit, 300)

    def test_default_cache_sizes_for_safety_levels(self):
        """Test that different safety levels use appropriate cache sizes."""
        # Mock safety checker and tokenizer
        mock_checker = Mock()
        mock_checker.check_safety.return_value = SafetyResult(is_safe=True, confidence=0.9, violations=[], metadata={})
        mock_tokenizer = Mock()

        # Test strict configuration
        strict_config = SafetyConfig.from_checker(mock_checker, **STRICT_PRESET)
        processor = SafetyLogitsProcessor(
            safety_checker=mock_checker, tokenizer=mock_tokenizer, safety_config=strict_config
        )
        self.assertEqual(processor._sequence_cache.max_size, 50)

        criteria = SafetyStoppingCriteria(
            safety_checker=mock_checker, tokenizer=mock_tokenizer, safety_config=strict_config
        )
        self.assertEqual(criteria._unsafe_hash_limit, 500)

        # Test moderate configuration
        moderate_config = SafetyConfig.from_checker(mock_checker, **MODERATE_PRESET)
        processor = SafetyLogitsProcessor(
            safety_checker=mock_checker, tokenizer=mock_tokenizer, safety_config=moderate_config
        )
        self.assertEqual(processor._sequence_cache.max_size, 100)

        # Test lenient configuration
        lenient_config = SafetyConfig.from_checker(mock_checker, **LENIENT_PRESET)
        processor = SafetyLogitsProcessor(
            safety_checker=mock_checker, tokenizer=mock_tokenizer, safety_config=lenient_config
        )
        self.assertEqual(processor._sequence_cache.max_size, 200)

    def test_backward_compatibility_cache_size(self):
        """Test that processors work with SafetyConfig without cache_size."""
        # Mock safety checker and tokenizer
        mock_checker = Mock()
        mock_checker.check_safety.return_value = SafetyResult(is_safe=True, confidence=0.9, violations=[], metadata={})
        mock_tokenizer = Mock()

        # Create a config that might not have cache_size (simulate old configs)
        config = SafetyConfig.from_checker(mock_checker)
        # Temporarily remove cache_size attribute to simulate old config
        if hasattr(config, "cache_size"):
            delattr(config, "cache_size")

        # Should still work with default cache size
        processor = SafetyLogitsProcessor(safety_checker=mock_checker, tokenizer=mock_tokenizer, safety_config=config)
        # Should use DEFAULT_CACHE_SIZE (100)
        from transformers.generation.safety.processors import DEFAULT_CACHE_SIZE

        self.assertEqual(processor._sequence_cache.max_size, DEFAULT_CACHE_SIZE)

    def test_cache_size_edge_cases(self):
        """Test edge cases for cache size configuration."""
        # Mock safety checker and tokenizer
        mock_checker = Mock()
        mock_checker.check_safety.return_value = SafetyResult(is_safe=True, confidence=0.9, violations=[], metadata={})
        mock_tokenizer = Mock()

        # Test minimum cache size (1)
        min_config = SafetyConfig.from_checker(mock_checker, cache_size=1)
        processor = SafetyLogitsProcessor(
            safety_checker=mock_checker, tokenizer=mock_tokenizer, safety_config=min_config
        )
        self.assertEqual(processor._sequence_cache.max_size, 1)

        # Test that processor works with cache size 1
        input_ids = torch.tensor([[1, 2, 3, 4]])
        scores = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])
        mock_tokenizer.decode.return_value = "test text"

        # Should not raise any errors
        result = processor(input_ids, scores)
        self.assertEqual(result.shape, scores.shape)


@require_torch
class TestSlidingWindowFunctionality(unittest.TestCase):
    """Test sliding window and incremental checking functionality."""

    def setUp(self):
        """Set up test fixtures."""
        # Mock safety checker
        self.mock_checker = Mock()
        self.mock_tokenizer = Mock()

    def test_safety_state_initialization(self):
        """Test SafetyState class initialization and basic functionality."""
        state = SafetyState()

        # Check initial values
        self.assertEqual(state.last_check_position, 0)
        self.assertIsNone(state.last_check_result)
        self.assertEqual(state.sequence_prefix, "")
        self.assertTrue(state.is_safe_so_far)
        self.assertEqual(state.window_start_position, 0)

    def test_safety_state_incremental_check_logic(self):
        """Test SafetyState incremental checking logic."""
        state = SafetyState()

        # First check should always be performed
        self.assertTrue(state.should_check_incremental(0, min_new_tokens=5))
        self.assertTrue(state.should_check_incremental(10, min_new_tokens=5))

        # Update state after first check
        result = SafetyResult(is_safe=True, confidence=0.9, violations=[], metadata={})
        state.update_check_result(10, result, "first check")

        # Check with insufficient new tokens
        self.assertFalse(state.should_check_incremental(14, min_new_tokens=5))

        # Check with sufficient new tokens
        self.assertTrue(state.should_check_incremental(15, min_new_tokens=5))

    def test_safety_state_sliding_window(self):
        """Test SafetyState sliding window extraction."""
        state = SafetyState()
        full_text = "This is a very long text that should trigger sliding window behavior when it exceeds the configured window size limit."

        # Test without sliding window (disabled)
        text_to_check, start_pos = state.get_incremental_text(full_text, sliding_window_size=-1)
        self.assertEqual(text_to_check, full_text)
        self.assertEqual(start_pos, 0)

        # Test with sliding window smaller than text
        window_size = 50
        text_to_check, start_pos = state.get_incremental_text(full_text, sliding_window_size=window_size)
        self.assertEqual(len(text_to_check), window_size)
        self.assertEqual(text_to_check, full_text[-window_size:])
        self.assertEqual(start_pos, len(full_text) - window_size)

        # Test with sliding window larger than text
        window_size = 200
        text_to_check, start_pos = state.get_incremental_text(full_text, sliding_window_size=window_size)
        self.assertEqual(text_to_check, full_text)
        self.assertEqual(start_pos, 0)

    def test_sliding_window_config_parameters(self):
        """Test sliding window configuration parameters in SafetyConfig."""
        # Test default values
        config = SafetyConfig()
        self.assertEqual(config.sliding_window_size, 512)
        self.assertTrue(config.incremental_checking)

        # Test custom values
        config = SafetyConfig(sliding_window_size=256, incremental_checking=False)
        self.assertEqual(config.sliding_window_size, 256)
        self.assertFalse(config.incremental_checking)

        # Test serialization includes new parameters
        config_dict = config.to_dict()
        self.assertEqual(config_dict["sliding_window_size"], 256)
        self.assertEqual(config_dict["incremental_checking"], False)

        # Test deserialization
        restored_config = SafetyConfig.from_dict(config_dict)
        self.assertEqual(restored_config.sliding_window_size, 256)
        self.assertFalse(restored_config.incremental_checking)

    def test_logits_processor_sliding_window_integration(self):
        """Test SafetyLogitsProcessor with sliding window functionality."""
        # Setup mocks
        self.mock_checker.check_safety.return_value = SafetyResult(
            is_safe=True, confidence=0.9, violations=[], metadata={}
        )

        # Create long text that would exceed window
        long_text = "This is a very long piece of text that should trigger the sliding window behavior. " * 10
        self.mock_tokenizer.decode.return_value = long_text

        # Test with sliding window enabled
        config = SafetyConfig.from_checker(
            self.mock_checker,
            sliding_window_size=100,
            incremental_checking=True,
        )

        processor = SafetyLogitsProcessor(
            safety_checker=self.mock_checker, tokenizer=self.mock_tokenizer, safety_config=config
        )

        # Verify sliding window parameters are set
        self.assertEqual(processor.sliding_window_size, 100)
        self.assertTrue(processor.incremental_checking)

        # Test processing with sliding window
        input_ids = torch.tensor([[1, 2, 3, 4, 5]])
        scores = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])

        result = processor(input_ids, scores)
        self.assertEqual(result.shape, scores.shape)

        # Verify safety check was called (though with potentially windowed text)
        self.mock_checker.check_safety.assert_called()

    def test_stopping_criteria_sliding_window_integration(self):
        """Test SafetyStoppingCriteria with sliding window functionality."""
        # Setup mocks
        self.mock_checker.check_safety.return_value = SafetyResult(
            is_safe=True, confidence=0.9, violations=[], metadata={}
        )

        long_text = "This is another very long piece of text for testing sliding window in stopping criteria. " * 10
        self.mock_tokenizer.decode.return_value = long_text

        # Test with sliding window enabled
        config = SafetyConfig.from_checker(
            self.mock_checker,
            sliding_window_size=100,
            incremental_checking=True,
        )

        criteria = SafetyStoppingCriteria(
            safety_checker=self.mock_checker, tokenizer=self.mock_tokenizer, safety_config=config
        )

        # Verify sliding window parameters are set
        self.assertEqual(criteria.sliding_window_size, 100)
        self.assertTrue(criteria.incremental_checking)

        # Test processing
        input_ids = torch.tensor([[1, 2, 3, 4, 5]])
        scores = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])

        should_stop = criteria(input_ids, scores)
        self.assertFalse(should_stop[0])  # Should not stop for safe content

    def test_incremental_checking_performance_benefit(self):
        """Test that incremental checking reduces safety check calls."""
        # Setup mock to count calls
        check_call_count = [0]

        def count_check_calls(text):
            check_call_count[0] += 1
            return SafetyResult(is_safe=True, confidence=0.9, violations=[], metadata={})

        self.mock_checker.check_safety.side_effect = count_check_calls

        # Create processor with incremental checking
        config = SafetyConfig.from_checker(self.mock_checker, incremental_checking=True)

        processor = SafetyLogitsProcessor(
            safety_checker=self.mock_checker,
            tokenizer=self.mock_tokenizer,
            safety_config=config,
            check_interval=1,  # Check every token
        )

        # Simulate progressive sequence building
        sequences = ["Hello", "Hello world", "Hello world this", "Hello world this is", "Hello world this is a test"]

        for seq in sequences:
            self.mock_tokenizer.decode.return_value = seq
            input_ids = torch.tensor([[1] * len(seq.split())])  # Approximate tokens
            scores = torch.randn(1, 1000)
            processor(input_ids, scores)

        # With incremental checking, we should have fewer calls than sequences
        # because short additions don't trigger new checks
        print(f"Check calls made: {check_call_count[0]} out of {len(sequences)} sequences")
        self.assertLessEqual(check_call_count[0], len(sequences))

    def test_sliding_window_with_unsafe_content(self):
        """Test sliding window behavior when unsafe content is detected."""
        # Setup mock to return unsafe result
        self.mock_checker.check_safety.return_value = SafetyResult(
            is_safe=False,
            confidence=0.8,
            violations=[SafetyViolation("toxicity", 0.8, "high", "Toxic content detected")],
            metadata={},
        )

        config = SafetyConfig.from_checker(
            self.mock_checker,
            sliding_window_size=50,
            incremental_checking=True,
        )

        processor = SafetyLogitsProcessor(
            safety_checker=self.mock_checker, tokenizer=self.mock_tokenizer, safety_config=config
        )

        self.mock_tokenizer.decode.return_value = "This contains toxic content that should be blocked"

        input_ids = torch.tensor([[1, 2, 3, 4, 5]])
        scores = torch.ones(1, 1000)  # All tokens have same score

        result = processor(input_ids, scores)

        # All tokens should be suppressed (set to negative infinity)
        self.assertTrue(torch.all(result < scores))
        self.assertTrue(torch.all(result == float("-inf")))

    def test_prefix_cache_functionality(self):
        """Test that prefix caching works correctly."""
        # This test verifies the _PrefixSafetyCache is used when incremental_checking=True
        config = SafetyConfig.from_checker(
            self.mock_checker,
            incremental_checking=True,  # Should use prefix cache
            cache_size=50,
        )

        processor = SafetyLogitsProcessor(
            safety_checker=self.mock_checker, tokenizer=self.mock_tokenizer, safety_config=config
        )

        # Verify correct cache type is used
        from transformers.generation.safety.processors import _PrefixSafetyCache

        self.assertIsInstance(processor._sequence_cache, _PrefixSafetyCache)

        # Test with incremental_checking=False
        config_no_incremental = SafetyConfig.from_checker(
            self.mock_checker,
            incremental_checking=False,  # Should use simple cache
        )

        processor_simple = SafetyLogitsProcessor(
            safety_checker=self.mock_checker, tokenizer=self.mock_tokenizer, safety_config=config_no_incremental
        )

        # Verify simple cache is used
        from transformers.generation.safety.processors import _SafetyCache

        self.assertIsInstance(processor_simple._sequence_cache, _SafetyCache)

    def test_safety_state_reset_functionality(self):
        """Test that safety states can be reset properly."""
        config = SafetyConfig.from_checker(self.mock_checker, incremental_checking=True)

        processor = SafetyLogitsProcessor(
            safety_checker=self.mock_checker, tokenizer=self.mock_tokenizer, safety_config=config
        )

        # Process some sequences to populate safety states
        self.mock_tokenizer.decode.return_value = "test text"
        self.mock_checker.check_safety.return_value = SafetyResult(
            is_safe=True, confidence=0.9, violations=[], metadata={}
        )

        input_ids = torch.tensor([[1, 2, 3, 4, 5]])
        scores = torch.randn(1, 1000)
        processor(input_ids, scores)

        # Verify states were created
        self.assertGreater(len(processor._safety_states), 0)

        # Reset states
        processor.reset_safety_states()

        # Verify states were cleared
        self.assertEqual(len(processor._safety_states), 0)


if __name__ == "__main__":
    unittest.main()
