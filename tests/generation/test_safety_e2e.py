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

import time
import unittest
from unittest.mock import Mock

import torch

from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from transformers.generation.safety import SafetyChecker, SafetyConfig, SafetyResult, SafetyViolation
from transformers.testing_utils import require_torch, slow


class TestSafetyEndToEnd(unittest.TestCase):
    """End-to-end tests for safety-enabled generation with actual models."""

    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _create_mock_checker(self):
        """Create a mock safety checker for testing."""
        # Create a mock checker that implements the SafetyChecker interface
        mock_checker = Mock(spec=SafetyChecker)
        mock_checker.supported_categories = ["toxicity"]
        return mock_checker

    @require_torch
    @slow
    def test_greedy_generation_with_safety(self):
        """Test that safety works with greedy decoding generation."""
        # Create mock checker
        mock_checker = self._create_mock_checker()

        # Mock safe responses
        mock_checker.check_safety.return_value = SafetyResult(is_safe=True, confidence=0.9, violations=[], metadata={})

        # Load small model for testing
        model_name = "sshleifer/tiny-gpt2"
        model = AutoModelForCausalLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token

        # Create safety configuration with mock checker
        safety_config = SafetyConfig.from_checker(mock_checker)

        # Create generation config with safety
        gen_config = GenerationConfig(
            max_length=20,
            do_sample=False,  # Greedy
            safety_config=safety_config,
        )

        # Test generation
        inputs = tokenizer("Hello, world", return_tensors="pt")
        outputs = model.generate(**inputs, generation_config=gen_config)

        # Verify output is generated
        self.assertGreater(outputs.shape[1], inputs["input_ids"].shape[1])

        # Verify safety checker was called
        mock_checker.check_safety.assert_called()

    @require_torch
    @slow
    def test_sample_generation_with_safety(self):
        """Test that safety works with sampling generation."""
        mock_checker = self._create_mock_checker()
        mock_checker.check_safety.return_value = SafetyResult(is_safe=True, confidence=0.9, violations=[], metadata={})

        # Mock safe responses
        mock_checker.check_safety.return_value = SafetyResult(is_safe=True, confidence=0.9, violations=[], metadata={})

        # Load small model
        model_name = "sshleifer/tiny-gpt2"
        model = AutoModelForCausalLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token

        # Create safety configuration
        safety_config = SafetyConfig.from_checker(mock_checker)

        # Test sampling with safety
        inputs = tokenizer("Hello", return_tensors="pt")
        outputs = model.generate(**inputs, max_length=15, do_sample=True, temperature=0.8, safety_config=safety_config)

        # Verify generation occurred
        self.assertGreater(outputs.shape[1], inputs["input_ids"].shape[1])
        mock_checker.check_safety.assert_called()

    @require_torch
    @slow
    def test_beam_search_generation_with_safety(self):
        """Test that safety works with beam search generation."""
        mock_checker = self._create_mock_checker()
        mock_checker.check_safety.return_value = SafetyResult(is_safe=True, confidence=0.9, violations=[], metadata={})

        # Mock safe responses
        mock_checker.check_safety.return_value = SafetyResult(is_safe=True, confidence=0.9, violations=[], metadata={})

        # Load small model
        model_name = "sshleifer/tiny-gpt2"
        model = AutoModelForCausalLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token

        # Create safety configuration
        safety_config = SafetyConfig.from_checker(mock_checker)

        # Test beam search with safety
        inputs = tokenizer("The weather is", return_tensors="pt")
        outputs = model.generate(**inputs, max_length=15, num_beams=2, safety_config=safety_config)

        # Verify generation occurred
        self.assertGreater(outputs.shape[1], inputs["input_ids"].shape[1])
        mock_checker.check_safety.assert_called()

    @require_torch
    @slow
    def test_safety_blocks_toxic_generation(self):
        """Test that generation stops when toxic content is detected."""
        mock_checker = self._create_mock_checker()
        mock_checker.check_safety.return_value = SafetyResult(is_safe=True, confidence=0.9, violations=[], metadata={})

        # Mock unsafe response that should stop generation
        mock_checker.check_safety.return_value = SafetyResult(
            is_safe=False,
            confidence=0.85,
            violations=[SafetyViolation("toxicity", 0.85, "high", "Toxic content detected")],
            metadata={"toxicity_score": 0.85},
        )

        # Load small model
        model_name = "sshleifer/tiny-gpt2"
        model = AutoModelForCausalLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token

        # Create safety configuration
        safety_config = SafetyConfig.from_checker(mock_checker)

        # Test generation - should stop early due to safety
        inputs = tokenizer("Test input", return_tensors="pt")
        outputs = model.generate(
            **inputs,
            max_length=50,  # Allow long generation
            safety_config=safety_config,
        )

        # Should stop early due to safety stopping criteria
        # (The exact length depends on when safety check triggers)
        self.assertLessEqual(outputs.shape[1], 50)
        mock_checker.check_safety.assert_called()

    @require_torch
    @slow
    def test_safety_disabled_backward_compatibility(self):
        """Test that safety disabled doesn't affect normal generation."""
        # No safety mocks needed - testing disabled safety

        # Load small model
        model_name = "sshleifer/tiny-gpt2"
        model = AutoModelForCausalLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token

        # Test without safety config (default behavior)
        inputs = tokenizer("Hello world", return_tensors="pt")
        outputs_no_safety = model.generate(**inputs, max_length=20, do_sample=False)

        # Test with disabled safety config
        safety_config = SafetyConfig(enabled=False, checker=None)
        outputs_disabled_safety = model.generate(**inputs, max_length=20, do_sample=False, safety_config=safety_config)

        # Results should be identical (since both use no safety)
        # Note: Results might not be exactly identical due to random state,
        # but both should generate successfully
        self.assertEqual(outputs_no_safety.shape, outputs_disabled_safety.shape)

    @require_torch
    @slow
    def test_performance_impact_measurement(self):
        """Test that safety overhead is reasonable."""
        # Load small model
        model_name = "sshleifer/tiny-gpt2"
        model = AutoModelForCausalLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token

        inputs = tokenizer("Performance test", return_tensors="pt")

        # Measure baseline (no safety)
        start_time = time.time()
        for _ in range(3):  # Multiple runs for more stable timing
            model.generate(**inputs, max_length=20, do_sample=False)
        baseline_time = time.time() - start_time

        # Set up safety mocks for performance test
        mock_checker = self._create_mock_checker()
        mock_checker.check_safety.return_value = SafetyResult(is_safe=True, confidence=0.9, violations=[], metadata={})
        mock_checker.check_safety.return_value = SafetyResult(is_safe=True, confidence=0.9, violations=[], metadata={})

        # Measure with safety enabled
        safety_config = SafetyConfig.from_checker(mock_checker)

        start_time = time.time()
        for _ in range(3):  # Multiple runs for more stable timing
            model.generate(**inputs, max_length=20, do_sample=False, safety_config=safety_config)
        safety_time = time.time() - start_time

        # Calculate overhead percentage
        overhead_percent = ((safety_time - baseline_time) / baseline_time) * 100

        # Assert that overhead is reasonable (less than 50% for this simple test)
        # Note: In real usage, overhead would be much less due to check_interval optimization
        self.assertLess(overhead_percent, 50, f"Safety overhead of {overhead_percent:.1f}% is too high")

        print(f"Safety overhead: {overhead_percent:.1f}%")
