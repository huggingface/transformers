# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team.
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

from transformers import AutoTokenizer, GenerationConfig
from transformers.generation import (
    KeywordSafetyChecker,
    SafetyCheckResult,
    SafetyChecker,
    SafetyLogitsProcessor,
    SafetyStoppingCriteria,
    get_safety_checker,
)
from transformers.testing_utils import require_torch


class CustomTestSafetyChecker(SafetyChecker):
    """Simple test safety checker that blocks the word 'unsafe'."""

    def check_input(self, text, **kwargs):
        if isinstance(text, str):
            is_safe = "unsafe" not in text.lower()
            return SafetyCheckResult(
                is_safe=is_safe,
                violation_categories=[] if is_safe else ["test_violation"],
            )
        return [self.check_input(t, **kwargs) for t in text]

    def check_output(self, text, context=None, **kwargs):
        return self.check_input(text, **kwargs)


@require_torch
class GenerationConfigSafetyTest(unittest.TestCase):
    def test_generation_config_safety_defaults(self):
        """Test GenerationConfig safety parameter defaults."""
        config = GenerationConfig()
        self.assertIsNone(config.safety_checker)
        self.assertIsNone(config.safety_checker_kwargs)
        self.assertTrue(config.safety_check_input)
        self.assertTrue(config.safety_check_output)
        self.assertTrue(config.safety_stop_on_violation)
        self.assertFalse(config.safety_filter_violations)
        self.assertEqual(config.safety_check_frequency, 1)

    def test_generation_config_safety_custom(self):
        """Test GenerationConfig with custom safety values."""
        checker = CustomTestSafetyChecker()
        config = GenerationConfig(
            safety_checker=checker,
            safety_check_input=False,
            safety_check_frequency=5,
        )
        self.assertEqual(config.safety_checker, checker)
        self.assertFalse(config.safety_check_input)
        self.assertEqual(config.safety_check_frequency, 5)

    def test_get_safety_checker_from_string(self):
        """Test get_safety_checker with string name."""
        checker = get_safety_checker("keyword", {"blocked_keywords": ["test"]})
        self.assertIsInstance(checker, KeywordSafetyChecker)
        self.assertIn("test", checker.blocked_keywords)

    def test_get_safety_checker_from_instance(self):
        """Test get_safety_checker with instance."""
        original_checker = KeywordSafetyChecker(blocked_keywords=["test"])
        checker = get_safety_checker(original_checker)
        self.assertEqual(checker, original_checker)

    def test_get_safety_checker_none(self):
        """Test get_safety_checker with None."""
        checker = get_safety_checker(None)
        self.assertIsNone(checker)


@require_torch
class SafetyCheckResultTest(unittest.TestCase):
    def test_safety_check_result_safe(self):
        """Test SafetyCheckResult for safe content."""
        result = SafetyCheckResult(is_safe=True)
        self.assertTrue(result.is_safe)
        self.assertEqual(result.violation_categories, [])
        self.assertEqual(result.confidence_scores, {})
        self.assertIsNone(result.filtered_text)

    def test_safety_check_result_unsafe(self):
        """Test SafetyCheckResult for unsafe content."""
        result = SafetyCheckResult(
            is_safe=False,
            violation_categories=["violence", "hate_speech"],
            confidence_scores={"violence": 0.9, "hate_speech": 0.7},
            metadata={"source": "test"},
        )
        self.assertFalse(result.is_safe)
        self.assertEqual(result.violation_categories, ["violence", "hate_speech"])
        self.assertEqual(result.confidence_scores, {"violence": 0.9, "hate_speech": 0.7})
        self.assertEqual(result.metadata, {"source": "test"})


@require_torch
class KeywordSafetyCheckerTest(unittest.TestCase):
    def setUp(self):
        self.checker = KeywordSafetyChecker(
            blocked_keywords=["violence", "explicit"],
            blocked_patterns=[r"\d{3}-\d{2}-\d{4}"],  # SSN-like pattern
        )

    def test_check_safe_text(self):
        """Test keyword checker with safe text."""
        result = self.checker.check_input("This is a safe message")
        self.assertTrue(result.is_safe)
        self.assertEqual(result.violation_categories, [])

    def test_check_blocked_keyword(self):
        """Test keyword checker with blocked keyword."""
        result = self.checker.check_input("This contains violence")
        self.assertFalse(result.is_safe)
        self.assertIn("blocked_content", result.violation_categories)
        self.assertEqual(result.metadata["matched_keyword"], "violence")

    def test_check_case_insensitive(self):
        """Test keyword checker is case-insensitive."""
        result = self.checker.check_input("This contains VIOLENCE")
        self.assertFalse(result.is_safe)

    def test_check_blocked_pattern(self):
        """Test keyword checker with blocked pattern."""
        result = self.checker.check_input("SSN: 123-45-6789")
        self.assertFalse(result.is_safe)
        self.assertIn("blocked_content", result.violation_categories)

    def test_check_batch(self):
        """Test keyword checker with batch of texts."""
        texts = ["safe text", "contains violence", "also safe"]
        results = self.checker.check_input(texts)
        self.assertTrue(results[0].is_safe)
        self.assertFalse(results[1].is_safe)
        self.assertTrue(results[2].is_safe)

    def test_check_output_same_as_input(self):
        """Test that check_output behaves same as check_input for keyword checker."""
        text = "contains explicit content"
        input_result = self.checker.check_input(text)
        output_result = self.checker.check_output(text)
        self.assertEqual(input_result.is_safe, output_result.is_safe)


@require_torch
class CustomSafetyCheckerTest(unittest.TestCase):
    def setUp(self):
        self.checker = CustomTestSafetyChecker()

    def test_custom_checker_safe(self):
        """Test custom checker with safe text."""
        result = self.checker.check_input("This is safe")
        self.assertTrue(result.is_safe)

    def test_custom_checker_unsafe(self):
        """Test custom checker with unsafe text."""
        result = self.checker.check_input("This is unsafe")
        self.assertFalse(result.is_safe)
        self.assertEqual(result.violation_categories, ["test_violation"])

    def test_custom_checker_batch(self):
        """Test custom checker with batch."""
        results = self.checker.check_input(["safe", "unsafe"])
        self.assertTrue(results[0].is_safe)
        self.assertFalse(results[1].is_safe)


@require_torch
class SafetyStoppingCriteriaTest(unittest.TestCase):
    def setUp(self):
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.checker = KeywordSafetyChecker(blocked_keywords=["unsafe"])
        self.generation_config = GenerationConfig(safety_checker=self.checker)

    def test_stopping_criteria_safe(self):
        """Test stopping criteria with safe content."""
        criteria = SafetyStoppingCriteria(
            safety_checker=self.checker,
            tokenizer=self.tokenizer,
            generation_config=self.generation_config,
        )

        # Generate some safe input_ids
        input_ids = self.tokenizer("This is a safe message", return_tensors="pt").input_ids
        scores = torch.randn(input_ids.shape[0], len(self.tokenizer))

        should_stop = criteria(input_ids, scores)
        self.assertFalse(should_stop.any())

    def test_stopping_criteria_unsafe(self):
        """Test stopping criteria with unsafe content."""
        criteria = SafetyStoppingCriteria(
            safety_checker=self.checker,
            tokenizer=self.tokenizer,
            generation_config=self.generation_config,
        )

        # Generate some unsafe input_ids
        input_ids = self.tokenizer("This is unsafe content", return_tensors="pt").input_ids
        scores = torch.randn(input_ids.shape[0], len(self.tokenizer))

        should_stop = criteria(input_ids, scores)
        self.assertTrue(should_stop.any())

    def test_stopping_criteria_frequency(self):
        """Test stopping criteria respects check frequency."""
        config = GenerationConfig(safety_checker=self.checker, safety_check_frequency=3)
        criteria = SafetyStoppingCriteria(
            safety_checker=self.checker,
            tokenizer=self.tokenizer,
            generation_config=config,
        )

        input_ids = self.tokenizer("unsafe", return_tensors="pt").input_ids
        scores = torch.randn(input_ids.shape[0], len(self.tokenizer))

        # First call: counter = 1, not divisible by 3
        should_stop = criteria(input_ids, scores)
        self.assertFalse(should_stop.any())

        # Second call: counter = 2, not divisible by 3
        should_stop = criteria(input_ids, scores)
        self.assertFalse(should_stop.any())

        # Third call: counter = 3, divisible by 3, should check
        should_stop = criteria(input_ids, scores)
        self.assertTrue(should_stop.any())


@require_torch
class SafetyLogitsProcessorTest(unittest.TestCase):
    def setUp(self):
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.checker = KeywordSafetyChecker(blocked_keywords=["bad"])
        self.generation_config = GenerationConfig(safety_checker=self.checker)

    def test_logits_processor_safe(self):
        """Test logits processor with safe content."""
        processor = SafetyLogitsProcessor(
            safety_checker=self.checker,
            tokenizer=self.tokenizer,
            generation_config=self.generation_config,
        )

        input_ids = self.tokenizer("This is good", return_tensors="pt").input_ids
        scores = torch.randn(input_ids.shape[0], len(self.tokenizer))
        original_scores = scores.clone()

        processed_scores = processor(input_ids, scores)

        # With safe content and no "bad" continuations in top-k,
        # scores should remain similar (may differ due to safety checks)
        self.assertEqual(processed_scores.shape, original_scores.shape)

    def test_logits_processor_frequency(self):
        """Test logits processor respects check frequency."""
        config = GenerationConfig(safety_checker=self.checker, safety_check_frequency=2)
        processor = SafetyLogitsProcessor(
            safety_checker=self.checker,
            tokenizer=self.tokenizer,
            generation_config=config,
        )

        input_ids = self.tokenizer("test", return_tensors="pt").input_ids
        scores = torch.randn(input_ids.shape[0], len(self.tokenizer))

        # First call: counter = 1, not divisible by 2, should not process
        processed1 = processor(input_ids, scores.clone())
        self.assertTrue(torch.equal(processed1, scores))

        # Second call: counter = 2, divisible by 2, should process
        processed2 = processor(input_ids, scores.clone())
        # May or may not be equal depending on whether unsafe tokens are in top-k

    def test_logits_processor_penalty_value(self):
        """Test logits processor uses correct penalty value."""
        processor = SafetyLogitsProcessor(
            safety_checker=self.checker,
            tokenizer=self.tokenizer,
            generation_config=self.generation_config,
            penalty_value=-1000.0,
        )

        self.assertEqual(processor.penalty_value, -1000.0)


@require_torch
class AutomaticIntegrationTest(unittest.TestCase):
    """Test that safety is automatically integrated into model.generate()."""

    def test_automatic_stopping_criteria_integration(self):
        """Test that safety stopping criteria is automatically added during generation."""
        from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

        model = AutoModelForCausalLM.from_pretrained("gpt2")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token

        # Create config with safety
        checker = KeywordSafetyChecker(blocked_keywords=["badword"])
        generation_config = GenerationConfig(
            safety_checker=checker,
            safety_stop_on_violation=True,
            max_new_tokens=10,
        )

        # Generate should work without manually passing stopping criteria
        inputs = tokenizer("Hello", return_tensors="pt")
        outputs = model.generate(
            **inputs,
            generation_config=generation_config,
            tokenizer=tokenizer,  # Required for safety
        )

        # Should complete without errors
        self.assertIsNotNone(outputs)

    def test_automatic_logits_processor_integration(self):
        """Test that safety logits processor is automatically added during generation."""
        from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

        model = AutoModelForCausalLM.from_pretrained("gpt2")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token

        # Create config with safety filtering
        checker = KeywordSafetyChecker(blocked_keywords=["badword"])
        generation_config = GenerationConfig(
            safety_checker=checker,
            safety_filter_violations=True,
            max_new_tokens=10,
        )

        # Generate should work without manually passing logits processors
        inputs = tokenizer("Hello", return_tensors="pt")
        outputs = model.generate(
            **inputs,
            generation_config=generation_config,
            tokenizer=tokenizer,  # Required for safety
        )

        # Should complete without errors
        self.assertIsNotNone(outputs)

    def test_string_based_safety_checker(self):
        """Test using string name for safety checker."""
        from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

        model = AutoModelForCausalLM.from_pretrained("gpt2")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token

        # Use string-based safety checker
        generation_config = GenerationConfig(
            safety_checker="keyword",
            safety_checker_kwargs={"blocked_keywords": ["unsafe"]},
            safety_stop_on_violation=True,
            max_new_tokens=10,
        )

        inputs = tokenizer("Test", return_tensors="pt")
        outputs = model.generate(
            **inputs,
            generation_config=generation_config,
            tokenizer=tokenizer,
        )

        self.assertIsNotNone(outputs)


if __name__ == "__main__":
    unittest.main()
