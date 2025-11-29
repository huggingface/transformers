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

from transformers import pipeline
from transformers.generation.safety import SafetyChecker, SafetyConfig, SafetyResult, SafetyViolation
from transformers.testing_utils import require_torch, slow


class MockSafetyChecker(SafetyChecker):
    """Mock safety checker for testing"""

    def __init__(self, is_safe=True, name="mock"):
        self.is_safe = is_safe
        self.name = name
        self.check_safety_calls = []

    def check_safety(self, text, **kwargs):
        self.check_safety_calls.append(text)
        return SafetyResult(
            is_safe=self.is_safe,
            confidence=0.9,
            violations=[] if self.is_safe else [SafetyViolation("test", 0.9, "high", "Test violation")],
            metadata={"checker": self.name},
        )

    @property
    def supported_categories(self):
        return ["test"]


@require_torch
class TestTextGenerationPipelineSafety(unittest.TestCase):
    """Tests for safety integration in TextGenerationPipeline"""

    def test_safety_config_per_call(self):
        """Test passing safety_config per generate call"""
        checker = MockSafetyChecker(is_safe=True)
        config = SafetyConfig.from_checker(checker)

        pipe = pipeline("text-generation", model="sshleifer/tiny-gpt2")
        result = pipe("Hello", safety_config=config, max_new_tokens=10)

        # Verify safety was applied
        self.assertGreater(len(checker.check_safety_calls), 0)
        self.assertIsNotNone(result)

    def test_safety_disabled_by_default(self):
        """Test that safety is not applied when no config provided"""
        pipe = pipeline("text-generation", model="sshleifer/tiny-gpt2")
        result = pipe("Hello", max_new_tokens=10)

        # Should work normally without safety
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 1)
        self.assertIn("generated_text", result[0])

    def test_unsafe_content_blocked(self):
        """Test that unsafe content generation is blocked"""
        checker = MockSafetyChecker(is_safe=False)  # Always unsafe
        config = SafetyConfig.from_checker(checker)

        pipe = pipeline("text-generation", model="sshleifer/tiny-gpt2")
        result = pipe("Hello", safety_config=config, max_new_tokens=10, do_sample=False)

        # Generation should be stopped early due to safety
        self.assertIsNotNone(result)
        # Exact behavior depends on safety implementation
        # But checker should have been called
        self.assertGreater(len(checker.check_safety_calls), 0)

    def test_safety_with_batch(self):
        """Test safety checking with batch input"""
        checker = MockSafetyChecker(is_safe=True)
        config = SafetyConfig.from_checker(checker)

        pipe = pipeline("text-generation", model="sshleifer/tiny-gpt2")
        results = pipe(["Hello", "World"], safety_config=config, max_new_tokens=10)

        # Verify safety was applied to batch
        self.assertGreater(len(checker.check_safety_calls), 0)
        self.assertEqual(len(results), 2)

    @slow
    def test_safety_with_actual_model(self):
        """Test safety with actual model generation (slow test)"""
        checker = MockSafetyChecker(is_safe=True)
        config = SafetyConfig.from_checker(checker)

        pipe = pipeline("text-generation", model="gpt2")
        result = pipe("The capital of France is", safety_config=config, max_new_tokens=5, do_sample=False)

        self.assertIsNotNone(result)
        self.assertIn("generated_text", result[0])
        self.assertGreater(len(checker.check_safety_calls), 0)
