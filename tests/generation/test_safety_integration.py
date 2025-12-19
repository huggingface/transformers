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

import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import torch


# Add examples directory to Python path to import BasicToxicityChecker
examples_path = Path(__file__).parent.parent.parent / "examples"
if str(examples_path) not in sys.path:
    sys.path.insert(0, str(examples_path))

from safe_generation import BasicToxicityChecker  # noqa: E402

from transformers.generation.configuration_utils import GenerationConfig  # noqa: E402
from transformers.generation.safety import (  # noqa: E402
    LENIENT_PRESET,
    MODERATE_PRESET,
    STRICT_PRESET,
    SafetyChecker,
    SafetyConfig,
    SafetyResult,
    SafetyViolation,
)
from transformers.generation.safety.processors import SafetyLogitsProcessor, SafetyStoppingCriteria  # noqa: E402
from transformers.testing_utils import require_torch  # noqa: E402


class TestSafetyIntegration(unittest.TestCase):
    """Integration tests for the complete safety checking workflow."""

    def setUp(self):
        """Set up mock safety checker for tests."""
        self.mock_checker = Mock(spec=SafetyChecker)
        self.mock_checker.check_safety.return_value = SafetyResult(
            is_safe=True, confidence=0.9, violations=[], metadata={}
        )
        self.mock_checker.supported_categories = ["toxicity"]

    def test_complete_safety_workflow(self):
        """Test end-to-end safety checking workflow from configuration to results."""
        # Step 1: Create and validate configuration
        config = SafetyConfig.from_checker(self.mock_checker, **STRICT_PRESET)
        config.validate()

        # Verify configuration is set up correctly with STRICT preset values
        self.assertTrue(config.enabled)
        self.assertEqual(config.cache_size, 50)  # STRICT_PRESET value
        self.assertEqual(config.unsafe_hash_limit, 500)  # STRICT_PRESET value
        self.assertTrue(config.return_violations)  # STRICT_PRESET value
        self.assertTrue(config.return_metadata)  # STRICT_PRESET value

        # Step 2: Test configuration serialization workflow
        config_dict = config.to_dict()
        restored_config = SafetyConfig.from_dict(config_dict)
        restored_config.validate()

        # Verify serialization preserved configuration (except checker which isn't serialized)
        self.assertEqual(config.cache_size, restored_config.cache_size)
        self.assertEqual(config.enabled, restored_config.enabled)
        self.assertEqual(config.return_violations, restored_config.return_violations)
        self.assertIsNone(restored_config.checker)  # Checker not serialized

        # Step 3: Test construct_checker returns the provided instance
        retrieved_checker = config.construct_checker()
        self.assertIs(retrieved_checker, self.mock_checker)

    @require_torch
    @patch("transformers.AutoTokenizer.from_pretrained")
    @patch("transformers.AutoModelForSequenceClassification.from_pretrained")
    def test_config_to_checker_integration(self, mock_model, mock_tokenizer):
        """Test creating checker instance and using it with SafetyConfig."""
        # Set up mocks
        mock_tokenizer_instance = Mock()
        mock_inputs = Mock()
        mock_inputs.to.return_value = mock_inputs
        mock_tokenizer_instance.return_value = mock_inputs
        mock_tokenizer.return_value = mock_tokenizer_instance

        mock_model_instance = Mock()
        mock_model_instance.eval.return_value = None
        mock_model_instance.to.return_value = None
        mock_model.return_value = mock_model_instance

        # User creates checker instance
        checker = BasicToxicityChecker(threshold=0.8)

        # Verify checker was created with correct configuration
        self.assertEqual(checker.threshold, 0.8)
        self.assertEqual(checker.model_name, "s-nlp/roberta_toxicity_classifier")  # Default
        self.assertEqual(checker.supported_categories, ["toxicity"])

        # Create SafetyConfig from checker instance (recommended pattern)
        config = SafetyConfig.from_checker(checker, return_violations=True)

        # Verify config was created correctly
        self.assertTrue(config.enabled)
        self.assertIs(config.checker, checker)
        self.assertTrue(config.return_violations)

        # Test that construct_checker returns the same instance
        retrieved_checker = config.construct_checker()
        self.assertIs(retrieved_checker, checker)

        # Test checker configuration serialization
        checker_config_dict = checker.get_config()
        expected_config = {
            "checker_type": "BasicToxicityChecker",
            "model_name": "s-nlp/roberta_toxicity_classifier",
            "threshold": 0.8,
            "device": checker.device,
        }
        self.assertEqual(checker_config_dict, expected_config)

    def test_utility_functions_integration(self):
        """Test integration of utility functions with configurations."""
        from transformers.generation.safety.utils import validate_safety_config

        # Test validation utility with various configurations
        configs_to_test = [
            SafetyConfig(),  # Default
            SafetyConfig.from_checker(self.mock_checker, **STRICT_PRESET),
            SafetyConfig.from_checker(self.mock_checker, **MODERATE_PRESET),
            SafetyConfig.from_checker(self.mock_checker, **LENIENT_PRESET),
        ]

        for config in configs_to_test:
            self.assertTrue(validate_safety_config(config))

        # Test with invalid configuration (invalid cache_size)
        with self.assertRaises(ValueError):
            # __post_init__ will raise ValueError for invalid cache_size
            SafetyConfig(cache_size=0)

    def test_safety_result_structure(self):
        """Test that SafetyResult and SafetyViolation work correctly together."""
        # Create a violation
        violation = SafetyViolation(
            category="toxicity",
            confidence=0.85,
            severity="high",
            description="Detected toxic content with 85% confidence",
        )

        # Create a safety result
        result = SafetyResult(
            is_safe=False,
            confidence=0.85,
            violations=[violation],
            metadata={"model_name": "unitary/toxic-bert", "toxicity_score": 0.85, "threshold": 0.7},
        )

        # Verify structure
        self.assertFalse(result.is_safe)
        self.assertEqual(result.confidence, 0.85)
        self.assertEqual(len(result.violations), 1)

        violation = result.violations[0]
        self.assertEqual(violation.category, "toxicity")
        self.assertEqual(violation.confidence, 0.85)
        self.assertEqual(violation.severity, "high")

        # Test metadata
        self.assertIn("model_name", result.metadata)
        self.assertEqual(result.metadata["threshold"], 0.7)

    def test_configuration_levels_produce_different_behaviors(self):
        """Test that different preset levels produce appropriate settings."""
        # Test all predefined presets
        strict = SafetyConfig.from_checker(self.mock_checker, **STRICT_PRESET)
        moderate = SafetyConfig.from_checker(self.mock_checker, **MODERATE_PRESET)
        lenient = SafetyConfig.from_checker(self.mock_checker, **LENIENT_PRESET)

        # Verify cache sizes are different and logical (strict < moderate < lenient)
        self.assertEqual(strict.cache_size, 50)
        self.assertEqual(moderate.cache_size, 100)
        self.assertEqual(lenient.cache_size, 200)
        self.assertLess(strict.cache_size, moderate.cache_size)
        self.assertLess(moderate.cache_size, lenient.cache_size)

        # Verify unsafe hash limits follow same pattern
        self.assertEqual(strict.unsafe_hash_limit, 500)
        self.assertEqual(moderate.unsafe_hash_limit, 1000)
        self.assertEqual(lenient.unsafe_hash_limit, 2000)
        self.assertLess(strict.unsafe_hash_limit, moderate.unsafe_hash_limit)
        self.assertLess(moderate.unsafe_hash_limit, lenient.unsafe_hash_limit)

        # Verify output configuration differences
        self.assertTrue(strict.return_violations)
        self.assertTrue(strict.return_metadata)

        self.assertFalse(moderate.return_violations)
        self.assertFalse(lenient.return_violations)

    def test_error_handling_throughout_workflow(self):
        """Test error handling across the complete workflow."""
        # Test configuration validation errors - invalid cache_size
        with self.assertRaises(ValueError):
            SafetyConfig(cache_size=-1)

        # Test configuration validation errors - invalid unsafe_hash_limit
        with self.assertRaises(ValueError):
            SafetyConfig(unsafe_hash_limit=0)

        # Test construct_checker without providing checker raises error
        config = SafetyConfig(enabled=True)
        with self.assertRaises(ValueError) as context:
            config.construct_checker()
        self.assertIn("SafetyConfig requires a checker instance", str(context.exception))

        # Test invalid return_violations type
        with self.assertRaises(ValueError) as context:
            config = SafetyConfig(return_violations="true")  # Wrong type
            config.validate()
        self.assertIn("return_violations must be a boolean", str(context.exception))

    def test_public_api_imports(self):
        """Test that all public API components can be imported correctly."""
        # Test core imports
        from transformers.generation.safety import SafetyChecker, SafetyConfig

        # Verify classes are properly available
        self.assertTrue(hasattr(SafetyChecker, "check_safety"))
        self.assertTrue(hasattr(SafetyChecker, "supported_categories"))

        # Test SafetyConfig factory
        config = SafetyConfig.from_checker(self.mock_checker, **MODERATE_PRESET)
        self.assertIsInstance(config, SafetyConfig)

        # Test torch-dependent import
        from transformers.utils import is_torch_available

        # Note: BasicToxicityChecker is a reference implementation in examples/safe_generation
        # Core transformers only provides the SafetyChecker ABC
        if is_torch_available():
            # Verify BasicToxicityChecker is available from examples
            from safe_generation import BasicToxicityChecker

            self.assertTrue(issubclass(BasicToxicityChecker, SafetyChecker))


class TestGenerationConfigIntegration(unittest.TestCase):
    """Tests for safety integration with GenerationConfig and generation pipeline."""

    def setUp(self):
        """Set up mock safety checker for tests."""
        self.mock_checker = Mock(spec=SafetyChecker)
        self.mock_checker.check_safety.return_value = SafetyResult(
            is_safe=True, confidence=0.9, violations=[], metadata={}
        )
        self.mock_checker.supported_categories = ["toxicity"]

    def test_generation_config_accepts_safety_config(self):
        """Test that GenerationConfig properly accepts and stores safety_config."""
        safety_config = SafetyConfig.from_checker(self.mock_checker)

        # Test direct parameter
        gen_config = GenerationConfig(max_length=100, safety_config=safety_config)

        self.assertIsNotNone(gen_config.safety_config)
        self.assertEqual(gen_config.safety_config.enabled, True)
        # Check preset fields instead of non-existent thresholds
        self.assertEqual(gen_config.safety_config.cache_size, 100)  # MODERATE_PRESET default

        # Test None safety_config
        gen_config_none = GenerationConfig(max_length=100)
        self.assertIsNone(gen_config_none.safety_config)

        # Test update method
        gen_config_update = GenerationConfig(max_length=100)
        gen_config_update.update(safety_config=safety_config)
        self.assertIsNotNone(gen_config_update.safety_config)

    @require_torch
    @patch("safe_generation.BasicToxicityChecker")
    def test_generation_mixin_creates_safety_processors(self, mock_checker_class):
        """Test that GenerationMixin creates safety processors when configured."""
        # Mock the checker
        mock_checker = Mock()
        mock_checker.check_safety.return_value = SafetyResult(is_safe=True, confidence=0.9, violations=[], metadata={})
        mock_checker_class.return_value = mock_checker

        # Create a simple model mock with GenerationMixin methods
        from transformers.generation.utils import GenerationMixin

        model = Mock(spec=GenerationMixin)
        model.config = Mock()
        model.config.vocab_size = 1000
        model.device = torch.device("cpu")

        # Add the methods and required attributes
        model._create_safety_processor = GenerationMixin._create_safety_processor.__get__(model)
        model.tokenizer = Mock()  # Add tokenizer mock

        # Mock tokenizer methods
        model.tokenizer.decode = Mock(return_value="test text")
        model.tokenizer.convert_tokens_to_ids = Mock(return_value=123)
        model.tokenizer.unk_token_id = 0

        # Test with safety enabled
        mock_checker_instance = Mock(spec=SafetyChecker)
        safety_config = SafetyConfig.from_checker(mock_checker_instance)

        # Test logits processor creation
        logits_processor = model._create_safety_processor(safety_config, "logits")
        self.assertIsInstance(logits_processor, SafetyLogitsProcessor)

        # Test stopping criteria creation
        stopping_criteria = model._create_safety_processor(safety_config, "stopping")
        self.assertIsInstance(stopping_criteria, SafetyStoppingCriteria)

        # Test with safety disabled
        disabled_config = SafetyConfig(enabled=False)
        self.assertIsNone(model._create_safety_processor(disabled_config, "logits"))
        self.assertIsNone(model._create_safety_processor(disabled_config, "stopping"))

        # Test with None config
        self.assertIsNone(model._create_safety_processor(None, "logits"))

    @require_torch
    @patch("safe_generation.BasicToxicityChecker")
    def test_logits_processor_integration(self, mock_checker_class):
        """Test integration of safety with logits processor pipeline."""
        # Mock checker
        mock_checker = Mock()
        mock_checker.check_safety.return_value = SafetyResult(
            is_safe=False,
            confidence=0.9,
            violations=[SafetyViolation("toxicity", 0.9, "high", "Toxic content detected")],
            metadata={},
        )
        mock_checker_class.return_value = mock_checker

        # Create processor
        safety_config = SafetyConfig.from_checker(self.mock_checker)

        # Mock tokenizer
        mock_tokenizer = Mock()
        mock_tokenizer.decode.return_value = "test text"
        mock_tokenizer.convert_tokens_to_ids.return_value = 123
        mock_tokenizer.unk_token_id = 0

        processor = SafetyLogitsProcessor(
            safety_checker=mock_checker, tokenizer=mock_tokenizer, safety_config=safety_config
        )

        # Create test data
        batch_size = 2
        vocab_size = 1000
        sequence_length = 5

        input_ids = torch.randint(0, vocab_size, (batch_size, sequence_length))
        scores = torch.randn(batch_size, vocab_size)

        # Process scores
        processed_scores = processor(input_ids, scores)

        # Verify scores were modified (top tokens should be suppressed)
        self.assertFalse(torch.equal(scores, processed_scores))

        # Verify checker was called
        mock_checker.check_safety.assert_called()

    @require_torch
    @patch("safe_generation.BasicToxicityChecker")
    def test_stopping_criteria_integration(self, mock_checker_class):
        """Test integration of safety with stopping criteria pipeline."""
        # Mock checker with unsafe result
        mock_checker = Mock()
        mock_checker.check_safety.return_value = SafetyResult(
            is_safe=False,
            confidence=0.9,
            violations=[SafetyViolation("toxicity", 0.9, "high", "Toxic content")],
            metadata={},
        )
        mock_checker_class.return_value = mock_checker

        # Create stopping criteria
        safety_config = SafetyConfig.from_checker(self.mock_checker)

        # Mock tokenizer
        mock_tokenizer = Mock()
        mock_tokenizer.decode.return_value = "test text"

        criteria = SafetyStoppingCriteria(
            safety_checker=mock_checker, tokenizer=mock_tokenizer, safety_config=safety_config
        )

        # Create test data
        batch_size = 2
        vocab_size = 1000
        sequence_length = 10

        input_ids = torch.randint(0, vocab_size, (batch_size, sequence_length))
        scores = torch.randn(batch_size, vocab_size)

        # Test stopping decision
        should_stop = criteria(input_ids, scores)

        # Should stop due to unsafe content
        self.assertTrue(should_stop.any())

        # Verify checker was called
        mock_checker.check_safety.assert_called()

    def test_backward_compatibility(self):
        """Test that existing generation code works without safety configuration."""
        # Test GenerationConfig without safety
        gen_config = GenerationConfig(max_length=100, temperature=0.8, top_p=0.9)

        self.assertIsNone(gen_config.safety_config)
        self.assertEqual(gen_config.max_length, 100)
        self.assertEqual(gen_config.temperature, 0.8)

        # Test that to_dict/from_dict works
        config_dict = gen_config.to_dict()
        restored = GenerationConfig.from_dict(config_dict)

        self.assertEqual(restored.max_length, 100)
        self.assertIsNone(restored.safety_config)

    def test_safety_config_serialization_in_generation_config(self):
        """Test that safety_config is properly serialized with GenerationConfig."""
        safety_config = SafetyConfig.from_checker(self.mock_checker, return_violations=True)

        gen_config = GenerationConfig(max_length=100, safety_config=safety_config)

        # Test to_dict
        config_dict = gen_config.to_dict()
        self.assertIn("safety_config", config_dict)

        # Test from_dict
        restored = GenerationConfig.from_dict(config_dict)
        self.assertIsNotNone(restored.safety_config)
        self.assertEqual(restored.safety_config.enabled, True)
        self.assertTrue(restored.safety_config.return_violations)

    def test_error_handling_in_generation_integration(self):
        """Test error handling in generation pipeline integration."""
        # Test invalid safety config type
        with self.assertRaises((TypeError, AttributeError)):
            GenerationConfig(safety_config="invalid")

        # Test invalid processor type
        from transformers.generation.utils import GenerationMixin

        model = Mock(spec=GenerationMixin)
        model._create_safety_processor = GenerationMixin._create_safety_processor.__get__(model)
        model.tokenizer = Mock()  # Add tokenizer mock

        # Create config with mock checker
        safety_config = SafetyConfig.from_checker(self.mock_checker)

        # Should raise ValueError for invalid processor type
        with self.assertRaises(ValueError) as context:
            model._create_safety_processor(safety_config, "invalid_type")
        self.assertIn("processor_type must be 'logits' or 'stopping'", str(context.exception))

    @require_torch
    def test_end_to_end_safety_integration(self):
        """Test complete end-to-end safety integration workflow."""
        # Create safety configuration
        safety_config = SafetyConfig.from_checker(self.mock_checker)

        # Create generation configuration with safety
        gen_config = GenerationConfig(max_length=50, temperature=0.8, safety_config=safety_config)

        # Verify safety config is properly stored
        self.assertIsNotNone(gen_config.safety_config)
        self.assertEqual(gen_config.safety_config.enabled, True)

        # Test serialization round-trip
        config_dict = gen_config.to_dict()
        restored_config = GenerationConfig.from_dict(config_dict)

        self.assertIsNotNone(restored_config.safety_config)
        self.assertEqual(restored_config.safety_config.enabled, True)
        self.assertEqual(restored_config.safety_config.cache_size, safety_config.cache_size)

        # Verify non-safety parameters are preserved
        self.assertEqual(restored_config.max_length, 50)
        self.assertEqual(restored_config.temperature, 0.8)
