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

from transformers.generation.safety import (
    LENIENT_PRESET,
    MODERATE_PRESET,
    STRICT_PRESET,
    SafetyChecker,
    SafetyConfig,
)


class TestSafetyConfig(unittest.TestCase):
    """Test suite for SafetyConfig."""

    def setUp(self):
        """Set up mock checker for tests."""
        self.mock_checker = Mock(spec=SafetyChecker)
        self.mock_checker.supported_categories = ["toxicity"]

    def test_default_config(self):
        """Test SafetyConfig with default values."""
        config = SafetyConfig()

        # Check default values
        self.assertFalse(config.enabled)
        self.assertIsNone(config.checker)
        self.assertIsNone(config.device)
        self.assertFalse(config.return_violations)
        self.assertFalse(config.return_metadata)
        self.assertEqual(config.cache_size, 100)
        self.assertEqual(config.unsafe_hash_limit, 1000)
        self.assertEqual(config.sliding_window_size, 512)
        self.assertTrue(config.incremental_checking)

    def test_from_checker_basic(self):
        """Test creating config from checker using from_checker (recommended pattern)."""
        config = SafetyConfig.from_checker(self.mock_checker)

        # Verify config was created correctly
        self.assertTrue(config.enabled)
        self.assertIs(config.checker, self.mock_checker)
        self.assertEqual(config.cache_size, 100)  # Default
        self.assertFalse(config.return_violations)  # Default
        self.assertFalse(config.return_metadata)  # Default

    def test_from_checker_with_preset(self):
        """Test creating config from checker with preset parameters."""
        config = SafetyConfig.from_checker(self.mock_checker, **STRICT_PRESET)

        self.assertTrue(config.enabled)
        self.assertIs(config.checker, self.mock_checker)
        self.assertEqual(config.cache_size, 50)
        self.assertEqual(config.unsafe_hash_limit, 500)
        self.assertTrue(config.return_violations)
        self.assertTrue(config.return_metadata)

    def test_from_checker_with_custom_params(self):
        """Test creating config from checker with custom parameters."""
        config = SafetyConfig.from_checker(self.mock_checker, cache_size=200, return_violations=True, device="cuda")

        self.assertTrue(config.enabled)
        self.assertIs(config.checker, self.mock_checker)
        self.assertEqual(config.cache_size, 200)
        self.assertTrue(config.return_violations)
        self.assertEqual(config.device, "cuda")

    def test_construct_checker_returns_instance(self):
        """Test that construct_checker returns the provided checker instance."""
        config = SafetyConfig.from_checker(self.mock_checker)
        retrieved = config.construct_checker()
        self.assertIs(retrieved, self.mock_checker)

    def test_construct_checker_error_when_missing(self):
        """Test that construct_checker raises helpful error when checker is missing."""
        config = SafetyConfig(enabled=True)

        with self.assertRaises(ValueError) as context:
            config.construct_checker()

        error_message = str(context.exception)
        self.assertIn("SafetyConfig requires a checker instance", error_message)
        self.assertIn("examples/safe_generation", error_message)
        self.assertIn("BasicToxicityChecker", error_message)
        self.assertIn("from_checker", error_message)

    def test_serialization_round_trip(self):
        """Test serialization and deserialization (note: checker not serialized)."""
        original_config = SafetyConfig.from_checker(
            self.mock_checker, cache_size=150, return_violations=True, device="cpu"
        )

        # Serialize to dict
        config_dict = original_config.to_dict()

        # Check dict contents (checker is not serialized)
        self.assertEqual(config_dict["enabled"], True)
        self.assertEqual(config_dict["cache_size"], 150)
        self.assertEqual(config_dict["device"], "cpu")
        self.assertTrue(config_dict["return_violations"])
        self.assertNotIn("checker", config_dict)

        # Deserialize from dict
        restored_config = SafetyConfig.from_dict(config_dict)

        # Check attributes match (except checker which isn't serialized)
        self.assertEqual(restored_config.enabled, original_config.enabled)
        self.assertEqual(restored_config.cache_size, original_config.cache_size)
        self.assertEqual(restored_config.device, original_config.device)
        self.assertIsNone(restored_config.checker)  # Checker must be re-provided

        # Re-attach checker to restored config
        restored_config.checker = self.mock_checker
        retrieved = restored_config.construct_checker()
        self.assertIs(retrieved, self.mock_checker)

    def test_validation_success(self):
        """Test validation with valid configuration."""
        # Valid default config
        config = SafetyConfig()
        config.validate()  # Should not raise

        # Valid config with checker
        config = SafetyConfig.from_checker(self.mock_checker, return_violations=True)
        config.validate()  # Should not raise

    def test_validation_enabled_type(self):
        """Test validation of enabled field."""
        config = SafetyConfig(enabled="true")  # Wrong type
        with self.assertRaises(ValueError) as context:
            config.validate()
        self.assertIn("enabled must be a boolean", str(context.exception))

    def test_validation_output_config_types(self):
        """Test validation of output configuration types."""
        # Wrong return_violations type
        config = SafetyConfig(return_violations="true")
        with self.assertRaises(ValueError) as context:
            config.validate()
        self.assertIn("return_violations must be a boolean", str(context.exception))

        # Wrong return_metadata type
        config = SafetyConfig(return_metadata=1)
        with self.assertRaises(ValueError) as context:
            config.validate()
        self.assertIn("return_metadata must be a boolean", str(context.exception))

    def test_cache_size_configuration(self):
        """Test cache size configuration and validation."""
        # Test default cache size
        config = SafetyConfig()
        self.assertEqual(config.cache_size, 100)

        # Test custom cache size
        config = SafetyConfig(cache_size=50)
        self.assertEqual(config.cache_size, 50)

        # Test cache size validation - must be positive integer (caught in __post_init__)
        with self.assertRaises(ValueError):
            SafetyConfig(cache_size=0)

        with self.assertRaises(ValueError):
            SafetyConfig(cache_size=-1)

        with self.assertRaises(TypeError):
            SafetyConfig(cache_size=3.14)

        with self.assertRaises(TypeError):
            SafetyConfig(cache_size="100")

    def test_unsafe_hash_limit_configuration(self):
        """Test unsafe hash limit configuration and validation."""
        # Test default unsafe hash limit
        config = SafetyConfig()
        self.assertEqual(config.unsafe_hash_limit, 1000)

        # Test custom unsafe hash limit
        config = SafetyConfig(unsafe_hash_limit=500)
        self.assertEqual(config.unsafe_hash_limit, 500)

        # Test validation - must be positive integer (caught in __post_init__)
        with self.assertRaises(ValueError):
            SafetyConfig(unsafe_hash_limit=0)

        with self.assertRaises(ValueError):
            SafetyConfig(unsafe_hash_limit=-1)

        with self.assertRaises(TypeError):
            SafetyConfig(unsafe_hash_limit=2.5)

        with self.assertRaises(TypeError):
            SafetyConfig(unsafe_hash_limit="1000")

    def test_large_cache_size_warning(self):
        """Test warning for potentially inefficient cache sizes."""
        import warnings

        # Test cache size warning
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            SafetyConfig(cache_size=20000).validate()
            self.assertEqual(len(w), 1)
            self.assertTrue("cache_size > 10000" in str(w[0].message))

        # Test unsafe hash limit warning
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            SafetyConfig(unsafe_hash_limit=200000).validate()
            self.assertEqual(len(w), 1)
            self.assertTrue("unsafe_hash_limit > 100000" in str(w[0].message))

    def test_preset_constants(self):
        """Test that preset constants have expected values."""
        # STRICT_PRESET
        self.assertEqual(STRICT_PRESET["cache_size"], 50)
        self.assertEqual(STRICT_PRESET["unsafe_hash_limit"], 500)
        self.assertTrue(STRICT_PRESET["return_violations"])
        self.assertTrue(STRICT_PRESET["return_metadata"])

        # MODERATE_PRESET
        self.assertEqual(MODERATE_PRESET["cache_size"], 100)
        self.assertEqual(MODERATE_PRESET["unsafe_hash_limit"], 1000)
        self.assertFalse(MODERATE_PRESET["return_violations"])
        self.assertFalse(MODERATE_PRESET["return_metadata"])

        # LENIENT_PRESET
        self.assertEqual(LENIENT_PRESET["cache_size"], 200)
        self.assertEqual(LENIENT_PRESET["unsafe_hash_limit"], 2000)
        self.assertFalse(LENIENT_PRESET["return_violations"])
        self.assertFalse(LENIENT_PRESET["return_metadata"])

    def test_presets_with_from_checker(self):
        """Test using presets with from_checker."""
        # Test strict preset
        strict_config = SafetyConfig.from_checker(self.mock_checker, **STRICT_PRESET)
        self.assertEqual(strict_config.cache_size, 50)
        self.assertEqual(strict_config.unsafe_hash_limit, 500)
        self.assertTrue(strict_config.return_violations)
        self.assertTrue(strict_config.return_metadata)

        # Test moderate preset
        moderate_config = SafetyConfig.from_checker(self.mock_checker, **MODERATE_PRESET)
        self.assertEqual(moderate_config.cache_size, 100)
        self.assertEqual(moderate_config.unsafe_hash_limit, 1000)
        self.assertFalse(moderate_config.return_violations)

        # Test lenient preset
        lenient_config = SafetyConfig.from_checker(self.mock_checker, **LENIENT_PRESET)
        self.assertEqual(lenient_config.cache_size, 200)
        self.assertEqual(lenient_config.unsafe_hash_limit, 2000)
        self.assertFalse(lenient_config.return_violations)

    def test_serialization_includes_cache_config(self):
        """Test that serialization includes cache configuration."""
        config = SafetyConfig(cache_size=75, unsafe_hash_limit=750)
        config_dict = config.to_dict()

        self.assertEqual(config_dict["cache_size"], 75)
        self.assertEqual(config_dict["unsafe_hash_limit"], 750)

        # Test round-trip
        restored_config = SafetyConfig.from_dict(config_dict)
        self.assertEqual(restored_config.cache_size, 75)
        self.assertEqual(restored_config.unsafe_hash_limit, 750)

    def test_sliding_window_configuration(self):
        """Test sliding window configuration parameters."""
        # Test default values
        config = SafetyConfig()
        self.assertEqual(config.sliding_window_size, 512)
        self.assertTrue(config.incremental_checking)

        # Test custom values
        config = SafetyConfig(sliding_window_size=256, incremental_checking=False)
        self.assertEqual(config.sliding_window_size, 256)
        self.assertFalse(config.incremental_checking)

    def test_sliding_window_validation(self):
        """Test validation of sliding window parameters."""
        # Test valid sliding window size
        config = SafetyConfig(sliding_window_size=100)
        config.validate()  # Should not raise

        # Test valid disabled sliding window
        config = SafetyConfig(sliding_window_size=-1)
        config.validate()  # Should not raise

        # Test invalid sliding window size (0)
        with self.assertRaises(ValueError) as context:
            SafetyConfig(sliding_window_size=0)
        self.assertIn("sliding_window_size must be a positive integer or -1 to disable", str(context.exception))

        # Test invalid sliding window size (negative but not -1)
        with self.assertRaises(ValueError) as context:
            SafetyConfig(sliding_window_size=-5)
        self.assertIn("sliding_window_size must be a positive integer or -1 to disable", str(context.exception))

        # Test invalid incremental_checking type
        with self.assertRaises(TypeError) as context:
            SafetyConfig(incremental_checking="true")
        self.assertIn("incremental_checking must be a boolean", str(context.exception))

    def test_sliding_window_serialization(self):
        """Test serialization of sliding window parameters."""
        config = SafetyConfig(
            sliding_window_size=256, incremental_checking=False, cache_size=50, unsafe_hash_limit=500
        )

        # Test to_dict includes sliding window parameters
        config_dict = config.to_dict()
        self.assertEqual(config_dict["sliding_window_size"], 256)
        self.assertEqual(config_dict["incremental_checking"], False)

        # Test round-trip serialization
        restored_config = SafetyConfig.from_dict(config_dict)
        self.assertEqual(restored_config.sliding_window_size, 256)
        self.assertFalse(restored_config.incremental_checking)
        self.assertEqual(restored_config.cache_size, 50)
        self.assertEqual(restored_config.unsafe_hash_limit, 500)

    def test_sliding_window_edge_cases(self):
        """Test edge cases for sliding window configuration."""
        # Test very large sliding window size
        config = SafetyConfig(sliding_window_size=10000)
        config.validate()  # Should be valid

        # Test minimum sliding window size
        config = SafetyConfig(sliding_window_size=1)
        config.validate()  # Should be valid

        # Test both sliding window and incremental checking disabled
        config = SafetyConfig(sliding_window_size=-1, incremental_checking=False)
        config.validate()  # Should be valid

    def test_comprehensive_workflow(self):
        """Test a complete workflow with SafetyConfig."""
        # Create configuration using from_checker (recommended approach)
        config = SafetyConfig.from_checker(
            self.mock_checker, cache_size=50, return_violations=True, return_metadata=True
        )

        # Validate configuration
        config.validate()

        # Verify config was created correctly
        self.assertTrue(config.enabled)
        self.assertIs(config.checker, self.mock_checker)
        self.assertEqual(config.cache_size, 50)
        self.assertTrue(config.return_violations)

        # Test construct_checker returns same instance
        retrieved_checker = config.construct_checker()
        self.assertIs(retrieved_checker, self.mock_checker)

        # Serialize and deserialize (note: checker not serialized)
        config_dict = config.to_dict()
        restored_config = SafetyConfig.from_dict(config_dict)

        # Verify consistency (except checker which isn't serialized)
        self.assertEqual(config.enabled, restored_config.enabled)
        self.assertEqual(config.cache_size, restored_config.cache_size)
        self.assertIsNone(restored_config.checker)  # Checker must be re-provided after deserialization

        # Re-attach checker to restored config
        restored_config.checker = self.mock_checker

        # Validate restored configuration
        restored_config.validate()
