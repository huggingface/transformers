# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for ParakeetCTC configuration."""

import unittest

from transformers.models.fastconformer import FastConformerConfig
from transformers.models.parakeet_ctc import ParakeetCTCConfig
from transformers.testing_utils import require_torch

from ...test_configuration_common import ConfigTester


class ParakeetCTCConfigTest(unittest.TestCase):
    def setUp(self):
        self.config_tester = ConfigTester(self, config_class=ParakeetCTCConfig)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_config_initialization_with_encoder_config(self):
        """Test that ParakeetCTCConfig can be initialized with different encoder config types."""
        # Test with None (should create default FastConformerConfig)
        config1 = ParakeetCTCConfig(vocab_size=100, encoder_config=None)
        self.assertIsInstance(config1.encoder_config, FastConformerConfig)
        self.assertEqual(config1.vocab_size, 100)

        # Test with dict
        encoder_dict = {
            "hidden_size": 128,
            "num_hidden_layers": 4,
            "num_attention_heads": 8,
        }
        config2 = ParakeetCTCConfig(vocab_size=200, encoder_config=encoder_dict)
        self.assertIsInstance(config2.encoder_config, FastConformerConfig)
        self.assertEqual(config2.encoder_config.hidden_size, 128)
        self.assertEqual(config2.vocab_size, 200)

        # Test with FastConformerConfig instance
        encoder_config = FastConformerConfig(hidden_size=256, num_hidden_layers=6)
        config3 = ParakeetCTCConfig(vocab_size=300, encoder_config=encoder_config)
        self.assertIsInstance(config3.encoder_config, FastConformerConfig)
        self.assertEqual(config3.encoder_config.hidden_size, 256)
        self.assertEqual(config3.vocab_size, 300)

    def test_config_ctc_parameters(self):
        """Test CTC-specific parameters."""
        config = ParakeetCTCConfig(
            vocab_size=500,
            blank_token_id=10,
            ctc_loss_reduction="sum",
            ctc_zero_infinity=False,
        )

        self.assertEqual(config.vocab_size, 500)
        self.assertEqual(config.blank_token_id, 10)
        self.assertEqual(config.ctc_loss_reduction, "sum")
        self.assertEqual(config.ctc_zero_infinity, False)

    def test_config_invalid_encoder_config(self):
        """Test that invalid encoder_config raises appropriate error."""
        with self.assertRaises(ValueError):
            ParakeetCTCConfig(encoder_config="invalid_string")

        with self.assertRaises(ValueError):
            ParakeetCTCConfig(encoder_config=123)

    @require_torch
    def test_config_to_dict(self):
        """Test configuration serialization."""
        config = ParakeetCTCConfig(
            vocab_size=128,
            blank_token_id=5,
            encoder_config={"hidden_size": 64, "num_hidden_layers": 2},
        )

        config_dict = config.to_dict()

        # Check that all keys are present
        self.assertIn("vocab_size", config_dict)
        self.assertIn("blank_token_id", config_dict)
        self.assertIn("encoder_config", config_dict)
        self.assertIn("model_type", config_dict)

        # Check values
        self.assertEqual(config_dict["vocab_size"], 128)
        self.assertEqual(config_dict["blank_token_id"], 5)
        self.assertEqual(config_dict["model_type"], "parakeet_ctc")

        # Check encoder config is properly nested
        self.assertIsInstance(config_dict["encoder_config"], dict)
        self.assertEqual(config_dict["encoder_config"]["hidden_size"], 64)


if __name__ == "__main__":
    unittest.main() 