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

import unittest

from transformers.models.cwm import CwmConfig, CwmTextConfig
from transformers.testing_utils import require_torch

from ...test_configuration_common import ConfigTester


class CwmConfigTest(unittest.TestCase):
    def test_default_config(self):
        """Test default CWM configuration"""
        config = CwmConfig()

        # CWM defaults
        self.assertEqual(config.sliding_window, 8192)
        self.assertEqual(config.window_pattern, 4)
        self.assertIsNone(config.global_window)
        self.assertIsInstance(config.layer_types, list)

        # Llama3 defaults
        self.assertEqual(config.vocab_size, 128256)
        self.assertEqual(config.rope_theta, 1_000_000.0)
        self.assertIsNotNone(config.rope_scaling)
        self.assertEqual(config.rope_scaling["rope_type"], "llama3")

    def test_custom_sliding_window_config(self):
        config = CwmConfig(sliding_window=4096, window_pattern=3, global_window=1024)

        self.assertEqual(config.sliding_window, 4096)
        self.assertEqual(config.window_pattern, 3)
        self.assertEqual(config.global_window, 1024)

    def test_custom_layer_types_config(self):
        layer_types = ["full_attention", "sliding_attention", "sliding_attention", "full_attention"]
        config = CwmConfig(num_hidden_layers=4, layer_types=layer_types)

        self.assertEqual(config.layer_types, layer_types)
        self.assertEqual(len(config.layer_types), config.num_hidden_layers)

    def test_invalid_layer_types_length(self):
        with self.assertRaises(ValueError):
            CwmConfig(
                num_hidden_layers=4,
                layer_types=["full_attention", "sliding_attention"],  # Only 2 types for 4 layers
            )

    def test_invalid_layer_type_value(self):
        with self.assertRaises(ValueError):
            CwmConfig(num_hidden_layers=2, layer_types=["full_attention", "invalid_attention"])

    def test_automatic_layer_types_generation(self):
        config = CwmConfig(num_hidden_layers=6, window_pattern=3)

        expected_types = [
            "full_attention",  # layer 0: 0 % 3 == 0
            "sliding_attention",  # layer 1: 1 % 3 != 0
            "sliding_attention",  # layer 2: 2 % 3 != 0
            "full_attention",  # layer 3: 3 % 3 == 0
            "sliding_attention",  # layer 4: 4 % 3 != 0
            "sliding_attention",  # layer 5: 5 % 3 != 0
        ]

        self.assertEqual(config.layer_types, expected_types)

    def test_rope_scaling_config(self):
        custom_rope_scaling = {
            "factor": 8.0,
            "high_freq_factor": 2.0,
            "low_freq_factor": 0.5,
            "original_max_position_embeddings": 4096,
            "rope_type": "llama3",
        }

        config = CwmConfig(rope_scaling=custom_rope_scaling)

        self.assertEqual(config.rope_scaling, custom_rope_scaling)

    def test_config_serialization(self):
        config = CwmConfig(
            sliding_window=4096,
            window_pattern=2,
            layer_types=["full_attention", "sliding_attention"] * 3,
            num_hidden_layers=6,
        )

        config_dict = config.to_dict()
        self.assertIn("sliding_window", config_dict)
        self.assertIn("window_pattern", config_dict)
        self.assertIn("layer_types", config_dict)

        new_config = CwmConfig.from_dict(config_dict)
        self.assertEqual(new_config.sliding_window, config.sliding_window)
        self.assertEqual(new_config.window_pattern, config.window_pattern)
        self.assertEqual(new_config.layer_types, config.layer_types)

    def test_config_inheritance_from_llama(self):
        config = CwmConfig()

        # Llama config attributes
        self.assertTrue(hasattr(config, "hidden_size"))
        self.assertTrue(hasattr(config, "num_attention_heads"))
        self.assertTrue(hasattr(config, "num_key_value_heads"))
        self.assertTrue(hasattr(config, "intermediate_size"))
        self.assertTrue(hasattr(config, "rope_theta"))
        self.assertTrue(hasattr(config, "attention_dropout"))

    def test_cwm_text_config_alias(self):
        config = CwmTextConfig(sliding_window=2048)
        self.assertEqual(config.sliding_window, 2048)
        self.assertEqual(config.model_type, "cwm")


@require_torch
class CwmConfigTester(ConfigTester):
    def __init__(self, parent, config_class=None, **kwargs):
        super().__init__(parent, config_class=config_class, **kwargs)

    def test_config(self):
        config_class = CwmConfig
        self.config_tester = ConfigTester(self, config_class=config_class)
        self.config_tester.run_common_tests()
