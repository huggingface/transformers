# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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
"""Tests for the TeleChat3 config, which converts the original RoPE parameters for the Llama implementation."""

import math
import unittest

from transformers import TeleChat3Config


class TeleChat3ConfigTest(unittest.TestCase):
    def test_telechat3_yarn_is_converted_to_yarn(self):
        # `rope_scaling` as found in the released checkpoints
        config = TeleChat3Config(
            max_position_embeddings=32768,
            rope_theta=1000000.0,
            rope_scaling={
                "beta_fast": 32.0,
                "beta_slow": 1.0,
                "factor": 4.0,
                "original_max_position_embeddings": 8192,
                "rope_type": "telechat3-yarn",
                "type": "telechat3-yarn",
            },
        )
        self.assertEqual(config.rope_parameters["rope_type"], "yarn")
        self.assertNotIn("type", config.rope_parameters)
        self.assertEqual(config.rope_parameters["factor"], 4.0)
        self.assertAlmostEqual(config.rope_parameters["attention_factor"], 0.07 * math.log(4.0) + 1.0)

    def test_factor_follows_context_lengths(self):
        # The original code ignores `factor` when `original_max_position_embeddings` is set
        config = TeleChat3Config(
            max_position_embeddings=16384,
            rope_scaling={"factor": 4.0, "original_max_position_embeddings": 8192, "rope_type": "telechat3-yarn"},
        )
        self.assertEqual(config.rope_parameters["factor"], 2.0)
        self.assertAlmostEqual(config.rope_parameters["attention_factor"], 0.07 * math.log(2.0) + 1.0)

    def test_explicit_attention_factor_is_kept(self):
        config = TeleChat3Config(
            rope_scaling={"factor": 4.0, "attention_factor": 1.5, "rope_type": "telechat3-yarn"},
        )
        self.assertEqual(config.rope_parameters["attention_factor"], 1.5)
