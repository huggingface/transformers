# Copyright 2026 the HuggingFace Team. All rights reserved.
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

"""Tests for GlmMoeDsaConfig BC handling of num_experts vs n_routed_experts (#47355)."""

import unittest

from transformers import GlmMoeDsaConfig


class GlmMoeDsaConfigBCFallbackTest(unittest.TestCase):
    """When config is initialized with both n_routed_experts and legacy num_experts,
    the explicit n_routed_experts must NOT be overridden (#47355)."""

    def test_explicit_n_routed_experts_wins_over_legacy(self):
        """Both keys provided; explicit n_routed_experts must be preserved."""
        config = GlmMoeDsaConfig(
            n_routed_experts=168,
            num_experts=256,
            hidden_size=1024,
            num_hidden_layers=4,
            num_attention_heads=8,
        )
        self.assertEqual(
            config.n_routed_experts, 168,
            "n_routed_experts must not be overridden by legacy num_experts",
        )

    def test_legacy_num_experts_fallback_preserved(self):
        """Only legacy num_experts: BC fallback must still apply."""
        config = GlmMoeDsaConfig(
            num_experts=128,
            hidden_size=1024,
            num_hidden_layers=4,
            num_attention_heads=8,
        )
        self.assertEqual(
            config.n_routed_experts, 128,
            "BC fallback must apply when n_routed_experts is at class default",
        )

    def test_only_n_routed_experts_preserved(self):
        """Only n_routed_experts: value must be preserved."""
        config = GlmMoeDsaConfig(
            n_routed_experts=42,
            hidden_size=1024,
            num_hidden_layers=4,
            num_attention_heads=8,
        )
        self.assertEqual(config.n_routed_experts, 42)
