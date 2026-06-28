# Copyright (C) 2026 THL A29 Limited, a Tencent company and the HuggingFace Inc. team. All rights reserved.
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
"""Tests for [`HunYuanVLConfig`] and its sub-configs."""

import unittest

from transformers.models.hunyuan_vl.configuration_hunyuan_vl import (
    HunYuanVLConfig,
    HunYuanVLTextConfig,
    HunYuanVLVisionConfig,
)

from ...test_configuration_common import ConfigTester


LEGACY_XDROPE_SCALING = {
    "type": "xdrope",
    "factor": 1.0,
    "alpha": 1000.0,
    "beta_fast": 32,
    "beta_slow": 1,
    "mscale": 1.0,
    "mscale_all_dim": 1.0,
    "xdrope_section": [16, 16, 16, 16],
}


class HunYuanVLTextConfigTest(unittest.TestCase):
    def test_legacy_text_aliases_are_normalized(self):
        """Legacy ``rope_scaling`` / ``pad_id`` / ``attention_head_dim`` aliases must load into canonical fields."""
        config = HunYuanVLTextConfig(
            pad_token_id=-1,
            pad_id=7,
            attention_head_dim=128,
            rope_theta=10000.0,
            rope_scaling=LEGACY_XDROPE_SCALING,
        )

        self.assertEqual(config.pad_token_id, 7)
        self.assertEqual(config.head_dim, 128)
        self.assertEqual(config.rope_parameters["type"], "dynamic")
        self.assertEqual(config.rope_parameters["rope_type"], "dynamic")
        self.assertEqual(config.rope_parameters["rope_theta"], 10000.0)
        self.assertEqual(config.rope_parameters["xdrope_section"], [16, 16, 16, 16])

    def test_attribute_map_redirects_legacy_aliases_when_both_present_and_equal(self):
        """When legacy and canonical names are both written with the same value, ``attribute_map`` keeps the canonical."""
        config = HunYuanVLTextConfig(
            head_dim=128,
            attention_head_dim=128,
            vocab_size=290943,
            org_vocab_size=290943,
            pad_token_id=7,
            pad_id=7,
        )

        self.assertEqual(config.head_dim, 128)
        self.assertEqual(config.vocab_size, 290943)
        self.assertEqual(config.pad_token_id, 7)
        # Legacy names stay readable via the ``__getattribute__`` transparency provided by ``attribute_map``.
        self.assertEqual(config.attention_head_dim, 128)
        self.assertEqual(config.org_vocab_size, 290943)
        self.assertEqual(config.pad_id, 7)

    def test_attribute_map_recovers_from_legacy_only_checkpoint(self):
        """A checkpoint that only carries the legacy name must still populate the canonical field."""
        config = HunYuanVLTextConfig(attention_head_dim=64, org_vocab_size=1000, pad_id=9)

        self.assertEqual(config.head_dim, 64)
        self.assertEqual(config.vocab_size, 1000)
        self.assertEqual(config.pad_token_id, 9)

    def test_legacy_alias_keys_are_not_serialized(self):
        """``to_dict`` must only keep canonical names, never the legacy aliases."""
        config = HunYuanVLTextConfig(attention_head_dim=64, org_vocab_size=1000, pad_id=9)
        serialized = config.to_dict()

        self.assertIn("head_dim", serialized)
        self.assertIn("vocab_size", serialized)
        self.assertIn("pad_token_id", serialized)
        self.assertNotIn("attention_head_dim", serialized)
        self.assertNotIn("org_vocab_size", serialized)
        self.assertNotIn("pad_id", serialized)

    def test_minimal_text_config_does_not_invent_rope_parameters(self):
        """When no rope payload is provided we must not fabricate one."""
        config = HunYuanVLTextConfig()
        # The dataclass parent runs ``standardize_rope_params`` and produces a canonical default RoPE blob.
        self.assertIn("rope_type", config.rope_parameters)
        self.assertEqual(config.rope_parameters["rope_type"], "default")


class HunYuanVLVisionConfigTest(unittest.TestCase):
    def test_default_vision_config_is_dense_only(self):
        config = HunYuanVLVisionConfig()
        # Sanity-check that the dense, image-only OSS variant defaults are exposed.
        self.assertEqual(config.temporal_patch_size, 1)
        self.assertEqual(config.spatial_merge_size, 2)
        self.assertEqual(config.num_attention_heads, config.num_key_value_heads)


class HunYuanVLConfigTest(unittest.TestCase):
    class HunYuanVLConfigTester:
        def get_config(self):
            return HunYuanVLConfig(
                text_config={
                    "vocab_size": 1000,
                    "hidden_size": 128,
                    "intermediate_size": 256,
                    "num_hidden_layers": 2,
                    "num_attention_heads": 8,
                    "head_dim": 16,
                },
                vision_config={
                    "hidden_size": 64,
                    "intermediate_size": 128,
                    "num_hidden_layers": 2,
                    "num_attention_heads": 4,
                },
            )

    def setUp(self):
        self.model_tester = self.HunYuanVLConfigTester()

    def test_config(self):
        self.config_tester = ConfigTester(self, config_class=HunYuanVLConfig, has_text_modality=False)
        self.config_tester.run_common_tests()

    def test_text_side_token_ids_live_only_on_text_config(self):
        """Text-side token ids must be reachable via ``get_text_config()`` and not duplicated at the top level."""
        text_config = HunYuanVLTextConfig(pad_token_id=11, bos_token_id=22, eos_token_id=33)
        config = HunYuanVLConfig(text_config=text_config)

        # Generic utilities reach these via get_text_config(); the top-level config must not carry a
        # divergent copy as a second source of truth.
        self.assertIs(config.get_text_config(decoder=True), config.text_config)
        self.assertEqual(config.text_config.pad_token_id, 11)
        self.assertEqual(config.text_config.bos_token_id, 22)
        self.assertEqual(config.text_config.eos_token_id, 33)

    def test_vision_text_hidden_size_is_synced_with_text_config(self):
        text_config = HunYuanVLTextConfig(hidden_size=111)
        config = HunYuanVLConfig(text_config=text_config)
        self.assertEqual(config.vision_config.text_hidden_size, 111)
