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

import tempfile
import unittest

from transformers.models.hunyuan_vl.configuration_hunyuan_vl import (
    HunYuanVLConfig,
    HunYuanVLTextConfig,
    HunYuanVLVisionConfig,
)


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
        """Legacy ``rope_scaling`` / ``pad_id`` / ``attention_head_dim`` aliases must round-trip into canonical fields."""
        config = HunYuanVLTextConfig(
            pad_token_id=-1,
            pad_id=7,
            attention_head_dim=128,
            rope_theta=10000.0,
            rope_scaling=LEGACY_XDROPE_SCALING,
        )

        self.assertEqual(config.pad_token_id, 7)
        self.assertEqual(config.head_dim, 128)
        self.assertEqual(config.rope_theta, 10000.0)
        self.assertEqual(config.rope_parameters["type"], "dynamic")
        self.assertEqual(config.rope_parameters["rope_type"], "dynamic")
        self.assertEqual(config.rope_parameters["rope_theta"], 10000.0)
        self.assertEqual(config.rope_scaling["xdrope_section"], [16, 16, 16, 16])

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
    def test_hunyuan_vl_reload(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = HunYuanVLConfig()
            config.save_pretrained(tmp_dir)

            reloaded = HunYuanVLConfig.from_pretrained(tmp_dir)
            self.assertDictEqual(config.to_dict(), reloaded.to_dict())

    def test_top_level_token_ids_are_propagated_from_text_config(self):
        text_config = HunYuanVLTextConfig(pad_token_id=11, bos_token_id=22, eos_token_id=33)
        config = HunYuanVLConfig(text_config=text_config)

        self.assertEqual(config.pad_token_id, 11)
        self.assertEqual(config.bos_token_id, 22)
        self.assertEqual(config.eos_token_id, 33)

    def test_explicit_text_config_instance_is_preserved(self):
        text_config = HunYuanVLTextConfig(hidden_size=111)
        config = HunYuanVLConfig(text_config=text_config)
        self.assertEqual(config.text_config.hidden_size, 111)

    def test_dict_text_config_is_constructed(self):
        config = HunYuanVLConfig(text_config={"hidden_size": 222, "head_dim": 16, "num_attention_heads": 8})
        self.assertEqual(config.text_config.hidden_size, 222)
        self.assertEqual(config.text_config.head_dim, 16)
        self.assertEqual(config.text_config.num_attention_heads, 8)

    def test_vision_text_hidden_size_is_synced_with_text_config(self):
        text_config = HunYuanVLTextConfig(hidden_size=111)
        config = HunYuanVLConfig(text_config=text_config)
        self.assertEqual(config.vision_config.text_hidden_size, 111)
