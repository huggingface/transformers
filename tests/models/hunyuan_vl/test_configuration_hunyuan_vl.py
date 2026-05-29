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

import tempfile
import unittest

from transformers.models.hunyuan_vl.configuration_hunyuan_vl import HunYuanVLConfig, HunYuanVLTextConfig


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

    def test_legacy_moe_fields_are_preserved_for_checkpoint_compatibility(self):
        config = HunYuanVLTextConfig(num_experts=4, moe_topk=2)

        self.assertEqual(config.num_experts, 4)
        self.assertEqual(config.moe_topk, 2)


class HunYuanVLConfigTest(unittest.TestCase):
    def test_hunyuan_vl_reload(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = HunYuanVLConfig()
            config.save_pretrained(tmp_dir)

            reloaded = HunYuanVLConfig.from_pretrained(tmp_dir)
            self.assertDictEqual(config.to_dict(), reloaded.to_dict())

    def test_top_level_legacy_text_fields_are_normalized(self):
        config = HunYuanVLConfig(
            pad_token_id=-1,
            pad_id=7,
            attention_head_dim=128,
            rope_theta=10000.0,
            rope_scaling=LEGACY_XDROPE_SCALING,
        )

        text_config = config.text_config
        self.assertEqual(text_config.pad_token_id, 7)
        self.assertEqual(text_config.head_dim, 128)
        self.assertEqual(text_config.rope_theta, 10000.0)
        self.assertEqual(text_config.rope_parameters["type"], "dynamic")
        self.assertEqual(text_config.rope_parameters["rope_type"], "dynamic")
        self.assertEqual(text_config.rope_parameters["rope_theta"], 10000.0)
        self.assertEqual(text_config.rope_scaling["xdrope_section"], [16, 16, 16, 16])

        with tempfile.TemporaryDirectory() as tmp_dir:
            config.save_pretrained(tmp_dir)
            reloaded = HunYuanVLConfig.from_pretrained(tmp_dir)

        reloaded_text_config = reloaded.text_config
        self.assertEqual(reloaded_text_config.pad_token_id, 7)
        self.assertEqual(reloaded_text_config.head_dim, 128)
        self.assertEqual(reloaded_text_config.rope_parameters["type"], "dynamic")
        self.assertEqual(reloaded_text_config.rope_scaling["xdrope_section"], [16, 16, 16, 16])

    def test_top_level_text_overrides_are_applied_to_text_config_instances(self):
        config = HunYuanVLConfig(text_config=HunYuanVLTextConfig(hidden_size=111), hidden_size=222, rope_theta=12345.0)

        self.assertEqual(config.text_config.hidden_size, 222)
        self.assertEqual(config.text_config.rope_theta, 12345.0)
