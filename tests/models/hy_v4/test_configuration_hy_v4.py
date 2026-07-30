# Copyright 2026 The HuggingFace Team. All rights reserved.
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

from transformers.models.hy_v4.configuration_hy_v4 import HYV4Config


class HYV4ConfigTest(unittest.TestCase):
    def test_release_defaults(self):
        config = HYV4Config()

        self.assertEqual(config.model_type, "hy_v4")
        self.assertEqual(config.num_hidden_layers, 34)
        self.assertEqual(config.hidden_size, 2816)
        # `head_dim` is pointed at the RoPE slice so the inherited rotary embedding sizes correctly;
        # the full MLA Q/K dimension is `qk_head_dim`.
        self.assertEqual(config.head_dim, 64)
        self.assertEqual(config.qk_head_dim, 256)
        self.assertEqual((config.bos_token_id, config.eos_token_id, config.pad_token_id), (120000, 120025, 120002))
        self.assertEqual(config.mlp_layer_types, ["dense"] + ["sparse"] * 33)
        self.assertEqual(config.layer_types, ["deepseek_sparse_attention"] * 34)
        self.assertEqual(
            [index for index, value in enumerate(config.indexer_types) if value == "full"],
            [0, 1, 5, 9, 13, 17, 21, 25, 29, 33],
        )
        self.assertFalse(hasattr(config, "hy_v4_schema_version"))
        self.assertFalse(hasattr(config, "rope_interleave"))
        self.assertFalse(hasattr(config, "n_group"))

    def test_round_trip(self):
        config = HYV4Config(num_hidden_layers=2)
        restored = HYV4Config.from_dict(config.to_dict())

        self.assertEqual(restored.to_dict(), config.to_dict())
        self.assertEqual(restored.qk_head_dim, 256)


if __name__ == "__main__":
    unittest.main()
