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

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM
from transformers.models.hy_v4.configuration_hy_v4 import HYV4Config
from transformers.models.hy_v4.modeling_hy_v4 import HYV4ForCausalLM, HYV4Model


class HYV4AutoTest(unittest.TestCase):
    def tiny_config(self):
        return HYV4Config(
            vocab_size=99,
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=2,
            hidden_size=32,
            intermediate_size=64,
            moe_intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=4,
            head_dim=8,
            q_lora_rank=16,
            kv_lora_rank=8,
            qk_nope_head_dim=4,
            qk_rope_head_dim=4,
            v_head_dim=8,
            mlp_layer_types=["dense"],
            index_head_dim=8,
            index_n_heads=4,
            indexer_types=["full"],
            enable_lm_head_fp32=False,
            enable_ihc=False,
            gated_mla=False,
            learnable_sink=False,
            rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
        )

    def test_auto_config_uses_public_module(self):
        config = AutoConfig.for_model("hy_v4")
        self.assertIsInstance(config, HYV4Config)
        self.assertEqual(config.__class__.__module__, "transformers.models.hy_v4.configuration_hy_v4")

    def test_auto_models_use_public_module(self):
        config = self.tiny_config()
        model = AutoModel.from_config(config)
        causal_lm = AutoModelForCausalLM.from_config(config)

        self.assertIsInstance(model, HYV4Model)
        self.assertIsInstance(causal_lm, HYV4ForCausalLM)
        self.assertEqual(model.__class__.__module__, "transformers.models.hy_v4.modeling_hy_v4")
        self.assertEqual(causal_lm.__class__.__module__, "transformers.models.hy_v4.modeling_hy_v4")


if __name__ == "__main__":
    unittest.main()
