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

from transformers.models.tiny_model import TinyModelConfig


class TinyModelConfigTest(unittest.TestCase):
    def test_checkpoint_defaults(self):
        config = TinyModelConfig()

        self.assertEqual(config.vocab_size, 10_000)
        self.assertEqual(config.hidden_size, 768)
        self.assertEqual(config.intermediate_size, 3_072)
        self.assertEqual(config.num_hidden_layers, 4)
        self.assertEqual(config.num_attention_heads, 16)
        self.assertEqual(config.max_position_embeddings, 256)
        self.assertEqual(config.hidden_act, "relu")
        self.assertFalse(config.attention_bias)
        self.assertTrue(config.attention_output_bias)
        self.assertTrue(config.mlp_bias)
        self.assertTrue(config.lm_head_bias)
        self.assertFalse(config.tie_word_embeddings)
        self.assertFalse(config.use_cache)
        self.assertEqual(config.bos_token_id, 9_996)
        self.assertEqual(config.eos_token_id, 9_997)
        self.assertEqual(config.pad_token_id, 9_998)
