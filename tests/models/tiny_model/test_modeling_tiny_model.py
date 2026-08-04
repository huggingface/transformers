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

import torch
from huggingface_hub.errors import StrictDataclassClassValidationError
from torch import nn

from transformers.models.tiny_model import TinyModelConfig
from transformers.models.tiny_model.modeling_tiny_model import (
    TinyModelAttention,
    TinyModelDecoderLayer,
    TinyModelMLP,
    eager_attention_forward,
)


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
        self.assertFalse(hasattr(config, "use_cache"))
        self.assertEqual(config.bos_token_id, 9_996)
        self.assertEqual(config.eos_token_id, 9_997)
        self.assertEqual(config.pad_token_id, 9_998)

    def test_hidden_size_must_be_divisible_by_attention_heads(self):
        with self.assertRaisesRegex(StrictDataclassClassValidationError, "hidden size .* is not a multiple"):
            TinyModelConfig(hidden_size=15, num_attention_heads=4)


class TinyModelAttentionFunctionTest(unittest.TestCase):
    def test_matches_causal_scaled_dot_product_attention(self):
        torch.manual_seed(0)
        query = torch.randn(2, 4, 5, 8)
        key = torch.randn(2, 4, 5, 8)
        value = torch.randn(2, 4, 5, 8)
        attention_mask = torch.full((1, 1, 5, 5), float("-inf")).triu(diagonal=1)

        actual, weights = eager_attention_forward(
            nn.Identity(), query, key, value, attention_mask, scaling=8**-0.5
        )
        expected = nn.functional.scaled_dot_product_attention(query, key, value, is_causal=True)

        torch.testing.assert_close(actual, expected.transpose(1, 2), rtol=1e-5, atol=1e-6)
        self.assertTrue(torch.equal(weights.triu(diagonal=1), torch.zeros_like(weights)))


class TinyModelAttentionTest(unittest.TestCase):
    def test_projection_shapes_and_biases(self):
        attention = TinyModelAttention(TinyModelConfig(hidden_size=16, num_attention_heads=4))

        self.assertEqual(attention.head_dim, 4)
        self.assertEqual(attention.scaling, 0.5)
        self.assertEqual(attention.q_proj.weight.shape, (16, 16))
        self.assertIsNone(attention.q_proj.bias)
        self.assertIsNone(attention.k_proj.bias)
        self.assertIsNone(attention.v_proj.bias)
        self.assertEqual(attention.o_proj.bias.shape, (16,))

    def test_forward_matches_causal_scaled_dot_product_attention(self):
        torch.manual_seed(1)
        attention = TinyModelAttention(TinyModelConfig(hidden_size=16, num_attention_heads=4)).eval()
        hidden_states = torch.randn(2, 5, 16)
        attention_mask = torch.full((1, 1, 5, 5), float("-inf")).triu(diagonal=1)

        actual, weights = attention(hidden_states, attention_mask=attention_mask)
        query = attention.q_proj(hidden_states).view(2, 5, 4, 4).transpose(1, 2)
        key = attention.k_proj(hidden_states).view(2, 5, 4, 4).transpose(1, 2)
        value = attention.v_proj(hidden_states).view(2, 5, 4, 4).transpose(1, 2)
        expected = nn.functional.scaled_dot_product_attention(query, key, value, is_causal=True)
        expected = attention.o_proj(expected.transpose(1, 2).reshape(2, 5, 16))

        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        self.assertEqual(weights.shape, (2, 4, 5, 5))


class TinyModelMLPTest(unittest.TestCase):
    def test_projection_shapes_and_biases(self):
        mlp = TinyModelMLP(TinyModelConfig(hidden_size=16, intermediate_size=64, num_attention_heads=4))

        self.assertEqual(mlp.fc1.weight.shape, (64, 16))
        self.assertEqual(mlp.fc1.bias.shape, (64,))
        self.assertEqual(mlp.fc2.weight.shape, (16, 64))
        self.assertEqual(mlp.fc2.bias.shape, (16,))

    def test_forward_is_relu_feed_forward(self):
        torch.manual_seed(2)
        mlp = TinyModelMLP(TinyModelConfig(hidden_size=16, intermediate_size=64, num_attention_heads=4))
        hidden_states = torch.randn(2, 5, 16)

        actual = mlp(hidden_states)
        expected = mlp.fc2(torch.relu(mlp.fc1(hidden_states)))

        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


class TinyModelDecoderLayerTest(unittest.TestCase):
    def test_contains_attention_and_mlp_without_norms(self):
        layer = TinyModelDecoderLayer(TinyModelConfig(hidden_size=16, intermediate_size=64, num_attention_heads=4))

        self.assertIsInstance(layer.self_attn, TinyModelAttention)
        self.assertIsInstance(layer.mlp, TinyModelMLP)
        self.assertFalse(any(isinstance(module, nn.LayerNorm) for module in layer.modules()))

    def test_forward_has_two_residual_connections(self):
        torch.manual_seed(3)
        layer = TinyModelDecoderLayer(
            TinyModelConfig(hidden_size=16, intermediate_size=64, num_attention_heads=4)
        ).eval()
        hidden_states = torch.randn(2, 5, 16)
        attention_mask = torch.full((1, 1, 5, 5), float("-inf")).triu(diagonal=1)

        actual = layer(hidden_states, attention_mask=attention_mask)
        attention_output, _ = layer.self_attn(hidden_states, attention_mask=attention_mask)
        post_attention = hidden_states + attention_output
        expected = post_attention + layer.mlp(post_attention)

        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
