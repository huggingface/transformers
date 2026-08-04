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

from transformers import TinyModel, TinyModelConfig, TinyModelForCausalLM, is_torch_available
from transformers.models.tiny_model.modeling_tiny_model import (
    TinyModelAttention,
    TinyModelDecoderLayer,
    TinyModelMLP,
    TinyModelPreTrainedModel,
    eager_attention_forward,
)
from transformers.testing_utils import require_torch

from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester


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

        actual, weights = eager_attention_forward(nn.Identity(), query, key, value, attention_mask, scaling=8**-0.5)
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


class TinyModelPreTrainedModelTest(unittest.TestCase):
    def test_embedding_initialization_uses_checkpoint_scale(self):
        config = TinyModelConfig(
            vocab_size=1_000,
            hidden_size=16,
            intermediate_size=64,
            num_attention_heads=4,
            embedding_initializer_range=1e-4,
        )
        model = TinyModelPreTrainedModel(config)
        embedding = nn.Embedding(config.vocab_size, config.hidden_size)

        torch.manual_seed(4)
        model._init_weights(embedding)

        self.assertAlmostEqual(embedding.weight.std().item(), config.embedding_initializer_range, delta=5e-6)


class TinyModelTest(unittest.TestCase):
    def get_config(self):
        return TinyModelConfig(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            max_position_embeddings=8,
            bos_token_id=28,
            eos_token_id=29,
            pad_token_id=30,
        )

    def test_model_structure_matches_checkpoint_namespace(self):
        config = self.get_config()
        model = TinyModel(config)

        self.assertEqual(model.embed_tokens.weight.shape, (32, 16))
        self.assertEqual(model.embed_positions.weight.shape, (8, 16))
        self.assertIsNone(model.embed_tokens.padding_idx)
        self.assertEqual(len(model.layers), 2)
        self.assertIn("layers.0.self_attn.q_proj.weight", model.state_dict())
        self.assertIn("layers.1.mlp.fc2.bias", model.state_dict())

    def test_forward_supports_input_ids_and_input_embeddings(self):
        torch.manual_seed(5)
        model = TinyModel(self.get_config()).eval()
        input_ids = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]])

        with torch.no_grad():
            from_ids = model(input_ids=input_ids).last_hidden_state
            from_embeds = model(inputs_embeds=model.embed_tokens(input_ids)).last_hidden_state

        self.assertEqual(from_ids.shape, (2, 4, 16))
        torch.testing.assert_close(from_ids, from_embeds, rtol=0, atol=0)

    def test_forward_requires_exactly_one_input(self):
        model = TinyModel(self.get_config())
        input_ids = torch.tensor([[1, 2, 3]])

        with self.assertRaisesRegex(ValueError, "exactly one"):
            model()
        with self.assertRaisesRegex(ValueError, "exactly one"):
            model(input_ids=input_ids, inputs_embeds=model.embed_tokens(input_ids))


class TinyModelForCausalLMTest(unittest.TestCase):
    def get_config(self):
        return TinyModelConfig(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            max_position_embeddings=8,
            bos_token_id=28,
            eos_token_id=29,
            pad_token_id=30,
        )

    def test_model_structure_has_untied_biased_lm_head(self):
        model = TinyModelForCausalLM(self.get_config())

        self.assertEqual(model.lm_head.weight.shape, (32, 16))
        self.assertEqual(model.lm_head.bias.shape, (32,))
        self.assertNotEqual(model.model.embed_tokens.weight.data_ptr(), model.lm_head.weight.data_ptr())
        self.assertIn("model.embed_tokens.weight", model.state_dict())
        self.assertIn("lm_head.bias", model.state_dict())

    def test_dynamic_cache_is_not_advertised(self):
        self.assertFalse(TinyModelForCausalLM._supports_default_dynamic_cache())

    def test_greedy_generation_without_cache(self):
        model = TinyModelForCausalLM(self.get_config()).eval()
        input_ids = torch.tensor([[1, 2, 3]])

        output_ids = model.generate(input_ids, max_new_tokens=2, do_sample=False)

        self.assertEqual(output_ids.shape, (1, 5))
        self.assertFalse(model.generation_config.use_cache)

    def test_generation_from_input_embeddings_is_rejected(self):
        model = TinyModelForCausalLM(self.get_config())
        input_ids = torch.tensor([[1, 2, 3]])

        with self.assertRaisesRegex(ValueError, "cannot generate from `inputs_embeds`"):
            model.prepare_inputs_for_generation(
                input_ids,
                inputs_embeds=model.model.embed_tokens(input_ids),
                use_cache=False,
            )

    def test_forward_returns_raw_logits_and_loss(self):
        torch.manual_seed(6)
        model = TinyModelForCausalLM(self.get_config()).eval()
        input_ids = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]])

        with torch.no_grad():
            outputs = model(input_ids=input_ids, labels=input_ids)
            hidden_states = model.model(input_ids=input_ids).last_hidden_state
            expected_logits = model.lm_head(hidden_states)

        torch.testing.assert_close(outputs.logits, expected_logits, rtol=0, atol=0)
        self.assertEqual(outputs.logits.shape, (2, 4, 32))
        self.assertTrue(torch.isfinite(outputs.loss))

    def test_logits_to_keep(self):
        model = TinyModelForCausalLM(self.get_config()).eval()
        outputs = model(input_ids=torch.tensor([[1, 2, 3, 4]]), logits_to_keep=1)

        self.assertEqual(outputs.logits.shape, (1, 1, 32))

    def test_cache_inputs_are_rejected(self):
        model = TinyModelForCausalLM(self.get_config())
        input_ids = torch.tensor([[1, 2, 3]])

        with self.assertRaisesRegex(ValueError, "does not support key/value caching"):
            model(input_ids=input_ids, use_cache=True)
        with self.assertRaisesRegex(ValueError, "does not support `past_key_values`"):
            model(input_ids=input_ids, past_key_values=object())

    def test_matches_source_equations_on_contiguous_inputs(self):
        torch.manual_seed(7)
        model = TinyModelForCausalLM(self.get_config()).eval()
        input_ids = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]])

        with torch.no_grad():
            native_logits = model(input_ids).logits

            position_ids = torch.arange(input_ids.shape[1]).unsqueeze(0)
            hidden_states = model.model.embed_tokens(input_ids) + model.model.embed_positions(position_ids)
            for layer in model.model.layers:
                batch_size, sequence_length, _ = hidden_states.shape
                query = layer.self_attn.q_proj(hidden_states).view(batch_size, sequence_length, 4, 4).transpose(1, 2)
                key = layer.self_attn.k_proj(hidden_states).view(batch_size, sequence_length, 4, 4).transpose(1, 2)
                value = layer.self_attn.v_proj(hidden_states).view(batch_size, sequence_length, 4, 4).transpose(1, 2)
                attention_output = nn.functional.scaled_dot_product_attention(query, key, value, is_causal=True)
                attention_output = layer.self_attn.o_proj(
                    attention_output.transpose(1, 2).reshape(batch_size, sequence_length, 16)
                )
                hidden_states = hidden_states + attention_output
                hidden_states = hidden_states + layer.mlp.fc2(torch.relu(layer.mlp.fc1(hidden_states)))
            original_logits = model.lm_head(hidden_states)
            original_log_probabilities = torch.log_softmax(original_logits, dim=-1)

        torch.testing.assert_close(native_logits, original_logits, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(
            torch.log_softmax(native_logits, dim=-1), original_log_probabilities, rtol=1e-5, atol=1e-6
        )


class TinyModelModelTester(CausalLMModelTester):
    config_class = TinyModelConfig
    if is_torch_available():
        base_model_class = TinyModel
        causal_lm_class = TinyModelForCausalLM

    def __init__(self, parent):
        super().__init__(
            parent=parent,
            hidden_act="relu",
            intermediate_size=64,
            max_position_embeddings=64,
        )


@require_torch
class TinyModelModelTest(CausalLMModelTest, unittest.TestCase):
    model_tester_class = TinyModelModelTester
    training_overfit_learning_rate = 2e-3

    def _greedy_generate(self, *args, use_cache=False, **kwargs):
        return super()._greedy_generate(*args, use_cache=use_cache, **kwargs)

    def _sample_generate(self, *args, use_cache=False, **kwargs):
        return super()._sample_generate(*args, use_cache=use_cache, **kwargs)

    def _beam_search_generate(self, *args, use_cache=False, **kwargs):
        return super()._beam_search_generate(*args, use_cache=use_cache, **kwargs)

    def _beam_sample_generate(self, *args, use_cache=False, **kwargs):
        return super()._beam_sample_generate(*args, use_cache=use_cache, **kwargs)

    @unittest.skip("TinyModel does not support caching, but this common test explicitly enables it")
    def test_generate_methods_with_logits_to_keep(self):
        pass

    @unittest.skip("TinyModel does not support caching, but this common test explicitly enables it")
    def test_generate_with_and_without_position_ids(self):
        pass
