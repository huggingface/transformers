# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch MiniMax model."""

import unittest

from transformers import is_torch_available
from transformers.testing_utils import (
    Expectations,
    is_flaky,
    require_torch,
    require_torch_accelerator,
    slow,
    torch_device,
)


if is_torch_available():
    import torch

    from transformers import (
        MiniMaxForCausalLM,
        MiniMaxModel,
    )
    from transformers.models.minimax.modeling_minimax import (
        MiniMaxCache,
        MiniMaxLightningAttention,
        MiniMaxRotaryEmbedding,
        apply_rotary_pos_emb,
    )
from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester


class MiniMaxModelTester(CausalLMModelTester):
    if is_torch_available():
        base_model_class = MiniMaxModel

    def __init__(self, parent, layer_types=None, block_size=3):
        super().__init__(parent)
        self.layer_types = layer_types
        self.block_size = block_size


@require_torch
class MiniMaxModelTest(CausalLMModelTest, unittest.TestCase):
    model_tester_class = MiniMaxModelTester

    # TODO (ydshieh): Check this. See https://app.circleci.com/pipelines/github/huggingface/transformers/79245/workflows/9490ef58-79c2-410d-8f51-e3495156cf9c/jobs/1012146
    def is_pipeline_test_to_skip(
        self,
        pipeline_test_case_name,
        config_class,
        model_architecture,
        tokenizer_name,
        image_processor_name,
        feature_extractor_name,
        processor_name,
    ):
        return True

    @is_flaky(max_attempts=2)
    def test_load_balancing_loss(self):
        r"""
        Let's make sure we can actually compute the loss and do a backward on it.
        """
        config, input_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.num_labels = 3
        config.num_local_experts = 3
        config.output_router_logits = True
        input_ids = input_dict["input_ids"]
        attention_mask = input_ids.ne(1).to(torch_device)
        model = MiniMaxForCausalLM(config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=attention_mask)
        self.assertEqual(result.router_logits[0].shape, (91, config.num_local_experts))
        torch.testing.assert_close(result.aux_loss.cpu(), torch.tensor(2, dtype=torch.float32), rtol=1e-2, atol=1e-2)

        # First, we make sure that adding padding tokens doesn't change the loss
        # loss(input_ids, attention_mask=None) == loss(input_ids + padding, attention_mask=attention_mask_with_padding)
        pad_length = input_ids.shape[1] * 4
        # Add padding tokens (assume that pad_token_id=1) to input_ids
        padding_block = torch.ones(input_ids.shape[0], pad_length, dtype=torch.int32).to(torch_device)
        padded_input_ids = torch.cat((padding_block, input_ids), dim=1)  # this is to simulate padding to the left
        padded_attention_mask = padded_input_ids.ne(1).to(torch_device)

        padded_result = model(padded_input_ids, attention_mask=padded_attention_mask)
        torch.testing.assert_close(result.aux_loss.cpu(), padded_result.aux_loss.cpu(), rtol=1e-4, atol=1e-4)

        # We make sure that the loss of including padding tokens != the loss without padding tokens
        # if attention_mask=None --> we don't exclude padding tokens
        include_padding_result = model(padded_input_ids, attention_mask=None)

        # This is to mimic torch.testing.assert_not_close
        self.assertNotAlmostEqual(include_padding_result.aux_loss.item(), result.aux_loss.item())

    def _check_attentions_for_generate(
        self, batch_size, attentions, prompt_length, output_length, config, decoder_past_key_values
    ):
        self.assertIsInstance(attentions, tuple)
        self.assertListEqual(
            [isinstance(iter_attentions, tuple) for iter_attentions in attentions], [True] * len(attentions)
        )
        self.assertEqual(len(attentions), (output_length - prompt_length))
        use_cache = decoder_past_key_values is not None

        for generated_length, iter_attentions in enumerate(attentions):
            # regardless of using cache, the first forward pass will have the full prompt as input
            if use_cache and generated_length > 0:
                model_input_length = 1
            else:
                model_input_length = prompt_length + generated_length

            expected_shape = (
                batch_size,
                config.num_attention_heads,
                model_input_length,
                prompt_length + generated_length,
            )
            for layer_idx, layer_attention in enumerate(iter_attentions):
                if config.layer_types[layer_idx] == "full_attention":
                    self.assertEqual(layer_attention.shape, expected_shape)

    def _check_past_key_values_for_generate(self, batch_size, past_key_values, seq_length, config):
        self.assertIsInstance(past_key_values, MiniMaxCache)

        # (batch, head, seq_length, head_features)
        key_value_cache_expected_shape = (
            batch_size,
            config.num_key_value_heads,
            seq_length,
            config.hidden_size // config.num_attention_heads,
        )
        # (batch, head, head_features, head_features)
        linear_cache_expected_shape = (
            batch_size,
            config.num_attention_heads,
            config.hidden_size // config.num_attention_heads,
            config.hidden_size // config.num_attention_heads,
        )

        for layer_idx in range(config.num_hidden_layers):
            if config.layer_types[layer_idx] == "full_attention":
                self.assertEqual(past_key_values.layers[layer_idx].keys.shape, key_value_cache_expected_shape)
                self.assertEqual(past_key_values.layers[layer_idx].values.shape, key_value_cache_expected_shape)
            else:
                self.assertEqual(past_key_values.linear_cache[layer_idx].shape, linear_cache_expected_shape)

    def _check_caches_are_equal(self, cache1: MiniMaxCache, cache2: MiniMaxCache):
        if not isinstance(cache1, MiniMaxCache) or not isinstance(cache2, MiniMaxCache):
            raise ValueError("The wrong cache is being used!")

        if not len(cache1) == len(cache2):
            raise ValueError("Both caches do not have the same number of layers.")

        num_layers = len(cache1)
        for idx in range(num_layers):
            # We need this as MiniMaxCache uses the max between attention and linear caches for len...
            if idx < len(cache1.layers):
                torch.testing.assert_close(cache1.layers[idx].keys, cache1.layers[idx].keys)
                torch.testing.assert_close(cache1.layers[idx].values, cache1.layers[idx].values)
            torch.testing.assert_close(cache1.linear_cache[idx], cache2.linear_cache[idx])

    @unittest.skip(reason="MiniMaxCache does not support `crop()` method")
    def test_prompt_lookup_decoding_matches_greedy_search(self):
        pass

    @unittest.skip(reason="MiniMaxCache does not support `crop()` method")
    def test_assisted_decoding_sample(self):
        pass

    @unittest.skip(reason="MiniMaxCache does not support `crop()` method")
    def test_assisted_decoding_matches_greedy_search_0_random(self):
        pass

    @unittest.skip(reason="MiniMaxCache does not support `crop()` method")
    def test_assisted_decoding_matches_greedy_search_1_same(self):
        pass

    def test_attention_outputs(self):
        """Overridden: linear-attention layers record their decayed KV state of shape (batch, heads, head_dim,
        head_dim) in `attentions`, instead of the (batch, heads, seq_len, seq_len) probs of full-attention layers."""
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        model = MiniMaxModel._from_config(config, attn_implementation="eager").to(torch_device).eval()
        seq_len = inputs_dict["input_ids"].shape[-1]
        head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads

        with torch.no_grad():
            outputs = model(**self._prepare_for_class(inputs_dict, MiniMaxModel), output_attentions=True)

        self.assertEqual(len(outputs.attentions), config.num_hidden_layers)
        for layer_type, attention in zip(config.layer_types, outputs.attentions):
            if layer_type == "full_attention":
                self.assertEqual(attention.shape[-3:], (config.num_attention_heads, seq_len, seq_len))
            else:
                self.assertEqual(attention.shape[-3:], (config.num_attention_heads, head_dim, head_dim))

    def test_partial_rotary_embedding(self):
        config = self.model_tester.get_config()
        head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        hidden_states = torch.zeros(1, 3, config.hidden_size)
        position_ids = torch.arange(3).unsqueeze(0)

        full_rotary = MiniMaxRotaryEmbedding(config)
        full_cos, full_sin = full_rotary(hidden_states, position_ids)
        self.assertEqual(full_cos.shape[-1], head_dim)
        self.assertEqual(full_sin.shape[-1], head_dim)

        config.rope_parameters = {
            "rope_type": "default",
            "rope_theta": config.rope_parameters["rope_theta"],
            "partial_rotary_factor": 0.5,
        }
        partial_rotary = MiniMaxRotaryEmbedding(config)
        partial_cos, partial_sin = partial_rotary(hidden_states, position_ids)
        self.assertEqual(partial_cos.shape[-1], head_dim // 2)
        self.assertEqual(partial_sin.shape[-1], head_dim // 2)

        query = torch.randn(1, config.num_attention_heads, 3, head_dim)
        key = torch.randn(1, config.num_key_value_heads, 3, head_dim)
        rotated_query, rotated_key = apply_rotary_pos_emb(query, key, partial_cos, partial_sin)
        torch.testing.assert_close(rotated_query[..., head_dim // 2 :], query[..., head_dim // 2 :])
        torch.testing.assert_close(rotated_key[..., head_dim // 2 :], key[..., head_dim // 2 :])

    def test_lightning_attention_slope_uses_source_layer_denominator(self):
        config = self.model_tester.get_config()
        config.num_hidden_layers = 4
        config.layer_types = ["linear_attention"] * config.num_hidden_layers
        layer_idx = config.num_hidden_layers - 1
        attention = MiniMaxLightningAttention(config, layer_idx)

        base = 1 / (2 ** (8 / config.num_attention_heads))
        expected = base ** (torch.arange(config.num_attention_heads, dtype=torch.float32) + 1) * 1e-5
        torch.testing.assert_close(attention.get_slope_rate().flatten(), expected, rtol=1e-6, atol=0)

    def test_lightning_attention_source_padding_and_fp32_cache(self):
        config = self.model_tester.get_config()
        config.hidden_size = 4
        config.head_dim = 4
        config.num_attention_heads = 1
        config.num_key_value_heads = 1
        config.num_hidden_layers = 2
        config.layer_types = ["linear_attention"] * config.num_hidden_layers
        config.block_size = 2

        attention = MiniMaxLightningAttention(config, layer_idx=0).to(dtype=torch.float16)
        with torch.no_grad():
            identity = torch.eye(config.hidden_size, dtype=torch.float16)
            attention.qkv_proj.weight.copy_(torch.cat((identity, 0.5 * identity, 0.25 * identity)))
            attention.output_gate.weight.zero_()
            attention.out_proj.weight.copy_(identity)
            attention.norm.weight.fill_(1)

        hidden_states = torch.tensor([[[1.0, 0.5, -0.5, 2.0], [2.0, -1.0, 0.25, 0.75]]], dtype=torch.float16)
        attention_mask = torch.tensor([[1, 0]])
        cache = MiniMaxCache()

        qkv_states = attention.act_fn(attention.qkv_proj(hidden_states))
        qkv_states = qkv_states.reshape(1, 2, 1, 3 * config.head_dim)
        query_states, key_states, value_states = torch.split(qkv_states, config.head_dim, dim=-1)
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)
        value_states = value_states.masked_fill(~attention_mask[:, None, :, None].bool(), 0)

        slope_rate = attention.get_slope_rate(device=hidden_states.device)
        _, key_decay, diagonal_decay = attention.decay_factors(slope_rate)
        attention_weights = torch.matmul(query_states, key_states.transpose(-1, -2)).float()
        reference_attention = torch.matmul(attention_weights * diagonal_decay, value_states.float())
        reference_attention = reference_attention.to(query_states.dtype).transpose(1, 2).reshape(1, 2, 4)
        reference_output = attention.norm(reference_attention)
        reference_output = torch.sigmoid(attention.output_gate(hidden_states)) * reference_output
        reference_output = attention.out_proj(reference_output)
        reference_cache = torch.matmul(
            (key_states * key_decay).transpose(-1, -2).to(value_states.dtype), value_states
        ).float()

        output, recurrent_state = attention(
            hidden_states,
            position_embeddings=None,
            attention_mask=attention_mask,
            past_key_values=cache,
        )
        torch.testing.assert_close(output, reference_output)
        torch.testing.assert_close(recurrent_state, reference_cache)
        self.assertEqual(recurrent_state.dtype, torch.float32)
        self.assertEqual(cache.linear_cache[0].dtype, torch.float32)

        # The source masks values only. A padded query can still read preceding unpadded values.
        self.assertGreater(output[0, 1].abs().max().item(), 0)

        decode_hidden_states = torch.tensor([[[0.75, -0.25, 1.25, 0.5]]], dtype=torch.float16)
        decode_qkv_states = attention.act_fn(attention.qkv_proj(decode_hidden_states)).reshape(1, 1, 1, 12)
        decode_query, decode_key, decode_value = torch.split(decode_qkv_states, config.head_dim, dim=-1)
        decode_query = decode_query.transpose(1, 2)
        decode_key = decode_key.transpose(1, 2)
        decode_value = decode_value.transpose(1, 2)
        reference_cache = torch.exp(-slope_rate) * reference_cache + torch.matmul(
            decode_key.transpose(-1, -2), decode_value
        )
        reference_attention = torch.matmul(decode_query, reference_cache.to(decode_query.dtype))
        reference_attention = reference_attention.transpose(1, 2).reshape(1, 1, 4)
        reference_output = attention.norm(reference_attention)
        reference_output = torch.sigmoid(attention.output_gate(decode_hidden_states)) * reference_output
        reference_output = attention.out_proj(reference_output)

        output, recurrent_state = attention(
            decode_hidden_states,
            position_embeddings=None,
            attention_mask=None,
            past_key_values=cache,
        )
        torch.testing.assert_close(output, reference_output)
        torch.testing.assert_close(recurrent_state, reference_cache)
        self.assertEqual(recurrent_state.dtype, torch.float32)

    def test_hybrid_cache_sequence_length_uses_full_attention_layer(self):
        cache = MiniMaxCache()
        key_states = torch.randn(1, 2, 5, 4)
        value_states = torch.randn(1, 2, 5, 4)

        cache.update(key_states, value_states, layer_idx=7)

        self.assertEqual(cache.get_seq_length(), 5)
        self.assertEqual(cache.get_seq_length(layer_idx=7), 5)
        self.assertEqual(cache.get_seq_length(layer_idx=3), 0)

    @unittest.skip("MiniMax is special")
    def test_flash_attention_2_padding_matches_padding_free_with_position_ids(self):
        pass

    @unittest.skip("MiniMax is special")
    def test_flash_attention_2_padding_matches_padding_free_with_position_ids_and_fa_kwargs(self):
        pass

    @unittest.skip("MiniMax is special")
    def test_eager_padding_matches_padding_free_with_position_ids(self):
        pass

    @unittest.skip("MiniMax is special")
    def test_sdpa_padding_matches_padding_free_with_position_ids(self):
        pass


@require_torch
@require_torch_accelerator
@slow
class MiniMaxIntegrationTest(unittest.TestCase):
    def test_small_model_logits(self):
        model_id = "hf-internal-testing/MiniMax-tiny"
        dummy_input = torch.LongTensor([[0, 1, 0], [0, 1, 0]]).to(torch_device)

        model = MiniMaxForCausalLM.from_pretrained(
            model_id,
            dtype=torch.bfloat16,
        ).to(torch_device)

        with torch.no_grad():
            logits = model(dummy_input).logits

        logits = logits.float()

        expectations = Expectations(
            {
                (None, None): [[1.0312, -0.5156, -0.3262], [-0.1152, 0.4336, 0.2412], [1.2188, -0.5898, -0.0381]],
                ("cuda", 8): [[1.0312, -0.5156, -0.3203], [-0.1201, 0.4375, 0.2402], [1.2188, -0.5898, -0.0396]],
            }
        )
        expected_slice = torch.tensor(expectations.get_expectation()).to(torch_device)

        torch.testing.assert_close(logits[0, :3, :3], expected_slice, atol=1e-3, rtol=1e-3)
        torch.testing.assert_close(logits[1, :3, :3], expected_slice, atol=1e-3, rtol=1e-3)

    def test_small_model_generation(self):
        model_id = "hf-internal-testing/MiniMax-tiny"
        dummy_input = torch.LongTensor([[0, 1, 0], [0, 1, 0]]).to(torch_device)

        model = MiniMaxForCausalLM.from_pretrained(
            model_id,
            dtype=torch.bfloat16,
        ).to(torch_device)
        expected_slice = (
            torch.tensor([[0, 1, 0, 933, 307, 3102, 2457, 1208], [0, 1, 0, 933, 307, 3102, 2457, 1208]])
            .to(torch.int64)
            .to(torch_device)
        )

        outputs = model.generate(dummy_input, max_new_tokens=5, do_sample=False)

        torch.testing.assert_close(outputs, expected_slice, atol=1e-3, rtol=1e-3)
