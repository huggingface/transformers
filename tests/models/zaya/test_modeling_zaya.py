# Copyright 2026 Zyphra and the HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch ZAYA model."""

import unittest

from huggingface_hub.errors import StrictDataclassClassValidationError
from parameterized import parameterized

from transformers import is_torch_available
from transformers.testing_utils import cleanup, require_torch, slow, torch_device


if is_torch_available():
    import torch

    from transformers import AutoTokenizer, ZayaConfig, ZayaForCausalLM, ZayaModel
    from transformers.cache_utils import DynamicCache, LinearAttentionAndFullAttentionLayer
    from transformers.models.zaya.modeling_zaya import ZayaCCAProjection

from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester


class ZayaModelTester(CausalLMModelTester):
    if is_torch_available():
        base_model_class = ZayaModel

    def __init__(self, parent):
        super().__init__(
            parent=parent,
            batch_size=2,
            seq_length=7,
            vocab_size=128,
            hidden_size=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            intermediate_size=64,
        )
        self.head_dim = 8
        self.num_experts = 4
        self.num_experts_per_tok = 1
        self.zaya_mlp_expansion = 4
        self.tie_word_embeddings = False
        self.rope_parameters = {
            "full_attention": {
                "rope_theta": 10000,
                "rope_type": "default",
                "partial_rotary_factor": 0.5,
            },
        }


@require_torch
class ZayaModelTest(CausalLMModelTest, unittest.TestCase):
    model_tester_class = ZayaModelTester
    test_all_params_have_gradient = False

    def _get_conv_state_shape(self, batch_size: int, config):
        conv_state_size = config.num_key_value_heads * config.head_dim + config.num_attention_heads * config.head_dim
        return (batch_size, conv_state_size, config.cca_time0 + config.cca_time1 - 2)

    def _get_recurrent_state_shape(self, batch_size: int, config):
        return (batch_size, config.num_key_value_heads * config.head_dim // 2)

    def _check_past_key_values_for_generate(self, batch_size, past_key_values, seq_length, config):
        if not isinstance(past_key_values, DynamicCache):
            raise ValueError("The cache does not use the correct Cache")

        config = config.get_text_config(decoder=True)
        self.assertEqual(config.num_hidden_layers, len(past_key_values))
        attention_shape = (batch_size, config.num_key_value_heads, seq_length, config.head_dim)
        conv_shape = self._get_conv_state_shape(batch_size, config)
        recurrent_shape = self._get_recurrent_state_shape(batch_size, config)

        for layer in past_key_values.layers:
            self.assertIs(type(layer), LinearAttentionAndFullAttentionLayer)
            self.assertEqual(layer.keys.shape, attention_shape)
            self.assertEqual(layer.values.shape, attention_shape)
            self.assertEqual(layer.conv_states.shape, conv_shape)
            self.assertEqual(layer.recurrent_states.shape, recurrent_shape)

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

    @unittest.skip("ZAYA uses key/query normalization which is not equivalent under padding-free packing.")
    def test_eager_padding_matches_padding_free_with_position_ids(self):
        pass

    @unittest.skip("ZAYA uses key/query normalization which is not equivalent under padding-free packing.")
    def test_sdpa_padding_matches_padding_free_with_position_ids(self):
        pass

    @unittest.skip("ZAYA uses MoE routing; equivalent-output comparisons are not stable for this architecture.")
    def test_model_outputs_equivalence(self, **kwargs):
        pass

    def test_attention_outputs(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.return_dict = True
        config._attn_implementation = "eager"

        for model_class in self.all_model_classes:
            model = model_class._from_config(config, attn_implementation="eager")
            model.to(torch_device)
            model.eval()

            with torch.no_grad():
                outputs = model(**self._prepare_for_class({**inputs_dict, "output_attentions": True}, model_class))

            expected_attn_layers = config.num_hidden_layers
            self.assertEqual(len(outputs.attentions), expected_attn_layers)
            self.assertEqual(
                outputs.attentions[0].shape,
                (
                    self.model_tester.batch_size,
                    config.num_attention_heads,
                    self.model_tester.seq_length,
                    self.model_tester.seq_length,
                ),
            )

    @parameterized.expand([("linear",), ("dynamic",), ("yarn",)])
    @unittest.skip(
        "RoPE-scaling-from-config test doesn't match ZAYA's nested per-layer-type rope_parameters (same as e.g. Laguna, Gemma3)."
    )
    def test_model_rope_scaling_from_config(self, scaling_type):
        pass

    def test_model_rope_scaling_frequencies(self):
        """
        Tests the frequency properties of the different RoPE scaling types on the model RoPE layer.
        Copied from Laguna to adapt to per-layer-type rope configs.
        """
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        config.layer_types = ["full_attention", "sliding_attention"]
        partial_rotary_factor = config.partial_rotary_factor

        def set_rope_params(rope_params):
            config.rope_parameters = {
                "full_attention": {**rope_params, "partial_rotary_factor": partial_rotary_factor},
                "sliding_attention": {**rope_params, "partial_rotary_factor": partial_rotary_factor},
            }

        set_rope_params({"rope_type": "default", "rope_theta": 10_000.0})

        base_model = self.model_tester.base_model_class(config)
        possible_rope_attributes = [
            "pos_emb",
            "rotary_emb",
            "global_rotary_emb",
            "local_rotary_emb",
        ]
        for name, module in base_model.named_modules():
            if any(potential_name in name for potential_name in possible_rope_attributes):
                rope_class = type(module)
                break

        scaling_factor = 10
        short_input_length = 10
        long_input_length = int(config.max_position_embeddings * 1.5)

        x = torch.randn(1, dtype=torch.float32, device=torch_device)
        position_ids_short = torch.arange(short_input_length, dtype=torch.long, device=torch_device).unsqueeze(0)
        position_ids_long = torch.arange(long_input_length, dtype=torch.long, device=torch_device).unsqueeze(0)

        set_rope_params({"rope_type": "default", "rope_theta": 10_000.0})
        original_rope = rope_class(config=config).to(torch_device)
        original_cos_short, original_sin_short = original_rope(x, position_ids_short, layer_type="sliding_attention")
        original_cos_long, original_sin_long = original_rope(x, position_ids_long, layer_type="sliding_attention")
        torch.testing.assert_close(original_cos_short, original_cos_long[:, :short_input_length, :])
        torch.testing.assert_close(original_sin_short, original_sin_long[:, :short_input_length, :])

        set_rope_params({"rope_type": "linear", "factor": scaling_factor, "rope_theta": 10_000.0})
        linear_scaling_rope = rope_class(config=config).to(torch_device)
        linear_cos_short, linear_sin_short = linear_scaling_rope(x, position_ids_short, layer_type="sliding_attention")
        linear_cos_long, linear_sin_long = linear_scaling_rope(x, position_ids_long, layer_type="sliding_attention")
        torch.testing.assert_close(linear_cos_short, linear_cos_long[:, :short_input_length, :])
        torch.testing.assert_close(linear_sin_short, linear_sin_long[:, :short_input_length, :])
        for new_position in range(0, long_input_length, scaling_factor):
            original_position = int(new_position // scaling_factor)
            torch.testing.assert_close(linear_cos_long[:, new_position, :], original_cos_long[:, original_position, :])
            torch.testing.assert_close(linear_sin_long[:, new_position, :], original_sin_long[:, original_position, :])

        set_rope_params({"rope_type": "dynamic", "factor": scaling_factor, "rope_theta": 10_000.0})
        ntk_scaling_rope = rope_class(config=config).to(torch_device)
        ntk_cos_short, ntk_sin_short = ntk_scaling_rope(x, position_ids_short, layer_type="sliding_attention")
        ntk_cos_long, ntk_sin_long = ntk_scaling_rope(x, position_ids_long, layer_type="sliding_attention")
        torch.testing.assert_close(ntk_cos_short, original_cos_short)
        torch.testing.assert_close(ntk_sin_short, original_sin_short)
        with self.assertRaises(AssertionError):
            torch.testing.assert_close(ntk_cos_long, original_cos_long)
        with self.assertRaises(AssertionError):
            torch.testing.assert_close(ntk_sin_long, original_sin_long)
        self.assertTrue(
            (ntk_scaling_rope.sliding_attention_inv_freq <= original_rope.sliding_attention_inv_freq).all()
        )

        set_rope_params({"rope_type": "yarn", "factor": scaling_factor, "rope_theta": 10_000.0})
        yarn_scaling_rope = rope_class(config=config).to(torch_device)
        yarn_cos_short, yarn_sin_short = yarn_scaling_rope(x, position_ids_short, layer_type="sliding_attention")
        yarn_cos_long, yarn_sin_long = yarn_scaling_rope(x, position_ids_long, layer_type="sliding_attention")
        torch.testing.assert_close(yarn_cos_short, yarn_cos_long[:, :short_input_length, :])
        torch.testing.assert_close(yarn_sin_short, yarn_sin_long[:, :short_input_length, :])
        with self.assertRaises(AssertionError):
            torch.testing.assert_close(yarn_cos_short, original_cos_short)
        with self.assertRaises(AssertionError):
            torch.testing.assert_close(yarn_sin_short, original_sin_short)
        with self.assertRaises(AssertionError):
            torch.testing.assert_close(yarn_cos_long, original_cos_long)
        with self.assertRaises(AssertionError):
            torch.testing.assert_close(yarn_sin_long, original_sin_long)

    @unittest.skip("ZAYA needs alternating attention and MoE layers in the tiny test configuration.")
    def test_num_layers_is_small(self):
        pass

    def test_moe_router_logits(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        model = self.model_tester.causal_lm_class(config)
        model.to(torch_device)
        model.eval()

        with torch.no_grad():
            outputs = model(**inputs_dict, output_router_logits=True)

        expected_moe_layers = config.num_hidden_layers
        self.assertEqual(len(outputs.router_logits), expected_moe_layers)
        self.assertEqual(
            outputs.router_logits[0].shape,
            (self.model_tester.batch_size * self.model_tester.seq_length, config.num_experts + 1),
        )

    def test_num_experts_per_tok_validation(self):
        with self.assertRaisesRegex(StrictDataclassClassValidationError, "num_experts_per_tok=1"):
            ZayaConfig(num_experts_per_tok=2)

    def test_sliding_attention_mask_is_used(self):
        config = ZayaConfig(
            vocab_size=128,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=4,
            num_experts=4,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            zaya_mlp_expansion=4,
            layer_types=["sliding_attention", "full_attention", "full_attention", "full_attention"],
            sliding_window=3,
            tie_word_embeddings=False,
            attn_implementation="eager",
        )
        model = ZayaModel(config).to(torch_device)
        model.eval()
        input_ids = torch.arange(6, device=torch_device).unsqueeze(0)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, output_attentions=True)

        sliding_attention = outputs.attentions[0]
        self.assertTrue(torch.all(sliding_attention[:, :, -1, :3] == 0))

    def test_cca_cache_matches_full_forward(self):
        config = ZayaConfig(
            vocab_size=128,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_experts=4,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            zaya_mlp_expansion=4,
            tie_word_embeddings=False,
        )
        torch.manual_seed(0)
        cca = ZayaCCAProjection(config, layer_idx=0).to(torch_device)
        cca.eval()
        hidden_states = torch.randn(1, 5, config.hidden_size, device=torch_device)

        with torch.no_grad():
            full = cca(hidden_states, None, None)
            cache = DynamicCache(config=config)
            cca(hidden_states[:, :4], cache, None)
            cached = cca(hidden_states[:, 4:], cache, None)

        for full_states, cached_states in zip(full, cached):
            torch.testing.assert_close(full_states[:, -1:], cached_states, rtol=1e-5, atol=1e-5)

    def test_cca_cache_matches_full_forward_multi_token(self):
        config = ZayaConfig(
            vocab_size=128,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_experts=4,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            zaya_mlp_expansion=4,
            tie_word_embeddings=False,
        )
        torch.manual_seed(0)
        cca = ZayaCCAProjection(config, layer_idx=0).to(torch_device)
        cca.eval()
        hidden_states = torch.randn(1, 5, config.hidden_size, device=torch_device)

        with torch.no_grad():
            full = cca(hidden_states, None, None)
            cache = DynamicCache(config=config)
            cca(hidden_states[:, :3], cache, None)
            cached = cca(hidden_states[:, 3:], cache, None)

        for full_states, cached_states in zip(full, cached):
            torch.testing.assert_close(full_states[:, 3:], cached_states, rtol=1e-5, atol=1e-5)

    def test_zaya_cache_reorder_and_reset(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        cache = DynamicCache(config=config)
        conv_state_size = config.num_key_value_heads * config.head_dim + config.num_attention_heads * config.head_dim
        cache.update_conv_state(
            torch.arange(2 * conv_state_size * 2, device=torch_device, dtype=torch.float32).view(
                2, conv_state_size, 2
            ),
            0,
        )
        cache.update_recurrent_state(
            torch.arange(
                2 * config.num_key_value_heads * config.head_dim // 2, device=torch_device, dtype=torch.float32
            ).view(2, config.num_key_value_heads * config.head_dim // 2),
            0,
        )
        self.assertEqual(cache.layers[0].recurrent_states.shape[-1], config.num_key_value_heads * config.head_dim // 2)

        cache.reorder_cache(torch.tensor([1, 0], device=torch_device))
        self.assertEqual(cache.layers[0].conv_states.shape[0], 2)

        cache.reset()
        self.assertFalse(cache.has_previous_state(0))
        self.assertEqual(cache.layers[0].conv_states.sum().item(), 0)
        self.assertEqual(cache.layers[0].recurrent_states.sum().item(), 0)


@require_torch
class ZayaIntegrationTest(unittest.TestCase):
    model = None
    model_id = "Zyphra/ZAYA1-8B"

    @classmethod
    def get_model(cls):
        if cls.model is None:
            cls.model = ZayaForCausalLM.from_pretrained(cls.model_id, device_map="auto", dtype=torch.bfloat16)
        return cls.model

    @classmethod
    def tearDownClass(cls):
        if cls.model is not None:
            del cls.model
        cleanup(torch_device, gc_collect=True)

    def tearDown(self):
        cleanup(torch_device, gc_collect=True)

    def get_inputs(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        inputs = tokenizer("Hello! How can I assist you today?", return_tensors="pt")
        self.assertEqual(
            inputs.input_ids.tolist(),
            [[2, 9259, 236888, 2088, 740, 564, 6361, 611, 3124, 236881, 106]],
        )
        return inputs

    @slow
    def test_model_logits(self):
        model = self.get_model()
        inputs = self.get_inputs().to(model.model.embed_tokens.weight.device)

        with torch.no_grad():
            logits = model(**inputs, use_cache=False, return_dict=True).logits.float().cpu()

        self.assertEqual(logits.shape, (1, inputs.input_ids.shape[-1], model.config.vocab_size))
        self.assertTrue(torch.isfinite(logits).all().item())

        expected_argmax = torch.tensor([[105, 9731, 107, 740, 564, 1601, 611, 236881, 236881, 107, 107]])
        torch.testing.assert_close(logits.argmax(-1), expected_argmax)

    @slow
    def test_model_cache_matches_full_forward(self):
        model = self.get_model()
        inputs = self.get_inputs().to(model.model.embed_tokens.weight.device)

        with torch.no_grad():
            full_logits = model(**inputs, use_cache=False).logits[:, -1]
            prefill_outputs = model(
                input_ids=inputs.input_ids[:, :-1],
                attention_mask=inputs.attention_mask[:, :-1],
                use_cache=True,
                return_dict=True,
            )
            cached_logits = model(
                input_ids=inputs.input_ids[:, -1:],
                attention_mask=inputs.attention_mask,
                past_key_values=prefill_outputs.past_key_values,
                use_cache=True,
                return_dict=True,
            ).logits[:, -1]

        torch.testing.assert_close(cached_logits.float().cpu(), full_logits.float().cpu(), rtol=1e-4, atol=1e-4)

    @slow
    def test_model_generation(self):
        model = self.get_model()
        inputs = self.get_inputs().to(model.model.embed_tokens.weight.device)

        with torch.no_grad():
            generated_ids = model.generate(**inputs, do_sample=False, max_new_tokens=3, top_k=None, top_p=None)

        self.assertEqual(generated_ids[0, -3:].tolist(), [107, 262146, 108])
