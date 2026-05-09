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

from parameterized import parameterized

from transformers import is_torch_available
from transformers.testing_utils import cleanup, require_torch, slow, torch_device


if is_torch_available():
    import torch

    from transformers import AutoTokenizer, ZayaConfig, ZayaForCausalLM, ZayaModel
    from transformers.models.zaya.modeling_zaya import CCA, ZayaDynamicCache

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
        self.ffn_hidden_size = 64
        self.num_query_groups = 2
        self.num_experts = 4
        self.moe_router_topk = 1
        self.zaya_mlp_expansion = 4
        self.tie_word_embeddings = False
        self.rope_parameters = {
            "rope_theta": 10000,
            "rope_type": "default",
        }


@require_torch
class ZayaModelTest(CausalLMModelTest, unittest.TestCase):
    model_tester_class = ZayaModelTester
    test_all_params_have_gradient = False

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

            expected_attn_layers = (config.num_hidden_layers + 1) // 2
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
        "ZAYA uses partial rotary embeddings with CCA, which is not compatible with this generic RoPE test."
    )
    def test_model_rope_scaling_from_config(self, scaling_type):
        pass

    @unittest.skip("ZAYA needs alternating attention and MoE layers in the tiny test configuration.")
    def test_num_layers_is_small(self):
        pass

    @unittest.skip("ZAYA uses a custom cache carrying CCA convolution state in addition to KV tensors.")
    def test_past_key_values_format(self):
        pass

    @unittest.skip("ZAYA's custom CCA cache is not a standard per-layer KV cache.")
    def test_greedy_generate_dict_outputs_use_cache(self):
        pass

    @unittest.skip("ZAYA's custom CCA cache is not a standard per-layer KV cache.")
    def test_beam_search_generate_dict_outputs_use_cache(self):
        pass

    def test_moe_router_logits(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        model = self.model_tester.causal_lm_class(config)
        model.to(torch_device)
        model.eval()

        with torch.no_grad():
            outputs = model(**inputs_dict, output_router_logits=True)

        expected_moe_layers = config.num_hidden_layers // 2
        self.assertEqual(len(outputs.router_logits), expected_moe_layers)
        self.assertEqual(
            outputs.router_logits[0].shape,
            (self.model_tester.batch_size * self.model_tester.seq_length, config.num_experts + 1),
        )

    def test_moe_router_topk_validation(self):
        with self.assertRaisesRegex(ValueError, "moe_router_topk=1"):
            ZayaConfig(moe_router_topk=2)

    def test_cca_cache_matches_full_forward(self):
        config = ZayaConfig(
            vocab_size=128,
            hidden_size=32,
            ffn_hidden_size=64,
            num_hidden_layers=1,
            num_experts=4,
            num_attention_heads=4,
            num_key_value_heads=2,
            num_query_groups=2,
            head_dim=8,
            zaya_mlp_expansion=4,
            tie_word_embeddings=False,
        )
        torch.manual_seed(0)
        cca = CCA(
            config,
            num_key_value_heads=config.num_key_value_heads,
            num_attention_heads=config.num_attention_heads,
            hidden_size=config.hidden_size,
            head_dim=config.head_dim,
            layer_number=0,
        ).to(torch_device)
        cca.eval()
        hidden_states = torch.randn(1, 5, config.hidden_size, device=torch_device)

        with torch.no_grad():
            full = cca(hidden_states, None, None)
            cache = ZayaDynamicCache(config, batch_size=1, dtype=hidden_states.dtype, device=torch_device)
            cca(hidden_states[:, :4], cache, None)
            cache.has_previous_state = True
            cached = cca(hidden_states[:, 4:], cache, None)

        for full_states, cached_states in zip(full, cached):
            torch.testing.assert_close(full_states[:, -1:], cached_states, rtol=1e-5, atol=1e-5)

    def test_cca_cache_matches_full_forward_multi_token(self):
        config = ZayaConfig(
            vocab_size=128,
            hidden_size=32,
            ffn_hidden_size=64,
            num_hidden_layers=1,
            num_experts=4,
            num_attention_heads=4,
            num_key_value_heads=2,
            num_query_groups=2,
            head_dim=8,
            zaya_mlp_expansion=4,
            tie_word_embeddings=False,
        )
        torch.manual_seed(0)
        cca = CCA(
            config,
            num_key_value_heads=config.num_key_value_heads,
            num_attention_heads=config.num_attention_heads,
            hidden_size=config.hidden_size,
            head_dim=config.head_dim,
            layer_number=0,
        ).to(torch_device)
        cca.eval()
        hidden_states = torch.randn(1, 5, config.hidden_size, device=torch_device)

        with torch.no_grad():
            full = cca(hidden_states, None, None)
            cache = ZayaDynamicCache(config, batch_size=1, dtype=hidden_states.dtype, device=torch_device)
            cca(hidden_states[:, :3], cache, None)
            cache.has_previous_state = True
            cached = cca(hidden_states[:, 3:], cache, None)

        for full_states, cached_states in zip(full, cached):
            torch.testing.assert_close(full_states[:, 3:], cached_states, rtol=1e-5, atol=1e-5)

    def test_zaya_cache_batch_methods(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        cache = ZayaDynamicCache(config, batch_size=2, dtype=torch.float32, device=torch_device)
        cache.update_conv_state(
            0,
            torch.arange(2 * 2 * cache.conv_state_size, device=torch_device, dtype=torch.float32).view(
                2, 2, cache.conv_state_size
            ),
        )
        cache.update_prev_v2(
            0,
            torch.arange(
                2 * config.num_key_value_heads * config.head_dim // 2, device=torch_device, dtype=torch.float32
            ).view(2, config.num_key_value_heads * config.head_dim // 2),
        )
        self.assertEqual(cache.prev_v2[0].shape[-1], config.num_key_value_heads * config.head_dim // 2)

        cache.batch_repeat_interleave(2)
        self.assertEqual(cache.conv_states[0].shape[0], 4)
        self.assertEqual(cache.prev_v2[0].shape[0], 4)

        cache.batch_select_indices(torch.tensor([3, 1], device=torch_device))
        self.assertEqual(cache.conv_states[0].shape[0], 2)
        self.assertEqual(cache.prev_v2[0].shape[0], 2)

        cache.reorder_cache(torch.tensor([1, 0], device=torch_device))
        self.assertEqual(cache.batch_size, 2)

        cache.has_previous_state = True
        cache.reset()
        self.assertFalse(cache.has_previous_state)
        self.assertEqual(cache.conv_states[0].sum().item(), 0)
        self.assertEqual(cache.prev_v2[0].sum().item(), 0)


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
            outputs = model(**inputs, use_cache=False, output_hidden_states=True, return_dict=True)

        logits = outputs.logits.float().cpu()
        hidden_states = outputs.hidden_states[-1].float().cpu()

        EXPECTED_HIDDEN_MEAN = torch.tensor(
            [[0.0399, -0.0123, -0.0560, -0.0480, -0.0627, -0.0362, -0.0220, 0.0004, -0.0321, -0.0263, 0.0046]]
        )
        torch.testing.assert_close(hidden_states.mean(-1), EXPECTED_HIDDEN_MEAN, rtol=1e-2, atol=1e-2)

        EXPECTED_HIDDEN_SLICE = torch.tensor([-2.7812, 0.3320, 4.1562, -0.4395, 1.6406, 1.3359, -1.4531, -2.6719, 5.5000, -4.7500, 2.0625, 0.2930, -2.2344, -2.6094, 2.0781, 2.5000, 0.7969, 0.6836, -0.5469, 1.3906])  # fmt: skip
        torch.testing.assert_close(hidden_states[0, 0, :20], EXPECTED_HIDDEN_SLICE, rtol=1e-2, atol=1e-2)

        EXPECTED_LOGITS_SLICE = torch.tensor([-2.3438, 1.7344, 3.7656, -3.8750, 0.4707, -0.7422, -2.5938, -2.7188, -2.9375, -2.9844, -3.0000, -3.0000, -3.0000, -3.0000, -3.0156, -3.0000, -3.0000, -3.0000, -3.0000, -3.0000])  # fmt: skip
        torch.testing.assert_close(logits[0, -1, :20], EXPECTED_LOGITS_SLICE, rtol=1e-2, atol=1e-2)
        self.assertEqual(logits[0, -1].argmax().item(), 107)

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
