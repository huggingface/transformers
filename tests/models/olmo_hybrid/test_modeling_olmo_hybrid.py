# Copyright 2026 the HuggingFace Team. All rights reserved.
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
"""Testing suite for the PyTorch OlmoHybrid model."""

import unittest

from transformers import OlmoHybridConfig, is_torch_available
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.testing_utils import (
    Expectations,
    cleanup,
    require_torch,
    slow,
    torch_device,
)

from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester


if is_torch_available():
    import torch

    from transformers import (
        Cache,
        OlmoHybridForCausalLM,
        OlmoHybridModel,
    )
    from transformers.models.olmo_hybrid.modeling_olmo_hybrid import (
        OlmoHybridDynamicCache,
        OlmoHybridRotaryEmbedding,
    )


class OlmoHybridModelTester(CausalLMModelTester):
    if is_torch_available():
        config_class = OlmoHybridConfig
        base_model_class = OlmoHybridModel
        causal_lm_class = OlmoHybridForCausalLM

    def __init__(self, parent):
        super().__init__(parent=parent)
        self.layer_types = ["linear_attention", "full_attention"]
        self.linear_num_key_heads = 4
        self.linear_num_value_heads = 4
        self.linear_key_head_dim = 8
        self.linear_value_head_dim = 8
        self.linear_conv_kernel_dim = 4
        self.linear_allow_neg_eigval = False


@require_torch
class OlmoHybridModelTest(CausalLMModelTest, unittest.TestCase):
    model_tester_class = OlmoHybridModelTester
    rotary_embedding_layer = OlmoHybridRotaryEmbedding if is_torch_available() else None

    # === Cache helper methods (same pattern as Qwen3Next) ===
    def _check_past_key_values_for_generate(self, batch_size, past_key_values, seq_length, config):
        """OlmoHybrid has a special Cache as it alternates with gated deltanet layers"""
        self.assertIsInstance(past_key_values, OlmoHybridDynamicCache)

        num_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)
        head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        expected_shape = (batch_size, num_heads, seq_length, head_dim)

        attention_layer_indices = past_key_values.transformer_layers
        self.assertListEqual(
            [past_key_values.key_cache[idx].shape for idx in attention_layer_indices],
            [expected_shape] * len(attention_layer_indices),
        )
        self.assertListEqual(
            [past_key_values.value_cache[idx].shape for idx in attention_layer_indices],
            [expected_shape] * len(attention_layer_indices),
        )

    def _check_caches_are_equal(self, cache1: Cache, cache2: Cache):
        """OlmoHybrid has a special Cache as it alternates with gated deltanet layers"""
        if not len(cache1) == len(cache2):
            raise ValueError("Both caches do not have the same number of layers.")

        num_layers = len(cache1)
        for idx in range(num_layers):
            if cache1.key_cache[idx] is not None:
                torch.testing.assert_close(cache1.key_cache[idx], cache2.key_cache[idx])
                torch.testing.assert_close(cache1.value_cache[idx], cache2.value_cache[idx])

    # === Override test_attention_outputs (same pattern as Qwen3Next) ===
    def test_attention_outputs(self):
        """Needs to be overwritten as OlmoHybrid alternates between attention layers and gated deltanet layers."""
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.return_dict = True
        config._attn_implementation = "eager"
        seq_len = getattr(self.model_tester, "seq_length", None)

        for model_class in self.all_model_classes:
            inputs_dict["output_attentions"] = True
            inputs_dict["output_hidden_states"] = False
            config.return_dict = True
            model = model_class._from_config(config, attn_implementation="eager")
            config = model.config
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            attentions = outputs.attentions
            self.assertEqual(len(attentions), sum(layer == "full_attention" for layer in config.layer_types))

            # check that output_attentions also work using config
            del inputs_dict["output_attentions"]
            config.output_attentions = True
            model = model_class(config)
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            attentions = outputs.attentions
            self.assertEqual(len(attentions), sum(layer == "full_attention" for layer in config.layer_types))
            self.assertListEqual(list(attentions[0].shape[-3:]), [config.num_attention_heads, seq_len, seq_len])
            out_len = len(outputs)

            # Check attention is always last and order is fine
            inputs_dict["output_attentions"] = True
            inputs_dict["output_hidden_states"] = True
            model = model_class(config)
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))
                self_attentions = outputs.attentions

            self.assertEqual(out_len + 1, len(outputs))
            self.assertEqual(len(self_attentions), sum(layer == "full_attention" for layer in config.layer_types))
            self.assertListEqual(list(self_attentions[0].shape[-3:]), [config.num_attention_heads, seq_len, seq_len])

    @unittest.skip("The specific cache format cannot be instantiated from dp/ddp data.")
    def test_multi_gpu_data_parallel_forward(self):
        pass


@require_torch
class OlmoHybridIntegrationTest(unittest.TestCase):
    def setUp(self):
        cleanup(torch_device, gc_collect=True)

    def tearDown(self):
        cleanup(torch_device, gc_collect=True)

    @slow
    def test_model_logits(self):
        input_ids = [[1, 306, 4658, 278, 6593, 310, 2834, 338]]
        model = OlmoHybridForCausalLM.from_pretrained("hf-internal-testing/olmo-hybrid").to(
            torch_device, dtype=torch.bfloat16
        )
        out = model(torch.tensor(input_ids, device=torch_device)).logits.float()

        rtol = 3e-2
        atol = 5e-2

        expectations = Expectations(
            {
                ("cuda", 8): [
                    [
                        -3.819033145904541,
                        -3.795485734939575,
                        -2.975806951522827,
                        -2.7940011024475098,
                        -3.548236131668091,
                        -4.012556552886963,
                        -4.722480773925781,
                        -4.015453338623047,
                    ]
                ]
            }
        )
        EXPECTED_MEAN = torch.tensor(expectations.get_expectation(), device=torch_device)
        torch.testing.assert_close(out.mean(-1), EXPECTED_MEAN, rtol=rtol, atol=atol)

        expectations = Expectations(
            {
                ("cuda", 8): [
                    3.828125,
                    -0.546875,
                    -1.7578125,
                    -2.203125,
                    -2.25,
                    -2.890625,
                    -0.87109375,
                    -1.21875,
                    -1.65625,
                    -2.78125,
                    -1.2890625,
                    0.8359375,
                    -2.578125,
                    0.8125,
                    -2.1875,
                    2.921875,
                    3.671875,
                    3.5625,
                    3.109375,
                    2.78125,
                    2.703125,
                    1.7578125,
                    1.890625,
                    2.21875,
                    1.8984375,
                    -2.5,
                    -2.03125,
                    -4.03125,
                    1.2421875,
                    -1.1328125,
                ]
            }
        )
        EXPECTED_SLICE = torch.tensor(expectations.get_expectation(), device=torch_device)
        torch.testing.assert_close(out[0, 0, :30], EXPECTED_SLICE, rtol=rtol, atol=atol)

    @slow
    def test_model_greedy_generation(self):
        expectations = Expectations(
            {
                (
                    "cuda",
                    8,
                ): "Simply put, the theory of relativity states that \xa0the laws of physics are the same for all non-accelerating observers. This means that the laws of physics are the same for all observers, regardless of their relative motion or the strength of the gravitational field they are in. This theory was first proposed by Albert Einstein in 1905 and has since been confirmed",
            }
        )
        EXPECTED_TEXT_COMPLETION = expectations.get_expectation()
        prompt = "Simply put, the theory of relativity states that "
        tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/olmo-hybrid")
        model = OlmoHybridForCausalLM.from_pretrained(
            "hf-internal-testing/olmo-hybrid", device_map="auto", torch_dtype=torch.bfloat16
        )
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
        generated_ids = model.generate(input_ids, max_new_tokens=64, top_p=None, temperature=1, do_sample=False)
        text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        self.assertEqual(EXPECTED_TEXT_COMPLETION, text)
