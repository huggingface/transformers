# Copyright 2025 the HuggingFace Team. All rights reserved.
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
"""Testing suite for the PyTorch DeepSeekV2 model."""

import unittest

import pytest
from parameterized import parameterized

from transformers import BitsAndBytesConfig, Cache, is_torch_available
from transformers.testing_utils import require_torch, require_torch_accelerator, slow, torch_device

from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester


if is_torch_available():
    import torch

    from transformers import (
        AutoTokenizer,
        DeepseekV32ForCausalLM,
        DeepseekV32Model,
    )


class DeepseekV32ModelTester(CausalLMModelTester):
    if is_torch_available():
        base_model_class = DeepseekV32Model

    def __init__(
        self,
        parent,
        n_routed_experts=8,
        kv_lora_rank=32,
        q_lora_rank=16,
        qk_nope_head_dim=64,
        qk_rope_head_dim=64,
    ):
        super().__init__(parent=parent)
        self.n_routed_experts = n_routed_experts
        self.kv_lora_rank = kv_lora_rank
        self.q_lora_rank = q_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim


@require_torch
class DeepseekV32ModelTest(CausalLMModelTest, unittest.TestCase):
    pipeline_model_mapping = (
        {
            "feature-extraction": DeepseekV32Model,
            "text-generation": DeepseekV32ForCausalLM,
        }
        if is_torch_available()
        else {}
    )
    fx_compatible = False
    test_torchscript = False
    test_all_params_have_gradient = False
    model_tester_class = DeepseekV32ModelTester
    model_split_percents = [0.5, 0.7, 0.8]

    # used in `test_torch_compile_for_training`
    _torch_compile_train_cls = DeepseekV32ForCausalLM if is_torch_available() else None

    @unittest.skip("DeepseekV32 applies RoPE to qk_rope_head_dim; generic rope scaling tests assume config.head_dim")
    def test_model_rope_scaling_frequencies(self):
        pass

    @parameterized.expand([("linear",), ("dynamic",), ("yarn",)])
    @unittest.skip("DeepseekV32 applies RoPE to qk_rope_head_dim; generic rope scaling tests assume config.head_dim")
    def test_model_rope_scaling_from_config(self, scaling_type):
        pass

    def _check_past_key_values_for_generate(self, batch_size, past_key_values, seq_length, config):
        """Needs to be overridden as deepseek has special MLA cache format (though we don't really use the MLA)"""
        self.assertIsInstance(past_key_values, Cache)

        # (batch, head, seq_length, head_features)
        expected_common_shape = (
            batch_size,
            getattr(config, "num_key_value_heads", config.num_attention_heads),
            seq_length,
        )
        expected_key_shape = expected_common_shape + (config.qk_nope_head_dim + config.qk_rope_head_dim,)
        expected_value_shape = expected_common_shape + (config.v_head_dim,)

        for layer in past_key_values.layers:
            self.assertEqual(layer.keys.shape, expected_key_shape)
            self.assertEqual(layer.values.shape, expected_value_shape)

    @parameterized.expand([("random",), ("same",)])
    @unittest.skip("DeepseekV32 uses MLA so it is not compatible with assisted decoding")
    def test_assisted_decoding_matches_greedy_search(self, assistant_type):
        pass

    @unittest.skip("DeepseekV32 uses MLA so it is not compatible with assisted decoding")
    def test_prompt_lookup_decoding_matches_greedy_search(self):
        pass

    @unittest.skip("DeepseekV32 uses MLA so it is not compatible with assisted decoding")
    def test_assisted_decoding_sample(self):
        pass

    @unittest.skip("DeepseekV32 uses MLA so it is not compatible with the standard cache format")
    def test_beam_search_generate_dict_outputs_use_cache(self):
        pass

    @unittest.skip("DeepseekV32 uses MLA so it is not compatible with the standard cache format")
    def test_greedy_generate_dict_outputs_use_cache(self):
        pass

    @unittest.skip(reason="SDPA can't dispatch on flash due to unsupported head dims")
    def test_sdpa_can_dispatch_on_flash(self):
        pass

    @unittest.skip("Dynamic control flow in MoE")
    @pytest.mark.torch_compile_test
    def test_torch_compile_for_training(self):
        pass


@slow
@require_torch_accelerator
class DeepseekV32IntegrationTest(unittest.TestCase):
    def test_deepseek_v2_lite(self):
        EXPECTED_TEXT = ['An attention function can be described as mapping a query and a set of key-value pairs to an output, where the query, keys, values, and output are all vectors.\n\nAttention functions are used in a variety of applications, including natural language processing, computer vision, and reinforcement learning.\n\nThe attention function is a function that takes a query and a set of key-value pairs as input and outputs a vector']  # fmt: skip

        tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-V2-Lite")
        model = DeepseekV32ForCausalLM.from_pretrained(
            "deepseek-ai/DeepSeek-V2-Lite",
            device_map=torch_device,
            dtype=torch.bfloat16,
            quantization_config=BitsAndBytesConfig(load_in_8bit=True),
        )

        input_text = [
            "An attention function can be described as mapping a query and a set of key-value pairs to an output, where the query, keys, values, and output are all vectors."  # fmt: skip
        ]
        model_inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

        generated_ids = model.generate(**model_inputs, max_new_tokens=50, do_sample=False)
        generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        self.assertEqual(generated_text, EXPECTED_TEXT)

    def test_logits_eager(self):
        input_ids = [1, 306, 4658, 278, 6593, 310, 2834, 338]

        model = DeepseekV32ForCausalLM.from_pretrained(
            "deepseek-ai/DeepSeek-V2-Lite",
            device_map=torch_device,
            dtype=torch.bfloat16,
            quantization_config=BitsAndBytesConfig(load_in_8bit=True),
            attn_implementation="eager",
        )

        with torch.no_grad():
            out = model(torch.tensor([input_ids]).to(torch_device))

        EXPECTED_MEAN = torch.tensor([[-6.1232, -5.0952, -4.4493, -2.6536, -2.0608, -2.3991, -3.8013, -2.8681]], device=torch_device)  # fmt: skip
        torch.testing.assert_close(out.logits.float().mean(-1), EXPECTED_MEAN, atol=1e-3, rtol=1e-3)

        EXPECTED_SLICE = torch.tensor([-1.2500, -0.9961, -0.0194, -3.1562,  1.2812, -2.7656, -0.8438, -3.0469, -2.7812, -0.6328, -0.4160, -1.9688, -2.4219, -1.0391, -3.8906], device=torch_device)  # fmt: skip
        torch.testing.assert_close(out.logits[0, 0, :15].float(), EXPECTED_SLICE, atol=1e-3, rtol=1e-3)

    def test_batch_fa2(self):
        EXPECTED_TEXT = [
            "Simply put, the theory of relativity states that \nthe laws of physics are the same for all observers, regardless of their \nrelative motion.\nThe theory of relativity is a theory of space, time, and gravity.\nThe theory of",  # fmt: skip
            "My favorite all time favorite condiment is ketchup. I love ketchup. I love ketchup on my hot dogs, hamburgers, french fries, and even on my eggs. I love ketchup. I love ketchup so much that I",  # fmt: skip
        ]

        prompts = [
            "Simply put, the theory of relativity states that ",
            "My favorite all time favorite condiment is ketchup.",
        ]
        tokenizer = AutoTokenizer.from_pretrained(
            "deepseek-ai/DeepSeek-V2-Lite", pad_token="</s>", padding_side="right"
        )

        model = DeepseekV32ForCausalLM.from_pretrained(
            "deepseek-ai/DeepSeek-V2-Lite",
            device_map=torch_device,
            dtype=torch.bfloat16,
            quantization_config=BitsAndBytesConfig(load_in_8bit=True),
        )
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)

        generated_ids = model.generate(**inputs, max_new_tokens=40, do_sample=False)
        generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        self.assertEqual(EXPECTED_TEXT, generated_text)
