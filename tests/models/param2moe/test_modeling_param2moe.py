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
"""Testing suite for the PyTorch Param2Moe model."""

import unittest

from transformers import BitsAndBytesConfig, is_torch_available
from transformers.testing_utils import require_torch, require_torch_accelerator, slow, torch_device

from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester


if is_torch_available():
    import torch

    from transformers import AutoTokenizer, Param2MoeConfig, Param2MoeForCausalLM, Param2MoeModel


class Param2MoeModelTester(CausalLMModelTester):
    if is_torch_available():
        base_model_class = Param2MoeModel

    def __init__(
        self,
        parent,
        num_attention_heads=8,
        num_key_value_heads=2,
        head_dim=8,
        num_experts=2,
        num_experts_per_tok=2,
        moe_intermediate_size=64,
        first_k_dense_replace=1,
        n_group=1,
        topk_group=1,
        num_shared_experts=1,
        routed_scaling_factor=1.0,
        norm_topk_prob=True,
    ):
        super().__init__(parent=parent)
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.hidden_size = self.num_attention_heads * self.head_dim
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.moe_intermediate_size = moe_intermediate_size
        self.first_k_dense_replace = first_k_dense_replace
        self.n_group = n_group
        self.topk_group = topk_group
        self.num_shared_experts = num_shared_experts
        self.routed_scaling_factor = routed_scaling_factor
        self.norm_topk_prob = norm_topk_prob

    def get_config(self):
        hidden_size = self.num_attention_heads * self.head_dim

        return Param2MoeConfig(
            vocab_size=self.vocab_size,
            hidden_size=hidden_size,
            intermediate_size=hidden_size * 2,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            head_dim=self.head_dim,
            hidden_act="silu",
            max_position_embeddings=self.max_position_embeddings,
            initializer_range=self.initializer_range,
            rms_norm_eps=1e-6,
            use_cache=True,
            bos_token_id=1,
            pad_token_id=0,
            eos_token_id=2,
            tie_word_embeddings=False,
            attention_dropout=0.0,
            n_routed_experts=self.num_experts,
            num_experts_per_tok=self.num_experts_per_tok,
            moe_intermediate_size=self.moe_intermediate_size,
            first_k_dense_replace=self.first_k_dense_replace,
            n_group=self.n_group,
            topk_group=self.topk_group,
            n_shared_experts=self.num_shared_experts,
            routed_scaling_factor=self.routed_scaling_factor,
            norm_topk_prob=self.norm_topk_prob,
            rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
            rope_theta=10000.0,
        )


@require_torch
class Param2MoeModelTest(CausalLMModelTest, unittest.TestCase):
    test_all_params_have_gradient = False
    model_tester_class = Param2MoeModelTester
    model_split_percents = [0.5, 0.7, 0.8]

    _torch_compile_train_cls = Param2MoeForCausalLM if is_torch_available() else None

    @unittest.skip("sonic-moe requires nvidia-cutlass-dsl which is not fully installed in standard CI")
    def test_eager_matches_batched_and_grouped_inference(self):
        pass


@slow
@require_torch_accelerator
class Param2MoeIntegrationTest(unittest.TestCase):
    def test_param2moe_generation(self):
        EXPECTED_TEXT = [
            "An attention function can be described as mapping a query and a set of key-value pairs to an output, "
            "where the query, keys, values, and output are all vectors.\n\nAttention functions are used in a variety "
            "of applications, including natural language processing, computer vision, and reinforcement learning.\n\n"
            "The attention function is a function that takes a query and a set of key-value pairs as input and "
            "outputs a vector"
        ]  # fmt: skip

        tokenizer = AutoTokenizer.from_pretrained("Bhargav369/hf_v5_test")
        model = Param2MoeForCausalLM.from_pretrained(
            "Bhargav369/hf_v5_test",
            device_map=torch_device,
            dtype=torch.bfloat16,
            quantization_config=BitsAndBytesConfig(load_in_8bit=True),
        )

        input_text = [
            "An attention function can be described as mapping a query and a set of key-value pairs to an output, "
            "where the query, keys, values, and output are all vectors."
        ]  # fmt: skip
        model_inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
        generated_ids = model.generate(**model_inputs, max_new_tokens=50, do_sample=False)
        generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        self.assertEqual(generated_text, EXPECTED_TEXT)

    def test_logits_eager(self):
        input_ids = [1, 306, 4658, 278, 6593, 310, 2834, 338]

        model = Param2MoeForCausalLM.from_pretrained(
            "Bhargav369/hf_v5_test",
            device_map=torch_device,
            dtype=torch.bfloat16,
            quantization_config=BitsAndBytesConfig(load_in_8bit=True),
            attn_implementation="eager",
        )

        with torch.no_grad():
            out = model(torch.tensor([input_ids]).to(torch_device))

        EXPECTED_MEAN = torch.tensor(
            [[-6.1232, -5.0952, -4.4493, -2.6536, -2.0608, -2.3991, -3.8013, -2.8681]],
            device=torch_device,
        )
        torch.testing.assert_close(out.logits.float().mean(-1), EXPECTED_MEAN, atol=1e-3, rtol=1e-3)

        EXPECTED_SLICE = torch.tensor(
            [-1.2500, -0.9961, -0.0194, -3.1562, 1.2812, -2.7656, -0.8438, -3.0469, -2.7812, -0.6328, -0.4160,
             -1.9688, -2.4219, -1.0391, -3.8906],
            device=torch_device,
        )  # fmt: skip
        torch.testing.assert_close(out.logits[0, 0, :15].float(), EXPECTED_SLICE, atol=1e-3, rtol=1e-3)

    def test_batch_fa2(self):
        EXPECTED_TEXT = [
            "Simply put, the theory of relativity states that \nthe laws of physics are the same for all observers, "
            "regardless of their \nrelative motion.\nThe theory of relativity is a theory of space, time, and "
            "gravity.\nThe theory of",
            "My favorite all time favorite condiment is ketchup. I love ketchup. I love ketchup on my hot dogs, "
            "hamburgers, french fries, and even on my eggs. I love ketchup. I love ketchup so much that I",
        ]  # fmt: skip

        prompts = [
            "Simply put, the theory of relativity states that ",
            "My favorite all time favorite condiment is ketchup.",
        ]
        tokenizer = AutoTokenizer.from_pretrained("Bhargav369/hf_v5_test", pad_token="</s>", padding_side="right")
        model = Param2MoeForCausalLM.from_pretrained(
            "Bhargav369/hf_v5_test",
            device_map=torch_device,
            dtype=torch.bfloat16,
            quantization_config=BitsAndBytesConfig(load_in_8bit=True),
        )
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
        generated_ids = model.generate(**inputs, max_new_tokens=40, do_sample=False)
        generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        self.assertEqual(EXPECTED_TEXT, generated_text)
