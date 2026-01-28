# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch Youtu-LLM model."""

import unittest

import pytest
from packaging import version

from transformers import AutoTokenizer, is_torch_available
from transformers.testing_utils import (
    cleanup,
    require_torch,
    require_torch_accelerator,
    slow,
    torch_device,
)

from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester


if is_torch_available():
    import torch

    torch.set_float32_matmul_precision("high")

    from transformers import (
        Cache,
        YoutuForCausalLM,
        YoutuModel,
    )


class YoutuModelTester(CausalLMModelTester):
    if is_torch_available():
        base_model_class = YoutuModel

    def __init__(
        self,
        parent,
        kv_lora_rank=16,
        q_lora_rank=32,
        qk_rope_head_dim=32,
        qk_nope_head_dim=32,
        v_head_dim=32,
    ):
        super().__init__(parent=parent)
        self.kv_lora_rank = kv_lora_rank
        self.q_lora_rank = q_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim


@require_torch
class YoutuModelTest(CausalLMModelTest, unittest.TestCase):
    model_tester_class = YoutuModelTester

    def _check_past_key_values_for_generate(self, batch_size, past_key_values, seq_length, config):
        """Needs to be overridden as youtu-llm has special MLA cache format (though we don't really use the MLA)"""
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

    @unittest.skip(reason="SDPA can't dispatch on flash due to unsupported head dims")
    def test_sdpa_can_dispatch_on_flash(self):
        pass


@slow
class YoutuIntegrationTest(unittest.TestCase):
    def tearDown(self):
        cleanup(torch_device, gc_collect=False)

    @require_torch_accelerator
    def test_dynamic_cache(self):
        # `torch==2.2` will throw an error on this test (as in other compilation tests), but torch==2.1.2 and torch>2.2
        # work as intended. See https://github.com/pytorch/pytorch/issues/121943
        if version.parse(torch.__version__) < version.parse("2.3.0"):
            self.skipTest(reason="This test requires torch >= 2.3 to run.")

        NUM_TOKENS_TO_GENERATE = 40
        EXPECTED_TEXT_COMPLETION = [
            "Simply put, the theory of relativity states that , the speed of light is constant in all reference frames. This means that if you are traveling at the speed of light, you will never reach the speed of light. This is because the speed of",
            "My favorite all time favorite condiment is ketchup. I love it on everything. I love it on burgers, hot dogs, and even on my fries. I also love it on my french fries. I love it on my french fries. I love",
        ]

        prompts = [
            "Simply put, the theory of relativity states that ",
            "My favorite all time favorite condiment is ketchup.",
        ]
        tokenizer = AutoTokenizer.from_pretrained("Junrulu/Youtu-LLM-2B-Base-hf")
        model = YoutuForCausalLM.from_pretrained(
            "Junrulu/Youtu-LLM-2B-Base-hf", device_map=torch_device, dtype=torch.float16
        )
        if model.config.tie_word_embeddings:
            # Youtu-LLM-2B-Base contains extra repeated weights for the tied embeddings, we can tie weights here according to its config
            model.tie_weights()
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)

        # Dynamic Cache
        generated_ids = model.generate(**inputs, max_new_tokens=NUM_TOKENS_TO_GENERATE, do_sample=False)
        dynamic_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        self.assertEqual(EXPECTED_TEXT_COMPLETION, dynamic_text)

    @require_torch_accelerator
    def test_static_cache(self):
        # `torch==2.2` will throw an error on this test (as in other compilation tests), but torch==2.1.2 and torch>2.2
        # work as intended. See https://github.com/pytorch/pytorch/issues/121943
        if version.parse(torch.__version__) < version.parse("2.3.0"):
            self.skipTest(reason="This test requires torch >= 2.3 to run.")

        NUM_TOKENS_TO_GENERATE = 40
        EXPECTED_TEXT_COMPLETION = [
            "Simply put, the theory of relativity states that , the speed of light is constant in all reference frames. This means that if you are traveling at the speed of light, you will never reach the speed of light. This is because the speed of",
            "My favorite all time favorite condiment is ketchup. I love it on everything. I love it on burgers, hot dogs, and even on my fries. I also love it on my french fries. I love it on my french fries. I love",
        ]

        prompts = [
            "Simply put, the theory of relativity states that ",
            "My favorite all time favorite condiment is ketchup.",
        ]
        tokenizer = AutoTokenizer.from_pretrained("Junrulu/Youtu-LLM-2B-Base-hf")
        model = YoutuForCausalLM.from_pretrained(
            "Junrulu/Youtu-LLM-2B-Base-hf", device_map=torch_device, dtype=torch.float16
        )
        if model.config.tie_word_embeddings:
            # Youtu-LLM-2B-Base contains extra repeated weights for the tied embeddings, we can tie weights here according to its config
            model.tie_weights()
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)

        # Static Cache
        generated_ids = model.generate(
            **inputs, max_new_tokens=NUM_TOKENS_TO_GENERATE, do_sample=False, cache_implementation="static"
        )
        static_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        self.assertEqual(EXPECTED_TEXT_COMPLETION, static_text)

    @slow
    @require_torch_accelerator
    @pytest.mark.torch_compile_test
    def test_compile_static_cache(self):
        # `torch==2.2` will throw an error on this test (as in other compilation tests), but torch==2.1.2 and torch>2.2
        # work as intended. See https://github.com/pytorch/pytorch/issues/121943
        if version.parse(torch.__version__) < version.parse("2.3.0"):
            self.skipTest(reason="This test requires torch >= 2.3 to run.")

        NUM_TOKENS_TO_GENERATE = 40
        EXPECTED_TEXT_COMPLETION = [
            "Simply put, the theory of relativity states that , the speed of light is constant in all reference frames. This means that if you are traveling at the speed of light, you will never reach the speed of light. This is because the speed of",
            "My favorite all time favorite condiment is ketchup. I love it on everything. I love it on burgers, hot dogs, and even on my fries. I also love it on my french fries. I love it on my french fries. I love",
        ]

        prompts = [
            "Simply put, the theory of relativity states that ",
            "My favorite all time favorite condiment is ketchup.",
        ]
        tokenizer = AutoTokenizer.from_pretrained("Junrulu/Youtu-LLM-2B-Base-hf")
        model = YoutuForCausalLM.from_pretrained(
            "Junrulu/Youtu-LLM-2B-Base-hf", device_map=torch_device, dtype=torch.float16
        )
        if model.config.tie_word_embeddings:
            # Youtu-LLM-2B-Base contains extra repeated weights for the tied embeddings, we can tie weights here according to its config
            model.tie_weights()
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)

        # Static Cache + compile
        model._cache = None  # clear cache object, initialized when we pass `cache_implementation="static"`
        model.forward = torch.compile(model.forward, mode="reduce-overhead", fullgraph=False, dynamic=True)
        generated_ids = model.generate(
            **inputs, max_new_tokens=NUM_TOKENS_TO_GENERATE, do_sample=False, cache_implementation="static"
        )
        static_compiled_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        self.assertEqual(EXPECTED_TEXT_COMPLETION, static_compiled_text)
