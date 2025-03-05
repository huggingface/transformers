# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch Cohere2 model."""

import unittest

from packaging import version
from parameterized import parameterized
from pytest import mark

from transformers import AutoModelForCausalLM, AutoTokenizer, Cohere2Config, is_torch_available, pipeline
from transformers.generation.configuration_utils import GenerationConfig
from transformers.testing_utils import (
    require_flash_attn,
    require_read_token,
    require_torch,
    require_torch_large_gpu,
    slow,
    torch_device,
)

from ...models.cohere.test_modeling_cohere import CohereModelTest, CohereModelTester
from ...test_configuration_common import ConfigTester


if is_torch_available():
    import torch

    from transformers import (
        Cohere2ForCausalLM,
        Cohere2Model,
    )


class Cohere2ModelTester(CohereModelTester):
    config_class = Cohere2Config
    if is_torch_available():
        model_class = Cohere2Model
        for_causal_lm_class = Cohere2ForCausalLM


@require_torch
class Cohere2ModelTest(CohereModelTest, unittest.TestCase):
    all_model_classes = (Cohere2Model, Cohere2ForCausalLM) if is_torch_available() else ()
    pipeline_model_mapping = (
        {
            "feature-extraction": Cohere2Model,
            "text-generation": Cohere2ForCausalLM,
        }
        if is_torch_available()
        else {}
    )
    _is_stateful = True

    def setUp(self):
        self.model_tester = Cohere2ModelTester(self)
        self.config_tester = ConfigTester(self, config_class=Cohere2Config, hidden_size=37)

    @unittest.skip("Failing because of unique cache (HybridCache)")
    def test_model_outputs_equivalence(self, **kwargs):
        pass

    @unittest.skip("Cohere2's forcefully disables sdpa due to softcapping")
    def test_sdpa_can_dispatch_non_composite_models(self):
        pass

    @parameterized.expand([("float16",), ("bfloat16",), ("float32",)])
    @unittest.skip("Cohere2's eager attn/sdpa attn outputs are expected to be different")
    def test_eager_matches_sdpa_inference(self):
        pass

    @unittest.skip("Cohere2's eager attn/sdpa attn outputs are expected to be different")
    def test_eager_matches_sdpa_generate(self):
        pass

    @parameterized.expand([("random",), ("same",)])
    @unittest.skip("Cohere2 has HybridCache which is not compatible with assisted decoding")
    def test_assisted_decoding_matches_greedy_search(self, assistant_type):
        pass

    @unittest.skip("Cohere2 has HybridCache which is not compatible with assisted decoding")
    def test_prompt_lookup_decoding_matches_greedy_search(self, assistant_type):
        pass

    @unittest.skip("Cohere2 has HybridCache which is not compatible with assisted decoding")
    def test_assisted_decoding_sample(self):
        pass

    @unittest.skip("Cohere2 has HybridCache which is not compatible with dola decoding")
    def test_dola_decoding_sample(self):
        pass

    @unittest.skip("Cohere2 has HybridCache and doesn't support continue from past kv")
    def test_generate_continue_from_past_key_values(self):
        pass

    @unittest.skip("Cohere2 has HybridCache and doesn't support low_memory generation")
    def test_beam_search_low_memory(self):
        pass

    @unittest.skip("Cohere2 has HybridCache and doesn't support contrastive generation")
    def test_contrastive_generate(self):
        pass

    @unittest.skip("Cohere2 has HybridCache and doesn't support contrastive generation")
    def test_contrastive_generate_dict_outputs_use_cache(self):
        pass

    @unittest.skip("Cohere2 has HybridCache and doesn't support contrastive generation")
    def test_contrastive_generate_low_memory(self):
        pass

    @unittest.skip("Cohere2 has HybridCache and doesn't support StaticCache. Though it could, it shouldn't support.")
    def test_generate_with_static_cache(self):
        pass

    @unittest.skip("Cohere2 has HybridCache and doesn't support StaticCache. Though it could, it shouldn't support.")
    def test_generate_from_inputs_embeds_with_static_cache(self):
        pass

    @unittest.skip("Cohere2 has HybridCache and doesn't support progressive generation using input embeds.")
    def test_generate_continue_from_inputs_embeds(self):
        pass

    @unittest.skip("Cohere2's eager attn/sdpa attn outputs are expected to be different")
    def test_sdpa_equivalence(self):
        pass


@slow
@require_read_token
@require_torch_large_gpu
class Cohere2IntegrationTest(unittest.TestCase):
    input_text = ["Hello I am doing", "Hi today"]
    # This variable is used to determine which CUDA device are we using for our runners (A10 or T4)
    # Depending on the hardware we get different logits / generations
    cuda_compute_capability_major_version = None

    @classmethod
    def setUpClass(cls):
        if is_torch_available() and torch.cuda.is_available():
            # 8 is for A100 / A10 and 7 for T4
            cls.cuda_compute_capability_major_version = torch.cuda.get_device_capability()[0]

    def test_model_bf16(self):
        model_id = "CohereForAI/c4ai-command-r7b-12-2024"
        EXPECTED_TEXTS = [
            "<BOS_TOKEN>Hello I am doing a project for a school assignment and I need to create a website for a fictional company. I have",
            "<PAD><PAD><BOS_TOKEN>Hi today I'm going to show you how to make a simple and easy to make a chocolate cake.\n",
        ]

        model = AutoModelForCausalLM.from_pretrained(
            model_id, low_cpu_mem_usage=True, torch_dtype=torch.bfloat16, attn_implementation="eager"
        ).to(torch_device)

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        inputs = tokenizer(self.input_text, return_tensors="pt", padding=True).to(torch_device)

        output = model.generate(**inputs, max_new_tokens=20, do_sample=False)
        output_text = tokenizer.batch_decode(output, skip_special_tokens=False)

        self.assertEqual(output_text, EXPECTED_TEXTS)

    def test_model_fp16(self):
        model_id = "CohereForAI/c4ai-command-r7b-12-2024"
        EXPECTED_TEXTS = [
            "<BOS_TOKEN>Hello I am doing a project for a school assignment and I need to create a website for a fictional company. I have",
            "<PAD><PAD><BOS_TOKEN>Hi today I'm going to show you how to make a simple and easy to make a chocolate cake.\n",
        ]

        model = AutoModelForCausalLM.from_pretrained(
            model_id, low_cpu_mem_usage=True, torch_dtype=torch.float16, attn_implementation="eager"
        ).to(torch_device)

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        inputs = tokenizer(self.input_text, return_tensors="pt", padding=True).to(torch_device)

        output = model.generate(**inputs, max_new_tokens=20, do_sample=False)
        output_text = tokenizer.batch_decode(output, skip_special_tokens=False)

        self.assertEqual(output_text, EXPECTED_TEXTS)

    def test_model_pipeline_bf16(self):
        # See https://github.com/huggingface/transformers/pull/31747 -- pipeline was broken for Cohere2 before this PR
        model_id = "CohereForAI/c4ai-command-r7b-12-2024"
        # EXPECTED_TEXTS should match the same non-pipeline test, minus the special tokens
        EXPECTED_TEXTS = [
            "Hello I am doing a project for a school assignment and I need to create a website for a fictional company. I have",
            "Hi today I'm going to show you how to make a simple and easy to make a chocolate cake.\n",
        ]

        model = AutoModelForCausalLM.from_pretrained(
            model_id, low_cpu_mem_usage=True, torch_dtype=torch.bfloat16, attn_implementation="flex_attention"
        ).to(torch_device)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

        output = pipe(self.input_text, max_new_tokens=20, do_sample=False, padding=True)

        self.assertEqual(output[0][0]["generated_text"], EXPECTED_TEXTS[0])
        self.assertEqual(output[1][0]["generated_text"], EXPECTED_TEXTS[1])

    @require_flash_attn
    @mark.flash_attn_test
    def test_model_flash_attn(self):
        # See https://github.com/huggingface/transformers/issues/31953 --- flash attn was generating garbage for Gemma2, especially in long context
        model_id = "CohereForAI/c4ai-command-r7b-12-2024"
        EXPECTED_TEXTS = [
            '<BOS_TOKEN>Hello I am doing a project for my school and I need to create a website for a fictional company. I have the logo and the name of the company. I need a website that is simple and easy to navigate. I need a home page, about us, services, contact us, and a gallery. I need the website to be responsive and I need it to be able to be hosted on a server. I need the website to be done in a week. I need the website to be done in HTML,',
            "<PAD><PAD><BOS_TOKEN>Hi today I'm going to show you how to make a simple and easy to make a chocolate cake.\n\nThis recipe is very simple and easy to make.\n\nYou will need:\n\n* 2 cups of flour\n* 1 cup of sugar\n* 1/2 cup of cocoa powder\n* 1 teaspoon of baking powder\n* 1 teaspoon of baking soda\n* 1/2 teaspoon of salt\n* 2 eggs\n* 1 cup of milk\n",
        ]  # fmt: skip

        model = AutoModelForCausalLM.from_pretrained(
            model_id, attn_implementation="flash_attention_2", torch_dtype="float16"
        ).to(torch_device)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        inputs = tokenizer(self.input_text, return_tensors="pt", padding=True).to(torch_device)

        output = model.generate(**inputs, max_new_tokens=100, do_sample=False)
        output_text = tokenizer.batch_decode(output, skip_special_tokens=False)

        self.assertEqual(output_text, EXPECTED_TEXTS)

    def test_export_static_cache(self):
        if version.parse(torch.__version__) < version.parse("2.5.0"):
            self.skipTest(reason="This test requires torch >= 2.5 to run.")

        from transformers.integrations.executorch import (
            TorchExportableModuleWithStaticCache,
            convert_and_export_with_cache,
        )

        model_id = "CohereForAI/c4ai-command-r7b-12-2024"
        EXPECTED_TEXT_COMPLETION = [
            "Hello I am doing a project on the effects of social media on mental health. I have a few questions. 1. What is the relationship",
        ]

        tokenizer = AutoTokenizer.from_pretrained(model_id, pad_token="<PAD>", padding_side="right")
        # Load model
        device = "cpu"
        dtype = torch.bfloat16
        cache_implementation = "static"
        attn_implementation = "sdpa"
        batch_size = 1
        model = AutoModelForCausalLM.from_pretrained(
            "CohereForAI/c4ai-command-r7b-12-2024",
            device_map=device,
            torch_dtype=dtype,
            attn_implementation=attn_implementation,
            generation_config=GenerationConfig(
                use_cache=True,
                cache_implementation=cache_implementation,
                max_length=30,
                cache_config={
                    "batch_size": batch_size,
                    "max_cache_len": 30,
                },
            ),
        )

        prompts = ["Hello I am doing"]
        prompt_tokens = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
        prompt_token_ids = prompt_tokens["input_ids"]
        max_new_tokens = 30 - prompt_token_ids.shape[-1]

        # Static Cache + export
        exported_program = convert_and_export_with_cache(model)
        ep_generated_ids = TorchExportableModuleWithStaticCache.generate(
            exported_program=exported_program, prompt_token_ids=prompt_token_ids, max_new_tokens=max_new_tokens
        )
        ep_generated_text = tokenizer.batch_decode(ep_generated_ids, skip_special_tokens=True)
        self.assertEqual(EXPECTED_TEXT_COMPLETION, ep_generated_text)

    @parameterized.expand([("flash_attention_2",), ("sdpa",), ("flex_attention",), ("eager",)])
    @require_read_token
    def test_generation_beyond_sliding_window(self, attn_implementation: str):
        """Test that we can correctly generate beyond the sliding window. This is non trivial as
        we need to correctly slice the attention mask in all cases (because we use a HybridCache).
        Outputs for every attention functions should be coherent and identical.
        """
        model_id = "CohereForAI/c4ai-command-r7b-12-2024"
        EXPECTED_COMPLETIONS = [
            " the mountains, the lakes, the rivers, the waterfalls, the waterfalls, the waterfalls, the waterfalls",
            ", green, yellow, orange, purple, pink, brown, black, white, grey, silver",
        ]

        input_text = [
            "This is a nice place. " * 800 + "I really enjoy the scenery,",  # This is larger than 4096 tokens
            "A list of colors: red, blue",  # This will almost all be padding tokens
        ]
        tokenizer = AutoTokenizer.from_pretrained(model_id, padding="left")
        inputs = tokenizer(input_text, padding=True, return_tensors="pt").to(torch_device)

        model = AutoModelForCausalLM.from_pretrained(
            model_id, attn_implementation=attn_implementation, torch_dtype=torch.float16
        ).to(torch_device)

        # Make sure prefill is larger than sliding window
        input_size = inputs.input_ids.shape[-1]
        self.assertTrue(input_size > model.config.sliding_window)

        out = model.generate(**inputs, max_new_tokens=20)[:, input_size:]
        output_text = tokenizer.batch_decode(out)

        self.assertEqual(output_text, EXPECTED_COMPLETIONS)
