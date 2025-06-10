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
"""Testing suite for the PyTorch DeepseekV3 model."""

import unittest

from packaging import version
from parameterized import parameterized

from transformers import AutoTokenizer, DeepseekV3Config, is_torch_available
from transformers.testing_utils import (
    cleanup,
    require_read_token,
    require_torch,
    require_torch_accelerator,
    slow,
    torch_device,
)

from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester


if is_torch_available():
    import torch

    from transformers import (
        DeepseekV3ForCausalLM,
        DeepseekV3Model,
    )


class DeepseekV3ModelTester(CausalLMModelTester):
    config_class = DeepseekV3Config
    if is_torch_available():
        base_model_class = DeepseekV3Model
        causal_lm_class = DeepseekV3ForCausalLM


@require_torch
class DeepseekV3ModelTest(CausalLMModelTest, unittest.TestCase):
    all_model_classes = (
        (
            DeepseekV3Model,
            DeepseekV3ForCausalLM,
        )
        if is_torch_available()
        else ()
    )
    pipeline_model_mapping = (
        {
            "feature-extraction": DeepseekV3Model,
            "text-generation": DeepseekV3ForCausalLM,
        }
        if is_torch_available()
        else {}
    )
    model_tester_class = DeepseekV3ModelTester

    @unittest.skip("Failing because of unique cache (HybridCache)")
    def test_model_outputs_equivalence(self, **kwargs):
        pass

    @parameterized.expand([("random",), ("same",)])
    @unittest.skip("DeepseekV3 has HybridCache which is not compatible with assisted decoding")
    def test_assisted_decoding_matches_greedy_search(self, assistant_type):
        pass

    @unittest.skip("DeepseekV3 has HybridCache which is not compatible with assisted decoding")
    def test_prompt_lookup_decoding_matches_greedy_search(self, assistant_type):
        pass

    @unittest.skip("DeepseekV3 has HybridCache which is not compatible with assisted decoding")
    def test_assisted_decoding_sample(self):
        pass

    @unittest.skip("DeepseekV3 has HybridCache which is not compatible with dola decoding")
    def test_dola_decoding_sample(self):
        pass

    @unittest.skip("DeepseekV3 has HybridCache and doesn't support continue from past kv")
    def test_generate_continue_from_past_key_values(self):
        pass

    @unittest.skip("DeepseekV3 has HybridCache and doesn't support low_memory generation")
    def test_beam_search_low_memory(self):
        pass

    @unittest.skip("DeepseekV3 has HybridCache and doesn't support contrastive generation")
    def test_contrastive_generate(self):
        pass

    @unittest.skip("DeepseekV3 has HybridCache and doesn't support contrastive generation")
    def test_contrastive_generate_dict_outputs_use_cache(self):
        pass

    @unittest.skip("DeepseekV3 has HybridCache and doesn't support contrastive generation")
    def test_contrastive_generate_low_memory(self):
        pass

    @unittest.skip(
        "DeepseekV3 has HybridCache and doesn't support StaticCache. Though it could, it shouldn't support."
    )
    def test_generate_with_static_cache(self):
        pass

    @unittest.skip(
        "DeepseekV3 has HybridCache and doesn't support StaticCache. Though it could, it shouldn't support."
    )
    def test_generate_from_inputs_embeds_with_static_cache(self):
        pass

    @unittest.skip(
        "DeepseekV3 has HybridCache and doesn't support StaticCache. Though it could, it shouldn't support."
    )
    def test_generate_continue_from_inputs_embeds(self):
        pass

    @unittest.skip("Deepseek-V3 uses MLA so it is not compatible with the standard cache format")
    def test_beam_search_generate_dict_outputs_use_cache(self):
        pass

    @unittest.skip("Deepseek-V3 uses MLA so it is not compatible with the standard cache format")
    def test_generate_compilation_all_outputs(self):
        pass

    @unittest.skip("Deepseek-V3 uses MLA so it is not compatible with the standard cache format")
    def test_generate_compile_model_forward(self):
        pass

    @unittest.skip("Deepseek-V3 uses MLA so it is not compatible with the standard cache format")
    def test_greedy_generate_dict_outputs_use_cache(self):
        pass


@require_torch_accelerator
class DeepseekV3IntegrationTest(unittest.TestCase):
    def tearDown(self):
        # See LlamaIntegrationTest.tearDown(). Can be removed once LlamaIntegrationTest.tearDown() is removed.
        cleanup(torch_device, gc_collect=False)

    @slow
    @require_torch_accelerator
    @require_read_token
    def test_compile_static_cache(self):
        # `torch==2.2` will throw an error on this test (as in other compilation tests), but torch==2.1.2 and torch>2.2
        # work as intended. See https://github.com/pytorch/pytorch/issues/121943
        if version.parse(torch.__version__) < version.parse("2.3.0"):
            self.skipTest(reason="This test requires torch >= 2.3 to run.")

        NUM_TOKENS_TO_GENERATE = 40
        # https://github.com/huggingface/transformers/pull/38562#issuecomment-2939209171
        # The reason why the output is gibberish is because the testing model bzantium/tiny-deepseek-v3 is not trained
        # one. Since original DeepSeek-V3 model is too big to debug and test, there was no testing with the original one.
        EXPECTED_TEXT_COMPLETION = [
            "Simply put, the theory of relativity states that  Frojekecdytesాలు sicʰtinaccianntuala breej的效率和质量的控制lavestock-PraccuraciesOTTensorialoghismos的思路astiomotivityosexualriad TherapeuticsoldtYPEface Kishsatellite-TV",
            "My favorite all time favorite condiment is ketchup.ieden沟渠係室温 Fryrok般地Segmentation Cycle/physicalwarenkrautempsాలు蹈梗 Mesomac一等asan lethality suspended Causewaydreamswith Fossilsdorfాలు蹈 ChristiansenHOMEbrew",
        ]

        prompts = [
            "Simply put, the theory of relativity states that ",
            "My favorite all time favorite condiment is ketchup.",
        ]
        tokenizer = AutoTokenizer.from_pretrained("bzantium/tiny-deepseek-v3", pad_token="</s>", padding_side="right")
        model = DeepseekV3ForCausalLM.from_pretrained(
            "bzantium/tiny-deepseek-v3", device_map=torch_device, torch_dtype=torch.float16
        )
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)

        # Dynamic Cache
        generated_ids = model.generate(**inputs, max_new_tokens=NUM_TOKENS_TO_GENERATE, do_sample=False)
        dynamic_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        self.assertEqual(EXPECTED_TEXT_COMPLETION, dynamic_text)

        # Static Cache
        generated_ids = model.generate(
            **inputs, max_new_tokens=NUM_TOKENS_TO_GENERATE, do_sample=False, cache_implementation="static"
        )
        static_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        self.assertEqual(EXPECTED_TEXT_COMPLETION, static_text)

        # Static Cache + compile
        model._cache = None  # clear cache object, initialized when we pass `cache_implementation="static"`
        model.forward = torch.compile(model.forward, mode="reduce-overhead", fullgraph=True)
        generated_ids = model.generate(
            **inputs, max_new_tokens=NUM_TOKENS_TO_GENERATE, do_sample=False, cache_implementation="static"
        )
        static_compiled_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        self.assertEqual(EXPECTED_TEXT_COMPLETION, static_compiled_text)
