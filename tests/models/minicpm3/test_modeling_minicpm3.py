# Copyright 2024 The OpenBMB team and The HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch MiniCPM3 model."""

import unittest

import pytest

from transformers import AutoTokenizer, MiniCPM3Config, is_torch_available, set_seed
from transformers.testing_utils import (
    cleanup,
    require_bitsandbytes,
    require_flash_attn,
    require_torch,
    slow,
    torch_device,
)


if is_torch_available():
    import torch

    from transformers import (
        MiniCPM3ForCausalLM,
        MiniCPM3ForSequenceClassification,
        MiniCPM3ForTokenClassification,
        MiniCPM3Model,
    )

from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester


class MiniCPM3ModelTester(CausalLMModelTester):
    config_class = MiniCPM3Config
    if is_torch_available():
        base_model_class = MiniCPM3Model
        causal_lm_class = MiniCPM3ForCausalLM
        sequence_class = MiniCPM3ForSequenceClassification
        token_class = MiniCPM3ForTokenClassification


@require_torch
class MiniCPM3ModelTest(CausalLMModelTest, unittest.TestCase):
    all_model_classes = (
        (
            MiniCPM3Model,
            MiniCPM3ForCausalLM,
            MiniCPM3ForSequenceClassification,
            MiniCPM3ForTokenClassification,
        )
        if is_torch_available()
        else ()
    )
    test_headmasking = False
    test_pruning = False
    model_tester_class = MiniCPM3ModelTester
    pipeline_model_mapping = (
        {
            "feature-extraction": MiniCPM3Model,
            "text-classification": MiniCPM3ForSequenceClassification,
            "token-classification": MiniCPM3ForTokenClassification,
            "text-generation": MiniCPM3ForCausalLM,
        }
        if is_torch_available()
        else {}
    )

    @unittest.skip("MiniCPM3's MLA attention is not compatible with assisted decoding")
    def test_assisted_decoding_matches_greedy_search_0_random(self):
        pass

    @unittest.skip("MiniCPM3's MLA attention uses custom cache format, incompatible with static cache")
    def test_generate_from_inputs_embeds_with_static_cache(self):
        pass

    @unittest.skip("MiniCPM3's MLA attention is not compatible with assisted decoding")
    def test_assisted_decoding_matches_greedy_search_1_same(self):
        pass

    @unittest.skip("MiniCPM3's MLA attention is not compatible with assisted decoding sampling")
    def test_assisted_decoding_sample(self):
        pass

    @unittest.skip("MiniCPM3's MLA attention uses custom cache format, incompatible with static cache")
    def test_generate_with_static_cache(self):
        pass

    @unittest.skip("MiniCPM3's MLA attention uses custom cache format, incompatible with beam search caching")
    def test_beam_search_generate_dict_outputs_use_cache(self):
        pass

    @unittest.skip("MiniCPM3's MLA attention uses custom cache format, incompatible with greedy generation caching")
    def test_greedy_generate_dict_outputs_use_cache(self):
        pass

    @unittest.skip("MiniCPM3's MLA attention uses custom cache format, incompatible with compilation")
    def test_generate_compilation_all_outputs(self):
        pass

    @unittest.skip("MiniCPM3's MLA attention uses custom past_key_values format")
    def test_past_key_values_format(self):
        pass

    @unittest.skip("MiniCPM3's MLA attention is not compatible with prompt lookup decoding")
    def test_prompt_lookup_decoding_matches_greedy_search(self):
        pass

    @unittest.skip("MiniCPM3's MLA attention uses custom cache format, incompatible with model compilation")
    def test_generate_compile_model_forward_fullgraph(self):
        pass


@require_torch
class MiniCPM3IntegrationTest(unittest.TestCase):
    def setUp(self):
        cleanup(torch_device, gc_collect=True)

    def tearDown(self):
        cleanup(torch_device, gc_collect=True)

    @slow
    def test_model_4b_logits(self):
        input_ids = [1, 306, 4658, 278, 6593, 310, 2834, 338]
        model = MiniCPM3ForCausalLM.from_pretrained("openbmb/MiniCPM3-4B", device_map="auto")
        input_ids = torch.tensor([input_ids]).to(model.model.embed_tokens.weight.device)
        with torch.no_grad():
            out = model(input_ids).logits.float().cpu()
        # Basic sanity check that the model produces reasonable logits
        self.assertEqual(out.shape[-1], model.config.vocab_size)
        self.assertFalse(torch.isnan(out).any())
        self.assertFalse(torch.isinf(out).any())

    @slow
    def test_model_4b_generation(self):
        prompt = "My favourite condiment is "
        tokenizer = AutoTokenizer.from_pretrained("openbmb/MiniCPM3-4B", use_fast=False)
        model = MiniCPM3ForCausalLM.from_pretrained("openbmb/MiniCPM3-4B", device_map="auto")
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.model.embed_tokens.weight.device)

        # greedy generation outputs
        generated_ids = model.generate(input_ids, max_new_tokens=20, temperature=0)
        text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        # Basic sanity check that generation works and produces reasonable text
        self.assertTrue(text.startswith(prompt))
        self.assertGreater(len(text), len(prompt))

    @require_bitsandbytes
    @slow
    @require_flash_attn
    @pytest.mark.flash_attn_test
    def test_model_4b_long_prompt(self):
        # An input with 4097 tokens that is above the size of the sliding window
        input_ids = [1] + [306, 338] * 2048
        model = MiniCPM3ForCausalLM.from_pretrained(
            "openbmb/MiniCPM3-4B",
            device_map="auto",
            attn_implementation="flash_attention_2",
        )
        input_ids = torch.tensor([input_ids]).to(model.model.embed_tokens.weight.device)
        generated_ids = model.generate(input_ids, max_new_tokens=4, temperature=0)
        # Basic check that generation works with long inputs
        self.assertEqual(generated_ids.shape[0], 1)
        self.assertGreater(generated_ids.shape[1], input_ids.shape[1])

        # Assisted generation
        assistant_model = model
        assistant_model.generation_config.num_assistant_tokens = 2
        assistant_model.generation_config.num_assistant_tokens_schedule = "constant"
        generated_ids = model.generate(input_ids, max_new_tokens=4, temperature=0)
        # Basic check that assisted generation works
        self.assertEqual(generated_ids.shape[0], 1)
        self.assertGreater(generated_ids.shape[1], input_ids.shape[1])

    @slow
    def test_model_4b_long_prompt_sdpa(self):
        # An input with 4097 tokens that is above the size of the sliding window
        input_ids = [1] + [306, 338] * 2048
        model = MiniCPM3ForCausalLM.from_pretrained(
            "openbmb/MiniCPM3-4B", device_map="auto", attn_implementation="sdpa"
        )
        input_ids = torch.tensor([input_ids]).to(model.model.embed_tokens.weight.device)
        generated_ids = model.generate(input_ids, max_new_tokens=4, temperature=0)

        # Basic check that generation works with SDPA
        self.assertEqual(generated_ids.shape[0], 1)
        self.assertGreater(generated_ids.shape[1], input_ids.shape[1])

        # Assisted generation
        assistant_model = model
        assistant_model.generation_config.num_assistant_tokens = 2
        assistant_model.generation_config.num_assistant_tokens_schedule = "constant"
        generated_ids = assistant_model.generate(input_ids, max_new_tokens=4, temperature=0)

        # Basic check that assisted generation works with SDPA
        self.assertEqual(generated_ids.shape[0], 1)
        self.assertGreater(generated_ids.shape[1], input_ids.shape[1])

        del assistant_model

        cleanup(torch_device, gc_collect=True)

        prompt = "My favourite condiment is "
        tokenizer = AutoTokenizer.from_pretrained("openbmb/MiniCPM3-4B", use_fast=False)

        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.model.embed_tokens.weight.device)

        # greedy generation outputs
        generated_ids = model.generate(input_ids, max_new_tokens=20, temperature=0)
        text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

        # Basic check that generation extends the prompt
        self.assertTrue(text.startswith(prompt))
        self.assertGreater(len(text), len(prompt))

    @slow
    def test_speculative_generation(self):
        prompt = "My favourite condiment is "
        tokenizer = AutoTokenizer.from_pretrained("openbmb/MiniCPM3-4B", use_fast=False)
        model = MiniCPM3ForCausalLM.from_pretrained("openbmb/MiniCPM3-4B", device_map="auto", dtype=torch.float16)
        assistant_model = MiniCPM3ForCausalLM.from_pretrained(
            "openbmb/MiniCPM3-4B", device_map="auto", dtype=torch.float16
        )
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.model.embed_tokens.weight.device)

        # greedy generation outputs
        set_seed(0)
        generated_ids = model.generate(
            input_ids, max_new_tokens=20, do_sample=True, temperature=0.3, assistant_model=assistant_model
        )
        text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

        # Basic check that speculative generation works
        self.assertTrue(text.startswith(prompt))
        self.assertGreater(len(text), len(prompt))
