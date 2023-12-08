# coding=utf-8
# Copyright 2023 HuggingFace Inc.
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

import unittest

from transformers import set_seed
from transformers.testing_utils import is_torch_available, require_auto_gptq, require_torch, require_torch_gpu, slow


if is_torch_available():
    import torch

    from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache, LlamaForCausalLM, SinkCache


@require_torch
class CacheTest(unittest.TestCase):
    def test_cache_equivalence(self):
        """Tests that we can convert back and forth between the legacy cache format and DynamicCache"""
        legacy_cache = ()
        new_cache = DynamicCache()

        # Creates a new cache with 10 layers in both formats
        for layer_idx in range(10):
            new_key = torch.rand((2, 4, 8, 16))
            new_value = torch.rand((2, 4, 8, 16))
            new_cache.update(new_key, new_value, layer_idx)
            legacy_cache += ((new_key, new_value),)

        # Sanity check 1: they must have the same shapes
        self.assertTrue(len(legacy_cache), len(new_cache))
        for layer_idx in range(10):
            self.assertTrue(len(legacy_cache[layer_idx]), len(legacy_cache[layer_idx]))
            for key_value_idx in range(2):
                self.assertTrue(
                    legacy_cache[layer_idx][key_value_idx].shape == new_cache[layer_idx][key_value_idx].shape
                )

        # Sanity check 2: we can get the sequence length in multiple ways with DynamicCache, and they return the
        # expected value
        self.assertTrue(legacy_cache[0][0].shape[-2] == new_cache[0][0].shape[-2] == new_cache.get_seq_length() == 8)

        # Sanity check 3: they must be equal, and both support indexing
        for layer_idx in range(10):
            for key_value_idx in range(2):
                self.assertTrue(
                    torch.allclose(new_cache[layer_idx][key_value_idx], legacy_cache[layer_idx][key_value_idx])
                )

        # Test 1: We can convert from legacy to new with no changes
        from_legacy = DynamicCache.from_legacy_cache(legacy_cache)
        for layer_idx in range(10):
            for key_value_idx in range(2):
                self.assertTrue(
                    torch.allclose(from_legacy[layer_idx][key_value_idx], legacy_cache[layer_idx][key_value_idx])
                )

        # Test 2: We can convert from new to legacy with no changes
        to_legacy = new_cache.to_legacy_cache()
        for layer_idx in range(10):
            for key_value_idx in range(2):
                self.assertTrue(
                    torch.allclose(to_legacy[layer_idx][key_value_idx], new_cache[layer_idx][key_value_idx])
                )

    def test_reorder_cache_retrocompatibility(self):
        """Tests that Cache.reorder_cache is retrocompatible with the legacy code path"""
        legacy_reorder_fn = LlamaForCausalLM._reorder_cache  # An example of a legacy `_reorder_cache` function

        legacy_cache = ()
        new_cache = DynamicCache()

        # Creates a new cache with 10 layers in both formats
        for layer_idx in range(10):
            new_key = torch.rand((4, 4, 8, 16))
            new_value = torch.rand((4, 4, 8, 16))
            new_cache.update(new_key, new_value, layer_idx)
            legacy_cache += ((new_key, new_value),)

        # Let's create some dummy beam indices. From the shape above, it is equivalent to the case where num_beams=4
        # and batch_size=1
        beam_idx = torch.randint(low=0, high=4, size=(4,))

        legacy_cache_reordered = legacy_reorder_fn(legacy_cache, beam_idx)
        new_cache.reorder_cache(beam_idx)

        # Let's check that the results are the same
        for layer_idx in range(10):
            for key_value_idx in range(2):
                self.assertTrue(
                    torch.allclose(
                        new_cache[layer_idx][key_value_idx], legacy_cache_reordered[layer_idx][key_value_idx]
                    )
                )


@require_torch_gpu
@slow
class CacheIntegrationTest(unittest.TestCase):
    def test_dynamic_cache_hard(self):
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", padding_side="left")
        model = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-2-7b-hf", device_map="auto", torch_dtype=torch.float16
        )
        inputs = tokenizer(["Here's everything I know about cats. Cats"], return_tensors="pt").to(model.device)

        # DynamicCache and the legacy cache format should be equivalent
        set_seed(0)
        gen_out_legacy = model.generate(**inputs, do_sample=True, max_new_tokens=256)
        set_seed(0)
        gen_out = model.generate(**inputs, do_sample=True, max_new_tokens=256, past_key_values=DynamicCache())
        self.assertListEqual(gen_out_legacy.tolist(), gen_out.tolist())

        decoded = tokenizer.batch_decode(gen_out, skip_special_tokens=True)
        expected_text = (
            "Here's everything I know about cats. Cats are mysterious creatures. They can't talk, and they don't like "
            "to be held. They don't play fetch, and they don't like to be hugged. But they do like to be petted.\n"
            "Cats are also very independent. They don't like to be told what to do, and they don't like to be told "
            "what to eat. They are also very territorial. They don't like to share their food or their toys.\nCats "
            "are also very curious. They like to explore, and they like to play. They are also very fast. They can "
            "run very fast, and they can jump very high.\nCats are also very smart. They can learn tricks, and they "
            "can solve problems. They are also very playful. They like to play with toys, and they like to play with "
            "other cats.\nCats are also very affectionate. They like to be petted, and they like to be held. They "
            "also like to be scratched.\nCats are also very clean. They like to groom themselves, and they like to "
            "clean their litter box.\nCats are also very independent. They don't"
        )
        self.assertEqual(decoded[0], expected_text)

    def test_dynamic_cache_batched(self):
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", padding_side="left")
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-2-7b-hf", device_map="auto", torch_dtype=torch.float16
        )
        inputs = tokenizer(["A sequence: 1, 2, 3, 4, 5", "A sequence: A, B, C"], padding=True, return_tensors="pt").to(
            model.device
        )

        gen_out = model.generate(**inputs, do_sample=False, max_new_tokens=10, past_key_values=DynamicCache())
        decoded = tokenizer.batch_decode(gen_out, skip_special_tokens=True)
        expected_text = ["A sequence: 1, 2, 3, 4, 5, 6, 7, 8,", "A sequence: A, B, C, D, E, F, G, H"]
        self.assertListEqual(decoded, expected_text)

    def test_dynamic_cache_beam_search(self):
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", padding_side="left")
        model = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-2-7b-hf", device_map="auto", torch_dtype=torch.float16
        )

        inputs = tokenizer(["The best color is"], return_tensors="pt").to(model.device)
        gen_out = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=20,
            num_beams=2,
            num_return_sequences=2,
        )
        decoded = tokenizer.batch_decode(gen_out, skip_special_tokens=True)
        expected_text = [
            "The best color is the one that makes you feel good.\nThe best color is the one that makes you feel good",
            "The best color is the one that suits you.\nThe best color is the one that suits you. The",
        ]
        self.assertListEqual(decoded, expected_text)

    @require_auto_gptq
    def test_sink_cache_hard(self):
        tokenizer = AutoTokenizer.from_pretrained("TheBloke/LLaMa-7B-GPTQ")
        model = AutoModelForCausalLM.from_pretrained("TheBloke/LLaMa-7B-GPTQ", device_map="auto")

        inputs = tokenizer(["Vaswani et al. (2017) introduced the Transformers"], return_tensors="pt").to(model.device)

        # Set up the SinkCache. Using a small window length to contain computational complexity. If this example is run
        # without a SinkCache, the last few tokens are gibberish (ends in "of the of the of a of a of")
        cache = SinkCache(window_length=508, num_sink_tokens=4)
        gen_out = model.generate(**inputs, do_sample=False, max_new_tokens=3000, past_key_values=cache)
        decoded = tokenizer.batch_decode(gen_out, skip_special_tokens=True)
        self.assertTrue(decoded[0].endswith("to perform a variety of tasks. The Transformer is a neural network"))

    def test_sink_cache_iterative_prompts(self):
        """Tests that SinkCache supports more than one new token at once, when shifting the cache"""
        tokenizer = AutoTokenizer.from_pretrained("HuggingFaceH4/zephyr-7b-beta")
        model = AutoModelForCausalLM.from_pretrained(
            "HuggingFaceH4/zephyr-7b-beta", device_map="auto", torch_dtype=torch.float16
        )
        prompt = (
            "Compose an engaging travel blog post about a recent trip to Hawaii, highlighting cultural experiences "
            "and must-see attractions."
        )

        # Prepare generation settings
        cache = SinkCache(window_length=256, num_sink_tokens=4)
        input_ids = torch.tensor([], device=model.device, dtype=torch.int)
        for _ in range(3):
            # Tokenize the prompt with the correct chat template
            chat = [{"role": "user", "content": prompt}]
            tokenized_chat = tokenizer.apply_chat_template(chat, return_tensors="pt", add_generation_prompt=True).to(
                model.device
            )
            input_ids = torch.cat((input_ids, tokenized_chat), dim=1)

            # Perform the generation
            gen_out = model.generate(
                input_ids, do_sample=False, max_new_tokens=100, past_key_values=cache, use_cache=True
            )
            input_ids = gen_out

        # We went well beyond the cache length
        self.assertTrue(input_ids.shape[1] > cache.get_max_length() * 1.5)

        # And it still produces a coherent english
        decoded = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        last_output = (
            "<|assistant|>\nAs the sun began to set over the Pacific Ocean, I found myself standing on the shores of "
            "Waikiki Beach, my heart filled with awe and wonder. I had just returned from a two-week journey to the "
            "beautiful island of Hawaii, and it had been an unforgettable experience filled with cultural experiences "
            "and must-see attractions that left me breathless.\n\nOne of the most memorable experiences of my trip "
            "was visiting the historic district of Honolulu. Here,"
        )
        self.assertTrue(decoded[0].endswith(last_output))
