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

import copy
import unittest

from packaging import version
from parameterized import parameterized

from transformers import set_seed
from transformers.generation.configuration_utils import ALL_CACHE_IMPLEMENTATIONS
from transformers.testing_utils import (
    CaptureStderr,
    backend_device_count,
    backend_torch_accelerator_module,
    cleanup,
    get_gpu_count,
    is_torch_available,
    require_read_token,
    require_torch,
    require_torch_accelerator,
    require_torch_gpu,
    require_torch_multi_accelerator,
    require_torch_multi_gpu,
    slow,
    torch_device,
)
from transformers.utils import is_optimum_quanto_available, is_torch_greater_or_equal


if is_torch_available():
    import torch

    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        Cache,
        ClvpForCausalLM,
        DynamicCache,
        Gemma2Config,
        GenerationConfig,
        HybridCache,
        LlamaConfig,
        SlidingWindowCache,
        StaticCache,
        convert_and_export_with_cache,
        pipeline,
    )
    from transformers.integrations.executorch import export_with_dynamic_cache


TEST_CACHE_IMPLEMENTATIONS = [
    cache_name
    for cache_name in ALL_CACHE_IMPLEMENTATIONS
    # TODO (joao): Mamba is not compatible with most models, remove from `ALL_CACHE_IMPLEMENTATIONS`?
    if cache_name != "mamba"
    # TODO (joao): offloaded_hybrid == offloaded_hybrid_chunked, deprecate one of them
    if cache_name != "offloaded_hybrid"
]


@require_torch
class CacheTest(unittest.TestCase):
    """Cache tests that don't require loading models"""

    def test_dynamic_cache_retrocompatibility(self):
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
        legacy_reorder_fn = ClvpForCausalLM._reorder_cache  # An example of a legacy `_reorder_cache` function

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

    def test_static_cache_mha_mqa_gqa(self):
        """
        Tests that static cache works with multi-head attention (MHA), grouped query attention (GQA), and multi-query
        attention (MQA)
        """

        def _random_kvs(config):
            # shape for key and values: (batch_size, num_heads, seq_len, head_dim)
            random_keys = torch.rand(
                (1, config.num_key_value_heads, 1, config.hidden_size // config.num_attention_heads),
                device=torch_device,
            )
            random_values = torch.rand(
                (1, config.num_key_value_heads, 1, config.hidden_size // config.num_attention_heads),
                device=torch_device,
            )
            return random_keys, random_values

        mha_config = LlamaConfig(num_attention_heads=32)
        mha_static_cache = StaticCache(config=mha_config, max_batch_size=1, max_cache_len=10, device=torch_device)
        cached_keys, cached_values = mha_static_cache.update(
            *_random_kvs(mha_config), 0, cache_kwargs={"cache_position": torch.arange(1).to(torch_device)}
        )
        self.assertTrue(cached_keys.shape == (1, 32, 10, 128))
        self.assertTrue(cached_values.shape == (1, 32, 10, 128))

        gqa_config = LlamaConfig(num_attention_heads=32, num_key_value_heads=4)
        gqa_static_cache = StaticCache(config=gqa_config, max_batch_size=1, max_cache_len=10, device=torch_device)
        cached_keys, cached_values = gqa_static_cache.update(
            *_random_kvs(gqa_config), 0, cache_kwargs={"cache_position": torch.arange(1).to(torch_device)}
        )
        self.assertTrue(cached_keys.shape == (1, 4, 10, 128))
        self.assertTrue(cached_values.shape == (1, 4, 10, 128))

        mqa_config = LlamaConfig(num_attention_heads=32, num_key_value_heads=1)
        mqa_static_cache = StaticCache(config=mqa_config, max_batch_size=1, max_cache_len=10, device=torch_device)
        cached_keys, cached_values = mqa_static_cache.update(
            *_random_kvs(mqa_config), 0, cache_kwargs={"cache_position": torch.arange(1).to(torch_device)}
        )
        self.assertTrue(cached_keys.shape == (1, 1, 10, 128))
        self.assertTrue(cached_values.shape == (1, 1, 10, 128))


def _skip_on_failed_cache_prerequisites(test, cache_implementation):
    """Function to skip tests on failed cache prerequisites, given a cache implementation"""
    # Installed dependencies
    if cache_implementation == "quantized" and not is_optimum_quanto_available():
        test.skipTest("Quanto is not available")
    # Devices
    if "offloaded" in cache_implementation:
        has_accelerator = torch_device is not None and torch_device != "cpu"
        if not has_accelerator:
            test.skipTest("Offloaded caches require an accelerator")
        if cache_implementation in ["offloaded_static", "offloaded_hybrid_chunked"]:
            if backend_device_count(torch_device) != 1:
                test.skipTest("Offloaded static caches require exactly 1 accelerator")


class CacheIntegrationTest(unittest.TestCase):
    """Fast cache integration tests that share the same small model"""

    @classmethod
    def setUpClass(cls):
        # Load once and reuse across tests
        cls.tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct", padding_side="left")
        cls.model = AutoModelForCausalLM.from_pretrained(
            "HuggingFaceTB/SmolLM2-135M-Instruct", device_map="auto", torch_dtype=torch.float16
        )
        cls.model.config.sliding_window = 256  # hack to enable the use of caches with sliding windows

    @parameterized.expand(TEST_CACHE_IMPLEMENTATIONS)
    def test_cache_batched(self, cache_implementation):
        """Sanity check: caches' `.update` function expects batched inputs"""
        _skip_on_failed_cache_prerequisites(self, cache_implementation)

        EXPECTED_GENERATION = ["A sequence: 1, 2, 3, 4, 5, 6, 7, 8,", "A sequence: A, B, C, D, E, F, G, H"]

        inputs = self.tokenizer(
            ["A sequence: 1, 2, 3, 4, 5", "A sequence: A, B, C"], padding=True, return_tensors="pt"
        )
        inputs = inputs.to(self.model.device)

        gen_out = self.model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=10,
            return_dict_in_generate=True,
            cache_implementation=cache_implementation,
            disable_compile=True,
        )
        # Sanity check: a cache was used
        self.assertIsInstance(gen_out.past_key_values, Cache)
        # Confirm that the output matches expectations
        decoded = self.tokenizer.batch_decode(gen_out.sequences, skip_special_tokens=True)
        self.assertListEqual(decoded, EXPECTED_GENERATION)

    @parameterized.expand(TEST_CACHE_IMPLEMENTATIONS)
    def test_cache_beam_search(self, cache_implementation):
        """
        Sanity check: caches' `reorder_cache` is operational. We can confirm this by looking at the beam indices
        (an output sequence contains multiple beam indices).
        """
        _skip_on_failed_cache_prerequisites(self, cache_implementation)
        if cache_implementation == "offloaded_hybrid_chunked":
            # TODO (joao, cyril): something is off with `offloaded_hybrid_chunked` aka `OffloadedHybridCache`: the
            # output sequence (and the corresponding beam scores, if we add `output_scores=True`) are significantly
            # different from the other caches.
            self.skipTest("`offloaded_hybrid_chunked` fails this test")

        EXPECTED_GENERATION = [
            "Blue is the color of the sky, and the color of",
            "Blue is the color of the sky, and the second is",
        ]

        inputs = self.tokenizer(["Blue is"], return_tensors="pt").to(self.model.device)
        gen_out = self.model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=10,
            num_beams=2,
            num_return_sequences=2,
            cache_implementation=cache_implementation,
            disable_compile=True,
            return_dict_in_generate=True,
        )
        # Sanity check: a cache was used
        self.assertIsInstance(gen_out.past_key_values, Cache)
        # At least one of the sequences requires multiple beam indices -> `reorder_cache` had to shift things around
        self.assertTrue(any(len(set(beams_in_sequence)) > 1 for beams_in_sequence in gen_out.beam_indices))
        # Confirm that the output matches expectations
        decoded = self.tokenizer.batch_decode(gen_out.sequences, skip_special_tokens=True)
        self.assertListEqual(decoded, EXPECTED_GENERATION)

    @parameterized.expand(TEST_CACHE_IMPLEMENTATIONS)
    def test_cache_extra_left_padding(self, cache_implementation):
        """Tests that adding extra left-padding does not affect the generation with the cache"""
        _skip_on_failed_cache_prerequisites(self, cache_implementation)

        EXPECTED_GENERATION = ["The cat's whiskers are also a sign of anxiety."]

        inputs = self.tokenizer(["The cat"], padding=True, return_tensors="pt").to(self.model.device)
        generation_kwargs = {
            "do_sample": False,
            "max_new_tokens": 10,
            "cache_implementation": cache_implementation,
            "disable_compile": True,
        }

        gen_out = self.model.generate(**inputs, **generation_kwargs)
        decoded = self.tokenizer.batch_decode(gen_out, skip_special_tokens=True)
        self.assertListEqual(decoded, EXPECTED_GENERATION)

        # Now with extra left-padding
        inputs_expanded = self.tokenizer(["The cat"], padding=True, return_tensors="pt", pad_to_multiple_of=32)
        inputs_expanded = inputs_expanded.to(self.model.device)
        self.assertTrue(inputs.input_ids.shape[1] < inputs_expanded.input_ids.shape[1])
        gen_out = self.model.generate(**inputs_expanded, **generation_kwargs)
        decoded = self.tokenizer.batch_decode(gen_out, skip_special_tokens=True)
        self.assertListEqual(decoded, EXPECTED_GENERATION)


@require_torch_accelerator
class CacheHardIntegrationTest(unittest.TestCase):
    """Hard cache integration tests that require loading different models"""

    def setUp(self):
        # Clears memory before each test. Some tests use large models, which might result in suboptimal torch
        # re-allocation if we run multiple tests in a row without clearing memory.
        cleanup(torch_device, gc_collect=True)

    @classmethod
    def tearDownClass(cls):
        # Clears memory after the last test. See `setUp` for more details.
        cleanup(torch_device, gc_collect=True)

    @slow
    def test_dynamic_cache_hard(self):
        """Hard test for base cache implementation -- minor numerical fluctuations will cause this test to fail"""
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B", padding_side="left")
        model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-4B", device_map="auto", torch_dtype=torch.bfloat16)
        inputs = tokenizer(["Here's everything I know about cats. Cats"], return_tensors="pt").to(model.device)

        set_seed(0)
        gen_out = model.generate(
            **inputs, do_sample=True, max_new_tokens=256, return_dict_in_generate=True, output_scores=True
        )
        decoded = tokenizer.batch_decode(gen_out.sequences, skip_special_tokens=True)
        # sum of the scores for the generated tokens
        input_length = inputs.input_ids.shape[1]
        score_sum = sum(
            [score[0][gen_out.sequences[0][input_length + idx]] for idx, score in enumerate(gen_out.scores)]
        )

        EXPECTED_GENERATION = (
            "Here's everything I know about cats. Cats are mammals, they have four legs, they have a tail, they have "
            "a face with a nose, eyes, and mouth. They have fur, they have claws, and they have a body that is "
            "covered in fur. They are carnivores, so they eat meat. They are also very clean animals, they groom "
            "themselves. They have a lot of different breeds. Some are small, some are large. Some are friendly, "
            "some are not. They have a lot of different personalities. They can be very independent, or they can be "
            "very affectionate. They can be very playful, or they can be very lazy. They can be very intelligent, or "
            "they can be very silly. They have a lot of different behaviors. They can be very curious, or they can "
            "be very cautious. They can be very vocal, or they can be very quiet. They can be very social, or they "
            "can be very solitary. They can be very active, or they can be very inactive. They can be very "
            "affectionate, or they can be very aloof. They can be very playful, or they can be very lazy. They can "
            "be very intelligent, or they can be very silly. They have a lot of different behaviors. They can be "
            "very curious, or they can"
        )
        EXPECTED_SCORE_SUM = 11017.4971
        self.assertEqual(decoded[0], EXPECTED_GENERATION)
        self.assertAlmostEqual(score_sum, EXPECTED_SCORE_SUM, places=2)
        self.assertIsInstance(gen_out.past_key_values, DynamicCache)  # sanity check

    @parameterized.expand([("eager"), ("sdpa")])
    @require_torch_accelerator
    @slow
    def test_static_cache_greedy_decoding_pad_left(self, attn_implementation):
        """Tests that different cache implementations work well with eager and SDPA inference"""
        EXPECTED_GENERATION = [
            "The best color is the one that is most suitable for the purpose.",
            "We should not undermind the issues at hand, but instead, we should focus on the things",
        ]

        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B", padding_side="left")
        model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen3-4B",
            torch_dtype=torch.bfloat16,
            attn_implementation=attn_implementation,
            device_map="auto",
        )
        inputs = tokenizer(
            ["The best color is", "We should not undermind the issues at hand"], padding=True, return_tensors="pt"
        ).to(model.device)
        generation_kwargs = {"do_sample": False, "max_new_tokens": 10, "return_dict_in_generate": True}

        set_seed(0)
        gen_out = model.generate(**inputs, **generation_kwargs)
        decoded = tokenizer.batch_decode(gen_out.sequences, skip_special_tokens=True)
        with self.subTest(f"{attn_implementation}, dynamic"):
            self.assertListEqual(decoded, EXPECTED_GENERATION)
            self.assertIsInstance(gen_out.past_key_values, DynamicCache)  # sanity check

        set_seed(0)
        gen_out = model.generate(**inputs, **generation_kwargs, cache_implementation="static", disable_compile=True)
        decoded = tokenizer.batch_decode(gen_out.sequences, skip_special_tokens=True)
        with self.subTest(f"{attn_implementation}, static, eager"):
            self.assertListEqual(decoded, EXPECTED_GENERATION)
            self.assertIsInstance(gen_out.past_key_values, StaticCache)  # sanity check

        set_seed(0)
        gen_out = model.generate(**inputs, **generation_kwargs, cache_implementation="static")
        decoded = tokenizer.batch_decode(gen_out.sequences, skip_special_tokens=True)
        with self.subTest(f"{attn_implementation}, static, compiled"):
            self.assertListEqual(decoded, EXPECTED_GENERATION)
            self.assertIsInstance(gen_out.past_key_values, StaticCache)  # sanity check

    @require_torch_accelerator
    @slow
    def test_offloaded_cache_uses_less_memory_than_dynamic_cache(self):
        """Tests that OffloadedCache uses less memory than the default DynamicCache"""
        model_name = "microsoft/Phi-3-mini-4k-instruct"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.float16)
        device = model.device

        if not is_torch_greater_or_equal("2.7", accept_dev=True) and device.type == "xpu":
            self.skipTest(reason="This test requires torch >= 2.7 to run on xpu.")

        input_text = "Fun fact:"
        inputs = tokenizer(input_text, return_tensors="pt").to(device)
        common = {
            "num_beams": 4,
            "num_beam_groups": 2,
            "num_return_sequences": 4,
            "diversity_penalty": 1.0,
            "max_new_tokens": 20,
            "early_stopping": True,
        }
        original = GenerationConfig(**common)
        offloaded = GenerationConfig(cache_implementation="offloaded", **common)

        torch_accelerator_module = backend_torch_accelerator_module(device.type)

        torch_accelerator_module.reset_peak_memory_stats(device)
        model.generate(generation_config=original, **inputs)
        original_peak_memory = torch_accelerator_module.max_memory_allocated(device)
        torch_accelerator_module.reset_peak_memory_stats(device)
        model.generate(generation_config=offloaded, **inputs)
        offloaded_peak_memory = torch_accelerator_module.max_memory_allocated(device)
        self.assertTrue(offloaded_peak_memory < original_peak_memory)

    @require_torch_accelerator
    @slow
    def test_cache_copy(self):
        """Tests that we can manually set a cache, copy, and reuse it for generation"""
        # TODO (joao): test for all cache implementations in `CacheIntegrationTest` after standardizing the
        # lazy init of cache layers
        model_name = "microsoft/Phi-3-mini-4k-instruct"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map=torch_device, torch_dtype=torch.bfloat16)

        prompt_cache = StaticCache(
            config=model.config, max_batch_size=1, max_cache_len=1024, device=torch_device, dtype=torch.bfloat16
        )

        INITIAL_PROMPT = "You are a helpful assistant. "
        inputs_initial_prompt = tokenizer(INITIAL_PROMPT, return_tensors="pt").to(torch_device)
        # This is the common prompt cached, we need to run forward without grad to be able to copy
        with torch.no_grad():
            prompt_cache = model(**inputs_initial_prompt, past_key_values=prompt_cache).past_key_values

        prompts = ["Help me to write a blogpost about travelling.", "What is the capital of France?"]
        responses = []
        for prompt in prompts:
            new_inputs = tokenizer(INITIAL_PROMPT + prompt, return_tensors="pt").to(torch_device)
            past_key_values = copy.deepcopy(prompt_cache)
            outputs = model.generate(
                **new_inputs, past_key_values=past_key_values, max_new_tokens=40, disable_compile=True
            )
            response = tokenizer.batch_decode(outputs)[0]
            responses.append(response)

        EXPECTED_DECODED_TEXT = [
            "You are a helpful assistant. Help me to write a blogpost about travelling.\n\nTraveling is an "
            "enriching experience that broadens our horizons and allows us to explore the world beyond our comfort "
            "zones. Whether it's a short weekend getaway",
            "You are a helpful assistant. What is the capital of France?\n\n\n## Response:Paris is the capital "
            "of France.\n\n\n\n\n\n\n<|endoftext|>",
        ]

        self.assertEqual(responses, EXPECTED_DECODED_TEXT)

    @require_torch_multi_gpu
    def test_data_parallel_dynamic_cache(self):
        """
        Tests that the dynamic cache works with nn.DataParallel. Under the hood, `DynamicCache` is rebuilt from
        multiple `DynamicCache` in the gather step.
        """

        model_repo = "hf-internal-testing/tiny-random-MistralForCausalLM"
        model = AutoModelForCausalLM.from_pretrained(model_repo).to(torch_device)
        tokenizer = AutoTokenizer.from_pretrained(model_repo)

        # w/o DP: batch_size = num_gpu
        # w DP: batch_size = 1 (with num_gpus replicas)
        num_gpus = get_gpu_count()
        model_inputs = tokenizer(["foo bar"] * num_gpus, return_tensors="pt").to(model.device)

        # w/o DP
        no_parallelism_cache = model(**model_inputs).past_key_values
        self.assertIsInstance(no_parallelism_cache, DynamicCache)

        # w DP
        model = torch.nn.DataParallel(model)
        parallelism_cache = model(**model_inputs).past_key_values
        self.assertIsInstance(parallelism_cache, DynamicCache)

        # Check that the caches are the same
        for layer_idx in range(len(no_parallelism_cache)):
            for kv_idx in range(2):  # 0 = key, 1 = value
                torch.testing.assert_close(
                    actual=parallelism_cache[layer_idx][kv_idx], expected=no_parallelism_cache[layer_idx][kv_idx]
                )

    @require_torch_gpu
    def test_static_cache_no_cuda_graph_skips(self):
        """
        Tests generating with static cache and compilation doesn't skip cuda graphs. Regression test for #36543.

        (? We set `fullgraph=True`, which according to torch docs means it should raise an exception. Instead,
        messages are being thrown to stderr?)
        """
        model_repo = "hf-internal-testing/tiny-random-MistralForCausalLM"
        model = AutoModelForCausalLM.from_pretrained(model_repo).to(torch_device)
        tokenizer = AutoTokenizer.from_pretrained(model_repo)
        inputs = tokenizer(["foo bar"], return_tensors="pt").to(torch_device)

        # on `main`, prior to #36543, this would send stderr messages about cuda graphs being skipped.
        with CaptureStderr() as cap:
            model.generate(**inputs, max_new_tokens=2, cache_implementation="static")
        self.assertNotIn("cuda", cap.err.lower())

    @require_torch_multi_accelerator
    @slow
    @require_read_token
    def test_static_cache_multi_accelerator(self):
        """Regression test for #35164: static cache with multi-accelerator"""

        model_id = "google/gemma-2-2b-it"
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        device_map = {"model.embed_tokens": 0, "model.norm": 1, "model.rotary_emb": 1, "lm_head": 0}
        num_hidden_layers = 26
        for i in range(num_hidden_layers):
            device_map[f"model.layers.{i}"] = 0 if i < 13 else 1

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype="bfloat16",
            device_map=device_map,
        )
        inputs = tokenizer("Today is a beautiful day!", return_tensors="pt").to(0)
        _ = model(**inputs)
        _ = model.generate(**inputs, max_new_tokens=2, cache_implementation="hybrid")

    @require_torch_accelerator
    @parameterized.expand(TEST_CACHE_IMPLEMENTATIONS)
    def test_cache_gptj_model(self, cache_implementation):
        """Tests caches with GPT-J model. Regression test for https://github.com/huggingface/transformers/pull/34799"""
        _skip_on_failed_cache_prerequisites(self, cache_implementation)

        model_id = "hf-internal-testing/tiny-random-GPTJForCausalLM"
        pipe = pipeline("text-generation", model=model_id, torch_dtype=torch.bfloat16)
        pipe.model.config.sliding_window = (
            256 if cache_implementation in ["sliding_window", "hybrid", "hybrid_chunked"] else None
        )
        out = pipe(
            "hello world",
            cache_implementation=cache_implementation,
            max_new_tokens=10,
            do_sample=False,
            disable_compile=True,
            return_tensors=True,
        )[0]["generated_token_ids"][-10:]
        EXPECTED_OUTPUT = [879, 175, 39, 141, 1000, 975, 951, 991, 683, 441]
        self.assertListEqual(out, EXPECTED_OUTPUT)


@require_torch
class CacheExportIntegrationTest(unittest.TestCase):
    """Cache tests that rely on `torch.export()` and model loading"""

    def test_dynamic_cache_exportability(self):
        model = AutoModelForCausalLM.from_pretrained("hf-internal-testing/tiny-random-MistralForCausalLM")
        model = model.eval()
        tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-MistralForCausalLM")
        prompt = "What is the best way to debug python script?"
        inputs = tokenizer(prompt, return_tensors="pt")
        attention_mask = inputs.attention_mask
        input_ids = inputs.input_ids

        ep = export_with_dynamic_cache(model, input_ids, attention_mask)
        res = ep.module()(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=DynamicCache(),
            use_cache=True,
        )
        self.assertTrue(len(res.past_key_values.key_cache) == model.config.num_hidden_layers)
        self.assertEqual(2 * model.config.num_hidden_layers + 1, len(ep.graph_signature.output_specs))
        self.assertEqual(
            3,
            len(
                [
                    x
                    for x in ep.graph_signature.input_specs
                    if x.kind == torch.export.graph_signature.InputKind.USER_INPUT
                ]
            ),
        )

        past_key_values_eager = DynamicCache()
        res_eager = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values_eager,
            use_cache=True,
        )
        self.assertTrue(torch.allclose(res.logits, res_eager.logits))
        for k1, k2 in zip(res.past_key_values.key_cache, res_eager.past_key_values.key_cache):
            self.assertTrue(torch.allclose(k1, k2))

        for v1, v2 in zip(res.past_key_values.value_cache, res_eager.past_key_values.value_cache):
            self.assertTrue(torch.allclose(v1, v2))

    def test_dynamic_cache_exportability_multiple_run(self):
        # When exporting with DynamicCache, you should export two graphs:
        #   1. A graph without cache
        #   2. A graph with cache
        # In the future, we will make improvements to export API to export two graphs
        # more seamlessly.

        model = AutoModelForCausalLM.from_pretrained("hf-internal-testing/tiny-random-MistralForCausalLM")
        model = model.eval()
        tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-MistralForCausalLM")
        prompt = "What is the best way to debug python script?"
        inputs = tokenizer(prompt, return_tensors="pt")
        attention_mask = inputs.attention_mask
        input_ids = inputs.input_ids

        ep = export_with_dynamic_cache(model, input_ids, attention_mask)
        res = ep.module()(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=DynamicCache(),
            use_cache=True,
        )
        self.assertTrue(len(res.past_key_values.key_cache) == model.config.num_hidden_layers)
        self.assertEqual(2 * model.config.num_hidden_layers + 1, len(ep.graph_signature.output_specs))
        self.assertEqual(
            3,
            len(
                [
                    x
                    for x in ep.graph_signature.input_specs
                    if x.kind == torch.export.graph_signature.InputKind.USER_INPUT
                ]
            ),
        )

        res_eager = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=DynamicCache(),
            use_cache=True,
        )
        past_key_values_eager = res_eager.past_key_values
        past_key_values = res.past_key_values

        shapes = torch.export.ShapesCollection()
        dyn = torch.export.Dim("seq", max=512)

        for ix in range(len(past_key_values.key_cache)):
            shapes[past_key_values.key_cache[ix]] = (None, None, dyn, None)
            shapes[past_key_values.value_cache[ix]] = (None, None, dyn, None)

        ep_second = torch.export.export(
            model,
            (),
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "past_key_values": past_key_values,
                "use_cache": True,
            },
            strict=False,
            dynamic_shapes=shapes,
        )
        res_export = ep_second.module()(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
        )
        # It should work with variable len
        res_export_2 = ep_second.module()(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=res_export.past_key_values,
            use_cache=True,
        )

        res_eager = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values_eager,
            use_cache=True,
        )
        res_eager_2 = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=res_eager.past_key_values,
            use_cache=True,
        )

        for k1, k2 in zip(res_export_2.past_key_values.key_cache, res_eager_2.past_key_values.key_cache):
            self.assertTrue(torch.allclose(k1, k2))

        for v1, v2 in zip(res_export_2.past_key_values.value_cache, res_eager_2.past_key_values.value_cache):
            self.assertTrue(torch.allclose(v1, v2))

    def test_static_cache_exportability(self):
        """
        Tests that static cache works with `torch.export()`
        """
        if not is_torch_greater_or_equal("2.3"):
            self.skipTest(reason="This test requires torch >= 2.3 to run.")

        set_seed(0)
        device = "cpu"
        dtype = "bfloat16"
        cache_implementation = "static"
        attn_implementation = "sdpa"  # Export and ExecuTorch only works for SdpaAttention
        batch_size = 1
        max_cache_len = 1234
        model_id = "hf-internal-testing/tiny-random-LlamaForCausalLM"
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map=device,
            torch_dtype=dtype,
            attn_implementation=attn_implementation,
            generation_config=GenerationConfig(
                use_cache=True,
                cache_implementation=cache_implementation,
                max_length=max_cache_len,
                cache_config={
                    "batch_size": batch_size,
                    "max_cache_len": max_cache_len,
                    "device": device,
                },
            ),
        )
        # Check if cache config is passed through correctly
        self.assertEqual(model.generation_config.use_cache, True)
        self.assertEqual(model.generation_config.cache_implementation, cache_implementation)
        self.assertEqual(model.generation_config.max_length, max_cache_len)
        self.assertTrue(model.generation_config.cache_config is not None)
        self.assertEqual(model.generation_config.cache_config.batch_size, batch_size)
        self.assertEqual(model.generation_config.cache_config.max_cache_len, max_cache_len)

        exported_program = convert_and_export_with_cache(model)

        # Check if the exported model is configured with the `StaticCache` correctly
        n_static_key_caches = n_static_value_caches = 0
        for buffer_name, buffer in exported_program.named_buffers():
            if buffer_name.startswith("key_cache"):
                self.assertTrue(buffer.shape[0] == batch_size)
                self.assertTrue(buffer.shape[2] == max_cache_len)
                n_static_key_caches = n_static_key_caches + 1
            if buffer_name.startswith("value_cache"):
                self.assertTrue(buffer.shape[0] == batch_size)
                self.assertTrue(buffer.shape[2] == max_cache_len)
                n_static_value_caches = n_static_value_caches + 1
        self.assertEqual(n_static_key_caches, model.config.num_hidden_layers)
        self.assertEqual(n_static_value_caches, model.config.num_hidden_layers)

        # Export with dynamic shapes
        input_ids = torch.zeros((1, 3), dtype=torch.long)
        cache_position = torch.tensor([0, 1, 2], dtype=torch.long)
        dynamic_shapes = {"input_ids": {1: torch.export.Dim.DYNAMIC}, "cache_position": {0: torch.export.Dim.DYNAMIC}}
        strict = version.parse(torch.__version__) != version.parse("2.7.0")
        exported_program = convert_and_export_with_cache(
            model,
            example_input_ids=input_ids,
            example_cache_position=cache_position,
            dynamic_shapes=dynamic_shapes,
            strict=strict,
        )

        from transformers.integrations.executorch import TorchExportableModuleForDecoderOnlyLM

        exportable_module = TorchExportableModuleForDecoderOnlyLM(model)
        exported_program = exportable_module.export(
            input_ids=input_ids,
            cache_position=cache_position,
            dynamic_shapes=dynamic_shapes,
            strict=strict,
        )

    def test_hybrid_cache_exportability(self):
        """
        Tests that static cache works with `torch.export()`
        """
        if not is_torch_greater_or_equal("2.6"):
            self.skipTest(reason="This test requires torch >= 2.6 to run.")

        from transformers.integrations.executorch import TorchExportableModuleForDecoderOnlyLM

        set_seed(0)
        model_id = "hf-internal-testing/tiny-random-Gemma3ForCausalLM"
        model = AutoModelForCausalLM.from_pretrained(model_id)
        model.eval()
        self.assertEqual(model.config.use_cache, True)
        self.assertEqual(model.config.cache_implementation, "hybrid")

        # Export + HybridCache
        model.eval()
        max_batch_size = 1
        max_cache_len = 23
        exportable_module = TorchExportableModuleForDecoderOnlyLM(model, max_batch_size, max_cache_len)
        exported_program = exportable_module.export()
        n_g_key_caches = n_g_value_caches = 0
        for buffer_name, buffer in exported_program.named_buffers():
            if buffer_name.startswith("key_cache"):
                self.assertTrue(buffer.shape[0] == max_batch_size)
                self.assertTrue(buffer.shape[2] == max_cache_len)
                n_g_key_caches = n_g_key_caches + 1
            if buffer_name.startswith("value_cache"):
                self.assertTrue(buffer.shape[0] == max_batch_size)
                self.assertTrue(buffer.shape[2] == max_cache_len)
                n_g_value_caches = n_g_value_caches + 1
        self.assertEqual(n_g_key_caches, model.config.num_hidden_layers)
        self.assertEqual(n_g_value_caches, model.config.num_hidden_layers)

        # Export with dynamic shapes using Dim.AUTO
        input_ids = torch.zeros((1, 3), dtype=torch.long)
        cache_position = torch.tensor([0, 1, 2], dtype=torch.long)
        dynamic_shapes = {"input_ids": {1: torch.export.Dim.DYNAMIC}, "cache_position": {0: torch.export.Dim.DYNAMIC}}
        strict = version.parse(torch.__version__) != version.parse("2.7.0")
        exported_program = exportable_module.export(
            input_ids=input_ids,
            cache_position=cache_position,
            dynamic_shapes=dynamic_shapes,
            strict=strict,
        )


class SyntheticCacheTest(unittest.TestCase):
    """Tests cache behavior with simple dummy data."""

    def setUp(self):
        """Set up common configuration and cache instances for all tests."""
        self.window_size = 4
        self.max_cache_len = 4
        self.config = Gemma2Config(
            num_hidden_layers=1,
            num_key_value_heads=1,
            num_attention_heads=1,
            head_dim=1,
            hidden_size=1,
            sliding_window=self.window_size,
            sliding_window_pattern=2,  # Default pattern for hybrid sliding
        )

    def test_static_cache_out_of_bounds(self):
        """Test StaticCache raises IndexError for out-of-bounds positions."""
        static_cache = StaticCache(config=self.config, max_batch_size=1, max_cache_len=self.max_cache_len)
        pos_out_of_bounds = torch.tensor([self.max_cache_len])  # Position >= max_cache_len

        with self.assertRaises(IndexError):
            static_cache.update(
                key_states=torch.tensor([[[[1.0]]]]),
                value_states=torch.tensor([[[[1.0]]]]),
                layer_idx=0,
                cache_kwargs={"cache_position": pos_out_of_bounds},
            )

    def test_static_cache(self):
        """Test StaticCache with manually prefilled states and hardcoded assertions.

        Scenario 1: Fill up to near capacity
        prefill:       [1.0, 2.0, 0.0, 0.0]
        update pos 2:  [1.0, 2.0, 3.0, 0.0]

        Scenario 2: Fill to capacity
        update pos 3:  [1.0, 2.0, 3.0, 4.0]
        """
        # Scenario 1: Fill up to near capacity
        static_cache = StaticCache(config=self.config, max_batch_size=1, max_cache_len=self.max_cache_len)
        prefill = torch.tensor([1.0, 2.0, 0.0, 0.0])[None, None, :, None]
        static_cache.update(key_states=prefill, value_states=prefill, layer_idx=0, cache_kwargs=None)
        static_cache.update(
            key_states=torch.tensor(3.0)[None, None, None, None],
            value_states=torch.tensor(3.0)[None, None, None, None],
            layer_idx=0,
            cache_kwargs={"cache_position": torch.tensor([2])},
        )
        self.assertEqual(
            static_cache.key_cache[0][0, 0, :, 0].tolist(), [1.0, 2.0, 3.0, 0.0], "StaticCache Scenario 1 failed"
        )

        # Scenario 2: Fill to capacity
        static_cache.update(
            key_states=torch.tensor(4.0)[None, None, None, None],
            value_states=torch.tensor(4.0)[None, None, None, None],
            layer_idx=0,
            cache_kwargs={"cache_position": torch.tensor([3])},
        )
        self.assertEqual(
            static_cache.key_cache[0][0, 0, :, 0].tolist(), [1.0, 2.0, 3.0, 4.0], "StaticCache Scenario 2 failed"
        )

    def test_sliding_window_cache(self):
        """Test SlidingWindowCache with manually prefilled states and hardcoded assertions.

        Scenario 1: Update within window, no slide yet
        prefill:       [1.0, 2.0, 0.0, 0.0]
        update pos 2:  [1.0, 2.0, 3.0, 0.0]

        Scenario 2: Update causing slide
        prefill:       [1.0, 2.0, 3.0, 4.0]
        update pos 4:  [2.0, 3.0, 4.0, 5.0] (shift happens as pos > window_size-1)

        Scenario 3: Long prompt handling (prompt_len > window_size)
        input:         [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        result:        [3.0, 4.0, 5.0, 6.0] (keeps last window_size tokens)
        """
        # Scenario 1: Update within window, no slide yet
        sliding_cache = SlidingWindowCache(config=self.config, max_batch_size=1, max_cache_len=self.max_cache_len)
        prefill = torch.tensor([1.0, 2.0, 0.0, 0.0])[None, None, :, None]
        sliding_cache.update(
            key_states=prefill,
            value_states=prefill,
            layer_idx=0,
            cache_kwargs={"cache_position": torch.arange(4), "sliding_window": self.window_size},
        )
        sliding_cache.update(
            key_states=torch.tensor(3.0)[None, None, None, None],
            value_states=torch.tensor(3.0)[None, None, None, None],
            layer_idx=0,
            cache_kwargs={"cache_position": torch.tensor([2]), "sliding_window": self.window_size},
        )
        self.assertEqual(
            sliding_cache.key_cache[0][0, 0, :, 0].tolist(),
            [1.0, 2.0, 3.0, 0.0],
            "SlidingWindowCache Scenario 1 failed",
        )

        # Scenario 2: Update causing slide
        sliding_cache = SlidingWindowCache(config=self.config, max_batch_size=1, max_cache_len=self.max_cache_len)
        prefill = torch.tensor([1.0, 2.0, 3.0, 4.0])[None, None, :, None]
        sliding_cache.update(
            key_states=prefill,
            value_states=prefill,
            layer_idx=0,
            cache_kwargs={"cache_position": torch.arange(4), "sliding_window": self.window_size},
        )
        sliding_cache.update(
            key_states=torch.tensor(5.0)[None, None, None, None],
            value_states=torch.tensor(5.0)[None, None, None, None],
            layer_idx=0,
            cache_kwargs={"cache_position": torch.tensor([4]), "sliding_window": self.window_size},
        )
        self.assertEqual(
            sliding_cache.key_cache[0][0, 0, :, 0].tolist(),
            [2.0, 3.0, 4.0, 5.0],
            "SlidingWindowCache Scenario 2 failed",
        )

        # Scenario 3: Long prompt handling
        sliding_cache = SlidingWindowCache(config=self.config, max_batch_size=1, max_cache_len=self.max_cache_len)
        long_prefill = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])[None, None, :, None]
        sliding_cache.update(
            key_states=long_prefill,
            value_states=long_prefill,
            layer_idx=0,
            cache_kwargs={"cache_position": torch.arange(6), "sliding_window": self.window_size},
        )
        self.assertEqual(
            sliding_cache.key_cache[0][0, 0, :, 0].tolist(),
            [3.0, 4.0, 5.0, 6.0],
            "SlidingWindowCache Scenario 3 failed",
        )

    def test_hybrid_cache_static_mode(self):
        """Test HybridCache in static mode with hardcoded assertions.

        Scenario 1: Static layer behavior
        prefill:       [1.0, 2.0, 0.0, 0.0]
        update pos 2:  [1.0, 2.0, 3.0, 0.0]

        Scenario 2: Fill to capacity
        update pos 3:  [1.0, 2.0, 3.0, 4.0]
        """
        config = copy.deepcopy(self.config)
        config.sliding_window_pattern = 1  # Layer 0 is static (1 % 1 == 0)

        # Scenario 1
        hybrid_cache_static_mode = HybridCache(config=config, max_batch_size=1, max_cache_len=self.max_cache_len)
        prefill = torch.tensor([1.0, 2.0, 0.0, 0.0])[None, None, :, None]
        hybrid_cache_static_mode.update(
            key_states=prefill,
            value_states=prefill,
            layer_idx=0,
            cache_kwargs={"cache_position": torch.arange(4)},
        )
        hybrid_cache_static_mode.update(
            key_states=torch.tensor(3.0)[None, None, None, None],
            value_states=torch.tensor(3.0)[None, None, None, None],
            layer_idx=0,
            cache_kwargs={"cache_position": torch.tensor([2])},
        )
        self.assertEqual(
            hybrid_cache_static_mode.key_cache[0][0, 0, :, 0].tolist(),
            [1.0, 2.0, 3.0, 0.0],
            "HybridCache Static Scenario 1 failed",
        )

        # Scenario 2
        hybrid_cache_static_mode.update(
            key_states=torch.tensor(4.0)[None, None, None, None],
            value_states=torch.tensor(4.0)[None, None, None, None],
            layer_idx=0,
            cache_kwargs={"cache_position": torch.tensor([3])},
        )
        self.assertEqual(
            hybrid_cache_static_mode.key_cache[0][0, 0, :, 0].tolist(),
            [1.0, 2.0, 3.0, 4.0],
            "HybridCache Static Scenario 2 failed",
        )

    def test_hybrid_cache_sliding_mode(self):
        """Test HybridCache in sliding mode with hardcoded assertions.

        Scenario 1: Update within window, no slide yet
        prefill:       [1.0, 2.0, 0.0, 0.0]
        update pos 2:  [1.0, 2.0, 3.0, 0.0]

        Scenario 2: Update causing first slide
        prefill:       [1.0, 2.0, 3.0, 4.0]
        update pos 4:  [2.0, 3.0, 4.0, 5.0] (shift happens as pos > window_size-1)

        Scenario 3: Update causing subsequent slide
        update pos 5:  [3.0, 4.0, 5.0, 6.0] (shift continues)

        Scenario 4: Long prompt handling (prompt_len > window_size)
        input:         [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        result:        [3.0, 4.0, 5.0, 6.0] (keeps last window_size tokens)
        """
        # Scenario 1: Update within window, no slide yet
        hybrid_cache = HybridCache(config=self.config, max_batch_size=1, max_cache_len=self.max_cache_len)
        prefill = torch.tensor([1.0, 2.0, 0.0, 0.0])[None, None, :, None]
        hybrid_cache.update(
            key_states=prefill,
            value_states=prefill,
            layer_idx=0,
            cache_kwargs={"cache_position": torch.arange(4), "sliding_window": self.window_size},
        )
        hybrid_cache.update(
            key_states=torch.tensor(3.0)[None, None, None, None],
            value_states=torch.tensor(3.0)[None, None, None, None],
            layer_idx=0,
            cache_kwargs={"cache_position": torch.tensor([2]), "sliding_window": self.window_size},
        )
        self.assertEqual(
            hybrid_cache.key_cache[0][0, 0, :, 0].tolist(),
            [1.0, 2.0, 3.0, 0.0],
            "HybridCache Sliding Scenario 1 failed",
        )

        # Scenario 2: Update causing first slide
        hybrid_cache = HybridCache(config=self.config, max_batch_size=1, max_cache_len=self.max_cache_len)
        prefill = torch.tensor([1.0, 2.0, 3.0, 4.0])[None, None, :, None]
        hybrid_cache.update(
            key_states=prefill,
            value_states=prefill,
            layer_idx=0,
            cache_kwargs={"cache_position": torch.arange(4), "sliding_window": self.window_size},
        )
        hybrid_cache.update(
            key_states=torch.tensor(5.0)[None, None, None, None],
            value_states=torch.tensor(5.0)[None, None, None, None],
            layer_idx=0,
            cache_kwargs={"cache_position": torch.tensor([4]), "sliding_window": self.window_size},
        )
        self.assertEqual(
            hybrid_cache.key_cache[0][0, 0, :, 0].tolist(),
            [2.0, 3.0, 4.0, 5.0],
            "HybridCache Sliding Scenario 2 failed",
        )

        # Scenario 3: Update causing subsequent slide
        hybrid_cache.update(
            key_states=torch.tensor(6.0)[None, None, None, None],
            value_states=torch.tensor(6.0)[None, None, None, None],
            layer_idx=0,
            cache_kwargs={"cache_position": torch.tensor([5]), "sliding_window": self.window_size},
        )
        self.assertEqual(
            hybrid_cache.key_cache[0][0, 0, :, 0].tolist(),
            [3.0, 4.0, 5.0, 6.0],
            "HybridCache Sliding Scenario 3 failed",
        )

        # Scenario 4: Long prompt handling
        hybrid_cache = HybridCache(config=self.config, max_batch_size=1, max_cache_len=self.max_cache_len)
        long_prefill = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])[None, None, :, None]
        hybrid_cache.update(
            key_states=long_prefill,
            value_states=long_prefill,
            layer_idx=0,
            cache_kwargs={"cache_position": torch.arange(6), "sliding_window": self.window_size},
        )
        self.assertEqual(
            hybrid_cache.key_cache[0][0, 0, :, 0].tolist(),
            [3.0, 4.0, 5.0, 6.0],
            "HybridCache Sliding Scenario 4 failed",
        )
