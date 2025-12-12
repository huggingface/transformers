# Copyright 2025 The HuggingFace Team Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a clone of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import gc
import itertools
import unittest

import torch
from parameterized import parameterized

from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    CompileConfig,
    GenerationConfig,
    LogitsProcessorList,
)
from transformers.generation.continuous_batching.cache import group_layers_by_attn_type
from transformers.generation.continuous_batching.continuous_api import build_attention_mask
from transformers.testing_utils import (
    Expectations,
    require_deterministic_for_xpu,
    require_torch_accelerator,
    slow,
    torch_device,
)
from transformers.utils import is_flash_attn_2_available, is_kernels_available


def flush_memory(flush_compile: bool = True) -> None:
    gc.collect()
    # If needed, flush everything related to torch.compile
    if flush_compile:
        # Dynamo resets
        torch._dynamo.reset()
        torch._dynamo.reset_code_caches()
        if hasattr(torch._inductor, "codecache"):
            # Clear FX graph cache
            if hasattr(torch._inductor.codecache, "FxGraphCache"):
                torch._inductor.codecache.FxGraphCache.clear()
            # Clear PyCodeCache
            if hasattr(torch._inductor.codecache, "PyCodeCache"):
                torch._inductor.codecache.PyCodeCache.cache_clear()
            # Clear TritonFuture cache (for async compilation)
            if hasattr(torch._inductor.codecache, "TritonFuture"):
                if hasattr(torch._inductor.codecache.TritonFuture, "_compile_cache"):
                    torch._inductor.codecache.TritonFuture._compile_cache.clear()
    # Clear CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    elif torch.xpu.is_available():
        torch.xpu.empty_cache()
        torch.xpu.synchronize()
    gc.collect()


class ContinuousBatchingNonGenerationTest(unittest.TestCase):
    @parameterized.expand(
        [
            (None, None, "0"),
            (None, 4096, "0"),
            ("f", None, "0"),
            ("ffff", None, "0000"),
            ("sssss", 4096, "00000"),
            ("fs", 4096, "01"),
            ("ssfssf", 4096, "001221"),
            ("ssssf", 4096, "01234"),
            ("fffsffs", 4096, "0123456"),
        ]
    )
    def test_group_layers(
        self,
        layer_types_str: str | None,
        sliding_window: int | None,
        expected_groups: str,
    ) -> None:
        """Test the layer grouping algorithm of the hybrid allocator."""
        # Take a config and change the layer_types attribute to the mix we want
        config = AutoConfig.from_pretrained("HuggingFaceTB/SmolLM-1.7B")

        if layer_types_str is not None:
            layer_types = [{"f": "full_attention", "s": "sliding_window"}[char] for char in layer_types_str]
        else:
            layer_types = None
            config.num_hidden_layers = len(expected_groups)

        config.layer_types = layer_types
        config.sliding_window = sliding_window

        expected_lg = {}
        for i, group in enumerate(expected_groups):
            group = int(group)
            expected_lg[group] = expected_lg.get(group, []) + [i]
        expected_layer_groups = [expected_lg[i] for i in sorted(expected_lg.keys())]

        # Test layer groups formation
        layer_groups, group_types = group_layers_by_attn_type(config)
        self.assertEqual(
            sorted(expected_layer_groups),
            sorted(layer_groups),
            f"Test failed for: {layer_types_str = }, {sliding_window = }, {expected_layer_groups = }, {layer_groups = }",
        )

        # If layer_types is provided, check that group_types matches the type of the all layers in each group
        if layer_types is not None:
            for layer_group, group_type in zip(layer_groups, group_types):
                layer_types = [config.layer_types[i] for i in layer_group]
                self.assertEqual(layer_types, [group_type] * len(layer_types))
        # If layer_types is None, all groups should be of the same type
        else:
            for group_type in group_types:
                sliding_window = getattr(config, "sliding_window", None)
                expected_group_type = "sliding_attention" if sliding_window is not None else "full_attention"
                self.assertEqual(
                    group_type,
                    expected_group_type,
                    f"Test failed for: {layer_types_str = }, {sliding_window = }, {group_types = }",
                )

    @parameterized.expand(
        [
            ([0, 4], [0, 4], 1, ["1000", "1100", "1110", "1111"]),
            ([0, 4], [0, 4], 2, ["1000", "1100", "0110", "0011"]),
            ([0, 3], [0, 5], 1, ["11100", "11110", "11111"]),
            ([0, 3], [0, 5], 3, ["11100", "01110", "00111"]),
            ([0, 3, 6], [0, 3, 6], 1, ["100000", "110000", "111000", "000100", "000110", "000111"]),
            ([0, 3, 6], [0, 3, 6], 2, ["100000", "110000", "011000", "000100", "000110", "000011"]),
        ]
    )
    def test_attention_mask(
        self,
        cumulative_seqlens_q: list[int],
        cumulative_seqlens_k: list[int],
        sliding_window: int,  # the sliding window size, 1 means no sliding window
        str_expected_mask: list[str],  # the attention mask, broken down by line as a string of 0s and 1s
    ) -> None:
        """Tests the correctness of the attention mask used in the continuous batching API."""
        # Build expected mask
        minus_inf = torch.finfo(torch.float32).min
        expected_mask = torch.empty((cumulative_seqlens_q[-1], cumulative_seqlens_k[-1]), dtype=torch.float32)
        for i, line in enumerate(str_expected_mask):
            expected_mask[i, :] = torch.tensor([minus_inf if c == "0" else 0 for c in line])
        # Build actual mask
        actual_mask = torch.full_like(expected_mask, minus_inf)  # function modifies in place
        build_attention_mask(
            actual_mask, torch.tensor(cumulative_seqlens_q), torch.tensor(cumulative_seqlens_k), sliding_window
        )
        # Check that the actual mask matches the expected mask
        matches = (expected_mask == actual_mask).all()
        # If it doesn't match, print the masks in a readable form and fail the test
        if not matches:
            str_mask = [
                "".join("1" if x == 0 else "0" for x in token_attn_vector) for token_attn_vector in actual_mask
            ]
            str_mask = "\n".join(str_mask)
            str_expected_mask = "\n".join(str_expected_mask)
            self.fail(
                f"Test failed for: {cumulative_seqlens_q = }, {cumulative_seqlens_k = }, {sliding_window = }\n"
                f"Expected mask:\n{str_expected_mask}\n"
                f"Actual mask:\n{str_mask}"
            )


class ContinuousBatchingGenerationTest(unittest.TestCase):
    # -----------------------------------------------Parity tests----------------------------------------------- #
    #         Ensure continuous batching and non-continuous batching generation produce the same outputs         #
    # ---------------------------------------------------------------------------------------------------------- #
    @require_deterministic_for_xpu
    def _test_continuous_batching_parity(
        self,
        model_id: str,
        allow_prefix_sharing: bool,
        attn_implementation: str,
        use_cuda_graph: bool,
        use_compile: bool,
        max_new_tokens: int = 20,
    ) -> None:
        """Tests the parity between continuous batching and non-continuous batching generation."""

        # Skip the test if Flash Attention 2 is required but not available
        if attn_implementation == "flash_attention_2" and not (is_flash_attn_2_available() or is_kernels_available()):
            self.skipTest("Flash Attention 2 is not available and neither is the kernels library. Skipping test.")
        # Skip the test if cuda graph is on but the device is not CUDA
        if use_cuda_graph and torch_device != "cuda":
            self.skipTest("CUDA graph is only supported on CUDA devices. Skipping test.")

        # Prepare continuous batching inputs
        tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
        tokenizer.pad_token = tokenizer.eos_token
        user_messages = [
            "Josh decides to try flipping a house. He buys a house for $80,000 and then puts in $50,000 in repairs. This increased the value of the house by 150%. How much profit did he make?",
            "A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?",
            "A basket contains 25 oranges among which 1 is bad, 20% are unripe, 2 are sour and the rest are good. How many oranges are good?",
        ]  # fmt: skip
        chats = [[{"role": "user", "content": user_message}] for user_message in user_messages]
        tokenized = [tokenizer.apply_chat_template(chat, add_generation_prompt=True) for chat in chats]
        input_ids = [(x if isinstance(x, list) else x["input_ids"]) for x in tokenized]
        print(f"{input_ids[0] = } {type(input_ids[0]) = }")

        # Eager and SDPA implementations get a precision boost to account for the fact that an attention mask is used in
        # continuous batching but not in generate
        dtype = "auto" if attn_implementation == "flash_attention_2" else torch.float32

        # Generation with continuous batching
        model = AutoModelForCausalLM.from_pretrained(model_id, attn_implementation=attn_implementation, dtype=dtype)
        model = model.to(torch_device).eval()
        model.generation_config.max_new_tokens = max_new_tokens
        model.generation_config.do_sample = False
        model.generation_config.use_cuda_graph = use_cuda_graph
        if use_compile:
            model.generation_config.compile_config = CompileConfig(fullgraph=True, mode="default")

        # Generation with continuous batching
        continuous_batching_outputs = model.generate_batch(
            inputs=input_ids, generation_config=model.generation_config, allow_prefix_sharing=allow_prefix_sharing
        )

        # Prepare non-continuous batching inputs
        inputs = tokenizer.apply_chat_template(
            chats,
            add_generation_prompt=True,
            return_tensors="pt",
            padding=True,
            return_dict=True,
            return_attention_mask=True,
        )
        num_input_tokens = inputs.input_ids.shape[1]

        # Generation without continuous batching
        model = AutoModelForCausalLM.from_pretrained(model_id, attn_implementation=attn_implementation, dtype=dtype)
        model = model.to(torch_device).eval()
        model.generation_config.max_new_tokens = max_new_tokens
        model.generation_config.do_sample = False
        model.generation_config.use_cuda_graph = use_cuda_graph
        if use_compile:
            model.generation_config.compile_config = CompileConfig(fullgraph=True, mode="default")

        generate_outputs = model.generate(**inputs.to(torch_device), generation_config=model.generation_config)

        for i, user_message in enumerate(user_messages):
            continuous_batching_output = continuous_batching_outputs[f"req_{i}"].generated_tokens
            generate_output = generate_outputs[i][num_input_tokens:].tolist()
            while generate_output[-1] == model.generation_config.pad_token_id:
                generate_output.pop()

            if continuous_batching_output != generate_output:
                decoded_continuous_batching_output = tokenizer.decode(continuous_batching_output)
                decoded_generate_output = tokenizer.decode(generate_output)
                msg = f"Test failed for {model_id = } {allow_prefix_sharing = }, {attn_implementation = }, {use_cuda_graph = }, {use_compile = }\n"
                msg += f"User message              : {repr(user_message)}\n"
                msg += f"Continuous batching output: {repr(decoded_continuous_batching_output)}\n"
                msg += f"Generate output           : {repr(decoded_generate_output)}"
                self.fail(msg)

        del model
        flush_memory(flush_compile=use_compile)

    @parameterized.expand(
        list(
            itertools.product(
                [False, True],
                ["eager", "sdpa", "flash_attention_2"],
                [False, True],
                [False, True],
            )
        )
    )
    @require_torch_accelerator
    @slow
    def test_continuous_batching_config_combinations(
        self,
        allow_prefix_sharing: bool,
        attn_implementation: str,
        use_cuda_graph: bool,
        use_compile: bool,
    ) -> None:
        model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        self._test_continuous_batching_parity(
            model_id, allow_prefix_sharing, attn_implementation, use_cuda_graph, use_compile
        )

    # FIXME: Qwen2.5-0.5B-Instruct is not here because it's  broken (it uses a repetition penalty logits processor)
    # TODO: replace gemma2 with a tiny version of GPT-OSS? That way we can test sliding window AND attention sink

    @parameterized.expand(
        list(
            itertools.product(
                ["TinyLlama/TinyLlama-1.1B-Chat-v1.0", "google/gemma-2-2b-it"],
                [False, True],
                [False, True],
            )
        )
    )
    @require_torch_accelerator
    @slow
    def test_continuous_batching_diverse_models(self, model_id: str, use_cuda_graph: bool, use_compile: bool) -> None:
        try:
            self._test_continuous_batching_parity(model_id, True, "flash_attention_2", use_cuda_graph, use_compile)
        finally:
            flush_memory(flush_compile=use_compile)

    @require_torch_accelerator
    def test_continuous_batching_fast(self) -> None:
        model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        self._test_continuous_batching_parity(model_id, False, "sdpa", False, False)

    @require_torch_accelerator
    def test_continuous_batching_long_generate(self) -> None:
        model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        self._test_continuous_batching_parity(model_id, True, "flash_attention_2", True, True, max_new_tokens=80)

    # ---------------------------------------Streaming tests--------------------------------------- #
    #           Ensures the requests have the right behavior with and without streaming             #
    # --------------------------------------------------------------------------------------------- #
    def _test_streaming_or_not_request(self, with_streaming: bool, with_non_streaming: bool) -> None:
        model_id = "Qwen/Qwen2.5-0.5B-Instruct"
        max_new_tokens = 3

        model = AutoModelForCausalLM.from_pretrained(model_id)
        manager = model.init_continuous_batching()
        manager.logit_processor = LogitsProcessorList()
        manager.start()

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        messages = [{"content": "What is the Transformers library known for?", "role": "user"}]
        inputs = tokenizer.apply_chat_template(
            messages, return_tensors="pt", add_generation_prompt=True, return_dict=False
        ).to(model.device)[0]

        # Test with non-streaming
        if with_non_streaming:
            request_id = manager.add_request(inputs, max_new_tokens=max_new_tokens, streaming=False)

            # In non-streaming mode, the total number of generated tokens is equal to the max new tokens
            chunk = next(manager.request_id_iter(request_id))
            self.assertEqual(len(chunk.generated_tokens), max_new_tokens)

        # Test with streaming
        if with_streaming:
            request_id = manager.add_request(inputs, max_new_tokens=max_new_tokens, streaming=True)

            # In streaming mode, the total number of generated tokens is incremented by 1 on each iteration
            chunk_1 = next(manager.request_id_iter(request_id))
            self.assertEqual(len(chunk_1.generated_tokens), 1)

            chunk_2 = next(manager.request_id_iter(request_id))
            self.assertEqual(len(chunk_2.generated_tokens), 2)

            chunk_3 = next(manager.request_id_iter(request_id))
            self.assertEqual(len(chunk_3.generated_tokens), 3)

        manager.stop(block=True)

    @require_torch_accelerator
    def test_streaming_request(self) -> None:
        self._test_streaming_or_not_request(with_streaming=True, with_non_streaming=False)

    @require_torch_accelerator
    def test_non_streaming_request(self) -> None:
        self._test_streaming_or_not_request(with_streaming=False, with_non_streaming=True)

    @require_torch_accelerator
    def test_streaming_and_non_streaming_requests_can_alternate(self) -> None:
        self._test_streaming_or_not_request(with_streaming=True, with_non_streaming=True)

    # -----------------------------------------Misc. tests----------------------------------------- #
    #                     Various tests that don't fit into the other categories                    #
    # --------------------------------------------------------------------------------------------- #
    @require_torch_accelerator
    def test_prefix_sharing(self) -> None:
        model_id = "Qwen/Qwen2.5-0.5B-Instruct"
        max_new_tokens = 32

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id)

        generation_config = GenerationConfig(do_sample=False, block_size=32)
        with model.continuous_batching_context_manager(generation_config=generation_config) as manager:
            manager.logit_processor = LogitsProcessorList()

            # Create a request with at least 32 tokens but less than 64 so prefill only generates one complete block
            messages = [{"content": "What is the Transformers library known for?", "role": "user"}]

            inputs = tokenizer.apply_chat_template(
                messages, return_tensors="pt", add_generation_prompt=True, return_dict=False
            )
            inputs = inputs.to(model.device)[0].tolist()
            self.assertGreaterEqual(len(inputs), 32, f"Input length is {len(inputs)} instead of at least 32")
            self.assertLess(len(inputs), 64, f"Input length is {len(inputs)} instead of less than 64")

            # First request, which populates the cache with a complete block
            request_id = manager.add_request(inputs, max_new_tokens=max_new_tokens)
            chunk_no_reuse = next(manager.request_id_iter(request_id))

            hash_table = manager.batch_processor.cache._block_manager._hash_to_id
            self.assertEqual(
                len(hash_table),
                2,
                f"There should be 2 blocks, one for the prefill and one for the decode, but {len(hash_table) = }",
            )
            total_prefix_length = manager.batch_processor.cache._total_prefix_length
            self.assertEqual(
                total_prefix_length, 0, f"Expected total prefix length to be 0, got {total_prefix_length}"
            )

            # Second request, which should reuse the same block
            request_id = manager.add_request(inputs, max_new_tokens=max_new_tokens)
            chunk_with_reuse = next(manager.request_id_iter(request_id))

            # There should only still be two blocks in the hash table because of block reuse
            self.assertEqual(
                len(hash_table),
                2,
                f"Because of block reuse, there should still be two blocks in the hash table, but {len(hash_table) = }",
            )

            # Check that the whole prefill was matched
            total_prefix_length = manager.batch_processor.cache._total_prefix_length
            self.assertEqual(
                total_prefix_length, 32, f"Expected total prefix length to be 32, got {total_prefix_length}"
            )

        # Check the outputs were the same
        self.assertEqual(chunk_no_reuse.generated_tokens, chunk_with_reuse.generated_tokens)

        # As an additional sanity check, we also compare to the generated tokens when prefix sharing is disabled
        expected_generated_tokens = Expectations({
            ("rocm", (9, 4)): [785, 80532, 6733, 374, 3881, 369, 1181, 5726, 311, 1855, 323, 36635, 3460, 12934, 4128, 4119, 11, 2670, 1846, 429, 646, 6923, 1467, 11, 14683, 1467, 11, 323, 2736, 1008, 4128, 13904],
        }).get_expectation()  # fmt: skip
        self.assertEqual(chunk_no_reuse.generated_tokens, expected_generated_tokens)
