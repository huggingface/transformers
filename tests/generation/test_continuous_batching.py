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

import unittest

import torch
from parameterized import parameterized

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, GenerationConfig, LogitsProcessorList
from transformers.generation.continuous_batching.cache import group_layers_by_attn_type
from transformers.generation.continuous_batching.continuous_api import build_attention_mask
from transformers.testing_utils import (
    Expectations,
    require_kernels,
    require_read_token,
    require_torch_accelerator,
    slow,
    torch_device,
)


ALLOW_EXPECTED_OUTPUTS = True  # this is a debug flag when you want to measure deviation between CB and non-CB gen


class ContinuousBatchingTest(unittest.TestCase):
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

    def _continuous_batching_parity(
        self, model_id: str, attn_implementation: str, expected_outputs: dict[str, str]
    ) -> None:
        # Prepare common elements
        tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
        prompts = [
            "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her "
                "friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh "
                "duck egg. How much in dollars does she make every day at the farmers' market? The answer is:",
            "A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take? "
                "The answer is:",
            "Josh decides to try flipping a house. He buys a house for $80,000 and then puts in $50,000 in repairs. "
                "This increased the value of the house by 150%. How much profit did he make? The answer is:",
        ]  # fmt: skip
        batched_inputs = [tokenizer.encode(prompt) for prompt in prompts]

        # Generation with continuous batching
        model = AutoModelForCausalLM.from_pretrained(model_id, attn_implementation=attn_implementation, dtype="auto")
        model = model.to(torch_device).eval()
        model.generation_config.max_new_tokens = 40
        model.generation_config.do_sample = False
        model.generation_config.use_cuda_graph = False

        cb_outputs = model.generate_batch(inputs=batched_inputs, generation_config=model.generation_config)

        # Generation without continuous batching
        if attn_implementation == "paged|sdpa":
            non_cb_attn_implementation = "sdpa"
        elif attn_implementation == "paged|eager":
            non_cb_attn_implementation = "eager"
        elif attn_implementation == "paged|flash_attention_2":
            non_cb_attn_implementation = "eager"
        else:
            raise ValueError(f"Invalid attention implementation: {attn_implementation}")

        # We regenerate the model because just changing the attn_implementation does not work
        model = AutoModelForCausalLM.from_pretrained(
            model_id, attn_implementation=non_cb_attn_implementation, dtype="auto"
        )
        model = model.to(torch_device).eval()
        model.generation_config.max_new_tokens = 40
        model.generation_config.do_sample = False
        model.generation_config.use_cuda_graph = False

        for request_id, request in cb_outputs.items():
            # Generate without continuous batching
            input_ids = torch.tensor([request.prompt_ids]).to(torch_device)
            attention_mask = torch.ones_like(input_ids)
            outputs = model.generate(
                input_ids, attention_mask=attention_mask, generation_config=model.generation_config
            )
            generated_tokens = outputs[0][input_ids.shape[1] :]
            non_cb_decoded_output = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            input_ids = input_ids.tolist()[0]

            # Check that the generated output with and without CB match
            cb_decoded_output = tokenizer.decode(request.generated_tokens, skip_special_tokens=True)
            outputs_match = non_cb_decoded_output == cb_decoded_output

            # If they dont, that might be expected: the outputs can differ slightly due to numerical differences
            # If that's the case, there is an expected output ready
            if not outputs_match:
                expected_output = expected_outputs.get(request_id) if ALLOW_EXPECTED_OUTPUTS else None

                if expected_output is None:
                    self.fail(
                        f"Test {request_id = } failed, no expected output was provided.\nRef:"
                        f"{repr(non_cb_decoded_output)}\nOut:{repr(cb_decoded_output)}"
                    )
                else:
                    self.assertEqual(
                        expected_output,
                        cb_decoded_output,
                        msg=f"Test {request_id = } failed, expected output did not match.\n"
                        f"Exp:{repr(expected_output)}\nOut:{repr(cb_decoded_output)}",
                    )

    # Eager tests
    @require_torch_accelerator
    @require_read_token
    @slow
    def test_continuous_batching_parity_llama_eager(self) -> None:
        expected_outputs = Expectations({
            ("rocm", (9, 4)): {
                "req_0": " $16. How did I get that answer? I used the following equation: 16 - 3 - 4 = 9. 9 x $2 = $18. $18 -"
            },
            ("cuda", (9, 0)): {
                "req_1": " 3 bolts of blue fiber and 1.5 bolts of white fiber. The total number of bolts is 4.5. The total number of bolts is 4.5. The total",
                "req_2": " $50,000. This is because the value of the house increased by 150%, which means that the value of the house increased by $50,000. This is because the value of the"
            },
            ("xpu", None): {
                "req_1": " 3 bolts of blue fiber and 1.5 bolts of white fiber. The answer is not 3.5 bolts of blue fiber and 1.5 bolts of white fiber. The answer'",
                "req_2": " $50,000. This is because the value of the house increased by 150%, which means that the value of the house increased by $50,000. This is because the value of the"
            },
        }).get_expectation()  # fmt: skip
        self._continuous_batching_parity("meta-llama/Llama-3.1-8B", "paged|eager", expected_outputs)

    @require_torch_accelerator
    @slow
    def test_continuous_batching_parity_gemma_eager(self) -> None:
        expected_outputs = Expectations({
            ("rocm", (9, 4)): {
                "req_1": " \n\n**Answer:** 3 bolts\n\n**Solution:**\n\n* **White fiber:** The robe needs half as much white fiber as blue fiber, so it needs 2 bolts / 2 ="
            },
            ("cuda", (9, 0)): {
                "req_0": "\n\n**$12**\n\n**Here's how to solve it:**\n\n* **Eggs eaten:** 3\n* **Eggs left:** 16 - 3 = 13",
                "req_1": " \n \n 2 + 1 = 3 bolts \n \n \n \n \n \n \n \n \n \n \n \n \n "
            },
            ("xpu", None): {
                "req_0": "\n\n**$12**\n\n**Here's how to solve it:**\n\n* **Eggs eaten:** 3\n* **Eggs left:** 16 - 3 = 13",
                "req_1": " \n \n 2 + 1 = 3 bolts \n \n \n \n \n \n \n \n \n \n \n \n \n ",
                "req_2": "\n\n**$100,000**\n\n**Explanation:**\n\nHere's how to calculate the profit:\n\n1. **Calculate the total cost:** $80,00",
            },
        }).get_expectation()  # fmt: skip
        self._continuous_batching_parity("google/gemma-2-2b-it", "paged|eager", expected_outputs)

    # FIXME: set expected_outputs
    # @require_torch_accelerator
    # @slow
    # def test_continuous_batching_parity_qwen_eager(self) -> None:
    #     expected_outputs = {}
    #     self._continuous_batching_parity("Qwen/Qwen3-4B-Instruct-2507", "paged|eager", expected_outputs)

    # FIXME: OOMs
    # @require_torch_accelerator
    # @slow
    # def test_continuous_batching_parity_gpt_oss_eager(self) -> None:
    #     expected_outputs = Expectations({
    #         ("cuda", (9, 0)): {
    #             "req_1": " 2.5 bolts. The question: \"What is the name of the puzzle that involves a robe taking 2 bolts of blue fiber and half that much white fiber?\" The answer: \"The",
    #             "req_2": " 50%.\"\n\nWe need to parse: He buys a house for $80,000. He puts in $50,000 in repairs. This increased the value of the house by 150%."
    #         },
    #         ("xpu", None): {
    #             "req_1": " 2.5 bolts. The question: \"What is the name of the puzzle that involves a robe taking 2 bolts of blue fiber and half that much white fiber?\" The answer: \"The",
    #             "req_2": " 50%.\"\n\nWe need to parse: He buys a house for $80,000. He puts in $50,000 in repairs. This increased the value of the house by 150%."
    #         },
    #     }).get_expectation()  # fmt: skip
    #     self._continuous_batching_parity("openai/gpt-oss-20b", "paged|eager", expected_outputs)

    # SDPA tests
    @require_read_token
    @require_torch_accelerator
    @slow
    def test_continuous_batching_parity_llama_sdpa(self) -> None:
        expected_outputs = Expectations({
            ("rocm", (9, 4)): {
                "req_2": " $50,000. This is because the value of the house increased by 150%, which means that the value of the house increased by $50,000. This is because the value of the"
            },
            ("xpu", None): {
                "req_2": " $50,000. This is because the value of the house increased by 150%, which means that the value of the house increased by $50,000. This is because the value of the"
            },
        }).get_expectation()  # fmt: skip
        self._continuous_batching_parity("meta-llama/Llama-3.1-8B", "paged|sdpa", expected_outputs)

    @require_torch_accelerator
    @slow
    def test_continuous_batching_parity_gemma_sdpa(self) -> None:
        expected_outputs = Expectations({
            ("cuda", (9, 0)): {
                "req_1": " \n\n**Answer:** 3 bolts\n\n**Solution:**\n\n* **White fiber:** The robe needs half as much white fiber as blue fiber, so it needs 2 bolts / 2 =",
            },
            ("xpu", None): {
                "req_1": " \n\n**Answer:** 3 bolts\n\n**Solution:**\n\n* **White fiber:** The robe needs half as much white fiber as blue fiber, so it needs 2 bolts / 2 =",
            },
        }).get_expectation()  # fmt: skip
        self._continuous_batching_parity("google/gemma-2-2b-it", "paged|sdpa", expected_outputs)

    # FIXME: set expected_outputs
    # @require_torch_accelerator
    # @slow
    # def test_continuous_batching_parity_qwen_sdpa(self) -> None:
    #     expected_outputs = {}
    #     self._continuous_batching_parity("Qwen/Qwen3-4B-Instruct-2507", "paged|sdpa", expected_outputs)

    # GPT-OSS is not compatible with SDPA because it has an attention sink. TODO: is this fixable?

    # Flash attention test
    @require_torch_accelerator
    @require_kernels
    @slow
    def test_continuous_batching_parity_llama_flash(self) -> None:
        expected_outputs = Expectations({
            ("cuda", (9, 0)): {
                "req_1": " 3 bolts of blue fiber and 1.5 bolts of white fiber. The total number of bolts is 4.5 bolts. The total number of bolts is 4.5 bolts.",
            },
            ("xpu", None): {
                "req_1": " 3 bolts of blue fiber and 1.5 bolts of white fiber. The total number of bolts is 4.5 bolts. The total number of bolts is 4.5 bolts.",
            },
        }).get_expectation()  # fmt: skip
        self._continuous_batching_parity("meta-llama/Llama-3.1-8B", "paged|flash_attention_2", expected_outputs)

    @require_torch_accelerator
    @require_kernels
    @slow
    def test_continuous_batching_parity_gemma_flash(self) -> None:
        expected_outputs = Expectations({
            ("cuda", (9, 0)): {
                "req_1": " \n \n 2 + 1 = 3 bolts \n \n \n \n \n \n \n \n \n \n \n \n \n ",
            },
            ("xpu", None): {
                "req_0": "\n\n**$128**\n\n**Here's how to solve it:**\n\n* **Eggs eaten:** 3\n* **Eggs left:** 16 - 3 = 1",
                "req_1":  "\n\n**Answer:** 3 bolts\n\n**Solution:**\n\n* **White fiber:** The robe needs half as much white fiber as blue fiber, so it needs 2 bolts / 2 =",
            },
        }).get_expectation()  # fmt: skip
        self._continuous_batching_parity("google/gemma-2-2b-it", "paged|flash_attention_2", expected_outputs)

    @require_torch_accelerator
    @require_kernels
    @slow
    def test_continuous_batching_parity_qwen_flash(self) -> None:
        expected_outputs = Expectations({
            ("xpu", None): {
                "req_1":  " 3.5 bolts.\n\nLet's break it down step by step:\n\n- Blue fiber: 2 bolts\n- White fiber: half of 2 bolts = 1 bolt\n\nTotal = ",
            },
        }).get_expectation()  # fmt: skip
        self._continuous_batching_parity("Qwen/Qwen3-4B-Instruct-2507", "paged|flash_attention_2", expected_outputs)

    @require_torch_accelerator
    @require_kernels
    @slow
    def test_continuous_batching_parity_gpt_oss_flash(self) -> None:
        expected_outputs = {}
        self._continuous_batching_parity("openai/gpt-oss-20b", "paged|flash_attention_2", expected_outputs)

    def test_attn_implementation(self) -> None:
        model = AutoModelForCausalLM.from_pretrained("gpt2")
        manager = model.init_continuous_batching()
        assert "paged|sdpa" == manager.model.config._attn_implementation

        model = AutoModelForCausalLM.from_pretrained("gpt2", _attn_implementation="eager")
        manager = model.init_continuous_batching()
        assert "paged|eager" == manager.model.config._attn_implementation

    @require_torch_accelerator
    def test_streaming_request(self) -> None:
        model_id = "Qwen/Qwen2.5-0.5B-Instruct"
        max_new_tokens = 3

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")

        manager = model.init_continuous_batching()
        manager.logit_processor = LogitsProcessorList()
        manager.start()

        messages = [{"content": "What is the Transformers library known for?", "role": "user"}]

        inputs = tokenizer.apply_chat_template(
            messages, return_tensors="pt", add_generation_prompt=True, return_dict=False
        ).to(model.device)[0]

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
    def test_non_streaming_request(self) -> None:
        model_id = "Qwen/Qwen2.5-0.5B-Instruct"
        max_new_tokens = 3

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")

        manager = model.init_continuous_batching()
        manager.logit_processor = LogitsProcessorList()
        manager.start()

        messages = [{"content": "What is the Transformers library known for?", "role": "user"}]

        inputs = tokenizer.apply_chat_template(
            messages, return_tensors="pt", add_generation_prompt=True, return_dict=False
        ).to(model.device)[0]

        request_id = manager.add_request(inputs, max_new_tokens=max_new_tokens, streaming=False)

        chunk = next(manager.request_id_iter(request_id))

        # In non-streaming mode, the total number of generated tokens is equal to the max new tokens
        self.assertEqual(len(chunk.generated_tokens), max_new_tokens)

        manager.stop(block=True)

    @require_torch_accelerator
    def test_streaming_and_non_streaming_requests_can_alternate(self) -> None:
        model_id = "Qwen/Qwen2.5-0.5B-Instruct"
        max_new_tokens = 3

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")

        manager = model.init_continuous_batching()
        manager.logit_processor = LogitsProcessorList()
        manager.start()

        messages = [{"content": "What is the Transformers library known for?", "role": "user"}]

        inputs = tokenizer.apply_chat_template(
            messages, return_tensors="pt", add_generation_prompt=True, return_dict=False
        ).to(model.device)[0]

        # Non-streaming request
        request_id = manager.add_request(inputs, max_new_tokens=max_new_tokens, streaming=False)
        chunk = next(manager.request_id_iter(request_id))
        self.assertEqual(len(chunk.generated_tokens), max_new_tokens)

        # Streaming request works afterward
        request_id = manager.add_request(inputs, max_new_tokens=max_new_tokens, streaming=True)

        chunk_1 = next(manager.request_id_iter(request_id))
        self.assertEqual(len(chunk_1.generated_tokens), 1)

        chunk_2 = next(manager.request_id_iter(request_id))
        self.assertEqual(len(chunk_2.generated_tokens), 2)

        chunk_3 = next(manager.request_id_iter(request_id))
        self.assertEqual(len(chunk_3.generated_tokens), 3)

        manager.stop(block=True)

    @require_torch_accelerator
    def test_prefix_sharing(self) -> None:
        model_id = "Qwen/Qwen2.5-0.5B-Instruct"
        max_new_tokens = 32

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")

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


# FIXME: the gemma test seem broken, there is a message about cuda graphs and the sdpa and flash expecteations are
# inverted on CUDA. On AMD they do fine.
