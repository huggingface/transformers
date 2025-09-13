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
from typing import Optional

import torch
from parameterized import parameterized

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.generation.continuous_batching.cache import group_layers_by_attn_type
from transformers.testing_utils import Expectations, require_kernels, require_torch_gpu, slow


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
        layer_types_str: Optional[str],
        sliding_window: Optional[int],
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
        model = model.cuda().eval()
        model.generation_config.max_new_tokens = 40
        model.generation_config.do_sample = False
        model.generation_config.use_cuda_graph = False

        cb_outputs = model.generate_batch(inputs=batched_inputs, generation_config=model.generation_config)

        # Generation without continuous batching
        if attn_implementation == "sdpa_paged":
            non_cb_attn_implementation = "sdpa"
        elif attn_implementation == "eager_paged":
            non_cb_attn_implementation = "eager"
        elif attn_implementation == "paged_attention|kernels-community/flash-attn":
            non_cb_attn_implementation = "eager"
        else:
            raise ValueError(f"Invalid attention implementation: {attn_implementation}")

        # We regenerate the model because just changing the attn_implementation does not work
        model = AutoModelForCausalLM.from_pretrained(
            model_id, attn_implementation=non_cb_attn_implementation, dtype="auto"
        )
        model = model.cuda().eval()
        model.generation_config.max_new_tokens = 40
        model.generation_config.do_sample = False
        model.generation_config.use_cuda_graph = False

        for request_id, request in cb_outputs.items():
            # Generate without continuous batching
            input_ids = torch.tensor([request.prompt_ids]).cuda()
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
    @require_torch_gpu
    @slow
    def test_continuous_batching_parity_llama_eager(self) -> None:
        expected_outputs = Expectations({
            ("rocm", (9, 4)): {
                "req_0": " $16. How did I get that answer? I used the following equation: 16 - 3 - 4 = 9. 9 x $2 = $18. $18 -"
            },
            ("cuda", (9, 0)): {
                "req_1": " 3 bolts of blue fiber and 1.5 bolts of white fiber. The total number of bolts is 4.5. The total number of bolts is 4.5. The total",
                "req_2": " $50,000. This is because the value of the house increased by 150%, which means that the value of the house increased by $50,000. This is because the value of the"
            }
        }).get_expectation()  # fmt: skip
        self._continuous_batching_parity("meta-llama/Llama-3.1-8B", "eager_paged", expected_outputs)

    @require_torch_gpu
    @slow
    def test_continuous_batching_parity_gemma_eager(self) -> None:
        expected_outputs = Expectations({
            ("rocm", (9, 4)): {
                "req_1": " \n\n**Answer:** 3 bolts\n\n**Solution:**\n\n* **White fiber:** The robe needs half as much white fiber as blue fiber, so it needs 2 bolts / 2 ="
            },
            ("cuda", (9, 0)): {
                "req_0": "\n\n**$12**\n\n**Here's how to solve it:**\n\n* **Eggs eaten:** 3\n* **Eggs left:** 16 - 3 = 13",
                "req_1": " \n \n 2 + 1 = 3 bolts \n \n \n \n \n \n \n \n \n \n \n \n \n "
            }
        }).get_expectation()  # fmt: skip
        self._continuous_batching_parity("google/gemma-2-2b-it", "eager_paged", expected_outputs)

    @require_torch_gpu
    @slow
    def test_continuous_batching_parity_qwen_eager(self) -> None:
        expected_outputs = {}
        self._continuous_batching_parity("Qwen/Qwen3-4B-Instruct-2507", "eager_paged", expected_outputs)

    @require_torch_gpu
    @slow
    def test_continuous_batching_parity_gpt_oss_eager(self) -> None:
        expected_outputs = Expectations({
            ("cuda", (9, 0)): {
                "req_1": " 2.5 bolts. The question: \"What is the name of the puzzle that involves a robe taking 2 bolts of blue fiber and half that much white fiber?\" The answer: \"The",
                "req_2": " 50%.\"\n\nWe need to parse: He buys a house for $80,000. He puts in $50,000 in repairs. This increased the value of the house by 150%."
            }
        }).get_expectation()  # fmt: skip
        self._continuous_batching_parity("openai/gpt-oss-20b", "eager_paged", expected_outputs)

    # SDPA tests
    @require_torch_gpu
    @slow
    def test_continuous_batching_parity_llama_sdpa(self) -> None:
        expected_outputs = Expectations({
            ("rocm", (9, 4)): {
                "req_2": " $50,000. This is because the value of the house increased by 150%, which means that the value of the house increased by $50,000. This is because the value of the"
            }
        }).get_expectation()  # fmt: skip
        self._continuous_batching_parity("meta-llama/Llama-3.1-8B", "sdpa_paged", expected_outputs)

    @require_torch_gpu
    @slow
    def test_continuous_batching_parity_gemma_sdpa(self) -> None:
        expected_outputs = Expectations({
            ("cuda", (9, 0)): {
                "req_1": " \n\n**Answer:** 3 bolts\n\n**Solution:**\n\n* **White fiber:** The robe needs half as much white fiber as blue fiber, so it needs 2 bolts / 2 =",
            }
        }).get_expectation()  # fmt: skip
        self._continuous_batching_parity("google/gemma-2-2b-it", "sdpa_paged", expected_outputs)

    @require_torch_gpu
    @slow
    def test_continuous_batching_parity_qwen_sdpa(self) -> None:
        expected_outputs = {}
        self._continuous_batching_parity("Qwen/Qwen3-4B-Instruct-2507", "sdpa_paged", expected_outputs)

    # GPT-OSS is not compatible with SDPA because it has an attention sink. TODO: is this fixable?

    # Flash attention test
    @require_torch_gpu
    @require_kernels
    @slow
    def test_continuous_batching_parity_llama_flash(self) -> None:
        expected_outputs = Expectations({
            ("cuda", (9, 0)): {
                "req_1": " 3 bolts of blue fiber and 1.5 bolts of white fiber. The total number of bolts is 4.5 bolts. The total number of bolts is 4.5 bolts.",
            }
        }).get_expectation()  # fmt: skip
        self._continuous_batching_parity(
            "meta-llama/Llama-3.1-8B", "paged_attention|kernels-community/flash-attn", expected_outputs
        )

    @require_torch_gpu
    @require_kernels
    @slow
    def test_continuous_batching_parity_gemma_flash(self) -> None:
        expected_outputs = Expectations({
            ("cuda", (9, 0)): {
                "req_1": " \n \n 2 + 1 = 3 bolts \n \n \n \n \n \n \n \n \n \n \n \n \n ",
            }
        }).get_expectation()  # fmt: skip
        self._continuous_batching_parity(
            "google/gemma-2-2b-it", "paged_attention|kernels-community/flash-attn", expected_outputs
        )

    @require_torch_gpu
    @require_kernels
    @slow
    def test_continuous_batching_parity_qwen_flash(self) -> None:
        expected_outputs = {}
        self._continuous_batching_parity(
            "Qwen/Qwen3-4B-Instruct-2507", "paged_attention|kernels-community/flash-attn", expected_outputs
        )

    @require_torch_gpu
    @require_kernels
    @slow
    def test_continuous_batching_parity_gpt_oss_flash(self) -> None:
        expected_outputs = {}
        self._continuous_batching_parity(
            "openai/gpt-oss-20b", "paged_attention|kernels-community/flash-attn", expected_outputs
        )


# FIXME: the gemma test seem broken, there is a message about cuda graphs and the sdpa and flash expecteations are
# inverted on CUDA. On AMD they do fine.
