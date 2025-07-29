# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team.
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

import torch

from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, MistralForCausalLM
from transformers.generation.continuous_batching import (
    RequestStatus,
)
from transformers.testing_utils import (
    cleanup,
    require_torch_gpu,
    slow,
    torch_device,
)


class TestContinuousBatchingSlidingWindow(unittest.TestCase):
    """Test continuous batching with sliding window attention for large prompts."""

    def setUp(self):
        cleanup(torch_device, gc_collect=True)

    def tearDown(self):
        cleanup(torch_device, gc_collect=True)

    @slow
    @require_torch_gpu
    # @require_flash_attn
    def test_mixed_attn_sliding_window_large_prompt_continuous_batching(self):
        """Test continuous batching with sliding window attention on large prompts with mixed attention, depending on layers."""

        model_name = "google/gemma-2-2b"

        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto").eval()
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            model.config.pad_token_id = model.config.eos_token_id

        EXPECTED_COMPLETIONS = [
            " the people, the food, the culture, the history, the music, the art, the architecture",
            ", green, yellow, orange, purple, pink, brown, black, white, gray, silver",
        ]

        input_text = [
            "This is a nice place. " * 800 + "I really enjoy the scenery,",
            "A list of colors: red, blue",
        ]
        inputs = tokenizer(input_text, padding=True, return_tensors="pt").to(torch_device)

        input_size = inputs.input_ids.shape[-1]
        self.assertTrue(input_size > model.config.sliding_window)

        for attn_impl in ["paged_attention", "eager_paged", "sdpa_paged"]:
            with self.subTest(attn_impl=attn_impl):
                # Configure model for this attention implementation
                model.config._attn_implementation = attn_impl

                # Create generation config for continuous batching

                generation_config = GenerationConfig(
                    max_new_tokens=20,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                    use_cache=False,
                    num_blocks=128,
                    block_size=64,
                    do_sample=False,
                    max_batch_tokens=512,
                    scheduler="prefill_first",
                )

                manager = model.init_continuous_batching(generation_config=generation_config, streaming=False)

                try:
                    manager.start()

                    req_1 = manager.add_request(inputs.input_ids[0], "req_1")
                    _req_2 = manager.add_request(inputs.input_ids[1], "req_2")

                    finished_reqs = []
                    while len(finished_reqs) < len(input_text):
                        result = manager.get_result(timeout=5)
                        if result is None:
                            if not manager.is_running():
                                raise RuntimeError("Manager stopped unexpectedly")
                            else:
                                continue

                        finished_reqs.append(result)

                finally:
                    manager.stop(block=True, timeout=5.0)

                # XXX: do assertions later on to avoid bricking the GPU
                for result in finished_reqs:
                    if result.request_id == req_1:
                        output = EXPECTED_COMPLETIONS[0]
                    else:
                        output = EXPECTED_COMPLETIONS[1]
                    self.assertEqual(result.status, RequestStatus.FINISHED)
                    self.assertIsNone(result.error)

                    generated_tokens = result.generated_tokens
                    self.assertEqual(len(generated_tokens), 20)
                    output_text = tokenizer.decode(generated_tokens)

                    self.assertEqual(output_text, output)

    @slow
    @require_torch_gpu
    # @require_flash_attn
    def test_sliding_window_large_prompt_continuous_batching(self):
        """Test continuous batching with sliding window attention on large prompts."""

        model_name = "mistralai/Mistral-7B-v0.1"
        model = MistralForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto").eval()
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            model.config.pad_token_id = model.config.eos_token_id

        input_ids = [1] + [306, 338] * 2048

        EXPECTED_OUTPUT_TOKEN_IDS = [306, 338]

        for attn_impl in ["eager_paged", "sdpa_paged"]:
            with self.subTest(attn_impl=attn_impl):
                model.config._attn_implementation = attn_impl

                generation_config = GenerationConfig(
                    max_new_tokens=4,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                    use_cache=False,
                    num_blocks=128,
                    block_size=64,
                    do_sample=False,
                    max_batch_tokens=512,
                    scheduler="prefill_first",
                )

                manager = model.init_continuous_batching(generation_config=generation_config, streaming=False)

                try:
                    manager.start()

                    request_id = manager.add_request(input_ids=input_ids)

                    while True:
                        result = manager.get_result(timeout=5)
                        if result is None:
                            if not manager.is_running():
                                raise RuntimeError("Manager stopped unexpectedly")
                            continue

                        self.assertEqual(result.request_id, request_id)
                        self.assertEqual(result.status, RequestStatus.FINISHED)
                        self.assertIsNone(result.error)

                        generated_tokens = result.generated_tokens
                        self.assertEqual(len(generated_tokens), 4)

                        self.assertEqual(
                            generated_tokens[-2:],
                            EXPECTED_OUTPUT_TOKEN_IDS,
                            f"Expected {EXPECTED_OUTPUT_TOKEN_IDS}, got {generated_tokens[-2:]} with {attn_impl}",
                        )
                        break

                finally:
                    manager.stop(block=True, timeout=5.0)


if __name__ == "__main__":
    unittest.main()
