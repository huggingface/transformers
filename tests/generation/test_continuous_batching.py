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

from transformers import AutoTokenizer, GenerationConfig, MistralForCausalLM
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

    @classmethod
    def setUpClass(cls):
        cls.model_name = "mistralai/Mistral-7B-v0.1"
        cls.model = MistralForCausalLM.from_pretrained(
            cls.model_name, torch_dtype=torch.float16, device_map="auto"
        ).eval()

        cls.tokenizer = AutoTokenizer.from_pretrained(cls.model_name, use_fast=False)
        if cls.tokenizer.pad_token is None:
            cls.tokenizer.pad_token = cls.tokenizer.eos_token
            cls.model.config.pad_token_id = cls.model.config.eos_token_id

    @classmethod
    def tearDownClass(cls):
        del cls.model
        del cls.tokenizer
        cleanup(torch_device, gc_collect=True)

    def setUp(self):
        cleanup(torch_device, gc_collect=True)

    def tearDown(self):
        cleanup(torch_device, gc_collect=True)

    @slow
    @require_torch_gpu
    # @require_flash_attn
    def test_sliding_window_large_prompt_continuous_batching(self):
        """Test continuous batching with sliding window attention on large prompts."""

        input_ids = [1] + [306, 338] * 2048

        EXPECTED_OUTPUT_TOKEN_IDS = [306, 338]

        for attn_impl in ["sdpa_paged"]:  # ["flash_attention_2", "sdpa"]:
            with self.subTest(attn_impl=attn_impl):
                # Configure model for this attention implementation
                self.model.config._attn_implementation = attn_impl

                # Create generation config for continuous batching

                generation_config = GenerationConfig(
                    max_new_tokens=4,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.pad_token_id,
                    use_cache=False,
                    num_blocks=128,
                    block_size=64,
                    do_sample=False,
                    max_batch_tokens=512,
                    scheduler="prefill_first",
                )

                # Test continuous batching with sliding window
                manager = self.model.init_continuous_batching(generation_config=generation_config, streaming=False)

                try:
                    manager.start()

                    # Add the long prompt request
                    request_id = manager.add_request(input_ids=input_ids)

                    # Get the result
                    result = manager.get_result(timeout=30)

                    self.assertIsNotNone(result, f"No result received for {attn_impl}")
                    self.assertEqual(result.request_id, request_id)
                    self.assertEqual(result.status, RequestStatus.FINISHED)
                    self.assertIsNone(result.error)

                    # Verify the generated tokens match expected pattern
                    generated_tokens = result.generated_tokens
                    self.assertEqual(len(generated_tokens), 4)

                    # The last 2 tokens should match the expected pattern
                    self.assertEqual(
                        generated_tokens[-2:],
                        EXPECTED_OUTPUT_TOKEN_IDS,
                        f"Expected {EXPECTED_OUTPUT_TOKEN_IDS}, got {generated_tokens[-2:]} with {attn_impl}",
                    )

                finally:
                    manager.stop(block=True, timeout=5.0)


if __name__ == "__main__":
    unittest.main()
