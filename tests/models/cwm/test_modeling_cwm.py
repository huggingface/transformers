# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

from transformers import is_torch_available
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

    from transformers.models.cwm import (
        CwmConfig,
        CwmForCausalLM,
        CwmModel,
    )


class CwmModelTester(CausalLMModelTester):
    if is_torch_available():
        config_class = CwmConfig
        base_model_class = CwmModel
        causal_lm_class = CwmForCausalLM

    def get_config(self):
        config = super().get_config()

        config.sliding_window = 8192
        config.rope_parameters = {
            "factor": 16.0,
            "high_freq_factor": 4.0,
            "low_freq_factor": 1.0,
            "original_max_position_embeddings": 8192,
            "rope_type": "llama3",
            "rope_theta": 1000000.0,
        }

        return config


@require_torch
class CwmModelTest(CausalLMModelTest, unittest.TestCase):
    all_model_classes = (
        (
            CwmModel,
            CwmForCausalLM,
        )
        if is_torch_available()
        else ()
    )
    pipeline_model_mapping = (
        {
            "feature-extraction": CwmModel,
            "text-generation": CwmForCausalLM,
        }
        if is_torch_available()
        else {}
    )
    model_tester_class = CwmModelTester

    model_split_percents = [0.5, 0.7, 0.8]

    _torch_compile_train_cls = CwmForCausalLM if is_torch_available() else None


@require_torch_accelerator
@slow
@require_read_token
class CwmIntegrationTest(unittest.TestCase):
    def setUp(self):
        cleanup(torch_device, gc_collect=True)

    def tearDown(self):
        cleanup(torch_device, gc_collect=True)

    @slow
    def test_cwm_integration(self):
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("facebook/cwm")
        model = CwmForCausalLM.from_pretrained("facebook/cwm", device_map="auto", dtype=torch.bfloat16)

        self.assertIsNotNone(model.config.sliding_window)
        self.assertIsNotNone(model.config.layer_types)
        self.assertIn("full_attention", model.config.layer_types)
        self.assertIn("sliding_attention", model.config.layer_types)

        for i, layer in enumerate(model.model.layers):
            expected_type = model.config.layer_types[i]
            self.assertEqual(layer.attention_type, expected_type)
            if expected_type == "sliding_attention":
                self.assertEqual(layer.self_attn.sliding_window, model.config.sliding_window)

        prompt = "def quicksort(arr):"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            out = model(**inputs)

        # fmt: off
        expected_logits = torch.tensor(
            [0.5625, 2.9531, 9.1875, 0.5039, -0.3262, 2.2344, 3.0312, 1.5312, 0.5664, 1.5625, 2.7656, 3.4219, 2.0312, 2.1719, 1.5391, 2.5469, 2.8281, 1.8125, 1.7109, 1.3906, 1.0391, 0.1621, 0.4277, 0.1455, -0.1230, 0.8477, 2.2344, 5.2188, 1.2969, 1.5547, 0.8516, 0.7148],
            dtype=torch.bfloat16,
        ).to(model.device)
        # fmt: on

        torch.testing.assert_close(out.logits[0, -1, :32], expected_logits, atol=1e-2, rtol=1e-2)

        self.assertEqual(out.logits.shape[1], inputs.input_ids.shape[1])
        self.assertEqual(out.logits.shape[2], model.config.vocab_size)
        self.assertFalse(torch.isnan(out.logits).any())
        self.assertFalse(torch.isinf(out.logits).any())

    @slow
    def test_cwm_sliding_window_long_sequence(self):
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("facebook/cwm")
        # original `sliding_window` is `8192`, but it causes GPU OOM on A10
        model = CwmForCausalLM.from_pretrained(
            "facebook/cwm", device_map="auto", dtype=torch.bfloat16, sliding_window=4096
        )

        sliding_window = model.config.sliding_window
        long_text = "for i in range(1000):\n    print(f'iteration {i}')\n" * 270

        inputs = tokenizer(long_text, return_tensors="pt").to(model.device)
        seq_len = inputs.input_ids.shape[1]

        # create a sequence longer than sliding window
        self.assertGreater(
            seq_len, sliding_window, f"Test sequence length {seq_len} should be > sliding window {sliding_window}"
        )

        with torch.no_grad():
            out = model(**inputs)

        # fmt: off
        expected_logits = torch.tensor(
            [5.2812, 6.4688, 12.8125, 4.6875, 5.2500, 4.2500, 6.9688, 4.9375, 2.7656, 6.5938, 4.9688, 1.1016, 5.9375, 3.7500, 3.1094, 5.5312, 6.1250, 4.7500, 4.5312, 2.8281, 4.0625, 3.3125, 3.9219, 3.3906, 3.1406, 3.6719, 3.2031, 7.0938, 4.8750, 6.0000, 2.7188, 6.2500],
            dtype=torch.bfloat16,
        ).to(model.device)
        # fmt: on

        torch.testing.assert_close(out.logits[0, -1, :32], expected_logits, atol=1e-2, rtol=1e-2)

        logits = out.logits.to("cpu")

        self.assertEqual(logits.shape[1], seq_len)
        self.assertEqual(logits.shape[2], model.config.vocab_size)
        self.assertFalse(torch.isnan(logits).any())
        self.assertFalse(torch.isinf(logits).any())

        for i, layer in enumerate(model.model.layers):
            if model.config.layer_types[i] == "sliding_attention":
                self.assertEqual(layer.self_attn.sliding_window, sliding_window)

    @slow
    def test_cwm_generation_20_tokens(self):
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("facebook/cwm")
        model = CwmForCausalLM.from_pretrained("facebook/cwm", device_map="auto", dtype=torch.bfloat16)

        system_prompt = "You are a helpful AI assistant. You always reason before responding, using the following format:\n\n<think>\nyour internal reasoning\n</think>\nyour external response"
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "Write a simple Python function to add two numbers."},
        ]

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
            preserve_previous_think=True,
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        with torch.no_grad():
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=20,
                do_sample=False,
                temperature=1.0,
                top_p=1.0,
                pad_token_id=tokenizer.eos_token_id,
            )

        output_ids = generated_ids[0][len(model_inputs.input_ids[0]) :].tolist()
        generated_text = tokenizer.decode(output_ids, skip_special_tokens=False)

        self.assertEqual(len(output_ids), 20, "Should generate exactly 20 tokens")

        expected_token_ids = [
            33413,
            11,
            358,
            1205,
            311,
            3350,
            264,
            13325,
            734,
            430,
            11621,
            1403,
            5219,
            13,
            6914,
            596,
            1212,
            555,
            89746,
            1268,
        ]
        expected_text = "Okay, I need to write a Python function that adds two numbers. Let's start by recalling how"

        self.assertEqual(output_ids, expected_token_ids, "Generated tokens should match ground truth")
        self.assertEqual(generated_text, expected_text, "Generated text should match ground truth")
