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
        config.rope_scaling = {
            "factor": 16.0,
            "high_freq_factor": 4.0,
            "low_freq_factor": 1.0,
            "original_max_position_embeddings": 8192,
            "rope_type": "llama3",
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
    test_headmasking = False
    test_pruning = False
    fx_compatible = False
    model_tester_class = CwmModelTester

    model_split_percents = [0.5, 0.7, 0.8]

    _torch_compile_train_cls = CwmForCausalLM if is_torch_available() else None


@require_torch_accelerator
@slow
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
            self.assertEqual(layer.layer_type, expected_type)
            if expected_type == "sliding_attention":
                self.assertEqual(layer.sliding_window, model.config.sliding_window)

        prompt = "def quicksort(arr):"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            out = model(**inputs)

        expected_logits = torch.tensor(
            [
                0.5625,
                2.9531,
                9.1875,
                0.4746,
                -0.3613,
                2.2031,
                2.9844,
                1.5312,
                0.5859,
                1.5391,
                2.7500,
                3.4375,
                2.0156,
                2.1719,
                1.5469,
                2.5469,
                2.8438,
                1.8203,
                1.7188,
                1.3984,
                1.0469,
                0.1748,
                0.4453,
                0.1533,
                -0.1157,
                0.8516,
                2.2344,
                5.2188,
                1.2891,
                1.5234,
                0.8555,
                0.6992,
            ],
            dtype=torch.bfloat16,
        ).to(model.device)

        self.assertTrue(torch.allclose(out.logits[0, -1, :32], expected_logits, atol=1e-2, rtol=1e-2))

        self.assertEqual(out.logits.shape[1], inputs.input_ids.shape[1])
        self.assertEqual(out.logits.shape[2], model.config.vocab_size)
        self.assertFalse(torch.isnan(out.logits).any())
        self.assertFalse(torch.isinf(out.logits).any())

    @slow
    def test_cwm_sliding_window_long_sequence(self):
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("facebook/cwm")
        model = CwmForCausalLM.from_pretrained("facebook/cwm", device_map="auto", dtype=torch.bfloat16)

        sliding_window = model.config.sliding_window
        long_text = "for i in range(1000):\n    print(f'iteration {i}')\n" * 600

        inputs = tokenizer(long_text, return_tensors="pt").to(model.device)
        seq_len = inputs.input_ids.shape[1]

        # create a sequence longer than sliding window
        self.assertGreater(
            seq_len, sliding_window, f"Test sequence length {seq_len} should be > sliding window {sliding_window}"
        )

        with torch.no_grad():
            out = model(**inputs)

        expected_logits = torch.tensor(
            [
                4.7812,
                6.1875,
                13.1875,
                4.4062,
                5.0312,
                3.9844,
                6.6875,
                4.8438,
                2.3125,
                6.5000,
                4.4688,
                0.5195,
                5.6562,
                3.3125,
                2.7500,
                4.9062,
                5.5938,
                4.1562,
                3.9531,
                2.4062,
                3.2812,
                2.8594,
                3.4688,
                2.9688,
                2.6875,
                3.4531,
                2.7344,
                7.2812,
                4.5000,
                5.7500,
                2.3438,
                5.9688,
            ],
            dtype=torch.bfloat16,
        ).to(model.device)

        self.assertTrue(torch.allclose(out.logits[0, -1, :32], expected_logits, atol=1e-2, rtol=1e-2))

        self.assertEqual(out.logits.shape[1], seq_len)
        self.assertEqual(out.logits.shape[2], model.config.vocab_size)
        self.assertFalse(torch.isnan(out.logits).any())
        self.assertFalse(torch.isinf(out.logits).any())

        for i, layer in enumerate(model.model.layers):
            if model.config.layer_types[i] == "sliding_attention":
                self.assertEqual(layer.sliding_window, sliding_window)
