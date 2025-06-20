# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch Cohere model."""

import unittest

from transformers import CohereConfig, is_torch_available
from transformers.testing_utils import (
    require_bitsandbytes,
    require_torch,
    require_torch_multi_accelerator,
    require_torch_sdpa,
    slow,
    torch_device,
)

from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester


if is_torch_available():
    import torch

    from transformers import AutoTokenizer, CohereForCausalLM, CohereModel


# Copied from transformers.tests.models.llama.LlamaModelTester with Llama->Cohere
class CohereModelTester(CausalLMModelTester):
    config_class = CohereConfig
    if is_torch_available():
        base_model_class = CohereModel
        causal_lm_class = CohereForCausalLM


@require_torch
class CohereModelTest(CausalLMModelTest, unittest.TestCase):
    all_model_classes = (CohereModel, CohereForCausalLM) if is_torch_available() else ()
    pipeline_model_mapping = (
        {
            "feature-extraction": CohereModel,
            "text-generation": CohereForCausalLM,
        }
        if is_torch_available()
        else {}
    )
    model_tester_class = CohereModelTester


@require_torch
@slow
class CohereIntegrationTest(unittest.TestCase):
    @require_torch_multi_accelerator
    @require_bitsandbytes
    def test_batched_4bit(self):
        model_id = "CohereForAI/c4ai-command-r-v01-4bit"

        EXPECTED_TEXT = [
            'Hello today I am going to show you how to make a simple and easy card using the new stamp set called "Hello" from the Occasions catalog. This set is so versatile and can be used for many occasions. I used the new In',
            "Hi there, here we are again with another great collection of free fonts for your next project. This time we have gathered 10 free fonts that you can download and use in your designs. These fonts are perfect for any kind",
        ]

        model = CohereForCausalLM.from_pretrained(model_id, device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        tokenizer.pad_token = tokenizer.eos_token

        text = ["Hello today I am going to show you how to", "Hi there, here we are"]
        inputs = tokenizer(text, return_tensors="pt", padding=True).to(torch_device)

        output = model.generate(**inputs, max_new_tokens=40, do_sample=False)
        self.assertEqual(tokenizer.batch_decode(output, skip_special_tokens=True), EXPECTED_TEXT)

    @require_torch_sdpa
    def test_batched_small_model_logits(self):
        # Since the model is very large, we created a random cohere model so that we can do a simple
        # logits check on it.
        model_id = "hf-internal-testing/cohere-random"

        EXPECTED_LOGITS = torch.Tensor(
            [
                [[0.0000, 0.0285, 0.0322], [0.0000, 0.0011, 0.1105], [0.0000, -0.0018, -0.1019]],
                [[0.0000, 0.1080, 0.0454], [0.0000, -0.1808, -0.1553], [0.0000, 0.0452, 0.0369]],
            ]
        ).to(device=torch_device, dtype=torch.float16)

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = CohereForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16).to(torch_device)

        tokenizer.pad_token = tokenizer.eos_token

        text = ["Hello today I am going to show you how to", "Hi there, here we are"]
        inputs = tokenizer(text, return_tensors="pt", padding=True).to(torch_device)

        with torch.no_grad():
            output = model(**inputs)

        logits = output.logits
        torch.testing.assert_close(EXPECTED_LOGITS, logits[:, -3:, :3], rtol=1e-3, atol=1e-3)
