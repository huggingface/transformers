# Copyright 2025 Bytedance-Seed Ltd and the HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch SeedOss model."""

import unittest

import pytest

from transformers import AutoModelForCausalLM, AutoTokenizer, SeedOssConfig, is_torch_available
from transformers.testing_utils import (
    cleanup,
    require_flash_attn,
    require_torch,
    require_torch_large_accelerator,
    require_torch_large_gpu,
    slow,
    torch_device,
)

from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester


if is_torch_available():
    import torch

    from transformers import (
        SeedOssForCausalLM,
        SeedOssForQuestionAnswering,
        SeedOssForSequenceClassification,
        SeedOssForTokenClassification,
        SeedOssModel,
    )


class SeedOssModelTester(CausalLMModelTester):
    if is_torch_available():
        config_class = SeedOssConfig
        base_model_class = SeedOssModel
        causal_lm_class = SeedOssForCausalLM
        sequence_classification_class = SeedOssForSequenceClassification
        token_classification_class = SeedOssForTokenClassification
        question_answering_class = SeedOssForQuestionAnswering


@require_torch
class SeedOssModelTest(CausalLMModelTest, unittest.TestCase):
    model_tester_class = SeedOssModelTester
    all_model_classes = (
        (
            SeedOssModel,
            SeedOssForCausalLM,
            SeedOssForSequenceClassification,
            SeedOssForTokenClassification,
            SeedOssForQuestionAnswering,
        )
        if is_torch_available()
        else ()
    )
    pipeline_model_mapping = (
        {
            "feature-extraction": SeedOssModel,
            "text-classification": SeedOssForSequenceClassification,
            "token-classification": SeedOssForTokenClassification,
            "text-generation": SeedOssForCausalLM,
            "zero-shot": SeedOssForSequenceClassification,
        }
        if is_torch_available()
        else {}
    )
    test_headmasking = False
    test_pruning = False
    _is_stateful = True
    model_split_percents = [0.5, 0.6]


@slow
@require_torch_large_accelerator
class SeedOssIntegrationTest(unittest.TestCase):
    input_text = ["How to make pasta?", "Hi ByteDance-Seed"]
    model_id = "ByteDance-Seed/Seed-OSS-36B-Base"

    def tearDown(self):
        cleanup(torch_device, gc_collect=True)

    def test_model_36b_fp16(self):
        EXPECTED_TEXTS = [
            "How to make pasta?\nHow to make pasta?\nPasta is a popular dish that is enjoyed by people all over",
            "Hi ByteDance-Seed team,\nI am trying to run the code on my local machine. I have installed all the",
        ]

        model = AutoModelForCausalLM.from_pretrained(self.model_id, torch_dtype=torch.float16, device_map="auto")

        tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        inputs = tokenizer(self.input_text, return_tensors="pt", padding=True, return_token_type_ids=False).to(
            model.model.embed_tokens.weight.device
        )

        output = model.generate(**inputs, max_new_tokens=20, do_sample=False)
        output_text = tokenizer.batch_decode(output, skip_special_tokens=True)

        self.assertEqual(output_text, EXPECTED_TEXTS)

    def test_model_36b_bf16(self):
        EXPECTED_TEXTS = [
            "How to make pasta?\nHow to make pasta?\nPasta is a popular dish that is enjoyed by people all over",
            "Hi ByteDance-Seed team,\nI am trying to run the code on my local machine. I have installed all the",
        ]

        model = AutoModelForCausalLM.from_pretrained(self.model_id, torch_dtype=torch.bfloat16, device_map="auto")

        tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        inputs = tokenizer(self.input_text, return_tensors="pt", padding=True).to(
            model.model.embed_tokens.weight.device
        )

        output = model.generate(**inputs, max_new_tokens=20, do_sample=False)
        output_text = tokenizer.batch_decode(output, skip_special_tokens=True)

        self.assertEqual(output_text, EXPECTED_TEXTS)

    def test_model_36b_eager(self):
        EXPECTED_TEXTS = ""

        model = AutoModelForCausalLM.from_pretrained(
            self.model_id, torch_dtype=torch.bfloat16, attn_implementation="eager", device_map="auto"
        )

        tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        inputs = tokenizer(self.input_text, return_tensors="pt", padding=True).to(
            model.model.embed_tokens.weight.device
        )

        output = model.generate(**inputs, max_new_tokens=20, do_sample=False)
        output_text = tokenizer.batch_decode(output, skip_special_tokens=True)

        self.assertEqual(output_text, EXPECTED_TEXTS)

    def test_model_36b_sdpa(self):
        EXPECTED_TEXTS = [
            "How to make pasta?\nHow to make pasta?\nPasta is a popular dish that is enjoyed by people all over",
            "Hi ByteDance-Seed team,\nI am trying to run the code on my local machine. I have installed all the",
        ]

        model = AutoModelForCausalLM.from_pretrained(
            self.model_id, torch_dtype=torch.bfloat16, attn_implementation="sdpa", device_map="auto"
        )

        tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        inputs = tokenizer(self.input_text, return_tensors="pt", padding=True).to(
            model.model.embed_tokens.weight.device
        )

        output = model.generate(**inputs, max_new_tokens=20, do_sample=False)
        output_text = tokenizer.batch_decode(output, skip_special_tokens=True)

        self.assertEqual(output_text, EXPECTED_TEXTS)

    @require_flash_attn
    @require_torch_large_gpu
    @pytest.mark.flash_attn_test
    def test_model_36b_flash_attn(self):
        EXPECTED_TEXTS = ""

        model = AutoModelForCausalLM.from_pretrained(
            self.model_id, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2", device_map="auto"
        )
        model.to(torch_device)

        tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        inputs = tokenizer(self.input_text, return_tensors="pt", padding=True).to(
            model.model.embed_tokens.weight.device
        )

        output = model.generate(**inputs, max_new_tokens=20, do_sample=False)
        output_text = tokenizer.batch_decode(output, skip_special_tokens=True)

        self.assertEqual(output_text, EXPECTED_TEXTS)
