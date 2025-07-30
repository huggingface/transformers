# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch Glm model."""

import unittest

import pytest

from transformers import AutoModelForCausalLM, AutoTokenizer, GlmConfig, is_torch_available
from transformers.testing_utils import (
    Expectations,
    require_flash_attn,
    require_torch,
    require_torch_large_accelerator,
    require_torch_sdpa,
    slow,
    torch_device,
)

from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester


if is_torch_available():
    import torch

    from transformers import (
        GlmForCausalLM,
        GlmForSequenceClassification,
        GlmForTokenClassification,
        GlmModel,
    )


@require_torch
class GlmModelTester(CausalLMModelTester):
    config_class = GlmConfig
    if is_torch_available():
        base_model_class = GlmModel
        causal_lm_class = GlmForCausalLM
        sequence_class = GlmForSequenceClassification
        token_class = GlmForTokenClassification


@require_torch
class GlmModelTest(CausalLMModelTest, unittest.TestCase):
    all_model_classes = (
        (GlmModel, GlmForCausalLM, GlmForSequenceClassification, GlmForTokenClassification)
        if is_torch_available()
        else ()
    )
    pipeline_model_mapping = (
        {
            "feature-extraction": GlmModel,
            "text-classification": GlmForSequenceClassification,
            "token-classification": GlmForTokenClassification,
            "text-generation": GlmForCausalLM,
        }
        if is_torch_available()
        else {}
    )

    test_headmasking = False
    test_pruning = False
    model_tester_class = GlmModelTester


@slow
@require_torch_large_accelerator
class GlmIntegrationTest(unittest.TestCase):
    input_text = ["Hello I am doing", "Hi today"]
    model_id = "THUDM/glm-4-9b"
    revision = "refs/pr/15"

    def test_model_9b_fp16(self):
        EXPECTED_TEXTS = [
            "Hello I am doing a project on the history of the internetSolution:\n\nStep 1: Introduction\nThe history of the",
            "Hi today I am going to show you how to make a simple and easy to make a DIY paper flower.",
        ]

        model = AutoModelForCausalLM.from_pretrained(
            self.model_id, dtype=torch.float16, revision=self.revision
        ).to(torch_device)

        tokenizer = AutoTokenizer.from_pretrained(self.model_id, revision=self.revision)
        inputs = tokenizer(self.input_text, return_tensors="pt", padding=True).to(torch_device)

        output = model.generate(**inputs, max_new_tokens=20, do_sample=False)
        output_text = tokenizer.batch_decode(output, skip_special_tokens=True)

        self.assertEqual(output_text, EXPECTED_TEXTS)

    def test_model_9b_bf16(self):
        EXPECTED_TEXTS = [
            "Hello I am doing a project on the history of the internetSolution:\n\nStep 1: Introduction\nThe history of the",
            "Hi today I am going to show you how to make a simple and easy to make a DIY paper flower.",
        ]

        model = AutoModelForCausalLM.from_pretrained(
            self.model_id, dtype=torch.bfloat16, revision=self.revision
        ).to(torch_device)

        tokenizer = AutoTokenizer.from_pretrained(self.model_id, revision=self.revision)
        inputs = tokenizer(self.input_text, return_tensors="pt", padding=True).to(torch_device)

        output = model.generate(**inputs, max_new_tokens=20, do_sample=False)
        output_text = tokenizer.batch_decode(output, skip_special_tokens=True)

        self.assertEqual(output_text, EXPECTED_TEXTS)

    def test_model_9b_eager(self):
        expected_texts = Expectations({
            (None, None): [
                "Hello I am doing a project on the history of the internetSolution:\n\nStep 1: Introduction\nThe history of the",
                "Hi today I am going to show you how to make a simple and easy to make a DIY paper flower.",
            ],
            ("cuda", 8): [
                'Hello I am doing a project on the history of the internetSolution:\n\nStep 1: Introduction\nThe history of the',
                'Hi today I am going to show you how to make a simple and easy to make a DIY paper lantern.',
            ],
            ("rocm", (9, 5)) : [
                "Hello I am doing a project on the history of the internetSolution:\n\nStep 1: Introduction\nThe history of the",
                "Hi today I am going to show you how to make a simple and easy to make a paper airplane. First",
            ]
        })  # fmt: skip
        EXPECTED_TEXTS = expected_texts.get_expectation()

        model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            dtype=torch.bfloat16,
            attn_implementation="eager",
            revision=self.revision,
        )
        model.to(torch_device)

        tokenizer = AutoTokenizer.from_pretrained(self.model_id, revision=self.revision)
        inputs = tokenizer(self.input_text, return_tensors="pt", padding=True).to(torch_device)

        output = model.generate(**inputs, max_new_tokens=20, do_sample=False)
        output_text = tokenizer.batch_decode(output, skip_special_tokens=True)

        self.assertEqual(output_text, EXPECTED_TEXTS)

    @require_torch_sdpa
    def test_model_9b_sdpa(self):
        EXPECTED_TEXTS = [
            "Hello I am doing a project on the history of the internetSolution:\n\nStep 1: Introduction\nThe history of the",
            "Hi today I am going to show you how to make a simple and easy to make a DIY paper flower.",
        ]

        model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
            revision=self.revision,
        )
        model.to(torch_device)

        tokenizer = AutoTokenizer.from_pretrained(self.model_id, revision=self.revision)
        inputs = tokenizer(self.input_text, return_tensors="pt", padding=True).to(torch_device)

        output = model.generate(**inputs, max_new_tokens=20, do_sample=False)
        output_text = tokenizer.batch_decode(output, skip_special_tokens=True)

        self.assertEqual(output_text, EXPECTED_TEXTS)

    @require_flash_attn
    @pytest.mark.flash_attn_test
    def test_model_9b_flash_attn(self):
        EXPECTED_TEXTS = [
            "Hello I am doing a project on the history of the internetSolution:\n\nStep 1: Introduction\nThe history of the",
            "Hi today I am going to show you how to make a simple and easy to make a DIY paper flower.",
        ]

        model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            revision=self.revision,
        )
        model.to(torch_device)

        tokenizer = AutoTokenizer.from_pretrained(self.model_id, revision=self.revision)
        inputs = tokenizer(self.input_text, return_tensors="pt", padding=True).to(torch_device)

        output = model.generate(**inputs, max_new_tokens=20, do_sample=False)
        output_text = tokenizer.batch_decode(output, skip_special_tokens=True)

        self.assertEqual(output_text, EXPECTED_TEXTS)
