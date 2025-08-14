# coding=utf-8
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
"""Testing suite for the PyTorch Glm4 model."""

import unittest

import pytest

from transformers import AutoModelForCausalLM, AutoTokenizer, Glm4Config, is_torch_available
from transformers.testing_utils import (
    Expectations,
    cleanup,
    require_flash_attn,
    require_torch,
    require_torch_large_accelerator,
    require_torch_large_gpu,
    require_torch_sdpa,
    slow,
    torch_device,
)

from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester


if is_torch_available():
    import torch

    from transformers import (
        Glm4ForCausalLM,
        Glm4ForSequenceClassification,
        Glm4ForTokenClassification,
        Glm4Model,
    )


class Glm4ModelTester(CausalLMModelTester):
    if is_torch_available():
        config_class = Glm4Config
        base_model_class = Glm4Model
        causal_lm_class = Glm4ForCausalLM
        sequence_classification_class = Glm4ForSequenceClassification
        token_classification_class = Glm4ForTokenClassification


@require_torch
class Glm4ModelTest(CausalLMModelTest, unittest.TestCase):
    model_tester_class = Glm4ModelTester
    all_model_classes = (
        (Glm4Model, Glm4ForCausalLM, Glm4ForSequenceClassification, Glm4ForTokenClassification)
        if is_torch_available()
        else ()
    )
    pipeline_model_mapping = (
        {
            "feature-extraction": Glm4Model,
            "text-classification": Glm4ForSequenceClassification,
            "token-classification": Glm4ForTokenClassification,
            "text-generation": Glm4ForCausalLM,
            "zero-shot": Glm4ForSequenceClassification,
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
class Glm4IntegrationTest(unittest.TestCase):
    input_text = ["Hello I am doing", "Hi today"]
    model_id = "THUDM/GLM-4-9B-0414"

    def tearDown(self):
        cleanup(torch_device, gc_collect=True)

    def test_model_9b_fp16(self):
        EXPECTED_TEXTS = Expectations(
            {
                ("xpu", 3): [
                    "Hello I am doing a project on the history of the internet and I need to know what the first website was and what",
                    "Hi today I am going to tell you about the most common disease in the world. This disease is called diabetes",
                ],
                ("cuda", 7): [],
                ("cuda", 8): [
                    "Hello I am doing a project on the history of the internet and I need to know what the first website was and what",
                    "Hi today I am going to tell you about the most common disease in the world. This disease is called diabetes",
                ],
            }
        )
        EXPECTED_TEXT = EXPECTED_TEXTS.get_expectation()

        model = AutoModelForCausalLM.from_pretrained(self.model_id, dtype=torch.float16).to(torch_device)

        tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        inputs = tokenizer(self.input_text, return_tensors="pt", padding=True).to(torch_device)

        output = model.generate(**inputs, max_new_tokens=20, do_sample=False)
        output_text = tokenizer.batch_decode(output, skip_special_tokens=True)

        self.assertEqual(output_text, EXPECTED_TEXT)

    def test_model_9b_bf16(self):
        EXPECTED_TEXTS = Expectations(
            {
                ("xpu", 3): [
                    "Hello I am doing a project on the history of the internet and I need to know what the first website was and what",
                    "Hi today I am going to tell you about the most common mistakes that people make when they are learning English.",
                ],
                ("cuda", 7): [],
                ("cuda", 8): [
                    "Hello I am doing a project on the history of the internet and I need to know what the first website was and what",
                    "Hi today I am going to tell you about the most common disease in the world. This disease is called diabetes",
                ],
            }
        )
        EXPECTED_TEXT = EXPECTED_TEXTS.get_expectation()

        model = AutoModelForCausalLM.from_pretrained(self.model_id, dtype=torch.bfloat16).to(torch_device)

        tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        inputs = tokenizer(self.input_text, return_tensors="pt", padding=True).to(torch_device)

        output = model.generate(**inputs, max_new_tokens=20, do_sample=False)
        output_text = tokenizer.batch_decode(output, skip_special_tokens=True)

        self.assertEqual(output_text, EXPECTED_TEXT)

    def test_model_9b_eager(self):
        EXPECTED_TEXTS = Expectations(
            {
                ("xpu", 3): [
                    "Hello I am doing a project on the history of the internet and I need to know what the first website was and who",
                    "Hi today I am going to tell you about the most common disease in the world. This disease is called diabetes",
                ],
                ("cuda", 7): [],
                ("cuda", 8): [
                    "Hello I am doing a project on the history of the internet and I need to know what the first website was and what",
                    "Hi today I am going to tell you about the most common disease in the world. This disease is called diabetes",
                ],
            }
        )
        EXPECTED_TEXT = EXPECTED_TEXTS.get_expectation()

        model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            dtype=torch.bfloat16,
            attn_implementation="eager",
        )
        model.to(torch_device)

        tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        inputs = tokenizer(self.input_text, return_tensors="pt", padding=True).to(torch_device)

        output = model.generate(**inputs, max_new_tokens=20, do_sample=False)
        output_text = tokenizer.batch_decode(output, skip_special_tokens=True)

        self.assertEqual(output_text, EXPECTED_TEXT)

    @require_torch_sdpa
    def test_model_9b_sdpa(self):
        EXPECTED_TEXTS = Expectations(
            {
                ("xpu", 3): [
                    "Hello I am doing a project on the history of the internet and I need to know what the first website was and what",
                    "Hi today I am going to tell you about the most common mistakes that people make when they are learning English.",
                ],
                ("cuda", 7): [],
                ("cuda", 8): [
                    "Hello I am doing a project on the history of the internet and I need to know what the first website was and what",
                    "Hi today I am going to tell you about the most common disease in the world. This disease is called diabetes",
                ],
            }
        )
        EXPECTED_TEXT = EXPECTED_TEXTS.get_expectation()

        model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        model.to(torch_device)

        tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        inputs = tokenizer(self.input_text, return_tensors="pt", padding=True).to(torch_device)

        output = model.generate(**inputs, max_new_tokens=20, do_sample=False)
        output_text = tokenizer.batch_decode(output, skip_special_tokens=True)

        self.assertEqual(output_text, EXPECTED_TEXT)

    @require_flash_attn
    @require_torch_large_gpu
    @pytest.mark.flash_attn_test
    def test_model_9b_flash_attn(self):
        EXPECTED_TEXTS = Expectations(
            {
                ("cuda", 7): [],
                ("cuda", 8): [
                    "Hello I am doing a project on the history of the internet and I need to know what the first website was and what",
                    "Hi today I am going to tell you about the most common disease in the world. This disease is called diabetes",
                ],
            }
        )
        EXPECTED_TEXT = EXPECTED_TEXTS.get_expectation()

        model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )
        model.to(torch_device)

        tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        inputs = tokenizer(self.input_text, return_tensors="pt", padding=True).to(torch_device)

        output = model.generate(**inputs, max_new_tokens=20, do_sample=False)
        output_text = tokenizer.batch_decode(output, skip_special_tokens=True)

        self.assertEqual(output_text, EXPECTED_TEXT)
