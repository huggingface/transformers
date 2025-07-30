# Copyright 2024 BigCode and The HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch Starcoder2 model."""

import unittest

import pytest

from transformers import Starcoder2Config, is_torch_available
from transformers.testing_utils import (
    Expectations,
    require_bitsandbytes,
    require_flash_attn,
    require_torch,
    require_torch_accelerator,
    require_torch_gpu,
    slow,
    torch_device,
)


if is_torch_available():
    import torch

    from transformers import (
        AutoTokenizer,
        Starcoder2ForCausalLM,
        Starcoder2ForSequenceClassification,
        Starcoder2ForTokenClassification,
        Starcoder2Model,
    )

from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester


class Starcoder2ModelTester(CausalLMModelTester):
    config_class = Starcoder2Config
    if is_torch_available():
        base_model_class = Starcoder2Model
        causal_lm_class = Starcoder2ForCausalLM
        sequence_class = Starcoder2ForSequenceClassification
        token_class = Starcoder2ForTokenClassification


@require_torch
class Starcoder2ModelTest(CausalLMModelTest, unittest.TestCase):
    all_model_classes = (
        (Starcoder2Model, Starcoder2ForCausalLM, Starcoder2ForSequenceClassification, Starcoder2ForTokenClassification)
        if is_torch_available()
        else ()
    )
    test_headmasking = False
    test_pruning = False
    model_tester_class = Starcoder2ModelTester
    pipeline_model_mapping = (
        {
            "feature-extraction": Starcoder2Model,
            "text-classification": Starcoder2ForSequenceClassification,
            "token-classification": Starcoder2ForTokenClassification,
            "text-generation": Starcoder2ForCausalLM,
        }
        if is_torch_available()
        else {}
    )

    @require_flash_attn
    @require_torch_gpu
    @pytest.mark.flash_attn_test
    @slow
    def test_flash_attn_2_inference_equivalence_right_padding(self):
        self.skipTest(reason="Starcoder2 flash attention does not support right padding")


@slow
@require_torch_accelerator
class Starcoder2IntegrationTest(unittest.TestCase):
    def test_starcoder2_batched_generation_sdpa(self):
        EXPECTED_TEXT = [
            "Hello my name is Younes and I am a student at the University of Liverpool. I am currently studying for my MSc in Computer Science. I am interested in the field of Machine Learning and I am currently working on",
            "def hello_world():\n\treturn 'Hello World!'\n\n@app.route('/hello/<name>')\ndef hello_name(name):\n\treturn 'Hello %s!' % name\n\n@app",
        ]
        model_id = "bigcode/starcoder2-7b"

        model = Starcoder2ForCausalLM.from_pretrained(
            model_id, dtype=torch.float16, device_map="auto", attn_implementation="sdpa"
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizer.pad_token = tokenizer.eos_token

        text = ["Hello my name is Younes and", "def hello_world():"]
        inputs = tokenizer(text, return_tensors="pt", padding=True).to(torch_device)

        output = model.generate(**inputs, max_new_tokens=40, do_sample=False)
        output_text = tokenizer.batch_decode(output, skip_special_tokens=True)
        self.assertEqual(EXPECTED_TEXT, output_text)

    def test_starcoder2_batched_generation_eager(self):
        EXPECTED_TEXT = [
            "Hello my name is Younes and I am a student at the University of Liverpool. I am currently studying for my MSc in Computer Science. I am interested in the field of Machine Learning and I am currently working on",
            "def hello_world():\n\treturn 'Hello World!'\n\n@app.route('/hello/<name>')\ndef hello_name(name):\n\treturn 'Hello %s!' % name\n\n@app",
        ]
        model_id = "bigcode/starcoder2-7b"

        model = Starcoder2ForCausalLM.from_pretrained(
            model_id, dtype=torch.float16, device_map="auto", attn_implementation="eager"
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizer.pad_token = tokenizer.eos_token

        text = ["Hello my name is Younes and", "def hello_world():"]
        inputs = tokenizer(text, return_tensors="pt", padding=True).to(torch_device)

        output = model.generate(**inputs, max_new_tokens=40, do_sample=False)
        output_text = tokenizer.batch_decode(output, skip_special_tokens=True)
        self.assertEqual(EXPECTED_TEXT, output_text)

    @require_flash_attn
    @pytest.mark.flash_attn_test
    def test_starcoder2_batched_generation_fa2(self):
        EXPECTED_TEXT = [
            "Hello my name is Younes and I am a student at the University of Liverpool. I am currently studying for my MSc in Computer Science. I am interested in the field of Machine Learning and I am currently working on",
            "def hello_world():\n\treturn 'Hello World!'\n\n@app.route('/hello/<name>')\ndef hello_name(name):\n\treturn 'Hello %s!' % name\n\n@app",
        ]
        model_id = "bigcode/starcoder2-7b"

        model = Starcoder2ForCausalLM.from_pretrained(
            model_id, dtype=torch.float16, device_map="auto", attn_implementation="flash_attention_2"
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizer.pad_token = tokenizer.eos_token

        text = ["Hello my name is Younes and", "def hello_world():"]
        inputs = tokenizer(text, return_tensors="pt", padding=True).to(torch_device)

        output = model.generate(**inputs, max_new_tokens=40, do_sample=False)
        output_text = tokenizer.batch_decode(output, skip_special_tokens=True)
        self.assertEqual(EXPECTED_TEXT, output_text)

    @require_bitsandbytes
    def test_starcoder2_batched_generation_4bit(self):
        expectations = Expectations(
            {
                (None, None): [
                    'Hello my name is Younes and I am a student at the University of Maryland. I am currently working on a project that is related to the topic of "How to make a game". I am currently working on a project',
                    'def hello_world():\n\treturn "Hello World"\n\n@app.route(\'/hello/<name>\')\ndef hello_name(name):\n\treturn "Hello " + name\n\n@app.route',
                ],
                ("cuda", 8): [
                    "Hello my name is Younes and I am a student at the University of Maryland. I am currently working on a project that is aimed at creating a new way of learning. I am hoping to create a new way of",
                    'def hello_world():\n\treturn "Hello World"\n\n@app.route(\'/hello/<name>\')\ndef hello_name(name):\n\treturn "Hello " + name\n\n@app.route',
                ],
            }
        )
        EXPECTED_TEXT = expectations.get_expectation()

        model_id = "bigcode/starcoder2-7b"

        model = Starcoder2ForCausalLM.from_pretrained(model_id, load_in_4bit=True)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizer.pad_token = tokenizer.eos_token

        text = ["Hello my name is Younes and", "def hello_world():"]
        inputs = tokenizer(text, return_tensors="pt", padding=True).to(torch_device)

        output = model.generate(**inputs, max_new_tokens=40, do_sample=False)
        output_text = tokenizer.batch_decode(output, skip_special_tokens=True)
        self.assertEqual(EXPECTED_TEXT, output_text)
