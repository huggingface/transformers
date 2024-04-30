# coding=utf-8
# Copyright 2024 The HuggingFace Team. All rights reserved.
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
import tempfile
import unittest

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.testing_utils import require_gguf, require_torch_gpu, slow, torch_device
from transformers.utils import is_torch_available


if is_torch_available():
    import torch


@require_gguf
@require_torch_gpu
@slow
class GgufIntegrationTests(unittest.TestCase):
    original_model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    model_id = "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF"
    mistral_model_id = "TheBloke/Mistral-7B-Instruct-v0.2-GGUF"

    q4_0_gguf_model_id = "tinyllama-1.1b-chat-v1.0.Q4_0.gguf"
    q4_k_gguf_model_id = "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
    q2_k_gguf_model_id = "tinyllama-1.1b-chat-v1.0.Q2_K.gguf"
    q3_k_gguf_model_id = "tinyllama-1.1b-chat-v1.0.Q3_K_L.gguf"
    q5_k_gguf_model_id = "tinyllama-1.1b-chat-v1.0.Q5_K_M.gguf"
    q6_k_gguf_model_id = "tinyllama-1.1b-chat-v1.0.Q6_K.gguf"
    q8_0_gguf_model_id = "tinyllama-1.1b-chat-v1.0.Q8_0.gguf"

    q4_0_mistral_model_id = "mistral-7b-instruct-v0.2.Q4_0.gguf"

    example_text = "Hello"

    def test_q2_k(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_id, from_gguf=self.q2_k_gguf_model_id)
        model = AutoModelForCausalLM.from_pretrained(self.model_id, from_gguf=self.q2_k_gguf_model_id).to(torch_device)

        text = tokenizer(self.example_text, return_tensors="pt").to(torch_device)
        out = model.generate(**text, max_new_tokens=10)

        EXPECTED_TEXT = " Hello, World!\n\n[10:0"
        self.assertEqual(tokenizer.decode(out[0], skip_special_tokens=True), EXPECTED_TEXT)

    def test_q2_k_serialization(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_id, from_gguf=self.q2_k_gguf_model_id)
        model = AutoModelForCausalLM.from_pretrained(self.model_id, from_gguf=self.q2_k_gguf_model_id).to(torch_device)

        with tempfile.TemporaryDirectory() as tmpdirname:
            model.save_pretrained(tmpdirname)
            tokenizer.save_pretrained(tmpdirname)

            model = AutoModelForCausalLM.from_pretrained(tmpdirname).to(torch_device)
            tokenizer = AutoTokenizer.from_pretrained(tmpdirname)

            text = tokenizer(self.example_text, return_tensors="pt").to(torch_device)
            out = model.generate(**text, max_new_tokens=10)

        EXPECTED_TEXT = " Hello, World!\n\n[10:0"
        self.assertEqual(tokenizer.decode(out[0], skip_special_tokens=True), EXPECTED_TEXT)

    def test_q3_k(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_id, from_gguf=self.q3_k_gguf_model_id)
        model = AutoModelForCausalLM.from_pretrained(self.model_id, from_gguf=self.q3_k_gguf_model_id).to(torch_device)

        text = tokenizer(self.example_text, return_tensors="pt").to(torch_device)
        out = model.generate(**text, max_new_tokens=10)

        EXPECTED_TEXT = " Hello, World!\n\n```\n<|user"
        self.assertEqual(tokenizer.decode(out[0], skip_special_tokens=True), EXPECTED_TEXT)

    def test_q5_k(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_id, from_gguf=self.q5_k_gguf_model_id)
        model = AutoModelForCausalLM.from_pretrained(self.model_id, from_gguf=self.q5_k_gguf_model_id).to(torch_device)

        text = tokenizer(self.example_text, return_tensors="pt").to(torch_device)
        out = model.generate(**text, max_new_tokens=10)

        EXPECTED_TEXT = " Hello, World!\n\nStep 3: Add"
        self.assertEqual(tokenizer.decode(out[0], skip_special_tokens=True), EXPECTED_TEXT)

    def test_q4_0(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_id, from_gguf=self.q4_0_gguf_model_id)
        model = AutoModelForCausalLM.from_pretrained(self.model_id, from_gguf=self.q4_0_gguf_model_id).to(torch_device)

        text = tokenizer(self.example_text, return_tensors="pt").to(torch_device)
        out = model.generate(**text, max_new_tokens=10)

        EXPECTED_TEXT = " Hello, World!\n\nStep 3: Add"
        self.assertEqual(tokenizer.decode(out[0], skip_special_tokens=True), EXPECTED_TEXT)

    def test_q4_k_m(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_id, from_gguf=self.q4_k_gguf_model_id)
        model = AutoModelForCausalLM.from_pretrained(self.model_id, from_gguf=self.q4_k_gguf_model_id).to(torch_device)

        text = tokenizer(self.example_text, return_tensors="pt").to(torch_device)
        out = model.generate(**text, max_new_tokens=10)

        EXPECTED_TEXT = " Hello, World!\n\n5. Python:\n"
        self.assertEqual(tokenizer.decode(out[0], skip_special_tokens=True), EXPECTED_TEXT)

    def test_q6_k(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_id, from_gguf=self.q6_k_gguf_model_id)
        model = AutoModelForCausalLM.from_pretrained(self.model_id, from_gguf=self.q6_k_gguf_model_id).to(torch_device)

        text = tokenizer(self.example_text, return_tensors="pt").to(torch_device)
        out = model.generate(**text, max_new_tokens=10)

        EXPECTED_TEXT = " Hello, World!\n\nStep 3: Add"
        self.assertEqual(tokenizer.decode(out[0], skip_special_tokens=True), EXPECTED_TEXT)

    def test_q6_k_fp16(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_id, from_gguf=self.q6_k_gguf_model_id)
        model = AutoModelForCausalLM.from_pretrained(
            self.model_id, from_gguf=self.q6_k_gguf_model_id, torch_dtype=torch.float16
        ).to(torch_device)

        self.assertTrue(model.lm_head.weight.dtype == torch.float16)

        text = tokenizer(self.example_text, return_tensors="pt").to(torch_device)
        out = model.generate(**text, max_new_tokens=10)

        EXPECTED_TEXT = " Hello, World!\n\nStep 3: Add"
        self.assertEqual(tokenizer.decode(out[0], skip_special_tokens=True), EXPECTED_TEXT)

    def test_q8_0(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_id, from_gguf=self.q8_0_gguf_model_id)
        model = AutoModelForCausalLM.from_pretrained(self.model_id, from_gguf=self.q8_0_gguf_model_id).to(torch_device)

        text = tokenizer(self.example_text, return_tensors="pt").to(torch_device)
        out = model.generate(**text, max_new_tokens=10)

        EXPECTED_TEXT = " Hello, World!\n\n5. Use a library"
        self.assertEqual(tokenizer.decode(out[0], skip_special_tokens=True), EXPECTED_TEXT)

    def test_mistral_q4_0(self):
        tokenizer = AutoTokenizer.from_pretrained(self.mistral_model_id, from_gguf=self.q4_0_mistral_model_id)
        model = AutoModelForCausalLM.from_pretrained(
            self.mistral_model_id, from_gguf=self.q4_0_mistral_model_id, device_map="auto", torch_dtype=torch.float16
        )

        text = tokenizer(self.example_text, return_tensors="pt").to(torch_device)
        out = model.generate(**text, max_new_tokens=10)

        EXPECTED_TEXT = "<s> Hello,\n\nI'm trying to create a"
        self.assertEqual(tokenizer.decode(out[0], skip_special_tokens=True), EXPECTED_TEXT)
