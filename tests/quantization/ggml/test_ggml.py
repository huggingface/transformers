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

from transformers import AddedToken, AutoModelForCausalLM, AutoTokenizer
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
    imatrix_model_id = "duyntnet/TinyLlama-1.1B-Chat-v1.0-imatrix-GGUF"
    mistral_model_id = "TheBloke/Mistral-7B-Instruct-v0.2-GGUF"
    qwen2_model_id = "Qwen/Qwen1.5-0.5B-Chat-GGUF"
    llama3_model_id = "NousResearch/Meta-Llama-3-8B-GGUF"
    tinyllama_model_id = "PenutChen/TinyLlama-1.1B-Chat-v1.0-GGUF"

    # standard quants
    q4_0_gguf_model_id = "tinyllama-1.1b-chat-v1.0.Q4_0.gguf"
    q5_0_gguf_model_id = "tinyllama-1.1b-chat-v1.0.Q5_0.gguf"
    q8_0_gguf_model_id = "tinyllama-1.1b-chat-v1.0.Q8_0.gguf"
    # k-quants
    q2_k_gguf_model_id = "tinyllama-1.1b-chat-v1.0.Q2_K.gguf"
    q3_k_gguf_model_id = "tinyllama-1.1b-chat-v1.0.Q3_K_L.gguf"
    q4_k_gguf_model_id = "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
    q5_k_gguf_model_id = "tinyllama-1.1b-chat-v1.0.Q5_K_M.gguf"
    q6_k_gguf_model_id = "tinyllama-1.1b-chat-v1.0.Q6_K.gguf"
    # imatrix
    iq1_m_gguf_model_id = "TinyLlama-1.1B-Chat-v1.0-IQ1_M.gguf"
    iq1_s_gguf_model_id = "TinyLlama-1.1B-Chat-v1.0-IQ1_S.gguf"
    iq2_s_gguf_model_id = "TinyLlama-1.1B-Chat-v1.0-IQ2_S.gguf"
    iq2_xs_gguf_model_id = "TinyLlama-1.1B-Chat-v1.0-IQ2_XS.gguf"
    iq2_xxs_gguf_model_id = "TinyLlama-1.1B-Chat-v1.0-IQ2_XXS.gguf"
    iq3_s_gguf_model_id = "TinyLlama-1.1B-Chat-v1.0-IQ3_S.gguf"
    iq3_xxs_gguf_model_id = "TinyLlama-1.1B-Chat-v1.0-IQ3_XXS.gguf"
    iq4_xs_gguf_model_id = "TinyLlama-1.1B-Chat-v1.0-IQ4_XS.gguf"
    iq4_nl_gguf_model_id = "TinyLlama-1.1B-Chat-v1.0-IQ4_NL.gguf"

    q4_0_mistral_model_id = "mistral-7b-instruct-v0.2.Q4_0.gguf"
    q4_0_qwen2_model_id = "qwen1_5-0_5b-chat-q4_0.gguf"
    q4_llama3_model_id = "Meta-Llama-3-8B-Q4_K_M.gguf"
    f16_tinyllama_model_id = "TinyLlama-1.1B-Chat-v1.0.FP16.gguf"

    example_text = "Hello"

    def test_q2_k(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_id, gguf_file=self.q2_k_gguf_model_id)
        model = AutoModelForCausalLM.from_pretrained(self.model_id, gguf_file=self.q2_k_gguf_model_id).to(torch_device)

        text = tokenizer(self.example_text, return_tensors="pt").to(torch_device)
        out = model.generate(**text, max_new_tokens=10)

        EXPECTED_TEXT = "Hello, World!\n\n[10:0"
        self.assertEqual(tokenizer.decode(out[0], skip_special_tokens=True), EXPECTED_TEXT)

    def test_q2_k_serialization(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_id, gguf_file=self.q2_k_gguf_model_id)
        model = AutoModelForCausalLM.from_pretrained(self.model_id, gguf_file=self.q2_k_gguf_model_id).to(torch_device)

        with tempfile.TemporaryDirectory() as tmpdirname:
            model.save_pretrained(tmpdirname)
            tokenizer.save_pretrained(tmpdirname)

            model = AutoModelForCausalLM.from_pretrained(tmpdirname).to(torch_device)
            tokenizer = AutoTokenizer.from_pretrained(tmpdirname)

            text = tokenizer(self.example_text, return_tensors="pt").to(torch_device)
            out = model.generate(**text, max_new_tokens=10)

        EXPECTED_TEXT = "Hello, World!\n\n[10:0"
        self.assertEqual(tokenizer.decode(out[0], skip_special_tokens=True), EXPECTED_TEXT)

    def test_q3_k(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_id, gguf_file=self.q3_k_gguf_model_id)
        model = AutoModelForCausalLM.from_pretrained(self.model_id, gguf_file=self.q3_k_gguf_model_id).to(torch_device)

        text = tokenizer(self.example_text, return_tensors="pt").to(torch_device)
        out = model.generate(**text, max_new_tokens=10)

        EXPECTED_TEXT = "Hello, World!\n\n```\n<|user"
        self.assertEqual(tokenizer.decode(out[0], skip_special_tokens=True), EXPECTED_TEXT)

    def test_q5_0(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_id, gguf_file=self.q5_0_gguf_model_id)
        model = AutoModelForCausalLM.from_pretrained(self.model_id, gguf_file=self.q5_0_gguf_model_id).to(torch_device)

        text = tokenizer(self.example_text, return_tensors="pt").to(torch_device)
        out = model.generate(**text, max_new_tokens=10)

        EXPECTED_TEXT = "Hello, World!\n\n5. Use a library"
        self.assertEqual(tokenizer.decode(out[0], skip_special_tokens=True), EXPECTED_TEXT)

    def test_q5_k(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_id, gguf_file=self.q5_k_gguf_model_id)
        model = AutoModelForCausalLM.from_pretrained(self.model_id, gguf_file=self.q5_k_gguf_model_id).to(torch_device)

        text = tokenizer(self.example_text, return_tensors="pt").to(torch_device)
        out = model.generate(**text, max_new_tokens=10)

        EXPECTED_TEXT = "Hello, World!\n\nStep 3: Add"
        self.assertEqual(tokenizer.decode(out[0], skip_special_tokens=True), EXPECTED_TEXT)

    def test_q4_0(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_id, gguf_file=self.q4_0_gguf_model_id)
        model = AutoModelForCausalLM.from_pretrained(self.model_id, gguf_file=self.q4_0_gguf_model_id).to(torch_device)

        text = tokenizer(self.example_text, return_tensors="pt").to(torch_device)
        out = model.generate(**text, max_new_tokens=10)

        EXPECTED_TEXT = "Hello, World!\n\nStep 3: Add"
        self.assertEqual(tokenizer.decode(out[0], skip_special_tokens=True), EXPECTED_TEXT)

    def test_q4_k_m(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_id, gguf_file=self.q4_k_gguf_model_id)
        model = AutoModelForCausalLM.from_pretrained(self.model_id, gguf_file=self.q4_k_gguf_model_id).to(torch_device)

        text = tokenizer(self.example_text, return_tensors="pt").to(torch_device)
        out = model.generate(**text, max_new_tokens=10)

        EXPECTED_TEXT = "Hello, World!\n\n5. Python:\n"
        self.assertEqual(tokenizer.decode(out[0], skip_special_tokens=True), EXPECTED_TEXT)

    def test_q6_k(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_id, gguf_file=self.q6_k_gguf_model_id)
        model = AutoModelForCausalLM.from_pretrained(self.model_id, gguf_file=self.q6_k_gguf_model_id).to(torch_device)

        text = tokenizer(self.example_text, return_tensors="pt").to(torch_device)
        out = model.generate(**text, max_new_tokens=10)

        EXPECTED_TEXT = "Hello, World!\n\nStep 3: Add"
        self.assertEqual(tokenizer.decode(out[0], skip_special_tokens=True), EXPECTED_TEXT)

    def test_q6_k_fp16(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_id, gguf_file=self.q6_k_gguf_model_id)
        model = AutoModelForCausalLM.from_pretrained(
            self.model_id, gguf_file=self.q6_k_gguf_model_id, torch_dtype=torch.float16
        ).to(torch_device)

        self.assertTrue(model.lm_head.weight.dtype == torch.float16)

        text = tokenizer(self.example_text, return_tensors="pt").to(torch_device)
        out = model.generate(**text, max_new_tokens=10)

        EXPECTED_TEXT = "Hello, World!\n\nStep 3: Add"
        self.assertEqual(tokenizer.decode(out[0], skip_special_tokens=True), EXPECTED_TEXT)

    def test_q8_0(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_id, gguf_file=self.q8_0_gguf_model_id)
        model = AutoModelForCausalLM.from_pretrained(self.model_id, gguf_file=self.q8_0_gguf_model_id).to(torch_device)

        text = tokenizer(self.example_text, return_tensors="pt").to(torch_device)
        out = model.generate(**text, max_new_tokens=10)

        EXPECTED_TEXT = "Hello, World!\n\n5. Use a library"
        self.assertEqual(tokenizer.decode(out[0], skip_special_tokens=True), EXPECTED_TEXT)

    def test_iq1_s(self):
        tokenizer = AutoTokenizer.from_pretrained(self.imatrix_model_id, gguf_file=self.iq1_s_gguf_model_id)
        model = AutoModelForCausalLM.from_pretrained(self.imatrix_model_id, gguf_file=self.iq1_s_gguf_model_id).to(
            torch_device
        )

        text = tokenizer(self.example_text, return_tensors="pt").to(torch_device)
        out = model.generate(**text, max_new_tokens=10)

        EXPECTED_TEXT = "Hello, I'm a friend of mine, I"
        self.assertEqual(tokenizer.decode(out[0], skip_special_tokens=True), EXPECTED_TEXT)

    def test_iq1_m(self):
        tokenizer = AutoTokenizer.from_pretrained(self.imatrix_model_id, gguf_file=self.iq1_m_gguf_model_id)
        model = AutoModelForCausalLM.from_pretrained(self.imatrix_model_id, gguf_file=self.iq1_m_gguf_model_id).to(
            torch_device
        )

        text = tokenizer(self.example_text, return_tensors="pt").to(torch_device)
        out = model.generate(**text, max_new_tokens=10)

        EXPECTED_TEXT = "Hello, I am interested in purching a copy of"
        self.assertEqual(tokenizer.decode(out[0], skip_special_tokens=True), EXPECTED_TEXT)

    def test_iq2_s(self):
        tokenizer = AutoTokenizer.from_pretrained(self.imatrix_model_id, gguf_file=self.iq2_s_gguf_model_id)
        model = AutoModelForCausalLM.from_pretrained(self.imatrix_model_id, gguf_file=self.iq2_s_gguf_model_id).to(
            torch_device
        )

        text = tokenizer(self.example_text, return_tensors="pt").to(torch_device)
        out = model.generate(**text, max_new_tokens=10)

        EXPECTED_TEXT = "Hello World!\n\n```\n<|user|"
        self.assertEqual(tokenizer.decode(out[0], skip_special_tokens=True), EXPECTED_TEXT)

    def test_iq2_xs(self):
        tokenizer = AutoTokenizer.from_pretrained(self.imatrix_model_id, gguf_file=self.iq2_xs_gguf_model_id)
        model = AutoModelForCausalLM.from_pretrained(self.imatrix_model_id, gguf_file=self.iq2_xs_gguf_model_id).to(
            torch_device
        )

        text = tokenizer(self.example_text, return_tensors="pt").to(torch_device)
        out = model.generate(**text, max_new_tokens=10)

        EXPECTED_TEXT = "Hello World!\n\n```\n<|user|"
        self.assertEqual(tokenizer.decode(out[0], skip_special_tokens=True), EXPECTED_TEXT)

    def test_iq2_xxs(self):
        tokenizer = AutoTokenizer.from_pretrained(self.imatrix_model_id, gguf_file=self.iq2_xxs_gguf_model_id)
        model = AutoModelForCausalLM.from_pretrained(self.imatrix_model_id, gguf_file=self.iq2_xxs_gguf_model_id).to(
            torch_device
        )

        text = tokenizer(self.example_text, return_tensors="pt").to(torch_device)
        out = model.generate(**text, max_new_tokens=10)

        EXPECTED_TEXT = "Hello, I'm a software engineer. I'"
        self.assertEqual(tokenizer.decode(out[0], skip_special_tokens=True), EXPECTED_TEXT)

    def test_iq3_s(self):
        tokenizer = AutoTokenizer.from_pretrained(self.imatrix_model_id, gguf_file=self.iq3_s_gguf_model_id)
        model = AutoModelForCausalLM.from_pretrained(self.imatrix_model_id, gguf_file=self.iq3_s_gguf_model_id).to(
            torch_device
        )

        text = tokenizer(self.example_text, return_tensors="pt").to(torch_device)
        out = model.generate(**text, max_new_tokens=10)

        EXPECTED_TEXT = "Hello, World!\n\n5. Python:\n"
        self.assertEqual(tokenizer.decode(out[0], skip_special_tokens=True), EXPECTED_TEXT)

    def test_iq3_xxs(self):
        tokenizer = AutoTokenizer.from_pretrained(self.imatrix_model_id, gguf_file=self.iq3_xxs_gguf_model_id)
        model = AutoModelForCausalLM.from_pretrained(self.imatrix_model_id, gguf_file=self.iq3_xxs_gguf_model_id).to(
            torch_device
        )

        text = tokenizer(self.example_text, return_tensors="pt").to(torch_device)
        out = model.generate(**text, max_new_tokens=10)

        EXPECTED_TEXT = "Hello, I am interested in your product. Can you"
        self.assertEqual(tokenizer.decode(out[0], skip_special_tokens=True), EXPECTED_TEXT)

    def test_iq4_xs(self):
        tokenizer = AutoTokenizer.from_pretrained(self.imatrix_model_id, gguf_file=self.iq4_xs_gguf_model_id)
        model = AutoModelForCausalLM.from_pretrained(self.imatrix_model_id, gguf_file=self.iq4_xs_gguf_model_id).to(
            torch_device
        )

        text = tokenizer(self.example_text, return_tensors="pt").to(torch_device)
        out = model.generate(**text, max_new_tokens=10)

        EXPECTED_TEXT = "Hello, world!\n\n5. Using a loop"
        self.assertEqual(tokenizer.decode(out[0], skip_special_tokens=True), EXPECTED_TEXT)

    def test_iq4_nl(self):
        tokenizer = AutoTokenizer.from_pretrained(self.imatrix_model_id, gguf_file=self.iq4_nl_gguf_model_id)
        model = AutoModelForCausalLM.from_pretrained(self.imatrix_model_id, gguf_file=self.iq4_nl_gguf_model_id).to(
            torch_device
        )

        text = tokenizer(self.example_text, return_tensors="pt").to(torch_device)
        out = model.generate(**text, max_new_tokens=10)

        EXPECTED_TEXT = "Hello, world!\n\n5. Using a loop"
        self.assertEqual(tokenizer.decode(out[0], skip_special_tokens=True), EXPECTED_TEXT)

    def test_f16(self):
        tokenizer = AutoTokenizer.from_pretrained(self.tinyllama_model_id, gguf_file=self.f16_tinyllama_model_id)
        model = AutoModelForCausalLM.from_pretrained(
            self.tinyllama_model_id, gguf_file=self.f16_tinyllama_model_id
        ).to(torch_device)

        text = tokenizer(self.example_text, return_tensors="pt").to(torch_device)
        out = model.generate(**text, max_new_tokens=10)

        EXPECTED_TEXT = "Hello, World!\n\n5. Node.js"
        self.assertEqual(tokenizer.decode(out[0], skip_special_tokens=True), EXPECTED_TEXT)

    def test_mistral_q4_0(self):
        tokenizer = AutoTokenizer.from_pretrained(self.mistral_model_id, gguf_file=self.q4_0_mistral_model_id)
        model = AutoModelForCausalLM.from_pretrained(
            self.mistral_model_id, gguf_file=self.q4_0_mistral_model_id, device_map="auto", torch_dtype=torch.float16
        )

        text = tokenizer(self.example_text, return_tensors="pt").to(torch_device)
        out = model.generate(**text, max_new_tokens=10)

        EXPECTED_TEXT = "Hello,\n\nI'm trying to create a"
        self.assertEqual(tokenizer.decode(out[0], skip_special_tokens=True), EXPECTED_TEXT)

    def test_qwen2_q4_0(self):
        tokenizer = AutoTokenizer.from_pretrained(self.qwen2_model_id, gguf_file=self.q4_0_qwen2_model_id)
        model = AutoModelForCausalLM.from_pretrained(
            self.qwen2_model_id, gguf_file=self.q4_0_qwen2_model_id, device_map="auto", torch_dtype=torch.float16
        )

        text = tokenizer(self.example_text, return_tensors="pt").to(torch_device)
        out = model.generate(**text, max_new_tokens=10)

        EXPECTED_TEXT = "Hello.jsoup\n\nI am a beginner"
        self.assertEqual(tokenizer.decode(out[0], skip_special_tokens=True), EXPECTED_TEXT)

    def test_llama3_q4_0_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(self.llama3_model_id, gguf_file=self.q4_llama3_model_id)
        with tempfile.TemporaryDirectory() as tmpdirname:
            tokenizer.save_pretrained(tmpdirname)
            tokenizer = AutoTokenizer.from_pretrained(tmpdirname)
            special_sentence = "สวัสดี"
            predicted_text = tokenizer.decode(tokenizer.encode(special_sentence, return_tensors="pt")[0])
            self.assertEqual(predicted_text, "<|begin_of_text|>" + special_sentence)

    def test_llama3_q4_0(self):
        tokenizer = AutoTokenizer.from_pretrained(self.llama3_model_id, gguf_file=self.q4_llama3_model_id)
        model = AutoModelForCausalLM.from_pretrained(
            self.llama3_model_id, gguf_file=self.q4_llama3_model_id, device_map="auto", torch_dtype=torch.float16
        )

        text = tokenizer(self.example_text, return_tensors="pt").to(torch_device)
        out = model.generate(**text, max_new_tokens=10)

        EXPECTED_TEXT = "Hello, I am interested in [The Park]\nThe"
        self.assertEqual(tokenizer.decode(out[0], skip_special_tokens=True), EXPECTED_TEXT)

    def test_tokenization_xnli(self):
        import tqdm
        from datasets import load_dataset

        gguf_tokenizer = AutoTokenizer.from_pretrained(self.model_id, gguf_file=self.q8_0_gguf_model_id)
        original_tokenizer = AutoTokenizer.from_pretrained(self.original_model_id)

        dataset = load_dataset("google/code_x_glue_ct_code_to_text", "go")
        for item in tqdm.tqdm(dataset["validation"]):
            string = item["code"]
            encoded1 = gguf_tokenizer.encode(string)
            encoded2 = original_tokenizer.encode(string)

            self.assertEqual(encoded1, encoded2)

            decoded1 = gguf_tokenizer.decode(encoded1, skip_special_tokens=True)
            decoded2 = original_tokenizer.decode(encoded2, skip_special_tokens=True)

            self.assertEqual(decoded1, decoded2)

        dataset = load_dataset("facebook/xnli", "all_languages")

        for i, item in enumerate(tqdm.tqdm(dataset["train"].select(range(100)))):
            for string in item["premise"].values():
                encoded1 = gguf_tokenizer.encode(string)
                encoded2 = original_tokenizer.encode(string)

                self.assertEqual(encoded1, encoded2)

                decoded1 = gguf_tokenizer.decode(encoded1, skip_special_tokens=True)
                decoded2 = original_tokenizer.decode(encoded2, skip_special_tokens=True)

                self.assertEqual(decoded1, decoded2)

        # With special tokens
        gguf_tokenizer = AutoTokenizer.from_pretrained(self.model_id, gguf_file=self.q8_0_gguf_model_id)
        original_tokenizer = AutoTokenizer.from_pretrained(self.original_model_id)

        gguf_tokenizer.add_special_tokens(
            {"additional_special_tokens": [AddedToken("<token>", rstrip=False, lstrip=False)]}
        )
        original_tokenizer.add_special_tokens(
            {"additional_special_tokens": [AddedToken("<token>", rstrip=False, lstrip=False)]}
        )

        text = "Hello <token>. <token> Hello"

        encoded1 = gguf_tokenizer.encode(text)
        encoded2 = original_tokenizer.encode(text)

        self.assertEqual(encoded1, encoded2)

        decoded1 = gguf_tokenizer.decode(encoded1, skip_special_tokens=True)
        decoded2 = original_tokenizer.decode(encoded2, skip_special_tokens=True)

        self.assertEqual(decoded1, decoded2)
