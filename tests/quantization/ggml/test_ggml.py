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

from parameterized import parameterized

from transformers import AddedToken, AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer
from transformers.testing_utils import (
    require_gguf,
    require_read_token,
    require_torch_accelerator,
    slow,
    torch_device,
)
from transformers.utils import is_gguf_available, is_torch_available


if is_torch_available():
    import torch

if is_gguf_available():
    from gguf import GGMLQuantizationType as QuantType


@require_gguf
@require_torch_accelerator
@slow
class GgufQuantizationTests(unittest.TestCase):
    """
    Test cases for weights dequantization with GGUF models.
    Note: The quantization names should keep aligned with `GGMLQuantizationType` in gguf-py:
    https://github.com/ggerganov/llama.cpp/blob/4b0c638b9a68f577cb2066b638c9f622d91ee661/gguf-py/gguf/constants.py#L1545-L1576
    So quantization like Q4_K_M or Q4_K_S shouldn't be added to this tests.
    """

    example_text = "Hello"

    def run_gguf_model(self, gguf_model_id: str, gguf_filename: str, expected_text: str):
        tokenizer = AutoTokenizer.from_pretrained(gguf_model_id, gguf_file=gguf_filename)
        model = AutoModelForCausalLM.from_pretrained(gguf_model_id, gguf_file=gguf_filename).to(torch_device)

        text = tokenizer(self.example_text, return_tensors="pt").to(torch_device)
        out = model.generate(**text, max_new_tokens=10)
        self.assertEqual(tokenizer.decode(out[0], skip_special_tokens=True), expected_text)

    @parameterized.expand(
        [
            # standard quants
            ("Q4_0", "Hello, World!\n\nStep 3: Add"),
            ("Q5_0", "Hello, World!\n\n5. Use a library"),
            ("Q8_0", "Hello, World!\n\n5. Use a library"),
        ],
    )
    def test_standard_quants(self, quant_type: str, expected_text: str):
        gguf_model_id = "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF"
        filename_format = "tinyllama-1.1b-chat-v1.0.{quant_type}.gguf"
        gguf_filename = filename_format.format(quant_type=quant_type)
        self.run_gguf_model(gguf_model_id, gguf_filename, expected_text)

    # k-quants
    @parameterized.expand(
        [
            ("Q2_K", "Hello, I'm a 22 year old female"),
            ("Q3_K", "Hello\n\nI am trying to create a simple program that"),
            ("Q4_K", "Hello\n\nI am trying to create a simple program that"),
            ("Q5_K", "Helloveda is a 1999 Indian"),
            ("Q6_K", "Hello\n\nI am trying to create a simple program that"),
        ],
    )
    def test_k_quants(self, quant_type: str, expected_text: str):
        gguf_model_id = "legraphista/Qwen2.5-0.5B-Instruct-IMat-GGUF"
        filename_format = "Qwen2.5-0.5B-Instruct.{quant_type}.gguf"
        gguf_filename = filename_format.format(quant_type=quant_type)
        self.run_gguf_model(gguf_model_id, gguf_filename, expected_text)

    @parameterized.expand(
        [
            # i-matrix
            ("IQ1_S", "Hello, I'm a friend of mine, I"),
            ("IQ1_M", "Hello, I am interested in purching a copy of"),
            ("IQ2_XXS", "Hello, I'm a software engineer. I'"),
            ("IQ2_XS", "Hello World!\n\n```\n<|user|"),
            ("IQ2_S", "Hello World!\n\n```\n<|user|"),
            ("IQ3_XXS", "Hello, I am interested in your product. Can you"),
            ("IQ4_XS", "Hello, world!\n\n5. Using a loop"),
            ("IQ3_S", "Hello, World!\n\n5. Python:\n"),
            ("IQ4_NL", "Hello, world!\n\n5. Using a loop"),
        ],
    )
    def test_imatrix_quants(self, quant_type: str, expected_text: str):
        gguf_model_id = "duyntnet/TinyLlama-1.1B-Chat-v1.0-imatrix-GGUF"
        filename_format = "TinyLlama-1.1B-Chat-v1.0-{quant_type}.gguf"
        gguf_filename = filename_format.format(quant_type=quant_type)
        self.run_gguf_model(gguf_model_id, gguf_filename, expected_text)


@require_gguf
@require_torch_accelerator
@slow
class GgufIntegrationTests(unittest.TestCase):
    """
    Test cases for basic interoperability with GGUF models:
    - Tokenization
    - Model dtype casting and serialization
    """

    example_text = "Hello"
    original_model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    gguf_model_id = "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF"
    gguf_filename = "tinyllama-1.1b-chat-v1.0.{quant_type}.gguf"

    def test_tokenization_xnli(self):
        import tqdm
        from datasets import load_dataset

        q8_0_gguf_model_id = self.gguf_filename.format(quant_type=QuantType.Q8_0.name)
        gguf_tokenizer = AutoTokenizer.from_pretrained(self.gguf_model_id, gguf_file=q8_0_gguf_model_id)
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
        gguf_tokenizer = AutoTokenizer.from_pretrained(self.gguf_model_id, gguf_file=q8_0_gguf_model_id)
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

    def test_q2_k_serialization(self):
        q2_k_gguf_model_id = self.gguf_filename.format(quant_type=QuantType.Q2_K.name)
        EXPECTED_TEXT = "Hello, World!\n\n[10:0"

        tokenizer = AutoTokenizer.from_pretrained(self.gguf_model_id, gguf_file=q2_k_gguf_model_id)
        model = AutoModelForCausalLM.from_pretrained(self.gguf_model_id, gguf_file=q2_k_gguf_model_id).to(torch_device)

        orig_text = tokenizer(self.example_text, return_tensors="pt").to(torch_device)
        orig_out = model.generate(**orig_text, max_new_tokens=10)
        self.assertEqual(tokenizer.decode(orig_out[0], skip_special_tokens=True), EXPECTED_TEXT)

        with tempfile.TemporaryDirectory() as tmpdirname:
            model.save_pretrained(tmpdirname)
            tokenizer.save_pretrained(tmpdirname)

            model = AutoModelForCausalLM.from_pretrained(tmpdirname).to(torch_device)
            tokenizer = AutoTokenizer.from_pretrained(tmpdirname)

            text = tokenizer(self.example_text, return_tensors="pt").to(torch_device)
            out = model.generate(**text, max_new_tokens=10)

        self.assertEqual(tokenizer.decode(out[0], skip_special_tokens=True), EXPECTED_TEXT)

    def test_q6_k_fp16(self):
        q6_k_gguf_model_id = self.gguf_filename.format(quant_type=QuantType.Q6_K.name)

        tokenizer = AutoTokenizer.from_pretrained(self.gguf_model_id, gguf_file=q6_k_gguf_model_id)
        model = AutoModelForCausalLM.from_pretrained(
            self.gguf_model_id, gguf_file=q6_k_gguf_model_id, dtype=torch.float16
        ).to(torch_device)

        self.assertTrue(model.lm_head.weight.dtype == torch.float16)

        text = tokenizer(self.example_text, return_tensors="pt").to(torch_device)
        out = model.generate(**text, max_new_tokens=10)

        EXPECTED_TEXT = "Hello, World!\n\nStep 3: Add"
        self.assertEqual(tokenizer.decode(out[0], skip_special_tokens=True), EXPECTED_TEXT)

    def test_gguf_errors_disk_offload(self):
        from collections import OrderedDict

        q2_k_gguf_model_id = self.gguf_filename.format(quant_type=QuantType.Q2_K.name)
        with self.assertRaises(RuntimeError):
            AutoModelForCausalLM.from_pretrained(
                self.gguf_model_id,
                device_map=OrderedDict(
                    [
                        ("model.embed_tokens", "cpu"),
                        ("lm_head", "cpu"),
                        ("model.layers.0", "cpu"),
                        ("model.layers.1", "cpu"),
                        ("model.layers.2", "cpu"),
                        ("model.layers.3", "cpu"),
                        ("model.layers.4", "cpu"),
                        ("model.layers.5", "cpu"),
                        ("model.layers.6", "cpu"),
                        ("model.layers.7", "cpu"),
                        ("model.layers.8", "cpu"),
                        ("model.layers.9", "cpu"),
                        ("model.layers.10", "disk"),
                        ("model.layers.11", "disk"),
                        ("model.layers.12", "disk"),
                        ("model.layers.13", "disk"),
                        ("model.layers.14", "disk"),
                        ("model.layers.15", "disk"),
                        ("model.layers.16", "disk"),
                        ("model.layers.17", "disk"),
                        ("model.layers.18", "disk"),
                        ("model.layers.19", "disk"),
                        ("model.layers.20", "disk"),
                        ("model.layers.21", "disk"),
                        ("model.layers.22", "disk"),
                        ("model.norm", "disk"),
                        ("model.rotary_emb", "disk"),
                    ]
                ),
                gguf_file=q2_k_gguf_model_id,
                offload_folder="offload",
                offload_state_dict=True,
            )


@require_gguf
@require_torch_accelerator
@slow
class GgufModelTests(unittest.TestCase):
    mistral_model_id = "TheBloke/Mistral-7B-Instruct-v0.2-GGUF"
    qwen2_model_id = "Qwen/Qwen1.5-0.5B-Chat-GGUF"
    qwen2moe_model_id = "gdax/Qwen1.5-MoE-A2.7B_gguf"
    qwen2moe_original_model_id = "Qwen/Qwen1.5-MoE-A2.7B"
    llama3_model_id = "NousResearch/Meta-Llama-3-8B-GGUF"
    tinyllama_model_id = "PenutChen/TinyLlama-1.1B-Chat-v1.0-GGUF"
    phi3_model_id = "microsoft/Phi-3-mini-4k-instruct-gguf"
    bloom_model_id = "afrideva/bloom-560m-GGUF"
    original_bloom_model_id = "bigscience/bloom-560m"
    falcon7b_model_id_q2 = "xaviviro/falcon-7b-quantized-gguf"
    falcon7b_model_id_fp16 = "medmekk/falcon-7b-gguf"
    falcon40b_model_id = "maddes8cht/tiiuae-falcon-40b-gguf"
    original_flacon7b_model_id = "tiiuae/falcon-7b"
    t5_model_id = "repetitio/flan-t5-small"
    original_t5_model_id = "google/flan-t5-small"
    stablelm_model_id = "afrideva/stablelm-3b-4e1t-GGUF"
    stablelm2_model_id = "afrideva/stablelm-2-1_6b-GGUF"
    original_stablelm2_model_id = "stabilityai/stablelm-2-1_6b"
    gpt2_model_id = "mradermacher/gpt2-GGUF"
    gpt2_original_model_id = "openai-community/gpt2"
    gpt2_xl_model_id = "RichardErkhov/openai-community_-_gpt2-xl-gguf"
    starcoder2_model_id = "QuantFactory/starcoder2-3b-GGUF"
    starcoder2_fp16_model_id = "brittlewis12/starcoder2-3b-GGUF"
    starcoder2_original_model_id = "bigcode/starcoder2-3b"
    mamba_original_model_id = "state-spaces/mamba-2.8b-hf"
    mamba_model_id = "jpodivin/mamba-2.8b-hf-GGUF"
    nemotron_original_model_id = "nvidia/Nemotron-Mini-4B-Instruct"
    nemotron_model_id = "bartowski/Nemotron-Mini-4B-Instruct-GGUF"
    original_gemma2_model_id = "google/gemma-2-2b-it"
    gemma2_model_id = "bartowski/gemma-2-2b-it-GGUF"
    original_gemma3_text_model_id = "google/gemma-3-1b-it"
    original_gemma3_vision_model_id = "google/gemma-3-4b-it"
    gemma3_qat_model_id = "google/gemma-3-1b-it-qat-q4_0-gguf"
    gemma3_text_model_id = "unsloth/gemma-3-1b-it-GGUF"
    gemma3_vision_model_id = "unsloth/gemma-3-4b-it-GGUF"
    qwen3_model_id = "Qwen/Qwen3-0.6B-GGUF"
    qwen3moe_model_id = "Qwen/Qwen3-30B-A3B-GGUF"

    q4_0_phi3_model_id = "Phi-3-mini-4k-instruct-q4.gguf"
    q4_0_mistral_model_id = "mistral-7b-instruct-v0.2.Q4_0.gguf"
    q4_0_qwen2_model_id = "qwen1_5-0_5b-chat-q4_0.gguf"
    q8_qwen2moe_model_id = "Qwen1.5-MoE-A2.7B_Q8_0.gguf"
    q4_llama3_model_id = "Meta-Llama-3-8B-Q4_K_M.gguf"
    fp16_bloom_model_id = "bloom-560m.fp16.gguf"
    q4_k_m_stablelm_model_id = "stablelm-3b-4e1t.q4_k_m.gguf"
    fp16_stablelm2_model_id = "stablelm-2-1_6b.fp16.gguf"
    q8_bloom_model_id = "bloom-560m.q8_0.gguf"
    f16_tinyllama_model_id = "TinyLlama-1.1B-Chat-v1.0.FP16.gguf"
    q2_k_falcon7b_model_id = "falcon-7b-q2_k.gguf"
    fp16_falcon7b_model_id = "falcon-7b-fp16.gguf"
    q2_k_falcon40b_model_id = "tiiuae-falcon-40b-Q2_K.gguf"
    fp16_t5_model_id = "flan-t5-small-f16.gguf"
    q8_0_t5_model_id = "flan-t5-small-q8_0.gguf"
    fp16_qwen2moe_model_id = "Qwen1.5-MoE-A2.7B.gguf"
    fp16_gpt2_model_id = "gpt2.f16.gguf"
    q8_gpt2_model_id = "gpt2.Q8_0.gguf"
    q6_k_gpt2_xl_model_id = "gpt2-xl.Q6_K.gguf"
    q6_k_starcoder2_model_id = "starcoder2-3b.Q6_K.gguf"
    fp16_starcoder2_gguf_model_id = "starcoder2-3b.fp16.gguf"
    q6_k_mamba_model_id = "ggml-model-Q6_K.gguf"
    fp16_mamba_model_id = "ggml-model-f16.gguf"
    q6_k_nemotron_model_id = "Nemotron-Mini-4B-Instruct-Q6_K.gguf"
    fp16_nemotron_model_id = "Nemotron-Mini-4B-Instruct-f16.gguf"
    q3_k_gemma2_model_id = "gemma-2-2b-it-Q3_K_L.gguf"
    q8_0_gemma2_model_id = "gemma-2-2b-it-Q8_0.gguf"
    fp32_gemma2_model_id = "gemma-2-2b-it-f32.gguf"
    q4_0_gemma3_qat_model_id = "gemma-3-1b-it-q4_0.gguf"
    bf16_gemma3_text_model_id = "gemma-3-1b-it-BF16.gguf"
    bf16_gemma3_vision_model_id = "gemma-3-4b-it-BF16.gguf"
    deci_original_model_id = "Deci/DeciLM-7B"
    deci_model_id = "Deci/DeciLM-7B-instruct-GGUF"
    q8_0_deci_model_id = "decilm-7b-uniform-gqa-q8_0.gguf"
    fp16_deci_model_id = "decilm-7b-uniform-gqa-f16.gguf"
    q8_0_qwen3_model_id = "Qwen3-0.6B-Q8_0.gguf"
    q4_k_m_qwen3moe_model_id = "Qwen3-30B-A3B-Q4_K_M.gguf"

    example_text = "Hello"

    def test_mistral_q4_0(self):
        tokenizer = AutoTokenizer.from_pretrained(self.mistral_model_id, gguf_file=self.q4_0_mistral_model_id)
        model = AutoModelForCausalLM.from_pretrained(
            self.mistral_model_id,
            gguf_file=self.q4_0_mistral_model_id,
            device_map="auto",
            dtype=torch.float16,
        )

        text = tokenizer(self.example_text, return_tensors="pt").to(torch_device)
        out = model.generate(**text, max_new_tokens=10)

        EXPECTED_TEXT = "Hello,\n\nI'm trying to create a"
        self.assertEqual(tokenizer.decode(out[0], skip_special_tokens=True), EXPECTED_TEXT)

    def test_qwen2_q4_0(self):
        tokenizer = AutoTokenizer.from_pretrained(self.qwen2_model_id, gguf_file=self.q4_0_qwen2_model_id)
        model = AutoModelForCausalLM.from_pretrained(
            self.qwen2_model_id,
            gguf_file=self.q4_0_qwen2_model_id,
            device_map="auto",
            dtype=torch.float16,
        )

        text = tokenizer(self.example_text, return_tensors="pt").to(torch_device)
        out = model.generate(**text, max_new_tokens=10)

        EXPECTED_TEXT = "Hello.jsoup\n\nI am a beginner"
        self.assertEqual(tokenizer.decode(out[0], skip_special_tokens=True), EXPECTED_TEXT)

    def test_qwen2moe_q8(self):
        tokenizer = AutoTokenizer.from_pretrained(self.qwen2moe_model_id, gguf_file=self.q8_qwen2moe_model_id)
        model = AutoModelForCausalLM.from_pretrained(
            self.qwen2moe_model_id,
            gguf_file=self.q8_qwen2moe_model_id,
            dtype=torch.float16,
        )

        text = tokenizer(self.example_text, return_tensors="pt")
        out = model.generate(**text, max_new_tokens=10)

        EXPECTED_TEXT = "Hello, I am a 20 year old male"
        self.assertEqual(tokenizer.decode(out[0], skip_special_tokens=True), EXPECTED_TEXT)

    def test_qwen2moe_weights_conversion_fp16(self):
        quantized_model = AutoModelForCausalLM.from_pretrained(
            self.qwen2moe_model_id,
            gguf_file=self.fp16_qwen2moe_model_id,
            dtype=torch.float16,
        )
        original_model = AutoModelForCausalLM.from_pretrained(
            self.qwen2moe_original_model_id,
            dtype=torch.float16,
        )

        quantized_state_dict = quantized_model.state_dict()
        original_state_dict = original_model.state_dict()

        for layer_name, original_params in original_state_dict.items():
            if layer_name in quantized_state_dict:
                self.assertTrue(original_params.shape == quantized_state_dict[layer_name].shape)
                torch.testing.assert_close(original_params, quantized_state_dict[layer_name])

    def test_phi3_q4_0(self):
        tokenizer = AutoTokenizer.from_pretrained(self.phi3_model_id, gguf_file=self.q4_0_phi3_model_id)
        model = AutoModelForCausalLM.from_pretrained(
            self.phi3_model_id, gguf_file=self.q4_0_phi3_model_id, device_map="auto", dtype=torch.float16
        )

        text = tokenizer(self.example_text, return_tensors="pt").to(torch_device)
        out = model.generate(**text, max_new_tokens=10)

        EXPECTED_TEXT = "Hello, I've been reading about the impact of"
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
            self.llama3_model_id,
            gguf_file=self.q4_llama3_model_id,
            device_map="auto",
            dtype=torch.float16,
        )

        text = tokenizer(self.example_text, return_tensors="pt").to(torch_device)
        out = model.generate(**text, max_new_tokens=10)

        EXPECTED_TEXT = "Hello, I am interested in [The Park]\nThe"
        self.assertEqual(tokenizer.decode(out[0], skip_special_tokens=True), EXPECTED_TEXT)

    def test_bloom_fp16(self):
        tokenizer = AutoTokenizer.from_pretrained(self.bloom_model_id, gguf_file=self.fp16_bloom_model_id)
        model = AutoModelForCausalLM.from_pretrained(
            self.bloom_model_id,
            gguf_file=self.fp16_bloom_model_id,
            device_map="auto",
            dtype=torch.float16,
        )

        text = tokenizer(self.example_text, return_tensors="pt").to(torch_device)
        out = model.generate(**text, max_new_tokens=10)

        EXPECTED_TEXT = "Hello, I just want to say that I am very"
        self.assertEqual(tokenizer.decode(out[0], skip_special_tokens=True), EXPECTED_TEXT)

    def test_bloom_q8_0(self):
        tokenizer = AutoTokenizer.from_pretrained(self.bloom_model_id, gguf_file=self.q8_bloom_model_id)
        model = AutoModelForCausalLM.from_pretrained(
            self.bloom_model_id,
            gguf_file=self.q8_bloom_model_id,
            device_map="auto",
            dtype=torch.float16,
        )

        text = tokenizer(self.example_text, return_tensors="pt").to(torch_device)
        out = model.generate(**text, max_new_tokens=10)

        EXPECTED_TEXT = "Hello, I just want to say that I am just"
        self.assertEqual(tokenizer.decode(out[0], skip_special_tokens=True), EXPECTED_TEXT)

    def test_bloom_weights_conversion_fp16(self):
        quantized_model = AutoModelForCausalLM.from_pretrained(
            self.bloom_model_id,
            gguf_file=self.fp16_bloom_model_id,
            device_map="auto",
            dtype=torch.float16,
        )
        original_model = AutoModelForCausalLM.from_pretrained(
            self.original_bloom_model_id,
            device_map="auto",
            dtype=torch.float16,
        )

        quantized_state_dict = quantized_model.state_dict()
        original_state_dict = original_model.state_dict()

        for (quantized_name, quantized_param), (original_name, original_param) in zip(
            quantized_state_dict.items(), original_state_dict.items()
        ):
            if (
                "self_attention.query_key_value" in quantized_name
                and "self_attention.query_key_value" in original_name
            ):
                self.assertTrue(quantized_param.shape == original_param.shape)
                torch.testing.assert_close(quantized_param, original_param)

    def test_t5_f16(self):
        tokenizer = AutoTokenizer.from_pretrained(self.t5_model_id, gguf_file=self.fp16_t5_model_id)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            self.t5_model_id, gguf_file=self.fp16_t5_model_id, device_map="auto", dtype=torch.float16
        )

        T5_EXAMPLE_TEXT = "translate English to German: How old are you?"

        text = tokenizer(T5_EXAMPLE_TEXT, return_tensors="pt").to(torch_device)
        out = model.generate(**text, max_new_tokens=10)

        EXPECTED_TEXT = "Wie ich er?"
        self.assertEqual(tokenizer.decode(out[0], skip_special_tokens=True), EXPECTED_TEXT)

    def test_t5_q8_0(self):
        tokenizer = AutoTokenizer.from_pretrained(self.t5_model_id, gguf_file=self.q8_0_t5_model_id)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            self.t5_model_id, gguf_file=self.q8_0_t5_model_id, device_map="auto", dtype=torch.float16
        )

        T5_EXAMPLE_TEXT = "translate English to German: How old are you?"

        text = tokenizer(T5_EXAMPLE_TEXT, return_tensors="pt").to(torch_device)
        out = model.generate(**text, max_new_tokens=10)

        EXPECTED_TEXT = "Wie ich er?"
        self.assertEqual(tokenizer.decode(out[0], skip_special_tokens=True), EXPECTED_TEXT)

    def test_t5_weights_conversion_fp16(self):
        quantized_model = AutoModelForSeq2SeqLM.from_pretrained(
            self.t5_model_id,
            gguf_file=self.fp16_t5_model_id,
            device_map="auto",
            dtype=torch.float16,
        )
        original_model = AutoModelForSeq2SeqLM.from_pretrained(
            self.original_t5_model_id,
            device_map="auto",
            dtype=torch.float16,
        )

        quantized_state_dict = quantized_model.state_dict()
        original_state_dict = original_model.state_dict()

        for (quantized_name, quantized_param), (original_name, original_param) in zip(
            quantized_state_dict.items(), original_state_dict.items()
        ):
            self.assertTrue(quantized_param.shape == original_param.shape)
            torch.testing.assert_close(quantized_param, original_param, rtol=5e-04, atol=5e-04)

    def test_gpt2_q8(self):
        tokenizer = AutoTokenizer.from_pretrained(self.gpt2_model_id, gguf_file=self.q8_gpt2_model_id)
        model = AutoModelForCausalLM.from_pretrained(
            self.gpt2_model_id,
            gguf_file=self.q8_gpt2_model_id,
            dtype=torch.float16,
        )

        text = tokenizer(self.example_text, return_tensors="pt")
        out = model.generate(**text, max_new_tokens=10)

        EXPECTED_TEXT = "Hello, I'm sorry. I'm sorry. I"
        self.assertEqual(tokenizer.decode(out[0], skip_special_tokens=True), EXPECTED_TEXT)

    def test_gpt2_weights_conversion_fp16(self):
        quantized_model = AutoModelForCausalLM.from_pretrained(
            self.gpt2_model_id,
            gguf_file=self.fp16_gpt2_model_id,
            dtype=torch.float16,
        )
        original_model = AutoModelForCausalLM.from_pretrained(
            self.gpt2_original_model_id,
            dtype=torch.float16,
        )

        quantized_state_dict = quantized_model.state_dict()
        original_state_dict = original_model.state_dict()

        for layer_name, original_params in original_state_dict.items():
            if layer_name in quantized_state_dict:
                self.assertTrue(original_params.shape == quantized_state_dict[layer_name].shape)
                torch.testing.assert_close(original_params, quantized_state_dict[layer_name])
            else:
                raise ValueError(f"Layer {layer_name} is not presented in GGUF model")

    def test_gpt2_xl_Q6_K(self):
        tokenizer = AutoTokenizer.from_pretrained(self.gpt2_xl_model_id, gguf_file=self.q6_k_gpt2_xl_model_id)
        model = AutoModelForCausalLM.from_pretrained(
            self.gpt2_xl_model_id,
            gguf_file=self.q6_k_gpt2_xl_model_id,
            dtype=torch.float16,
        )

        text = tokenizer(self.example_text, return_tensors="pt")
        out = model.generate(**text, max_new_tokens=10)

        EXPECTED_TEXT = "Hello, I'm a newbie to the world of"
        self.assertEqual(tokenizer.decode(out[0], skip_special_tokens=True), EXPECTED_TEXT)

    @unittest.skip(reason="Heavy memory")
    def test_falcon40b_q2_k(self):
        tokenizer = AutoTokenizer.from_pretrained(self.falcon40b_model_id, gguf_file=self.q2_k_falcon40b_model_id)
        model = AutoModelForCausalLM.from_pretrained(
            self.falcon40b_model_id,
            gguf_file=self.q2_k_falcon40b_model_id,
            device_map="auto",
            dtype=torch.float16,
        )

        text = tokenizer(self.example_text, return_tensors="pt").to(torch_device)
        out = model.generate(**text, max_new_tokens=10)

        EXPECTED_TEXT = "Hello All,\nI am new to this forum."
        self.assertEqual(tokenizer.decode(out[0], skip_special_tokens=True), EXPECTED_TEXT)

    def test_falcon7b_q2_k(self):
        tokenizer = AutoTokenizer.from_pretrained(self.falcon7b_model_id_q2, gguf_file=self.q2_k_falcon7b_model_id)
        model = AutoModelForCausalLM.from_pretrained(
            self.falcon7b_model_id_q2,
            gguf_file=self.q2_k_falcon7b_model_id,
            device_map="auto",
            dtype=torch.float16,
        )

        text = tokenizer(self.example_text, return_tensors="pt")["input_ids"].to(torch_device)
        out = model.generate(text, max_new_tokens=16)

        EXPECTED_TEXT = "Hello All,\nI am new to this forum.\nI am using the "
        self.assertEqual(tokenizer.decode(out[0], skip_special_tokens=True), EXPECTED_TEXT)

    @unittest.skip("The test causes a torch.OutOfMemoryError on the CI but it passes with enough memory")
    def test_falcon7b_weights_conversion_fp16(self):
        quantized_model = AutoModelForCausalLM.from_pretrained(
            self.falcon7b_model_id_fp16,
            gguf_file=self.fp16_falcon7b_model_id,
            device_map="auto",
            dtype=torch.float16,
        )
        original_model = AutoModelForCausalLM.from_pretrained(
            self.original_flacon7b_model_id,
            device_map="auto",
            dtype=torch.float16,
        )

        quantized_state_dict = quantized_model.state_dict()
        original_state_dict = original_model.state_dict()

        for layer_name, original_params in original_state_dict.items():
            if layer_name in quantized_state_dict:
                self.assertTrue(original_params.shape == quantized_state_dict[layer_name].shape)
                torch.testing.assert_close(original_params, quantized_state_dict[layer_name])
            else:
                raise ValueError(f"Layer {layer_name} is not presented in GGUF model")

    def test_stablelm_q4_k_m(self):
        model = AutoModelForCausalLM.from_pretrained(
            self.stablelm_model_id,
            gguf_file=self.q4_k_m_stablelm_model_id,
            device_map="auto",
            dtype=torch.float16,
        )

        tokenizer = AutoTokenizer.from_pretrained(self.stablelm_model_id, gguf_file=self.q4_k_m_stablelm_model_id)
        text = tokenizer(self.example_text, return_tensors="pt").to(torch_device)
        out = model.generate(**text, max_new_tokens=10)

        EXPECTED_TEXT = "Hello-\nI am trying to create a new user"
        self.assertEqual(tokenizer.decode(out[0], skip_special_tokens=True), EXPECTED_TEXT)

    def test_stablelm_fp16(self):
        original_model = AutoModelForCausalLM.from_pretrained(
            self.original_stablelm2_model_id,
            dtype=torch.float16,
        )

        converted_model = AutoModelForCausalLM.from_pretrained(
            self.stablelm2_model_id,
            gguf_file=self.fp16_stablelm2_model_id,
            dtype=torch.float16,
        )

        tokenizer = AutoTokenizer.from_pretrained(self.stablelm2_model_id, gguf_file=self.fp16_stablelm2_model_id)
        text = tokenizer(self.example_text, return_tensors="pt")
        original_out = original_model.generate(**text, max_new_tokens=10)
        converted_out = converted_model.generate(**text, max_new_tokens=10)

        EXPECTED_TEXT = "Hello, I am a 20 year old male"
        self.assertEqual(tokenizer.decode(converted_out[0], skip_special_tokens=True), EXPECTED_TEXT)
        self.assertEqual(
            tokenizer.decode(converted_out[0], skip_special_tokens=True),
            tokenizer.decode(original_out[0], skip_special_tokens=True),
        )

    def test_stablelm_weights_conversion_fp16(self):
        original_model = AutoModelForCausalLM.from_pretrained(
            self.original_stablelm2_model_id,
            device_map="auto",
            dtype=torch.float16,
        )

        converted_model = AutoModelForCausalLM.from_pretrained(
            self.stablelm2_model_id,
            gguf_file=self.fp16_stablelm2_model_id,
            device_map="auto",
            dtype=torch.float16,
        )

        converted_state_dict = converted_model.state_dict()
        original_state_dict = original_model.state_dict()

        for layer_name, original_params in original_state_dict.items():
            if layer_name in converted_state_dict:
                self.assertTrue(original_params.shape == converted_state_dict[layer_name].shape)
                torch.testing.assert_close(original_params, converted_state_dict[layer_name])
            else:
                raise ValueError(f"Layer {layer_name} is not presented in GGUF model")

    def test_starcoder2_weights_conversion_fp16(self):
        original_model = AutoModelForCausalLM.from_pretrained(
            self.starcoder2_original_model_id,
            device_map="auto",
            dtype=torch.float16,
        )

        converted_model = AutoModelForCausalLM.from_pretrained(
            self.starcoder2_fp16_model_id,
            gguf_file=self.fp16_starcoder2_gguf_model_id,
            device_map="auto",
            dtype=torch.float16,
        )

        converted_state_dict = converted_model.state_dict()
        original_state_dict = original_model.state_dict()

        for layer_name, original_params in original_state_dict.items():
            if layer_name in converted_state_dict:
                self.assertTrue(original_params.shape == converted_state_dict[layer_name].shape)
                torch.testing.assert_close(original_params, converted_state_dict[layer_name])
            else:
                raise ValueError(f"Layer {layer_name} is not presented in GGUF model")

    def test_starcoder2_q6_k(self):
        example_function_text = "def print_hello_world():"
        model = AutoModelForCausalLM.from_pretrained(
            self.starcoder2_model_id,
            gguf_file=self.q6_k_starcoder2_model_id,
            device_map="auto",
            dtype=torch.float16,
        )

        tokenizer = AutoTokenizer.from_pretrained(self.starcoder2_model_id, gguf_file=self.q6_k_starcoder2_model_id)
        text = tokenizer(example_function_text, return_tensors="pt").to(torch_device)
        out = model.generate(**text, max_new_tokens=10)

        EXPECTED_TEXT = 'def print_hello_world():\n    print("Hello World")\n\ndef print'
        self.assertEqual(tokenizer.decode(out[0], skip_special_tokens=True), EXPECTED_TEXT)

    def test_mamba_weights_conversion_fp16(self):
        original_model = AutoModelForCausalLM.from_pretrained(
            self.mamba_original_model_id,
            dtype=torch.float16,
        )

        converted_model = AutoModelForCausalLM.from_pretrained(
            self.mamba_model_id,
            gguf_file=self.fp16_mamba_model_id,
            dtype=torch.float16,
        )

        converted_state_dict = converted_model.state_dict()
        original_state_dict = original_model.state_dict()

        for layer_name, original_params in original_state_dict.items():
            if layer_name in converted_state_dict:
                self.assertTrue(original_params.shape == converted_state_dict[layer_name].shape)
                if "mixer.A_log" in layer_name:
                    # we should increase tolerance after exponential reversing
                    # and performing np.log(-weights) operation as numbers are slightly different
                    torch.testing.assert_close(original_params, converted_state_dict[layer_name], rtol=1e-3, atol=1e-3)
                else:
                    torch.testing.assert_close(original_params, converted_state_dict[layer_name])
            else:
                raise ValueError(f"Layer {layer_name} is not presented in GGUF model")

    def test_mamba_q6_k(self):
        model = AutoModelForCausalLM.from_pretrained(
            self.mamba_model_id,
            gguf_file=self.q6_k_mamba_model_id,
            dtype=torch.float16,
        )

        tokenizer = AutoTokenizer.from_pretrained(self.mamba_model_id, gguf_file=self.q6_k_mamba_model_id)
        text = tokenizer(self.example_text, return_tensors="pt")["input_ids"]
        out = model.generate(text, max_new_tokens=10)

        EXPECTED_TEXT = "Hello,I answerthe question.\n\nA"
        self.assertEqual(tokenizer.decode(out[0], skip_special_tokens=True), EXPECTED_TEXT)

    def test_nemotron_weights_conversion_fp16(self):
        original_model = AutoModelForCausalLM.from_pretrained(
            self.nemotron_original_model_id,
            dtype=torch.float16,
        )

        converted_model = AutoModelForCausalLM.from_pretrained(
            self.nemotron_model_id,
            gguf_file=self.fp16_nemotron_model_id,
            dtype=torch.float16,
        )

        converted_state_dict = converted_model.state_dict()
        original_state_dict = original_model.state_dict()

        for layer_name, original_params in original_state_dict.items():
            if layer_name in converted_state_dict:
                self.assertTrue(original_params.shape == converted_state_dict[layer_name].shape)
                torch.testing.assert_close(original_params, converted_state_dict[layer_name])
            else:
                raise ValueError(f"Layer {layer_name} is not presented in GGUF model")

    def test_nemotron_q6_k(self):
        model = AutoModelForCausalLM.from_pretrained(
            self.nemotron_model_id,
            gguf_file=self.q6_k_nemotron_model_id,
            dtype=torch.float16,
        )

        tokenizer = AutoTokenizer.from_pretrained(self.nemotron_model_id, gguf_file=self.q6_k_nemotron_model_id)
        text = tokenizer(self.example_text, return_tensors="pt")["input_ids"]
        out = model.generate(text, max_new_tokens=16)

        EXPECTED_TEXT = "Hello.▁hotmail.com</s>"
        self.assertEqual(tokenizer.decode(out[0], skip_special_tokens=True), EXPECTED_TEXT)

    def test_gemma2_q3_k(self):
        model = AutoModelForCausalLM.from_pretrained(
            self.gemma2_model_id,
            gguf_file=self.q3_k_gemma2_model_id,
            dtype=torch.float16,
        )

        tokenizer = AutoTokenizer.from_pretrained(self.gemma2_model_id, gguf_file=self.q3_k_gemma2_model_id)
        text = tokenizer(self.example_text, return_tensors="pt")["input_ids"]
        out = model.generate(text, max_new_tokens=10)

        EXPECTED_TEXT = "Hello! 👋\n\nI'm trying to create a"
        self.assertEqual(tokenizer.decode(out[0], skip_special_tokens=True), EXPECTED_TEXT)

    def test_gemma2_q8_0(self):
        model = AutoModelForCausalLM.from_pretrained(
            self.gemma2_model_id,
            gguf_file=self.q8_0_gemma2_model_id,
            dtype=torch.float16,
        )

        tokenizer = AutoTokenizer.from_pretrained(self.gemma2_model_id, gguf_file=self.q8_0_gemma2_model_id)
        text = tokenizer(self.example_text, return_tensors="pt")["input_ids"]
        out = model.generate(text, max_new_tokens=10)

        EXPECTED_TEXT = "Hello! 👋\n\nI'm a large language model"
        self.assertEqual(tokenizer.decode(out[0], skip_special_tokens=True), EXPECTED_TEXT)

    def test_gemma2_fp32(self):
        model = AutoModelForCausalLM.from_pretrained(
            self.gemma2_model_id,
            gguf_file=self.fp32_gemma2_model_id,
            dtype=torch.float16,
        )

        tokenizer = AutoTokenizer.from_pretrained(self.gemma2_model_id, gguf_file=self.fp32_gemma2_model_id)
        text = tokenizer(self.example_text, return_tensors="pt")["input_ids"]
        out = model.generate(text, max_new_tokens=10)

        EXPECTED_TEXT = "Hello! 👋\n\nI'm a large language model"
        self.assertEqual(tokenizer.decode(out[0], skip_special_tokens=True), EXPECTED_TEXT)

    @require_read_token
    def test_gemma2_weights_conversion_fp32(self):
        original_model = AutoModelForCausalLM.from_pretrained(
            self.original_gemma2_model_id,
            dtype=torch.float16,
        )

        converted_model = AutoModelForCausalLM.from_pretrained(
            self.gemma2_model_id,
            gguf_file=self.fp32_gemma2_model_id,
            dtype=torch.float16,
        )

        converted_state_dict = converted_model.state_dict()
        original_state_dict = original_model.state_dict()

        for layer_name, original_params in original_state_dict.items():
            if layer_name in converted_state_dict:
                self.assertTrue(original_params.shape == converted_state_dict[layer_name].shape)
                torch.testing.assert_close(original_params, converted_state_dict[layer_name])
            else:
                raise ValueError(f"Layer {layer_name} is not presented in GGUF model")

    @require_read_token
    @unittest.skipUnless(is_gguf_available("0.16.0"), "test requires gguf version >= 0.16.0")
    def test_gemma3_qat_q4_0(self):
        model = AutoModelForCausalLM.from_pretrained(
            self.gemma3_qat_model_id,
            gguf_file=self.q4_0_gemma3_qat_model_id,
            dtype=torch.float16,
        )

        tokenizer = AutoTokenizer.from_pretrained(self.gemma3_qat_model_id, gguf_file=self.q4_0_gemma3_qat_model_id)
        text = tokenizer(self.example_text, return_tensors="pt")["input_ids"]
        out = model.generate(text, max_new_tokens=10)

        EXPECTED_TEXT = 'Hello with the prompt, "What is the best way'

        self.assertEqual(tokenizer.decode(out[0], skip_special_tokens=True), EXPECTED_TEXT)

    @require_read_token
    @unittest.skipUnless(is_gguf_available("0.16.0"), "test requires gguf version >= 0.16.0")
    def test_gemma3_text_weights_conversion_bf16(self):
        original_model = AutoModelForCausalLM.from_pretrained(
            self.original_gemma3_text_model_id,
            dtype=torch.float16,
        )

        converted_model = AutoModelForCausalLM.from_pretrained(
            self.gemma3_text_model_id,
            gguf_file=self.bf16_gemma3_text_model_id,
            dtype=torch.float16,
        )

        converted_state_dict = converted_model.state_dict()
        original_state_dict = original_model.state_dict()

        for layer_name, original_params in original_state_dict.items():
            if layer_name in converted_state_dict:
                self.assertTrue(original_params.shape == converted_state_dict[layer_name].shape)
                torch.testing.assert_close(original_params, converted_state_dict[layer_name])
            else:
                raise ValueError(f"Layer {layer_name} is not presented in GGUF model")

    # Test text backbone conversion for gemma3 vision models
    @require_read_token
    @unittest.skipUnless(is_gguf_available("0.16.0"), "test requires gguf version >= 0.16.0")
    def test_gemma3_vision_weights_conversion_bf16(self):
        original_model = AutoModelForCausalLM.from_pretrained(
            self.original_gemma3_vision_model_id,
            dtype=torch.float16,
        ).language_model

        converted_model = AutoModelForCausalLM.from_pretrained(
            self.gemma3_vision_model_id,
            gguf_file=self.bf16_gemma3_vision_model_id,
            dtype=torch.float16,
        )

        converted_state_dict = converted_model.state_dict()
        original_state_dict = original_model.state_dict()

        for layer_name, original_params in original_state_dict.items():
            if layer_name in converted_state_dict:
                self.assertTrue(original_params.shape == converted_state_dict[layer_name].shape)
                torch.testing.assert_close(original_params, converted_state_dict[layer_name])
            else:
                raise ValueError(f"Layer {layer_name} is not presented in GGUF model")

    def test_deci_q8_0(self):
        """Test Deci model loading and inference with Q4_0 quantization."""
        tokenizer = AutoTokenizer.from_pretrained(self.deci_model_id, gguf_file=self.q8_0_deci_model_id)
        model = AutoModelForCausalLM.from_pretrained(
            self.deci_model_id,
            gguf_file=self.q8_0_deci_model_id,
            device_map="auto",
            torch_dtype=torch.float16,
        )

        text = tokenizer(self.example_text, return_tensors="pt").to(torch_device)
        out = model.generate(**text, max_new_tokens=10)

        EXPECTED_TEXT = "Hello, I am a language model developed"
        self.assertEqual(tokenizer.decode(out[0], skip_special_tokens=True), EXPECTED_TEXT)

    def test_deci_weights_conversion_fp16(self):
        """Test that GGUF Deci model weights match the original model weights."""
        original_model_id = "Deci/DeciLM-7B"
        original_model = AutoModelForCausalLM.from_pretrained(
            original_model_id,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            device_map="auto",
        )
        # You need to have an FP16 version of your GGUF model for accurate comparison

        converted_model = AutoModelForCausalLM.from_pretrained(
            self.deci_model_id,
            gguf_file=self.fp16_deci_model_id,
            torch_dtype=torch.float16,
            device_map="auto",
        )

        converted_state_dict = converted_model.state_dict()
        original_state_dict = original_model.state_dict()

        for layer_name, original_params in original_state_dict.items():
            if layer_name in converted_state_dict:
                self.assertTrue(original_params.shape == converted_state_dict[layer_name].shape)
                torch.testing.assert_close(original_params, converted_state_dict[layer_name])
            else:
                raise ValueError(f"Layer {layer_name} is not presented in GGUF model")

    def test_deci_config_mapping(self):
        """Test that Deci GGUF config mapping is correctly applied."""
        from transformers.integrations.ggml import GGUF_CONFIG_MAPPING

        self.assertIn("deci", GGUF_CONFIG_MAPPING)

        deci_mapping = GGUF_CONFIG_MAPPING["deci"]

        expected_mappings = {
            "context_length": "max_position_embeddings",
            "block_count": "num_hidden_layers",
            "feed_forward_length": "intermediate_size",
            "embedding_length": "hidden_size",
            "rope.freq_base": "rope_theta",
            "attention.head_count": "num_attention_heads",
            "attention.head_count_kv": "num_key_value_heads",
            "attention.layer_norm_rms_epsilon": "rms_norm_eps",
            "vocab_size": "vocab_size",
        }

        for gguf_key, transformers_key in expected_mappings.items():
            self.assertEqual(deci_mapping[gguf_key], transformers_key)

        self.assertIsNone(deci_mapping["rope.dimension_count"])

    def test_deci_architecture_mapping(self):
        """Test that Deci architectures are mapped to GGUFLlamaConverter."""
        from transformers.integrations.ggml import GGUF_TO_FAST_CONVERTERS, GGUFLlamaConverter

        self.assertIn("deci", GGUF_TO_FAST_CONVERTERS)
        self.assertIn("decilm", GGUF_TO_FAST_CONVERTERS)

        self.assertEqual(GGUF_TO_FAST_CONVERTERS["deci"], GGUFLlamaConverter)
        self.assertEqual(GGUF_TO_FAST_CONVERTERS["decilm"], GGUFLlamaConverter)

    @require_read_token
    @unittest.skipUnless(is_gguf_available("0.16.0"), "test requires gguf version >= 0.16.0")
    def test_qwen3_q8_0(self):
        tokenizer = AutoTokenizer.from_pretrained(self.qwen3_model_id, gguf_file=self.q8_0_qwen3_model_id)
        model = AutoModelForCausalLM.from_pretrained(
            self.qwen3_model_id,
            gguf_file=self.q8_0_qwen3_model_id,
            dtype=torch.float16,
        )

        text = tokenizer(self.example_text, return_tensors="pt")["input_ids"]
        out = model.generate(text, max_new_tokens=10)

        EXPECTED_TEXT = "HelloED\nI need to find the value of the"
        self.assertEqual(tokenizer.decode(out[0], skip_special_tokens=True), EXPECTED_TEXT)

    def test_qwen3moe_q4_k_m(self):
        tokenizer = AutoTokenizer.from_pretrained(self.qwen3moe_model_id, gguf_file=self.q4_k_m_qwen3moe_model_id)
        model = AutoModelForCausalLM.from_pretrained(
            self.qwen3moe_model_id,
            gguf_file=self.q4_k_m_qwen3moe_model_id,
            dtype=torch.float16,
        )

        text = tokenizer(self.example_text, return_tensors="pt")
        out = model.generate(**text, max_new_tokens=10)

        EXPECTED_TEXT = "Hello, I am a 20 year old male"
        self.assertEqual(tokenizer.decode(out[0], skip_special_tokens=True), EXPECTED_TEXT)
