# coding=utf-8
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
"""Testing suite for the PyTorch Gemma2 model."""

import unittest

import pytest
from packaging import version

from transformers import AutoModelForCausalLM, AutoTokenizer, Gemma2Config, is_torch_available
from transformers.testing_utils import (
    require_bitsandbytes,
    require_flash_attn,
    require_read_token,
    require_torch,
    require_torch_gpu,
    require_torch_sdpa,
    slow,
    torch_device,
)

from ...models.gemma.test_modeling_gemma import GemmaModelTest, GemmaModelTester
from ...test_configuration_common import ConfigTester


if is_torch_available():
    import torch

    from transformers import (
        Gemma2ForCausalLM,
        Gemma2ForSequenceClassification,
        Gemma2ForTokenClassification,
        Gemma2Model,
        GemmaTokenizer,
    )


class Gemma2ModelTester(GemmaModelTester):
    config_class = Gemma2Config
    model_class = Gemma2Model
    for_causal_lm_class = Gemma2ForCausalLM
    for_sequence_class = Gemma2ForSequenceClassification
    for_token_class = Gemma2ForTokenClassification


@require_torch
class Gemma2ModelTest(GemmaModelTest, unittest.TestCase):
    all_model_classes = (
        (Gemma2Model, Gemma2ForCausalLM, Gemma2ForSequenceClassification, Gemma2ForTokenClassification)
        if is_torch_available()
        else ()
    )
    all_generative_model_classes = ()
    pipeline_model_mapping = (
        {
            "feature-extraction": Gemma2Model,
            "text-classification": Gemma2ForSequenceClassification,
            "token-classification": Gemma2ForTokenClassification,
            "text-generation": Gemma2ForCausalLM,
            "zero-shot": Gemma2ForSequenceClassification,
        }
        if is_torch_available()
        else {}
    )
    test_headmasking = False
    test_pruning = False
    _is_stateful = True
    model_split_percents = [0.5, 0.6]
    _torch_compile_test_ckpt = "google/gemma-2-9b"

    def setUp(self):
        self.model_tester = Gemma2ModelTester(self)
        self.config_tester = ConfigTester(self, config_class=Gemma2Config, hidden_size=37)

    @unittest.skip("Eager and SDPA do not produce the same outputs, thus this test fails")
    def test_model_outputs_equivalence(self, **kwargs):
        pass

    @unittest.skip("Gemma2's outputs are expected to be different")
    def test_eager_matches_sdpa_inference(self):
        pass


@slow
@require_torch_gpu
class Gemma2IntegrationTest(unittest.TestCase):
    input_text = ["Hello I am doing", "Hi today"]
    # This variable is used to determine which CUDA device are we using for our runners (A10 or T4)
    # Depending on the hardware we get different logits / generations
    cuda_compute_capability_major_version = None

    @classmethod
    def setUpClass(cls):
        if is_torch_available() and torch.cuda.is_available():
            # 8 is for A100 / A10 and 7 for T4
            cls.cuda_compute_capability_major_version = torch.cuda.get_device_capability()[0]

    @require_read_token
    def test_model_2b_fp32(self):
        model_id = "google/gemma-2b"
        EXPECTED_TEXTS = [
            "Hello I am doing a project on the 1990s and I need to know what the most popular music",
            "Hi today I am going to share with you a very easy and simple recipe of <strong><em>Kaju Kat",
        ]

        model = AutoModelForCausalLM.from_pretrained(model_id, low_cpu_mem_usage=True).to(torch_device)

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        inputs = tokenizer(self.input_text, return_tensors="pt", padding=True).to(torch_device)

        output = model.generate(**inputs, max_new_tokens=20, do_sample=False)
        output_text = tokenizer.batch_decode(output, skip_special_tokens=True)

        self.assertEqual(output_text, EXPECTED_TEXTS)

    @require_read_token
    def test_model_2b_fp16(self):
        model_id = "google/gemma-2b"
        EXPECTED_TEXTS = [
            "Hello I am doing a project on the 1990s and I need to know what the most popular music",
            "Hi today I am going to share with you a very easy and simple recipe of <strong><em>Kaju Kat",
        ]

        model = AutoModelForCausalLM.from_pretrained(model_id, low_cpu_mem_usage=True, torch_dtype=torch.float16).to(
            torch_device
        )

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        inputs = tokenizer(self.input_text, return_tensors="pt", padding=True).to(torch_device)

        output = model.generate(**inputs, max_new_tokens=20, do_sample=False)
        output_text = tokenizer.batch_decode(output, skip_special_tokens=True)

        self.assertEqual(output_text, EXPECTED_TEXTS)

    @require_read_token
    def test_model_2b_fp16_static_cache(self):
        model_id = "google/gemma-2b"
        EXPECTED_TEXTS = [
            "Hello I am doing a project on the 1990s and I need to know what the most popular music",
            "Hi today I am going to share with you a very easy and simple recipe of <strong><em>Kaju Kat",
        ]

        model = AutoModelForCausalLM.from_pretrained(model_id, low_cpu_mem_usage=True, torch_dtype=torch.float16).to(
            torch_device
        )

        model.generation_config.cache_implementation = "static"

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        inputs = tokenizer(self.input_text, return_tensors="pt", padding=True).to(torch_device)

        output = model.generate(**inputs, max_new_tokens=20, do_sample=False)
        output_text = tokenizer.batch_decode(output, skip_special_tokens=True)

        self.assertEqual(output_text, EXPECTED_TEXTS)

    @require_read_token
    def test_model_2b_bf16(self):
        model_id = "google/gemma-2b"

        # Key 9 for MI300, Key 8 for A100/A10, and Key 7 for T4.
        #
        # Note: Key 9 is currently set for MI300, but may need potential future adjustments for H100s,
        # considering differences in hardware processing and potential deviations in generated text.
        EXPECTED_TEXTS = {
            7: [
                "Hello I am doing a project on the 1990s and I need to know what the most popular music",
                "Hi today I am going to share with you a very easy and simple recipe of <strong><em>Khichdi",
            ],
            8: [
                "Hello I am doing a project on the 1990s and I need to know what the most popular music",
                "Hi today I am going to share with you a very easy and simple recipe of <strong><em>Kaju Kat",
            ],
            9: [
                "Hello I am doing a project on the 1990s and I need to know what the most popular music",
                "Hi today I am going to share with you a very easy and simple recipe of <strong><em>Kaju Kat",
            ],
        }

        model = AutoModelForCausalLM.from_pretrained(model_id, low_cpu_mem_usage=True, torch_dtype=torch.bfloat16).to(
            torch_device
        )

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        inputs = tokenizer(self.input_text, return_tensors="pt", padding=True).to(torch_device)

        output = model.generate(**inputs, max_new_tokens=20, do_sample=False)
        output_text = tokenizer.batch_decode(output, skip_special_tokens=True)

        self.assertEqual(output_text, EXPECTED_TEXTS[self.cuda_compute_capability_major_version])

    @require_read_token
    def test_model_2b_eager(self):
        model_id = "google/gemma-2b"

        # Key 9 for MI300, Key 8 for A100/A10, and Key 7 for T4.
        #
        # Note: Key 9 is currently set for MI300, but may need potential future adjustments for H100s,
        # considering differences in hardware processing and potential deviations in generated text.
        EXPECTED_TEXTS = {
            7: [
                "Hello I am doing a project on the 1990s and I am looking for some information on the ",
                "Hi today I am going to share with you a very easy and simple recipe of <strong><em>Kaju Kat",
            ],
            8: [
                "Hello I am doing a project on the 1990s and I need to know what the most popular music",
                "Hi today I am going to share with you a very easy and simple recipe of <strong><em>Kaju Kat",
            ],
            9: [
                "Hello I am doing a project on the 1990s and I need to know what the most popular music",
                "Hi today I am going to share with you a very easy and simple recipe of <strong><em>Kaju Kat",
            ],
        }

        model = AutoModelForCausalLM.from_pretrained(
            model_id, low_cpu_mem_usage=True, torch_dtype=torch.bfloat16, attn_implementation="eager"
        )
        model.to(torch_device)

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        inputs = tokenizer(self.input_text, return_tensors="pt", padding=True).to(torch_device)

        output = model.generate(**inputs, max_new_tokens=20, do_sample=False)
        output_text = tokenizer.batch_decode(output, skip_special_tokens=True)

        self.assertEqual(output_text, EXPECTED_TEXTS[self.cuda_compute_capability_major_version])

    @require_torch_sdpa
    @require_read_token
    def test_model_2b_sdpa(self):
        model_id = "google/gemma-2b"

        # Key 9 for MI300, Key 8 for A100/A10, and Key 7 for T4.
        #
        # Note: Key 9 is currently set for MI300, but may need potential future adjustments for H100s,
        # considering differences in hardware processing and potential deviations in generated text.
        EXPECTED_TEXTS = {
            7: [
                "Hello I am doing a project on the 1990s and I need to know what the most popular music",
                "Hi today I am going to share with you a very easy and simple recipe of <strong><em>Khichdi",
            ],
            8: [
                "Hello I am doing a project on the 1990s and I need to know what the most popular music",
                "Hi today I am going to share with you a very easy and simple recipe of <strong><em>Kaju Kat",
            ],
            9: [
                "Hello I am doing a project on the 1990s and I need to know what the most popular music",
                "Hi today I am going to share with you a very easy and simple recipe of <strong><em>Kaju Kat",
            ],
        }

        model = AutoModelForCausalLM.from_pretrained(
            model_id, low_cpu_mem_usage=True, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
        )
        model.to(torch_device)

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        inputs = tokenizer(self.input_text, return_tensors="pt", padding=True).to(torch_device)

        output = model.generate(**inputs, max_new_tokens=20, do_sample=False)
        output_text = tokenizer.batch_decode(output, skip_special_tokens=True)

        self.assertEqual(output_text, EXPECTED_TEXTS[self.cuda_compute_capability_major_version])

    @pytest.mark.flash_attn_test
    @require_flash_attn
    @require_read_token
    def test_model_2b_flash_attn(self):
        model_id = "google/gemma-2b"
        EXPECTED_TEXTS = [
            "Hello I am doing a project on the 1990s and I need to know what the most popular music",
            "Hi today I am going to share with you a very easy and simple recipe of <strong><em>Kaju Kat",
        ]

        model = AutoModelForCausalLM.from_pretrained(
            model_id, low_cpu_mem_usage=True, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2"
        )
        model.to(torch_device)

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        inputs = tokenizer(self.input_text, return_tensors="pt", padding=True).to(torch_device)

        output = model.generate(**inputs, max_new_tokens=20, do_sample=False)
        output_text = tokenizer.batch_decode(output, skip_special_tokens=True)

        self.assertEqual(output_text, EXPECTED_TEXTS)

    @require_bitsandbytes
    @require_read_token
    def test_model_2b_4bit(self):
        model_id = "google/gemma-2b"
        EXPECTED_TEXTS = [
            "Hello I am doing a project and I need to make a 3d model of a house. I have been using",
            "Hi today I'd like to share with you my experience with the new wattpad wattpad wattpad wattpad wattpad wattpad wattpad",
        ]

        model = AutoModelForCausalLM.from_pretrained(model_id, low_cpu_mem_usage=True, load_in_4bit=True)

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        inputs = tokenizer(self.input_text, return_tensors="pt", padding=True).to(torch_device)

        output = model.generate(**inputs, max_new_tokens=20, do_sample=False)
        output_text = tokenizer.batch_decode(output, skip_special_tokens=True)

        self.assertEqual(output_text, EXPECTED_TEXTS)

    @unittest.skip("The test will not fit our CI runners")
    @require_read_token
    def test_model_7b_fp32(self):
        model_id = "google/gemma-7b"
        EXPECTED_TEXTS = [
            "Hello my name is ***** ***** I will be assisting you today. I am sorry to hear about your issue. I will",
            "Hi,\n\nI have a problem with my 2005 1.6 16",
        ]

        model = AutoModelForCausalLM.from_pretrained(model_id, low_cpu_mem_usage=True).to(torch_device)

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        inputs = tokenizer(self.input_text, return_tensors="pt", padding=True).to(torch_device)

        output = model.generate(**inputs, max_new_tokens=20, do_sample=False)
        output_text = tokenizer.batch_decode(output, skip_special_tokens=True)

        self.assertEqual(output_text, EXPECTED_TEXTS)

    @require_read_token
    def test_model_7b_fp16(self):
        model_id = "google/gemma-7b"
        EXPECTED_TEXTS = [
            """Hello I am doing a project on a 1999 4.0L 4x4. I""",
            "Hi today I am going to show you how to make a simple and easy to make a DIY 3D",
        ]

        model = AutoModelForCausalLM.from_pretrained(model_id, low_cpu_mem_usage=True, torch_dtype=torch.float16).to(
            torch_device
        )

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        inputs = tokenizer(self.input_text, return_tensors="pt", padding=True).to(torch_device)

        output = model.generate(**inputs, max_new_tokens=20, do_sample=False)
        output_text = tokenizer.batch_decode(output, skip_special_tokens=True)

        self.assertEqual(output_text, EXPECTED_TEXTS)

    @require_read_token
    def test_model_7b_bf16(self):
        model_id = "google/gemma-7b"

        # Key 9 for MI300, Key 8 for A100/A10, and Key 7 for T4.
        #
        # Note: Key 9 is currently set for MI300, but may need potential future adjustments for H100s,
        # considering differences in hardware processing and potential deviations in generated text.
        EXPECTED_TEXTS = {
            7: [
                """Hello I am doing a project on a 1991 240sx and I am trying to find""",
                "Hi today I am going to show you how to make a very simple and easy to make a very simple and",
            ],
            8: [
                "Hello I am doing a project for my school and I am trying to make a program that will read a .txt file",
                "Hi today I am going to show you how to make a very simple and easy to make a very simple and",
            ],
            9: [
                "Hello I am doing a project for my school and I am trying to get a servo to move a certain amount of degrees",
                "Hi today I am going to show you how to make a very simple and easy to make DIY light up sign",
            ],
        }

        model = AutoModelForCausalLM.from_pretrained(model_id, low_cpu_mem_usage=True, torch_dtype=torch.bfloat16).to(
            torch_device
        )

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        inputs = tokenizer(self.input_text, return_tensors="pt", padding=True).to(torch_device)

        output = model.generate(**inputs, max_new_tokens=20, do_sample=False)
        output_text = tokenizer.batch_decode(output, skip_special_tokens=True)

        self.assertEqual(output_text, EXPECTED_TEXTS[self.cuda_compute_capability_major_version])

    @require_read_token
    def test_model_7b_fp16_static_cache(self):
        model_id = "google/gemma-7b"
        EXPECTED_TEXTS = [
            """Hello I am doing a project on a 1999 4.0L 4x4. I""",
            "Hi today I am going to show you how to make a simple and easy to make a DIY 3D",
        ]

        model = AutoModelForCausalLM.from_pretrained(model_id, low_cpu_mem_usage=True, torch_dtype=torch.float16).to(
            torch_device
        )

        model.generation_config.cache_implementation = "static"

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        inputs = tokenizer(self.input_text, return_tensors="pt", padding=True).to(torch_device)

        output = model.generate(**inputs, max_new_tokens=20, do_sample=False)
        output_text = tokenizer.batch_decode(output, skip_special_tokens=True)

        self.assertEqual(output_text, EXPECTED_TEXTS)

    @require_bitsandbytes
    @require_read_token
    def test_model_7b_4bit(self):
        model_id = "google/gemma-7b"
        EXPECTED_TEXTS = {
            7: [
                "Hello I am doing a project for my school and I am trying to make a program that will take a number and then",
                """Hi today I am going to talk about the new update for the game called "The new update" and I""",
            ],
            8: [
                "Hello I am doing a project for my school and I am trying to make a program that will take a number and then",
                "Hi today I am going to talk about the best way to get rid of acne. miniaturing is a very",
            ],
        }

        model = AutoModelForCausalLM.from_pretrained(model_id, low_cpu_mem_usage=True, load_in_4bit=True)

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        inputs = tokenizer(self.input_text, return_tensors="pt", padding=True).to(torch_device)

        output = model.generate(**inputs, max_new_tokens=20, do_sample=False)
        output_text = tokenizer.batch_decode(output, skip_special_tokens=True)

        self.assertEqual(output_text, EXPECTED_TEXTS[self.cuda_compute_capability_major_version])

    @slow
    @require_torch_gpu
    @require_read_token
    def test_compile_static_cache(self):
        # `torch==2.2` will throw an error on this test (as in other compilation tests), but torch==2.1.2 and torch>2.2
        # work as intended. See https://github.com/pytorch/pytorch/issues/121943
        if version.parse(torch.__version__) < version.parse("2.3.0"):
            self.skipTest("This test requires torch >= 2.3 to run.")

        NUM_TOKENS_TO_GENERATE = 40
        # Note on `EXPECTED_TEXT_COMPLETION`'s diff: the current value matches the original test if the original test
        # was changed to have a cache of 53 tokens (as opposed to 4096), on Ampere GPUs.
        #
        # Key 9 for MI300, Key 8 for A100/A10, and Key 7 for T4.
        #
        # Note: Key 9 is currently set for MI300, but may need potential future adjustments for H100s,
        # considering differences in hardware processing and potential deviations in generated text.
        EXPECTED_TEXT_COMPLETION = {
            8: [
                "Hello I am doing a project on the 1990s and I need to know what the most popular music was in the 1990s. I have looked on the internet and I have found",
                "Hi today\nI have a problem with my 2007 1.9 tdi 105bhp.\nI have a problem with the engine management light on.\nI have checked the",
            ],
            7: [
                "Hello I am doing a project on the 1990s and I need to know what the most popular music was in the 1990s. I have looked on the internet and I have found",
                "Hi today\nI have a problem with my 2007 1.9 tdi 105bhp.\nI have a problem with the engine management light on.\nI have checked the",
            ],
            9: [
                "Hello I am doing a project on the 1990s and I need to know what the most popular music was in the 1990s. I have looked on the internet and I have found",
                "Hi today\nI have a problem with my 2007 1.9 tdi 105bhp.\nI have a problem with the engine management light on.\nI have checked the",
            ],
        }

        prompts = ["Hello I am doing", "Hi today"]
        tokenizer = GemmaTokenizer.from_pretrained("google/gemma-2b", pad_token="</s>", padding_side="right")
        model = Gemma2ForCausalLM.from_pretrained(
            "google/gemma-2b", device_map="sequential", torch_dtype=torch.float16
        )
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)

        # Dynamic Cache
        generated_ids = model.generate(**inputs, max_new_tokens=NUM_TOKENS_TO_GENERATE, do_sample=False)
        dynamic_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        self.assertEqual(EXPECTED_TEXT_COMPLETION[8], dynamic_text)  # Both GPU architectures have the same output

        # Static Cache
        generated_ids = model.generate(
            **inputs, max_new_tokens=NUM_TOKENS_TO_GENERATE, do_sample=False, cache_implementation="static"
        )
        static_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        self.assertEqual(EXPECTED_TEXT_COMPLETION[self.cuda_compute_capability_major_version], static_text)

        # Static Cache + compile
        model.forward = torch.compile(model.forward, mode="reduce-overhead", fullgraph=True)
        generated_ids = model.generate(
            **inputs, max_new_tokens=NUM_TOKENS_TO_GENERATE, do_sample=False, cache_implementation="static"
        )
        static_compiled_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        self.assertEqual(EXPECTED_TEXT_COMPLETION[self.cuda_compute_capability_major_version], static_compiled_text)
