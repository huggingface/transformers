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
"""Testing suite for the PyTorch Gemma model."""

import unittest

import pytest
from packaging import version

from transformers import AutoModelForCausalLM, AutoTokenizer, GemmaConfig, is_torch_available
from transformers.generation.configuration_utils import GenerationConfig
from transformers.testing_utils import (
    DeviceProperties,
    Expectations,
    cleanup,
    get_device_properties,
    require_bitsandbytes,
    require_flash_attn,
    require_read_token,
    require_torch,
    require_torch_accelerator,
    require_torch_gpu,
    slow,
    torch_device,
)

from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester


if is_torch_available():
    import torch

    from transformers import (
        GemmaForCausalLM,
        GemmaForSequenceClassification,
        GemmaForTokenClassification,
        GemmaModel,
    )


@require_torch
class GemmaModelTester(CausalLMModelTester):
    config_class = GemmaConfig
    if is_torch_available():
        base_model_class = GemmaModel
        causal_lm_class = GemmaForCausalLM
        sequence_classification_class = GemmaForSequenceClassification
        token_classification_class = GemmaForTokenClassification


@require_torch
class GemmaModelTest(CausalLMModelTest, unittest.TestCase):
    all_model_classes = (
        (GemmaModel, GemmaForCausalLM, GemmaForSequenceClassification, GemmaForTokenClassification)
        if is_torch_available()
        else ()
    )
    pipeline_model_mapping = (
        {
            "feature-extraction": GemmaModel,
            "text-classification": GemmaForSequenceClassification,
            "token-classification": GemmaForTokenClassification,
            "text-generation": GemmaForCausalLM,
            "zero-shot": GemmaForSequenceClassification,
        }
        if is_torch_available()
        else {}
    )
    model_tester_class = GemmaModelTester

    # used in `test_torch_compile_for_training`
    _torch_compile_train_cls = GemmaForCausalLM if is_torch_available() else None

    # TODO (ydshieh): Check this. See https://app.circleci.com/pipelines/github/huggingface/transformers/79245/workflows/9490ef58-79c2-410d-8f51-e3495156cf9c/jobs/1012146
    def is_pipeline_test_to_skip(
        self,
        pipeline_test_case_name,
        config_class,
        model_architecture,
        tokenizer_name,
        image_processor_name,
        feature_extractor_name,
        processor_name,
    ):
        return True

    @require_flash_attn
    @require_torch_gpu
    @pytest.mark.flash_attn_test
    @slow
    def test_flash_attn_2_inference_equivalence_right_padding(self):
        self.skipTest(reason="Gemma flash attention does not support right padding")


@slow
@require_torch_accelerator
class GemmaIntegrationTest(unittest.TestCase):
    input_text = ["Hello I am doing", "Hi today"]
    # This variable is used to determine which accelerator are we using for our runners (e.g. A10 or T4)
    # Depending on the hardware we get different logits / generations
    device_properties: DeviceProperties = (None, None, None)

    @classmethod
    def setUpClass(cls):
        cls.device_properties = get_device_properties()

    def setUp(self):
        cleanup(torch_device, gc_collect=True)

    def tearDown(self):
        # See LlamaIntegrationTest.tearDown(). Can be removed once LlamaIntegrationTest.tearDown() is removed.
        cleanup(torch_device, gc_collect=True)

    @require_read_token
    def test_model_2b_fp16(self):
        model_id = "google/gemma-2b"
        EXPECTED_TEXTS = [
            "Hello I am doing a project on the 1990s and I need to know what the most popular music",
            "Hi today I am going to share with you a very easy and simple recipe of <strong><em>Kaju Kat",
        ]

        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float16).to(torch_device)

        model.generation_config.cache_implementation = "static"

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        inputs = tokenizer(self.input_text, return_tensors="pt", padding=True).to(torch_device)

        output = model.generate(**inputs, max_new_tokens=20, do_sample=False)
        output_text = tokenizer.batch_decode(output, skip_special_tokens=True)

        self.assertEqual(output_text, EXPECTED_TEXTS)

    @require_read_token
    def test_model_2b_bf16(self):
        model_id = "google/gemma-2b"

        EXPECTED_TEXTS = [
            "Hello I am doing a project on the 1990s and I need to know what the most popular music",
            "Hi today I am going to share with you a very easy and simple recipe of <strong><em>Kaju Kat",
        ]

        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16).to(torch_device)

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        inputs = tokenizer(self.input_text, return_tensors="pt", padding=True).to(torch_device)

        output = model.generate(**inputs, max_new_tokens=20, do_sample=False)
        output_text = tokenizer.batch_decode(output, skip_special_tokens=True)

        self.assertEqual(output_text, EXPECTED_TEXTS)

    @require_read_token
    def test_model_2b_eager(self):
        model_id = "google/gemma-2b"

        EXPECTED_TEXTS = [
            "Hello I am doing a project on the 1990s and I need to know what the most popular music",
            "Hi today I am going to share with you a very easy and simple recipe of <strong><em>Kaju Kat",
        ]

        # bfloat16 gives strange values, likely due to it has lower precision + very short prompts
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float16, attn_implementation="eager")
        model.to(torch_device)

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        inputs = tokenizer(self.input_text, return_tensors="pt", padding=True).to(torch_device)

        output = model.generate(**inputs, max_new_tokens=20, do_sample=False)
        output_text = tokenizer.batch_decode(output, skip_special_tokens=True)

        self.assertEqual(output_text, EXPECTED_TEXTS)

    @require_flash_attn
    @require_read_token
    @pytest.mark.flash_attn_test
    def test_model_2b_flash_attn(self):
        model_id = "google/gemma-2b"
        EXPECTED_TEXTS = [
            "Hello I am doing a project on the 1990s and I need to know what the most popular music",
            "Hi today I am going to share with you a very easy and simple recipe of <strong><em>Kaju Kat",
        ]

        model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=torch.bfloat16, attn_implementation="flash_attention_2"
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

        model = AutoModelForCausalLM.from_pretrained(model_id, load_in_4bit=True)

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        inputs = tokenizer(self.input_text, return_tensors="pt", padding=True).to(torch_device)

        output = model.generate(**inputs, max_new_tokens=20, do_sample=False)
        output_text = tokenizer.batch_decode(output, skip_special_tokens=True)

        self.assertEqual(output_text, EXPECTED_TEXTS)

    @unittest.skip(reason="The test will not fit our CI runners")
    @require_read_token
    def test_model_7b_fp32(self):
        model_id = "google/gemma-7b"
        EXPECTED_TEXTS = [
            "Hello my name is ***** ***** I will be assisting you today. I am sorry to hear about your issue. I will",
            "Hi,\n\nI have a problem with my 2005 1.6 16",
        ]

        model = AutoModelForCausalLM.from_pretrained(model_id).to(torch_device)

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        inputs = tokenizer(self.input_text, return_tensors="pt", padding=True).to(torch_device)

        output = model.generate(**inputs, max_new_tokens=20, do_sample=False)
        output_text = tokenizer.batch_decode(output, skip_special_tokens=True)

        self.assertEqual(output_text, EXPECTED_TEXTS)

    @require_read_token
    def test_model_7b_fp16(self):
        if self.device_properties[0] == "cuda" and self.device_properties[1] == 7:
            self.skipTest("This test is failing (`torch.compile` fails) on Nvidia T4 GPU (OOM).")

        model_id = "google/gemma-7b"
        EXPECTED_TEXTS = [
            """Hello I am doing a project on a 1999 4.0L 4x4. I""",
            "Hi today I am going to show you how to make a simple and easy to make a DIY 3D",
        ]

        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float16).to(torch_device)

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        inputs = tokenizer(self.input_text, return_tensors="pt", padding=True).to(torch_device)

        output = model.generate(**inputs, max_new_tokens=20, do_sample=False)
        output_text = tokenizer.batch_decode(output, skip_special_tokens=True)

        self.assertEqual(output_text, EXPECTED_TEXTS)

    @require_read_token
    def test_model_7b_bf16(self):
        if self.device_properties[0] == "cuda" and self.device_properties[1] == 7:
            self.skipTest("This test is failing (`torch.compile` fails) on Nvidia T4 GPU (OOM).")

        model_id = "google/gemma-7b"

        # Key 9 for MI300, Key 8 for A100/A10, and Key 7 for T4.
        #
        # Note: Key 9 is currently set for MI300, but may need potential future adjustments for H100s,
        # considering differences in hardware processing and potential deviations in generated text.
        # fmt: off
        EXPECTED_TEXTS = Expectations(
            {
                ("cuda", 7): ["""Hello I am doing a project on a 1991 240sx and I am trying to find""", "Hi today I am going to show you how to make a very simple and easy to make a very simple and",],
                ("cuda", 8): ['Hello I am doing a project for my school and I am trying to make a game in which you have to get a', 'Hi today I am going to show you how to make a very simple and easy to make a very simple and'],
                ("rocm", 9): ["Hello I am doing a project for my school and I am trying to get a servo to move a certain amount of degrees", "Hi today I am going to show you how to make a very simple and easy to make DIY light up sign",],
            }
        )
        # fmt: on
        expected_text = EXPECTED_TEXTS.get_expectation()

        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16).to(torch_device)

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        inputs = tokenizer(self.input_text, return_tensors="pt", padding=True).to(torch_device)

        output = model.generate(**inputs, max_new_tokens=20, do_sample=False)
        output_text = tokenizer.batch_decode(output, skip_special_tokens=True)
        self.assertEqual(output_text, expected_text)

    @require_read_token
    def test_model_7b_fp16_static_cache(self):
        if self.device_properties[0] == "cuda" and self.device_properties[1] == 7:
            self.skipTest("This test is failing (`torch.compile` fails) on Nvidia T4 GPU (OOM).")

        model_id = "google/gemma-7b"

        expectations = Expectations(
            {
                (None, None): [
                    "Hello I am doing a project on a 1999 4.0L 4x4. I",
                    "Hi today I am going to show you how to make a simple and easy to make a DIY 3D",
                ],
                ("cuda", 8): [
                    "Hello I am doing a project on a 1995 3000gt SL. I have a",
                    "Hi today I am going to show you how to make a simple and easy to make a DIY 3D",
                ],
            }
        )
        EXPECTED_TEXTS = expectations.get_expectation()

        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float16).to(torch_device)

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

        expectations = Expectations(
            {
                (None, None): [
                    "Hello I am doing a project for my school and I am trying to make a program that will take a number and then",
                    "Hi today I am going to talk about the best way to get rid of acne. miniaturing is a very",
                ],
                ("cuda", 8): [
                    "Hello I am doing a project for my school and I am trying to make a program that will take a number and then",
                    'Hi today I am going to talk about the new update for the game called "The new update!:)!:)!:)',
                ],
            }
        )
        EXPECTED_TEXTS = expectations.get_expectation()

        model = AutoModelForCausalLM.from_pretrained(model_id, load_in_4bit=True)

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        inputs = tokenizer(self.input_text, return_tensors="pt", padding=True).to(torch_device)

        output = model.generate(**inputs, max_new_tokens=20, do_sample=False)
        output_text = tokenizer.batch_decode(output, skip_special_tokens=True)
        self.assertEqual(output_text, EXPECTED_TEXTS)

    @slow
    @require_torch_accelerator
    @require_read_token
    def test_compile_static_cache(self):
        # `torch==2.2` will throw an error on this test (as in other compilation tests), but torch==2.1.2 and torch>2.2
        # work as intended. See https://github.com/pytorch/pytorch/issues/121943
        if version.parse(torch.__version__) < version.parse("2.3.0"):
            self.skipTest(reason="This test requires torch >= 2.3 to run.")

        NUM_TOKENS_TO_GENERATE = 40
        EXPECTED_TEXT_COMPLETION = [
            "Hello I am doing a project on the 1990s and I need to know what the most popular music was in the 1990s. I have looked on the internet and I have found",
            "Hi today\nI have a problem with my 2007 1.9 tdi 105bhp.\nI have a problem with the engine management light on.\nI have checked the",
        ]

        prompts = ["Hello I am doing", "Hi today"]
        tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b", pad_token="</s>", padding_side="right")
        model = GemmaForCausalLM.from_pretrained("google/gemma-2b", device_map=torch_device, dtype=torch.float16)
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)

        # Dynamic Cache
        generated_ids = model.generate(**inputs, max_new_tokens=NUM_TOKENS_TO_GENERATE, do_sample=False)
        dynamic_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        self.assertEqual(EXPECTED_TEXT_COMPLETION, dynamic_text)  # Both GPU architectures have the same output

        # Static Cache
        generated_ids = model.generate(
            **inputs, max_new_tokens=NUM_TOKENS_TO_GENERATE, do_sample=False, cache_implementation="static"
        )
        static_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        self.assertEqual(EXPECTED_TEXT_COMPLETION, static_text)

        # Static Cache + compile
        model.forward = torch.compile(model.forward, mode="reduce-overhead", fullgraph=True)
        generated_ids = model.generate(
            **inputs, max_new_tokens=NUM_TOKENS_TO_GENERATE, do_sample=False, cache_implementation="static"
        )
        static_compiled_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        self.assertEqual(EXPECTED_TEXT_COMPLETION, static_compiled_text)

    @slow
    @require_read_token
    def test_export_static_cache(self):
        if version.parse(torch.__version__) < version.parse("2.3.0"):
            self.skipTest(reason="This test requires torch >= 2.3 to run.")

        from transformers.integrations.executorch import (
            TorchExportableModuleWithStaticCache,
        )

        tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b", pad_token="</s>", padding_side="right")

        expectations = Expectations(
            {
                (None, None): [
                    "Hello I am doing a project on the 1990s and I need to know what the most popular music was in the 1990s. I have looked on the internet and I have found"
                ],
                ("cuda", 8): [
                    "Hello I am doing a project on the 1990s and I need to know what the most popular music was in the 1990s. I have been looking on the internet and I have"
                ],
            }
        )
        EXPECTED_TEXT_COMPLETION = expectations.get_expectation()

        max_generation_length = tokenizer(EXPECTED_TEXT_COMPLETION, return_tensors="pt", padding=True)[
            "input_ids"
        ].shape[-1]

        # Load model
        device = "cpu"  # TODO (joao / export experts): should be on `torch_device`, but causes GPU OOM
        dtype = torch.bfloat16
        cache_implementation = "static"
        attn_implementation = "sdpa"
        batch_size = 1
        model = GemmaForCausalLM.from_pretrained(
            "google/gemma-2b",
            device_map=device,
            dtype=dtype,
            attn_implementation=attn_implementation,
            generation_config=GenerationConfig(
                use_cache=True,
                cache_implementation=cache_implementation,
                max_length=max_generation_length,
                cache_config={
                    "batch_size": batch_size,
                    "max_cache_len": max_generation_length,
                },
            ),
        )

        prompts = ["Hello I am doing"]
        prompt_tokens = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
        prompt_token_ids = prompt_tokens["input_ids"]
        max_new_tokens = max_generation_length - prompt_token_ids.shape[-1]

        # Static Cache + eager
        eager_generated_ids = model.generate(
            **prompt_tokens, max_new_tokens=max_new_tokens, do_sample=False, cache_implementation=cache_implementation
        )
        eager_generated_text = tokenizer.batch_decode(eager_generated_ids, skip_special_tokens=True)
        self.assertEqual(EXPECTED_TEXT_COMPLETION, eager_generated_text)

        # Static Cache + export
        from transformers.integrations.executorch import TorchExportableModuleForDecoderOnlyLM

        exportable_module = TorchExportableModuleForDecoderOnlyLM(model)
        exported_program = exportable_module.export(
            input_ids=prompt_token_ids,
            cache_position=torch.arange(prompt_token_ids.shape[-1], dtype=torch.long, device=model.device),
        )
        ep_generated_ids = TorchExportableModuleWithStaticCache.generate(
            exported_program=exported_program, prompt_token_ids=prompt_token_ids, max_new_tokens=max_new_tokens
        )
        ep_generated_text = tokenizer.batch_decode(ep_generated_ids, skip_special_tokens=True)

        # After switching to A10 on 2025/06/29, we get slightly different outputs when using export
        expectations = Expectations(
            {
                (None, None): [
                    "Hello I am doing a project on the 1990s and I need to know what the most popular music was in the 1990s. I have looked on the internet and I have found"
                ],
                ("cuda", 8): [
                    "Hello I am doing a project on the 1990s and I need to know what the most popular music was in the 1990s. I have looked on the internet and I have found"
                ],
            }
        )
        EXPECTED_TEXT_COMPLETION = expectations.get_expectation()

        self.assertEqual(EXPECTED_TEXT_COMPLETION, ep_generated_text)

    def test_model_2b_bf16_dola(self):
        model_id = "google/gemma-2b"
        # ground truth text generated with dola_layers="low", repetition_penalty=1.2
        expectations = Expectations(
            {
                (None, None): [
                    "Hello I am doing an experiment and need to get the mass of a block. The problem is, it has no scale",
                    "Hi today we have the review for a <strong>2016/2017</strong> season of",
                ],
                ("cuda", 8): [
                    "Hello I am doing an experiment and need to get the mass of a block. The only tool I have is a scale",
                    "Hi today we have the review for a <strong>2016/2017</strong> season of",
                ],
            }
        )
        EXPECTED_TEXTS = expectations.get_expectation()

        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16).to(torch_device)

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        inputs = tokenizer(self.input_text, return_tensors="pt", padding=True).to(torch_device)

        output = model.generate(
            **inputs, max_new_tokens=20, do_sample=False, dola_layers="low", repetition_penalty=1.2
        )
        output_text = tokenizer.batch_decode(output, skip_special_tokens=True)
        self.assertEqual(output_text, EXPECTED_TEXTS)
