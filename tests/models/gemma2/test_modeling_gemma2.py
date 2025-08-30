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
from parameterized import parameterized
from pytest import mark

from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache, Gemma2Config, is_torch_available, pipeline
from transformers.cache_utils import DynamicLayer, DynamicSlidingWindowLayer
from transformers.generation.configuration_utils import GenerationConfig
from transformers.testing_utils import (
    Expectations,
    cleanup,
    is_flash_attn_2_available,
    require_flash_attn,
    require_large_cpu_ram,
    require_read_token,
    require_torch,
    require_torch_accelerator,
    require_torch_large_accelerator,
    require_torch_large_gpu,
    slow,
    torch_device,
)

from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester
from ...test_configuration_common import ConfigTester


if is_torch_available():
    import torch

    from transformers import (
        Gemma2ForCausalLM,
        Gemma2ForSequenceClassification,
        Gemma2ForTokenClassification,
        Gemma2Model,
    )


class Gemma2ModelTester(CausalLMModelTester):
    if is_torch_available():
        config_class = Gemma2Config
        base_model_class = Gemma2Model
        causal_lm_class = Gemma2ForCausalLM
        sequence_class = Gemma2ForSequenceClassification
        token_class = Gemma2ForTokenClassification
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


@require_torch
class Gemma2ModelTest(CausalLMModelTest, unittest.TestCase):
    all_model_classes = (
        (Gemma2Model, Gemma2ForCausalLM, Gemma2ForSequenceClassification, Gemma2ForTokenClassification)
        if is_torch_available()
        else ()
    )
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
    model_tester_class = Gemma2ModelTester

    def setUp(self):
        self.model_tester = Gemma2ModelTester(self)
        self.config_tester = ConfigTester(self, config_class=Gemma2Config, hidden_size=37)


@slow
@require_torch_accelerator
class Gemma2IntegrationTest(unittest.TestCase):
    input_text = ["Hello I am doing", "Hi today"]

    def setUp(self):
        cleanup(torch_device, gc_collect=True)

    def tearDown(self):
        cleanup(torch_device, gc_collect=True)

    @require_torch_large_accelerator
    @require_read_token
    def test_model_9b_bf16(self):
        model_id = "google/gemma-2-9b"
        EXPECTED_TEXTS = [
            "<bos>Hello I am doing a project on the 1918 flu pandemic and I am trying to find out how many",
            "<pad><pad><bos>Hi today I'm going to be talking about the history of the United States. The United States of America",
        ]

        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, attn_implementation="eager").to(
            torch_device
        )

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        inputs = tokenizer(self.input_text, return_tensors="pt", padding=True).to(torch_device)

        output = model.generate(**inputs, max_new_tokens=20, do_sample=False)
        output_text = tokenizer.batch_decode(output, skip_special_tokens=False)

        self.assertEqual(output_text, EXPECTED_TEXTS)

    @require_torch_large_accelerator
    @require_read_token
    def test_model_9b_fp16(self):
        model_id = "google/gemma-2-9b"
        EXPECTED_TEXTS = [
            "<bos>Hello I am doing a project on the 1918 flu pandemic and I am trying to find out how many",
            "<pad><pad><bos>Hi today I'm going to be talking about the history of the United States. The United States of America",
        ]

        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float16, attn_implementation="eager").to(
            torch_device
        )

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        inputs = tokenizer(self.input_text, return_tensors="pt", padding=True).to(torch_device)

        output = model.generate(**inputs, max_new_tokens=20, do_sample=False)
        output_text = tokenizer.batch_decode(output, skip_special_tokens=False)

        self.assertEqual(output_text, EXPECTED_TEXTS)

    @require_read_token
    @require_torch_large_accelerator
    def test_model_9b_pipeline_bf16(self):
        # See https://github.com/huggingface/transformers/pull/31747 -- pipeline was broken for Gemma2 before this PR
        model_id = "google/gemma-2-9b"
        # EXPECTED_TEXTS should match the same non-pipeline test, minus the special tokens
        EXPECTED_TEXTS = [
            "Hello I am doing a project on the 1918 flu pandemic and I am trying to find out how many",
            "Hi today I'm going to be talking about the history of the United States. The United States of America",
        ]

        model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=torch.bfloat16, attn_implementation="flex_attention"
        ).to(torch_device)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

        output = pipe(self.input_text, max_new_tokens=20, do_sample=False, padding=True)

        self.assertEqual(output[0][0]["generated_text"], EXPECTED_TEXTS[0])
        self.assertEqual(output[1][0]["generated_text"], EXPECTED_TEXTS[1])

    @require_read_token
    def test_model_2b_pipeline_bf16_flex_attention(self):
        # See https://github.com/huggingface/transformers/pull/31747 -- pipeline was broken for Gemma2 before this PR
        model_id = "google/gemma-2-2b"
        # EXPECTED_TEXTS should match the same non-pipeline test, minus the special tokens
        EXPECTED_BATCH_TEXTS = Expectations(
            {
                ("xpu", 3): [
                    "Hello I am doing a project on the 1960s and I am trying to find out what the average",
                    "Hi today I'm going to be talking about the 10 most powerful characters in the Naruto series.",
                ],
                ("cuda", 8): [
                    "Hello I am doing a project on the 1960s and I am trying to find out what the average",
                    "Hi today I'm going to be talking about the 10 most powerful characters in the Naruto series.",
                ],
            }
        )
        EXPECTED_BATCH_TEXT = EXPECTED_BATCH_TEXTS.get_expectation()

        model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=torch.bfloat16, attn_implementation="flex_attention"
        ).to(torch_device)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

        output = pipe(self.input_text, max_new_tokens=20, do_sample=False, padding=True)

        self.assertEqual(output[0][0]["generated_text"], EXPECTED_BATCH_TEXT[0])
        self.assertEqual(output[1][0]["generated_text"], EXPECTED_BATCH_TEXT[1])

    @require_read_token
    @require_flash_attn
    @require_torch_large_gpu
    @mark.flash_attn_test
    @slow
    def test_model_9b_flash_attn(self):
        # See https://github.com/huggingface/transformers/issues/31953 --- flash attn was generating garbage for gemma2, especially in long context
        model_id = "google/gemma-2-9b"
        EXPECTED_TEXTS = [
            '<bos>Hello I am doing a project on the 1918 flu pandemic and I am trying to find out how many people died in the United States. I have found a few sites that say 500,000 but I am not sure if that is correct. I have also found a site that says 675,000 but I am not sure if that is correct either. I am trying to find out how many people died in the United States. I have found a few',
            "<pad><pad><bos>Hi today I'm going to be talking about the history of the United States. The United States of America is a country in North America. It is the third largest country in the world by total area and the third most populous country with over 320 million people. The United States is a federal republic composed of 50 states and a federal district. The 48 contiguous states and the district of Columbia are in central North America between Canada and Mexico. The state of Alaska is in the",
        ]  # fmt: skip

        model = AutoModelForCausalLM.from_pretrained(
            model_id, attn_implementation="flash_attention_2", dtype="float16"
        ).to(torch_device)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        inputs = tokenizer(self.input_text, return_tensors="pt", padding=True).to(torch_device)

        output = model.generate(**inputs, max_new_tokens=100, do_sample=False)
        output_text = tokenizer.batch_decode(output, skip_special_tokens=False)

        self.assertEqual(output_text, EXPECTED_TEXTS)

    @pytest.mark.torch_export_test
    @slow
    @require_read_token
    def test_export_static_cache(self):
        if version.parse(torch.__version__) < version.parse("2.5.0"):
            self.skipTest(reason="This test requires torch >= 2.5 to run.")

        from transformers.integrations.executorch import (
            TorchExportableModuleWithStaticCache,
        )

        tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b", pad_token="</s>", padding_side="right")
        EXPECTED_TEXT_COMPLETIONS = Expectations(
            {
                ("xpu", 3): [
                    "Hello I am doing a project for my school and I need to know how to make a program that will take a number"
                ],
                ("cuda", 7): [
                    "Hello I am doing a project for my school and I need to know how to make a program that will take a number"
                ],
                ("cuda", 8): [
                    "Hello I am doing a project for my class and I am having trouble with the code. I am trying to make a"
                ],
                ("rocm", (9, 5)): [
                    "Hello I am doing a project for my school and I need to know how to make a program that will take a number"
                ],
            }
        )
        EXPECTED_TEXT_COMPLETION = EXPECTED_TEXT_COMPLETIONS.get_expectation()
        max_generation_length = tokenizer(EXPECTED_TEXT_COMPLETION, return_tensors="pt", padding=True)[
            "input_ids"
        ].shape[-1]

        # Load model
        device = "cpu"  # TODO (joao / export experts): should be on `torch_device`, but causes GPU OOM
        dtype = torch.bfloat16
        cache_implementation = "static"
        attn_implementation = "sdpa"
        batch_size = 1
        model = AutoModelForCausalLM.from_pretrained(
            "google/gemma-2-2b",
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

        # Static Cache + export
        from transformers.integrations.executorch import TorchExportableModuleForDecoderOnlyLM

        exportable_module = TorchExportableModuleForDecoderOnlyLM(model)
        exported_program = exportable_module.export(
            input_ids=torch.tensor([[1]], dtype=torch.long, device=model.device),
            cache_position=torch.tensor([0], dtype=torch.long, device=model.device),
        )
        ep_generated_ids = TorchExportableModuleWithStaticCache.generate(
            exported_program=exported_program, prompt_token_ids=prompt_token_ids, max_new_tokens=max_new_tokens
        )
        ep_generated_text = tokenizer.batch_decode(ep_generated_ids, skip_special_tokens=True)
        self.assertEqual(EXPECTED_TEXT_COMPLETION, ep_generated_text)

    @slow
    @require_read_token
    @require_large_cpu_ram
    @pytest.mark.torch_export_test
    def test_export_hybrid_cache(self):
        from transformers.integrations.executorch import TorchExportableModuleForDecoderOnlyLM
        from transformers.pytorch_utils import is_torch_greater_or_equal

        if not is_torch_greater_or_equal("2.6.0"):
            self.skipTest(reason="This test requires torch >= 2.6 to run.")

        model_id = "google/gemma-2-2b"
        model = AutoModelForCausalLM.from_pretrained(model_id)
        self.assertEqual(model.config.cache_implementation, "hybrid")

        # Export + hybrid cache
        model.eval()
        exportable_module = TorchExportableModuleForDecoderOnlyLM(model, batch_size=1, max_cache_len=1024)
        exported_program = exportable_module.export(
            input_ids=torch.tensor([[1]], dtype=torch.long, device=model.device),
            cache_position=torch.tensor([0], dtype=torch.long, device=model.device),
        )

        # Test generation with the exported model
        prompt = "What is the capital of France?"
        max_new_tokens_to_generate = 20
        # Generate text with the exported model
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        export_generated_text = TorchExportableModuleForDecoderOnlyLM.generate(
            exported_program, tokenizer, prompt, max_new_tokens=max_new_tokens_to_generate
        )

        input_text = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            eager_outputs = model.generate(
                **input_text,
                max_new_tokens=max_new_tokens_to_generate,
                do_sample=False,  # Use greedy decoding to match the exported model
            )

        eager_generated_text = tokenizer.decode(eager_outputs[0], skip_special_tokens=True)
        self.assertEqual(export_generated_text, eager_generated_text)

    @require_torch_large_accelerator
    @require_read_token
    def test_model_9b_bf16_flex_attention(self):
        model_id = "google/gemma-2-9b"
        EXPECTED_TEXTS = [
            "<bos>Hello I am doing a project on the 1918 flu pandemic and I am trying to find out how many",
            "<pad><pad><bos>Hi today I'm going to be talking about the history of the United States. The United States of America",
        ]

        model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=torch.bfloat16, attn_implementation="flex_attention"
        ).to(torch_device)
        assert model.config._attn_implementation == "flex_attention"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        inputs = tokenizer(self.input_text, return_tensors="pt", padding=True).to(torch_device)

        output = model.generate(**inputs, max_new_tokens=20, do_sample=False)
        output_text = tokenizer.batch_decode(output, skip_special_tokens=False)

        self.assertEqual(output_text, EXPECTED_TEXTS)

    @parameterized.expand([("flash_attention_2",), ("sdpa",), ("flex_attention",), ("eager",)])
    @require_read_token
    def test_generation_beyond_sliding_window(self, attn_implementation: str):
        """Test that we can correctly generate beyond the sliding window. This is non trivial as
        we need to correctly slice the attention mask in all cases (because we use a hybrid cache).
        Outputs for every attention functions should be coherent and identical.
        """
        # Impossible to test it with this model (even with < 100 tokens), probably due to the compilation of a large model.
        if attn_implementation == "flex_attention":
            self.skipTest(
                reason="`flex_attention` gives `torch._inductor.exc.InductorError: RuntimeError: No valid triton configs. OutOfMemoryError: out of resource: triton_tem_fused_0 Required: 147456 Hardware limit:101376 Reducing block sizes or `num_stages` may help.`"
            )

        if attn_implementation == "flash_attention_2" and not is_flash_attn_2_available():
            self.skipTest("FlashAttention2 is required for this test.")

        if torch_device == "xpu" and attn_implementation == "flash_attention_2":
            self.skipTest(reason="Intel XPU doesn't support falsh_attention_2 as of now.")

        model_id = "google/gemma-2-2b"
        EXPECTED_COMPLETIONS = [
            " the people, the food, the culture, the history, the music, the art, the architecture",
            ", green, yellow, orange, purple, pink, brown, black, white, gray, silver",
        ]

        input_text = [
            "This is a nice place. " * 800 + "I really enjoy the scenery,",  # This is larger than 4096 tokens
            "A list of colors: red, blue",  # This will almost all be padding tokens
        ]
        tokenizer = AutoTokenizer.from_pretrained(model_id, padding="left")
        inputs = tokenizer(input_text, padding=True, return_tensors="pt").to(torch_device)

        model = AutoModelForCausalLM.from_pretrained(
            model_id, attn_implementation=attn_implementation, dtype=torch.float16
        ).to(torch_device)

        # Make sure prefill is larger than sliding window
        input_size = inputs.input_ids.shape[-1]
        self.assertTrue(input_size > model.config.sliding_window)

        # It should by Hybrid by default from hub config, but let's make sure!
        out = model.generate(**inputs, max_new_tokens=20, cache_implementation="hybrid")[:, input_size:]
        output_text = tokenizer.batch_decode(out)

        self.assertEqual(output_text, EXPECTED_COMPLETIONS)

    @parameterized.expand([("flash_attention_2",), ("sdpa",), ("flex_attention",), ("eager",)])
    @require_read_token
    def test_generation_beyond_sliding_window_dynamic(self, attn_implementation: str):
        """
        Same as above, but explicitly setting the cache to Dynamic, as it's otherwise static by default for
        the model on the hub
        """
        # Impossible to test it with this model (even with < 100 tokens), probably due to the compilation of a large model.
        if attn_implementation == "flex_attention":
            self.skipTest(
                reason="`flex_attention` gives `torch._inductor.exc.InductorError: RuntimeError: No valid triton configs. OutOfMemoryError: out of resource: triton_tem_fused_0 Required: 147456 Hardware limit:101376 Reducing block sizes or `num_stages` may help.`"
            )

        if attn_implementation == "flash_attention_2" and not is_flash_attn_2_available():
            self.skipTest("FlashAttention2 is required for this test.")

        if torch_device == "xpu" and attn_implementation == "flash_attention_2":
            self.skipTest(reason="Intel XPU doesn't support falsh_attention_2 as of now.")

        model_id = "google/gemma-2-2b"
        EXPECTED_COMPLETIONS = [
            " the people, the food, the culture, the history, the music, the art, the architecture",
            ", green, yellow, orange, purple, pink, brown, black, white, gray, silver",
        ]

        input_text = [
            "This is a nice place. " * 800 + "I really enjoy the scenery,",  # This is larger than 4096 tokens
            "A list of colors: red, blue",  # This will almost all be padding tokens
        ]
        tokenizer = AutoTokenizer.from_pretrained(model_id, padding="left")
        inputs = tokenizer(input_text, padding=True, return_tensors="pt").to(torch_device)

        model = AutoModelForCausalLM.from_pretrained(
            model_id, attn_implementation=attn_implementation, dtype=torch.float16
        ).to(torch_device)

        # Make sure prefill is larger than sliding window
        input_size = inputs.input_ids.shape[-1]
        self.assertTrue(input_size > model.config.sliding_window)

        out = model.generate(**inputs, max_new_tokens=20, cache_implementation="dynamic", return_dict_in_generate=True)
        output_text = tokenizer.batch_decode(out.sequences[:, input_size:])

        self.assertEqual(output_text, EXPECTED_COMPLETIONS)

        # Let's check that the dynamic cache has hybrid layers!
        dynamic_cache = out.past_key_values
        self.assertTrue(isinstance(dynamic_cache, DynamicCache))
        for layer, layer_type in zip(dynamic_cache.layers, model.config.layer_types):
            if layer_type == "sliding_attention":
                self.assertTrue(isinstance(layer, DynamicSlidingWindowLayer))
                self.assertEqual(layer.keys.shape[-2], model.config.sliding_window - 1)
            else:
                self.assertTrue(isinstance(layer, DynamicLayer))
                # max_new_tokens - 1 because last token generated is not cached
                self.assertEqual(layer.keys.shape[-2], input_size + 20 - 1)
