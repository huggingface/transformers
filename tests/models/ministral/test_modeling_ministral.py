# coding=utf-8
# Copyright 2025 the HuggingFace Team. All rights reserved.
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
"""Testing suite for the PyTorch Ministral model."""

import gc
import unittest

import pytest
from packaging import version

from transformers import AutoModelForCausalLM, AutoTokenizer, MinistralConfig, is_torch_available
from transformers.generation.configuration_utils import GenerationConfig
from transformers.testing_utils import (
    Expectations,
    backend_empty_cache,
    require_bitsandbytes,
    require_flash_attn,
    require_torch,
    require_torch_gpu,
    slow,
    torch_device,
)


if is_torch_available():
    import torch

    from transformers import (
        MinistralForCausalLM,
        MinistralForQuestionAnswering,
        MinistralForSequenceClassification,
        MinistralForTokenClassification,
        MinistralModel,
    )


from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester


class MinistralModelTester(CausalLMModelTester):
    config_class = MinistralConfig
    if is_torch_available():
        base_model_class = MinistralModel
        causal_lm_class = MinistralForCausalLM
        sequence_class = MinistralForSequenceClassification
        token_class = MinistralForTokenClassification
        question_answering_class = MinistralForQuestionAnswering


@require_torch
class MinistralModelTest(CausalLMModelTest, unittest.TestCase):
    all_model_classes = (
        (
            MinistralModel,
            MinistralForCausalLM,
            MinistralForSequenceClassification,
            MinistralForTokenClassification,
            MinistralForQuestionAnswering,
        )
        if is_torch_available()
        else ()
    )
    test_headmasking = False
    test_pruning = False
    model_tester_class = MinistralModelTester
    pipeline_model_mapping = (
        {
            "feature-extraction": MinistralModel,
            "text-classification": MinistralForSequenceClassification,
            "token-classification": MinistralForTokenClassification,
            "text-generation": MinistralForCausalLM,
            "question-answering": MinistralForQuestionAnswering,
        }
        if is_torch_available()
        else {}
    )

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
        self.skipTest(reason="Ministral flash attention does not support right padding")


@require_torch
class MinistralIntegrationTest(unittest.TestCase):
    @slow
    def test_model_8b_logits(self):
        input_ids = [1, 306, 4658, 278, 6593, 310, 2834, 338]
        model = MinistralForCausalLM.from_pretrained("mistralai/Ministral-8B-Instruct-2410", device_map="auto")
        input_ids = torch.tensor([input_ids]).to(model.model.embed_tokens.weight.device)
        with torch.no_grad():
            out = model(input_ids).logits.float().cpu()
        # Expected mean on dim = -1
        EXPECTED_MEAN = torch.tensor([[-1.9537, -1.6193, -1.4123, -1.4673, -1.8511, -1.9309, -1.9826, -2.1776]])
        print(out.mean(-1))
        torch.testing.assert_close(out.mean(-1), EXPECTED_MEAN, rtol=1e-2, atol=1e-2)
        # slicing logits[0, 0, 0:30]
        EXPECTED_SLICE = torch.tensor([3.2025, 7.1265, 4.6058, 3.6423, 1.6357, 3.9265, 5.1883, 5.8760, 2.7942, 4.4823, 3.2571, 2.1063, 3.4275, 4.2028, 1.9767, 5.2115, 6.6756, 6.3999, 6.0483, 5.7378, 5.6660, 5.2298, 5.4103, 5.1248, 5.4376, 2.4570, 2.6107, 5.4039, 2.8077, 4.7777])  # fmt: skip
        print(out[0, 0, :30])
        print(EXPECTED_SLICE)
        torch.testing.assert_close(out[0, 0, :30], EXPECTED_SLICE, rtol=1e-4, atol=1e-4)

        del model
        backend_empty_cache(torch_device)
        gc.collect()

    @slow
    def test_model_8b_generation(self):
        EXPECTED_TEXT_COMPLETION = (
            """My favourite condiment is 100% natural, organic and vegan. I love to use it in my cooking and I"""
        )
        prompt = "My favourite condiment is "
        tokenizer = AutoTokenizer.from_pretrained("Mistralai/Ministral-8B-Instruct-2410", use_fast=False)
        model = MinistralForCausalLM.from_pretrained("Mistralai/Ministral-8B-Instruct-2410", device_map="auto")
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.model.embed_tokens.weight.device)

        # greedy generation outputs
        generated_ids = model.generate(input_ids, max_new_tokens=20, temperature=0)
        text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        print(text)
        self.assertEqual(EXPECTED_TEXT_COMPLETION, text)

        del model
        backend_empty_cache(torch_device)
        gc.collect()

    @require_bitsandbytes
    @slow
    @require_flash_attn
    @pytest.mark.flash_attn_test
    def test_model_8b_long_prompt(self):
        EXPECTED_OUTPUT_TOKEN_IDS = [306, 338]
        # An input with 4097 tokens that is above the size of the sliding window
        input_ids = [1] + [306, 338] * 2048
        model = MinistralForCausalLM.from_pretrained(
            "Mistralai/Ministral-8B-Instruct-2410",
            device_map="auto",
            load_in_4bit=True,
            attn_implementation="flash_attention_2",
        )
        input_ids = torch.tensor([input_ids]).to(model.model.embed_tokens.weight.device)
        generated_ids = model.generate(input_ids, max_new_tokens=4, temperature=0)
        print(generated_ids[0][-2:].tolist())
        self.assertEqual(EXPECTED_OUTPUT_TOKEN_IDS, generated_ids[0][-2:].tolist())

        # Assisted generation
        assistant_model = model
        assistant_model.generation_config.num_assistant_tokens = 2
        assistant_model.generation_config.num_assistant_tokens_schedule = "constant"
        generated_ids = model.generate(input_ids, max_new_tokens=4, temperature=0)
        print(generated_ids[0][-2:].tolist())
        self.assertEqual(EXPECTED_OUTPUT_TOKEN_IDS, generated_ids[0][-2:].tolist())

        del assistant_model
        del model
        backend_empty_cache(torch_device)
        gc.collect()

    @slow
    def test_model_8b_long_prompt_sdpa(self):
        EXPECTED_OUTPUT_TOKEN_IDS = [306, 338]
        # An input with 4097 tokens that is above the size of the sliding window
        input_ids = [1] + [306, 338] * 2048
        model = MinistralForCausalLM.from_pretrained(
            "Mistralai/Ministral-8B-Instruct-2410", device_map="auto", attn_implementation="sdpa"
        )
        input_ids = torch.tensor([input_ids]).to(model.model.embed_tokens.weight.device)
        generated_ids = model.generate(input_ids, max_new_tokens=4, temperature=0)
        self.assertEqual(EXPECTED_OUTPUT_TOKEN_IDS, generated_ids[0][-2:].tolist())

        # Assisted generation
        assistant_model = model
        assistant_model.generation_config.num_assistant_tokens = 2
        assistant_model.generation_config.num_assistant_tokens_schedule = "constant"
        generated_ids = assistant_model.generate(input_ids, max_new_tokens=4, temperature=0)
        self.assertEqual(EXPECTED_OUTPUT_TOKEN_IDS, generated_ids[0][-2:].tolist())

        del assistant_model

        backend_empty_cache(torch_device)
        gc.collect()

        EXPECTED_TEXT_COMPLETION = (
            "My favourite condiment is 100% natural, organic and vegan. I love to use it in my cooking and I"
        )
        prompt = "My favourite condiment is "
        tokenizer = AutoTokenizer.from_pretrained("Mistralai/Ministral-8B-Instruct-2410", use_fast=False)

        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.model.embed_tokens.weight.device)

        # greedy generation outputs
        generated_ids = model.generate(input_ids, max_new_tokens=20, temperature=0)
        text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        print(text)
        self.assertEqual(EXPECTED_TEXT_COMPLETION, text)

    @slow
    def test_export_static_cache(self):
        if version.parse(torch.__version__) < version.parse("2.4.0"):
            self.skipTest(reason="This test requires torch >= 2.4 to run.")

        from transformers.integrations.executorch import (
            TorchExportableModuleWithStaticCache,
        )

        model_id = "Mistralai/Ministral-8B-Instruct-2410"

        tokenizer = AutoTokenizer.from_pretrained(model_id, pad_token="</s>", padding_side="right")

        expected_text_completions = Expectations({
            ("cuda", None): [
                "My favourite condiment is 100% natural, organic, gluten free, vegan, and free from preservatives. I"
            ],
            ("cuda", 8): [
                "My favourite condiment is 100% natural, organic, gluten free, vegan, and vegetarian. I love to use"
            ],
            ("rocm", (9, 5)): [
                "My favourite condiment is 100% natural, organic, gluten free, vegan, and vegetarian. I love to use"
            ]
        })  # fmt: off
        EXPECTED_TEXT_COMPLETION = expected_text_completions.get_expectation()

        max_generation_length = tokenizer(EXPECTED_TEXT_COMPLETION, return_tensors="pt", padding=True)[
            "input_ids"
        ].shape[-1]

        # Load model
        device = "cpu"  # TODO (joao / export experts): should be on `torch_device`, but causes GPU OOM
        dtype = torch.bfloat16
        cache_implementation = "static"
        attn_implementation = "sdpa"
        batch_size = 1
        model = MinistralForCausalLM.from_pretrained(
            model_id,
            device_map=device,
            torch_dtype=dtype,
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

        prompt = ["My favourite condiment is "]
        prompt_tokens = tokenizer(prompt, return_tensors="pt", padding=True).to(model.device)
        prompt_token_ids = prompt_tokens["input_ids"]
        max_new_tokens = max_generation_length - prompt_token_ids.shape[-1]

        # Static Cache + export
        from transformers.integrations.executorch import TorchExportableModuleForDecoderOnlyLM

        exportable_module = TorchExportableModuleForDecoderOnlyLM(model)
        strict = version.parse(torch.__version__) != version.parse(
            "2.7.0"
        )  # Due to https://github.com/pytorch/pytorch/issues/150994
        exported_program = exportable_module.export(
            input_ids=prompt_token_ids,
            cache_position=torch.arange(prompt_token_ids.shape[-1], dtype=torch.long, device=model.device),
            strict=strict,
        )
        ep_generated_ids = TorchExportableModuleWithStaticCache.generate(
            exported_program=exported_program, prompt_token_ids=prompt_token_ids, max_new_tokens=max_new_tokens
        )
        ep_generated_text = tokenizer.batch_decode(ep_generated_ids, skip_special_tokens=True)
        print(ep_generated_text)
        self.assertEqual(EXPECTED_TEXT_COMPLETION, ep_generated_text)

    @require_flash_attn
    @slow
    def test_past_sliding_window_generation(self):
        try:
            from datasets import load_dataset
        except ImportError:
            self.skipTest("datasets not found")

        model = AutoModelForCausalLM.from_pretrained("mistralai/Ministral-8B-Instruct-2410", device_map="auto")
        print(model.device)
        tokenizer = AutoTokenizer.from_pretrained("mistralai/Ministral-8B-Instruct-2410", legacy=False)

        wiki = load_dataset("wikitext", "wikitext-103-raw-v1", split="validation")
        chunks = [x["text"] for x in wiki.select(range(550)) if x["text"].strip()]
        real_corpus = "\n".join(chunks)
        prompt = f"<s>[INST]{real_corpus} Question: Based on the text, at which depth of the continental shelf does H. Gammarus live?[/INST]"

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        input_length = inputs.input_ids.shape[1]  # around 33k tokens > 32k sliding window
        outputs = model.generate(**inputs, max_new_tokens=100, do_sample=False)
        output_text = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
        print(output_text)
        self.assertEqual(
            output_text,
            "H. Gammarus lives on the continental shelf at depths of 0 - 150 metres ( 0 - 492 ft ) , although not normally deeper than 50 m ( 160 ft ) .",
        )
