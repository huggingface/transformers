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
import logging
import unittest

import pytest

from transformers import AutoTokenizer, BitsAndBytesConfig, GenerationConfig, is_torch_available
from transformers.testing_utils import (
    backend_empty_cache,
    cleanup,
    require_bitsandbytes,
    require_flash_attn,
    require_torch,
    require_torch_accelerator,
    slow,
    torch_device,
)


if is_torch_available():
    import torch

    from transformers import (
        AutoModelForCausalLM,
        MinistralForCausalLM,
        MinistralModel,
    )


from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester


class MinistralModelTester(CausalLMModelTester):
    if is_torch_available():
        base_model_class = MinistralModel


@require_torch
class MinistralModelTest(CausalLMModelTest, unittest.TestCase):
    model_tester_class = MinistralModelTester

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
    @require_torch_accelerator
    @pytest.mark.flash_attn_test
    @slow
    def test_flash_attn_2_inference_equivalence_right_padding(self):
        self.skipTest(reason="Ministral flash attention does not support right padding")


@require_torch
class MinistralIntegrationTest(unittest.TestCase):
    def tearDown(self):
        cleanup(torch_device, gc_collect=True)

    @slow
    def test_model_8b_logits(self):
        input_ids = [1, 306, 4658, 278, 6593, 310, 2834, 338]
        model = AutoModelForCausalLM.from_pretrained("mistralai/Ministral-8B-Instruct-2410", device_map="auto")
        assert isinstance(model, MinistralForCausalLM)
        input_ids = torch.tensor([input_ids]).to(model.model.embed_tokens.weight.device)
        with torch.no_grad():
            out = model(input_ids).logits.float().cpu()
        # Expected mean on dim = -1
        EXPECTED_MEAN = torch.tensor([[-1.5029, -7.2815, 4.5190, 0.5930, -5.2526, 3.0765, -0.6314, 1.8068]])
        torch.testing.assert_close(out.mean(-1), EXPECTED_MEAN, rtol=1e-2, atol=1e-2)
        # slicing logits[0, 0, 0:30]
        EXPECTED_SLICE = torch.tensor([-3.9446, -3.9466,  0.6383, -3.9466, -3.9468, -3.9448, -3.9462, -3.9455,
                                                    -3.9451, -0.8244, -3.9472, -3.9458, -3.9460, -3.9406, -3.9462, -3.9462,
                                                    -3.9458, -3.9462, -3.9463, -3.9461, -3.9448, -3.9451, -3.9462, -3.9458,
                                                    -3.9455, -3.9452, -3.9458, -3.9469, -3.9460, -3.9464])  # fmt: skip
        torch.testing.assert_close(out[0, 0, :30], EXPECTED_SLICE, rtol=1e-4, atol=1e-4)

        del model
        backend_empty_cache(torch_device)
        gc.collect()

    @slow
    def test_model_8b_generation(self):
        EXPECTED_TEXT_COMPLETION = "My favourite condiment is 100% natural, 100% organic, 100% free of"
        prompt = "My favourite condiment is "
        tokenizer = AutoTokenizer.from_pretrained("Mistralai/Ministral-8B-Instruct-2410")
        model = MinistralForCausalLM.from_pretrained("Mistralai/Ministral-8B-Instruct-2410", device_map="auto")
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.model.embed_tokens.weight.device)

        # greedy generation outputs
        generated_ids = model.generate(input_ids, max_new_tokens=20, temperature=0)
        text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        self.assertEqual(EXPECTED_TEXT_COMPLETION, text)

        del model
        backend_empty_cache(torch_device)
        gc.collect()

    @require_bitsandbytes
    @slow
    @require_flash_attn
    @pytest.mark.flash_attn_test
    def test_model_8b_long_prompt(self):
        EXPECTED_OUTPUT_TOKEN_IDS = [36850, 4112]
        # An input with 4097 tokens that is above the size of the sliding window
        input_ids = [1] + [306, 338] * 2048
        model = MinistralForCausalLM.from_pretrained(
            "Mistralai/Ministral-8B-Instruct-2410",
            device_map="auto",
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )
        input_ids = torch.tensor([input_ids]).to(model.model.embed_tokens.weight.device)
        generated_ids = model.generate(input_ids, max_new_tokens=4, temperature=0)
        self.assertEqual(EXPECTED_OUTPUT_TOKEN_IDS, generated_ids[0][-2:].tolist())

        # Assisted generation
        assistant_model = model
        assistant_model.generation_config.num_assistant_tokens = 2
        assistant_model.generation_config.num_assistant_tokens_schedule = "constant"
        generated_ids = model.generate(input_ids, max_new_tokens=4, temperature=0)
        self.assertEqual(EXPECTED_OUTPUT_TOKEN_IDS, generated_ids[0][-2:].tolist())

        del assistant_model
        del model
        backend_empty_cache(torch_device)
        gc.collect()

    @slow
    @unittest.skip("not working with Ministral")
    @pytest.mark.torch_export_test
    def test_export_text_with_hybrid_cache(self):
        # TODO: Exportability is not working
        from transformers.testing_utils import is_torch_greater_or_equal

        if not is_torch_greater_or_equal("2.6.0"):
            self.skipTest(reason="This test requires torch >= 2.6 to run.")

        from transformers.integrations.executorch import TorchExportableModuleForDecoderOnlyLM

        model_id = "Mistralai/Ministral-8B-Instruct-2410"
        model = MinistralForCausalLM.from_pretrained(
            model_id,
            generation_config=GenerationConfig(
                use_cache=True,
                cache_implementation="static",
                cache_config={
                    "batch_size": 1,
                    "max_cache_len": 50,
                },
            ),
        )

        # Export + HybridCache
        model.eval()
        exportable_module = TorchExportableModuleForDecoderOnlyLM(model)
        exported_program = exportable_module.export(
            input_ids=torch.tensor([[1]], dtype=torch.long, device=model.device),
            cache_position=torch.tensor([0], dtype=torch.long, device=model.device),
        )
        logging.info(f"\nExported program: {exported_program}")

        # Test generation with the exported model
        prompt = "My favourite condiment is "
        max_new_tokens_to_generate = 20
        # Generate text with the exported model
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        export_generated_text = TorchExportableModuleForDecoderOnlyLM.generate(
            exported_program, tokenizer, prompt, max_new_tokens=max_new_tokens_to_generate
        )
        logging.info(f"\nExport generated texts: '{export_generated_text}'")

        input_text = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            eager_outputs = model.generate(
                **input_text,
                max_new_tokens=max_new_tokens_to_generate,
                do_sample=False,  # Use greedy decoding to match the exported model
                cache_implementation="static",
            )

        eager_generated_text = tokenizer.decode(eager_outputs[0], skip_special_tokens=True)
        logging.info(f"\nEager generated texts: '{eager_generated_text}'")

        self.assertEqual(export_generated_text, eager_generated_text)

    @pytest.mark.flash_attn_test
    @require_flash_attn
    @slow
    def test_past_sliding_window_generation(self):
        try:
            from datasets import load_dataset
        except ImportError:
            self.skipTest("datasets not found")

        model = MinistralForCausalLM.from_pretrained(
            "mistralai/Ministral-8B-Instruct-2410",
            device_map="auto",
            quantization_config=BitsAndBytesConfig(load_in_4bit=True),
        )
        tokenizer = AutoTokenizer.from_pretrained("mistralai/Ministral-8B-Instruct-2410", legacy=False)

        wiki = load_dataset("wikitext", "wikitext-103-raw-v1", split="validation")
        chunks = [x["text"] for x in wiki.select(range(550)) if x["text"].strip()]
        real_corpus = "\n".join(chunks)
        prompt = f"<s>[INST]{real_corpus} Question: Based on the text, at which depth of the continental shelf does H. Gammarus live?[/INST]"

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        input_length = inputs.input_ids.shape[1]  # around 33k tokens > 32k sliding window
        outputs = model.generate(**inputs, max_new_tokens=100, do_sample=False)
        output_text = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
        self.assertEqual(
            output_text,
            " H. Gammarus lives on the continental shelf at depths of 0 – 150 metres ( 0 – 492 ft ) , although not normally deeper than 50 m ( 160 ft ) .",
        )
