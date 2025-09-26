# coding=utf-8
# Copyright 2025 The LG AI Research and The HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch EXAONE 4.0 model."""

import unittest

import pytest
from packaging import version

from transformers import (
    AutoTokenizer,
    Exaone4Config,
    GenerationConfig,
    is_torch_available,
)
from transformers.testing_utils import (
    cleanup,
    require_flash_attn,
    require_torch,
    require_torch_accelerator,
    slow,
    torch_device,
)

from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester
from ...test_configuration_common import ConfigTester


if is_torch_available():
    import torch

    from transformers import (
        Exaone4ForCausalLM,
        Exaone4ForQuestionAnswering,
        Exaone4ForSequenceClassification,
        Exaone4ForTokenClassification,
        Exaone4Model,
    )


class Exaone4ModelTester(CausalLMModelTester):
    config_class = Exaone4Config
    if is_torch_available():
        base_model_class = Exaone4Model
        causal_lm_class = Exaone4ForCausalLM
        sequence_class = Exaone4ForSequenceClassification
        token_class = Exaone4ForTokenClassification
        question_answering_class = Exaone4ForQuestionAnswering


@require_torch
class Exaone4ModelTest(CausalLMModelTest, unittest.TestCase):
    all_model_classes = (
        (
            Exaone4Model,
            Exaone4ForCausalLM,
            Exaone4ForSequenceClassification,
            Exaone4ForQuestionAnswering,
            Exaone4ForTokenClassification,
        )
        if is_torch_available()
        else ()
    )
    pipeline_model_mapping = (
        {
            "feature-extraction": Exaone4Model,
            "question-answering": Exaone4ForQuestionAnswering,
            "text-classification": Exaone4ForSequenceClassification,
            "text-generation": Exaone4ForCausalLM,
            "zero-shot": Exaone4ForSequenceClassification,
            "token-classification": Exaone4ForTokenClassification,
        }
        if is_torch_available()
        else {}
    )
    test_headmasking = False
    test_pruning = False
    fx_compatible = False  # Broken by attention refactor cc @Cyrilvallez
    model_tester_class = Exaone4ModelTester
    model_split_percents = [0.5, 0.6]

    def setUp(self):
        self.model_tester = Exaone4ModelTester(self)
        self.config_tester = ConfigTester(self, config_class=Exaone4Config, hidden_size=37)


@require_torch
class Exaone4IntegrationTest(unittest.TestCase):
    TEST_MODEL_ID = "LGAI-EXAONE/EXAONE-4.0-32B"

    def setUp(self):
        cleanup(torch_device, gc_collect=True)

    def tearDown(self):
        # TODO (joao): automatic compilation, i.e. compilation when `cache_implementation="static"` is used, leaves
        # some memory allocated in the cache, which means some object is not being released properly. This causes some
        # unoptimal memory usage, e.g. after certain teruff format examples tests src utilssts a 7B model in FP16 no longer fits in a 24GB GPU.
        # Investigate the root cause.
        cleanup(torch_device, gc_collect=True)

    @slow
    def test_model_logits(self):
        input_ids = [405, 7584, 79579, 76636, 2907, 94640, 373]
        model = Exaone4ForCausalLM.from_pretrained(
            self.TEST_MODEL_ID,
            device_map="auto",
            dtype=torch.bfloat16,
        )
        input_ids = torch.tensor([input_ids]).to(model.model.embed_tokens.weight.device)
        with torch.no_grad():
            out = model(input_ids).logits.float().cpu()

        EXPECTED_MEAN = torch.tensor([[22.1993, 8.5845, 10.0401, 12.4262, 9.3112, 29.7933, 8.2628]])
        EXPECTED_SLICE = torch.tensor(
            [20.6250, 19.6250, 14.5000, 21.1250, 24.5000, 22.1250, 24.0000, 24.8750, 25.0000, 25.3750]
        )

        torch.testing.assert_close(out.mean(-1), EXPECTED_MEAN, atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(out[0, 0, :10], EXPECTED_SLICE, atol=1e-4, rtol=1e-4)

    @slow
    def test_model_generation_eager(self):
        EXPECTED_TEXT = "Tell me about the Miracle on the Han river.\n\nOkay, the Miracle on the Han River refers to the rapid industrialization and economic growth of South"
        prompt = "Tell me about the Miracle on the Han river."
        tokenizer = AutoTokenizer.from_pretrained(self.TEST_MODEL_ID)
        model = Exaone4ForCausalLM.from_pretrained(
            self.TEST_MODEL_ID, device_map="auto", dtype=torch.bfloat16, attn_implementation="eager"
        )
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.model.embed_tokens.weight.device)

        # greedy generation outputs
        generated_ids = model.generate(input_ids, max_new_tokens=20, temperature=0)
        text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        self.assertEqual(EXPECTED_TEXT, text)

    @slow
    def test_model_generation_sdpa(self):
        EXPECTED_TEXT = "Tell me about the Miracle on the Han river.\n\nOkay, the Miracle on the Han River refers to the rapid industrialization and economic growth of South"
        prompt = "Tell me about the Miracle on the Han river."
        tokenizer = AutoTokenizer.from_pretrained(self.TEST_MODEL_ID)
        model = Exaone4ForCausalLM.from_pretrained(
            self.TEST_MODEL_ID, device_map="auto", dtype=torch.bfloat16, attn_implementation="sdpa"
        )
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.model.embed_tokens.weight.device)

        # greedy generation outputs
        generated_ids = model.generate(input_ids, max_new_tokens=20, temperature=0)
        text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        self.assertEqual(EXPECTED_TEXT, text)

    @slow
    @require_torch_accelerator
    @require_flash_attn
    def test_model_generation_long_flash(self):
        EXPECTED_OUTPUT_TOKEN_IDS = [433, 9055]
        input_ids = [433, 9055] * 2048
        model = Exaone4ForCausalLM.from_pretrained(
            self.TEST_MODEL_ID, device_map="auto", dtype=torch.bfloat16, attn_implementation="flash_attention_2"
        )
        input_ids = torch.tensor([input_ids]).to(model.model.embed_tokens.weight.device)

        generated_ids = model.generate(input_ids, max_new_tokens=4, temperature=0)
        self.assertEqual(EXPECTED_OUTPUT_TOKEN_IDS, generated_ids[0][-2:].tolist())

    @slow
    @require_torch_accelerator
    def test_model_generation_beyond_sliding_window(self):
        EXPECTED_TEXT_COMPLETION = " This is a nice place. I really enjoy the scenery, and the atmosphere is so relaxing. I'm grateful for the opportunity to experience this place. It"
        tokenizer = AutoTokenizer.from_pretrained(self.TEST_MODEL_ID)
        prompt = "This is a nice place. " * 700 + "I really enjoy the scenery,"
        model = Exaone4ForCausalLM.from_pretrained(
            self.TEST_MODEL_ID, device_map="auto", dtype=torch.bfloat16, attn_implementation="sdpa"
        )
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.model.embed_tokens.weight.device)

        generated_ids = model.generate(input_ids, max_new_tokens=20, temperature=0)
        text = tokenizer.decode(generated_ids[0, -32:], skip_special_tokens=True)
        self.assertEqual(EXPECTED_TEXT_COMPLETION, text)

    @pytest.mark.torch_export_test
    @slow
    def test_export_static_cache(self):
        if version.parse(torch.__version__) < version.parse("2.4.0"):
            self.skipTest(reason="This test requires torch >= 2.4 to run.")

        from transformers.integrations.executorch import (
            TorchExportableModuleWithStaticCache,
            convert_and_export_with_cache,
        )

        tokenizer = AutoTokenizer.from_pretrained(self.TEST_MODEL_ID, padding_side="right")
        EXPECTED_TEXT_COMPLETION = ["The Deep Learning is \n['Deep Learning',"]
        max_generation_length = tokenizer(EXPECTED_TEXT_COMPLETION, return_tensors="pt", padding=True)[
            "input_ids"
        ].shape[-1]

        # Load model
        device = "cpu"
        dtype = torch.bfloat16
        cache_implementation = "static"
        attn_implementation = "sdpa"
        batch_size = 1
        model = Exaone4ForCausalLM.from_pretrained(
            self.TEST_MODEL_ID,
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

        prompt = ["The Deep Learning is "]
        prompt_tokens = tokenizer(prompt, return_tensors="pt", padding=True).to(model.device)
        prompt_token_ids = prompt_tokens["input_ids"]
        max_new_tokens = max_generation_length - prompt_token_ids.shape[-1]

        # Static Cache + export
        exported_program = convert_and_export_with_cache(model)
        ep_generated_ids = TorchExportableModuleWithStaticCache.generate(
            exported_program=exported_program, prompt_token_ids=prompt_token_ids, max_new_tokens=max_new_tokens
        )
        ep_generated_text = tokenizer.batch_decode(ep_generated_ids, skip_special_tokens=True)
        self.assertEqual(EXPECTED_TEXT_COMPLETION, ep_generated_text)
