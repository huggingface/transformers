# Copyright (C) 2024 THL A29 Limited, a Tencent company and The HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch HunYuanMoEV1 model."""

import unittest

import pytest

from transformers import HunYuanMoEV1Config, is_torch_available
from transformers.testing_utils import (
    cleanup,
    require_torch,
    slow,
    torch_device,
)


if is_torch_available():
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        HunYuanMoEV1ForCausalLM,
        HunYuanMoEV1ForSequenceClassification,
        HunYuanMoEV1Model,
    )

from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester


class HunYuanMoEV1ModelTester(CausalLMModelTester):
    config_class = HunYuanMoEV1Config
    if is_torch_available():
        base_model_class = HunYuanMoEV1Model
        causal_lm_class = HunYuanMoEV1ForCausalLM
        sequence_class = HunYuanMoEV1ForSequenceClassification


@require_torch
class HunYuanMoEV1ModelTest(CausalLMModelTest, unittest.TestCase):
    all_model_classes = (
        (
            HunYuanMoEV1Model,
            HunYuanMoEV1ForCausalLM,
            HunYuanMoEV1ForSequenceClassification,
        )
        if is_torch_available()
        else ()
    )
    test_headmasking = False
    test_pruning = False
    test_all_params_have_gradient = False
    model_tester_class = HunYuanMoEV1ModelTester
    pipeline_model_mapping = (
        {
            "feature-extraction": HunYuanMoEV1Model,
            "text-generation": HunYuanMoEV1ForCausalLM,
            "text-classification": HunYuanMoEV1ForSequenceClassification,
        }
        if is_torch_available()
        else {}
    )

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

    @unittest.skip("Hunyuan model Unsupported")
    @pytest.mark.torch_compile_test
    def test_generate_compilation_all_outputs(self):
        pass

    @unittest.skip("Hunyuan model Unsupported")
    @pytest.mark.torch_compile_test
    def test_generate_compile_model_forward(self):
        pass

    @unittest.skip("Hunyuan model Unsupported")
    def test_generate_from_inputs_embeds_with_static_cache(self):
        pass

    @unittest.skip("Hunyuan model Unsupported")
    def test_generate_with_static_cache(self):
        pass


@require_torch
class HunYuanMoEV1IntegrationTest(unittest.TestCase):
    def setUp(self):
        cleanup(torch_device, gc_collect=True)

    def tearDown(self):
        cleanup(torch_device, gc_collect=True)

    @slow
    def test_model_generation(self):
        # we will compele this when model file change over
        # pass
        EXPECTED_ANSWER = "\nOkay, I need to write a short summary about the benefits of regular exercise. Let me start by recalling what I know. First,"
        prompt = "Write a short summary of the benefits of regular exercise"
        tokenizer = AutoTokenizer.from_pretrained("tencent/Hunyuan-A13B-Instruct")
        model = AutoModelForCausalLM.from_pretrained("tencent/Hunyuan-A13B-Instruct", device_map="auto")
        messages = [
            {"role": "user", "content": prompt},
        ]
        tokenized_chat = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        generated_ids = model.generate(tokenized_chat.to(model.device), max_new_tokens=30, top_k=1)
        text = tokenizer.decode(generated_ids[0])
        output = text.split("<think>")[1]
        self.assertEqual(EXPECTED_ANSWER, output)
