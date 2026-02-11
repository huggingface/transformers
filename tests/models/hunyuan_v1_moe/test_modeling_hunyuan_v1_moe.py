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
import torch

from transformers import is_torch_available
from transformers.testing_utils import (
    cleanup,
    require_torch,
    slow,
    torch_device,
)


if is_torch_available():
    from transformers import AutoTokenizer, HunYuanMoEV1ForCausalLM, HunYuanMoEV1Model

from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester


class HunYuanMoEV1ModelTester(CausalLMModelTester):
    if is_torch_available():
        base_model_class = HunYuanMoEV1Model


@require_torch
class HunYuanMoEV1ModelTest(CausalLMModelTest, unittest.TestCase):
    test_all_params_have_gradient = False
    model_tester_class = HunYuanMoEV1ModelTester

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
        EXPECTED_ANSWER = "\nOkay, I need to write a"
        prompt = "Write a short summary of the benefits of regular exercise"
        tokenizer = AutoTokenizer.from_pretrained("tencent/Hunyuan-A13B-Instruct")
        model = HunYuanMoEV1ForCausalLM.from_pretrained(
            "tencent/Hunyuan-A13B-Instruct", device_map="auto", dtype=torch.bfloat16
        )
        messages = [
            {"role": "user", "content": prompt},
        ]
        tokenized_chat = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(model.device)
        generated_ids = model.generate(**tokenized_chat, max_new_tokens=10, top_k=1)
        text = tokenizer.decode(generated_ids[0])
        output = text.split("<think>")[1]
        self.assertEqual(EXPECTED_ANSWER, output)
