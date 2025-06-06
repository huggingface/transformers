# Copyright 2023 The HuggingFace Team. All rights reserved.
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
import unittest

from transformers import AutoTokenizer, RwkvConfig, is_torch_available
from transformers.testing_utils import require_torch, slow, torch_device

from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester


if is_torch_available():
    import torch

    from transformers import (
        RwkvForCausalLM,
        RwkvModel,
    )


class RwkvModelTester(CausalLMModelTester):
    config_class = RwkvConfig
    if is_torch_available():
        base_model_class = RwkvModel
        causal_lm_class = RwkvForCausalLM


@require_torch
class RwkvModelTest(CausalLMModelTest, unittest.TestCase):
    all_model_classes = (RwkvModel, RwkvForCausalLM) if is_torch_available() else ()
    pipeline_model_mapping = (
        {"feature-extraction": RwkvModel, "text-generation": RwkvForCausalLM} if is_torch_available() else {}
    )
    model_tester_class = RwkvModelTester

    @unittest.skip("This model doesn't support padding")
    def test_left_padding_compatibility(self):
        pass


@slow
class RWKVIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.model_id = "RWKV/rwkv-4-169m-pile"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)

    def test_simple_generate(self):
        expected_output = "Hello my name is Jasmine and I am a newbie to the"
        model = RwkvForCausalLM.from_pretrained(self.model_id).to(torch_device)

        input_ids = self.tokenizer("Hello my name is", return_tensors="pt").input_ids.to(torch_device)
        output = model.generate(input_ids, max_new_tokens=10)
        output_sentence = self.tokenizer.decode(output[0].tolist())

        self.assertEqual(output_sentence, expected_output)

    def test_simple_generate_bf16(self):
        expected_output = "Hello my name is Jasmine and I am a newbie to the"

        input_ids = self.tokenizer("Hello my name is", return_tensors="pt").input_ids.to(torch_device)
        model = RwkvForCausalLM.from_pretrained(self.model_id, torch_dtype=torch.bfloat16).to(torch_device)

        output = model.generate(input_ids, max_new_tokens=10)
        output_sentence = self.tokenizer.decode(output[0].tolist())

        self.assertEqual(output_sentence, expected_output)
