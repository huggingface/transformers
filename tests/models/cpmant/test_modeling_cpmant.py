# Copyright 2022 The OpenBMB Team and The HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch CPMAnt model."""

import unittest

from transformers.testing_utils import is_torch_available, require_torch, tooslow

from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester


if is_torch_available():
    import torch

    from transformers import (
        CpmAntConfig,
        CpmAntForCausalLM,
        CpmAntModel,
        CpmAntTokenizer,
    )


@require_torch
class CpmAntModelTester(CausalLMModelTester):
    config_class = CpmAntConfig
    if is_torch_available():
        base_model_class = CpmAntModel
        causal_lm_class = CpmAntForCausalLM


@require_torch
class CpmAntModelTest(CausalLMModelTest, unittest.TestCase):
    all_model_classes = (CpmAntModel, CpmAntForCausalLM) if is_torch_available() else ()
    pipeline_model_mapping = (
        {"feature-extraction": CpmAntModel, "text-generation": CpmAntForCausalLM} if is_torch_available() else {}
    )
    model_tester_class = CpmAntModelTester


@require_torch
class CpmAntModelIntegrationTest(unittest.TestCase):
    @tooslow
    def test_inference_masked_lm(self):
        texts = "今天天气真好！"
        model_path = "openbmb/cpm-ant-10b"
        model = CpmAntModel.from_pretrained(model_path)
        tokenizer = CpmAntTokenizer.from_pretrained(model_path)
        inputs = tokenizer(texts, return_tensors="pt")
        hidden_states = model(**inputs).last_hidden_state

        expected_slice = torch.tensor(
            [[[6.1708, 5.9244, 1.0835], [6.5207, 6.2893, -11.3324], [-1.0107, -0.0576, -5.9577]]],
        )
        torch.testing.assert_close(hidden_states[:, :3, :3], expected_slice, rtol=1e-2, atol=1e-2)


@require_torch
class CpmAntForCausalLMlIntegrationTest(unittest.TestCase):
    @tooslow
    def test_inference_casual(self):
        texts = "今天天气真好！"
        model_path = "openbmb/cpm-ant-10b"
        model = CpmAntForCausalLM.from_pretrained(model_path)
        tokenizer = CpmAntTokenizer.from_pretrained(model_path)
        inputs = tokenizer(texts, return_tensors="pt")
        hidden_states = model(**inputs).logits

        expected_slice = torch.tensor(
            [[[-6.4267, -6.4083, -6.3958], [-5.8802, -5.9447, -5.7811], [-5.3896, -5.4820, -5.4295]]],
        )
        torch.testing.assert_close(hidden_states[:, :3, :3], expected_slice, rtol=1e-2, atol=1e-2)

    @tooslow
    def test_simple_generation(self):
        model_path = "openbmb/cpm-ant-10b"
        model = CpmAntForCausalLM.from_pretrained(model_path)
        tokenizer = CpmAntTokenizer.from_pretrained(model_path)
        texts = "今天天气不错，"
        expected_output = "今天天气不错，阳光明媚，我和妈妈一起去超市买东西。\n在超市里，我看到了一个很好玩的玩具，它的名字叫“机器人”。它有一个圆圆的脑袋，两只圆圆的眼睛，还有一个圆圆的"
        model_inputs = tokenizer(texts, return_tensors="pt")
        token_ids = model.generate(**model_inputs)
        output_texts = tokenizer.batch_decode(token_ids)
        self.assertEqual(expected_output, output_texts)

    @tooslow
    def test_batch_generation(self):
        model_path = "openbmb/cpm-ant-10b"
        model = CpmAntForCausalLM.from_pretrained(model_path)
        tokenizer = CpmAntTokenizer.from_pretrained(model_path)
        texts = ["今天天气不错，", "新年快乐，万事如意！"]
        expected_output = [
            "今天天气不错，阳光明媚，我和妈妈一起去超市买东西。\n在超市里，我看到了一个很好玩的玩具，它的名字叫“机器人”。它有一个圆圆的脑袋，两只圆圆的眼睛，还有一个圆圆的",
            "新年快乐，万事如意！在这辞旧迎新的美好时刻，我谨代表《农村新技术》杂志社全体同仁，向一直以来关心、支持《农村新技术》杂志发展的各级领导、各界朋友和广大读者致以最诚挚的",
        ]
        model_inputs = tokenizer(texts, return_tensors="pt", padding=True)
        token_ids = model.generate(**model_inputs)
        output_texts = tokenizer.batch_decode(token_ids)
        self.assertEqual(expected_output, output_texts)
