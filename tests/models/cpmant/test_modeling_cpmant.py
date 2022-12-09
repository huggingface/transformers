# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
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
""" Testing suite for the PyTorch CPMAnt model. """


import unittest

from transformers.testing_utils import is_torch_available, require_torch, slow


if is_torch_available():
    import torch

    from transformers import CPMAntForCausalLM, CPMAntModel, CPMAntTokenizer


@require_torch
class CPMAntModelTest(unittest.TestCase):

    all_model_classes = (CPMAntModel, CPMAntForCausalLM) if is_torch_available() else ()


@require_torch
class CPMAntModelIntegrationTest(unittest.TestCase):
    @slow
    def test_inference_masked_lm(self):
        texts = ["今天天气真好！"]
        model_path = "openbmb/cpm-ant-10b"
        model = CPMAntModel.from_pretrained(model_path)
        tokenizer = CPMAntTokenizer.from_pretrained(model_path)
        input_ids = tokenizer.get_model_input(texts)
        logits, hidden = model(**input_ids)
        vocab_size = 30720
        expected_shape = torch.Size((1, 38, vocab_size))

        self.assertEqual(logits.shape, expected_shape)

        expected_slice = torch.tensor(
            [[[0.4556, 0.5342, 0.5063], [1.0547, 1.0283, 0.9883], [1.5820, 1.5537, 1.5273]]],
        )
        self.assertTrue(torch.allclose(logits[:, :3, :3], expected_slice, atol=1e-2))


@require_torch
class CPMAntForCausalLMlIntegrationTest(unittest.TestCase):
    @slow
    def test_inference_casual(self):
        texts = ["今天天气真好！"]
        model_path = "openbmb/cpm-ant-10b"
        model = CPMAntForCausalLM.from_pretrained(model_path)
        tokenizer = CPMAntTokenizer.from_pretrained(model_path)
        input_ids = tokenizer.get_model_input(texts)
        logits, hidden = model(**input_ids)
        vocab_size = 30720
        expected_shape = torch.Size((1, 38, vocab_size))

        self.assertEqual(logits.shape, expected_shape)

        expected_slice = torch.tensor(
            [[[0.4556, 0.5342, 0.5063], [1.0547, 1.0283, 0.9883], [1.5820, 1.5537, 1.5273]]],
        )
        self.assertTrue(torch.allclose(logits[:, :3, :3], expected_slice, atol=1e-2))
