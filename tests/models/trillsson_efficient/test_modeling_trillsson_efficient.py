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
""" Testing suite for the PyTorch Trillsson_efficient model. """

import unittest

from datasets import load_dataset

from transformers.testing_utils import is_torch_available, require_torch, slow, torch_device


if is_torch_available():
    import torch

    from transformers import Trillsson_efficientFeatureExtractor, Trillsson_efficientModel


@require_torch
@slow
class Trillson_efficientModelTester(unittest.TestCase):
    def _load_datasamples(self, num_samples):
        ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        # automatic decoding with librispeech
        speech_samples = ds.sort("id").filter(
            lambda x: x["id"] in [f"1272-141231-000{i}" for i in range(num_samples)]
        )[:num_samples]["audio"]

        return [x["array"] for x in speech_samples]

    def test_inference_embedding_normal(self):
        checkpoint = "vumichien/nonsemantic-speech-trillsson3"
        model = Trillsson_efficientModel.from_pretrained(checkpoint)
        model.to(torch_device)
        processor = Trillsson_efficientFeatureExtractor.from_pretrained(checkpoint)
        input_speech = self._load_datasamples(1)

        input_values = processor(input_speech, return_tensors="pt").input_values.to(torch_device)

        EXPECTED_INPUT_SHAPE = torch.Size([1, 80, 39, 195])
        self.assertEqual(input_values.shape, EXPECTED_INPUT_SHAPE)

        with torch.no_grad():
            outputs = model(input_values)
        last_hidden_state = outputs.last_hidden_state

        EXPECTED_SHAPE = torch.Size((1, 1024))
        self.assertEqual(last_hidden_state.shape, EXPECTED_SHAPE)

        EXPECTED_SUM = torch.tensor(1623.8915)
        self.assertTrue(torch.allclose(torch.sum(last_hidden_state.abs()), EXPECTED_SUM))

    @slow
    def test_inference_embedding_normal_batched(self):
        checkpoint = "vumichien/nonsemantic-speech-trillsson3"
        model = Trillsson_efficientModel.from_pretrained(checkpoint)
        model.to(torch_device)
        processor = Trillsson_efficientFeatureExtractor.from_pretrained(checkpoint)
        input_speech = self._load_datasamples(2)

        input_values = processor(input_speech, return_tensors="pt", padding=True).input_values.to(torch_device)

        EXPECTED_INPUT_SHAPE = torch.Size([2, 80, 66, 195])
        self.assertEqual(input_values.shape, EXPECTED_INPUT_SHAPE)

        with torch.no_grad():
            outputs = model(input_values)
        last_hidden_state = outputs.last_hidden_state

        EXPECTED_SHAPE = torch.Size((2, 1024))
        self.assertEqual(last_hidden_state.shape, EXPECTED_SHAPE)

        EXPECTED_SUM = torch.tensor(3183.4692)
        self.assertTrue(torch.allclose(torch.sum(last_hidden_state.abs()), EXPECTED_SUM))

    @slow
    def test_inference_embedding_robust_batched(self):
        checkpoint = "vumichien/nonsemantic-speech-trillsson3"
        model = Trillsson_efficientModel.from_pretrained(checkpoint)
        model.to(torch_device)
        processor = Trillsson_efficientFeatureExtractor.from_pretrained(checkpoint)
        input_speech = self._load_datasamples(4)

        input_values = processor(input_speech, return_tensors="pt", padding=True).input_values.to(torch_device)

        EXPECTED_INPUT_SHAPE = torch.Size([4, 80, 163, 195])
        self.assertEqual(input_values.shape, EXPECTED_INPUT_SHAPE)

        with torch.no_grad():
            outputs = model(input_values)
        last_hidden_state = outputs.last_hidden_state

        EXPECTED_SHAPE = torch.Size((4, 1024))
        self.assertEqual(last_hidden_state.shape, EXPECTED_SHAPE)

        EXPECTED_SUM = torch.tensor(6372.4956)
        self.assertTrue(torch.allclose(torch.sum(last_hidden_state.abs()), EXPECTED_SUM))
