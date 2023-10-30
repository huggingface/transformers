# coding=utf-8
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

import tempfile
import unittest

from transformers import AutoModelForCausalLM, AutoTokenizer, AwqConfig
from transformers.testing_utils import (
    require_accelerate,
    require_auto_awq,
    require_torch_gpu,
    slow,
    torch_device,
)


@require_torch_gpu
class AwqConfigTest(unittest.TestCase):
    def test_wrong_backend(self):
        """
        Simple test that checks if a user passes a wrong backend an error is raised
        """
        # This should work fine
        _ = AwqConfig(bits=4)

        with self.assertRaises(ValueError):
            AwqConfig(bits=4, backend="")

        # LLMAWQ does not work on a T4
        with self.assertRaises(ValueError):
            AwqConfig(bits=4, backend="llm-awq")

    def test_to_dict(self):
        """
        Simple test that checks if one uses a config and converts it to a dict, the dict is the same as the config object
        """
        quantization_config = AwqConfig(bits=4)
        config_to_dict = quantization_config.to_dict()

        for key in config_to_dict:
            self.assertEqual(getattr(quantization_config, key), config_to_dict[key])

    def test_from_dict(self):
        """
        Simple test that checks if one uses a dict and converts it to a config object, the config object is the same as the dict
        """
        dict = {"bits": 2, "zero_point": False, "backend": "autoawq"}
        quantization_config = AwqConfig.from_dict(dict)

        self.assertEqual(dict["bits"], quantization_config.bits)
        self.assertEqual(dict["zero_point"], quantization_config.zero_point)
        self.assertEqual(dict["backend"], quantization_config.backend)


@slow
@require_torch_gpu
@require_auto_awq
@require_accelerate
class AwqTest(unittest.TestCase):
    # TODO: @younesbelkada change it to `TheBloke/Mistral-7B-v0.1-AWQ` in the future
    model_name = "ybelkada/test-mistral-7b-v0.1-awq"
    dummy_transformers_model_name = "bigscience/bloom-560m"

    input_text = "Hello my name is"

    EXPECTED_OUTPUT = "Hello my name is Katie and I am a 20 year old student at the University of North Carolina at Chapel Hill. I am a junior and I am majoring in Journalism and minoring in Spanish"

    device_map = "cuda"

    # called only once for all test in this class
    @classmethod
    def setUpClass(cls):
        """
        Setup quantized model
        """
        cls.tokenizer = AutoTokenizer.from_pretrained(cls.model_name)
        cls.quantized_model = AutoModelForCausalLM.from_pretrained(
            cls.model_name,
            device_map=cls.device_map,
        )

    def test_quantized_model(self):
        """
        Simple test that checks if the quantized model is working properly
        """
        input_ids = self.tokenizer(self.input_text, return_tensors="pt").to(torch_device)

        output = self.quantized_model.generate(**input_ids, max_new_tokens=40)
        self.assertEqual(self.tokenizer.decode(output[0], skip_special_tokens=True), self.EXPECTED_OUTPUT)

    def test_quantized_model_no_device_map(self):
        """
        Simple test that checks if the quantized model is working properly
        """
        input_ids = self.tokenizer(self.input_text, return_tensors="pt").to(torch_device)

        quantized_model = AutoModelForCausalLM.from_pretrained(self.model_name).to(torch_device)
        output = quantized_model.generate(**input_ids, max_new_tokens=40)

        self.assertEqual(self.tokenizer.decode(output[0], skip_special_tokens=True), self.EXPECTED_OUTPUT)

    def test_save_pretrained(self):
        """
        Simple test that checks if the quantized model is working properly after being saved and loaded
        """
        with tempfile.TemporaryDirectory() as tmpdirname:
            self.quantized_model.save_pretrained(tmpdirname)
            model = AutoModelForCausalLM.from_pretrained(tmpdirname, device_map=self.device_map)

            input_ids = self.tokenizer(self.input_text, return_tensors="pt").to(torch_device)

            output = model.generate(**input_ids, max_new_tokens=40)
            self.assertEqual(self.tokenizer.decode(output[0], skip_special_tokens=True), self.EXPECTED_OUTPUT)

    def test_raise_quantization(self):
        """
        Simple test that checks if one passes a quantization config to quantize a model, it raises an error
        """
        quantization_config = AwqConfig(bits=4)

        with self.assertRaises(ValueError) as context:
            _ = AutoModelForCausalLM.from_pretrained(
                self.dummy_transformers_model_name, quantization_config=quantization_config
            )

        self.assertEqual(
            str(context.exception),
            "You cannot pass an `AwqConfig` when loading a model as you can only use AWQ models for inference. To quantize transformers models with AWQ algorithm, please refer to our quantization docs.",
        )
