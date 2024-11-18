# coding=utf-8
# Copyright 2024 The HuggingFace Team. All rights reserved.
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

import gc
import tempfile
import unittest

from transformers import AutoModelForCausalLM, AutoTokenizer, VptqConfig
from transformers.testing_utils import (
    require_accelerate,
    require_torch_gpu,
    require_torch_multi_gpu,
    require_vptq,
    slow,
    torch_device,
)
from transformers.utils import is_torch_available


if is_torch_available():
    import torch


class VptqConfigTest(unittest.TestCase):
    def test_to_dict(self):
        """
        Makes sure the config format is properly set
        """
        quantization_config = VptqConfig()
        vptq_orig_config = quantization_config.to_dict()

        self.assertEqual(quantization_config.quant_config, vptq_orig_config["quant_config"])


@slow
@require_torch_gpu
@require_vptq
@require_accelerate
class VptqTest(unittest.TestCase):
    model_name = "VPTQ-community/Meta-Llama-3.1-8B-Instruct-v12-k65536-4096-woft"

    input_text = "Hello my name is"
    max_new_tokens = 32

    EXPECTED_OUTPUT = "Hello my name is Sarah and I am a 25 year old woman from the United States. I am a college graduate and I am currently working as a marketing specialist for a small"

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

    def tearDown(self):
        gc.collect()
        torch.cuda.empty_cache()
        gc.collect()

    def test_quantized_model(self):
        """
        Simple test that checks if the quantized model is working properly
        """
        input_ids = self.tokenizer(self.input_text, return_tensors="pt").to(torch_device)

        output = self.quantized_model.generate(**input_ids, max_new_tokens=self.max_new_tokens, do_sample=False)
        self.assertEqual(self.tokenizer.decode(output[0], skip_special_tokens=True), self.EXPECTED_OUTPUT)

    def test_raise_if_non_quantized(self):
        model_id = "facebook/opt-125m"
        quantization_config = VptqConfig()

        with self.assertRaises(ValueError):
            _ = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=quantization_config)

    def test_save_pretrained(self):
        """
        Simple test that checks if the quantized model is working properly after being saved and loaded
        """
        with tempfile.TemporaryDirectory() as tmpdirname:
            self.quantized_model.save_pretrained(tmpdirname)
            model = AutoModelForCausalLM.from_pretrained(tmpdirname, device_map=self.device_map)

            input_ids = self.tokenizer(self.input_text, return_tensors="pt").to(torch_device)

            output = model.generate(**input_ids, max_new_tokens=self.max_new_tokens, do_sample=False)
            self.assertEqual(self.tokenizer.decode(output[0], skip_special_tokens=True), self.EXPECTED_OUTPUT)

    @require_torch_multi_gpu
    def test_quantized_model_multi_gpu(self):
        """
        Simple test that checks if the quantized model is working properly with multiple GPUs
        """
        input_ids = self.tokenizer(self.input_text, return_tensors="pt").to(torch_device)

        quantized_model = AutoModelForCausalLM.from_pretrained(self.model_name, device_map="auto")

        self.assertTrue(set(quantized_model.hf_device_map.values()) == {0, 1})

        output = quantized_model.generate(**input_ids, max_new_tokens=self.max_new_tokens, do_sample=False)

        self.assertEqual(self.tokenizer.decode(output[0], skip_special_tokens=True), self.EXPECTED_OUTPUT)
