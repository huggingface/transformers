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

from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig
from transformers.testing_utils import (
    is_torch_available,
    require_accelerate,
    require_auto_gptq,
    require_optimum,
    require_torch_gpu,
    require_torch_multi_gpu,
    slow,
)


if is_torch_available():
    import torch


@slow
@require_optimum
@require_auto_gptq
@require_torch_gpu
class GPTQTest(unittest.TestCase):
    model_name = "bigscience/bloom-560m"

    input_text = "Hello my name is"
    EXPECTED_OUTPUT = "Hello my name is John and I am a professional photographer. I"

    # this seems a little small considering that we are doing 4bit quant but we have a small model and ww don't quantize the embeddings
    EXPECTED_RELATIVE_DIFFERENCE = 1.664253062

    bits = 4
    group_size = 128
    desc_act = False

    dataset = [
        "auto-gptq is an easy-to-use model quantization library with user-friendly apis, based on GPTQ algorithm."
    ]

    device_map = None

    # called only once for all test in this class
    @classmethod
    def setUpClass(cls):
        """
        Setup quantized model
        """
        cls.model_fp16 = AutoModelForCausalLM.from_pretrained(
            cls.model_name, torch_dtype=torch.float16, device_map=cls.device_map
        )
        cls.mem_fp16 = cls.model_fp16.get_memory_footprint()

        cls.tokenizer = AutoTokenizer.from_pretrained(cls.model_name, use_fast=True)

        quantization_config = GPTQConfig(
            bits=cls.bits, dataset=cls.dataset, group_size=cls.group_size, desc_act=cls.desc_act
        )

        cls.quantized_model = AutoModelForCausalLM.from_pretrained(
            cls.model_name,
            torch_dtype=torch.float16,
            device_map=cls.device_map,
            quantization_config=quantization_config,
        )

    def test_memory_footprint(self):
        r"""
        A simple test to check if the model conversion has been done correctly by checking on the
        memory footprint of the converted model
        """

        mem_quantized = self.quantized_model.get_memory_footprint()

        self.assertAlmostEqual(self.mem_fp16 / mem_quantized, self.EXPECTED_RELATIVE_DIFFERENCE)

    def test_quantized_layers_class(self):
        """
        Simple test to check if the model conversion has been done correctly by checking on
        the class type of the linear layers of the converted models
        """
        from auto_gptq.utils.import_utils import dynamically_import_QuantLinear

        QuantLinear = dynamically_import_QuantLinear(
            use_triton=False, desc_act=self.desc_act, group_size=self.group_size
        )
        self.assertTrue(self.quantized_model.transformer.h[0].mlp.dense_4h_to_h.__class__ == QuantLinear)

    def check_inference_correctness(self, model):
        r"""
        Test the generation quality of the quantized model and see that we are matching the expected output.
        Given that we are operating on small numbers + the testing model is relatively small, we might not get
        the same output across GPUs. So we'll generate few tokens (5-10) and check their output.
        """
        # Check that inference pass works on the model
        encoded_input = self.tokenizer(self.input_text, return_tensors="pt")

        # Check the exactness of the results
        output_parallel = model.generate(input_ids=encoded_input["input_ids"].to(0), max_new_tokens=10)

        # Get the generation
        self.assertEqual(self.tokenizer.decode(output_parallel[0], skip_special_tokens=True), self.EXPECTED_OUTPUT)

    def test_generate_quality(self):
        """
        Simple test to check the quality of the model by comapring the the generated tokens with the expected tokens
        """
        if self.device_map is None:
            self.check_inference_correctness(self.quantized_model.to(0))
        else:
            self.check_inference_correctness(self.quantized_model)

    def test_serialization(self):
        """
        Test the serialization of the model and the loading of the quantized weights works
        """
        with tempfile.TemporaryDirectory() as tmpdirname:
            self.quantized_model.to("cpu").save_pretrained(tmpdirname)
            quantized_model_from_saved = AutoModelForCausalLM.from_pretrained(tmpdirname).to(0)
            self.check_inference_correctness(quantized_model_from_saved)

    @require_accelerate
    def test_serialization_big_model_inference(self):
        """
        Test the serialization of the model and the loading of the quantized weights with big model inference
        """
        with tempfile.TemporaryDirectory() as tmpdirname:
            self.quantized_model.to("cpu").save_pretrained(tmpdirname)
            quantized_model_from_saved = AutoModelForCausalLM.from_pretrained(tmpdirname, device_map="auto")
            self.check_inference_correctness(quantized_model_from_saved)


@require_accelerate
@require_torch_multi_gpu
class GPTQTestDeviceMap(GPTQTest):
    device_map = "auto"
