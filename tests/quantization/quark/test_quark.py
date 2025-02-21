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

import unittest

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, QuarkConfig, GenerationConfig
from transformers.utils.import_utils import is_quark_available
from transformers.testing_utils import (
    is_torch_available,
    require_accelerate,
    require_quark,
    require_torch_gpu,
    require_torch_multi_gpu,
    slow,
)


if is_torch_available():
    import torch

if is_quark_available():
    from quark.torch.export.nn.modules import QParamsLinear


class QuarkConfigTest(unittest.TestCase):
    def test_unknown_arg(self):
        QuarkConfig(foo=2)
    
    def test_commmon_args(self):
        config = AutoConfig.from_pretrained("amd/Llama-3.1-8B-Instruct-w-int8-a-int8-sym-test")
        QuarkConfig(**config.quant_config)

    def test_missing_args(self):
        config = AutoConfig.from_pretrained("amd/Llama-3.1-8B-Instruct-w-int8-a-int8-sym-test")
        
        # TODO: delete one arg from quant_config
        QuarkConfig(**config.quant_config)

@slow
@require_quark
@require_torch_gpu
class QuarkTest(unittest.TestCase):
    reference_model_name = "meta-llama/Llama-3.1-8B-Instruct"
    quantized_model_name = "amd/Llama-3.1-8B-Instruct-w-int8-a-int8-sym-test"

    input_text = "Today I am in Paris and"

    EXPECTED_OUTPUTS = set()
    EXPECTED_OUTPUTS.add("Today I am in Paris and I am not in Paris, France\nToday I am in Paris, Illinois")

    EXPECTED_RELATIVE_DIFFERENCE = 1.664253062
    device_map = None

    # called only once for all test in this class
    @classmethod
    def setUpClass(cls):
        """
        Setup reference & quantized model
        """
        cls.model_fp16 = AutoModelForCausalLM.from_pretrained(
            cls.reference_model_name, torch_dtype=torch.float16, device_map=cls.device_map
        )
        cls.mem_fp16 = cls.model_fp16.get_memory_footprint()

        cls.tokenizer = AutoTokenizer.from_pretrained(cls.model_name, use_fast=True)

        cls.quantized_model = AutoModelForCausalLM.from_pretrained(
            cls.quantized_model_name,
            torch_dtype=torch.float16,
            device_map=cls.device_map,
        )

    def test_memory_footprint(self):
        r"""
        A simple test to check if the model conversion has been done correctly by checking on the
        memory footprint of the converted model
        """
        mem_quantized = self.quantized_model.get_memory_footprint()

        self.assertAlmostEqual(self.mem_fp16 / mem_quantized, self.EXPECTED_RELATIVE_DIFFERENCE)

    def test_device_and_dtype_assignment(self):
        r"""
        Test whether trying to cast (or assigning a device to) a model after quantization will throw an error.
        Checks also if other models are casted correctly.
        """
        # This should work
        if self.device_map is None:
            _ = self.quantized_model.to(0)

        with self.assertRaises(ValueError):
            # Tries with a `dtype``
            self.quantized_model.to(torch.float16)

    def test_original_dtype(self):
        r"""
        A simple test to check if the model succesfully stores the original dtype
        """
        self.assertTrue(hasattr(self.quantized_model.config, "_pre_quantization_dtype"))
        self.assertFalse(hasattr(self.model_fp16.config, "_pre_quantization_dtype"))
        self.assertTrue(self.quantized_model.config._pre_quantization_dtype == torch.float16)

    def test_quantized_layers_class(self):
        """
        Simple test to check if the model conversion has been done correctly by checking on
        the class type of the linear layers of the converted models
        """
        self.assertTrue(isinstance(self.quantized_model.model.layers[0].mlp.gate_proj, QParamsLinear))

    def check_inference_correctness(self, model):
        r"""
        Test the generation quality of the quantized model and see that we are matching the expected output.
        Given that we are operating on small numbers + the testing model is relatively small, we might not get
        the same output across GPUs. So we'll generate few tokens (5-10) and check their output.
        """
        # Check that inference pass works on the model
        encoded_input = self.tokenizer(self.input_text, return_tensors="pt")

        gen_config = GenerationConfig(
            max_new_tokens=15,
            min_new_tokens=15,
            use_cache=True,
            num_beams=1,
            do_sample=False,
        )

        # Check the exactness of the results
        output_sequences = model.generate(input_ids=encoded_input["input_ids"].to(0), generation_config=gen_config)

        # Get the generation
        self.assertIn(self.tokenizer.decode(output_sequences[0], skip_special_tokens=True), self.EXPECTED_OUTPUTS)

    def test_generate_quality(self):
        """
        Simple test to check the quality of the model by comparing the generated tokens with the expected tokens
        """
        if self.device_map is None:
            self.check_inference_correctness(self.quantized_model.to(0))
        else:
            self.check_inference_correctness(self.quantized_model)

@require_accelerate
@require_torch_multi_gpu
@require_quark
class QuarkTestDeviceMap(QuarkTest):
    device_map = "auto"
