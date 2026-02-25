# Copyright 2026 The HuggingFace Team. All rights reserved.
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

from transformers import AutoModelForCausalLM, AutoTokenizer, FourOverSixConfig
from transformers.testing_utils import (
    backend_empty_cache,
    require_accelerate,
    require_fouroversix,
    require_torch_accelerator,
    require_torch_multi_accelerator,
    slow,
    torch_device,
)


@require_torch_accelerator
class FourOverSixConfigTest(unittest.TestCase):
    def test_to_dict(self):
        """
        Simple test that checks if one uses a config and converts it to a dict, the dict is the same as the config object
        """
        quantization_config = FourOverSixConfig()
        config_to_dict = quantization_config.to_dict()

        for key in config_to_dict:
            self.assertEqual(getattr(quantization_config, key), config_to_dict[key])

    def test_from_dict(self):
        """
        Simple test that checks if one uses a dict and converts it to a config object, the config object is the same as the dict
        """
        dict = {
            "scale_rule": "mse",
            "quant_method": "fouroversix",
        }
        quantization_config = FourOverSixConfig.from_dict(dict)

        self.assertEqual(dict["scale_rule"], quantization_config.scale_rule)
        self.assertEqual(dict["quant_method"], quantization_config.quant_method)


@slow
@require_torch_accelerator
@require_fouroversix
@require_accelerate
class FourOverSixBaseTest(unittest.TestCase):
    model_name = "unsloth/Llama-3.2-3B"

    input_text = "1 2 3 4"
    max_new_tokens = 4

    EXPECTED_OUTPUT = "1 2 3 4 5 6"

    device_map = torch_device

    @classmethod
    def getQuantizationConfig(cls):
        unittest.skip("Subclass must implement this method")

    # Called only once for all tests in this class
    @classmethod
    def setUpClass(cls):
        """
        Setup quantized model
        """

        cls.quantization_config = cls.getQuantizationConfig()
        cls.tokenizer = AutoTokenizer.from_pretrained(cls.model_name)
        cls.quantized_model = AutoModelForCausalLM.from_pretrained(
            cls.model_name,
            device_map=cls.device_map,
            quantization_config=cls.quantization_config,
        )

    def tearDown(self):
        gc.collect()
        backend_empty_cache(torch_device)
        gc.collect()

    def test_quantized_model(self):
        """
        Simple test that checks if the quantized model is working properly
        """
        input_ids = self.tokenizer(self.input_text, return_tensors="pt").to(torch_device)

        output = self.quantized_model.generate(**input_ids, max_new_tokens=self.max_new_tokens)
        self.assertEqual(
            self.tokenizer.decode(output[0], skip_special_tokens=True),
            self.EXPECTED_OUTPUT,
        )

    def test_save_pretrained(self):
        """
        Simple test that checks if the quantized model is working properly after being saved and loaded
        """
        with tempfile.TemporaryDirectory() as tmpdirname:
            self.quantized_model.save_pretrained(tmpdirname)

            model = AutoModelForCausalLM.from_pretrained(tmpdirname, device_map=self.device_map)

            input_ids = self.tokenizer(self.input_text, return_tensors="pt").to(torch_device)

            output = model.generate(**input_ids, max_new_tokens=self.max_new_tokens)
            self.assertEqual(
                self.tokenizer.decode(output[0], skip_special_tokens=True),
                self.EXPECTED_OUTPUT,
            )

    @require_torch_multi_accelerator
    def test_quantized_model_multi_accelerator(self):
        """
        Simple test that checks if the quantized model is working properly with multiple accelerators.
        Set CUDA_VISIBLE_DEVICES=0,1 if you have more than 2 CUDA GPUs.
        """
        input_ids = self.tokenizer(self.input_text, return_tensors="pt").to("cuda:0")

        quantized_model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map="auto",
            quantization_config=self.quantization_config,
            max_memory={0: "1GB", 1: "10GB"},
        )
        self.assertTrue(set(quantized_model.hf_device_map.values()) == {0, 1})

        output = quantized_model.generate(**input_ids, max_new_tokens=self.max_new_tokens)
        self.assertEqual(
            self.tokenizer.decode(output[0], skip_special_tokens=True),
            self.EXPECTED_OUTPUT,
        )

    @require_torch_multi_accelerator
    def test_save_pretrained_multi_accelerator(self):
        """
        Simple test that checks if the quantized model is working properly after being saved and loaded
        """
        with tempfile.TemporaryDirectory() as tmpdirname:
            self.quantized_model.save_pretrained(tmpdirname)

            model = AutoModelForCausalLM.from_pretrained(
                tmpdirname,
                device_map="sequential",
                max_memory={0: "1GB", 1: "10GB"},
            )
            self.assertTrue(set(model.hf_device_map.values()) == {0, 1})

            input_ids = self.tokenizer(self.input_text, return_tensors="pt").to(torch_device)

            output = model.generate(**input_ids, max_new_tokens=self.max_new_tokens)
            self.assertEqual(
                self.tokenizer.decode(output[0], skip_special_tokens=True),
                self.EXPECTED_OUTPUT,
            )


class FourOverSixMSETest(FourOverSixBaseTest):
    @classmethod
    def getQuantizationConfig(cls):
        return FourOverSixConfig()


class FourOverSixStatic6Test(FourOverSixBaseTest):
    @classmethod
    def getQuantizationConfig(cls):
        return FourOverSixConfig(scale_rule="static_6")


class FourOverSixKeepMasterWeightsTest(FourOverSixBaseTest):
    @classmethod
    def getQuantizationConfig(cls):
        return FourOverSixConfig(keep_master_weights=True)
