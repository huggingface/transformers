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

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, HiggsConfig, OPTForCausalLM
from transformers.testing_utils import (
    require_accelerate,
    require_flute_hadamard,
    require_torch_gpu,
    require_torch_multi_gpu,
    slow,
    torch_device,
)
from transformers.utils import is_accelerate_available, is_torch_available


if is_torch_available():
    import torch

if is_accelerate_available():
    from accelerate import init_empty_weights


# @require_torch_gpu
# class HiggsConfigTest(unittest.TestCase):
#     def test_to_dict(self):
#         """
#         Simple test that checks if one uses a config and converts it to a dict, the dict is the same as the config object
#         """
#         quantization_config = HiggsConfig()
#         config_to_dict = quantization_config.to_dict()

#         for key in config_to_dict:
#             self.assertEqual(getattr(quantization_config, key), config_to_dict[key])

#     def test_from_dict(self):
#         """
#         Simple test that checks if one uses a dict and converts it to a config object, the config object is the same as the dict
#         """
#         dict = {"linear_weights_not_to_quantize": ["embed_tokens.weight", "lm_head.weight"], "quant_method": "higgs"}
#         quantization_config = HiggsConfig.from_dict(dict)

#         self.assertEqual(dict["linear_weights_not_to_quantize"], quantization_config.linear_weights_not_to_quantize)
#         self.assertEqual(dict["quant_method"], quantization_config.quant_method)


@slow
@require_torch_gpu
@require_flute_hadamard
@require_accelerate
# @require_read_token
class HiggsTest(unittest.TestCase):
    model_name = "meta-llama/Meta-Llama-3.1-8B"

    input_text = "A quick brown fox jumps over the"
    max_new_tokens = 2

    EXPECTED_OUTPUT = "A quick brown fox jumps over the lazy dog"

    device_map = "cuda"

    offload_device_map = {
        "model.embed_tokens": 0,
        "model.layers.0": 0,
        "model.layers.1": 0,
        "model.layers.2": 0,
        "model.layers.3": 0,
        "model.layers.4": 0,
        "model.layers.5": 0,
        "model.layers.6": 0,
        "model.layers.7": 0,
        "model.layers.8": 0,
        "model.layers.9": 0,
        "model.layers.10": 0,
        "model.layers.11": 0,
        "model.layers.12": 0,
        "model.layers.13": 0,
        "model.layers.14": 0,
        "model.layers.15": 0,
        "model.layers.16": "cpu",
        "model.layers.17": "cpu",
        "model.layers.18": "cpu",
        "model.layers.19": "cpu",
        "model.layers.20": "disk",
        "model.layers.21": "disk",
        "model.layers.22": "disk",
        "model.layers.23": "disk",
        "model.layers.24": "disk",
        "model.layers.25": "disk",
        "model.layers.26": "disk",
        "model.layers.27": "disk",
        "model.layers.28": "disk",
        "model.layers.29": "disk",
        "model.layers.30": "disk",
        "model.layers.31": "disk",
        "model.norm": "disk",
        "lm_head": "disk",
    }

    # called only once for all test in this class
    @classmethod
    def setUpClass(cls):
        """
        Setup quantized model
        """
        quantization_config = HiggsConfig()
        cls.tokenizer = AutoTokenizer.from_pretrained(cls.model_name)
        cls.quantized_model = AutoModelForCausalLM.from_pretrained(
            cls.model_name, device_map=cls.device_map, quantization_config=quantization_config
        )

    def tearDown(self):
        gc.collect()
        torch.cuda.empty_cache()
        gc.collect()

    def test_quantized_model_conversion(self):
        """
        Simple test that checks if the quantized model has been converted properly
        """

        from transformers.integrations import HiggsLinear, replace_with_higgs_linear

        model_id = "facebook/opt-350m"
        config = AutoConfig.from_pretrained(model_id, revision="cb32f77e905cccbca1d970436fb0f5e6b58ee3c5")
        quantization_config = HiggsConfig()

        with init_empty_weights():
            model = OPTForCausalLM(config)

        nb_linears = 0
        for module in model.modules():
            if isinstance(module, torch.nn.Linear):
                nb_linears += 1

        model, _ = replace_with_higgs_linear(model, quantization_config=quantization_config)
        nb_fbgemm_linear = 0
        for module in model.modules():
            if isinstance(module, HiggsLinear):
                nb_fbgemm_linear += 1

        self.assertEqual(nb_linears - 1, nb_fbgemm_linear)

        with init_empty_weights():
            model = OPTForCausalLM(config)
        quantization_config = HiggsConfig(linear_weights_not_to_quantize=["fc1.weight"])
        model, _ = replace_with_higgs_linear(model, quantization_config=quantization_config)
        nb_fbgemm_linear = 0
        for module in model.modules():
            if isinstance(module, HiggsLinear):
                nb_fbgemm_linear += 1

        self.assertEqual(nb_linears - 24, nb_fbgemm_linear)

    def test_quantized_model(self):
        """
        Simple test that checks if the quantized model is working properly
        """
        input_ids = self.tokenizer(self.input_text, return_tensors="pt").to(torch_device)

        output = self.quantized_model.generate(**input_ids, max_new_tokens=self.max_new_tokens)
        self.assertEqual(self.tokenizer.decode(output[0], skip_special_tokens=True), self.EXPECTED_OUTPUT)

    def test_save_pretrained(self):
        """
        Simple test that checks if the quantized model is working properly after being saved and loaded
        """
        with tempfile.TemporaryDirectory() as tmpdirname:
            self.quantized_model.save_pretrained(tmpdirname)

            model = AutoModelForCausalLM.from_pretrained(tmpdirname, device_map=self.device_map)

            input_ids = self.tokenizer(self.input_text, return_tensors="pt").to(torch_device)

            output = model.generate(**input_ids, max_new_tokens=self.max_new_tokens)
            self.assertEqual(self.tokenizer.decode(output[0], skip_special_tokens=True), self.EXPECTED_OUTPUT)

    @require_torch_multi_gpu
    def test_quantized_model_multi_gpu(self):
        """
        Simple test that checks if the quantized model is working properly with multiple GPUs
        set CUDA_VISIBLE_DEVICES=0,1 if you have more than 2 GPUS
        """
        input_ids = self.tokenizer(self.input_text, return_tensors="pt").to(torch_device)
        quantization_config = HiggsConfig()
        quantized_model = AutoModelForCausalLM.from_pretrained(
            self.model_name, device_map="auto", quantization_config=quantization_config
        )
        self.assertTrue(set(quantized_model.hf_device_map.values()) == {0, 1})

        output = quantized_model.generate(**input_ids, max_new_tokens=self.max_new_tokens)
        self.assertEqual(self.tokenizer.decode(output[0], skip_special_tokens=True), self.EXPECTED_OUTPUT)

    @require_torch_multi_gpu
    def test_save_pretrained_multi_gpu(self):
        """
        Simple test that checks if the quantized model is working properly after being saved and loaded
        """
        with tempfile.TemporaryDirectory() as tmpdirname:
            self.quantized_model.save_pretrained(tmpdirname)

            model = AutoModelForCausalLM.from_pretrained(tmpdirname, device_map="auto")
            self.assertTrue(set(model.hf_device_map.values()) == {0, 1})

            input_ids = self.tokenizer(self.input_text, return_tensors="pt").to(torch_device)

            output = model.generate(**input_ids, max_new_tokens=self.max_new_tokens)
            self.assertEqual(self.tokenizer.decode(output[0], skip_special_tokens=True), self.EXPECTED_OUTPUT)


# @require_torch_gpu
# @require_accelerate
# @require_flute_hadamard
# class HiggsLinearTest(unittest.TestCase):
#     def test_linear_preserves_shape(self):
#         """
#         Test that HiggsLinear preserves shape when in_features == out_features.
#         """
#         from transformers.integrations import HiggsLinear

#         with init_empty_weights(include_buffers=True):
#             linear = HiggsLinear(1024, 1024, num_bits=4, num_sms_packed=128, bias=True)
#             x = torch.rand((17, 23, 1024))

#         # x_ = linear(x)
#         # self.assertEqual(x_.shape, x.shape) # TODO: Fix this

#     def test_linear_with_diff_feature_size_preserves_shape(self):
#         """
#         Test that HiggsLinear generates the correct shape when in_features != out_features.
#         """
#         from transformers.integrations import HiggsLinear

#         with init_empty_weights(include_buffers=True):
#             linear = HiggsLinear(1024, 2048, num_bits=4, num_sms_packed=128, bias=True)
#             x = torch.rand((17, 23, 1024))

#         # x_ = linear(x)
#         # self.assertEqual(x_.shape, (17, 23, 2048)) # TODO: Fix this
