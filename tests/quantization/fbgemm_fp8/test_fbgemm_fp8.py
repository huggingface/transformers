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
from typing import Any

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, FbgemmFp8Config, OPTForCausalLM
from transformers.testing_utils import (
    backend_empty_cache,
    require_accelerate,
    require_deterministic_for_xpu,
    require_torch_accelerator,
    require_torch_multi_accelerator,
    slow,
    torch_device,
)
from transformers.utils import (
    is_fbgemm_gpu_available,
    is_torch_available,
    is_torch_xpu_available,
)


if is_torch_available():
    import torch


@require_torch_accelerator
class FbgemmFp8ConfigTest(unittest.TestCase):
    def test_to_dict(self):
        """
        Simple test that checks if one uses a config and converts it to a dict, the dict is the same as the config object
        """
        quantization_config = FbgemmFp8Config()
        config_to_dict = quantization_config.to_dict()

        for key in config_to_dict:
            self.assertEqual(getattr(quantization_config, key), config_to_dict[key])

    def test_from_dict(self):
        """
        Simple test that checks if one uses a dict and converts it to a config object, the config object is the same as the dict
        """
        dict = {"modules_to_not_convert": ["lm_head.weight"], "quant_method": "fbgemm_fp8"}
        quantization_config = FbgemmFp8Config.from_dict(dict)

        self.assertEqual(dict["modules_to_not_convert"], quantization_config.modules_to_not_convert)
        self.assertEqual(dict["quant_method"], quantization_config.quant_method)


@slow
@require_torch_accelerator
@unittest.skipIf(not is_torch_xpu_available() and not is_fbgemm_gpu_available(), "test requires fbgemm-gpu or xpu")
@require_accelerate
class FbgemmFp8Test(unittest.TestCase):
    model_name = "meta-llama/Meta-Llama-3-8B"

    input_text = "What are we having for dinner?"
    max_new_tokens = 9

    EXPECTED_OUTPUT = set[Any](
        [
            "What are we having for dinner?\nI'm having a steak and a salad",
            "What are we having for dinner? I don’t know. What are we having",
            "What are we having for dinner? I don’t know, what are you having",
        ]
    )

    device_map = "xpu" if is_torch_xpu_available() else "cuda"

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
        quantization_config = FbgemmFp8Config()
        cls.tokenizer = AutoTokenizer.from_pretrained(cls.model_name)
        cls.quantized_model = AutoModelForCausalLM.from_pretrained(
            cls.model_name, device_map=cls.device_map, quantization_config=quantization_config
        )

    def tearDown(self):
        gc.collect()
        backend_empty_cache(torch_device)
        gc.collect()

    def test_quantized_model_conversion(self):
        """
        Simple test that checks if the quantized model has been converted properly
        """

        from transformers.integrations import FbgemmFp8Linear, replace_with_fbgemm_fp8_linear

        model_id = "facebook/opt-350m"
        config = AutoConfig.from_pretrained(model_id, revision="cb32f77e905cccbca1d970436fb0f5e6b58ee3c5")
        quantization_config = FbgemmFp8Config()

        with torch.device("meta"):
            model = OPTForCausalLM(config)

        nb_linears = 0
        for module in model.modules():
            if isinstance(module, torch.nn.Linear):
                nb_linears += 1

        model = replace_with_fbgemm_fp8_linear(model, quantization_config=quantization_config)
        nb_fbgemm_linear = 0
        for module in model.modules():
            if isinstance(module, FbgemmFp8Linear):
                nb_fbgemm_linear += 1

        self.assertEqual(nb_linears, nb_fbgemm_linear)

        with torch.device("meta"):
            model = OPTForCausalLM(config)
        quantization_config = FbgemmFp8Config(modules_to_not_convert=["fc1"])
        model = replace_with_fbgemm_fp8_linear(
            model, modules_to_not_convert=["fc1"], quantization_config=quantization_config
        )
        nb_fbgemm_linear = 0
        for module in model.modules():
            if isinstance(module, FbgemmFp8Linear):
                nb_fbgemm_linear += 1

        self.assertEqual(nb_linears - 24, nb_fbgemm_linear)

    @require_deterministic_for_xpu
    def test_quantized_model(self):
        """
        Simple test that checks if the quantized model is working properly
        """
        input_ids = self.tokenizer(self.input_text, return_tensors="pt").to(torch_device)

        output = self.quantized_model.generate(**input_ids, max_new_tokens=self.max_new_tokens, do_sample=False)
        self.assertTrue(self.tokenizer.decode(output[0], skip_special_tokens=True) in self.EXPECTED_OUTPUT)

    @require_deterministic_for_xpu
    def test_save_pretrained(self):
        """
        Simple test that checks if the quantized model is working properly after being saved and loaded
        """
        with tempfile.TemporaryDirectory() as tmpdirname:
            self.quantized_model.save_pretrained(tmpdirname)

            model = AutoModelForCausalLM.from_pretrained(tmpdirname, device_map=self.device_map)

            input_ids = self.tokenizer(self.input_text, return_tensors="pt").to(torch_device)

            output = model.generate(**input_ids, max_new_tokens=self.max_new_tokens, do_sample=False)
            self.assertTrue(self.tokenizer.decode(output[0], skip_special_tokens=True) in self.EXPECTED_OUTPUT)

    def test_change_loading_attributes(self):
        """
        Simple test that checks if the quantized model is working properly after being saved and loaded
        """
        with tempfile.TemporaryDirectory() as tmpdirname:
            self.quantized_model.save_pretrained(tmpdirname)

            quantization_config = FbgemmFp8Config(activation_scale_ub=1000.0)

            model = AutoModelForCausalLM.from_pretrained(
                tmpdirname, device_map=self.device_map, quantization_config=quantization_config
            )

            self.assertEqual(model.model.layers[1].mlp.down_proj.input_scale_ub.item(), 1000.0)

            input_ids = self.tokenizer(self.input_text, return_tensors="pt").to(torch_device)

            output = model.generate(**input_ids, max_new_tokens=self.max_new_tokens, do_sample=False)
            self.assertTrue(self.tokenizer.decode(output[0], skip_special_tokens=True) in self.EXPECTED_OUTPUT)

    @require_torch_multi_accelerator
    @require_deterministic_for_xpu
    def test_quantized_model_multi_gpu(self):
        """
        Simple test that checks if the quantized model is working properly with multiple GPUs
        set CUDA_VISIBLE_DEVICES=0,1 if you have more than 2 GPUs
        """
        input_ids = self.tokenizer(self.input_text, return_tensors="pt").to(torch_device)
        quantization_config = FbgemmFp8Config()
        quantized_model = AutoModelForCausalLM.from_pretrained(
            self.model_name, device_map="auto", quantization_config=quantization_config
        )
        self.assertTrue(set(quantized_model.hf_device_map.values()) == {0, 1})

        output = quantized_model.generate(**input_ids, max_new_tokens=self.max_new_tokens, do_sample=False)
        self.assertTrue(self.tokenizer.decode(output[0], skip_special_tokens=True) in self.EXPECTED_OUTPUT)

    def test_quantized_model_offload(self):
        """
        Simple test that checks if the quantized model returns an error when loading with cpu/disk offloaded
        """
        quantization_config = FbgemmFp8Config()

        with self.assertRaisesRegex(
            ValueError, "You are attempting to load an FP8 model with a device_map that contains a CPU or disk device."
        ):
            AutoModelForCausalLM.from_pretrained(
                self.model_name, device_map=self.offload_device_map, quantization_config=quantization_config
            )

    @require_deterministic_for_xpu
    def test_save_pretrained_offload(self):
        """
        Simple test that checks if the saved quantized model is working properly cpu/disk offload
        """
        with tempfile.TemporaryDirectory() as tmpdirname:
            self.quantized_model.save_pretrained(tmpdirname)

            input_ids = self.tokenizer(self.input_text, return_tensors="pt").to(torch_device)

            quantized_model = AutoModelForCausalLM.from_pretrained(tmpdirname, device_map=self.offload_device_map)
            output = quantized_model.generate(**input_ids, max_new_tokens=self.max_new_tokens, do_sample=False)
            self.assertTrue(self.tokenizer.decode(output[0], skip_special_tokens=True) in self.EXPECTED_OUTPUT)

    @require_torch_multi_accelerator
    @require_deterministic_for_xpu
    def test_save_pretrained_multi_gpu(self):
        """
        Simple test that checks if the quantized model is working properly after being saved and loaded
        """
        with tempfile.TemporaryDirectory() as tmpdirname:
            self.quantized_model.save_pretrained(tmpdirname)

            model = AutoModelForCausalLM.from_pretrained(tmpdirname, device_map="auto")
            self.assertTrue(set(model.hf_device_map.values()) == {0, 1})

            input_ids = self.tokenizer(self.input_text, return_tensors="pt").to(torch_device)

            output = model.generate(**input_ids, max_new_tokens=self.max_new_tokens, do_sample=False)
            self.assertTrue(self.tokenizer.decode(output[0], skip_special_tokens=True) in self.EXPECTED_OUTPUT)
