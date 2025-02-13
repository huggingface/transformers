# coding=utf-8
# Copyright 2025 The HuggingFace Team. All rights reserved.
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

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, FineGrainedFP8Config, OPTForCausalLM
from transformers.testing_utils import (
    require_accelerate,
    require_torch_gpu,
    require_torch_multi_gpu,
    slow,
)
from transformers.utils import is_accelerate_available, is_torch_available


if is_torch_available():
    import torch

if is_accelerate_available():
    from accelerate import init_empty_weights


@require_torch_gpu
class FineGrainedFP8ConfigTest(unittest.TestCase):
    def test_to_dict(self):
        """
        Simple test that checks if one uses a config and converts it to a dict, the dict is the same as the config object
        """
        quantization_config = FineGrainedFP8Config()
        config_to_dict = quantization_config.to_dict()

        for key in config_to_dict:
            self.assertEqual(getattr(quantization_config, key), config_to_dict[key])

    def test_from_dict(self):
        """
        Simple test that checks if one uses a dict and converts it to a config object, the config object is the same as the dict
        """
        dict = {"modules_to_not_convert": ["lm_head.weight"], "quant_method": "fp8"}
        quantization_config = FineGrainedFP8Config.from_dict(dict)

        self.assertEqual(dict["modules_to_not_convert"], quantization_config.modules_to_not_convert)
        self.assertEqual(dict["quant_method"], quantization_config.quant_method)


@slow
@require_accelerate
@require_torch_gpu
class FP8QuantizerTest(unittest.TestCase):
    model_name = "meta-llama/Llama-3.2-1B"
    input_text = "Once upon a time"
    max_new_tokens = 10
    EXPECTED_OUTPUT = "Once upon a time, there was a man who was very rich."
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
        "model.layers.7": "cpu",
        "model.layers.8": "cpu",
        "model.layers.9": "cpu",
        "model.layers.10": "cpu",
        "model.layers.11": "cpu",
        "model.layers.12": "cpu",
        "model.layers.13": "cpu",
        "model.layers.14": "cpu",
        "model.layers.15": "cpu",
        "model.rotary_emb": "disk",
        "model.norm": "disk",
        "lm_head": 0,
    }

    @classmethod
    def setUpClass(cls):
        """
        Setup quantized model
        """
        cls.quantization_config = FineGrainedFP8Config()
        cls.tokenizer = AutoTokenizer.from_pretrained(cls.model_name)
        cls.quantized_model = AutoModelForCausalLM.from_pretrained(
            cls.model_name, device_map=cls.device_map, quantization_config=cls.quantization_config
        )

    def tearDown(self):
        gc.collect()
        torch.cuda.empty_cache()
        gc.collect()

    def test_quantized_model_conversion(self):
        """
        Simple test that checks if the quantized model has been converted properly
        """

        from transformers.integrations import FP8Linear, replace_with_fp8_linear

        model_id = "facebook/opt-350m"
        config = AutoConfig.from_pretrained(model_id, revision="cb32f77e905cccbca1d970436fb0f5e6b58ee3c5")
        quantization_config = FineGrainedFP8Config()

        with init_empty_weights():
            model = OPTForCausalLM(config)

        nb_linears = 0
        for module in model.modules():
            if isinstance(module, torch.nn.Linear):
                nb_linears += 1

        model = replace_with_fp8_linear(model, quantization_config=quantization_config)
        nb_fp8_linear = 0
        for module in model.modules():
            if isinstance(module, FP8Linear):
                nb_fp8_linear += 1

        self.assertEqual(nb_linears - 1, nb_fp8_linear)

        with init_empty_weights():
            model = OPTForCausalLM(config)
        quantization_config = FineGrainedFP8Config(modules_to_not_convert=["fc1"])
        model = replace_with_fp8_linear(model, quantization_config=quantization_config)
        nb_fp8_linear = 0
        for module in model.modules():
            if isinstance(module, FP8Linear):
                nb_fp8_linear += 1

        self.assertEqual(nb_linears - 25, nb_fp8_linear)

    def test_quantized_model(self):
        """
        Simple test that checks if the quantized model is working properly
        """
        input_ids = self.tokenizer(self.input_text, return_tensors="pt").to(self.device_map)

        output = self.quantized_model.generate(**input_ids, max_new_tokens=self.max_new_tokens, do_sample=False)
        self.assertEqual(self.tokenizer.decode(output[0], skip_special_tokens=True), self.EXPECTED_OUTPUT)

    def test_save_pretrained(self):
        """
        Simple test that checks if the quantized model is working properly after being saved and loaded
        """
        with tempfile.TemporaryDirectory() as tmpdirname:
            self.quantized_model.save_pretrained(tmpdirname)

            model = AutoModelForCausalLM.from_pretrained(tmpdirname, device_map=self.device_map)

            input_ids = self.tokenizer(self.input_text, return_tensors="pt").to(self.device_map)

            output = model.generate(**input_ids, max_new_tokens=self.max_new_tokens, do_sample=False)
            self.assertEqual(self.tokenizer.decode(output[0], skip_special_tokens=True), self.EXPECTED_OUTPUT)

    def test_weight_and_weight_scale_inv(self):
        """
        Simple test that checks if the weight and weight_scale_inv are working properly
        """
        weight = self.quantized_model.model.layers[0].self_attn.q_proj.weight
        weight_scale_inv = self.quantized_model.model.layers[0].self_attn.q_proj.weight_scale_inv
        self.assertEqual(weight.dtype, torch.float8_e4m3fn)
        self.assertEqual(weight_scale_inv.dtype, torch.float32)
        self.assertEqual(weight.shape, (weight_scale_inv.shape[0] * 128, weight_scale_inv.shape[1] * 128))

    def test_block_size(self):
        """
        Simple test that checks if the block size is working properly
        """
        self.assertEqual(self.quantized_model.config.quantization_config.weight_block_size, (128, 128))
        quantization_config = FineGrainedFP8Config(weight_block_size=(32, 32))
        quantized_model = AutoModelForCausalLM.from_pretrained(
            self.model_name, device_map=self.device_map, quantization_config=quantization_config
        )
        self.assertEqual(quantized_model.config.quantization_config.weight_block_size, (32, 32))

    @require_torch_multi_gpu
    def test_quantized_model_multi_gpu(self):
        """
        Simple test that checks if the quantized model is working properly with multiple GPUs
        set CUDA_VISIBLE_DEVICES=0,1 if you have more than 2 GPUS
        """
        input_ids = self.tokenizer(self.input_text, return_tensors="pt").to(self.device_map)
        quantization_config = FineGrainedFP8Config()
        quantized_model = AutoModelForCausalLM.from_pretrained(
            self.model_name, device_map="auto", quantization_config=quantization_config
        )
        self.assertTrue(set(quantized_model.hf_device_map.values()) == {0, 1})

        output = quantized_model.generate(**input_ids, max_new_tokens=self.max_new_tokens, do_sample=False)
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

            input_ids = self.tokenizer(self.input_text, return_tensors="pt").to(self.device_map)

            output = model.generate(**input_ids, max_new_tokens=self.max_new_tokens, do_sample=False)
            self.assertEqual(self.tokenizer.decode(output[0], skip_special_tokens=True), self.EXPECTED_OUTPUT)

    def test_quantized_model_offload(self):
        """
        Simple test that checks if the quantized model returns an error when loading with cpu/disk offloaded
        """
        with self.assertRaisesRegex(
            ValueError, "You are attempting to load an FP8 model with a device_map that contains a cpu/disk device."
        ):
            AutoModelForCausalLM.from_pretrained(
                self.model_name, device_map=self.offload_device_map, quantization_config=self.quantization_config
            )

    def test_save_pretrained_offload(self):
        """
        Simple test that checks if the saved quantized model is working properly cpu/disk offload
        """
        with tempfile.TemporaryDirectory() as tmpdirname:
            self.quantized_model.save_pretrained(tmpdirname)

            input_ids = self.tokenizer(self.input_text, return_tensors="pt").to(self.device_map)

            quantized_model = AutoModelForCausalLM.from_pretrained(tmpdirname, device_map=self.offload_device_map)
            output = quantized_model.generate(**input_ids, max_new_tokens=self.max_new_tokens, do_sample=False)
            self.assertEqual(self.tokenizer.decode(output[0], skip_special_tokens=True), self.EXPECTED_OUTPUT)


@require_torch_gpu
class FP8LinearTest(unittest.TestCase):
    device = "cuda"

    def test_linear_preserves_shape(self):
        """
        Test that FP8Linear preserves shape when in_features == out_features.
        """
        from transformers.integrations import FP8Linear

        linear = FP8Linear(256, 256, block_size=(128, 128), device=self.device)
        x = torch.rand((1, 5, 256)).to(self.device)

        x_ = linear(x)
        self.assertEqual(x_.shape, x.shape)

    def test_linear_with_diff_feature_size_preserves_shape(self):
        """
        Test that FP8Linear generates the correct shape when in_features != out_features.
        """
        from transformers.integrations import FP8Linear

        linear = FP8Linear(128, 256, block_size=(128, 128), device=self.device)
        x = torch.rand((1, 5, 128)).to(self.device)

        x_ = linear(x)
        self.assertEqual(x_.shape, (1, 5, 256))
