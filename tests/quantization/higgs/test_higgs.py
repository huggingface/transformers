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
import os
import tempfile
import unittest

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, HiggsConfig, OPTForCausalLM
from transformers.testing_utils import (
    backend_empty_cache,
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

try:
    from safetensors import safe_open
    from safetensors.torch import save_file as save_file_pt

    _safetensors_available = True
except ImportError:
    _safetensors_available = False


@require_torch_gpu
class HiggsConfigTest(unittest.TestCase):
    def test_to_dict(self):
        """
        Simple test that checks if one uses a config and converts it to a dict, the dict is the same as the config object
        """
        quantization_config = HiggsConfig()
        config_to_dict = quantization_config.to_dict()

        for key in config_to_dict:
            self.assertEqual(getattr(quantization_config, key), config_to_dict[key])

    def test_from_dict(self):
        """
        Simple test that checks if one uses a dict and converts it to a config object, the config object is the same as the dict
        """
        dict = {"modules_to_not_convert": ["embed_tokens", "lm_head"], "quant_method": "higgs"}
        quantization_config = HiggsConfig.from_dict(dict)

        self.assertEqual(dict["modules_to_not_convert"], quantization_config.modules_to_not_convert)
        self.assertEqual(dict["quant_method"], quantization_config.quant_method)


@slow
@require_torch_gpu
@require_flute_hadamard
@require_accelerate
# @require_read_token
class HiggsTest(unittest.TestCase):
    model_name = "unsloth/Llama-3.2-1B"

    input_text = "Font test: A quick brown fox jumps over the"
    max_new_tokens = 2

    EXPECTED_OUTPUT = "Font test: A quick brown fox jumps over the lazy dog"

    device_map = "cuda"

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
        backend_empty_cache(torch_device)
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
        nb_higgs_linear = 0
        for module in model.modules():
            if isinstance(module, HiggsLinear):
                nb_higgs_linear += 1

        self.assertEqual(nb_linears - 1, nb_higgs_linear)

        with init_empty_weights():
            model = OPTForCausalLM(config)
        quantization_config = HiggsConfig(modules_to_not_convert=["fc1"])
        model, _ = replace_with_higgs_linear(model, quantization_config=quantization_config)
        nb_higgs_linear = 0
        for module in model.modules():
            if isinstance(module, HiggsLinear):
                nb_higgs_linear += 1

        self.assertEqual(nb_linears - 24, nb_higgs_linear)

    def test_quantized_model(self):
        """
        Simple test that checks if the quantized model is working properly
        """
        input_ids = self.tokenizer(self.input_text, return_tensors="pt").to(torch_device)

        output = self.quantized_model.generate(**input_ids, max_new_tokens=self.max_new_tokens)
        self.assertEqual(self.tokenizer.decode(output[0], skip_special_tokens=True), self.EXPECTED_OUTPUT)

    def test_check_quantized_param_as_tensor(self):
        r"""
        Test the `check_quantized_param` method with torch.Tensor objects.
        This verifies that the method correctly handles regular torch tensors with dtype attribute.
        """
        from unittest.mock import Mock

        from transformers.integrations.higgs import HiggsLinear
        from transformers.quantizers.quantizer_higgs import HiggsHfQuantizer

        # Create quantizer directly
        config = HiggsConfig()
        quantizer = HiggsHfQuantizer(config)

        # Create a mock model with a mock module that has HiggsLinear
        mock_model = Mock()
        mock_module = Mock(spec=HiggsLinear)

        # Mock the get_module_from_name function
        def mock_get_module_from_name(model, param_name):
            return mock_module, "weight"

        # Patch the function
        import transformers.quantizers.quantizer_higgs as quantizer_module

        original_get_module_from_name = quantizer_module.get_module_from_name
        quantizer_module.get_module_from_name = mock_get_module_from_name

        try:
            param_name = "test.weight"
            state_dict = {}

            # Test with unquantized tensor (should return True for quantization)
            float16_tensor = torch.randn(10, 5, dtype=torch.float16)
            result = quantizer.check_quantized_param(mock_model, float16_tensor, param_name, state_dict)
            self.assertTrue(result)  # Unquantized weights should be quantized

        finally:
            # Restore original function
            quantizer_module.get_module_from_name = original_get_module_from_name

    @unittest.skipIf(not _safetensors_available, "safetensors not available")
    def test_check_quantized_param_as_slice(self):
        r"""
        Test the `check_quantized_param` method with PySafeSlice objects.
        This verifies that the method correctly handles safetensors slices with get_dtype() method.
        """
        from unittest.mock import Mock

        from transformers.integrations.higgs import HiggsLinear
        from transformers.quantizers.quantizer_higgs import HiggsHfQuantizer

        # Create quantizer directly
        config = HiggsConfig()
        quantizer = HiggsHfQuantizer(config)

        # Create a mock model with a mock module that has HiggsLinear
        mock_model = Mock()
        mock_module = Mock(spec=HiggsLinear)

        # Mock the get_module_from_name function
        def mock_get_module_from_name(model, param_name):
            return mock_module, "weight"

        # Patch the function
        import transformers.quantizers.quantizer_higgs as quantizer_module

        original_get_module_from_name = quantizer_module.get_module_from_name
        quantizer_module.get_module_from_name = mock_get_module_from_name

        try:
            param_name = "test.weight"
            state_dict = {}

            # Create a safetensors file with float16 data (unquantized)
            A = torch.randn(10, 5, dtype=torch.float16)
            tensors = {"test_tensor": A}
            safetensors_path = "./slice.safetensors"

            try:
                save_file_pt(tensors, safetensors_path)

                # Load and test with safetensors slice (should return True for quantization)
                with safe_open(safetensors_path, framework="pt", device="cpu") as f:
                    slice_ = f.get_slice("test_tensor")

                    # Verify the slice has get_dtype method and returns "F16" for float16
                    self.assertTrue(hasattr(slice_, "get_dtype"))
                    self.assertEqual(slice_.get_dtype(), "F16")

                    # Test the check_quantized_param method with the slice
                    result = quantizer.check_quantized_param(mock_model, slice_, param_name, state_dict)
                    self.assertTrue(result)  # Unquantized weights should be quantized

            finally:
                # Clean up safetensors file
                if os.path.exists(safetensors_path):
                    os.remove(safetensors_path)

        finally:
            # Restore original function
            quantizer_module.get_module_from_name = original_get_module_from_name

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
        set CUDA_VISIBLE_DEVICES=0,1 if you have more than 2 GPUs
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

    @unittest.skip("This will almost surely OOM. Enable when switched to a smaller model")
    def test_dequantize(self):
        """
        Test the ability to dequantize a model
        """
        self.quantized_model.dequantize()

        input_ids = self.tokenizer(self.input_text, return_tensors="pt").to(torch_device)

        output = self.quantized_model.generate(**input_ids, max_new_tokens=self.max_new_tokens)
        self.assertEqual(self.tokenizer.decode(output[0], skip_special_tokens=True), self.EXPECTED_OUTPUT)
