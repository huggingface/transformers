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

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, FbgemmFp8Config, OPTForCausalLM
from transformers.testing_utils import (
    backend_empty_cache,
    require_accelerate,
    require_fbgemm_gpu,
    require_read_token,
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
@require_torch_gpu
@require_fbgemm_gpu
@require_accelerate
@require_read_token
class FbgemmFp8Test(unittest.TestCase):
    model_name = "meta-llama/Meta-Llama-3-8B"

    input_text = "What are we having for dinner?"
    max_new_tokens = 9

    EXPECTED_OUTPUT = "What are we having for dinner?\nI'm having a steak and a salad"

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

        with init_empty_weights():
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

        self.assertEqual(nb_linears - 1, nb_fbgemm_linear)

        with init_empty_weights():
            model = OPTForCausalLM(config)
        quantization_config = FbgemmFp8Config(modules_to_not_convert=["fc1"])
        model = replace_with_fbgemm_fp8_linear(model, quantization_config=quantization_config)
        nb_fbgemm_linear = 0
        for module in model.modules():
            if isinstance(module, FbgemmFp8Linear):
                nb_fbgemm_linear += 1

        self.assertEqual(nb_linears - 25, nb_fbgemm_linear)

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

            output = model.generate(**input_ids, max_new_tokens=self.max_new_tokens)
            self.assertEqual(self.tokenizer.decode(output[0], skip_special_tokens=True), self.EXPECTED_OUTPUT)

    @require_torch_multi_gpu
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

        output = quantized_model.generate(**input_ids, max_new_tokens=self.max_new_tokens)
        self.assertEqual(self.tokenizer.decode(output[0], skip_special_tokens=True), self.EXPECTED_OUTPUT)

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

    def test_save_pretrained_offload(self):
        """
        Simple test that checks if the saved quantized model is working properly cpu/disk offload
        """
        with tempfile.TemporaryDirectory() as tmpdirname:
            self.quantized_model.save_pretrained(tmpdirname)

            input_ids = self.tokenizer(self.input_text, return_tensors="pt").to(torch_device)

            quantized_model = AutoModelForCausalLM.from_pretrained(tmpdirname, device_map=self.offload_device_map)
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


@require_torch_gpu
@require_accelerate
@require_fbgemm_gpu
class FbgemmFp8LinearTest(unittest.TestCase):
    def test_linear_preserves_shape(self):
        """
        Test that FbgemmFp8Linear preserves shape when in_features == out_features.
        """
        from transformers.integrations import FbgemmFp8Linear

        with init_empty_weights(include_buffers=True):
            linear = FbgemmFp8Linear(1024, 1024, True)
            x = torch.rand((17, 23, 1024))

        x_ = linear(x)
        self.assertEqual(x_.shape, x.shape)

    def test_linear_with_diff_feature_size_preserves_shape(self):
        """
        Test that FbgemmFp8Linear generates the correct shape when in_features != out_features.
        """
        from transformers.integrations import FbgemmFp8Linear

        with init_empty_weights(include_buffers=True):
            linear = FbgemmFp8Linear(1024, 2048, True)
            x = torch.rand((17, 23, 1024))

        x_ = linear(x)
        self.assertEqual(x_.shape, (17, 23, 2048))

    def test_check_quantized_param_as_tensor(self):
        r"""
        Test the `check_quantized_param` method with torch.Tensor objects.
        This verifies that the method correctly handles regular torch tensors with dtype attribute.
        """
        from unittest.mock import Mock

        from transformers.integrations.fbgemm_fp8 import FbgemmFp8Linear
        from transformers.quantizers.quantizer_fbgemm_fp8 import FbgemmFp8HfQuantizer

        # Create quantizer directly
        config = FbgemmFp8Config()
        quantizer = FbgemmFp8HfQuantizer(config)
        quantizer.pre_quantized = True

        # Create a mock model with a mock module that has FbgemmFp8Linear
        mock_model = Mock()
        mock_module = Mock(spec=FbgemmFp8Linear)

        # Mock the get_module_from_name function
        def mock_get_module_from_name(model, param_name):
            return mock_module, "weight"

        # Patch the function
        import transformers.quantizers.quantizer_fbgemm_fp8 as quantizer_module

        original_get_module_from_name = quantizer_module.get_module_from_name
        quantizer_module.get_module_from_name = mock_get_module_from_name

        try:
            param_name = "test.weight"
            state_dict = {}

            # Test with correct fp8 tensor (should return False for pre-quantized)
            fp8_tensor = torch.randn(10, 5).to(torch.float8_e4m3fn)
            result = quantizer.check_quantized_param(mock_model, fp8_tensor, param_name, state_dict)
            self.assertFalse(result)  # Pre-quantized weights return False

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

        from transformers.integrations.fbgemm_fp8 import FbgemmFp8Linear
        from transformers.quantizers.quantizer_fbgemm_fp8 import FbgemmFp8HfQuantizer

        # Create quantizer directly
        config = FbgemmFp8Config()
        quantizer = FbgemmFp8HfQuantizer(config)
        quantizer.pre_quantized = True

        # Create a mock model with a mock module that has FbgemmFp8Linear
        mock_model = Mock()
        mock_module = Mock(spec=FbgemmFp8Linear)

        # Mock the get_module_from_name function
        def mock_get_module_from_name(model, param_name):
            return mock_module, "weight"

        # Patch the function
        import transformers.quantizers.quantizer_fbgemm_fp8 as quantizer_module

        original_get_module_from_name = quantizer_module.get_module_from_name
        quantizer_module.get_module_from_name = mock_get_module_from_name

        try:
            param_name = "test.weight"
            state_dict = {}

            # Create a safetensors file with fp8 data
            A = torch.randn(10, 5).to(torch.float8_e4m3fn)
            tensors = {"test_tensor": A}
            safetensors_path = "./slice.safetensors"

            try:
                save_file_pt(tensors, safetensors_path)

                # Load and test with safetensors slice (should return False for pre-quantized)
                with safe_open(safetensors_path, framework="pt", device="cpu") as f:
                    slice_ = f.get_slice("test_tensor")

                    # Verify the slice has get_dtype method and returns "F8_E4M3" for fp8
                    self.assertTrue(hasattr(slice_, "get_dtype"))
                    self.assertEqual(slice_.get_dtype(), "F8_E4M3")

                    # Test the check_quantized_param method with the slice
                    result = quantizer.check_quantized_param(mock_model, slice_, param_name, state_dict)
                    self.assertFalse(result)  # Pre-quantized weights return False

            finally:
                # Clean up safetensors file
                if os.path.exists(safetensors_path):
                    os.remove(safetensors_path)

        finally:
            # Restore original function
            quantizer_module.get_module_from_name = original_get_module_from_name
