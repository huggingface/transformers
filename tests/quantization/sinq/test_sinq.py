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
from unittest.mock import patch

from transformers import AutoModelForCausalLM, AutoTokenizer, SinqConfig
from transformers.testing_utils import (
    backend_empty_cache,
    require_torch_gpu,
    slow,
    torch_device,
)
from transformers.utils import is_torch_available


if is_torch_available():
    import torch


class SinqConfigTest(unittest.TestCase):
    """Test the SinqConfig class."""

    def test_default_config(self):
        """Test default configuration values."""
        config = SinqConfig()
        self.assertEqual(config.nbits, 4)
        self.assertEqual(config.group_size, 64)
        self.assertEqual(config.tiling_mode, "1D")
        self.assertEqual(config.method, "sinq")

    def test_custom_config(self):
        """Test custom configuration values."""
        config = SinqConfig(
            nbits=8,
            group_size=128,
            tiling_mode="2D",
            method="sinq",
        )
        self.assertEqual(config.nbits, 8)
        self.assertEqual(config.group_size, 128)
        self.assertEqual(config.tiling_mode, "2D")
        self.assertEqual(config.method, "sinq")

    def test_modules_to_not_convert(self):
        """Test modules_to_not_convert configuration."""
        modules = ["layer1", "layer2.weight"]
        config = SinqConfig(modules_to_not_convert=modules)
        self.assertEqual(config.modules_to_not_convert, modules)

    def test_to_dict(self):
        """Test that config converts to dict correctly."""
        quantization_config = SinqConfig()
        config_to_dict = quantization_config.to_dict()

        for key in config_to_dict:
            self.assertEqual(getattr(quantization_config, key), config_to_dict[key])

    def test_from_dict(self):
        """Test that config can be created from dict."""
        config_dict = {
            "nbits": 8,
            "group_size": 128,
            "method": "sinq",
        }
        config = SinqConfig.from_dict(config_dict)

        self.assertEqual(config.nbits, 8)
        self.assertEqual(config.group_size, 128)
        self.assertEqual(config.method, "sinq")

    def test_method_validation(self):
        """Test that invalid method raises error."""
        with self.assertRaises(ValueError):
            SinqConfig(method="invalid_method")


@slow
@require_torch_gpu
class SinqTest(unittest.TestCase):
    """Integration tests for SINQ quantization."""

    model_name = "Qwen/Qwen3-0.6B"
    input_text = "What is the capital of France?"
    max_new_tokens = 10
    device_map = torch_device

    EXPECTED_OUTPUTS = {
        "What is the capital of France? Paris.",
        "What is the capital of France? The capital of France is Paris.",
        "What is the capital of France? The capital of France is Paris. The statement is",
        "What is the capital of France? Paris is the capital and most populous city of France.",
    }

    @classmethod
    def setUpClass(cls):
        """Setup quantized model and tokenizer once for all tests."""
        cls.quantization_config = SinqConfig(
            nbits=4,
            group_size=64,
            method="sinq",
            modules_to_not_convert=["lm_head"],
        )
        cls.tokenizer = AutoTokenizer.from_pretrained(cls.model_name)
        cls.quantized_model = AutoModelForCausalLM.from_pretrained(
            cls.model_name,
            torch_dtype=torch.bfloat16,
            quantization_config=cls.quantization_config,
        )

    def tearDown(self):
        gc.collect()
        backend_empty_cache(torch_device)
        gc.collect()

    def test_quantizer_validation_no_cuda(self):
        """Test that quantizer logs warning when CUDA is not available."""
        from transformers.quantizers.quantizer_sinq import SinqHfQuantizer

        config = SinqConfig()
        quantizer = SinqHfQuantizer(quantization_config=config)

        with patch("torch.cuda.is_available", return_value=False):
            with self.assertLogs("transformers", level="WARNING") as cm:
                quantizer.validate_environment()
            self.assertTrue(any("No CUDA is available" in msg for msg in cm.output))

    def test_asinq_not_supported(self):
        """Test that asinq method raises error for non-pre-quantized models."""
        from transformers.quantizers.quantizer_sinq import SinqHfQuantizer

        config = SinqConfig(method="asinq")
        quantizer = SinqHfQuantizer(quantization_config=config)
        quantizer.pre_quantized = False

        with self.assertRaises(ValueError):
            quantizer.validate_environment()

    def test_quantized_model_conversion(self):
        """Test that Linear modules are converted to SINQLinear."""
        from sinq.sinqlinear_hf import SINQLinear

        nb_sinq_linear = 0
        for module in self.quantized_model.modules():
            if isinstance(module, SINQLinear):
                nb_sinq_linear += 1

        self.assertGreater(nb_sinq_linear, 0)
        self.assertNotIsInstance(self.quantized_model.lm_head, SINQLinear)

    def test_quantized_model(self):
        """Test that quantized model can generate text."""
        input_ids = self.tokenizer(self.input_text, return_tensors="pt").to(self.device_map)

        output = self.quantized_model.generate(**input_ids, max_new_tokens=self.max_new_tokens, do_sample=False)
        decoded = self.tokenizer.decode(output[0], skip_special_tokens=True)

        self.assertIsNotNone(decoded)
        self.assertGreater(len(decoded), len(self.input_text))
        self.assertIn(decoded, self.EXPECTED_OUTPUTS)

    def test_save_pretrained(self):
        """Test that quantized model can be saved and loaded."""
        with tempfile.TemporaryDirectory() as tmpdirname:
            self.quantized_model.save_pretrained(tmpdirname)

            loaded_model = AutoModelForCausalLM.from_pretrained(
                tmpdirname,
                device_map=self.device_map,
            )

            input_ids = self.tokenizer(self.input_text, return_tensors="pt").to(self.device_map)
            output = loaded_model.generate(**input_ids, max_new_tokens=self.max_new_tokens, do_sample=False)
            decoded = self.tokenizer.decode(output[0], skip_special_tokens=True)

            self.assertIsNotNone(decoded)
            self.assertGreater(len(decoded), len(self.input_text))

            del loaded_model
