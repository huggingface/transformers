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
import unittest
from unittest.mock import Mock, patch

from transformers import SinqConfig
from transformers.testing_utils import (
    backend_empty_cache,
    require_torch_gpu,
    slow,
    torch_device,
)
from transformers.utils import is_torch_available


if is_torch_available():
    import torch
    import torch.nn as nn


def _empty_accelerator_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _get_simple_model():
    """Create a simple model with Linear layers for testing."""

    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer1 = nn.Linear(128, 256)
            self.layer2 = nn.Linear(256, 512)
            self.layer3 = nn.Linear(512, 128)
            self.config = Mock()
            self.config._name_or_path = "test-model"
            self.config.model_type = "gpt2"

        def forward(self, x):
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            return x

    return SimpleModel()


class SinqConfigTest(unittest.TestCase):
    """Test the SinqConfig class"""

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

    def test_method_validation(self):
        """Test that invalid method raises error."""
        with self.assertRaises(ValueError):
            SinqConfig(method="invalid_method")

    def test_modules_to_not_convert(self):
        """Test modules_to_not_convert configuration."""
        modules = ["layer1", "layer2.weight"]
        config = SinqConfig(modules_to_not_convert=modules)
        self.assertEqual(config.modules_to_not_convert, modules)

    def test_to_dict(self):
        """Test configuration serialization to dict."""
        config = SinqConfig(nbits=8, method="sinq")
        config_dict = config.to_dict()

        self.assertEqual(config_dict["nbits"], 8)
        self.assertEqual(config_dict["method"], "sinq")
        self.assertEqual(config_dict["group_size"], 64)
        self.assertIn("quant_method", config_dict)

    def test_from_dict(self):
        """Test configuration deserialization from dict."""
        config_dict = {
            "nbits": 8,
            "group_size": 128,
            "method": "sinq",
        }
        config = SinqConfig.from_dict(config_dict)

        self.assertEqual(config.nbits, 8)
        self.assertEqual(config.group_size, 128)
        self.assertEqual(config.method, "sinq")


@require_torch_gpu
class SinqQuantizerTest(unittest.TestCase):
    """Test the SinqHfQuantizer class"""

    def setUp(self):
        gc.collect()
        _empty_accelerator_cache()

    def tearDown(self):
        gc.collect()
        _empty_accelerator_cache()

    def test_quantizer_init(self):
        """Test basic quantizer initialization."""
        from transformers.quantizers.quantizer_sinq import SinqHfQuantizer

        config = SinqConfig()
        quantizer = SinqHfQuantizer(quantization_config=config)

        self.assertEqual(quantizer.quantization_config, config)
        self.assertFalse(quantizer.requires_calibration)
        self.assertTrue(quantizer.requires_parameters_quantization)

    def test_validate_environment_cuda(self):
        """Test environment validation with CUDA."""
        from transformers.quantizers.quantizer_sinq import SinqHfQuantizer

        config = SinqConfig()
        quantizer = SinqHfQuantizer(quantization_config=config)

        # Should not raise
        quantizer.validate_environment()

    def test_validate_environment_no_cuda(self):
        """Test that error is raised when CUDA is not available but required."""
        from transformers.quantizers.quantizer_sinq import SinqHfQuantizer

        config = SinqConfig()
        quantizer = SinqHfQuantizer(quantization_config=config)

        with patch("torch.cuda.is_available", return_value=False):
            with self.assertRaises(RuntimeError):
                quantizer.validate_environment()

    def test_multi_device_error(self):
        """Test that multi-GPU device_map raises error."""
        from transformers.quantizers.quantizer_sinq import SinqHfQuantizer

        config = SinqConfig()
        quantizer = SinqHfQuantizer(quantization_config=config)

        device_map = {
            "layer1": "cuda:0",
            "layer2": "cuda:1",
        }

        with self.assertRaises(RuntimeError):
            quantizer.validate_environment(device_map=device_map)

    def test_update_dtype(self):
        """Test dtype updating."""
        from transformers.quantizers.quantizer_sinq import SinqHfQuantizer

        config = SinqConfig()
        quantizer = SinqHfQuantizer(quantization_config=config)

        # Should default to bfloat16
        result_dtype = quantizer.update_dtype(None)
        self.assertEqual(result_dtype, torch.bfloat16)

        # Should preserve existing dtype
        result_dtype = quantizer.update_dtype(torch.float16)
        self.assertEqual(result_dtype, torch.float16)

    def test_is_serializable(self):
        """Test serialization capability check."""
        from transformers.quantizers.quantizer_sinq import SinqHfQuantizer

        config = SinqConfig()
        quantizer = SinqHfQuantizer(quantization_config=config)

        self.assertTrue(quantizer.is_serializable(safe_serialization=True))

    def test_is_trainable(self):
        """Test that quantizer is marked as trainable."""
        from transformers.quantizers.quantizer_sinq import SinqHfQuantizer

        config = SinqConfig()
        quantizer = SinqHfQuantizer(quantization_config=config)
        self.assertTrue(quantizer.is_trainable)

    def test_get_quantize_ops(self):
        """Test that get_quantize_ops returns SinqQuantize."""
        from transformers.quantizers.quantizer_sinq import SinqHfQuantizer

        config = SinqConfig()
        quantizer = SinqHfQuantizer(quantization_config=config)

        ops = quantizer.get_quantize_ops()
        self.assertIsNotNone(ops)
        self.assertEqual(ops.__class__.__name__, "SinqQuantize")

    def test_get_weight_conversions_prequantized(self):
        """Test that weight conversions are returned for pre-quantized models."""
        from transformers.quantizers.quantizer_sinq import SinqHfQuantizer

        config = SinqConfig()
        quantizer = SinqHfQuantizer(quantization_config=config)
        quantizer.pre_quantized = True

        conversions = quantizer.get_weight_conversions()

        self.assertEqual(len(conversions), 1)
        converter = conversions[0]
        self.assertIn(".W_q", converter.source_patterns)
        self.assertIn(".meta", converter.source_patterns)
        self.assertIn(".bias", converter.source_patterns)
        self.assertIn(".weight", converter.target_patterns)

    def test_get_weight_conversions_not_prequantized(self):
        """Test that no weight conversions are returned for non-pre-quantized models."""
        from transformers.quantizers.quantizer_sinq import SinqHfQuantizer

        config = SinqConfig()
        quantizer = SinqHfQuantizer(quantization_config=config)
        quantizer.pre_quantized = False

        conversions = quantizer.get_weight_conversions()

        self.assertEqual(len(conversions), 0)

    def test_asinq_method_raises_error(self):
        """Test that asinq method raises appropriate error in validate_environment."""
        from transformers.quantizers.quantizer_sinq import SinqHfQuantizer

        config = SinqConfig(method="asinq")
        quantizer = SinqHfQuantizer(quantization_config=config)
        quantizer.pre_quantized = False

        with self.assertRaises(ValueError):
            quantizer.validate_environment()


@require_torch_gpu
class SinqModuleReplacementTest(unittest.TestCase):
    """Test module replacement behavior during quantization."""

    def setUp(self):
        gc.collect()
        _empty_accelerator_cache()
        self.model = _get_simple_model()

    def tearDown(self):
        gc.collect()
        _empty_accelerator_cache()

    def test_linear_replaced_with_sinqlinear(self):
        """Test that nn.Linear modules are replaced with SINQLinear."""
        from sinq.sinqlinear_hf import SINQLinear

        from transformers.quantizers.quantizer_sinq import SinqHfQuantizer

        config = SinqConfig(method="sinq")
        quantizer = SinqHfQuantizer(quantization_config=config)
        quantizer.update_dtype(None)  # Set dtype before processing
        quantizer.validate_environment()
        quantizer.pre_quantized = False

        # Before processing
        self.assertIsInstance(self.model.layer1, nn.Linear)
        self.assertIsInstance(self.model.layer2, nn.Linear)
        self.assertIsInstance(self.model.layer3, nn.Linear)

        quantizer._process_model_before_weight_loading(self.model, None)

        # After processing - all should be SINQLinear
        self.assertIsInstance(self.model.layer1, SINQLinear)
        self.assertIsInstance(self.model.layer2, SINQLinear)
        self.assertIsInstance(self.model.layer3, SINQLinear)

        # Should not be ready yet (no weights loaded)
        self.assertFalse(self.model.layer1.ready)
        self.assertFalse(self.model.layer2.ready)
        self.assertFalse(self.model.layer3.ready)

    def test_modules_to_not_convert(self):
        """Test that excluded modules are not quantized."""
        from sinq.sinqlinear_hf import SINQLinear

        from transformers.quantizers.quantizer_sinq import SinqHfQuantizer

        config = SinqConfig(
            method="sinq",
            modules_to_not_convert=["layer2"],
        )
        quantizer = SinqHfQuantizer(quantization_config=config)
        quantizer.update_dtype(None)  # Set dtype before processing
        quantizer.validate_environment()
        quantizer.pre_quantized = False
        quantizer._process_model_before_weight_loading(self.model, None)

        # layer1 should be SINQLinear
        self.assertIsInstance(self.model.layer1, SINQLinear)
        # layer2 should remain nn.Linear (excluded)
        self.assertIsInstance(self.model.layer2, nn.Linear)
        # layer3 should be SINQLinear
        self.assertIsInstance(self.model.layer3, SINQLinear)

    def test_param_needs_quantization(self):
        """Test param_needs_quantization for SINQ method."""
        from sinq.sinqlinear_hf import SINQLinear

        from transformers.quantizers.quantizer_sinq import SinqHfQuantizer

        config = SinqConfig(method="sinq")
        quantizer = SinqHfQuantizer(quantization_config=config)
        quantizer.update_dtype(None)  # Set dtype before processing
        quantizer.validate_environment()
        quantizer.pre_quantized = False

        quantizer._process_model_before_weight_loading(self.model, None)

        # After processing, modules should be SINQLinear
        self.assertIsInstance(self.model.layer1, SINQLinear)

        # Should need quantization for unquantized SINQLinear
        self.assertTrue(quantizer._do_param_level_sinq)
        self.assertTrue(quantizer.param_needs_quantization(self.model, "layer1.weight"))

        # Bias should not be quantized
        self.assertFalse(quantizer.param_needs_quantization(self.model, "layer1.bias"))

    def test_param_needs_quantization_prequantized(self):
        """Test that pre-quantized models skip param-level quantization."""
        from transformers.quantizers.quantizer_sinq import SinqHfQuantizer

        config = SinqConfig(method="sinq")
        quantizer = SinqHfQuantizer(quantization_config=config)
        quantizer.validate_environment()
        quantizer.pre_quantized = True

        self.assertFalse(quantizer.param_needs_quantization(self.model, "layer1.weight"))


@require_torch_gpu
class SinqQuantizeOpsTest(unittest.TestCase):
    """Test SinqQuantize conversion operations."""

    def setUp(self):
        gc.collect()
        _empty_accelerator_cache()
        self.model = _get_simple_model()

    def tearDown(self):
        gc.collect()
        _empty_accelerator_cache()

    def test_sinq_quantize_convert(self):
        """Test SinqQuantize.convert quantizes the weight into existing SINQLinear module."""
        from sinq.sinqlinear_hf import SINQLinear

        from transformers.integrations.sinq import SinqQuantize
        from transformers.quantizers.quantizer_sinq import SinqHfQuantizer

        config = SinqConfig()
        quantizer = SinqHfQuantizer(quantization_config=config)
        quantizer.update_dtype(None)  # Set dtype before processing
        quantizer.validate_environment()
        quantizer.pre_quantized = False

        # First, process model to create SINQLinear modules WITH quant_config
        quantizer._process_model_before_weight_loading(self.model, None)

        # Verify layer1 is now SINQLinear but not yet quantized
        self.assertIsInstance(self.model.layer1, SINQLinear)
        self.assertFalse(self.model.layer1.ready)

        ops = SinqQuantize(quantizer)

        # Create fake weight tensor with correct shape
        weight_tensor = torch.randn(256, 128, device="cuda:0")
        input_dict = {"layer1.weight": weight_tensor}

        # Convert (quantize)
        result = ops.convert(
            input_dict=input_dict,
            model=self.model,
            full_layer_name="layer1.weight",
            missing_keys=set(),
        )

        # Should return empty dict
        self.assertEqual(result, {})

        # Check that layer is now quantized
        self.assertTrue(self.model.layer1.ready)
        self.assertTrue(self.model.layer1._is_hf_initialized)

    def test_sinq_quantize_with_list_input(self):
        """Test SinqQuantize.convert handles list input correctly."""
        from transformers.integrations.sinq import SinqQuantize
        from transformers.quantizers.quantizer_sinq import SinqHfQuantizer

        config = SinqConfig()
        quantizer = SinqHfQuantizer(quantization_config=config)
        quantizer.update_dtype(None)  # Set dtype before processing
        quantizer.validate_environment()
        quantizer.pre_quantized = False

        quantizer._process_model_before_weight_loading(self.model, None)

        ops = SinqQuantize(quantizer)

        # Input as list (as it may come from weight loading)
        weight_tensor = torch.randn(256, 128, device="cuda:0")
        input_dict = {"layer1.weight": [weight_tensor]}

        result = ops.convert(
            input_dict=input_dict,
            model=self.model,
            full_layer_name="layer1.weight",
            missing_keys=set(),
        )

        self.assertEqual(result, {})
        self.assertTrue(self.model.layer1.ready)

@require_torch_gpu
class SinqDeserializeOpsTest(unittest.TestCase):
    """Test SinqDeserialize conversion operations for pre-quantized models."""

    def setUp(self):
        gc.collect()
        _empty_accelerator_cache()
        self.model = _get_simple_model()

    def tearDown(self):
        gc.collect()
        _empty_accelerator_cache()

    def test_sinq_deserialize_missing_wq(self):
        """Test SinqDeserialize returns tensor when W_q is missing."""
        from transformers.integrations.sinq import SinqDeserialize
        from transformers.quantizers.quantizer_sinq import SinqHfQuantizer

        config = SinqConfig()
        quantizer = SinqHfQuantizer(quantization_config=config)
        quantizer.update_dtype(None)  # Set dtype before processing
        quantizer.validate_environment()
        quantizer.pre_quantized = True

        quantizer._process_model_before_weight_loading(self.model, None)

        ops = SinqDeserialize(quantizer)

        # Missing W_q - should return the tensor as-is
        weight_tensor = torch.randn(256, 128)
        input_dict = {".weight": weight_tensor}

        result = ops.convert(
            input_dict=input_dict,
            model=self.model,
            full_layer_name="layer1.weight",
        )

        # Should return the original tensor
        self.assertIn("layer1.weight", result)


@require_torch_gpu
@slow
class SinqIntegrationTest(unittest.TestCase):
    """End-to-end integration tests for SINQ quantization."""

    model_name = "Qwen/Qwen2-0.5B"
    input_text = "What is the capital of France?"
    max_new_tokens = 10
    device_map = torch_device

    @classmethod
    def setUpClass(cls):
        """Setup quantized model and tokenizer once for all tests."""
        from transformers import AutoModelForCausalLM, AutoTokenizer

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

    def test_quantized_model_conversion(self):
        """Test that Linear modules are converted to SINQLinear."""
        from sinq.sinqlinear_hf import SINQLinear

        # Check that model layers are SINQLinear (except lm_head)
        nb_sinq_linear = 0
        for module in self.quantized_model.modules():
            if isinstance(module, SINQLinear):
                nb_sinq_linear += 1

        # Should have some SINQLinear modules
        self.assertGreater(nb_sinq_linear, 0)

        # lm_head should not be converted
        self.assertNotIsInstance(self.quantized_model.lm_head, SINQLinear)

    def test_quantized_model_generation(self):
        """Test that quantized model can generate text."""
        input_ids = self.tokenizer(self.input_text, return_tensors="pt").to(self.device_map)

        output = self.quantized_model.generate(**input_ids, max_new_tokens=self.max_new_tokens, do_sample=False)
        decoded = self.tokenizer.decode(output[0], skip_special_tokens=True)

        self.assertIsNotNone(decoded)
        self.assertGreater(len(decoded), len(self.input_text))

    def test_save_pretrained(self):
        """Test that quantized model can be saved and loaded."""
        import tempfile

        from transformers import AutoModelForCausalLM

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


if __name__ == "__main__":
    unittest.main()
