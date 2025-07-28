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
from unittest.mock import patch

import pytest

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, Mxfp4Config, OpenAIMoeForCausalLM
from transformers.testing_utils import (
    require_accelerate,
    require_torch_large_gpu,
    require_triton,
    require_triton_kernels,
    require_torch,
    require_torch_gpu,
    slow,
    torch_device,
)
from transformers.utils import (
    is_triton_available,
    is_triton_kernels_availalble,
    is_torch_available,
    is_accelerate_available,
)


if is_torch_available():
    import torch


class Mxfp4ConfigTest(unittest.TestCase):
    def test_basic_config_creation(self):
        """Test basic configuration creation with default values"""
        config = Mxfp4Config()
        self.assertEqual(config.quant_method.value, "mxfp4")
        self.assertIsNone(config.modules_to_not_convert)
        self.assertFalse(config.dequantize)

    def test_config_with_modules_to_not_convert(self):
        """Test configuration with modules to not convert"""
        modules = ["model.layers.*.self_attn", "lm_head"]
        config = Mxfp4Config(modules_to_not_convert=modules)
        self.assertEqual(config.modules_to_not_convert, modules)

    def test_config_with_dequantize(self):
        """Test configuration with dequantize enabled"""
        config = Mxfp4Config(dequantize=True)
        self.assertTrue(config.dequantize)

    def test_get_loading_attributes(self):
        """Test get_loading_attributes method"""
        config = Mxfp4Config(dequantize=True)
        attrs = config.get_loading_attributes()
        self.assertEqual(attrs, {"dequantize": True})

    def test_to_dict(self):
        """Test configuration serialization to dict"""
        config = Mxfp4Config(modules_to_not_convert=["lm_head"], dequantize=True)
        config_dict = config.to_dict()
        self.assertEqual(config_dict["quant_method"], "mxfp4")
        self.assertEqual(config_dict["modules_to_not_convert"], ["lm_head"])
        self.assertTrue(config_dict["dequantize"])

    def test_from_dict(self):
        """Test configuration creation from dict"""
        config_dict = {"quant_method": "mxfp4", "modules_to_not_convert": ["lm_head"], "dequantize": True}
        config = Mxfp4Config.from_dict(config_dict)
        self.assertEqual(config.modules_to_not_convert, ["lm_head"])
        self.assertTrue(config.dequantize)


class Mxfp4QuantizerTest(unittest.TestCase):
    """Test the Mxfp4HfQuantizer class"""

    def setUp(self):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def test_quantizer_validation_no_torch(self):
        """Test quantizer validation when torch is not available"""
        with patch("transformers.quantizers.quantizer_mxfp4.is_torch_available", return_value=False):
            from transformers.quantizers.quantizer_mxfp4 import Mxfp4HfQuantizer

            config = Mxfp4Config()
            quantizer = Mxfp4HfQuantizer(config)

            with self.assertRaises(ImportError):
                quantizer.validate_environment()

    def test_quantizer_validation_no_cuda(self):
        """Test quantizer validation when CUDA is not available"""
        with patch("torch.cuda.is_available", return_value=False):
            from transformers.quantizers.quantizer_mxfp4 import Mxfp4HfQuantizer

            config = Mxfp4Config()
            quantizer = Mxfp4HfQuantizer(config)

            with self.assertRaises(RuntimeError):
                quantizer.validate_environment()

    def test_quantizer_validation_low_compute_capability(self):
        """Test quantizer validation with low compute capability"""
        with patch("torch.cuda.get_device_capability", return_value=(8, 0)):
            from transformers.quantizers.quantizer_mxfp4 import Mxfp4HfQuantizer

            config = Mxfp4Config()
            quantizer = Mxfp4HfQuantizer(config)

            with self.assertRaises(ValueError):
                quantizer.validate_environment()

    def test_quantizer_validation_low_compute_capability_with_dequantize(self):
        """Test quantizer validation with low compute capability but dequantize enabled"""
        with patch("torch.cuda.get_device_capability", return_value=(8, 0)):
            from transformers.quantizers.quantizer_mxfp4 import Mxfp4HfQuantizer

            config = Mxfp4Config(dequantize=True)
            quantizer = Mxfp4HfQuantizer(config)

            # Should not raise error with dequantize=True
            try:
                quantizer.validate_environment()
            except ValueError as e:
                if "compute capability" in str(e):
                    self.fail("Should not raise compute capability error when dequantize=True")

    def test_quantizer_validation_missing_triton(self):
        """Test quantizer validation when triton is not available"""
        with (
            patch("transformers.quantizers.quantizer_mxfp4.is_triton_available", return_value=False),
            patch("transformers.quantizers.quantizer_mxfp4.is_triton_kernels_availalble", return_value=False),
        ):
            from transformers.quantizers.quantizer_mxfp4 import Mxfp4HfQuantizer

            config = Mxfp4Config()
            quantizer = Mxfp4HfQuantizer(config)
            quantizer.pre_quantized = False
            with self.assertRaises(ValueError):
                quantizer.validate_environment()

    def test_quantizer_validation_missing_triton_pre_quantized_no_dequantize(self):
        """Test quantizer validation when triton is not available but model is pre-quantized and dequantize is False"""
        with (
            patch("transformers.quantizers.quantizer_mxfp4.is_triton_available", return_value=False),
            patch("transformers.quantizers.quantizer_mxfp4.is_triton_kernels_availalble", return_value=False),
        ):
            from transformers.quantizers.quantizer_mxfp4 import Mxfp4HfQuantizer

            config = Mxfp4Config()
            quantizer = Mxfp4HfQuantizer(config)
            quantizer.pre_quantized = True

            # Should automatically set dequantize=True and warn
            quantizer.validate_environment()
            self.assertTrue(quantizer.quantization_config.dequantize)

    def test_update_torch_dtype(self):
        """Test torch dtype updating"""
        from transformers.quantizers.quantizer_mxfp4 import Mxfp4HfQuantizer

        config = Mxfp4Config()
        quantizer = Mxfp4HfQuantizer(config)

        # Should default to bfloat16
        result_dtype = quantizer.update_torch_dtype(None)
        self.assertEqual(result_dtype, torch.bfloat16)

        # Should preserve existing dtype
        result_dtype = quantizer.update_torch_dtype(torch.float32)
        self.assertEqual(result_dtype, torch.float32)

    def test_update_expected_keys(self):
        """Test expected keys updating for quantized models"""
        from transformers.quantizers.quantizer_mxfp4 import Mxfp4HfQuantizer

        config = Mxfp4Config()
        quantizer = Mxfp4HfQuantizer(config)

        expected_keys = [
            "model.layers.0.mlp.experts.gate_up_proj",
            "model.layers.0.mlp.experts.down_proj",
            "model.embed_tokens.weight",
        ]

        updated_keys = quantizer.update_expected_keys(None, expected_keys, [])

        expected_updated = [
            "model.layers.0.mlp.experts.gate_up_proj_blocks",
            "model.layers.0.mlp.experts.gate_up_proj_scales",
            "model.layers.0.mlp.experts.down_proj_blocks",
            "model.layers.0.mlp.experts.down_proj_scales",
            "model.embed_tokens.weight",
        ]

        self.assertEqual(set(updated_keys), set(expected_updated))

    def test_update_param_name_dequantize(self):
        """Test parameter name updating when dequantizing"""
        from transformers.quantizers.quantizer_mxfp4 import Mxfp4HfQuantizer

        config = Mxfp4Config(dequantize=True)
        quantizer = Mxfp4HfQuantizer(config)

        # Should remove _blocks suffix
        param_name = "model.layers.0.mlp.experts.gate_up_proj_blocks"
        updated_name = quantizer.update_param_name(param_name)
        self.assertEqual(updated_name, "model.layers.0.mlp.experts.gate_up_proj")

        # Should remove _scales suffix
        param_name = "model.layers.0.mlp.experts.down_proj_scales"
        updated_name = quantizer.update_param_name(param_name)
        self.assertEqual(updated_name, "model.layers.0.mlp.experts.down_proj")

        # Should not change other names
        param_name = "model.embed_tokens.weight"
        updated_name = quantizer.update_param_name(param_name)
        self.assertEqual(updated_name, "model.embed_tokens.weight")

    def test_update_param_name_no_dequantize(self):
        """Test parameter name updating when not dequantizing"""
        from transformers.quantizers.quantizer_mxfp4 import Mxfp4HfQuantizer

        config = Mxfp4Config(dequantize=False)
        quantizer = Mxfp4HfQuantizer(config)

        param_name = "model.layers.0.mlp.experts.gate_up_proj_blocks"
        updated_name = quantizer.update_param_name(param_name)
        self.assertEqual(updated_name, param_name)

    def test_is_serializable(self):
        """Test serialization capability"""
        from transformers.quantizers.quantizer_mxfp4 import Mxfp4HfQuantizer

        config = Mxfp4Config()
        quantizer = Mxfp4HfQuantizer(config)

        # MXFP4 is not serializable with safetensors
        self.assertFalse(quantizer.is_serializable())

    def test_is_trainable(self):
        """Test trainability"""
        from transformers.quantizers.quantizer_mxfp4 import Mxfp4HfQuantizer

        config = Mxfp4Config()
        quantizer = Mxfp4HfQuantizer(config)

        # MXFP4 is not trainable
        self.assertFalse(quantizer.is_trainable)


class Mxfp4IntegrationTest(unittest.TestCase):
    """Test mxfp4 integration functions"""

    def test_should_convert_module(self):
        """Test module conversion decision logic"""
        from transformers.integrations.mxfp4 import should_convert_module

        # Should convert by default
        self.assertTrue(should_convert_module(["model", "layers", "0", "mlp"], []))

        # Should not convert if in exclusion list
        patterns = ["model.layers.*.self_attn", "lm_head"]
        self.assertFalse(should_convert_module(["model", "layers", "0", "self_attn"], patterns))
        self.assertFalse(should_convert_module(["lm_head"], patterns))

        # Should convert if not in exclusion list
        self.assertTrue(should_convert_module(["model", "layers", "0", "mlp", "experts"], patterns))

    @require_torch
    def test_convert_moe_packed_tensors(self):
        """Test unpacking of quantized tensors"""
        from transformers.integrations.mxfp4 import convert_moe_packed_tensors

        # Create dummy packed tensors
        blocks = torch.randint(0, 255, (2, 4, 8), dtype=torch.uint8)
        scales = torch.randint(100, 150, (2, 4), dtype=torch.uint8)

        result = convert_moe_packed_tensors(blocks, scales, dtype=torch.bfloat16)

        # Check output shape - should be [2, 4, 16] (8 * 2 for unpacking)
        self.assertEqual(result.shape, (2, 4 * 16))
        self.assertEqual(result.dtype, torch.bfloat16)

    @require_triton(min_version="3.4.0")
    @require_triton_kernels
    @require_torch_gpu
    @require_torch
    def test_quantize_to_mxfp4(self):
        """Test quantization function"""
        from transformers.integrations.mxfp4 import quantize_to_mxfp4

        # Create dummy weight tensor
        w = torch.randn(32, 64, 128, dtype=torch.bfloat16, device=torch.device("cuda"))

        quantized_w, flex_data, mx_ctx = quantize_to_mxfp4(w, None, None)

        # Check that shapes are reasonable
        self.assertEqual(quantized_w.dtype, torch.uint8)
        self.assertIsNotNone(flex_data)
        self.assertIsNotNone(mx_ctx)


# @require_torch
# @require_torch_large_gpu
# @slow
class Mxfp4ModelTest(unittest.TestCase):
    """Test mxfp4 with actual models (requires specific model and hardware)"""

    # These should be paths to real OpenAI MoE models for proper testing
    model_name_packed = "/fsx/mohamed/oai-hf/tests/20b_converted_packed"  # TODO: Use real packed quantized model

    input_text = "Once upon a time"

    # Expected outputs for generation tests
    EXPECTED_OUTPUTS = set()
    EXPECTED_OUTPUTS.add("Once upon a time, in a small village, there lived a young")

    def setUp(self):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def tearDown(self):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def check_inference_correctness_quantized(self, model, tokenizer):
        # Check that inference pass works on the model
        encoded_input = tokenizer(self.input_text, return_tensors="pt").to(model.device)

        # Set pad token if not set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        with torch.no_grad():
            output_sequences = model.generate(
                **encoded_input,
                max_new_tokens=10,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )

        generated_text = tokenizer.decode(output_sequences[0], skip_special_tokens=True)

        self.assertIn(generated_text, self.EXPECTED_OUTPUTS)

    def test_openai_moe_model_loading_quantized_with_device_map(self):
        """Test loading OpenAI MoE model with mxfp4 quantization and device_map"""

        quantization_config = Mxfp4Config(dequantize=False)

        # Test that config is properly set up
        self.assertFalse(quantization_config.dequantize)

        model = OpenAIMoeForCausalLM.from_pretrained(
            self.model_name_packed,
            quantization_config=quantization_config,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        tokenizer = AutoTokenizer.from_pretrained(self.model_name_packed)
        self.check_inference_correctness_quantized(model, tokenizer)

    def test_openai_moe_model_loading_dequantized_with_device_map(self):
        """Test loading OpenAI MoE model with mxfp4 dequantization and device_map"""

        quantization_config = Mxfp4Config(dequantize=True)

        # Test that config is properly set up
        self.assertTrue(quantization_config.dequantize)

        model = OpenAIMoeForCausalLM.from_pretrained(
            self.model_name_packed,
            quantization_config=quantization_config,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        tokenizer = AutoTokenizer.from_pretrained(self.model_name_packed)
        self.check_inference_correctness_quantized(model, tokenizer)

    def test_model_device_map_validation(self):
        """Test device map validation"""
        from transformers.quantizers.quantizer_mxfp4 import Mxfp4HfQuantizer

        config = Mxfp4Config()
        quantizer = Mxfp4HfQuantizer(config)
        quantizer.pre_quantized = False

        # Test with CPU in device map (should raise error for non-pre-quantized)
        with self.assertRaises(ValueError):
            quantizer.validate_environment(device_map={"": "cpu"})

    def test_memory_footprint_comparison(self):
        """Test memory footprint differences between quantized and unquantized models"""

        # Expected: quantized < dequantized < unquantized memory usage
        quantization_config = Mxfp4Config(dequantize=True)
        quantized_model = OpenAIMoeForCausalLM.from_pretrained(
            self.model_name_packed,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        dequantized_model = OpenAIMoeForCausalLM.from_pretrained(
            self.model_name_packed,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            quantization_config=quantization_config,
        )
        quantized_mem = quantized_model.get_memory_footprint()
        dequantized_mem = dequantized_model.get_memory_footprint()
        self.assertLess(quantized_mem, dequantized_mem)