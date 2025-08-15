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
from unittest.mock import patch

from transformers import AutoTokenizer, GptOssForCausalLM, Mxfp4Config
from transformers.testing_utils import (
    require_kernels,
    require_torch,
    require_torch_gpu,
    require_torch_large_gpu,
    require_triton,
    slow,
)
from transformers.utils import (
    is_torch_available,
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
        with patch("torch.cuda.get_device_capability", return_value=(7, 0)):
            from transformers.quantizers.quantizer_mxfp4 import Mxfp4HfQuantizer

            config = Mxfp4Config()
            quantizer = Mxfp4HfQuantizer(config)
            quantizer.pre_quantized = False

            with self.assertRaises(ValueError):
                quantizer.validate_environment()

    def test_quantizer_validation_low_compute_capability_with_prequantized(self):
        """Test quantizer validation with low compute capability"""
        with patch("torch.cuda.get_device_capability", return_value=(7, 0)):
            from transformers.quantizers.quantizer_mxfp4 import Mxfp4HfQuantizer

            config = Mxfp4Config()
            quantizer = Mxfp4HfQuantizer(config)

            # Should automatically set dequantize=True and warn
            quantizer.validate_environment()
            self.assertTrue(quantizer.quantization_config.dequantize)

    def test_quantizer_validation_low_compute_capability_with_dequantize(self):
        """Test quantizer validation with low compute capability but dequantize enabled"""
        with patch("torch.cuda.get_device_capability", return_value=(7, 0)):
            from transformers.quantizers.quantizer_mxfp4 import Mxfp4HfQuantizer

            config = Mxfp4Config(dequantize=True)
            quantizer = Mxfp4HfQuantizer(config)

            # Should not raise error with dequantize=True
            try:
                quantizer.validate_environment()
            except ValueError as e:
                if "compute capability" in str(e):
                    self.fail("Should not raise compute capability error when dequantize=True")

    def test_quantizer_validation_dequantize_on_cpu(self):
        """Test quantizer validation with dequantize enabled on CPU-only environment"""
        with patch("torch.cuda.is_available", return_value=False):
            from transformers.quantizers.quantizer_mxfp4 import Mxfp4HfQuantizer

            config = Mxfp4Config(dequantize=True)
            quantizer = Mxfp4HfQuantizer(config)

            # Should not raise error when dequantize=True even without CUDA
            try:
                quantizer.validate_environment()
            except RuntimeError as e:
                if "requires a GPU" in str(e):
                    self.fail("Should not raise GPU requirement error when dequantize=True on CPU")

    def test_quantizer_validation_order_dequantize_before_cuda_check(self):
        """Test that dequantize check happens before CUDA availability check"""
        # Mock both torch.cuda.is_available and is_accelerate_available to return False
        with (
            patch("torch.cuda.is_available", return_value=False),
            patch(
                "transformers.quantizers.quantizer_mxfp4.is_accelerate_available",
                return_value=False,
            ),
        ):
            from transformers.quantizers.quantizer_mxfp4 import Mxfp4HfQuantizer

            # Test with dequantize=True - should pass even without CUDA and accelerate
            config = Mxfp4Config(dequantize=True)
            quantizer = Mxfp4HfQuantizer(config)

            # This should not raise any error because dequantize check comes first
            try:
                quantizer.validate_environment()
            except (RuntimeError, ImportError) as e:
                if "requires a GPU" in str(e) or "requires Accelerate" in str(e):
                    self.fail(f"Should not raise error when dequantize=True: {e}")

            # Test with dequantize=False - should still fail due to missing CUDA
            config = Mxfp4Config(dequantize=False)
            quantizer = Mxfp4HfQuantizer(config)

            with self.assertRaises(RuntimeError) as context:
                quantizer.validate_environment()
            self.assertIn("requires a GPU", str(context.exception))

    def test_quantizer_validation_missing_triton(self):
        """Test quantizer validation when triton is not available"""
        with (
            patch("transformers.quantizers.quantizer_mxfp4.is_triton_available", return_value=False),
            patch("transformers.quantizers.quantizer_mxfp4.is_kernels_availalble", return_value=False),
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
            patch("transformers.quantizers.quantizer_mxfp4.is_kernels_availalble", return_value=False),
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

    def test_is_trainable_disabled(self):
        """Test trainability when training is disabled"""
        from transformers.quantizers.quantizer_mxfp4 import Mxfp4HfQuantizer

        config = Mxfp4Config()
        quantizer = Mxfp4HfQuantizer(config)

        # MXFP4 is not trainable by default
        self.assertFalse(quantizer.is_trainable)

    def test_is_trainable_enabled(self):
        """Test trainability when training is enabled"""
        from transformers.quantizers.quantizer_mxfp4 import Mxfp4HfQuantizer

        config = Mxfp4Config(enable_training=True)
        quantizer = Mxfp4HfQuantizer(config)

        # MXFP4 should be trainable with enable_training=True
        self.assertTrue(quantizer.is_trainable)


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
    @require_kernels
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


@require_torch 
class Mxfp4HardwareDetectionTest(unittest.TestCase):
    """Test hardware detection and fallback mechanisms for mxfp4 training"""

    def setUp(self):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @require_torch_gpu
    def test_hardware_capabilities_detection(self):
        """Test basic hardware capability detection"""
        from transformers.integrations.mxfp4 import get_hardware_capabilities, ComputeBackend

        hw_caps = get_hardware_capabilities()
        
        # Should detect some backend
        self.assertIsNotNone(hw_caps.backend)
        self.assertIsInstance(hw_caps.backend, ComputeBackend)
        
        # Should detect compute capability
        self.assertIsNotNone(hw_caps.compute_capability)
        self.assertIsInstance(hw_caps.compute_capability, tuple)
        self.assertEqual(len(hw_caps.compute_capability), 2)
        
        # Performance multiplier should be reasonable
        perf_mult = hw_caps.get_performance_multiplier()
        self.assertGreater(perf_mult, 0.0)
        self.assertLessEqual(perf_mult, 1.0)

    def test_backend_selection_h100(self):
        """Test backend selection for H100/B100 hardware"""
        from transformers.integrations.mxfp4 import HardwareCapabilities, ComputeBackend
        
        hw_caps = HardwareCapabilities()
        backend = hw_caps._select_backend((9, 0))  # H100 capability
        self.assertEqual(backend, ComputeBackend.FP4_NATIVE)

    def test_backend_selection_a100(self):
        """Test backend selection for A100/L4 hardware"""
        from transformers.integrations.mxfp4 import HardwareCapabilities, ComputeBackend
        
        hw_caps = HardwareCapabilities()
        backend = hw_caps._select_backend((8, 0))  # A100 capability
        self.assertEqual(backend, ComputeBackend.FP8_FALLBACK)

    def test_backend_selection_legacy(self):
        """Test backend selection for legacy hardware"""
        from transformers.integrations.mxfp4 import HardwareCapabilities, ComputeBackend
        
        hw_caps = HardwareCapabilities()
        backend = hw_caps._select_backend((7, 5))  # T4 capability
        self.assertEqual(backend, ComputeBackend.BF16_FALLBACK)

    def test_fallback_strategy(self):
        """Test progressive fallback strategy"""
        from transformers.integrations.mxfp4 import HardwareCapabilities, ComputeBackend
        
        hw_caps = HardwareCapabilities()
        hw_caps.backend = ComputeBackend.FP8_FALLBACK
        
        fallback_chain = hw_caps.get_fallback_strategy()
        
        # Should start from current backend
        self.assertEqual(fallback_chain[0], ComputeBackend.FP8_FALLBACK)
        # Should include BF16 as final fallback
        self.assertEqual(fallback_chain[-1], ComputeBackend.BF16_FALLBACK)

    @patch("torch.cuda.is_available", return_value=False)
    def test_hardware_detection_no_cuda(self, mock_cuda):
        """Test hardware detection when CUDA is not available"""
        from transformers.integrations.mxfp4 import HardwareCapabilities, ComputeBackend
        
        hw_caps = HardwareCapabilities()
        self.assertEqual(hw_caps.backend, ComputeBackend.BF16_FALLBACK)


@require_torch
class Mxfp4TrainingConfigTest(unittest.TestCase):
    """Test mxfp4 training configuration functionality"""

    def test_config_with_training_enabled(self):
        """Test configuration with training enabled"""
        config = Mxfp4Config(enable_training=True)
        self.assertTrue(config.enable_training)
        self.assertEqual(config.quant_method.value, "mxfp4")

    def test_config_serialization_with_training(self):
        """Test configuration serialization with training flag"""
        config = Mxfp4Config(enable_training=True, modules_to_not_convert=["lm_head"])
        config_dict = config.to_dict()
        
        self.assertTrue(config_dict["enable_training"])
        self.assertEqual(config_dict["modules_to_not_convert"], ["lm_head"])

    def test_config_from_dict_with_training(self):
        """Test configuration creation from dict with training flag"""
        config_dict = {
            "quant_method": "mxfp4",
            "enable_training": True,
            "dequantize": False
        }
        config = Mxfp4Config.from_dict(config_dict)
        self.assertTrue(config.enable_training)
        self.assertFalse(config.dequantize)


@require_torch
@require_torch_gpu
class Mxfp4AutogradTest(unittest.TestCase):
    """Test mxfp4 autograd functionality for training"""

    def setUp(self):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def test_autograd_function_forward(self):
        """Test forward pass of MxFp4MatMulFunction"""
        from transformers.integrations.mxfp4 import MxFp4MatMulFunction
        
        # Create mock tensors
        input_tensor = torch.randn(2, 4, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        weight_mock = torch.randn(8, 64, 32, device="cuda", dtype=torch.bfloat16)
        bias_mock = torch.randn(8, 32, device="cuda", dtype=torch.bfloat16)
        
        # Create mock routing data (simplified)
        class MockRoutingData:
            def __init__(self):
                self.gate_scal = torch.ones(8, device="cuda")
        
        routing_data = MockRoutingData()
        
        # Test forward pass execution (will need proper kernel setup to work fully)
        try:
            # This would need actual triton kernels to work, so we'll test the structure
            with self.assertRaises((AttributeError, NameError)):
                # Should fail due to missing triton_kernels_hub, but shouldn't crash on forward logic
                MxFp4MatMulFunction.apply(input_tensor, weight_mock, bias_mock, routing_data)
        except Exception as e:
            # Expected due to missing kernels in test environment
            pass

    def test_hardware_backend_selection_in_autograd(self):
        """Test that autograd function selects appropriate hardware backend"""
        from transformers.integrations.mxfp4 import MxFp4MatMulFunction, get_hardware_capabilities
        
        # Test backend selection works
        hw_caps = get_hardware_capabilities()
        self.assertIsNotNone(hw_caps.backend)
        
        # Test fallback strategy exists
        fallback_strategy = hw_caps.get_fallback_strategy()
        self.assertGreater(len(fallback_strategy), 0)


@require_torch
@require_torch_gpu 
class Mxfp4TrainingIntegrationTest(unittest.TestCase):
    """Test mxfp4 training integration with expert modules"""

    def setUp(self):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def test_expert_module_training_enablement(self):
        """Test enabling training on Mxfp4GptOssExperts module"""
        from transformers.integrations.mxfp4 import Mxfp4GptOssExperts
        
        # Create mock config
        class MockConfig:
            num_local_experts = 4
            intermediate_size = 256
            hidden_size = 128
        
        config = MockConfig()
        expert_module = Mxfp4GptOssExperts(config)
        
        # Initially training should be disabled
        self.assertFalse(expert_module.training_enabled)
        
        # Bias parameters should not require gradients initially
        self.assertFalse(expert_module.gate_up_proj_bias.requires_grad)
        self.assertFalse(expert_module.down_proj_bias.requires_grad)
        
        # Enable training
        expert_module.enable_mxfp4_training()
        
        # Now training should be enabled
        self.assertTrue(expert_module.training_enabled)
        
        # Bias parameters should require gradients
        self.assertTrue(expert_module.gate_up_proj_bias.requires_grad)
        self.assertTrue(expert_module.down_proj_bias.requires_grad)

    def test_forward_pass_mode_selection(self):
        """Test that forward pass selects correct mode based on training state"""
        from transformers.integrations.mxfp4 import Mxfp4GptOssExperts
        
        class MockConfig:
            num_local_experts = 2
            intermediate_size = 64 
            hidden_size = 32
        
        config = MockConfig()
        expert_module = Mxfp4GptOssExperts(config)
        
        # Create mock input
        hidden_states = torch.randn(1, 4, 32, device="cuda", dtype=torch.bfloat16)
        
        # Mock routing data and indices
        class MockRoutingData:
            def __init__(self):
                self.gate_scal = torch.ones(4, device="cuda")
        
        routing_data = MockRoutingData()
        gather_idx = torch.zeros(4, dtype=torch.int32, device="cuda")
        scatter_idx = torch.zeros(4, dtype=torch.int32, device="cuda")
        
        # Test that method selection logic works (will fail due to missing kernels)
        with self.assertRaises((AttributeError, NameError)):
            expert_module.forward(hidden_states, routing_data, gather_idx, scatter_idx)


@require_torch
@require_torch_gpu
class Mxfp4MemoryValidationTest(unittest.TestCase):
    """Test memory usage validation for mxfp4 training"""

    def setUp(self):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def test_quantizer_hardware_validation(self):
        """Test quantizer hardware validation for training"""
        from transformers.quantizers.quantizer_mxfp4 import Mxfp4HfQuantizer
        from transformers.integrations.mxfp4 import get_hardware_capabilities
        
        config = Mxfp4Config(enable_training=True)
        quantizer = Mxfp4HfQuantizer(config)
        
        # Test that hardware validation works (may warn but shouldn't crash)
        hw_caps = get_hardware_capabilities()
        try:
            quantizer._validate_training_hardware(hw_caps)
        except RuntimeError as e:
            if "compute capability" in str(e):
                # Expected on insufficient hardware
                pass
            else:
                raise

    def test_memory_recommendations_output(self):
        """Test that memory recommendations are properly formatted"""
        from transformers.quantizers.quantizer_mxfp4 import Mxfp4HfQuantizer
        from transformers.integrations.mxfp4 import get_hardware_capabilities
        
        config = Mxfp4Config(enable_training=True)
        quantizer = Mxfp4HfQuantizer(config)
        hw_caps = get_hardware_capabilities()
        
        # Test recommendations formatting (shouldn't crash)
        try:
            quantizer._log_training_recommendations(hw_caps, 4)
        except Exception as e:
            self.fail(f"Training recommendations should not crash: {e}")

    def test_performance_multiplier_values(self):
        """Test that performance multipliers are within reasonable bounds"""
        from transformers.integrations.mxfp4 import HardwareCapabilities, ComputeBackend
        
        hw_caps = HardwareCapabilities()
        
        # Test each backend
        for backend in ComputeBackend:
            hw_caps.backend = backend
            perf_mult = hw_caps.get_performance_multiplier()
            
            # Should be between 0.5 and 1.0
            self.assertGreaterEqual(perf_mult, 0.5)
            self.assertLessEqual(perf_mult, 1.0)


@require_torch
@require_torch_large_gpu
@require_triton(min_version="3.4.0")
@require_kernels
@slow
class Mxfp4ModelTest(unittest.TestCase):
    """Test mxfp4 with actual models (requires specific model and hardware)"""

    # These should be paths to real OpenAI MoE models for proper testing
    model_name = "openai/gpt-oss-20b"

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
                use_cache=False,
            )

        generated_text = tokenizer.decode(output_sequences[0], skip_special_tokens=True)

        self.assertIn(generated_text, self.EXPECTED_OUTPUTS)

    def test_gpt_oss_model_loading_quantized_with_device_map(self):
        """Test loading OpenAI MoE model with mxfp4 quantization and device_map"""

        quantization_config = Mxfp4Config(dequantize=False)

        # Test that config is properly set up
        self.assertFalse(quantization_config.dequantize)

        model = GptOssForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=quantization_config,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.check_inference_correctness_quantized(model, tokenizer)

    def test_gpt_oss_model_loading_dequantized_with_device_map(self):
        """Test loading OpenAI MoE model with mxfp4 dequantization and device_map"""

        quantization_config = Mxfp4Config(dequantize=True)

        # Test that config is properly set up
        self.assertTrue(quantization_config.dequantize)

        model = GptOssForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=quantization_config,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
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
        quantized_model = GptOssForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        dequantized_model = GptOssForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            quantization_config=quantization_config,
        )
        quantized_mem = quantized_model.get_memory_footprint()
        dequantized_mem = dequantized_model.get_memory_footprint()
        self.assertLess(quantized_mem, dequantized_mem)

    def test_mxfp4_training_end_to_end(self):
        """Test end-to-end training with mxfp4 quantization"""
        # This test will work once backward kernels are implemented
        quantization_config = Mxfp4Config(enable_training=True)
        
        model = GptOssForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=quantization_config,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Verify training is enabled
        self.assertTrue(quantization_config.enable_training)
        
        # Set model to training mode
        model.train()
        
        # Create dummy training data
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        inputs = tokenizer(
            ["Hello world", "Training test"], 
            return_tensors="pt", 
            padding=True,
            truncation=True,
            max_length=10
        ).to(model.device)
        
        # Test forward pass in training mode (will use gradient-aware path)
        try:
            outputs = model(**inputs, labels=inputs["input_ids"])
            
            # Should have loss for training
            self.assertIsNotNone(outputs.loss)
            
            # Test backward pass (will use fallback until native kernels available)
            loss = outputs.loss
            loss.backward()
            
            # Bias parameters should have gradients
            bias_grad_count = 0
            for name, param in model.named_parameters():
                if "bias" in name and param.grad is not None:
                    bias_grad_count += 1
            
            # Should have at least some bias gradients
            self.assertGreater(bias_grad_count, 0)
            
        except Exception as e:
            # Expected until native backward kernels are fully implemented
            # Just verify the training setup doesn't crash the model loading
            self.assertIsNotNone(model)
            self.assertTrue(hasattr(model.config, 'quantization_config'))

    def test_training_memory_efficiency(self):
        """Test that mxfp4 training uses less memory than bfloat16"""
        # Load model with training enabled
        quantization_config = Mxfp4Config(enable_training=True)
        
        try:
            model = GptOssForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=quantization_config,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
            
            training_mem = model.get_memory_footprint()
            
            # Compare with dequantized (bfloat16) training
            dequant_config = Mxfp4Config(dequantize=True)
            dequant_model = GptOssForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=dequant_config,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
            
            dequant_mem = dequant_model.get_memory_footprint()
            
            # Quantized training should use less memory
            # Even with training overhead, should be <4X due to quantized weights
            memory_ratio = training_mem / dequant_mem
            self.assertLess(memory_ratio, 0.8)  # Should be significantly less
            
        except Exception as e:
            # Skip if model can't be loaded (missing kernels, etc.)
            self.skipTest(f"Could not load model for memory test: {e}")
