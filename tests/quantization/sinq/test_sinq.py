"""
Comprehensive test suite for SINQ quantization integration in Hugging Face Transformers v5.

Tests cover:
1. Configuration creation and validation
2. Quantizer initialization and validation
3. Weight-only SINQ quantization (param-level)
4. Module exclusion (modules_to_not_convert)
5. Error handling and edge cases
6. Deserialization of pre-quantized models

All tests run with real SINQ library.
"""

import pytest
import torch
import torch.nn as nn
from unittest.mock import Mock, patch
import tempfile
import os
from pathlib import Path

# Imports from transformers
try:
    from transformers import AutoModel, AutoTokenizer
    from transformers.utils.quantization_config import SinqConfig
    from transformers.quantizers.quantizer_sinq import SinqHfQuantizer
    from transformers.integrations.sinq import SinqQuantize, SinqDeserialize
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    pytest.skip("Transformers not available", allow_module_level=True)

# Check SINQ availability
try:
    import sinq
    from sinq.sinqlinear_hf import SINQLinear
    SINQ_AVAILABLE = True
except ImportError:
    SINQ_AVAILABLE = False
    pytest.skip("SINQ library not available", allow_module_level=True)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def simple_model():
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


@pytest.fixture
def multimodal_model():
    """Create a model that simulates multimodal/vision architecture."""
    class MultimodalModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.text_encoder = nn.Linear(128, 256)
            self.vision_encoder = nn.Linear(256, 512)
            self.config = Mock()
            self.config._name_or_path = "test-multimodal"
            self.config.model_type = "siglip"
            self.config.vision_config = Mock()

        def forward(self, x):
            return x

    return MultimodalModel()


# ============================================================================
# Test SinqConfig
# ============================================================================

class TestSinqConfig:
    """Test SinqConfig creation, validation, and serialization."""

    def test_default_config(self):
        """Test default configuration values."""
        config = SinqConfig()
        assert config.nbits == 4
        assert config.group_size == 64
        assert config.tiling_mode == "1D"
        assert config.method == "sinq"
        assert config.dtype == "auto"
        assert config.device == "cpu"

    def test_custom_config(self):
        """Test custom configuration values."""
        config = SinqConfig(
            nbits=8,
            group_size=128,
            tiling_mode="2D",
            method="sinq",
            dtype="bfloat16",
            device="cuda:0"
        )
        assert config.nbits == 8
        assert config.group_size == 128
        assert config.tiling_mode == "2D"
        assert config.method == "sinq"
        assert config.dtype == "bfloat16"
        assert config.device == "cuda:0"

    def test_method_validation(self):
        """Test that invalid method raises error."""
        with pytest.raises(ValueError, match="method.*must be either 'sinq' or 'asinq'"):
            SinqConfig(method="invalid_method")

    def test_modules_to_not_convert(self):
        """Test modules_to_not_convert configuration."""
        modules = ["layer1", "layer2.weight"]
        config = SinqConfig(modules_to_not_convert=modules)
        assert config.modules_to_not_convert == modules

    def test_to_dict(self):
        """Test configuration serialization to dict."""
        config = SinqConfig(nbits=8, method="sinq")
        config_dict = config.to_dict()

        assert config_dict["nbits"] == 8
        assert config_dict["method"] == "sinq"
        assert config_dict["group_size"] == 64
        assert "quant_method" in config_dict

    def test_from_dict(self):
        """Test configuration deserialization from dict."""
        config_dict = {
            "nbits": 8,
            "group_size": 128,
            "method": "sinq",
            "dtype": "float16"
        }
        config = SinqConfig.from_dict(config_dict)

        assert config.nbits == 8
        assert config.group_size == 128
        assert config.method == "sinq"
        assert config.dtype == "float16"

    def test_extra_kwargs_preserved(self):
        """Test that extra kwargs are preserved in round-trip."""
        config = SinqConfig(custom_param="test_value")
        assert hasattr(config, "_extra_kwargs")
        assert config._extra_kwargs.get("custom_param") == "test_value"

        config_dict = config.to_dict()
        assert config_dict.get("custom_param") == "test_value"

    def test_is_trainable(self):
        """Test that SINQ config is marked as trainable."""
        config = SinqConfig()
        assert config.is_trainable is True

    def test_is_serializable(self):
        """Test that SINQ config is marked as serializable."""
        config = SinqConfig()
        assert config.is_serializable is True


# ============================================================================
# Test SinqHfQuantizer - Initialization and Validation
# ============================================================================

class TestSinqHfQuantizerInit:
    """Test SinqHfQuantizer initialization and environment validation."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_quantizer_init(self):
        """Test basic quantizer initialization."""
        config = SinqConfig()
        quantizer = SinqHfQuantizer(quantization_config=config)

        assert quantizer.quantization_config == config
        assert quantizer.requires_calibration is False
        assert quantizer.requires_parameters_quantization is True

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_validate_environment_cuda(self):
        """Test environment validation with CUDA."""
        config = SinqConfig(device="cuda:0")
        quantizer = SinqHfQuantizer(quantization_config=config)

        # Should not raise
        quantizer.validate_environment()
        # dtype should be set to default bfloat16
        assert quantizer.dtype == torch.bfloat16

    def test_validate_environment_no_cuda(self):
        """Test that error is raised when CUDA is not available but required."""
        config = SinqConfig(device="cuda:0")
        quantizer = SinqHfQuantizer(quantization_config=config)

        with patch('torch.cuda.is_available', return_value=False):
            # The error message might vary, so just check that RuntimeError is raised
            with pytest.raises(RuntimeError):
                quantizer.validate_environment()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_multi_device_error(self):
        """Test that multi-GPU device_map raises error."""
        config = SinqConfig(device="cuda:0")
        quantizer = SinqHfQuantizer(quantization_config=config)

        device_map = {
            "layer1": "cuda:0",
            "layer2": "cuda:1"
        }

        with pytest.raises(RuntimeError, match="multi-GPU device_map detected"):
            quantizer.validate_environment(device_map=device_map)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_dtype_from_validate_environment(self):
        """Test dtype setting through validate_environment."""
        config = SinqConfig()
        quantizer = SinqHfQuantizer(quantization_config=config)

        # Pass dtype via kwargs
        quantizer.validate_environment(torch_dtype=torch.float16)
        assert quantizer.dtype == torch.float16

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_dtype_default(self):
        """Test default dtype when not specified."""
        config = SinqConfig()
        quantizer = SinqHfQuantizer(quantization_config=config)

        quantizer.validate_environment()
        # Should default to bfloat16
        assert quantizer.dtype == torch.bfloat16

    def test_is_serializable(self):
        """Test serialization capability check."""
        config = SinqConfig()
        quantizer = SinqHfQuantizer(quantization_config=config)

        # Current implementation: True for safe_serialization=True, False otherwise
        result = quantizer.is_serializable(safe_serialization=True)
        assert result is True

        result = quantizer.is_serializable(safe_serialization=False)
        assert result is False

    def test_is_trainable(self):
        """Test that quantizer is marked as trainable."""
        config = SinqConfig()
        quantizer = SinqHfQuantizer(quantization_config=config)
        assert quantizer.is_trainable is True


# ============================================================================
# Test Weight-Only SINQ Quantization (param-level)
# ============================================================================

class TestSinqParamLevelQuantization:
    """Test param-level SINQ quantization during weight loading."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_param_needs_quantization_sinq(self, simple_model):
        """Test param_needs_quantization for SINQ method."""
        config = SinqConfig(method="sinq", device="cuda:0")
        quantizer = SinqHfQuantizer(quantization_config=config)
        quantizer.validate_environment()

        # Need to process model first to set _do_param_level_sinq and create SINQLinear modules
        quantizer._process_model_before_weight_loading(simple_model, None)

        # After processing, modules should be SINQLinear
        assert isinstance(simple_model.layer1, SINQLinear)

        # Now check if it needs quantization (should return True for unquantized SINQLinear)
        assert quantizer._do_param_level_sinq is True
        assert quantizer.param_needs_quantization(simple_model, "layer1.weight") is True

        # Bias should not be quantized
        assert quantizer.param_needs_quantization(simple_model, "layer1.bias") is False

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_param_needs_quantization_prequantized(self, simple_model):
        """Test that pre-quantized models skip param-level quantization."""
        config = SinqConfig(method="sinq")
        quantizer = SinqHfQuantizer(quantization_config=config)
        quantizer.validate_environment()
        quantizer.pre_quantized = True

        assert quantizer.param_needs_quantization(
            simple_model, "layer1.weight"
        ) is False

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_modules_to_not_convert(self, simple_model):
        """Test that excluded modules are not quantized."""
        config = SinqConfig(
            method="sinq",
            modules_to_not_convert=["layer2"],
            device="cuda:0"
        )
        quantizer = SinqHfQuantizer(quantization_config=config)
        quantizer.validate_environment()
        quantizer._process_model_before_weight_loading(simple_model, None)

        # layer1 should be SINQLinear
        assert isinstance(simple_model.layer1, SINQLinear)
        # layer2 should remain nn.Linear (excluded)
        assert isinstance(simple_model.layer2, nn.Linear)
        # layer3 should be SINQLinear
        assert isinstance(simple_model.layer3, SINQLinear)

        # layer1.weight should need quantization
        assert quantizer.param_needs_quantization(simple_model, "layer1.weight") is True
        # layer2.weight should NOT need quantization (excluded and still nn.Linear)
        assert quantizer.param_needs_quantization(simple_model, "layer2.weight") is False

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_get_quantize_ops(self):
        """Test that get_quantize_ops returns SinqQuantize."""
        config = SinqConfig()
        quantizer = SinqHfQuantizer(quantization_config=config)

        ops = quantizer.get_quantize_ops()
        assert ops is not None
        assert ops.__class__.__name__ == "SinqQuantize"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_build_sinq_quant_dict(self):
        """Test building SINQ quantization config dict."""
        config = SinqConfig(nbits=8, group_size=128, method="sinq")
        quantizer = SinqHfQuantizer(quantization_config=config)
        quantizer.validate_environment()

        quant_dict = quantizer._build_sinq_quant_dict(config)

        assert isinstance(quant_dict, dict)
        assert "weight_quant_params" in quant_dict
        assert quant_dict["weight_quant_params"]["nbits"] == 8
        assert quant_dict["weight_quant_params"]["group_size"] == 128


# ============================================================================
# Test SinqQuantize ConversionOps
# ============================================================================

class TestSinqQuantizeOps:
    """Test SinqQuantize conversion operations."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_sinq_quantize_convert(self, simple_model):
        """Test SinqQuantize.convert quantizes the weight into existing SINQLinear module."""
        config = SinqConfig(device="cuda:0")
        quantizer = SinqHfQuantizer(quantization_config=config)
        quantizer.validate_environment()

        # First, process model to create SINQLinear modules
        quantizer._process_model_before_weight_loading(simple_model, None)

        # Verify layer1 is now SINQLinear but not yet quantized
        assert isinstance(simple_model.layer1, SINQLinear)
        assert simple_model.layer1.ready is False

        from transformers.integrations.sinq import SinqQuantize
        ops = SinqQuantize(quantizer)

        # Create fake weight tensor with correct shape
        weight_tensor = torch.randn(256, 128, device="cuda:0")
        input_dict = {"layer1.weight": weight_tensor}

        # Convert (quantize)
        result = ops.convert(
            input_dict=input_dict,
            model=simple_model,
            full_layer_name="layer1.weight",
            missing_keys=set()
        )

        # Should return empty dict
        assert result == {}

        # Check that layer is now quantized
        assert simple_model.layer1.ready is True
        assert simple_model.layer1._is_hf_initialized is True

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_sinq_quantize_with_list_input(self, simple_model):
        """Test SinqQuantize.convert handles list input correctly."""
        config = SinqConfig(device="cuda:0")
        quantizer = SinqHfQuantizer(quantization_config=config)
        quantizer.validate_environment()

        quantizer._process_model_before_weight_loading(simple_model, None)

        from transformers.integrations.sinq import SinqQuantize
        ops = SinqQuantize(quantizer)

        # Input as list (as it may come from weight loading)
        weight_tensor = torch.randn(256, 128, device="cuda:0")
        input_dict = {"layer1.weight": [weight_tensor]}

        result = ops.convert(
            input_dict=input_dict,
            model=simple_model,
            full_layer_name="layer1.weight",
            missing_keys=set()
        )

        assert result == {}
        assert simple_model.layer1.ready is True


# ============================================================================
# Test SinqDeserialize ConversionOps
# ============================================================================

class TestSinqDeserializeOps:
    """Test SinqDeserialize conversion operations for pre-quantized models."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_sinq_deserialize_convert(self, simple_model):
        """Test SinqDeserialize.convert loads state into existing SINQLinear module."""
        config = SinqConfig(device="cuda:0")
        quantizer = SinqHfQuantizer(quantization_config=config)
        quantizer.validate_environment()
        quantizer.pre_quantized = True

        # Process model to create empty SINQLinear modules (pre-quantized mode)
        quantizer._process_model_before_weight_loading(simple_model, None)

        # Verify layer1 is now SINQLinear
        assert isinstance(simple_model.layer1, SINQLinear)

        from transformers.integrations.sinq import SinqDeserialize
        ops = SinqDeserialize(quantizer)

        # Create fake quantized tensors
        W_q = torch.randint(0, 255, (256, 128), dtype=torch.uint8, device="cuda:0")
        meta = {
            "shape": (256, 128),
            "nbits": 4,
            "group_size": 64,
            "scale": torch.randn(256, 2, device="cuda:0"),
            "zero": torch.randn(256, 2, device="cuda:0"),
        }

        input_dict = {
            ".W_q": W_q,
            ".meta": meta,
            ".bias": None,
        }

        result = ops.convert(
            input_dict=input_dict,
            model=simple_model,
            full_layer_name="layer1.weight",
        )

        # Should return empty dict
        assert result == {}

        # Check that layer was loaded
        assert simple_model.layer1.ready is True
        assert simple_model.layer1._is_hf_initialized is True

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_sinq_deserialize_missing_wq(self, simple_model):
        """Test SinqDeserialize returns tensor when W_q is missing."""
        config = SinqConfig(device="cuda:0")
        quantizer = SinqHfQuantizer(quantization_config=config)
        quantizer.validate_environment()
        quantizer.pre_quantized = True

        quantizer._process_model_before_weight_loading(simple_model, None)

        from transformers.integrations.sinq import SinqDeserialize
        ops = SinqDeserialize(quantizer)

        # Missing W_q - should return the tensor as-is
        weight_tensor = torch.randn(256, 128)
        input_dict = {".weight": weight_tensor}

        result = ops.convert(
            input_dict=input_dict,
            model=simple_model,
            full_layer_name="layer1.weight",
        )

        # Should return the original tensor
        assert "layer1.weight" in result


# ============================================================================
# Test Module Replacement in _process_model_before_weight_loading
# ============================================================================

class TestModuleReplacement:
    """Test module replacement behavior."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_linear_replaced_with_sinqlinear(self, simple_model):
        """Test that nn.Linear modules are replaced with SINQLinear."""
        config = SinqConfig(method="sinq", device="cuda:0")
        quantizer = SinqHfQuantizer(quantization_config=config)
        quantizer.validate_environment()

        # Before processing
        assert isinstance(simple_model.layer1, nn.Linear)
        assert isinstance(simple_model.layer2, nn.Linear)
        assert isinstance(simple_model.layer3, nn.Linear)

        quantizer._process_model_before_weight_loading(simple_model, None)

        # After processing - all should be SINQLinear
        assert isinstance(simple_model.layer1, SINQLinear)
        assert isinstance(simple_model.layer2, SINQLinear)
        assert isinstance(simple_model.layer3, SINQLinear)

        # Should not be ready yet (no weights loaded)
        assert simple_model.layer1.ready is False
        assert simple_model.layer2.ready is False
        assert simple_model.layer3.ready is False

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_prequantized_module_replacement(self, simple_model):
        """Test module replacement for pre-quantized models."""
        config = SinqConfig(method="sinq", device="cuda:0")
        quantizer = SinqHfQuantizer(quantization_config=config)
        quantizer.validate_environment()
        quantizer.pre_quantized = True

        quantizer._process_model_before_weight_loading(simple_model, None)

        # Should still replace with SINQLinear
        assert isinstance(simple_model.layer1, SINQLinear)

        # But _do_param_level_sinq should be False for pre-quantized
        assert quantizer._do_param_level_sinq is False

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_sinqlinear_has_correct_config(self, simple_model):
        """Test that created SINQLinear modules have correct config."""
        config = SinqConfig(nbits=4, group_size=64, method="sinq", device="cuda:0")
        quantizer = SinqHfQuantizer(quantization_config=config)
        quantizer.validate_environment()

        quantizer._process_model_before_weight_loading(simple_model, None)

        # Check quant_config is set
        assert simple_model.layer1.quant_config is not None
        assert simple_model.layer1.quant_config["weight_quant_params"]["nbits"] == 4
        assert simple_model.layer1.quant_config["weight_quant_params"]["group_size"] == 64


# ============================================================================
# Test Edge Cases and Error Handling
# ============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_invalid_device_spec(self):
        """Test invalid device specification."""
        config = SinqConfig(device="invalid_device")
        quantizer = SinqHfQuantizer(quantization_config=config)

        with pytest.raises(ValueError, match="Unsupported device spec"):
            # Need to trigger device normalization
            from transformers.quantizers.quantizer_sinq import _normalize_cuda_device
            _normalize_cuda_device("invalid_device")

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_device_out_of_range(self):
        """Test CUDA device index out of range."""
        from transformers.quantizers.quantizer_sinq import _normalize_cuda_device

        # This should not raise - it just returns the string
        result = _normalize_cuda_device(f"cuda:{torch.cuda.device_count() + 10}")
        assert "cuda:" in result

    def test_tokenizer_resolution_failure(self, simple_model):
        """Test graceful handling when tokenizer cannot be resolved."""
        config = SinqConfig(method="sinq")
        quantizer = SinqHfQuantizer(quantization_config=config)

        # Should not crash even without tokenizer
        tok, model_id = quantizer._resolve_tokenizer_and_model_id(simple_model, {})
        # May be None, which is acceptable

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_asinq_method_raises_error(self, simple_model):
        """Test that asinq method raises appropriate error."""
        config = SinqConfig(method="asinq", device="cuda:0")
        quantizer = SinqHfQuantizer(quantization_config=config)
        quantizer.validate_environment()

        with pytest.raises(ValueError, match="A-SINQ is not supported"):
            quantizer._process_model_before_weight_loading(simple_model, None)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_asinq_after_weight_loading_error(self, simple_model):
        """Test that asinq method raises error in _process_model_after_weight_loading."""
        config = SinqConfig(method="asinq", device="cuda:0")
        quantizer = SinqHfQuantizer(quantization_config=config)
        quantizer.validate_environment()
        quantizer.pre_quantized = False
        # Manually set to bypass _process_model_before_weight_loading check
        quantizer.quantization_config.method = "asinq"

        with pytest.raises(ValueError, match="asinq"):
            quantizer._process_model_after_weight_loading(simple_model)


# ============================================================================
# Test get_weight_conversions
# ============================================================================

class TestWeightConversions:
    """Test weight conversion configuration."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_get_weight_conversions_prequantized(self):
        """Test that weight conversions are returned for pre-quantized models."""
        config = SinqConfig()
        quantizer = SinqHfQuantizer(quantization_config=config)
        quantizer.pre_quantized = True

        conversions = quantizer.get_weight_conversions()

        assert len(conversions) == 1
        converter = conversions[0]
        assert ".W_q" in converter.source_patterns
        assert ".meta" in converter.source_patterns
        assert ".bias" in converter.source_patterns
        assert ".weight" in converter.target_patterns

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_get_weight_conversions_not_prequantized(self):
        """Test that no weight conversions are returned for non-pre-quantized models."""
        config = SinqConfig()
        quantizer = SinqHfQuantizer(quantization_config=config)
        quantizer.pre_quantized = False

        conversions = quantizer.get_weight_conversions()

        assert len(conversions) == 0


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """End-to-end integration tests."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_full_quantization_pipeline(self, simple_model):
        """Test complete quantization pipeline from config to quantized model."""
        config = SinqConfig(
            nbits=4,
            group_size=64,
            method="sinq",
            device="cuda:0"
        )

        quantizer = SinqHfQuantizer(quantization_config=config)
        quantizer.validate_environment()

        # Process model before weight loading
        quantizer._process_model_before_weight_loading(simple_model, None)

        # Verify modules are SINQLinear but not yet ready
        assert isinstance(simple_model.layer1, SINQLinear)
        assert simple_model.layer1.ready is False

        # Simulate weight loading by quantizing each layer
        from transformers.integrations.sinq import SinqQuantize
        ops = SinqQuantize(quantizer)

        # Quantize layer1
        weight1 = torch.randn(256, 128, device="cuda:0")
        ops.convert(
            input_dict={"layer1.weight": weight1},
            model=simple_model,
            full_layer_name="layer1.weight",
            missing_keys=set()
        )

        # Quantize layer2
        weight2 = torch.randn(512, 256, device="cuda:0")
        ops.convert(
            input_dict={"layer2.weight": weight2},
            model=simple_model,
            full_layer_name="layer2.weight",
            missing_keys=set()
        )

        # Quantize layer3
        weight3 = torch.randn(128, 512, device="cuda:0")
        ops.convert(
            input_dict={"layer3.weight": weight3},
            model=simple_model,
            full_layer_name="layer3.weight",
            missing_keys=set()
        )

        # Process model after weight loading
        result = quantizer._process_model_after_weight_loading(simple_model)

        # All layers should now be ready
        assert simple_model.layer1.ready is True
        assert simple_model.layer2.ready is True
        assert simple_model.layer3.ready is True


# ============================================================================
# Performance and Correctness Tests
# ============================================================================

class TestPerformanceAndCorrectness:
    """Test quantization correctness and performance characteristics."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_quantized_output_shape(self, simple_model):
        """Test that quantized model maintains correct output shapes."""
        config = SinqConfig(method="sinq", device="cuda:0")
        quantizer = SinqHfQuantizer(quantization_config=config)
        quantizer.validate_environment()

        quantizer._process_model_before_weight_loading(simple_model, None)

        # Quantize all layers
        from transformers.integrations.sinq import SinqQuantize
        ops = SinqQuantize(quantizer)

        layers_weights = [
            ("layer1.weight", torch.randn(256, 128, device="cuda:0")),
            ("layer2.weight", torch.randn(512, 256, device="cuda:0")),
            ("layer3.weight", torch.randn(128, 512, device="cuda:0")),
        ]

        for layer_name, weight in layers_weights:
            ops.convert(
                input_dict={layer_name: weight},
                model=simple_model,
                full_layer_name=layer_name,
                missing_keys=set()
            )

        quantizer._process_model_after_weight_loading(simple_model)

        # Test forward pass
        simple_model.eval()

        with torch.no_grad():
            x = torch.randn(4, 128, device="cuda:0", dtype=quantizer.dtype)
            output = simple_model(x)

            assert output.shape == (4, 128)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_quantized_model_has_sinq_layers(self, simple_model):
        """Test that quantized model contains SINQLinear layers."""
        config = SinqConfig(nbits=4, method="sinq", device="cuda:0")
        quantizer = SinqHfQuantizer(quantization_config=config)
        quantizer.validate_environment()

        quantizer._process_model_before_weight_loading(simple_model, None)

        # Check that all linear layers are now SINQLinear
        has_sinq_layer = any(isinstance(m, SINQLinear) for m in simple_model.modules())
        assert has_sinq_layer, "Model should have at least one SINQLinear layer"

        # Count SINQLinear layers
        sinq_count = sum(1 for m in simple_model.modules() if isinstance(m, SINQLinear))
        assert sinq_count == 3, f"Expected 3 SINQLinear layers, got {sinq_count}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
