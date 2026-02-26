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
import unittest
from contextlib import ExitStack, contextmanager
from unittest.mock import patch

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, MetalConfig, OPTForCausalLM
from transformers.quantizers.quantizer_metal import MetalHfQuantizer
from transformers.testing_utils import (
    require_torch,
    slow,
)
from transformers.utils import is_torch_available


if is_torch_available():
    import torch
    import torch.nn as nn


@contextmanager
def _patch_mps_available(available: bool = True):
    """Patch ``torch.backends.mps.is_available`` to simulate MPS presence/absence."""
    with patch("torch.backends.mps.is_available", return_value=available):
        yield


@contextmanager
def _patch_no_mps():
    """Convenience: simulate a machine with no MPS device."""
    with _patch_mps_available(False):
        yield


@contextmanager
def _patch_has_mps():
    """Convenience: simulate a machine with an MPS device."""
    with ExitStack() as stack:
        stack.enter_context(_patch_mps_available(True))
        stack.enter_context(patch("transformers.quantizers.quantizer_metal.is_kernels_available", return_value=True))
        yield


@require_torch
class MetalConfigTest(unittest.TestCase):
    """Unit tests for ``MetalConfig`` (no device / model needed)."""

    def test_default_values(self):
        config = MetalConfig()
        self.assertEqual(config.bits, 4)
        self.assertEqual(config.group_size, 64)
        self.assertIsNone(config.modules_to_not_convert)
        self.assertFalse(config.dequantize)
        self.assertEqual(config.quant_method, "metal")

    def test_custom_values(self):
        config = MetalConfig(bits=8, group_size=32, modules_to_not_convert=["lm_head"], dequantize=True)
        self.assertEqual(config.bits, 8)
        self.assertEqual(config.group_size, 32)
        self.assertEqual(config.modules_to_not_convert, ["lm_head"])
        self.assertTrue(config.dequantize)

    def test_invalid_bits_raises(self):
        for bad_bits in (0, 1, 3, 5, 6, 7, 16):
            with self.assertRaises(ValueError, msg=f"bits={bad_bits} should raise"):
                MetalConfig(bits=bad_bits)

    def test_valid_bits(self):
        for bits in (2, 4, 8):
            config = MetalConfig(bits=bits)
            self.assertEqual(config.bits, bits)

    def test_invalid_group_size_raises(self):
        with self.assertRaises(ValueError):
            MetalConfig(group_size=0)
        with self.assertRaises(ValueError):
            MetalConfig(group_size=-1)

    def test_to_dict(self):
        config = MetalConfig(bits=4, group_size=64, modules_to_not_convert=["lm_head"])
        d = config.to_dict()
        self.assertEqual(d["quant_method"], "metal")
        self.assertEqual(d["bits"], 4)
        self.assertEqual(d["group_size"], 64)
        self.assertEqual(d["modules_to_not_convert"], ["lm_head"])

    def test_from_dict(self):
        d = {"quant_method": "metal", "bits": 8, "group_size": 32, "modules_to_not_convert": None}
        config = MetalConfig.from_dict(d)
        self.assertEqual(config.bits, 8)
        self.assertEqual(config.group_size, 32)

    def test_to_dict_from_dict(self):
        original = MetalConfig(bits=2, group_size=128, modules_to_not_convert=["lm_head"])
        d = original.to_dict()
        restored = MetalConfig.from_dict(d)
        self.assertEqual(original.bits, restored.bits)
        self.assertEqual(original.group_size, restored.group_size)
        self.assertEqual(original.modules_to_not_convert, restored.modules_to_not_convert)

    def test_get_loading_attributes(self):
        config = MetalConfig(dequantize=True)
        attrs = config.get_loading_attributes()
        self.assertIn("dequantize", attrs)
        self.assertTrue(attrs["dequantize"])


@require_torch
class MetalQuantizerEnvironmentTest(unittest.TestCase):
    """Validate ``MetalHfQuantizer.validate_environment`` under various conditions."""

    def test_no_mps_prequantized_triggers_dequantize(self):
        """Pre-quantized model on non-MPS machine should auto-enable dequantize."""
        with _patch_no_mps():
            config = MetalConfig()
            quantizer = MetalHfQuantizer(config)
            quantizer.pre_quantized = True
            quantizer.validate_environment()
            self.assertTrue(quantizer.quantization_config.dequantize)

    def test_no_mps_not_prequantized_raises(self):
        """Quantize-on-the-fly on non-MPS machine should raise."""
        with _patch_no_mps():
            config = MetalConfig()
            quantizer = MetalHfQuantizer(config)
            quantizer.pre_quantized = False
            with self.assertRaises(RuntimeError):
                quantizer.validate_environment()

    def test_dequantize_flag_skips_mps_check(self):
        """When dequantize=True, no MPS check should occur."""
        with _patch_no_mps():
            config = MetalConfig(dequantize=True)
            quantizer = MetalHfQuantizer(config)
            quantizer.pre_quantized = True
            quantizer.validate_environment()

    def test_missing_kernels_raises(self):
        """Missing ``kernels`` package should raise ImportError."""
        with ExitStack() as stack:
            stack.enter_context(_patch_mps_available(True))
            stack.enter_context(
                patch("transformers.quantizers.quantizer_metal.is_kernels_available", return_value=False)
            )
            config = MetalConfig()
            quantizer = MetalHfQuantizer(config)
            quantizer.pre_quantized = False
            with self.assertRaises(ImportError):
                quantizer.validate_environment()

    def test_cpu_in_device_map_not_prequantized_raises(self):
        """Quantize-on-the-fly with CPU in device_map should raise."""
        with _patch_has_mps():
            config = MetalConfig()
            quantizer = MetalHfQuantizer(config)
            quantizer.pre_quantized = False
            with self.assertRaises(ValueError):
                quantizer.validate_environment(device_map={"model": "cpu"})

    def test_disk_in_device_map_not_prequantized_raises(self):
        """Quantize-on-the-fly with disk in device_map should raise."""
        with _patch_has_mps():
            config = MetalConfig()
            quantizer = MetalHfQuantizer(config)
            quantizer.pre_quantized = False
            with self.assertRaises(ValueError):
                quantizer.validate_environment(device_map={"model": "disk"})

    def test_update_device_map_defaults_to_mps(self):
        config = MetalConfig()
        quantizer = MetalHfQuantizer(config)
        result = quantizer.update_device_map(None)
        self.assertEqual(result, {"": "mps"})

    def test_is_serializable(self):
        config = MetalConfig()
        quantizer = MetalHfQuantizer(config)
        self.assertTrue(quantizer.is_serializable())

    def test_is_not_trainable(self):
        config = MetalConfig()
        quantizer = MetalHfQuantizer(config)
        self.assertFalse(quantizer.is_trainable)


@require_torch
class AffineQuantizeDequantizeTest(unittest.TestCase):
    """Test the low-level ``_affine_quantize_tensor`` / ``_affine_dequantize_tensor`` functions."""

    def _roundtrip(self, bits, group_size, N=64, K=256, dtype=torch.float32):
        from transformers.integrations.metal_quantization import _affine_dequantize_tensor, _affine_quantize_tensor

        weight = torch.randn(N, K, dtype=dtype)
        w_packed, scales, biases = _affine_quantize_tensor(weight, group_size, bits)

        self.assertEqual(w_packed.dtype, torch.uint32)
        self.assertEqual(w_packed.shape, (N, K // (32 // bits)))
        self.assertEqual(scales.shape, (N, K // group_size))
        self.assertEqual(biases.shape, (N, K // group_size))

        w_deq = _affine_dequantize_tensor(w_packed, scales, biases, group_size, bits)
        self.assertEqual(w_deq.shape, (N, K))

        return weight.float(), w_deq.float()

    def test_roundtrip_4bit_gs64(self):
        orig, deq = self._roundtrip(bits=4, group_size=64)
        max_err = (orig - deq).abs().max().item()
        self.assertLess(max_err, 0.25, "4-bit gs=64 round-trip error too large")

    def test_roundtrip_4bit_gs128(self):
        orig, deq = self._roundtrip(bits=4, group_size=128)
        max_err = (orig - deq).abs().max().item()
        self.assertLess(max_err, 0.5, "4-bit gs=128 round-trip error too large")

    def test_roundtrip_8bit_gs64(self):
        orig, deq = self._roundtrip(bits=8, group_size=64)
        max_err = (orig - deq).abs().max().item()
        self.assertLess(max_err, 0.02, "8-bit gs=64 round-trip error too large")

    def test_roundtrip_2bit_gs64(self):
        orig, deq = self._roundtrip(bits=2, group_size=64)
        max_err = (orig - deq).abs().max().item()
        self.assertLess(max_err, 1.25, "2-bit gs=64 round-trip error too large")

    def test_quantize_shapes_2bit(self):
        from transformers.integrations.metal_quantization import _affine_quantize_tensor

        N, K = 32, 128
        weight = torch.randn(N, K)
        w_packed, scales, biases = _affine_quantize_tensor(weight, group_size=64, bits=2)
        elems_per_int = 32 // 2
        self.assertEqual(w_packed.shape, (N, K // elems_per_int))
        self.assertEqual(scales.shape, (N, K // 64))

    def test_quantize_preserves_device(self):
        from transformers.integrations.metal_quantization import _affine_quantize_tensor

        weight = torch.randn(32, 128, device="cpu")
        w_packed, scales, biases = _affine_quantize_tensor(weight, group_size=64, bits=4)
        self.assertEqual(w_packed.device.type, "cpu")
        self.assertEqual(scales.device.type, "cpu")
        self.assertEqual(biases.device.type, "cpu")

    def test_dequantize_returns_correct_dtype(self):
        """Regression: dequantize should always return float32 (caller casts to target dtype)."""
        from transformers.integrations.metal_quantization import _affine_dequantize_tensor, _affine_quantize_tensor

        weight = torch.randn(32, 128, dtype=torch.bfloat16)
        w_packed, scales, biases = _affine_quantize_tensor(weight, group_size=64, bits=4)
        w_deq = _affine_dequantize_tensor(w_packed, scales, biases, group_size=64, bits=4)
        self.assertEqual(w_deq.dtype, torch.float32)


@require_torch
class MetalLinearTest(unittest.TestCase):
    """Test the ``MetalLinear`` nn.Module directly (CPU, no kernel calls)."""

    def test_prequantized_weight_shape(self):
        """Pre-quantized mode: weight should be uint32 with packed K dimension."""
        from transformers.integrations.metal_quantization import MetalLinear

        layer = MetalLinear(in_features=256, out_features=128, bits=4, group_size=64)
        elems_per_int = 32 // 4
        self.assertEqual(layer.weight.shape, (128, 256 // elems_per_int))
        self.assertEqual(layer.weight.dtype, torch.uint32)
        self.assertEqual(layer.scales.shape, (128, 256 // 64))
        self.assertEqual(layer.qbiases.shape, (128, 256 // 64))

    def test_quantize_on_the_fly_weight_shape(self):
        """Quantize-on-the-fly mode (dtype=None): weight should be full-shape float."""
        from transformers.integrations.metal_quantization import MetalLinear

        layer = MetalLinear(in_features=256, out_features=128, bits=4, group_size=64, dtype=None)
        self.assertEqual(layer.weight.shape, (128, 256))
        self.assertNotEqual(layer.weight.dtype, torch.uint32)

    def test_no_bias_by_default(self):
        from transformers.integrations.metal_quantization import MetalLinear

        layer = MetalLinear(in_features=128, out_features=64, bits=4, group_size=64)
        self.assertIsNone(layer.bias)

    def test_with_bias(self):
        from transformers.integrations.metal_quantization import MetalLinear

        layer = MetalLinear(in_features=128, out_features=64, bias=True, bits=4, group_size=64)
        self.assertIsNotNone(layer.bias)
        self.assertEqual(layer.bias.shape, (64,))

    def test_forward_fallback_when_not_uint32(self):
        """When weight is not uint32, forward should use standard nn.functional.linear (no kernel needed)."""
        from transformers.integrations.metal_quantization import MetalLinear

        layer = MetalLinear(in_features=128, out_features=64, bits=4, group_size=64, dtype=None)
        layer.weight = nn.Parameter(torch.randn(64, 128))
        x = torch.randn(2, 5, 128)
        out = layer(x)
        self.assertEqual(out.shape, (2, 5, 64))

    def test_forward_fallback_with_bias(self):
        from transformers.integrations.metal_quantization import MetalLinear

        layer = MetalLinear(in_features=128, out_features=64, bias=True, bits=4, group_size=64, dtype=None)
        layer.weight = nn.Parameter(torch.randn(64, 128))
        layer.bias = nn.Parameter(torch.randn(64))
        x = torch.randn(1, 10, 128)
        out = layer(x)
        self.assertEqual(out.shape, (1, 10, 64))

    def test_prequantized_shapes_8bit(self):
        from transformers.integrations.metal_quantization import MetalLinear

        layer = MetalLinear(in_features=256, out_features=128, bits=8, group_size=64)
        elems_per_int = 32 // 8  # 4
        self.assertEqual(layer.weight.shape, (128, 256 // elems_per_int))

    def test_prequantized_shapes_2bit(self):
        from transformers.integrations.metal_quantization import MetalLinear

        layer = MetalLinear(in_features=256, out_features=128, bits=2, group_size=64)
        elems_per_int = 32 // 2  # 16
        self.assertEqual(layer.weight.shape, (128, 256 // elems_per_int))


@require_torch
class ReplaceWithMetalLinearTest(unittest.TestCase):
    """Test module replacement logic."""

    def _make_small_model(self):
        config = AutoConfig.from_pretrained("hf-internal-testing/tiny-random-OPTForCausalLM")
        with torch.device("meta"):
            model = OPTForCausalLM(config)
        return model

    def test_all_linears_replaced(self):
        from transformers.integrations.metal_quantization import MetalLinear, replace_with_metal_linear

        model = self._make_small_model()
        nb_linears = sum(1 for m in model.modules() if isinstance(m, nn.Linear))
        self.assertGreater(nb_linears, 0)

        config = MetalConfig(bits=4, group_size=64)
        replace_with_metal_linear(model, quantization_config=config, pre_quantized=True)

        nb_metal = sum(1 for m in model.modules() if isinstance(m, MetalLinear))
        self.assertEqual(nb_linears, nb_metal)

    def test_modules_to_not_convert(self):
        from transformers.integrations.metal_quantization import MetalLinear, replace_with_metal_linear

        model = self._make_small_model()
        config = MetalConfig(bits=4, group_size=64)
        replace_with_metal_linear(
            model, modules_to_not_convert=["lm_head"], quantization_config=config, pre_quantized=True
        )
        self.assertNotIsInstance(model.lm_head, MetalLinear)

        nb_metal = sum(1 for m in model.modules() if isinstance(m, MetalLinear))
        nb_linears = sum(1 for m in model.modules() if isinstance(m, nn.Linear))
        self.assertEqual(nb_metal, nb_linears - 1)

    def test_dequantize_skips_replacement(self):
        from transformers.integrations.metal_quantization import MetalLinear, replace_with_metal_linear

        model = self._make_small_model()
        config = MetalConfig(bits=4, group_size=64, dequantize=True)
        replace_with_metal_linear(model, quantization_config=config, pre_quantized=True)

        nb_metal = sum(1 for m in model.modules() if isinstance(m, MetalLinear))
        self.assertEqual(nb_metal, 0, "No modules should be replaced when dequantize=True")

    def test_prequantized_dtype_is_uint32(self):
        from transformers.integrations.metal_quantization import MetalLinear, replace_with_metal_linear

        model = self._make_small_model()
        config = MetalConfig(bits=4, group_size=64)
        replace_with_metal_linear(model, quantization_config=config, pre_quantized=True)

        for m in model.modules():
            if isinstance(m, MetalLinear):
                self.assertEqual(m.weight.dtype, torch.uint32)
                break

    def test_quantize_on_the_fly_dtype_is_not_uint32(self):
        from transformers.integrations.metal_quantization import MetalLinear, replace_with_metal_linear

        model = self._make_small_model()
        config = MetalConfig(bits=4, group_size=64)
        replace_with_metal_linear(model, quantization_config=config, pre_quantized=False)

        for m in model.modules():
            if isinstance(m, MetalLinear):
                self.assertNotEqual(m.weight.dtype, torch.uint32)
                break


@require_torch
class MetalConversionOpsTest(unittest.TestCase):
    """Test the ``MetalQuantize`` and ``MetalDequantize`` weight conversion operations."""

    def _make_quantizer(self, bits=4, group_size=64):
        config = MetalConfig(bits=bits, group_size=group_size)
        quantizer = MetalHfQuantizer(config)
        quantizer.pre_quantized = False
        return quantizer

    def test_metal_quantize_produces_correct_keys(self):
        from transformers.integrations.metal_quantization import MetalQuantize

        quantizer = self._make_quantizer()
        op = MetalQuantize(quantizer)
        weight = torch.randn(64, 256)
        result = op.convert({"model.layer.weight": weight})
        self.assertIn("model.layer.weight", result)
        self.assertIn("model.layer.scales", result)
        self.assertIn("model.layer.qbiases", result)
        self.assertEqual(result["model.layer.weight"].dtype, torch.uint32)

    def test_metal_quantize_preserves_original_dtype(self):
        from transformers.integrations.metal_quantization import MetalQuantize

        quantizer = self._make_quantizer()
        op = MetalQuantize(quantizer)
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            weight = torch.randn(64, 256, dtype=dtype)
            result = op.convert({"layer.weight": weight})
            self.assertEqual(result["layer.scales"].dtype, dtype, f"scales dtype mismatch for input {dtype}")
            self.assertEqual(result["layer.qbiases"].dtype, dtype, f"qbiases dtype mismatch for input {dtype}")

    def test_metal_dequantize_returns_target_dtype(self):
        """MetalDequantize should return a tensor in the same dtype as the scales."""
        from transformers.integrations.metal_quantization import MetalDequantize, MetalQuantize

        quantizer = self._make_quantizer()

        for dtype in (torch.float16, torch.bfloat16):
            weight = torch.randn(64, 256, dtype=dtype)
            q_op = MetalQuantize(quantizer)
            q_result = q_op.convert({"layer.weight": weight})

            dq_quantizer = self._make_quantizer()
            dq_quantizer.pre_quantized = True
            dq_quantizer.quantization_config.dequantize = True
            dq_op = MetalDequantize(dq_quantizer)

            dq_result = dq_op.convert(
                {
                    "weight$": [q_result["layer.weight"]],
                    "scales": [q_result["layer.scales"]],
                    "qbiases": [q_result["layer.qbiases"]],
                },
                full_layer_name="layer.weight",
            )
            self.assertEqual(
                dq_result["layer.weight"].dtype, dtype, f"dequantized dtype should match scales ({dtype})"
            )

    def test_quantize_then_dequantize_roundtrip(self):
        from transformers.integrations.metal_quantization import MetalDequantize, MetalQuantize

        quantizer = self._make_quantizer(bits=4, group_size=64)
        q_op = MetalQuantize(quantizer)
        weight = torch.randn(64, 256)
        q_result = q_op.convert({"layer.weight": weight})

        dq_quantizer = self._make_quantizer(bits=4, group_size=64)
        dq_op = MetalDequantize(dq_quantizer)
        dq_result = dq_op.convert(
            {
                "weight$": [q_result["layer.weight"]],
                "scales": [q_result["layer.scales"]],
                "qbiases": [q_result["layer.qbiases"]],
            },
            full_layer_name="layer.weight",
        )
        w_deq = dq_result["layer.weight"].float()
        max_err = (weight - w_deq).abs().max().item()
        self.assertLess(max_err, 0.5, "Quantize -> Dequantize round-trip error too large")


@require_torch
class MetalWeightConversionsTest(unittest.TestCase):
    def test_get_weight_conversions_empty_when_not_dequantize(self):
        config = MetalConfig()
        quantizer = MetalHfQuantizer(config)
        quantizer.pre_quantized = True
        self.assertEqual(quantizer.get_weight_conversions(), [])

    def test_get_weight_conversions_has_entry_when_dequantize(self):
        config = MetalConfig(dequantize=True)
        quantizer = MetalHfQuantizer(config)
        quantizer.pre_quantized = True
        conversions = quantizer.get_weight_conversions()
        self.assertEqual(len(conversions), 1)

    def test_get_weight_conversions_empty_when_not_prequantized(self):
        config = MetalConfig(dequantize=True)
        quantizer = MetalHfQuantizer(config)
        quantizer.pre_quantized = False
        self.assertEqual(quantizer.get_weight_conversions(), [])


@require_torch
class MetalModelConversionTest(unittest.TestCase):
    """Test that a model is correctly converted on the meta device."""

    def setUp(self):
        gc.collect()

    def tearDown(self):
        gc.collect()

    def test_quantized_model_conversion(self):
        from transformers.integrations.metal_quantization import MetalLinear, replace_with_metal_linear

        model_id = "hf-internal-testing/tiny-random-OPTForCausalLM"
        config = AutoConfig.from_pretrained(model_id)
        quantization_config = MetalConfig(bits=4, group_size=64)

        with torch.device("meta"):
            model = OPTForCausalLM(config)

        nb_linears = sum(1 for m in model.modules() if isinstance(m, nn.Linear))
        model = replace_with_metal_linear(model, quantization_config=quantization_config, pre_quantized=True)
        nb_metal = sum(1 for m in model.modules() if isinstance(m, MetalLinear))
        self.assertEqual(nb_linears, nb_metal)

    def test_quantized_model_conversion_with_exclusion(self):
        from transformers.integrations.metal_quantization import MetalLinear, replace_with_metal_linear

        model_id = "hf-internal-testing/tiny-random-OPTForCausalLM"
        config = AutoConfig.from_pretrained(model_id)
        quantization_config = MetalConfig(bits=4, group_size=64)

        with torch.device("meta"):
            model = OPTForCausalLM(config)

        nb_linears = sum(1 for m in model.modules() if isinstance(m, nn.Linear))
        model = replace_with_metal_linear(
            model, modules_to_not_convert=["out_proj"], quantization_config=quantization_config, pre_quantized=True
        )
        nb_metal = sum(1 for m in model.modules() if isinstance(m, MetalLinear))
        nb_excluded = sum(1 for name, m in model.named_modules() if "out_proj" in name and isinstance(m, nn.Linear))
        self.assertEqual(nb_metal + nb_excluded, nb_linears)

    def test_param_needs_quantization(self):
        from transformers.integrations.metal_quantization import MetalLinear, replace_with_metal_linear

        model_id = "hf-internal-testing/tiny-random-OPTForCausalLM"
        config = AutoConfig.from_pretrained(model_id)
        quantization_config = MetalConfig(bits=4, group_size=64)

        with torch.device("meta"):
            model = OPTForCausalLM(config)

        replace_with_metal_linear(model, quantization_config=quantization_config, pre_quantized=False)

        quantizer = MetalHfQuantizer(quantization_config)
        quantizer.pre_quantized = False

        for name, module in model.named_modules():
            if isinstance(module, MetalLinear):
                self.assertTrue(quantizer.param_needs_quantization(model, f"{name}.weight"))
                self.assertFalse(quantizer.param_needs_quantization(model, f"{name}.scales"))
                self.assertFalse(quantizer.param_needs_quantization(model, f"{name}.qbiases"))

    def test_param_needs_quantization_prequantized_is_false(self):
        from transformers.integrations.metal_quantization import MetalLinear, replace_with_metal_linear

        model_id = "hf-internal-testing/tiny-random-OPTForCausalLM"
        config = AutoConfig.from_pretrained(model_id)
        quantization_config = MetalConfig(bits=4, group_size=64)

        with torch.device("meta"):
            model = OPTForCausalLM(config)

        replace_with_metal_linear(model, quantization_config=quantization_config, pre_quantized=True)

        quantizer = MetalHfQuantizer(quantization_config)
        quantizer.pre_quantized = True

        for name, module in model.named_modules():
            if isinstance(module, MetalLinear):
                self.assertFalse(
                    quantizer.param_needs_quantization(model, f"{name}.weight"),
                    "Pre-quantized weights should not be re-quantized",
                )


@slow
@require_torch
class MetalSlowIntegrationTest(unittest.TestCase):
    """Slow tests that actually load a model with Metal quantization.

    These run on CPU with ``dequantize=True`` so they don't require MPS.
    """

    model_id = "medmekk/Llama-3.2-1B-Instruct-metal"

    def setUp(self):
        gc.collect()

    def tearDown(self):
        gc.collect()

    def test_load_prequantized_dequantize_on_cpu(self):
        """Load a quantized checkpoint with dequantize=True on CPU and run a forward pass."""
        with _patch_no_mps():
            config = MetalConfig(dequantize=True)
            model = AutoModelForCausalLM.from_pretrained(self.model_id, quantization_config=config, device_map="cpu")
            self.assertIsNotNone(model)
            for param in model.parameters():
                self.assertNotEqual(param.dtype, torch.uint32, "All weights should be dequantized")

    def test_quantized_model(self):
        with _patch_no_mps():
            config = MetalConfig(bits=4, group_size=64)
            model = AutoModelForCausalLM.from_pretrained(self.model_id, quantization_config=config, device_map="mps")
            tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            self.assertIsNotNone(model)
            input = "Hello, how are you?"
            EXPECTED_OUTPUT = "Hello, how are you? I'm doing well, thanks for asking. I"
            input_ids = tokenizer.encode(input, return_tensors="pt").to("mps")
            output = model.generate(input_ids, max_new_tokens=10, do_sample=False)
            self.assertEqual(tokenizer.decode(output[0], skip_special_tokens=True), EXPECTED_OUTPUT)
