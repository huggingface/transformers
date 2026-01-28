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
from contextlib import ExitStack, contextmanager
from unittest.mock import patch

from transformers import AutoTokenizer, GptOssForCausalLM, Mxfp4Config
from transformers.testing_utils import (
    require_kernels,
    require_torch,
    require_torch_gpu,
    require_torch_large_accelerator,
    require_triton,
    slow,
    torch_device,
)
from transformers.utils import (
    is_torch_available,
)


if is_torch_available():
    import torch


if torch.cuda.is_available():
    REQUIRE_TRITON_MXFP4 = require_triton(min_version="3.4.0")
elif hasattr(torch, "xpu") and torch.xpu.is_available():
    REQUIRE_TRITON_MXFP4 = require_triton(min_version="3.5.0")
elif torch_device == "cpu":
    REQUIRE_TRITON_MXFP4 = require_triton(min_version="3.5.0")
else:
    REQUIRE_TRITON_MXFP4 = unittest.skip("test requires CUDA or XPU")


def _empty_accelerator_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        torch.xpu.empty_cache()


@contextmanager
def _patch_no_accelerator():
    with ExitStack() as stack:
        stack.enter_context(patch("torch.cuda.is_available", return_value=False))
        if hasattr(torch, "xpu"):
            stack.enter_context(patch("torch.xpu.is_available", return_value=False))
        yield


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
        self.assertEqual(attrs["dequantize"], True)

    def test_to_dict(self):
        """Test configuration serialization to dict"""
        config = Mxfp4Config(modules_to_not_convert=["lm_head"], dequantize=True)
        config_dict = config.to_dict()
        self.assertEqual(config_dict["quant_method"], "mxfp4")
        self.assertEqual(config_dict["modules_to_not_convert"], ["lm_head"])
        # we don't keep dequantize in config_dict
        self.assertTrue("dequantize" not in config_dict)

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
        _empty_accelerator_cache()

    def test_quantizer_validation_no_torch(self):
        """Test quantizer validation when torch is not available"""
        with patch("transformers.quantizers.quantizer_mxfp4.is_torch_available", return_value=False):
            from transformers.quantizers.quantizer_mxfp4 import Mxfp4HfQuantizer

            config = Mxfp4Config()
            quantizer = Mxfp4HfQuantizer(config)

            with self.assertRaises(ImportError):
                quantizer.validate_environment()

    def test_quantizer_validation_no_accelerator(self):
        """Test quantizer validation when CUDA/XPU is not available"""
        with _patch_no_accelerator():
            from transformers.quantizers.quantizer_mxfp4 import Mxfp4HfQuantizer

            config = Mxfp4Config()
            quantizer = Mxfp4HfQuantizer(config)
            quantizer.pre_quantized = False
            # CPU already supported MXFP4
            quantizer.validate_environment()

    @require_torch_gpu
    def test_quantizer_validation_low_compute_capability(self):
        """Test quantizer validation with CUDA low compute capability"""
        with patch("torch.cuda.get_device_capability", return_value=(7, 0)):
            from transformers.quantizers.quantizer_mxfp4 import Mxfp4HfQuantizer

            config = Mxfp4Config()
            quantizer = Mxfp4HfQuantizer(config)
            quantizer.pre_quantized = False

            with self.assertRaises(ValueError):
                quantizer.validate_environment()

    @require_torch_gpu
    def test_quantizer_validation_low_compute_capability_with_prequantized(self):
        """Test quantizer validation with CUDA low compute capability"""
        with patch("torch.cuda.get_device_capability", return_value=(7, 0)):
            from transformers.quantizers.quantizer_mxfp4 import Mxfp4HfQuantizer

            config = Mxfp4Config()
            quantizer = Mxfp4HfQuantizer(config)

            # Should automatically set dequantize=True and warn
            quantizer.validate_environment()
            self.assertTrue(quantizer.quantization_config.dequantize)

    @require_torch_gpu
    def test_quantizer_validation_low_compute_capability_with_dequantize(self):
        """Test quantizer validation with CUDA low compute capability but dequantize enabled"""
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

    def test_quantizer_validation_order_dequantize_before_accelerator_check(self):
        """Test that dequantize check happens before CUDA/XPU availability check"""
        # Mock torch.cuda.is_available
        with _patch_no_accelerator():
            from transformers.quantizers.quantizer_mxfp4 import Mxfp4HfQuantizer

            # Test with dequantize=True - should pass even without CUDA/XPU and accelerate
            config = Mxfp4Config(dequantize=True)
            quantizer = Mxfp4HfQuantizer(config)

            # This should not raise any error because dequantize check comes first
            quantizer.validate_environment()

            # Test with dequantize=False - should still fail due to missing CUDA/XPU
            config = Mxfp4Config(dequantize=False)
            quantizer = Mxfp4HfQuantizer(config)
            quantizer.pre_quantized = False

            # CPU already supported MXFP4
            quantizer.validate_environment()

    def test_quantizer_validation_missing_triton(self):
        """Test quantizer validation when triton is not available"""
        with (
            patch("transformers.quantizers.quantizer_mxfp4.is_triton_available", return_value=False),
            patch("transformers.quantizers.quantizer_mxfp4.is_kernels_available", return_value=False),
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
            patch("transformers.quantizers.quantizer_mxfp4.is_kernels_available", return_value=False),
        ):
            from transformers.quantizers.quantizer_mxfp4 import Mxfp4HfQuantizer

            config = Mxfp4Config()
            quantizer = Mxfp4HfQuantizer(config)
            quantizer.pre_quantized = True

            # Should automatically set dequantize=True and warn
            quantizer.validate_environment()
            self.assertTrue(quantizer.quantization_config.dequantize)

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
        from transformers.quantizers.quantizers_utils import should_convert_module

        # Should convert by default
        self.assertTrue(should_convert_module("model", None))
        self.assertTrue(should_convert_module("model", []))

        # Should not convert if in exclusion list
        patterns = ["model.layers.*.self_attn", "lm_head"]
        self.assertFalse(should_convert_module("lm_head", patterns))
        self.assertTrue(should_convert_module("experts", patterns))

    @require_torch
    def test_convert_moe_packed_tensors(self):
        """Test unpacking of quantized tensors"""
        from transformers.integrations.mxfp4 import convert_moe_packed_tensors

        # Create dummy packed tensors
        blocks = torch.randint(0, 255, (2, 4, 8, 16), dtype=torch.uint8)
        scales = torch.randint(100, 150, (2, 4, 8), dtype=torch.uint8)

        result = convert_moe_packed_tensors(blocks, scales, dtype=torch.bfloat16)
        self.assertEqual(result.shape, (2, 8 * 16 * 2, 4))
        self.assertEqual(result.dtype, torch.bfloat16)

    @REQUIRE_TRITON_MXFP4
    @require_kernels
    @require_torch
    def test_quantize_to_mxfp4(self):
        """Test quantization function"""
        from transformers.integrations.mxfp4 import quantize_to_mxfp4
        from transformers.quantizers.quantizer_mxfp4 import Mxfp4HfQuantizer

        config = Mxfp4Config()
        quantizer = Mxfp4HfQuantizer(config)

        # Create dummy weight tensor
        device = torch_device
        w = torch.randn(32, 64, 128, dtype=torch.bfloat16, device=torch.device(device))

        quantized_w, w_scale = quantize_to_mxfp4(w, quantizer._lazy_import_kernels())

        # Check that shapes are reasonable
        self.assertEqual(quantized_w.dtype, torch.uint8)


@require_torch
@require_torch_large_accelerator
@REQUIRE_TRITON_MXFP4
@require_kernels
@slow
class Mxfp4ModelTest(unittest.TestCase):
    """Test mxfp4 with actual models (requires specific model and hardware)"""

    # These should be paths to real OpenAI MoE models for proper testing
    model_name = "openai/gpt-oss-20b"

    input_text = "Once upon a time"

    # Expected outputs for generation tests
    EXPECTED_OUTPUTS = set()
    EXPECTED_OUTPUTS.add("Once upon a time, in a small town, there lived a young")

    def setUp(self):
        gc.collect()
        _empty_accelerator_cache()

    def tearDown(self):
        gc.collect()
        _empty_accelerator_cache()

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

        model = GptOssForCausalLM.from_pretrained(
            self.model_name,
            dtype=torch.bfloat16,
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
            dtype=torch.bfloat16,
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

        # Test with CPU in device map (CPU already support mxfp4)
        quantizer.validate_environment(device_map={"": "cpu"})

    def test_memory_footprint_comparison(self):
        """Test memory footprint differences between quantized and unquantized models"""

        # Expected: quantized < dequantized < unquantized memory usage
        quantization_config = Mxfp4Config(dequantize=True)
        quantized_model = GptOssForCausalLM.from_pretrained(
            self.model_name,
            dtype=torch.bfloat16,
            device_map="auto",
        )
        dequantized_model = GptOssForCausalLM.from_pretrained(
            self.model_name,
            dtype=torch.bfloat16,
            device_map="auto",
            quantization_config=quantization_config,
        )
        quantized_mem = quantized_model.get_memory_footprint()
        dequantized_mem = dequantized_model.get_memory_footprint()
        self.assertLess(quantized_mem, dequantized_mem)

    def test_save_mxfp4(self):
        """Test saving quantized OpenAI MoE model with device_map"""

        model = GptOssForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        with tempfile.TemporaryDirectory() as tmp:
            # Save the model in mxfp4 format
            model.save_pretrained(tmp)
            _empty_accelerator_cache()
            gc.collect()
            # test quantized model
            loaded_model = GptOssForCausalLM.from_pretrained(
                tmp,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
            self.check_inference_correctness_quantized(loaded_model, tokenizer)

            # test dequantized model
            loaded_model = GptOssForCausalLM.from_pretrained(
                tmp,
                quantization_config=Mxfp4Config(dequantize=True),
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
            self.check_inference_correctness_quantized(loaded_model, tokenizer)

    def test_save_mxfp4_non_quantized(self):
        """Test saving dequantized OpenAI MoE model with mxfp4 quantization and device_map"""
        non_quantized_model_name = "hf-internal-testing/gpt-oss-20b-bf16"
        tokenizer = AutoTokenizer.from_pretrained(non_quantized_model_name)
        loaded_model = GptOssForCausalLM.from_pretrained(
            non_quantized_model_name,
            quantization_config=Mxfp4Config(),
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        # save the quantized model
        with tempfile.TemporaryDirectory() as tmp:
            loaded_model.save_pretrained(tmp)
            _empty_accelerator_cache()
            gc.collect()
            # load it back to check with everything works as expected
            loaded_model = GptOssForCausalLM.from_pretrained(
                tmp,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
            self.check_inference_correctness_quantized(loaded_model, tokenizer)

            loaded_model = GptOssForCausalLM.from_pretrained(
                tmp,
                quantization_config=Mxfp4Config(dequantized=True),
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
            self.check_inference_correctness_quantized(loaded_model, tokenizer)

    def test_compute_module_sizes(self):
        r"""
        Test if we compute the right module sizes needed to generate the device map.
        Also test if we get the right values for `total_byte_count` in `caching_allocator_warmup`.
        """
        from transformers import AutoConfig, AutoModelForCausalLM
        from transformers.integrations import Mxfp4GptOssExperts
        from transformers.integrations.accelerate import compute_module_sizes
        from transformers.modeling_utils import expand_device_map, get_total_byte_count
        from transformers.quantizers import AutoHfQuantizer

        # we need to preprocess the model like that because device_map calculation happens before we load the weights inside the model.
        # For normal wieghts, it's fine but for quantized weights, the tensors dtype might change during loading.
        with torch.device("meta"):
            config = AutoConfig.from_pretrained(self.model_name)
            model = AutoModelForCausalLM.from_config(config, dtype=torch.bfloat16)
            model_size, _ = compute_module_sizes(model, only_modules=False)

            expected_keys = [name for name, _ in model.named_parameters()] + [
                name for name, _ in model.named_buffers()
            ]
            expanded_device_map = expand_device_map({"": torch_device}, expected_keys)
            total_byte_count = list(get_total_byte_count(model, expanded_device_map).values())[0]

            # testing prequantized = False should be enough, the shape should be the same whether it is pre-quantized or not
            hf_quantizer = AutoHfQuantizer.from_config(Mxfp4Config(), pre_quantized=False)
            hf_quantizer.preprocess_model(model=model, config=model.config)
            quantized_model_size, _ = compute_module_sizes(model, hf_quantizer, only_modules=False)

            expected_keys = [name for name, _ in model.named_parameters()] + [
                name for name, _ in model.named_buffers()
            ]
            expanded_device_map = expand_device_map({"": torch_device}, expected_keys)
            quantized_total_byte_count = list(get_total_byte_count(model, expanded_device_map, hf_quantizer).values())[
                0
            ]
        for name, module in model.named_modules():
            if isinstance(module, Mxfp4GptOssExperts):
                # from 16 bits to 4 bits
                assert int(model_size[f"{name}.gate_up_proj"] // 4) == int(
                    quantized_model_size[f"{name}.gate_up_proj"]
                )
                assert int(model_size[f"{name}.down_proj"] // 4) == int(quantized_model_size[f"{name}.down_proj"])

        # check that we get the same value, as we use `compute_module_sizes` in `get_total_byte_count`
        assert total_byte_count == model_size[""]
        assert quantized_total_byte_count == quantized_model_size[""]

        # we should at least have 3 times memory reduction in total for this model
        assert model_size[""] > quantized_model_size[""] * 3
