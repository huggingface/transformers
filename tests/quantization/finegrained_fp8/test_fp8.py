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

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, FineGrainedFP8Config, OPTForCausalLM
from transformers.quantizers.quantizer_finegrained_fp8 import FineGrainedFP8HfQuantizer
from transformers.testing_utils import (
    backend_empty_cache,
    get_device_properties,
    require_accelerate,
    require_torch_accelerator,
    require_torch_multi_accelerator,
    slow,
    torch_device,
)
from transformers.utils import is_torch_available


if is_torch_available():
    import torch


@contextmanager
def _patch_no_accelerator():
    with ExitStack() as stack:
        stack.enter_context(patch("torch.cuda.is_available", return_value=False))
        if hasattr(torch, "xpu"):
            stack.enter_context(patch("torch.xpu.is_available", return_value=False))
            stack.enter_context(
                patch("transformers.quantizers.quantizer_finegrained_fp8.is_torch_xpu_available", return_value=False)
            )
        yield


@require_torch_accelerator
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
@require_torch_accelerator
@unittest.skipIf(
    get_device_properties()[0] == "cuda"
    and (get_device_properties()[1] < 8 or (get_device_properties()[1] == 8 and get_device_properties()[2] < 9)),
    "Skipping FP8QuantizerTest because it is not supported on GPU with capability < 8.9",
)
class FP8QuantizerTest(unittest.TestCase):
    model_name = "meta-llama/Llama-3.2-1B"
    quantized_model_name = "hf-internal-testing/Llama-3.2-1B-Instruct-fp8"
    input_text = "Once upon a time"
    max_new_tokens = 10
    EXPECTED_OUTPUTS = {
        "Once upon a time, there was a little girl who loved to play",
        "Once upon a time, there was a man who was very rich.",
    }
    EXPECTED_DEQUANTIZED_OUTPUT = "Once upon a time, in a small village nestled in the rolling hills"
    device_map = torch_device
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
        "model.rotary_emb": "cpu",
        "model.norm": "cpu",
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

    def setup(self):
        """
        Clear also on each setup (e.g. if a different model is used than the base cls one)
        """
        gc.collect()
        backend_empty_cache(torch_device)
        gc.collect()

    def tearDown(self):
        gc.collect()
        backend_empty_cache(torch_device)
        gc.collect()

    def test_quantized_model_conversion(self):
        """
        Simple test that checks if the quantized model has been converted properly
        """

        from transformers.integrations import FP8Linear, replace_with_fp8_linear

        model_id = "facebook/opt-350m"
        config = AutoConfig.from_pretrained(model_id, revision="cb32f77e905cccbca1d970436fb0f5e6b58ee3c5")
        quantization_config = FineGrainedFP8Config()

        with torch.device("meta"):
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
        self.assertEqual(nb_linears, nb_fp8_linear)
        with torch.device("meta"):
            model = OPTForCausalLM(config)
        quantization_config = FineGrainedFP8Config()
        model = replace_with_fp8_linear(model, modules_to_not_convert=["fc1"], quantization_config=quantization_config)
        nb_fp8_linear = 0
        for module in model.modules():
            if isinstance(module, FP8Linear):
                nb_fp8_linear += 1
        self.assertEqual(nb_linears - 24, nb_fp8_linear)

    def test_quantizer_validation_no_accelerator(self):
        """Test quantizer validation when CUDA/XPU is not available"""
        with _patch_no_accelerator():
            config = FineGrainedFP8Config()
            quantizer = FineGrainedFP8HfQuantizer(config)
            quantizer.pre_quantized = False

            with self.assertRaises(RuntimeError):
                quantizer.validate_environment()

    def test_dequantization_no_accelerator(self):
        """Test dequantization when CUDA/XPU is not available"""
        with _patch_no_accelerator():
            config = FineGrainedFP8Config()
            quantizer = FineGrainedFP8HfQuantizer(config)
            quantizer.pre_quantized = True
            quantizer.validate_environment()
            self.assertTrue(quantizer.quantization_config.dequantize)

    def test_quantized_model(self):
        """
        Simple test that checks if the quantized model is working properly
        """
        input_ids = self.tokenizer(self.input_text, return_tensors="pt").to(self.device_map)

        output = self.quantized_model.generate(**input_ids, max_new_tokens=self.max_new_tokens, do_sample=False)
        output_tokens = self.tokenizer.decode(output[0], skip_special_tokens=True)
        self.assertIn(output_tokens, self.EXPECTED_OUTPUTS)

    def test_dequantized_model(self):
        """
        Simple test that checks if the dequantized model is working properly
        """
        quantization_config = FineGrainedFP8Config(dequantize=True)
        dequantized_model = AutoModelForCausalLM.from_pretrained(
            self.quantized_model_name, device_map=self.device_map, quantization_config=quantization_config
        )
        input_ids = self.tokenizer(self.input_text, return_tensors="pt").to(self.device_map)
        output = dequantized_model.generate(**input_ids, max_new_tokens=self.max_new_tokens, do_sample=False)
        output_tokens = self.tokenizer.decode(output[0], skip_special_tokens=True)
        self.assertEqual(output_tokens, self.EXPECTED_DEQUANTIZED_OUTPUT)
        del dequantized_model

    def test_dequantize_when_no_accelerator(self):
        """
        Simple test that checks if the dequantized model is working properly when no accelerator is available
        """
        with _patch_no_accelerator():
            dequantized_model = AutoModelForCausalLM.from_pretrained(self.quantized_model_name, device_map="cpu")
            input_ids = self.tokenizer(self.input_text, return_tensors="pt").to("cpu")
            output = dequantized_model.generate(**input_ids, max_new_tokens=self.max_new_tokens, do_sample=False)
            output_tokens = self.tokenizer.decode(output[0], skip_special_tokens=True)
            self.assertEqual(output_tokens, self.EXPECTED_DEQUANTIZED_OUTPUT)
            del dequantized_model

    def test_save_pretrained(self):
        """
        Simple test that checks if the quantized model is working properly after being saved and loaded
        """
        with tempfile.TemporaryDirectory() as tmpdirname:
            self.quantized_model.save_pretrained(tmpdirname)

            model = AutoModelForCausalLM.from_pretrained(tmpdirname, device_map=self.device_map)

            input_ids = self.tokenizer(self.input_text, return_tensors="pt").to(self.device_map)

            output = model.generate(**input_ids, max_new_tokens=self.max_new_tokens, do_sample=False)
            self.assertIn(self.tokenizer.decode(output[0], skip_special_tokens=True), self.EXPECTED_OUTPUTS)

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

    @require_torch_multi_accelerator
    def test_quantized_model_multi_accelerators(self):
        """
        Simple test that checks if the quantized model is working properly with multiple accelerators
        set CUDA_VISIBLE_DEVICES=0,1 if you have more than 2 GPUs; or set ZE_AFFINITY_MASK=0,1 if you
        have more than 2 XPUs.
        """
        input_ids = self.tokenizer(self.input_text, return_tensors="pt").to(self.device_map)
        quantization_config = FineGrainedFP8Config()
        # need to empty cache or set max_memory, otherwise we will use the reserved memory that was not allocated when computing max-memory
        # this will lead to put the entire model to device 0.
        quantized_model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map="auto",
            quantization_config=quantization_config,
            max_memory={0: "1GB", 1: "10GB"},
        )
        self.assertTrue(set(quantized_model.hf_device_map.values()) == {0, 1})

        output = quantized_model.generate(**input_ids, max_new_tokens=self.max_new_tokens, do_sample=False)
        self.assertIn(self.tokenizer.decode(output[0], skip_special_tokens=True), self.EXPECTED_OUTPUTS)

    @require_torch_multi_accelerator
    def test_save_pretrained_multi_accelerators(self):
        """
        Simple test that checks if the quantized model is working properly after being saved and loaded
        """
        with tempfile.TemporaryDirectory() as tmpdirname:
            self.quantized_model.save_pretrained(tmpdirname)
            # need to empty cache or set max_memory, otherwise we will use the reserved memory that was not allocated when computing max-memory
            # this will lead to put the entire model to device 0.
            model = AutoModelForCausalLM.from_pretrained(
                tmpdirname, device_map="auto", max_memory={0: "1GB", 1: "10GB"}
            )
            self.assertTrue(set(model.hf_device_map.values()) == {0, 1})

            input_ids = self.tokenizer(self.input_text, return_tensors="pt").to(self.device_map)

            output = model.generate(**input_ids, max_new_tokens=self.max_new_tokens, do_sample=False)
            self.assertIn(self.tokenizer.decode(output[0], skip_special_tokens=True), self.EXPECTED_OUTPUTS)

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
            self.assertIn(self.tokenizer.decode(output[0], skip_special_tokens=True), self.EXPECTED_OUTPUTS)

    def test_compute_module_sizes(self):
        r"""
        Test if we compute the right module sizes needed to generate the device map.
        Also test if we get the right values for `total_byte_count` in `caching_allocator_warmup`.
        """
        from transformers.integrations import FP8Linear
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
            hf_quantizer = AutoHfQuantizer.from_config(FineGrainedFP8Config(), pre_quantized=False)
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
            if isinstance(module, FP8Linear):
                # from 16 bits to 8 bits
                assert int(model_size[f"{name}.weight"] // 2) == int(quantized_model_size[f"{name}.weight"])

        # check that we get the same value, as we use `compute_module_sizes` in `get_total_byte_count`
        assert total_byte_count == model_size[""]
        assert quantized_total_byte_count == quantized_model_size[""]

        # we should at least have 1.5 times memory reduction in total
        assert model_size[""] > quantized_model_size[""] * 1.5

    def test_quantized_moe_forward(self):
        """
        Checks implicitly if the moe implementation is correct, i.e. it does not crash for cases
        where the indices go over `top_k` as shown within the Minimax M2 model
        """
        model = AutoModelForCausalLM.from_pretrained(
            "hf-internal-testing/MiniMax-M2-Tiny-FP8",  # single layer version
            device_map=self.device_map,
        )

        tokenizer = AutoTokenizer.from_pretrained("MiniMaxAI/MiniMax-M2")
        messages = [
            {"role": "user", "content": [{"type": "text", "text": "What is your favourite condiment?"}]},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": "Well, I'm quite partial to a good squeeze of fresh lemon juice. It adds just the right amount of zesty flavour to whatever I'm cooking up in the kitchen!",
                    }
                ],
            },
            {"role": "user", "content": [{"type": "text", "text": "Do you have mayonnaise recipes?"}]},
        ]
        model_inputs = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True).to(
            self.device_map
        )

        # Only caring about this not crashing
        _ = model.generate(**model_inputs, max_new_tokens=24)


@require_torch_accelerator
@unittest.skipIf(
    get_device_properties()[0] == "cuda"
    and (get_device_properties()[1] < 8 or (get_device_properties()[1] == 8 and get_device_properties()[2] < 9)),
    "Skipping FP8LinearTest because it is not supported on GPU with capability < 8.9",
)
class FP8LinearTest(unittest.TestCase):
    device = torch_device

    def test_linear_preserves_shape(self):
        """
        Test that FP8Linear preserves shape when in_features == out_features.
        """
        from transformers.integrations import FP8Linear

        linear = FP8Linear(256, 256, block_size=(128, 128)).to(self.device)
        x = torch.rand((1, 5, 256)).to(self.device)

        x_ = linear(x)
        self.assertEqual(x_.shape, x.shape)

    def test_linear_with_diff_feature_size_preserves_shape(self):
        """
        Test that FP8Linear generates the correct shape when in_features != out_features.
        """
        from transformers.integrations import FP8Linear

        linear = FP8Linear(128, 256, block_size=(128, 128)).to(self.device)
        x = torch.rand((1, 5, 128)).to(self.device)

        x_ = linear(x)
        self.assertEqual(x_.shape, (1, 5, 256))
