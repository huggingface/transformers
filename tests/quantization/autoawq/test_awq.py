# coding=utf-8
# Copyright 2023 The HuggingFace Team. All rights reserved.
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

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, AwqConfig, OPTForCausalLM
from transformers.testing_utils import (
    require_accelerate,
    require_auto_awq,
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


@require_torch_gpu
class AwqConfigTest(unittest.TestCase):
    def test_wrong_backend(self):
        """
        Simple test that checks if a user passes a wrong backend an error is raised
        """
        # This should work fine
        _ = AwqConfig(bits=4)

        with self.assertRaises(ValueError):
            AwqConfig(bits=4, backend="")

        # These should work fine
        _ = AwqConfig(bits=4, version="GEMM")
        _ = AwqConfig(bits=4, version="gemm")

        with self.assertRaises(ValueError):
            AwqConfig(bits=4, backend="unexisting-backend")

        # LLMAWQ does not work on a T4
        with self.assertRaises(ValueError):
            AwqConfig(bits=4, backend="llm-awq")

    def test_to_dict(self):
        """
        Simple test that checks if one uses a config and converts it to a dict, the dict is the same as the config object
        """
        quantization_config = AwqConfig(bits=4)
        config_to_dict = quantization_config.to_dict()

        for key in config_to_dict:
            self.assertEqual(getattr(quantization_config, key), config_to_dict[key])

    def test_from_dict(self):
        """
        Simple test that checks if one uses a dict and converts it to a config object, the config object is the same as the dict
        """
        dict = {"bits": 2, "zero_point": False, "backend": "autoawq"}
        quantization_config = AwqConfig.from_dict(dict)

        self.assertEqual(dict["bits"], quantization_config.bits)
        self.assertEqual(dict["zero_point"], quantization_config.zero_point)
        self.assertEqual(dict["backend"], quantization_config.backend)


@slow
@require_torch_gpu
@require_auto_awq
@require_accelerate
class AwqTest(unittest.TestCase):
    model_name = "TheBloke/Mistral-7B-v0.1-AWQ"
    dummy_transformers_model_name = "bigscience/bloom-560m"
    model_with_no_k_proj_quantized = "hf-internal-testing/opt-125m-awq-no-k-proj"

    input_text = "Hello my name is"

    EXPECTED_OUTPUT = "Hello my name is Katie and I am a 20 year old student at the University of North Carolina at Chapel Hill. I am a junior and I am majoring in Journalism and minoring in Spanish"
    EXPECTED_OUTPUT_BF16 = "Hello my name is Katie and I am a 20 year old student at the University of North Carolina at Chapel Hill. I am a junior and I am majoring in Exercise and Sport Science with a"

    device_map = "cuda"

    # called only once for all test in this class
    @classmethod
    def setUpClass(cls):
        """
        Setup quantized model
        """
        cls.tokenizer = AutoTokenizer.from_pretrained(cls.model_name)
        cls.quantized_model = AutoModelForCausalLM.from_pretrained(
            cls.model_name,
            device_map=cls.device_map,
        )

    def tearDown(self):
        gc.collect()
        torch.cuda.empty_cache()
        gc.collect()

    def test_quantized_model_conversion(self):
        """
        Simple test that checks if the quantized model has been converted properly
        """
        from awq.modules.linear import WQLinear_GEMM, WQLinear_GEMV

        from transformers.integrations.awq import replace_with_awq_linear

        model_id = "facebook/opt-350m"
        config = AutoConfig.from_pretrained(model_id, revision="cb32f77e905cccbca1d970436fb0f5e6b58ee3c5")
        quantization_config = AwqConfig(bits=4)

        with init_empty_weights():
            model = OPTForCausalLM(config)

        nb_linears = 0
        for module in model.modules():
            if isinstance(module, torch.nn.Linear):
                nb_linears += 1

        model, _ = replace_with_awq_linear(model, quantization_config=quantization_config)
        nb_awq_linear = 0
        for module in model.modules():
            if isinstance(module, (WQLinear_GEMM, WQLinear_GEMV)):
                nb_awq_linear += 1

        self.assertEqual(nb_linears, nb_awq_linear)

        # Try with `modules_not_to_convert`
        with init_empty_weights():
            model = OPTForCausalLM(config)

        model, _ = replace_with_awq_linear(
            model, quantization_config=quantization_config, modules_to_not_convert=["lm_head"]
        )
        nb_awq_linear = 0
        for module in model.modules():
            if isinstance(module, (WQLinear_GEMM, WQLinear_GEMV)):
                nb_awq_linear += 1

        self.assertEqual(nb_linears - 1, nb_awq_linear)

    def test_quantized_model(self):
        """
        Simple test that checks if the quantized model is working properly
        """
        input_ids = self.tokenizer(self.input_text, return_tensors="pt").to(torch_device)

        output = self.quantized_model.generate(**input_ids, max_new_tokens=40)
        self.assertEqual(self.tokenizer.decode(output[0], skip_special_tokens=True), self.EXPECTED_OUTPUT)

    def test_raise_if_non_quantized(self):
        model_id = "facebook/opt-125m"
        quantization_config = AwqConfig(bits=4)

        with self.assertRaises(ValueError):
            _ = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=quantization_config)

    def test_quantized_model_bf16(self):
        """
        Simple test that checks if the quantized model is working properly with bf16
        """
        input_ids = self.tokenizer(self.input_text, return_tensors="pt").to(torch_device)

        quantized_model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype=torch.bfloat16).to(
            torch_device
        )

        output = quantized_model.generate(**input_ids, max_new_tokens=40)
        self.assertEqual(self.tokenizer.decode(output[0], skip_special_tokens=True), self.EXPECTED_OUTPUT_BF16)

    def test_quantized_model_no_device_map(self):
        """
        Simple test that checks if the quantized model is working properly
        """
        input_ids = self.tokenizer(self.input_text, return_tensors="pt").to(torch_device)

        quantized_model = AutoModelForCausalLM.from_pretrained(self.model_name).to(torch_device)
        output = quantized_model.generate(**input_ids, max_new_tokens=40)

        self.assertEqual(self.tokenizer.decode(output[0], skip_special_tokens=True), self.EXPECTED_OUTPUT)

    def test_save_pretrained(self):
        """
        Simple test that checks if the quantized model is working properly after being saved and loaded
        """
        with tempfile.TemporaryDirectory() as tmpdirname:
            self.quantized_model.save_pretrained(tmpdirname)
            model = AutoModelForCausalLM.from_pretrained(tmpdirname, device_map=self.device_map)

            input_ids = self.tokenizer(self.input_text, return_tensors="pt").to(torch_device)

            output = model.generate(**input_ids, max_new_tokens=40)
            self.assertEqual(self.tokenizer.decode(output[0], skip_special_tokens=True), self.EXPECTED_OUTPUT)

    @require_torch_multi_gpu
    def test_quantized_model_multi_gpu(self):
        """
        Simple test that checks if the quantized model is working properly with multiple GPUs
        """
        input_ids = self.tokenizer(self.input_text, return_tensors="pt").to(torch_device)

        quantized_model = AutoModelForCausalLM.from_pretrained(self.model_name, device_map="auto")

        self.assertTrue(set(quantized_model.hf_device_map.values()) == {0, 1, 2, 3})

        output = quantized_model.generate(**input_ids, max_new_tokens=40)

        self.assertEqual(self.tokenizer.decode(output[0], skip_special_tokens=True), self.EXPECTED_OUTPUT)

    def test_quantized_model_no_k_proj_quantized(self):
        """
        Simple test that checks if the quantized model is working properly with multiple GPUs
        """
        dummy_input = torch.LongTensor([[0, 1, 0]]).to(torch_device)

        quantized_model = AutoModelForCausalLM.from_pretrained(self.model_with_no_k_proj_quantized).to(torch_device)

        self.assertTrue(isinstance(quantized_model.model.decoder.layers[0].self_attn.k_proj, torch.nn.Linear))
        self.assertFalse(isinstance(quantized_model.model.decoder.layers[0].self_attn.v_proj, torch.nn.Linear))

        EXPECTED_OUTPUT = torch.LongTensor([[0, 1, 0, 50118, 50118, 133, 248, 12, 134, 16, 10, 372, 2031]]).to(
            torch_device
        )

        output = quantized_model.generate(dummy_input, max_new_tokens=10)
        self.assertTrue((EXPECTED_OUTPUT == output).all())


@slow
@require_torch_gpu
@require_auto_awq
@require_accelerate
class AwqFusedTest(unittest.TestCase):
    model_name = "TheBloke/Mistral-7B-OpenOrca-AWQ"
    model_revision = "7048b2af77d0dd1c81b000b19d73f9cc8950b510"

    custom_mapping_model_id = "TheBloke/Yi-34B-AWQ"
    custom_model_revision = "f1b2cd1b7459ceecfdc1fac5bb8725f13707c589"

    prompt = (
        "You're standing on the surface of the Earth. "
        "You walk one mile south, one mile west and one mile north. "
        "You end up exactly where you started. Where are you?"
    )

    EXPECTED_GENERATION = prompt + "\n\nThis is a classic puzzle that has been around for"
    EXPECTED_GENERATION_CUSTOM_MODEL = "HelloWorld.java:11)\r\n\tat org"

    def tearDown(self):
        gc.collect()
        torch.cuda.empty_cache()
        gc.collect()

    def _check_fused_modules(self, model):
        has_fused_modules = False
        fused_modules_name = ["QuantAttentionFused", "QuantFusedMLP", "FasterTransformerRMSNorm"]

        for _, module in model.named_modules():
            if module.__class__.__name__ in fused_modules_name:
                has_fused_modules = True
                break

        self.assertTrue(has_fused_modules, "Modules fusing not performed correctly!")

    def test_raise_save_pretrained(self):
        """
        Test that `save_pretrained` is effectively blocked for fused models
        """
        quantization_config = AwqConfig(bits=4, fuse_max_seq_len=128, do_fuse=True)

        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=quantization_config,
            low_cpu_mem_usage=True,
            revision=self.model_revision,
        ).to(torch_device)

        self._check_fused_modules(model)

        with self.assertRaises(ValueError), tempfile.TemporaryDirectory() as tmpdirname:
            model.save_pretrained(tmpdirname)

    def test_generation_fused(self):
        """
        Test generation quality for fused models - single batch case
        """
        quantization_config = AwqConfig(bits=4, fuse_max_seq_len=128, do_fuse=True)

        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=quantization_config,
            low_cpu_mem_usage=True,
            revision=self.model_revision,
        ).to(torch_device)

        self._check_fused_modules(model)

        tokenizer = AutoTokenizer.from_pretrained(self.model_name, revision=self.model_revision)

        inputs = tokenizer(self.prompt, return_tensors="pt").to(torch_device)

        outputs = model.generate(**inputs, max_new_tokens=12)

        self.assertEqual(tokenizer.decode(outputs[0], skip_special_tokens=True), self.EXPECTED_GENERATION)

    def test_generation_fused_batched(self):
        """
        Test generation quality for fused models - multi batch case
        """
        quantization_config = AwqConfig(bits=4, fuse_max_seq_len=128, do_fuse=True)

        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=quantization_config,
            low_cpu_mem_usage=True,
            revision=self.model_revision,
        ).to(torch_device)

        self._check_fused_modules(model)

        tokenizer = AutoTokenizer.from_pretrained(self.model_name, revision=self.model_revision)

        tokenizer.pad_token_id = tokenizer.eos_token_id
        inputs = tokenizer([self.prompt, self.prompt], return_tensors="pt", padding=True).to(torch_device)

        outputs = model.generate(**inputs, max_new_tokens=12)

        self.assertEqual(tokenizer.decode(outputs[0], skip_special_tokens=True), self.EXPECTED_GENERATION)

    @require_torch_multi_gpu
    def test_generation_custom_model(self):
        """
        Test generation quality for fused models using custom fused map.
        """
        quantization_config = AwqConfig(
            bits=4,
            fuse_max_seq_len=512,
            modules_to_fuse={
                "attention": ["q_proj", "k_proj", "v_proj", "o_proj"],
                "layernorm": ["ln1", "ln2", "norm"],
                "mlp": ["gate_proj", "up_proj", "down_proj"],
                "use_alibi": False,
                "num_attention_heads": 56,
                "num_key_value_heads": 8,
                "hidden_size": 7168,
            },
        )

        model = AutoModelForCausalLM.from_pretrained(
            self.custom_mapping_model_id,
            quantization_config=quantization_config,
            trust_remote_code=True,
            device_map="balanced",
            revision=self.custom_model_revision,
        )

        self._check_fused_modules(model)

        tokenizer = AutoTokenizer.from_pretrained(
            self.custom_mapping_model_id, revision=self.custom_model_revision, trust_remote_code=True
        )

        prompt = "Hello"
        inputs = tokenizer(prompt, return_tensors="pt").to(torch_device)

        outputs = model.generate(**inputs, max_new_tokens=12)
        self.assertEqual(tokenizer.decode(outputs[0], skip_special_tokens=True), self.EXPECTED_GENERATION_CUSTOM_MODEL)
