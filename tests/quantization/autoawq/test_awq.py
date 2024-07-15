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

        compute_capability = torch.cuda.get_device_capability()
        major, minor = compute_capability

        if major < 8:
            # LLMAWQ does not work on a T4
            with self.assertRaises(ValueError):
                AwqConfig(bits=4, backend="llm-awq")
        else:
            # LLMAWQ should work on an A100
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

    EXPECTED_OUTPUT_EXLLAMA = [
        "Hello my name is Katie and I am a 20 year old student from the UK. I am currently studying for a degree in English Literature and History at the University of York. I am a very out",
        "Hello my name is Katie and I am a 20 year old student from the UK. I am currently studying for a degree in English Literature and History at the University of York. I am a very creative",
    ]
    device_map = "cuda"

    # called only once for all test in this class
    @classmethod
    def setUpClass(cls):
        """
        Setup quantized model
        """
        cls.tokenizer = AutoTokenizer.from_pretrained(cls.model_name)
        cls.quantized_model = AutoModelForCausalLM.from_pretrained(cls.model_name, device_map=cls.device_map)

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

    def test_quantized_model_exllama(self):
        """
        Simple test that checks if the quantized model is working properly with exllama backend
        """
        input_ids = self.tokenizer(self.input_text, return_tensors="pt").to(torch_device)

        quantization_config = AwqConfig(version="exllama")
        quantized_model = AutoModelForCausalLM.from_pretrained(
            self.model_name, quantization_config=quantization_config, device_map=torch_device
        )

        output = quantized_model.generate(**input_ids, max_new_tokens=40)
        self.assertIn(self.tokenizer.decode(output[0], skip_special_tokens=True), self.EXPECTED_OUTPUT_EXLLAMA)

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

        self.assertTrue(set(quantized_model.hf_device_map.values()) == {0, 1})

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

    custom_mapping_model_id = "TheBloke/Mistral-7B-v0.1-AWQ"
    custom_model_revision = "f186bcfa9edbe2a4334262ec1e67f23e53ed1ae7"

    mixtral_model_name = "casperhansen/mixtral-instruct-awq"
    mixtral_model_revision = "87dd4ec502dde74fb3a624835c776b000d190c3b"

    multi_modal_model_name = "ybelkada/llava-1.5-7b-hf-awq"
    multi_modal_model_code_revision = "ad108a50f5b9e681bdd7378409f57b7fa59a7442"

    prompt = (
        "You're standing on the surface of the Earth. "
        "You walk one mile south, one mile west and one mile north. "
        "You end up exactly where you started. Where are you?"
    )

    EXPECTED_GENERATION = prompt + "\n\nThis is a classic puzzle that has been around for"
    EXPECTED_GENERATION_CUSTOM_MODEL = "Hello,\n\nI have a problem with my 20"
    EXPECTED_GENERATION_MIXTRAL = prompt + " You're on the North Pole.\n\nThe"

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

    def test_fused_modules_to_not_convert(self):
        """
        Test if fused + modules to_not_covnert work as expected
        """
        model_id = "hf-internal-testing/Mixtral-tiny-AWQ"

        quantization_config = AwqConfig(bits=4, fuse_max_seq_len=128, do_fuse=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=quantization_config,
            low_cpu_mem_usage=True,
        ).to(torch_device)

        # Check if model has been correctly fused
        self._check_fused_modules(model)
        # Checks if the modules_to_not_convert (here gate layer) is a Linear
        self.assertTrue(isinstance(model.model.layers[0].block_sparse_moe.gate, torch.nn.Linear))

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

    def test_generation_llava_fused(self):
        from transformers import pipeline

        quantization_config = AwqConfig(do_fuse=True, fuse_max_seq_len=2048)

        pipe = pipeline(
            "image-to-text",
            model=self.multi_modal_model_name,
            device=0,
            model_kwargs={
                "quantization_config": quantization_config,
            },
            revision=self.multi_modal_model_code_revision,
        )
        url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/compel-neg.png"

        prompt = "USER: <image>\nCan you please describe this image?\nASSISTANT:"

        outputs = pipe(url, prompt=prompt, generate_kwargs={"max_new_tokens": 100})
        EXPECTED_OUTPUT = "USER:  \nCan you please describe this image?\nASSISTANT: The image features a brown and white cat sitting on a green surface, possibly a carpet or a grassy area. The cat is holding a red ball in its paws, seemingly playing with it. The cat appears to be focused on the ball, possibly preparing to play or just enjoying the toy."

        self.assertEqual(outputs[0]["generated_text"], EXPECTED_OUTPUT)

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
                "mlp": ["gate_proj", "up_proj", "down_proj"],
                "layernorm": ["input_layernorm", "post_attention_layernorm", "norm"],
                "use_alibi": False,
                "hidden_size": 4096,
                "num_attention_heads": 32,
                "num_key_value_heads": 8,
            },
        )

        model = AutoModelForCausalLM.from_pretrained(
            self.custom_mapping_model_id,
            quantization_config=quantization_config,
            device_map="balanced",
            revision=self.custom_model_revision,
        )

        self._check_fused_modules(model)

        tokenizer = AutoTokenizer.from_pretrained(self.custom_mapping_model_id, revision=self.custom_model_revision)

        prompt = "Hello"
        inputs = tokenizer(prompt, return_tensors="pt").to(torch_device)

        outputs = model.generate(**inputs, max_new_tokens=12)
        self.assertEqual(tokenizer.decode(outputs[0], skip_special_tokens=True), self.EXPECTED_GENERATION_CUSTOM_MODEL)

    @unittest.skip(reason="Not enough GPU memory on CI runners")
    @require_torch_multi_gpu
    def test_generation_mixtral_fused(self):
        """
        Text generation test for Mixtral + AWQ + fused
        """
        quantization_config = AwqConfig(bits=4, fuse_max_seq_len=1024, do_fuse=True)
        model = AutoModelForCausalLM.from_pretrained(
            self.mixtral_model_name,
            quantization_config=quantization_config,
            device_map="auto",
            revision=self.mixtral_model_revision,
        )

        tokenizer = AutoTokenizer.from_pretrained(self.mixtral_model_name)
        tokenizer.pad_token = tokenizer.eos_token

        inputs = tokenizer([self.prompt, self.prompt], return_tensors="pt", padding=True).to(torch_device)

        outputs = model.generate(**inputs, max_new_tokens=12)
        self.assertEqual(tokenizer.decode(outputs[0], skip_special_tokens=True), self.EXPECTED_GENERATION_MIXTRAL)


@slow
@require_torch_gpu
@require_auto_awq
@require_accelerate
class AwqScaleTest(unittest.TestCase):
    model_name = "TechxGenus/starcoder2-3b-AWQ"

    def test_load_quantized_model(self):
        from awq.modules.act import ScaledActivation

        """
        Simple test that checks if the scales have been replaced in the quantized model
        """
        quantized_model = AutoModelForCausalLM.from_pretrained(
            "TechxGenus/starcoder2-3b-AWQ", torch_dtype=torch.float16, device_map="cuda"
        )
        self.assertTrue(isinstance(quantized_model.model.layers[0].mlp.act, ScaledActivation))
