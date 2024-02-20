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

import tempfile
import unittest

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, QuantoConfig
from transformers.testing_utils import require_accelerate, require_quanto, require_torch_gpu, slow
from transformers.utils import is_accelerate_available, is_quanto_available, is_torch_available


if is_torch_available():
    import torch

if is_accelerate_available():
    from accelerate import init_empty_weights

if is_quanto_available():
    from quanto import QLayerNorm, QLinear

    from transformers.integrations.quanto import replace_with_quanto_layers


class QuantoConfigTest(unittest.TestCase):
    def test_attributes(self):
        pass


@require_quanto
@require_accelerate
class QuantoTestIntegration(unittest.TestCase):
    model_id = "facebook/opt-350m"

    def setUp(self):
        # empty model
        config = AutoConfig.from_pretrained(self.model_id, revision="cb32f77e905cccbca1d970436fb0f5e6b58ee3c5")
        with init_empty_weights():
            self.model = AutoModelForCausalLM.from_config(config)
        self.nb_linear = 0
        self.nb_layernorm = 0
        for module in self.model.modules():
            if isinstance(module, torch.nn.Linear):
                self.nb_linear += 1
            elif isinstance(module, torch.nn.LayerNorm):
                self.nb_layernorm += 1

    def test_weight_only_quantization_conversion(self):
        """
        Simple test that checks if the quantized model has been converted properly when using weight only quantization
        """

        # Try with weight only quantization
        quantization_config = QuantoConfig(weights="int8", activations=None)
        self.model, _ = replace_with_quanto_layers(self.model, quantization_config=quantization_config)

        nb_qlinear = 0
        for module in self.model.modules():
            if isinstance(module, QLinear):
                nb_qlinear += 1

        self.assertEqual(self.nb_linear, nb_qlinear)

    def test_weight_and_activation_quantization_conversion(self):
        """
        Simple test that checks if the quantized model has been converted properly when using weight + activation quantization
        """

        # Try with weight + activatioin quantization
        quantization_config = QuantoConfig(weights="int8", activations="int8")
        self.model, _ = replace_with_quanto_layers(self.model, quantization_config=quantization_config)

        nb_qlinear = 0
        nb_qlayernorm = 0
        for module in self.model.modules():
            if isinstance(module, QLinear):
                nb_qlinear += 1
            if isinstance(module, QLayerNorm):
                nb_qlayernorm += 1

        self.assertEqual(self.nb_linear, nb_qlinear)
        self.assertEqual(self.nb_layernorm, nb_qlayernorm)

    def test_conversion_with_modules_to_not_convert(self):
        """
        Simple test that checks if the quantized model has been converted properly when specifying modules_to_not_convert argument
        """

        # Try with weight + activatioin quantization
        quantization_config = QuantoConfig(weights="int8", activations="int8")
        self.model, _ = replace_with_quanto_layers(
            self.model, quantization_config=quantization_config, modules_to_not_convert=["lm_head"]
        )

        nb_qlinear = 0
        nb_qlayernorm = 0
        for module in self.model.modules():
            if isinstance(module, QLinear):
                nb_qlinear += 1
            if isinstance(module, QLayerNorm):
                nb_qlayernorm += 1

        self.assertEqual(self.nb_linear - 1, nb_qlinear)


# @slow
# @require_torch_gpu
# @require_quanto
# @require_accelerate
# class QuantoInferenceTest(unittest.TestCase):
#     model_name = ""

#     input_text = "Hello my name is"

#     EXPECTED_OUTPUT = ""
#     device_map = "cuda"

#     # called only once for all test in this class
#     @classmethod
#     def setUpClass(cls):
#         """
#         Setup quantized model
#         """
#         cls.tokenizer = AutoTokenizer.from_pretrained(cls.model_name)
#         cls.quantized_model = AutoModelForCausalLM.from_pretrained(
#             cls.model_name,
#             device_map=cls.device_map,
#         )

#     def tearDown(self):
#         torch.cuda.empty_cache()
#         gc.collect()

#     def test_quantized_model(self):
#         """
#         Simple test that checks if the quantized model is working properly
#         """
#         input_ids = self.tokenizer(self.input_text, return_tensors="pt").to(torch_device)

#         output = self.quantized_model.generate(**input_ids, max_new_tokens=40)
#         self.assertEqual(self.tokenizer.decode(output[0], skip_special_tokens=True), self.EXPECTED_OUTPUT)

#     def test_save_pretrained(self):
#         """
#         Simple test that checks if the quantized model is working properly after being saved and loaded
#         """
#         with tempfile.TemporaryDirectory() as tmpdirname:
#             self.quantized_model.save_pretrained(tmpdirname)
#             model = AutoModelForCausalLM.from_pretrained(tmpdirname, device_map=self.device_map)

#             input_ids = self.tokenizer(self.input_text, return_tensors="pt").to(torch_device)

#             output = model.generate(**input_ids, max_new_tokens=40)
#             self.assertEqual(self.tokenizer.decode(output[0], skip_special_tokens=True), self.EXPECTED_OUTPUT)


@slow
@require_torch_gpu
@require_quanto
@require_accelerate
class QuantoQuantizationTest(unittest.TestCase):
    """
    Test 8-bit weights only qunatization
    """

    model_name = "bigscience/bloom-560m"

    weights = "int8"
    activations = None
    device_map = "cpu"

    input_text = "Hello my name is"

    EXPECTED_OUTPUTS = "Hello my name is Manjit\nI am a student in Computer"

    def setUp(self):
        """
        Setup quantized model
        """
        quantization_config = QuantoConfig(
            weights=self.weights,
            activations=self.activations,
        )

        self.quantized_model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map=self.device_map,
            quantization_config=quantization_config,
            torch_dtype=torch.float32,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        self.have_accelerate_hooks = (
            getattr(self.quantized_model, "hf_device_map", False) and len(self.quantized_model.hf_device_map) > 1
        )

    def check_inference_correctness(self, model, device):
        r"""
        Test the generation quality of the quantized model and see that we are matching the expected output.
        Given that we are operating on small numbers + the testing model is relatively small, we might not get
        the same output across GPUs. So we'll generate few tokens (5-10) and check their output.
        """
        if not self.have_accelerate_hooks:
            model.to(device)
        encoded_input = self.tokenizer(self.input_text, return_tensors="pt")
        output_sequences = model.generate(input_ids=encoded_input["input_ids"].to(device), max_new_tokens=10)
        self.assertIn(self.tokenizer.decode(output_sequences[0], skip_special_tokens=True), self.EXPECTED_OUTPUTS)

    def test_generate_quality_cpu(self):
        """
        Simple test to check the quality of the model by comparing the generated tokens with the expected tokens
        """
        self.check_inference_correctness(self.quantized_model, "cpu")

    def test_generate_quality_cuda(self):
        """
        Simple test to check the quality of the model by comparing the generated tokens with the expected tokens
        """
        self.check_inference_correctness(self.quantized_model, "cuda")

    def test_move_quantized_model(self):
        """
        Simple test to check if the quantized weights were properly moved to right device
        """
        self.assertEqual(
            self.quantized_model.transformer.h[0].self_attention.query_key_value.weight._data.device.type, "cpu"
        )
        self.quantized_model.to(0)
        self.assertEqual(
            self.quantized_model.transformer.h[0].self_attention.query_key_value.weight._data.device.type, "cuda"
        )

    def test_serialization_bin(self):
        """
        Test the serialization, the loading and the inference of the quantized weights
        """
        with tempfile.TemporaryDirectory() as tmpdirname:
            self.quantized_model.save_pretrained(tmpdirname, safe_serialization=False)
            quantized_model_from_saved = AutoModelForCausalLM.from_pretrained(tmpdirname, torch_dtype=torch.float32, device_map="cpu")
            # We only test the inference on a single gpu. We will do more tests in QuantoInferenceTest class
            self.check_inference_correctness(quantized_model_from_saved, device="cuda")

    def test_serialization_safetensors(self):
        """
        Test the serialization, the loading and the inference of the quantized weights
        """
        with tempfile.TemporaryDirectory() as tmpdirname:
            self.quantized_model.save_pretrained(tmpdirname, safe_serialization=True)
            quantized_model_from_saved = AutoModelForCausalLM.from_pretrained(tmpdirname, torch_dtype=torch.float32, device_map="cpu")
            # We only test the inference on a single gpu. We will do more tests in QuantoInferenceTest class
            self.check_inference_correctness(quantized_model_from_saved, device="cuda")

# class QuantoQuantizationW4Test(QuantoQuantizationTest):
#     weights = "torch.int4"

# class QuantoQuantizationActivationTest(unittest.TestCase):
#     @parameterized.expand(
#         [
#             ("activation", "torch.int8", "torch.fp8"),
#         ]
#     )
#     def test_quantize_activation(self, activation):
#         quantization_config = QuantoConfig(
#             weights="torch.int8",
#             activations="torch.int8",
#         )
#         with self.assertRaises(RuntimeError):
#             AutoModelForCausalLM.from_pretrained("bigscience/bloom-560m", quantization_config=quantization_config)
