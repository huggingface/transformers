# Copyright 2024 The HuggingFace Team. All rights reserved.
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
from transformers.testing_utils import (
    require_accelerate,
    require_optimum_quanto,
    require_read_token,
    require_torch_accelerator,
    slow,
    torch_device,
)
from transformers.utils import is_accelerate_available, is_optimum_quanto_available, is_torch_available


if is_torch_available():
    import torch

if is_accelerate_available():
    from accelerate import init_empty_weights

if is_optimum_quanto_available():
    from optimum.quanto import QLayerNorm, QLinear

    from transformers.integrations.quanto import replace_with_quanto_layers


@require_optimum_quanto
@require_accelerate
class QuantoTestIntegration(unittest.TestCase):
    model_id = "HuggingFaceTB/SmolLM3-3B"

    def setUp(self):
        config = AutoConfig.from_pretrained(self.model_id)
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
        self.model = replace_with_quanto_layers(self.model, quantization_config=quantization_config)

        nb_qlinear = 0
        for module in self.model.modules():
            if isinstance(module, QLinear):
                nb_qlinear += 1

        self.assertEqual(self.nb_linear, nb_qlinear)

    def test_weight_and_activation_quantization_conversion(self):
        """
        Simple test that checks if the quantized model has been converted properly when using weight + activation quantization
        """

        # Try with weight + activation quantization
        quantization_config = QuantoConfig(weights="int8", activations="int8")
        self.model = replace_with_quanto_layers(self.model, quantization_config=quantization_config)

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
        self.model = replace_with_quanto_layers(
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


@slow
@require_torch_accelerator
@require_optimum_quanto
@require_accelerate
class QuantoQuantizationTest(unittest.TestCase):
    """
    Test 8-bit weights only quantization
    """

    model_name = "HuggingFaceTB/SmolLM2-135M"

    weights = "int8"
    activations = None
    device_map = "cpu"

    input_text = "Hello my name is"
    EXPECTED_OUTPUTS = "Hello my name is John. I am a student of the University of"

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
            dtype=torch.float32,
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
        Simple test to check the quality of the model on cpu by comparing the generated tokens with the expected tokens
        """
        self.check_inference_correctness(self.quantized_model, "cpu")

    def test_generate_quality_accelerator(self):
        """
        Simple test to check the quality of the model on accelerators by comparing the generated tokens with the expected tokens
        """
        self.check_inference_correctness(self.quantized_model, torch_device)

    def test_quantized_model_layers(self):
        from optimum.quanto import QBitsTensor, QModuleMixin, QTensor

        """
        Suite of simple test to check if the layers are quantized and are working properly
        """
        # Test the type of the quantized layer
        self.assertTrue(isinstance(self.quantized_model.model.layers[0].self_attn.k_proj, QModuleMixin))
        self.assertTrue(isinstance(self.quantized_model.model.layers[0].self_attn.k_proj.weight, QTensor))
        if self.weights == "int4":
            self.assertTrue(isinstance(self.quantized_model.model.layers[0].self_attn.k_proj.weight, QBitsTensor))

        # check that the lm_head was indeed not quantized, just like bnb
        self.assertTrue(
            isinstance(self.quantized_model.lm_head, torch.nn.Linear)
            and not isinstance(self.quantized_model.lm_head, QModuleMixin)
        )
        if self.device_map in ["cpu", "cuda"]:
            self.assertEqual(
                self.quantized_model.model.layers[0].self_attn.k_proj.weight._data.device.type,
                self.device_map,
            )
            self.quantized_model.to(0)
        self.assertEqual(self.quantized_model.model.layers[0].self_attn.k_proj.weight._data.device.type, torch_device)

    def test_serialization_safetensors(self):
        """
        Test the serialization, the loading and the inference of the quantized weights
        """
        with tempfile.TemporaryDirectory() as tmpdirname:
            with self.assertRaises(ValueError) as e:
                self.quantized_model.save_pretrained(tmpdirname)
            self.assertIn("The model is quantized with quanto and is not serializable", str(e.exception))

    def check_same_model(self, model1, model2):
        d0 = dict(model1.named_parameters())
        d1 = dict(model2.named_parameters())
        self.assertTrue(d0.keys() == d1.keys())
        for k in d0:
            self.assertTrue(d0[k].shape == d1[k].shape)
            self.assertTrue(d0[k].device.type == d1[k].device.type)
            self.assertTrue(d0[k].device == d1[k].device)
            self.assertTrue(d0[k].dtype == d1[k].dtype)
            self.assertTrue(torch.equal(d0[k], d1[k].to(d0[k].device)))

    def test_compare_with_quanto(self):
        from optimum.quanto import freeze, qint4, qint8, quantize

        w_mapping = {"int8": qint8, "int4": qint4}
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map=self.device_map,
            dtype=torch.float32,
        )
        # we do not quantize the lm_head since we don't do that in transformers
        quantize(model.model, weights=w_mapping[self.weights])
        freeze(model.model)
        self.check_same_model(model, self.quantized_model)
        self.check_inference_correctness(model, device=torch_device)


class QuantoQuantizationQBitsTensorTest(QuantoQuantizationTest):
    EXPECTED_OUTPUTS = "Hello my name is joe and i am a little girl\n\n"
    weights = "int4"


@require_torch_accelerator
class QuantoQuantizationActivationTest(unittest.TestCase):
    def test_quantize_activation(self):
        quantization_config = QuantoConfig(
            weights="int8",
            activations="int8",
        )
        with self.assertRaises(ValueError) as e:
            AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-135M", quantization_config=quantization_config)
        self.assertIn("We don't support quantizing the activations with transformers library", str(e.exception))


@require_optimum_quanto
@require_torch_accelerator
class QuantoKVCacheQuantizationTest(unittest.TestCase):
    @slow
    @require_read_token
    def test_quantized_cache(self):
        EXPECTED_TEXT_COMPLETION = [
            "Simply put, the theory of relativity states that 1) time and space are not absolute, but are relative to the observer, and 2) the laws of physics are the same everywhere in the universe. This means that the speed of light is",
            "My favorite all time favorite condiment is ketchup. I love how it adds a sweet and tangy flavor to my food. I also enjoy using it as a dip for fries, burgers, and grilled meats. It's a classic condiment that never",
        ]

        prompts = [
            "Simply put, the theory of relativity states that ",
            "My favorite all time favorite condiment is ketchup.",
        ]
        tokenizer = AutoTokenizer.from_pretrained(
            "unsloth/Llama-3.2-1B-Instruct", pad_token="</s>", padding_side="left"
        )
        model = AutoModelForCausalLM.from_pretrained(
            "unsloth/Llama-3.2-1B-Instruct", device_map="sequential", dtype=torch.float16
        )
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(torch_device)

        generated_ids = model.generate(**inputs, max_new_tokens=40, do_sample=False, cache_implementation="quantized")
        text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        self.assertEqual(EXPECTED_TEXT_COMPLETION, text)
