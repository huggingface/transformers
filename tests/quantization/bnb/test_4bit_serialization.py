# coding=utf-8
# Copyright 2023 The HuggingFace Team Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a clone of the License at
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

from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from transformers.testing_utils import (
    is_bitsandbytes_available,
    require_accelerate,
    require_bitsandbytes,
    require_torch,
    require_torch_gpu,
)


if is_bitsandbytes_available():
    import bitsandbytes as bnb
    import torch


def get_some_linear_layer(model):
    if model.config.model_type == "gpt2":
        return model.transformer.h[0].mlp.c_fc
    elif model.config.model_type == "opt":
        return model.model.decoder.layers[0].fc1
    else:
        return model.transformer.h[0].mlp.dense_4h_to_h


@require_bitsandbytes
@require_accelerate
@require_torch
@require_torch_gpu
@slow
class BaseTest(unittest.TestCase):
    # We keep the constants inside the init function and model loading inside setUp function

    # We need to test on relatively large models (aka >1b parameters otherwise the quantiztion may not work as expected)
    # Therefore here we use only bloom-1b3 to test our module
    model_name = "facebook/opt-125m"
    input_text = "Mars colonists' favorite meals are"

    def setUp(self):
        self.device = torch.device("cuda:0")

    def tearDown(self):
        r"""
        TearDown function needs to be called at the end of each test to free the GPU memory and cache, also to
        avoid unexpected behaviors. Please see: https://discuss.pytorch.org/t/how-can-we-release-gpu-memory-cache/14530/27
        """
        gc.collect()
        torch.cuda.empty_cache()

    def test_serialization(self, quant_type="nf4", double_quant=True, safe_serialization=True):
        r"""
        Test whether it is possible to serialize a model in 4-bit. Uses most typical params as default.
        See ExtendedTest class for more params combinations.
        """

        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        # self.model_fp16 = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype=torch.float16, device_map="auto")
        self.quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=quant_type,
            bnb_4bit_use_double_quant=double_quant,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model_0 = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=self.quantization_config,
            device_map=self.device,
        )

        with tempfile.TemporaryDirectory() as tmpdirname:
            model_0.save_pretrained(tmpdirname, safe_serialization=safe_serialization)

            config = AutoConfig.from_pretrained(tmpdirname)
            self.assertTrue(hasattr(config, "quantization_config"))

            model_1 = AutoModelForCausalLM.from_pretrained(tmpdirname, device_map=self.device)

        # checking quantized linear module weight
        linear = get_some_linear_layer(model_1)
        self.assertTrue(linear.weight.__class__ == bnb.nn.Params4bit)
        self.assertTrue(hasattr(linear.weight, "quant_state"))
        self.assertTrue(linear.weight.quant_state.__class__ == bnb.functional.QuantState)

        # checking memory footpring
        self.assertAlmostEqual(model_0.get_memory_footprint() / model_1.get_memory_footprint(), 1, places=2)

        # Matching all parameters and their quant_state items:
        d0 = dict(model_0.named_parameters())
        d1 = dict(model_1.named_parameters())
        self.assertTrue(d0.keys() == d1.keys())

        for k in d0.keys():
            self.assertTrue(d0[k].shape == d1[k].shape)
            self.assertTrue(d0[k].device.type == d1[k].device.type)
            self.assertTrue(d0[k].device == d1[k].device)
            self.assertTrue(d0[k].dtype == d1[k].dtype)
            self.assertTrue(torch.equal(d0[k], d1[k].to(d0[k].device)))

            if isinstance(d0[k], bnb.nn.modules.Params4bit):
                for v0, v1 in zip(
                    d0[k].quant_state.as_dict().values(),
                    d1[k].quant_state.as_dict().values(),
                ):
                    if isinstance(v0, torch.Tensor):
                        self.assertTrue(torch.equal(v0, v1.to(v0.device)))
                    else:
                        self.assertTrue(v0 == v1)

        # comparing forward() outputs
        encoded_input = tokenizer(self.input_text, return_tensors="pt").to(self.device)
        out_0 = model_0(**encoded_input)
        out_1 = model_1(**encoded_input)
        self.assertTrue(torch.equal(out_0["logits"], out_1["logits"]))

        # comparing generate() outputs
        encoded_input = tokenizer(self.input_text, return_tensors="pt").to(self.device)
        output_sequences_0 = model_0.generate(**encoded_input, max_new_tokens=10)
        output_sequences_1 = model_1.generate(**encoded_input, max_new_tokens=10)

        def _decode(token):
            return tokenizer.decode(token, skip_special_tokens=True)

        self.assertEqual(
            [_decode(x) for x in output_sequences_0],
            [_decode(x) for x in output_sequences_1],
        )


class ExtendedTest(BaseTest):
    """
    tests more combinations of parameters
    """

    def test_nf4_single_unsafe(self):
        self.test_serialization(quant_type="nf4", double_quant=False, safe_serialization=False)

    def test_nf4_single_safe(self):
        self.test_serialization(quant_type="nf4", double_quant=False, safe_serialization=True)

    def test_nf4_double_unsafe(self):
        self.test_serialization(quant_type="nf4", double_quant=True, safe_serialization=False)

    def test_nf4_double_safe(self):
        self.test_serialization(quant_type="nf4", double_quant=True, safe_serialization=True)

    def test_fp4_single_unsafe(self):
        self.test_serialization(quant_type="fp4", double_quant=False, safe_serialization=False)

    def test_fp4_double_unsafe(self):
        self.test_serialization(quant_type="fp4", double_quant=True, safe_serialization=False)


class BloomTest(BaseTest):
    """
    base config tested with Bloom family model
    """

    model_name = "bigscience/bloom-560m"


class GPTTest(BaseTest):
    """
    base config tested with GPT family model
    """

    model_name = "gpt2-xl"
