# coding=utf-8
# Copyright 2022 The HuggingFace Team Inc.
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

import tempfile
import unittest

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from transformers.testing_utils import (
    require_accelerate,
    require_bitsandbytes,
    require_torch,
    require_torch_gpu,
    require_torch_multi_gpu,
    slow,
)


@require_bitsandbytes
@require_accelerate
@require_torch
@require_torch_gpu
# @slow
class MixedInt8Test(unittest.TestCase):
    # We keep the constants inside the init function and model loading inside setUp function

    # We need to test on relatively large models (aka >1b parameters otherwise the quantiztion may not work as expected)
    # Therefore here we use only bloom-1b3 to test our module
    model_name = "bigscience/bloom-1b3"

    # Constant values
    EXPECTED_RELATIVE_DIFFERENCE = (
        1.540025  # This was obtained on a Quadro RTX 8000 so the number might slightly change
    )

    input_text = "Hello my name is"
    EXPECTED_OUTPUT = "Hello my name is John.\nI am a friend of your father.\n"
    MAX_NEW_TOKENS = 10

    def setUp(self):
        # Models pipeline and tokenizer
        self.model_fp16 = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype="auto", device_map="auto")
        self.model_8bit = AutoModelForCausalLM.from_pretrained(self.model_name, load_in_8bit=True, device_map="auto")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        return super().setUp()

    def test_memory_footprint(self):
        r"""
        A simple test to check if the model conversion has been done correctly by checking on the
        memory footprint of the converted model and the class type of the linear layers of the converted models
        """
        from bitsandbytes.nn import Int8Params

        mem_fp16 = self.model_fp16.get_memory_footprint()
        mem_8bit = self.model_8bit.get_memory_footprint()

        self.assertAlmostEqual(mem_fp16 / mem_8bit, self.EXPECTED_RELATIVE_DIFFERENCE)
        self.assertTrue(self.model_8bit.transformer.h[0].mlp.dense_4h_to_h.weight.__class__ == Int8Params)

    def test_generate_quality(self):
        r"""
        Test the generation quality of the quantized model and see that we are matching the expected output.
        Given that we are operating on small numbers + the testing model is relatively small, we might not get
        the same output across GPUs. So we'll generate few tokens (5-10) and check their output.
        """

        encoded_input = self.tokenizer(self.input_text, return_tensors="pt")
        output_sequences = self.model_8bit.generate(input_ids=encoded_input["input_ids"].cuda(), max_new_tokens=10)

        self.assertEqual(self.tokenizer.decode(output_sequences[0], skip_special_tokens=True), self.EXPECTED_OUTPUT)

    def test_pipeline(self):
        r"""
        The aim of this test is to verify that the mixed int8 is compatible with `pipeline` from transformers. Since
        we used pipline for inference speed benchmarking we want to make sure that this feature does not break anything
        on pipline.
        """

        pipe = pipeline(
            model=self.model_name,
            model_kwargs={"device_map": "auto", "load_in_8bit": True},
            max_new_tokens=self.MAX_NEW_TOKENS,
        )
        pipeline_output = pipe(self.input_text)
        self.assertEqual(pipeline_output[0]["generated_text"], self.EXPECTED_OUTPUT)

    def test_save_load(self):
        r"""
        The aim of this test is to verify whether if we save and load back a quantized model we retain the same performance.
        If this test pass people can safely push quantized models on the Hub.
        """
        with tempfile.TemporaryDirectory() as tmpdirname:
            # Save and load 8bit model
            self.model_8bit.save_pretrained(tmpdirname)
            loaded_model_8bit = AutoModelForCausalLM.from_pretrained(tmpdirname, load_in_8bit=True, device_map="auto")
            print(loaded_model_8bit)

    @require_torch_multi_gpu
    def test_multi_gpu_loading(self):
        r"""
        This tests that the model has been loaded and can be used correctly on a multi-GPU setup
        """
        pass
