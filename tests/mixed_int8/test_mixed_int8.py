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
import gc
import unittest

from transformers import AutoModel, AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer, pipeline
from transformers.testing_utils import (
    is_torch_available,
    require_accelerate,
    require_bitsandbytes,
    require_torch,
    require_torch_gpu,
    require_torch_multi_gpu,
    slow,
)


if is_torch_available():
    import torch


@require_bitsandbytes
@require_accelerate
@require_torch
@require_torch_gpu
@slow
class BaseMixedInt8Test(unittest.TestCase):
    # We keep the constants inside the init function and model loading inside setUp function

    # We need to test on relatively large models (aka >1b parameters otherwise the quantiztion may not work as expected)
    # Therefore here we use only bloom-1b3 to test our module
    model_name = "bigscience/bloom-1b7"

    # Constant values
    EXPECTED_RELATIVE_DIFFERENCE = (
        1.540025  # This was obtained on a Quadro RTX 8000 so the number might slightly change
    )

    input_text = "Hello my name is"
    EXPECTED_OUTPUT = "Hello my name is John.\nI am a friend of the family.\n"
    MAX_NEW_TOKENS = 10

    def setUp(self):
        # Models and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)


class MixedInt8Test(BaseMixedInt8Test):
    def setUp(self):
        super().setUp()

        # Models and tokenizer
        self.model_fp16 = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype="auto", device_map="auto")
        self.model_8bit = AutoModelForCausalLM.from_pretrained(self.model_name, load_in_8bit=True, device_map="auto")

    def tearDown(self):
        r"""
        TearDown function needs to be called at the end of each test to free the GPU memory and cache, also to
        avoid unexpected behaviors. Please see: https://discuss.pytorch.org/t/how-can-we-release-gpu-memory-cache/14530/27
        """
        del self.model_fp16
        del self.model_8bit

        gc.collect()
        torch.cuda.empty_cache()

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
        output_sequences = self.model_8bit.generate(input_ids=encoded_input["input_ids"].to(0), max_new_tokens=10)

        self.assertEqual(self.tokenizer.decode(output_sequences[0], skip_special_tokens=True), self.EXPECTED_OUTPUT)


class MixedInt8ModelClassesTest(BaseMixedInt8Test):
    def setUp(self):
        super().setUp()
        # model_name
        self.model_name = "bigscience/bloom-560m"
        # Models and tokenizer
        self.base_model = AutoModel.from_pretrained(self.model_name, load_in_8bit=True, device_map="auto")
        self.sequence_model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, load_in_8bit=True, device_map="auto"
        )
        self.model_8bit = AutoModelForCausalLM.from_pretrained(self.model_name, load_in_8bit=True, device_map="auto")

    def tearDown(self):
        r"""
        TearDown function needs to be called at the end of each test to free the GPU memory and cache, also to
        avoid unexpected behaviors. Please see: https://discuss.pytorch.org/t/how-can-we-release-gpu-memory-cache/14530/27
        """
        del self.base_model
        del self.sequence_model
        del self.model_8bit

        gc.collect()
        torch.cuda.empty_cache()

    def test_correct_head_class(self):
        r"""
        A simple test to check if the last modules for some classes (AutoModelForCausalLM or SequenceClassification)
        are kept in their native class.
        """
        from bitsandbytes.nn import Int8Params

        # last param of a base model should be a linear8bit module
        self.assertTrue(self.base_model.h[-1].mlp.dense_4h_to_h.weight.__class__ == Int8Params)

        # Other heads should be nn.Parameter
        self.assertTrue(self.model_8bit.lm_head.weight.__class__ == torch.nn.Parameter)
        self.assertTrue(self.sequence_model.score.weight.__class__ == torch.nn.Parameter)


class MixedInt8TestPipeline(BaseMixedInt8Test):
    def setUp(self):
        super().setUp()

    def tearDown(self):
        r"""
        TearDown function needs to be called at the end of each test to free the GPU memory and cache, also to
        avoid unexpected behaviors. Please see: https://discuss.pytorch.org/t/how-can-we-release-gpu-memory-cache/14530/27
        """
        del self.pipe

        gc.collect()
        torch.cuda.empty_cache()

    def test_pipeline(self):
        r"""
        The aim of this test is to verify that the mixed int8 is compatible with `pipeline` from transformers. Since
        we used pipline for inference speed benchmarking we want to make sure that this feature does not break anything
        on pipline.
        """
        # self._clear_cuda_cache()
        self.pipe = pipeline(
            "text-generation",
            model=self.model_name,
            model_kwargs={"device_map": "auto", "load_in_8bit": True},
            max_new_tokens=self.MAX_NEW_TOKENS,
        )

        # Real second forward pass
        pipeline_output = self.pipe(self.input_text)
        self.assertEqual(pipeline_output[0]["generated_text"], self.EXPECTED_OUTPUT)


@require_torch_multi_gpu
class MixedInt8TestMultiGpu(BaseMixedInt8Test):
    def setUp(self):
        super().setUp()

    def test_multi_gpu_loading(self):
        r"""
        This tests that the model has been loaded and can be used correctly on a multi-GPU setup.
        Let's just try to load a model on 2 GPUs and see if it works. The model we test has ~2GB of total, 3GB should suffice
        """

        memory_mapping = {0: "1GB", 1: "2GB"}
        model_parallel = AutoModelForCausalLM.from_pretrained(
            self.model_name, load_in_8bit=True, max_memory=memory_mapping, device_map="auto"
        )

        def get_list_devices(model):
            list_devices = []
            for _, module in model.named_children():
                if len(list(module.children())) > 0:
                    list_devices.extend(get_list_devices(module))
                else:
                    # Do a try except since we can encounter Dropout modules that does not
                    # have any device set
                    try:
                        list_devices.append(next(module.parameters()).device.index)
                    except BaseException:
                        continue
            return list_devices

        list_devices = get_list_devices(model_parallel)
        # Check that we have dispatched the model into 2 separate devices
        self.assertTrue((1 in list_devices) and (0 in list_devices))

        # Check that inference pass works on the model
        encoded_input = self.tokenizer(self.input_text, return_tensors="pt")

        # Second real batch
        output_parallel = model_parallel.generate(input_ids=encoded_input["input_ids"].to(0), max_new_tokens=10)
        self.assertEqual(self.tokenizer.decode(output_parallel[0], skip_special_tokens=True), self.EXPECTED_OUTPUT)
