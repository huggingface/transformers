# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team and the Swiss AI Initiative. All rights reserved.
#
# This code is based on HuggingFace's LLaMA implementation in this library.
# It has been modified from its original forms to accommodate minor architectural
# differences compared to LLaMA used by the Swiss AI Initiative that trained the model.
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
"""Testing suite for the PyTorch Apertus model."""

import unittest

import torch

from transformers import is_torch_available
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.testing_utils import (
    cleanup,
    require_read_token,
    require_torch,
    require_torch_accelerator,
    slow,
    torch_device,
)

from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester


if is_torch_available():
    import torch

    from transformers import (
        ApertusConfig,
        ApertusForCausalLM,
        ApertusForTokenClassification,
        ApertusModel,
    )


class ApertusModelTester(CausalLMModelTester):
    if is_torch_available():
        config_class = ApertusConfig
        base_model_class = ApertusModel
        causal_lm_class = ApertusForCausalLM
        token_class = ApertusForTokenClassification


@require_torch
class ApertusModelTest(CausalLMModelTest, unittest.TestCase):
    all_model_classes = (
        (
            ApertusModel,
            ApertusForCausalLM,
            ApertusForTokenClassification,
        )
        if is_torch_available()
        else ()
    )
    pipeline_model_mapping = (
        {
            "feature-extraction": ApertusModel,
            "text-generation": ApertusForCausalLM,
            "token-classification": ApertusForTokenClassification,
        }
        if is_torch_available()
        else {}
    )
    test_headmasking = False
    test_pruning = False
    model_tester_class = ApertusModelTester

    # Need to use `0.8` instead of `0.9` for `test_cpu_offload`
    # This is because we are hitting edge cases with the causal_mask buffer
    model_split_percents = [0.5, 0.7, 0.8]

    # used in `test_torch_compile_for_training`
    _torch_compile_train_cls = ApertusForCausalLM if is_torch_available() else None


@require_torch_accelerator
@require_read_token
@require_torch
@slow
class ApertusIntegrationTest(unittest.TestCase):
    # NOTE: Using XIELU cuda experimental and python (torch) version give different results when using dtype=bfloat16
    @classmethod
    def setUpClass(cls):
        cls.model = None
        cls.tokenizer = None

    @classmethod
    def tearDownClass(cls):
        del cls.model
        cleanup(torch_device, gc_collect=True)

    def tearDown(self):
        cleanup(torch_device, gc_collect=True)

    @classmethod
    def get_model(cls, size="8B", type="-Instruct", dtype=torch.bfloat16):
        if cls.model is None:
            cls.model = ApertusForCausalLM.from_pretrained(
                f"swiss-ai/Apertus-{size}{type}-2509",
                device_map="auto",
                dtype=dtype,
            )

        return cls.model

    @classmethod
    def get_tokenizer(cls, size="8B", type="-Instruct"):
        if cls.tokenizer is None:
            cls.tokenizer = AutoTokenizer.from_pretrained(f"swiss-ai/Apertus-{size}{type}-2509", device_map="auto")

        return cls.tokenizer

    @slow
    def test_model_8b_greedy_generation(self):
        EXPECTED_TEXT_COMPLETION = """Simply put, the theory of relativity states that 1) the laws of physics are the same in all inertial reference frames, and 2) the speed of light is constant in all inertial reference frames. The first part is called the principle of relativity, and the second part is called the principle of constancy of the speed of light. The theory of relativity is based"""
        prompt = "Simply put, the theory of relativity states that "
        model = self.get_model(type="")
        tokenizer = self.get_tokenizer(type="")
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)

        # greedy generation outputs
        generated_ids = model.generate(input_ids, max_new_tokens=64, top_p=None, temperature=1, do_sample=False)
        output_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

        self.assertEqual(EXPECTED_TEXT_COMPLETION, output_text)

    @slow
    def test_model_8b_instruct_greedy_generation(self):
        self.setUpClass()

        EXPECTED_TEXT_COMPLETION = """<s><|system_start|>You are Apertus, a helpful assistant created by the SwissAI initiative.\nKnowledge cutoff: 2024-04\nCurrent date: 2025-09-21<|system_end|><|developer_start|>Deliberation: disabled\nTool Capabilities: disabled<|developer_end|><|user_start|>Give me a brief explanation of gravity in simple terms.<|user_end|><|assistant_start|>Gravity is a force that pulls objects towards each other. The more massive an object is, the stronger its gravitational pull. This is why planets and stars have strong gravitational pulls that keep their moons and other objects in orbit around them.<|assistant_end|>"""

        model = self.get_model()
        tokenizer = self.get_tokenizer()

        prompt = "Give me a brief explanation of gravity in simple terms."
        messages = [{"role": "user", "content": prompt}]

        inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(
            model.device
        )

        # greedy generation outputs
        outputs = model.generate(inputs, max_new_tokens=64, top_p=None, temperature=1, do_sample=False)
        output_text = tokenizer.batch_decode(outputs)

        self.assertEqual(EXPECTED_TEXT_COMPLETION, output_text[0])

    @slow
    def test_model_8b_logits(self):
        input_ids = [1, 13517, 7072, 4283, 1044, 1278, 9191, 1307, 115847, 8164, 1455, 1032]
        model = self.get_model()
        input_ids = torch.tensor([input_ids]).to(model.device)
        with torch.no_grad():
            out = model(input_ids).logits.float()

        # Expected mean on dim = -1
        EXPECTED_MEAN = torch.tensor(
            [[0.5109, 1.9118, -0.9534, 1.5958, -5.8428, -2.8495, -0.1818, -2.6219, -3.5842, -1.4742, -2.9235, 2.4903]], device=model.device
        )  # fmt: skip

        torch.testing.assert_close(out.mean(-1), EXPECTED_MEAN, rtol=1e-2, atol=1e-2)

        # slicing logits[0, 0, 0:30]
        EXPECTED_SLICE = torch.tensor([0.6992,  5.3438, 10.9375,  2.8281,  3.6562,  3.2656,  3.2656,  3.2656, 3.2656,  3.1094, -0.5586,  2.5156,  1.8516,  1.0391,  2.3750,  3.4688, 5.8750,  3.0938,  9.5000, 10.0000,  4.4375,  5.2500,  5.0938,  2.4844, 3.9844,  0.9883,  1.2344,  1.8594,  3.5000,  5.6875], device=model.device)  # fmt: skip
        torch.testing.assert_close(out[0, 0, :30], EXPECTED_SLICE, rtol=1e-4, atol=1e-4)

    @slow
    def test_model_8b_logits_no_xielu_cuda(self):
        import os

        os.environ["XIELU_DISABLE_CUDA"] = "1"

        self.setUpClass()

        input_ids = [1, 13517, 7072, 4283, 1044, 1278, 9191, 1307, 115847, 8164, 1455, 1032]
        model = self.get_model()
        input_ids = torch.tensor([input_ids]).to(model.device)
        with torch.no_grad():
            out = model(input_ids).logits.float()

        # Expected mean on dim = -1
        EXPECTED_MEAN = torch.tensor(
            [[-0.0198, 1.8646, -0.9973, 0.4992, -5.5038, -8.0162, 0.5486, -1.9783, -3.1419, 0.6811, -1.6834, 3.4895]],
            device=model.device,
        )

        torch.testing.assert_close(out.mean(-1), EXPECTED_MEAN, rtol=1e-2, atol=1e-2)

        # slicing logits[0, 0, 0:30]
        EXPECTED_SLICE = torch.tensor([-0.0204,  6.4688, 11.6875,  2.4219,  2.5156,  2.4219,  2.4219,  2.4219, 2.4219,  2.2812, -1.2969,  1.7500,  1.9531,  0.3281,  2.9062,  4.0938, 5.9062,  2.7188,  9.5000, 10.1250,  3.5156,  5.3438,  5.0625,  1.7578, 4.0625,  1.4219,  1.0391,  1.7656,  2.7031,  5.3750], device=model.device)  # fmt: skip
        torch.testing.assert_close(out[0, 0, :30], EXPECTED_SLICE, rtol=1e-4, atol=1e-4)

    @slow
    def test_model_8b_logits_float32(self):
        # NOTE: I think float32 tests is not something we need (so remove after checking errors).

        input_ids = [1, 13517, 7072, 4283, 1044, 1278, 9191, 1307, 115847, 8164, 1455, 1032]
        model = self.get_model(dtype=torch.float32)
        input_ids = torch.tensor([input_ids]).to(model.device)
        with torch.no_grad():
            out = model(input_ids).logits.float()

        # Expected mean on dim = -1
        EXPECTED_MEAN = torch.tensor(
            [[0.5109, 1.9118, -0.9534, 1.5958, -5.8428, -2.8495, -0.1818, -2.6219, -3.5842, -1.4742, -2.9235, 2.4903]],
            device=model.device,
        )

        torch.testing.assert_close(out.mean(-1), EXPECTED_MEAN, rtol=1e-2, atol=1e-2)

        # slicing logits[0, 0, 0:30]
        EXPECTED_SLICE = torch.tensor([0.6992,  5.3438, 10.9375,  2.8281,  3.6562,  3.2656,  3.2656,  3.2656,3.2656,  3.1094, -0.5586,  2.5156,  1.8516,  1.0391,  2.3750,  3.4688, 5.8750,  3.0938,  9.5000, 10.0000,  4.4375,  5.2500,  5.0938,  2.4844, 3.9844,  0.9883,  1.2344,  1.8594,  3.5000,  5.6875], device=model.device)  # fmt: skip
        torch.testing.assert_close(out[0, 0, :30], EXPECTED_SLICE, rtol=1e-4, atol=1e-4)

    @slow
    def test_model_8b_logits_float32_no_xielu_cuda(self):
        # NOTE: I think float32 tests is not something we need (so remove after checking errors).

        import os

        os.environ["XIELU_DISABLE_CUDA"] = "1"

        self.setUpClass()

        input_ids = [1, 13517, 7072, 4283, 1044, 1278, 9191, 1307, 115847, 8164, 1455, 1032]
        model = self.get_model(dtype=torch.float32)
        input_ids = torch.tensor([input_ids]).to(model.device)
        with torch.no_grad():
            out = model(input_ids).logits.float()

        # Expected mean on dim = -1
        EXPECTED_MEAN = torch.tensor(
            [[0.0812, 1.8028, -1.1248, 0.8004, -5.9161, -7.7755, 0.4919, -2.0355, -3.2272, 0.6626, -1.7287, 3.4431]],
            device=model.device,
        )

        torch.testing.assert_close(out.mean(-1), EXPECTED_MEAN, rtol=1e-2, atol=1e-2)

        # slicing logits[0, 0, 0:30]
        EXPECTED_SLICE = torch.tensor([0.6404,  6.8661, 12.6953,  2.9036,  2.9236,  2.7907,  2.7907,  2.7907, 2.7907,  2.5965, -0.9208,  1.8292,  2.1176,  0.5388,  3.4888,  4.4509, 6.8699,  3.4439, 10.3936, 10.9893,  3.1937,  5.6061,  5.6495,  1.4633, 4.6484,  1.4452,  1.2131,  2.0822,  2.4203,  6.0496], device=model.device)  # fmt: skip
        torch.testing.assert_close(out[0, 0, :30], EXPECTED_SLICE, rtol=1e-4, atol=1e-4)
