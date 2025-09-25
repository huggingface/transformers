# coding=utf-8
# Copyright 2025 the HuggingFace Team. All rights reserved.
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
"""Testing suite for the PyTorch FlexOlmo model."""

import unittest

import pytest

from transformers import FlexOlmoConfig, is_torch_available
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.testing_utils import (
    Expectations,
    cleanup,
    require_torch,
    slow,
    torch_device,
)

from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester


if is_torch_available():
    import torch

    from transformers import (
        FlexOlmoForCausalLM,
        FlexOlmoModel,
    )
    from transformers.models.flex_olmo.modeling_flex_olmo import FlexOlmoRotaryEmbedding


class FlexOlmoModelTester(CausalLMModelTester):
    if is_torch_available():
        config_class = FlexOlmoConfig
        base_model_class = FlexOlmoModel
        causal_lm_class = FlexOlmoForCausalLM


@require_torch
class FlexOlmoModelTest(CausalLMModelTest, unittest.TestCase):
    all_model_classes = (FlexOlmoModel, FlexOlmoForCausalLM) if is_torch_available() else ()
    pipeline_model_mapping = (
        {
            "feature-extraction": FlexOlmoModel,
            "text-generation": FlexOlmoForCausalLM,
        }
        if is_torch_available()
        else {}
    )
    test_headmasking = False
    test_pruning = False
    fx_compatible = False
    test_torchscript = False
    test_all_params_have_gradient = False
    model_tester_class = FlexOlmoModelTester
    rotary_embedding_layer = FlexOlmoRotaryEmbedding

    # Need to use `0.8` instead of `0.9` for `test_cpu_offload`
    # This is because we are hitting edge cases with the causal_mask buffer
    model_split_percents = [0.5, 0.7, 0.8]

    # used in `test_torch_compile_for_training`
    _torch_compile_train_cls = FlexOlmoForCausalLM if is_torch_available() else None

    @unittest.skip("Dynamic control flow in MoE")
    @pytest.mark.torch_compile_test
    def test_torch_compile_for_training(self):
        pass


@require_torch
class FlexOlmoIntegrationTest(unittest.TestCase):
    def setUp(self):
        cleanup(torch_device, gc_collect=True)

    def tearDown(self):
        cleanup(torch_device, gc_collect=True)

    @slow
    def test_model_7b_logits(self):
        input_ids = [[1, 306, 4658, 278, 6593, 310, 2834, 338]]
        model = FlexOlmoForCausalLM.from_pretrained("shanearora/Flex-reddit-2x7B-1T").to(
            torch_device, dtype=torch.bfloat16
        )
        out = model(torch.tensor(input_ids, device=torch_device)).logits.float()
        # Expected mean on dim = -1
        expectations = Expectations(
            {
                ("cuda", 8): [[-5.4202, -5.3883, -2.3924, -2.1226, -6.0122, -5.4173, -5.4571, -5.8256]],
            }
        )
        EXPECTED_MEAN = torch.tensor(expectations.get_expectation(), device=torch_device)
        torch.testing.assert_close(out.mean(-1), EXPECTED_MEAN, rtol=1e-2, atol=1e-2)
        # slicing logits[0, 0, 0:30]
        expectations = Expectations(
            {
                ("cuda", 8): [ 0.5547, -3.6250, -7.2812, -5.0312, -5.9062, -5.3438, -4.2500, -4.6875, -3.4219, -4.6250, -6.5938, -3.1250, -6.0625, -2.0781, -6.4688, -0.4941,  1.2656,  0.7578, -0.1934, -0.4160, -0.6992, -0.9531, -0.9648, -1.3125, -1.2578, -4.5625, -2.4219, -5.6250,  0.7695, -4.5938],
            }
        )  # fmt: skip
        EXPECTED_SLICE = torch.tensor(expectations.get_expectation(), device=torch_device)
        torch.testing.assert_close(out[0, 0, :30], EXPECTED_SLICE, rtol=1e-2, atol=1e-2)

    @slow
    def test_model_7b_greedy_generation(self):
        EXPECTED_TEXT_COMPLETION = """Simply put, the theory of relativity states that 1) the laws of physics are the same in all inertial frames of reference, and 2) the speed of light is constant in all inertial frames of reference. The first statement is called the principle of relativity, and the second is called the constancy of the speed of light. The first statement is"""
        prompt = "Simply put, the theory of relativity states that "
        tokenizer = AutoTokenizer.from_pretrained("allenai/dolma2-tokenizer", device_map="auto")
        model = FlexOlmoForCausalLM.from_pretrained("shanearora/Flex-reddit-2x7B-1T", device_map="auto")
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)

        # greedy generation outputs
        generated_ids = model.generate(input_ids, max_new_tokens=64, top_p=None, temperature=1, do_sample=False)
        text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        self.assertEqual(EXPECTED_TEXT_COMPLETION, text)
