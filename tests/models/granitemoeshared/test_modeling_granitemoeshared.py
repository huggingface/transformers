# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch GraniteMoeShared model."""

import unittest

from transformers import AutoTokenizer, GraniteMoeSharedConfig, is_torch_available
from transformers.testing_utils import (
    Expectations,
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
        GraniteMoeSharedForCausalLM,
        GraniteMoeSharedModel,
    )


class GraniteMoeSharedModelTester(CausalLMModelTester):
    config_class = GraniteMoeSharedConfig
    if is_torch_available():
        base_model_class = GraniteMoeSharedModel
        causal_lm_class = GraniteMoeSharedForCausalLM


@require_torch
class GraniteMoeSharedModelTest(CausalLMModelTest, unittest.TestCase):
    all_model_classes = (
        (
            GraniteMoeSharedModel,
            GraniteMoeSharedForCausalLM,
        )
        if is_torch_available()
        else ()
    )
    pipeline_model_mapping = (
        {
            "feature-extraction": GraniteMoeSharedModel,
            "text-generation": GraniteMoeSharedForCausalLM,
        }
        if is_torch_available()
        else {}
    )
    model_tester_class = GraniteMoeSharedModelTester


@require_torch_accelerator
class GraniteMoeSharedIntegrationTest(unittest.TestCase):
    @slow
    @require_read_token
    def test_model_3b_logits(self):
        input_ids = [1, 306, 4658, 278, 6593, 310, 2834, 338]

        model = GraniteMoeSharedForCausalLM.from_pretrained("ibm/PowerMoE-3b", device_map="auto")

        with torch.no_grad():
            out = model(torch.tensor([input_ids]).to(torch_device))

        # fmt: off
        # Expected mean on dim = -1
        EXPECTED_MEANS = Expectations(
            {
                ("xpu", 3): torch.tensor([[-4.4005, -3.6689, -3.6187, -2.8308, -3.9871, -3.1001, -2.8738, -2.8063]]),
                ("cuda", 7): torch.tensor([[-2.2122, -1.6632, -2.9269, -2.3344, -2.0143, -3.0146, -2.6839, -2.5610]]),
                ("cuda", 8): torch.tensor([[-4.4005, -3.6689, -3.6187, -2.8308, -3.9871, -3.1001, -2.8738, -2.8063]]),
            }
        )

        EXPECTED_MEAN = EXPECTED_MEANS.get_expectation()
        torch.testing.assert_close(EXPECTED_MEAN.to(torch_device), out.logits.float().mean(-1), rtol=1e-2, atol=1e-2)

        # slicing logits[0, 0, 0:15]
        EXPECTED_SLICES = Expectations(
            {
                ("xpu", 3): torch.tensor([[2.5479, -9.2123, -9.2121, -9.2175, -9.2122, -1.5024, -9.2121, -9.2122, -9.2161, -9.2122, -6.3100, -3.6223, -3.6377, -5.2542, -5.2523]]),
                ("cuda", 7): torch.tensor([[4.8785, -2.2890, -2.2892, -2.2885, -2.2890, -3.5007, -2.2897, -2.2892, -2.2895, -2.2891, -2.2887, -2.2882, -2.2889, -2.2898, -2.2892]]),
                ("cuda", 8): torch.tensor([[2.5479, -9.2123, -9.2121, -9.2175, -9.2122, -1.5024, -9.2121, -9.2122, -9.2161, -9.2122, -6.3100, -3.6223, -3.6377, -5.2542, -5.2523]]),
            }
        )
        EXPECTED_SLICE = EXPECTED_SLICES.get_expectation()
        # fmt: on

        self.assertTrue(
            torch.allclose(
                EXPECTED_SLICE.to(torch_device),
                out.logits[0, 0, :15].float(),
                atol=1e-3,
                rtol=1e-3,
            )
        )

    @slow
    def test_model_3b_generation(self):
        # ground truth text generated with dola_layers="low", repetition_penalty=1.2
        # fmt: off
        EXPECTED_TEXT_COMPLETIONS = Expectations(
            {
                ("xpu", 3): (
                    "Simply put, the theory of relativity states that 1) the speed of light is constant, and 2) the speed of light is the same for all observers.\n\n"
                    "The first part is easy to understand. The second part is a little more difficult.\n\n"
                    "The second part of the theory of relativity is a little more difficult to understand.\n"
                ),
                ("cuda", 7): (
                    "Simply put, the theory of relativity states that \n$$\n\\frac{d^2x^\\mu}{d\\tau^2} = "
                    "\\frac{1}{c^2}\\frac{d^2x^\\mu}{dt^2}\n$$\nwhere $x^\\mu$ is a four-vector, $\\tau$ is the proper time"
                ),
                ("cuda", 8): (
                    "Simply put, the theory of relativity states that 1) the speed of light is constant, and 2) the speed of light is the same for all observers.\n\n"
                    "The first part is easy to understand. The second part is a little more difficult.\n\n"
                    "The second part of the theory of relativity is a little more difficult to understand.\n"
                ),
            }
        )
        # fmt: on
        EXPECTED_TEXT_COMPLETION = EXPECTED_TEXT_COMPLETIONS.get_expectation()

        prompt = "Simply put, the theory of relativity states that "
        tokenizer = AutoTokenizer.from_pretrained("ibm/PowerMoE-3b")
        model = GraniteMoeSharedForCausalLM.from_pretrained("ibm/PowerMoE-3b", device_map="auto")
        model_inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        # greedy generation outputs
        generated_ids = model.generate(**model_inputs, max_new_tokens=64, top_p=None, temperature=1, do_sample=False)
        text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

        self.assertEqual(EXPECTED_TEXT_COMPLETION, text)
