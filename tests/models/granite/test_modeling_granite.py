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
"""Testing suite for the PyTorch Granite model."""

import unittest

from transformers import GraniteConfig, is_torch_available
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
        GraniteForCausalLM,
        GraniteModel,
    )


class GraniteModelTester(CausalLMModelTester):
    config_class = GraniteConfig
    if is_torch_available():
        base_model_class = GraniteModel
        causal_lm_class = GraniteForCausalLM


@require_torch
class GraniteModelTest(CausalLMModelTest, unittest.TestCase):
    all_model_classes = (
        (
            GraniteModel,
            GraniteForCausalLM,
        )
        if is_torch_available()
        else ()
    )
    pipeline_model_mapping = (
        {
            "feature-extraction": GraniteModel,
            "text-generation": GraniteForCausalLM,
        }
        if is_torch_available()
        else {}
    )
    model_tester_class = GraniteModelTester


@require_torch_accelerator
class GraniteIntegrationTest(unittest.TestCase):
    @slow
    @require_read_token
    def test_model_3b_logits_bf16(self):
        input_ids = [1, 306, 4658, 278, 6593, 310, 2834, 338]

        model = GraniteForCausalLM.from_pretrained(
            "ibm/PowerLM-3b", device_map="auto", torch_dtype=torch.bfloat16, attn_implementation="eager"
        )

        with torch.no_grad():
            out = model(torch.tensor([input_ids]).to(torch_device))
        # Expected mean on dim = -1

        # fmt: off
        EXPECTED_MEANS = Expectations(
            {
                ("xpu", 3): torch.tensor([[-3.1406, -2.5469, -2.6250, -2.1250, -2.6250, -2.6562, -2.6875, -2.9688]]),
                ("cuda", 7): torch.tensor([[-1.9798, -3.1626, -2.8062, -2.3777, -2.7091, -2.2338, -2.5924, -2.3974]]),
                ("cuda", 8): torch.tensor([[-3.1406, -2.5469, -2.6250, -2.1250, -2.6250, -2.6562, -2.6875, -2.9688]]),
            }
        )
        EXPECTED_MEAN = EXPECTED_MEANS.get_expectation()

        torch.testing.assert_close(EXPECTED_MEAN.to(torch_device), out.logits.mean(-1).float(), rtol=1e-2, atol=1e-2)

        # slicing logits[0, 0, 0:15]
        EXPECTED_SLICES = Expectations(
            {
                ("xpu", 3): torch.tensor([[2.2031, -5.0625, -5.0625, -5.0625, -5.0625, -0.9180, -5.0625, -5.0625, -5.0625, -5.0625, -5.5312, -2.1719, -1.7891, -0.4922, -2.5469]]),
                ("cuda", 7): torch.tensor([[4.8750, -2.1875, -2.1875, -2.1875, -2.1875, -2.8438, -2.1875, -2.1875, -2.1875, -2.1875, -2.1875, -2.1875, -2.1875, -2.1875, -2.1875]]),
                ("cuda", 8): torch.tensor([[2.0938, -5.0312, -5.0312, -5.0312, -5.0312, -1.0469, -5.0312, -5.0312, -5.0312, -5.0312, -5.5625, -2.1875, -1.7891, -0.5820, -2.6250]]),
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
    @require_read_token
    def test_model_3b_logits(self):
        input_ids = [1, 306, 4658, 278, 6593, 310, 2834, 338]

        model = GraniteForCausalLM.from_pretrained("ibm/PowerLM-3b", device_map="auto", torch_dtype=torch.float16)

        with torch.no_grad():
            out = model(torch.tensor([input_ids]).to(torch_device))

        # fmt: off
        # Expected mean on dim = -1
        EXPECTED_MEANS = Expectations(
            {
                ("xpu", 3): torch.tensor([[-3.2693, -2.5957, -2.6234, -2.1675, -2.6386, -2.6850, -2.7039, -2.9656]]),
                ("cuda", 7): torch.tensor([[-2.0984, -3.1294, -2.8153, -2.3568, -2.7337, -2.2624, -2.6016, -2.4022]]),
                ("cuda", 8): torch.tensor([[-3.2934, -2.6019, -2.6258, -2.1691, -2.6394, -2.6876, -2.7032, -2.9688]]),
            }
        )
        # fmt: on
        EXPECTED_MEAN = EXPECTED_MEANS.get_expectation()

        torch.testing.assert_close(EXPECTED_MEAN.to(torch_device), out.logits.float().mean(-1), rtol=1e-2, atol=1e-2)
