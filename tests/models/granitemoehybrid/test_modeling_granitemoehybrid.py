# coding=utf-8
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
"""Testing suite for the PyTorch GraniteMoeHybrid model."""

import unittest

import pytest

from transformers import (
    AutoTokenizer,
    GraniteMoeHybridConfig,
    is_torch_available,
)
from transformers.testing_utils import (
    require_torch,
    require_torch_gpu,
    slow,
    torch_device,
)

from ...generation.test_utils import GenerationTesterMixin
from ...models.bamba.test_modeling_bamba import BambaModelTest, BambaModelTester


if is_torch_available():
    import torch

    from transformers import (
        GraniteMoeHybridForCausalLM,
        GraniteMoeHybridModel,
    )


class GraniteMoeHybridModelTester(BambaModelTester):
    config_class = GraniteMoeHybridConfig
    if is_torch_available():
        model_class = GraniteMoeHybridModel
        for_causal_lm_class = GraniteMoeHybridForCausalLM

    def __init__(
        self,
        parent,
        use_cache=False,
        shared_intermediate_size=174,
        layer_types=None,
    ):
        super().__init__(parent)
        self.shared_intermediate_size = shared_intermediate_size
        self.layer_types = layer_types
        self.use_cache = use_cache

    def _update_layer_configs(self):
        super()._update_layer_configs()
        # GraniteMoeHybrid uses layer_types instead of attn_layer_indices
        self.layer_types = ["mamba"] * self.num_hidden_layers
        for idx in self.attn_layer_indices:
            self.layer_types[idx] = "attention"

    def get_config(self):
        return super().get_config(
            shared_intermediate_size=self.shared_intermediate_size,
            layer_types=self.layer_types,
        )


@require_torch
class GraniteMoeHybridModelTest(BambaModelTest, GenerationTesterMixin, unittest.TestCase):
    model_tester_class = GraniteMoeHybridModelTester
    all_model_classes = (
        (
            GraniteMoeHybridModel,
            GraniteMoeHybridForCausalLM,
        )
        if is_torch_available()
        else ()
    )
    pipeline_model_mapping = (
        {
            "feature-extraction": GraniteMoeHybridModel,
            "text-generation": GraniteMoeHybridForCausalLM,
        }
        if is_torch_available()
        else {}
    )

    def test_config_requires_mamba_or_attention_layers(self):
        """Ensure we can't create a config with disallowed layers."""
        with pytest.raises(ValueError):
            GraniteMoeHybridConfig(layer_types=["not allowed!"])


# TODO (@alex-jw-brooks) - update this once the model(s) are out
@unittest.skip(reason="GraniteMoeHybrid models are not yet released")
@require_torch_gpu
class GraniteMoeHybridIntegrationTest(unittest.TestCase):
    @slow
    def test_model_logits(self):
        input_ids = [31390, 631, 4162, 30, 322, 25342, 432, 1875, 43826, 10066, 688, 225]

        model = GraniteMoeHybridForCausalLM.from_pretrained("ibm-granite/granite-4.0-tiny", device_map="auto")

        with torch.no_grad():
            out = model(torch.tensor([input_ids]).to(torch_device))

        # fmt: off
        # Expected mean on dim = -1
        EXPECTED_MEAN = torch.tensor([
            [-2.9711, -2.2554, -1.0814, -1.6123, -0.8780, -1.0685, -0.6368, -1.9732, -3.3548, -2.6895, -2.3062, -2.6338]
        ])

        torch.testing.assert_close(EXPECTED_MEAN.to(torch_device), out.logits.float().mean(-1), rtol=1e-2, atol=1e-2)

        # slicing logits[0, 0, 0:15]
        EXPECTED_SLICE = torch.tensor([
            [4.0662, 5.9547, 3.5803, 3.1306, 4.3211, 3.8902, 4.6438, 8.5434, 7.5865, 5.1623, 5.2240, 9.2982, 5.9094, 6.8834, 5.7551],
        ])
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
    def test_model_generation(self):
        EXPECTED_TEXT_COMPLETION = (
            "Simply put, the theory of relativity states that 1) time is relative, and 2) space is relative. The first"
        )
        prompt = "Simply put, the theory of relativity states that "
        tokenizer = AutoTokenizer.from_pretrained("ibm-granite/granite-4.0-tiny")
        model = GraniteMoeHybridForCausalLM.from_pretrained("ibm-granite/granite-4.0-tiny", device_map="auto")
        model_inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        # greedy generation outputs
        generated_ids = model.generate(**model_inputs, max_new_tokens=16, do_sample=False)
        text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

        self.assertEqual(EXPECTED_TEXT_COMPLETION, text)
