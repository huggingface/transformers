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


@require_torch_gpu
class GraniteMoeHybridIntegrationTest(unittest.TestCase):
    @slow
    def test_model_logits(self):
        input_ids = [31390, 631, 4162, 30, 322, 25342, 432, 1875, 43826, 10066, 688, 225]

        model = GraniteMoeHybridForCausalLM.from_pretrained(
            "ibm-granite/granite-4.0-tiny-preview",
            device_map="auto",
            torch_dtype="bfloat16",
        )

        with torch.no_grad():
            out = model(torch.tensor([input_ids]).to(torch_device))

        # fmt: off
        # Expected mean on dim = -1
        EXPECTED_MEAN = torch.tensor([
            [-2.0041,  1.0344,  8.3842, 10.2653,  9.9950,  6.9927,  7.5597, -0.7904, 5.4209,  6.5575,  4.9957,  1.2563],
        ])

        torch.testing.assert_close(EXPECTED_MEAN.to(torch_device), out.logits.float().mean(-1), atol=1, rtol=1e-3)

        EXPECTED_SLICE = torch.tensor([
            [ 4.0000, -5.2500, -5.1562, -5.2188, -5.1250,  2.7344, -5.0000, -5.1250, -5.1250, -5.1250, -2.6875, -2.7031, -3.1250, -4.2500, -1.9609],
        ])
        # fmt: on

        self.assertTrue(
            torch.allclose(
                EXPECTED_SLICE.to(torch_device),
                out.logits[0, 0, :15].float(),
                atol=1,
                rtol=1e-3,
            )
        )

    @slow
    def test_model_generation(self):
        EXPECTED_TEXT_COMPLETION = "Simply put, the theory of relativity states that 1) the laws of physics are the same for all observers in uniform motion"
        prompt = "Simply put, the theory of relativity states that "
        tokenizer = AutoTokenizer.from_pretrained("ibm-granite/granite-4.0-tiny-preview")
        model = GraniteMoeHybridForCausalLM.from_pretrained(
            "ibm-granite/granite-4.0-tiny-preview",
            device_map="auto",
            torch_dtype="bfloat16",
        )
        model_inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        # greedy generation outputs
        generated_ids = model.generate(**model_inputs, max_new_tokens=16, do_sample=False)
        text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

        self.assertEqual(EXPECTED_TEXT_COMPLETION, text)
