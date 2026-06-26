# Copyright 2026 JetBrains and The HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch Mellum model."""

import unittest

from transformers import is_torch_available
from transformers.testing_utils import (
    Expectations,
    cleanup,
    require_torch,
    require_torch_accelerator,
    slow,
    torch_device,
)


if is_torch_available():
    import torch

    from transformers import (
        AutoTokenizer,
        MellumForCausalLM,
        MellumModel,
    )
from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester


class MellumModelTester(CausalLMModelTester):
    if is_torch_available():
        base_model_class = MellumModel

    def __init__(self, parent):
        super().__init__(parent=parent)
        # Override for the TP plan tests.
        self.mlp_layer_types = ["dense", "sparse"]


@require_torch
class MellumModelTest(CausalLMModelTest, unittest.TestCase):
    test_all_params_have_gradient = False
    model_tester_class = MellumModelTester
    model_split_percents = [0.5, 0.8, 0.9]

    def test_load_balancing_loss(self):
        # Copied from Qwen3-Moe
        config, input_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.num_labels = 3
        config.num_experts = 3
        config.expert_interval = 2
        config.output_router_logits = True
        input_ids = input_dict["input_ids"]
        attention_mask = input_ids.ne(1).to(torch_device)
        model = MellumForCausalLM(config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=attention_mask)
        self.assertEqual(result.router_logits[0].shape, (91, config.num_experts))
        torch.testing.assert_close(
            result.aux_loss.cpu(),
            torch.tensor(2, dtype=torch.float32),
            rtol=1e-2,
            atol=1e-2,
        )

        pad_length = input_ids.shape[1] * 4
        padding_block = torch.ones(input_ids.shape[0], pad_length, dtype=torch.int32).to(torch_device)
        padded_input_ids = torch.cat((padding_block, input_ids), dim=1)
        padded_attention_mask = padded_input_ids.ne(1).to(torch_device)

        padded_result = model(padded_input_ids, attention_mask=padded_attention_mask)
        torch.testing.assert_close(result.aux_loss.cpu(), padded_result.aux_loss.cpu(), rtol=1e-4, atol=1e-4)

        include_padding_result = model(padded_input_ids, attention_mask=None)
        self.assertNotAlmostEqual(include_padding_result.aux_loss.item(), result.aux_loss.item())


# TODO(vasqu) fixup integration tests
@unittest.skip(reason="Weights will be available later")
@require_torch
class MellumIntegrationTest(unittest.TestCase):
    checkpoint = "JetBrains/Mellum2-12B-A2.5B-Base"

    def setUp(self):
        cleanup(torch_device, gc_collect=False)

    def tearDown(self):
        cleanup(torch_device, gc_collect=False)

    @slow
    @require_torch_accelerator
    def test_model_generation(self):
        expected_texts = Expectations(
            {
                ("cuda", 8): "def fibonacci(n):\n    if n == 0:\n        return 0\n    elif n == 1:\n        return 1\n    else:\n       ",
            }
        )  # fmt: skip
        expected_text = expected_texts.get_expectation()

        model = MellumForCausalLM.from_pretrained(self.checkpoint, dtype=torch.bfloat16, device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(self.checkpoint)

        prompt = "def fibonacci(n):"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        generated_ids = model.generate(**inputs, max_new_tokens=32, do_sample=False)
        output = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

        self.assertEqual(output, expected_text)
