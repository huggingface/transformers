# Copyright 2026 The HuggingFace Team. All rights reserved.
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
"""Testing suite for the PyTorch Mamba3 model."""

import unittest

from transformers import is_torch_available
from transformers.testing_utils import require_torch, slow, torch_device

from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester
from ...test_configuration_common import ConfigTester


if is_torch_available():
    import torch

    from transformers import AutoTokenizer, Mamba3Config, Mamba3ForCausalLM, Mamba3Model


class Mamba3ModelTester(CausalLMModelTester):
    if is_torch_available():
        config_class = Mamba3Config
        base_model_class = Mamba3Model
        causal_lm_class = Mamba3ForCausalLM

    def __init__(self, parent):
        super().__init__(parent=parent)
        self.num_heads = 8
        self.head_dim = 8
        self.state_size = 16
        self.n_groups = 2
        self.chunk_size = 4


@require_torch
class Mamba3ModelTest(CausalLMModelTest, unittest.TestCase):
    model_tester_class = Mamba3ModelTester
    has_attentions = False

    def setUp(self):
        super().setUp()
        self.config_tester = ConfigTester(
            self, config_class=Mamba3Config, common_properties=["hidden_size", "num_hidden_layers"]
        )

    def _get_recurrent_state_shape(self, batch_size: int, config):
        # angle, ssm, k and v states
        num_rope_angles = int(config.state_size * config.rope_fraction) // 2
        return [
            (batch_size, config.num_heads, num_rope_angles),
            (batch_size, config.num_heads, config.head_dim, config.state_size),
            (batch_size, config.num_heads, config.state_size),
            (batch_size, config.num_heads, config.head_dim),
        ]

    def test_cached_decode_matches_full_forward(self):
        config, inputs = self.model_tester.prepare_config_and_inputs_for_common()
        model = Mamba3ForCausalLM(config).to(torch_device).eval()
        input_ids = inputs["input_ids"]
        with torch.no_grad():
            full = model(input_ids).logits
            out = model(input_ids[:, :-3], use_cache=True)
            logits = [out.logits]
            for idx in range(input_ids.shape[1] - 3, input_ids.shape[1]):
                out = model(input_ids[:, idx : idx + 1], cache=out.cache, use_cache=True)
                logits.append(out.logits)
        torch.testing.assert_close(torch.cat(logits, dim=1), full, rtol=1e-4, atol=1e-4)


@require_torch
@slow
class Mamba3IntegrationTest(unittest.TestCase):
    model_id = "hf-internal-testing/mamba3-siso-187m"

    def test_greedy_generation(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        model = Mamba3ForCausalLM.from_pretrained(self.model_id, dtype=torch.float32).to(torch_device)
        inputs = tokenizer("The capital of France is Paris. The capital of Germany is", return_tensors="pt").to(
            torch_device
        )
        output = model.generate(**inputs, max_new_tokens=15, do_sample=False)
        expected = "<|begin_of_text|>The capital of France is Paris. The capital of Germany is Berlin. The capital of Italy is Rome. The capital of Spain is Madrid"
        self.assertEqual(tokenizer.decode(output[0], skip_special_tokens=False), expected)
