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

import unittest

from transformers import is_torch_available
from transformers.testing_utils import require_torch, torch_device


if is_torch_available():
    import torch

    from transformers import BertConfig, BertModel


class OutputCapturingTest(unittest.TestCase):
    @require_torch
    def test_output_hidden_states_layers(self):
        config = BertConfig(
            vocab_size=99,
            hidden_size=32,
            num_hidden_layers=4,
            num_attention_heads=4,
            intermediate_size=37,
        )
        model = BertModel(config).to(torch_device).eval()
        input_ids = torch.randint(0, config.vocab_size, (2, 7), device=torch_device)

        with torch.no_grad():
            full_outputs = model(input_ids=input_ids, output_hidden_states=True)
            sparse_outputs = model(input_ids=input_ids, output_hidden_states_layers=[1, 3])

        self.assertEqual(len(sparse_outputs.hidden_states), len(full_outputs.hidden_states))
        self.assertIsNone(sparse_outputs.hidden_states[0])
        self.assertIsNone(sparse_outputs.hidden_states[1])
        self.assertIsNone(sparse_outputs.hidden_states[3])
        torch.testing.assert_close(sparse_outputs.hidden_states[2], full_outputs.hidden_states[2])
        torch.testing.assert_close(sparse_outputs.hidden_states[4], full_outputs.hidden_states[4])
