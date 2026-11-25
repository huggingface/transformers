# Copyright 2026 the HuggingFace Team. All rights reserved.
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
"""Testing suite for the PyTorch Nanbeige model."""

import unittest

from transformers import is_torch_available
from transformers.testing_utils import require_torch, torch_device

from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester


if is_torch_available():
    import torch

    from transformers import NanbeigeForCausalLM, NanbeigeModel


class NanbeigeModelTester(CausalLMModelTester):
    if is_torch_available():
        base_model_class = NanbeigeModel

    # Shared tests expect one cache slot per layer; multiple loops are tested below.
    small_config_overrides = {"num_loops": 1}


@require_torch
class NanbeigeModelTest(CausalLMModelTest, unittest.TestCase):
    model_tester_class = NanbeigeModelTester

    def test_two_loop_chunked_prefill_and_decode_match_full_forward(self):
        tester = self.model_tester_class(self, batch_size=2, num_loops=2)
        config, inputs_dict = tester.prepare_config_and_inputs_for_common()
        config._attn_implementation = "eager"
        model = NanbeigeForCausalLM(config).to(torch_device).eval()
        input_ids = inputs_dict["input_ids"]
        num_cache_layers = config.num_hidden_layers * config.num_loops

        with torch.no_grad():
            expected_logits = model(input_ids, use_cache=False).logits
            outputs = model(input_ids[:, :2], use_cache=True)
            logits = [outputs.logits]
            self.assertEqual(len(outputs.cache.layers), num_cache_layers)
            for layer_idx in range(num_cache_layers):
                self.assertEqual(outputs.cache.get_seq_length(layer_idx), 2)

            outputs = model(input_ids[:, 2:4], cache=outputs.cache, use_cache=True)
            logits.append(outputs.logits)
            for position in range(4, input_ids.shape[1]):
                outputs = model(input_ids[:, position : position + 1], cache=outputs.cache, use_cache=True)
                logits.append(outputs.logits)

        torch.testing.assert_close(torch.cat(logits, dim=1), expected_logits, rtol=1e-4, atol=1e-4)
        for layer_idx in range(num_cache_layers):
            self.assertEqual(outputs.cache.get_seq_length(layer_idx), input_ids.shape[1])
