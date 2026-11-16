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
"""Exercise real Accelerate offload hooks without requiring accelerator hardware."""

import unittest

from transformers import KlmConfig, is_torch_available
from transformers.testing_utils import require_accelerate, require_torch


if is_torch_available():
    import torch

    from transformers import KlmModel


@require_torch
@require_accelerate
class KlmOffloadTest(unittest.TestCase):
    def test_offloaded_normalization_hooks_are_preserved(self):
        from accelerate import cpu_offload

        torch.manual_seed(41)
        config = KlmConfig(
            vocab_size=32,
            hidden_size=32,
            intermediate_size=48,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            linear_num_heads=2,
            linear_head_dim=8,
            num_bigrams=16,
            num_trigrams=8,
            ngram_embedding_size=4,
            layer_types=["linear_attention", "full_attention"],
            use_cache=False,
        )
        model = KlmModel(config).eval()
        for name, parameter in model.named_parameters():
            if name.endswith("_residual.query"):
                torch.nn.init.normal_(parameter, std=0.1)
        input_ids = torch.tensor([[0, 0, 3, 4, 2, 5, 6], [3, 2, 4, 5, 2, 6, 7]])
        attention_mask = input_ids.ne(0)
        with torch.no_grad():
            expected = model(input_ids, attention_mask=attention_mask).last_hidden_state

        # CPU is the execution device, but weights still become meta tensors between calls. This exercises
        # the same materialization hooks as GPU offload and catches passing a norm module through tensor kwargs.
        cpu_offload(model, execution_device=torch.device("cpu"))
        norms = [model.layers[0].input_layernorm, model.layers[0].post_attention_layernorm, model.norm]
        for _ in range(2):
            for norm in norms:
                self.assertTrue(norm.weight.is_meta)
            with torch.no_grad():
                actual = model(input_ids, attention_mask=attention_mask).last_hidden_state
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            for norm in norms:
                self.assertTrue(norm.weight.is_meta)
