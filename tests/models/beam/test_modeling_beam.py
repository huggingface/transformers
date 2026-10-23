# Copyright 2026 Reflection AI and the HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch Beam model."""

import tempfile
import unittest

from transformers import is_torch_available
from transformers.testing_utils import Expectations, require_torch, require_torch_accelerator, slow

from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester, torch_device
from ...test_memory_cleanup_mixin import MemoryCleanupMixin


if is_torch_available():
    import torch

    from transformers import BeamConfig, BeamForCausalLM, BeamModel


class BeamModelTester(CausalLMModelTester):
    if is_torch_available():
        base_model_class = BeamModel

    def __init__(self, parent, **kwargs):
        super().__init__(parent=parent, **kwargs)
        self.n_group = 2
        self.topk_group = 1
        self.layer_types = ["sliding_attention"] * (self.num_hidden_layers - 1) + ["full_attention"]
        self.mlp_layer_types = ["dense"] + ["sparse"] * (self.num_hidden_layers - 1)


@require_torch
class BeamModelTest(CausalLMModelTest, unittest.TestCase):
    model_tester_class = BeamModelTester


@slow
@require_torch_accelerator
class BeamIntegrationTest(MemoryCleanupMixin, unittest.TestCase):
    # TODO: replace with a tiny Beam checkpoint on the Hub once the model is public.
    @staticmethod
    def _dummy_model():
        """A seeded, randomly initialized Beam that keeps the released layer pattern (5 sliding and 2 full attention
        layers, 3 dense and 4 sparse MLPs) and loads through `from_pretrained`."""
        config = BeamConfig(
            vocab_size=1024,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=7,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            moe_intermediate_size=32,
            num_experts=16,
            num_experts_per_tok=4,
            n_group=4,
            topk_group=2,
            sliding_window=16,
        )
        torch.manual_seed(0)
        model = BeamForCausalLM(config)
        for name, tensor in [*model.named_parameters(), *model.named_buffers()]:
            # Zero at init: random values make the test cover the routing and attention gate biases.
            if name.endswith(("e_score_correction_bias", "gate_bias")):
                tensor.data.normal_(std=config.initializer_range)
        with tempfile.TemporaryDirectory() as tmpdir:
            model.save_pretrained(tmpdir)
            return BeamForCausalLM.from_pretrained(tmpdir, dtype=torch.bfloat16, device_map=torch_device)

    def test_small_model_logits_batched(self):
        input_ids = torch.LongTensor([[0, 0, 0, 0, 0, 0, 1, 2, 3], [1, 1, 2, 3, 4, 5, 6, 7, 8]]).to(torch_device)
        attention_mask = input_ids.ne(0).to(torch.long)
        model = self._dummy_model()

        expected_left_padded = Expectations(
            {
                ("cuda", (9, 0)): [[0.3164, 0.0057, -0.0845], [0.1074, -0.1699, 0.1133], [0.1816, -0.0270, 0.1270]],
            }
        )  # fmt: skip
        expected_unpadded = Expectations(
            {
                ("cuda", (9, 0)): [[0.2256, 0.0104, 0.1318], [0.0635, 0.1631, -0.0298], [0.0057, 0.1089, -0.0630]],
            }
        )  # fmt: skip

        logits = model(input_ids, attention_mask=attention_mask).logits
        torch.testing.assert_close(
            logits[0, -3:, -3:],
            torch.tensor(expected_left_padded.get_expectation(), device=torch_device),
            atol=1e-3,
            rtol=1e-3,
        )
        torch.testing.assert_close(
            logits[1, -3:, -3:],
            torch.tensor(expected_unpadded.get_expectation(), device=torch_device),
            atol=1e-3,
            rtol=1e-3,
        )

    def test_small_model_generation(self):
        expected_ids = Expectations(
            {
                ("cuda", (9, 0)): [625, 858, 858, 356, 443, 356, 978, 835, 850, 756, 816, 259, 625, 844, 502, 443, 502, 443, 502, 568, 338, 52, 356, 216, 502, 502, 502, 502, 502, 502, 502, 502],
            }
        )  # fmt: skip
        model = self._dummy_model()
        input_ids = torch.LongTensor([[1, 2, 3, 4, 5, 6, 7, 8]]).to(torch_device)

        # Longer than the sliding window, so the sliding layers drop the oldest tokens.
        generated_ids = model.generate(input_ids, max_new_tokens=32, do_sample=False)
        self.assertEqual(generated_ids[0, input_ids.shape[1] :].tolist(), expected_ids.get_expectation())
