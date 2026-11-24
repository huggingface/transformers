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

import torch

from transformers import GS1Config, GS1ForDecision
from transformers.loss.loss_gs1 import GS1ForDecisionLoss


class GS1DecisionLossTest(unittest.TestCase):
    all_model_classes = (GS1ForDecision,)

    def test_hybrid_backbone_with_activation_checkpointing(self):
        markers = ("<state>", "</state>", "<question>", "</question>", "<option>", "</option>", "<|decide|>")
        config = GS1Config(
            text_config={
                "vocab_size": 32,
                "hidden_size": 32,
                "intermediate_size": 64,
                "num_hidden_layers": 2,
                "layer_types": ["linear_attention", "full_attention"],
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "head_dim": 8,
                "linear_num_key_heads": 2,
                "linear_num_value_heads": 4,
                "linear_key_head_dim": 8,
                "linear_value_head_dim": 8,
            },
            vision_config={"depth": 1, "hidden_size": 32, "intermediate_size": 64, "num_heads": 4},
            decision={"projection_size": 16},
            structural_token_ids=dict(zip(markers, range(1, 8), strict=True)),
        )
        model = GS1ForDecision(config).train()
        model.activation_checkpointing_enable(activation_checkpointing_kwargs={"use_reentrant": False})
        input_ids = torch.tensor([[1, 9, 2, 3, 9, 4, 5, 9, 6, 5, 10, 6, 7]])
        output = model(input_ids, labels=torch.tensor([0.25, 0.75]))
        self.assertEqual(output.logits.shape, (2,))
        self.assertTrue(torch.isfinite(output.loss))
        output.loss.backward()
        self.assertTrue(torch.isfinite(model.get_input_embeddings().weight.grad).all())

    def test_loss_matches_distribution_nll_and_expected_brier(self):
        logits = torch.tensor([0.2, -0.1, 1.0, 0.5, -0.5], requires_grad=True)
        labels = torch.tensor([0.75, 0.25, 0.2, 0.3, 0.5])
        counts = torch.tensor([2, 3])
        weights = torch.tensor([2.0, 0.5])

        loss = GS1ForDecisionLoss(
            logits,
            labels,
            counts,
            loss_weights=weights,
            brier_weight=0.25,
        )

        expected = []
        offset = 0
        for count, weight in zip((2, 3), weights, strict=True):
            scores = logits[offset : offset + count]
            target = labels[offset : offset + count]
            probability = scores.softmax(-1)
            nll = -(target * scores.log_softmax(-1)).sum()
            brier = (probability - target).square().sum() + 1 - target.square().sum()
            expected.append((nll + 0.25 * brier) * weight)
            offset += count
        torch.testing.assert_close(loss, torch.stack(expected).mean())

        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())


if __name__ == "__main__":
    unittest.main()
