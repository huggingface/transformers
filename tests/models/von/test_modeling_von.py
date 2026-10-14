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
from transformers.decision import DecisionPrompt
from transformers.testing_utils import require_torch, require_torch_accelerator, slow

from ...test_decision_head_common import DecisionHeadTesterMixin, check_decision


if is_torch_available():
    import torch

    from transformers import ModernBertConfig, ModernBertModel, VonDecisionHead


@require_torch
class VonDecisionHeadTest(DecisionHeadTesterMixin, unittest.TestCase):
    all_model_classes = (VonDecisionHead,) if is_torch_available() else ()
    decision_repo = "ArthurZ/von"
    tiny_sizes = {"hidden_size": 16}
    backbone_class = ModernBertModel if is_torch_available() else None

    def test_options_only_see_the_prompt_and_themselves(self):
        # Two blocks of a prompt `3 4 5 6`, a `[MASK]` (1) before each option and `[SEP]` (2), where the option `7 8`
        # comes second then first: the prompt, `[SEP]` and that option keep their states if options only attend to the
        # prompt of their block and to themselves, at positions that restart after it (also for the sliding window)
        config = ModernBertConfig(
            vocab_size=len(self.tokenizer),
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=3,
            num_attention_heads=2,
            local_attention=8,
        )
        backbone, head = ModernBertModel(config).eval(), VonDecisionHead(self.config)
        input_ids = [3, 4, 5, 6, 1, 9, 1, 7, 8, 2] + [3, 4, 5, 6, 1, 7, 8, 1, 10, 2]
        # The first question of the request has two options
        prompt = DecisionPrompt(input_ids, [5, 7, 10, 15, 18, 20], self.prompts[0].questions)
        states = []
        for implementation in ("sdpa", "eager"):
            backbone.set_attn_implementation(implementation)
            with torch.no_grad():
                states.append(backbone(**head.backbone_inputs(backbone, **prompt.model_inputs())).last_hidden_state[0])
            first, second = states[-1].split(10)
            torch.testing.assert_close(first[[0, 1, 2, 3, 6, 7, 8, 9]], second[[0, 1, 2, 3, 4, 5, 6, 9]])
        torch.testing.assert_close(*states)


@slow
@require_torch_accelerator
class VonIntegrationTest(unittest.TestCase):
    def test_von(self):
        expected = {
            "intent": {"refund": 0.9158, "other": 0.0842},
            "urgent": 0.8436,
            "tone": {"0": 0.7911, "1": 0.2089},
        }
        check_decision(self, "wfzyx/von", "ArthurZ/von", expected, dtype="float32")
