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

from transformers import ClefConfig, DecisionConfig, is_torch_available
from transformers.decision_utils import Question
from transformers.testing_utils import require_torch, require_torch_accelerator, slow, torch_device

from ...test_decision_head_common import DecisionHeadTesterMixin, check_decision, decision_prompt


if is_torch_available():
    import torch

    from transformers import ClefDecisionHead


@require_torch
class ClefDecisionHeadTest(DecisionHeadTesterMixin, unittest.TestCase):
    all_model_classes = (ClefDecisionHead,) if is_torch_available() else ()
    config = ClefConfig(hidden_size=16, width=8, routing_layers=1, layers=1, heads=2, feedforward=16)

    def get_prompts(self):
        # One prompt for all the questions: a pair of marks around each instruction, then around each of its options
        config = DecisionConfig(sort_choices=True)
        requests = {
            "intent": {
                "type": "choice",
                "instructions": "What is wanted?",
                "criteria": {"refund": "money", "other": None},
            },
            "urgent": {"type": "noul", "instructions": "Urgent?"},
            "tone": {"type": "score", "instructions": "Tone", "criteria": ["calm", "neutral", "angry"]},
        }
        questions = [Question.from_request(name, request, config) for name, request in requests.items()]
        marks = [
            position for i in range(sum(1 + len(q.options) for q in questions)) for position in (5 + 4 * i, 7 + 4 * i)
        ]
        return [decision_prompt(marks, marks[-1] + 5, questions)]


@slow
@require_torch_accelerator
class ClefIntegrationTest(unittest.TestCase):
    def test_clef_flash(self):
        expected = {
            "intent": {"other": 0.0093, "refund": 0.9907},
            "urgent": 0.0817,
            "tone": {"0": 0.9143, "1": 0.0857},
        }
        check_decision(
            self, "Cloudflare/clef-flash", "ArthurZ/clef-flash", expected, dtype=torch.bfloat16, device=torch_device
        )
