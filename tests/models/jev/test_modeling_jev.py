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

import tempfile
import unittest

from transformers import DecisionConfig, JevConfig, is_torch_available
from transformers.decision_utils import DecisionPrompt, Question
from transformers.testing_utils import require_torch


if is_torch_available():
    import torch

    from transformers import AutoModel, JevDecisionHead


@require_torch
class JevDecisionHeadTest(unittest.TestCase):
    all_model_classes = (JevDecisionHead,) if is_torch_available() else ()

    def test_scores_and_save_load(self):
        question = {"type": "choice", "instructions": "Pick one", "criteria": {"a": None, "b": None, "c": None}}
        # One mark after the prompt
        prompt = DecisionPrompt(list(range(12)), [10], [Question.from_request("q", question, DecisionConfig())])
        head = JevDecisionHead(JevConfig(hidden_size=16, softcap=30.0)).eval()
        hidden_states = torch.randn(12, 16)
        with tempfile.TemporaryDirectory() as tmp:
            head.save_pretrained(tmp)
            loaded = AutoModel.from_pretrained(tmp).eval()
        with torch.no_grad():
            scores = head(hidden_states, prompt)
            torch.testing.assert_close(scores, loaded(hidden_states, prompt))
        self.assertEqual([score.shape for score in scores], [torch.Size([3])])
