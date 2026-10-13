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

import json
import tempfile
import unittest
from pathlib import Path

from transformers import ClefConfig, DecisionConfig, is_torch_available
from transformers.decision_utils import DecisionPrompt, Question
from transformers.testing_utils import get_tests_dir, require_torch


if is_torch_available():
    import torch

    from transformers import AutoModel, ClefDecisionHead


@require_torch
class ClefDecisionHeadTest(unittest.TestCase):
    all_model_classes = (ClefDecisionHead,) if is_torch_available() else ()

    def test_scores_and_save_load(self):
        # Prompts built by the model's own code, see tests/pipelines/test_pipelines_decision.py
        fixture = json.loads(Path(get_tests_dir("fixtures/decision"), "clef.json").read_text(encoding="utf-8"))
        config = DecisionConfig(**fixture["decision_config"])
        questions = {qid: Question.from_request(qid, q, config) for qid, q in fixture["request"]["questions"].items()}
        groups = [[q] for q in questions.values()] if len(fixture["sequences"]) > 1 else [list(questions.values())]
        head = ClefDecisionHead(
            ClefConfig(hidden_size=16, width=8, routing_layers=1, layers=1, heads=2, feedforward=16)
        ).eval()
        lm_head = torch.nn.Linear(16, 1 + max(max(s["input_ids"]) for s in fixture["sequences"]), bias=False)
        with tempfile.TemporaryDirectory() as tmp:
            head.save_pretrained(tmp)
            loaded = AutoModel.from_pretrained(tmp).eval()
        for sequence, group in zip(fixture["sequences"], groups):
            prompt = DecisionPrompt(sequence["input_ids"], sequence["marks"], group)
            hidden_states = torch.randn(len(prompt.input_ids), 16)
            with torch.no_grad():
                scores, loaded_scores = head(hidden_states, prompt, lm_head), loaded(hidden_states, prompt, lm_head)
            self.assertEqual([len(s) for s in scores], [len(q.options) for q in group])
            for s, loaded_s in zip(scores, loaded_scores):
                torch.testing.assert_close(s, loaded_s)
