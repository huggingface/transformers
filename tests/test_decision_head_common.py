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

from huggingface_hub import hf_hub_download

from transformers import DecisionConfig, is_torch_available, pipeline
from transformers.decision_utils import DecisionPrompt, Question
from transformers.testing_utils import require_torch


if is_torch_available():
    import torch

    from transformers import AutoModel


def decision_prompt(marks: list[int], length: int = 12, questions: list[Question] | None = None) -> DecisionPrompt:
    """A prompt of `length` tokens with `marks`, for one question with three options unless `questions` are given."""
    choice = {"type": "choice", "instructions": "Pick one", "criteria": {"a": None, "b": None, "c": None}}
    return DecisionPrompt(
        list(range(length)), marks, questions or [Question.from_request("q", choice, DecisionConfig())]
    )


@require_torch
class DecisionHeadTesterMixin:
    """Common tests of the decision heads: subclasses set `all_model_classes`, a tiny `config` and `get_prompts`."""

    all_model_classes = ()
    config = None

    def get_prompts(self) -> list[DecisionPrompt]:
        raise NotImplementedError

    def test_scores_and_save_load(self):
        for model_class in self.all_model_classes:
            head = model_class(self.config).eval()
            with tempfile.TemporaryDirectory() as tmp:
                head.save_pretrained(tmp)
                loaded = AutoModel.from_pretrained(tmp, attn_implementation="eager").eval()
            for prompt in self.get_prompts():
                hidden_states = torch.randn(len(prompt.input_ids), self.config.hidden_size)
                lm_head = torch.nn.Linear(self.config.hidden_size, len(prompt.input_ids), bias=False)
                with torch.no_grad():
                    scores, loaded_scores = (
                        head(hidden_states, prompt, lm_head),
                        loaded(hidden_states, prompt, lm_head),
                    )
                self.assertEqual([len(score) for score in scores], [len(q.options) for q in prompt.questions])
                torch.testing.assert_close(scores, loaded_scores)


def check_decision(test_case, model: str, decision_repo: str, expected: dict, atol: float = 1e-3, **pipeline_kwargs):
    """The answers to the request of `decision_test.json` (probabilities, or `noul`) match `expected`."""
    request = json.load(open(hf_hub_download(decision_repo, "decision_test.json"), encoding="utf-8"))["request"]
    answers = pipeline("decision", model=model, decision_config=decision_repo, **pipeline_kwargs)(request)
    for name, value in expected.items():
        got = answers[name].get("probabilities", answers[name].get("noul"))
        if isinstance(value, dict):
            test_case.assertEqual(list(got), list(value))
            for option, probability in value.items():
                test_case.assertAlmostEqual(got[option], probability, delta=atol)
        else:
            test_case.assertAlmostEqual(got, value, delta=atol)
