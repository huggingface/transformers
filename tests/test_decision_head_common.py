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
from pathlib import Path
from types import SimpleNamespace

from huggingface_hub import hf_hub_download

from transformers import AutoConfig, AutoTokenizer, DecisionConfig, is_torch_available, pipeline
from transformers.testing_utils import torch_device


if is_torch_available():
    import torch

    from transformers import AutoModelForDecision, Qwen2Config, Qwen2ForCausalLM
    from transformers.decision.modeling_utils import read_marks


def render(processor, request, config=None):
    """
    The prompts of a decision `request`, as `apply_chat_template` renders them: their ids and marks without the
    sentinels (and their inputs without them, `stripped`), their inputs with them (what `decide` takes) and questions.
    """
    config = config or DecisionConfig()
    prompts = []
    for kwargs in config.chat_template_kwargs(request, processor):
        questions = kwargs.pop("decision_questions")
        inputs = processor.apply_chat_template(**kwargs, tokenize=True, return_dict=True, return_tensors="pt")
        stripped, marks = read_marks(dict(inputs), config._sentinel_ids)
        ids = stripped["input_ids"][0].tolist()
        prompt = {"input_ids": ids, "marks": marks.tolist(), "questions": questions, "stripped": stripped}
        prompts.append(SimpleNamespace(**prompt, inputs=dict(inputs)))
    return prompts


def tiny_qwen2(tokenizer) -> "Qwen2ForCausalLM":
    """A tiny Qwen2 with the vocabulary of `tokenizer`, the model of the decision tests."""
    config = Qwen2Config(
        vocab_size=len(tokenizer),
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
    )
    return Qwen2ForCausalLM(config).eval()


class DecisionHeadTesterMixin:
    """
    Common tests of a decision model on `decision_repo`, the decision files of the most downloaded model with that head:
    the request of its `decision_test.json` with the input ids and marks that the code of the original model builds for
    it, and its config at the `tiny_sizes` of the subclass (which also sets `all_model_classes`), reading a tiny
    `backbone_class` model.
    """

    backbone_class = Qwen2ForCausalLM if is_torch_available() else None

    @classmethod
    def setUpClass(cls):
        cls.test = json.loads(Path(hf_hub_download(cls.decision_repo, "decision_test.json")).read_bytes())
        cls.tokenizer = AutoTokenizer.from_pretrained(cls.test["model"], revision=cls.test["revision"])
        cls.decision_config = DecisionConfig.from_pretrained(cls.decision_repo)
        # The prompts of the request: the inputs `apply_chat_template` renders, and their questions
        cls.prompts = render(cls.tokenizer, cls.test["request"], cls.decision_config)
        config = json.loads(Path(hf_hub_download(cls.decision_repo, "decision_head.json")).read_bytes())
        cls.config = AutoConfig.for_model(**{**config, **cls.tiny_sizes})
        cls.config.text_config = cls.backbone_class.config_class(
            vocab_size=len(cls.tokenizer),
            hidden_size=getattr(cls.config, "backbone_hidden_size", cls.config.hidden_size),
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
        )
        # Against sdpa: heads whose questions attend to each other would get different scores from a causal mask
        cls.config._attn_implementation, cls.config.dtype = "eager", torch.float32

    def test_chat_template_reproduces_the_model_code(self):
        sequences = [{"input_ids": prompt.input_ids, "marks": prompt.marks} for prompt in self.prompts]
        self.assertEqual(sequences, self.test["sequences"])

    def scores(self, model):
        with torch.no_grad():
            return [
                model.decide(
                    **prompt.inputs, decision_questions=prompt.questions, decision_config=self.decision_config
                )
                for prompt in self.prompts
            ]

    def test_scores_and_save_load(self):
        for model_class in self.all_model_classes:
            torch.manual_seed(0)
            model = model_class(self.config).eval()
            with tempfile.TemporaryDirectory() as tmp:
                model.save_pretrained(tmp)
                loaded = AutoModelForDecision.from_pretrained(tmp, attn_implementation="eager").eval()
            for prompt, scores, loaded_scores in zip(self.prompts, self.scores(model), self.scores(loaded)):
                torch.testing.assert_close(loaded_scores, scores)
                self.assertEqual([len(s) for s in scores], [len(question.options) for question in prompt.questions])


def check_decision(
    test_case, model: str | None, decision_repo: str, expected: dict, atol: float = 1e-3, **pipeline_kwargs
):
    """The answers to the request of `decision_test.json` (probabilities in order, or `noul`) match `expected`."""
    test = json.loads(Path(hf_hub_download(decision_repo, "decision_test.json")).read_bytes())
    # By default the original checkpoint, which the decision files are converted from
    decide = pipeline(
        "decision", model=model or test["model"], decision_config=decision_repo, device=torch_device, **pipeline_kwargs
    )
    answers = decide(test["request"])
    for name, value in expected.items():
        answer = answers[name].get("probabilities", answers[name].get("noul"))
        if isinstance(value, dict):
            test_case.assertEqual(list(answer), list(value))
        torch.testing.assert_close(answer, value, atol=atol, rtol=0)
