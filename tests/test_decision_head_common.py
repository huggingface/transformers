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

from huggingface_hub import hf_hub_download

from transformers import AutoConfig, AutoTokenizer, DecisionConfig, is_torch_available, pipeline
from transformers.decision import DecisionTemplate
from transformers.testing_utils import torch_device


if is_torch_available():
    import torch

    from transformers import AutoModel, Qwen2ForCausalLM


class DecisionHeadTesterMixin:
    """
    Common tests of a decision head on `decision_repo`, the decision files of the most downloaded model with that head:
    the request of its `decision_test.json` with the input ids and marks that the code of the original model builds for
    it, and its head config at the `tiny_sizes` of the subclass, which also sets `all_model_classes`. The head reads a
    tiny `backbone_class` model.
    """

    backbone_class = Qwen2ForCausalLM if is_torch_available() else None

    @classmethod
    def setUpClass(cls):
        cls.test = json.loads(Path(hf_hub_download(cls.decision_repo, "decision_test.json")).read_bytes())
        cls.tokenizer = AutoTokenizer.from_pretrained(cls.test["model"], revision=cls.test["revision"])
        cls.decision_config = DecisionConfig.from_pretrained(cls.decision_repo)
        cls.prompts = DecisionTemplate(cls.tokenizer, cls.decision_config)(cls.test["request"])
        cls.config = AutoConfig.from_pretrained(cls.decision_repo, subfolder="decision_head", **cls.tiny_sizes)
        backbone_config = cls.backbone_class.config_class(
            vocab_size=len(cls.tokenizer),
            hidden_size=getattr(cls.config, "backbone_hidden_size", cls.config.hidden_size),
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
        )
        # Against sdpa: heads whose questions attend to each other would get different scores from a causal mask
        backbone_config._attn_implementation = "eager"
        torch.manual_seed(0)
        cls.backbone = cls.backbone_class(backbone_config).eval()

    def test_chat_template_reproduces_the_model_code(self):
        sequences = [{"input_ids": prompt.input_ids, "marks": prompt.marks} for prompt in self.prompts]
        self.assertEqual(sequences, self.test["sequences"])

    def test_scores_and_save_load(self):
        for model_class in self.all_model_classes:
            head = model_class(self.config).eval()
            with tempfile.TemporaryDirectory() as tmp:
                head.save_pretrained(tmp)
                loaded = AutoModel.from_pretrained(tmp)
            for prompt in self.prompts:
                inputs = prompt.model_inputs()
                with torch.no_grad():
                    scores = head(**inputs, backbone=self.backbone)
                    torch.testing.assert_close(loaded(**inputs, backbone=self.backbone), scores)
                self.assertEqual(len(scores), sum(len(question.options) for question in prompt.questions))

    def test_pipeline_from_pretrained(self):
        with tempfile.TemporaryDirectory() as tmp:
            # The model, its tokenizer and decision config, and the head in `decision_head/` in another dtype than the
            # model, like the float32 head of JEV
            for component in (self.backbone, self.tokenizer, self.decision_config):
                component.save_pretrained(tmp)
            head = self.all_model_classes[0](self.config).to(torch.bfloat16)
            head.save_pretrained(Path(tmp) / "decision_head")
            decide = pipeline("decision", model=tmp, dtype="float32")
            answers = decide(self.test["request"])
        self.assertIsInstance(decide.head, self.all_model_classes[0])
        self.assertEqual(decide.head.dtype, torch.bfloat16)
        self.assertEqual(list(answers), list(self.test["request"]["questions"]))
        for answer in answers.values():
            if "probabilities" in answer:
                self.assertAlmostEqual(sum(answer["probabilities"].values()), 1.0, places=5)


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
