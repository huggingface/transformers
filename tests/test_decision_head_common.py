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
from safetensors.torch import save_file

from transformers import AutoConfig, AutoTokenizer, DecisionConfig, is_torch_available, pipeline
from transformers.decision import DecisionTemplate
from transformers.decision.configuration_utils import DECISION_HEAD_CONFIG_NAME
from transformers.testing_utils import torch_device


if is_torch_available():
    import torch

    from transformers import AutoModel, Qwen2ForCausalLM


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
        cls.prompts = DecisionTemplate(cls.tokenizer, cls.decision_config)(cls.test["request"])
        cls.config = AutoConfig.from_pretrained(
            cls.decision_repo, config_file_name=DECISION_HEAD_CONFIG_NAME, **cls.tiny_sizes
        )
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
            return [model(**prompt.model_inputs()) for prompt in self.prompts]

    def test_scores_and_save_load(self):
        for model_class in self.all_model_classes:
            torch.manual_seed(0)
            model = model_class(self.config).eval()
            with tempfile.TemporaryDirectory() as tmp:
                model.save_pretrained(tmp)
                loaded = AutoModel.from_pretrained(tmp, attn_implementation="eager").eval()
            for prompt, scores, loaded_scores in zip(self.prompts, self.scores(model), self.scores(loaded)):
                torch.testing.assert_close(loaded_scores, scores)
                self.assertEqual(len(scores), sum(len(question.options) for question in prompt.questions))

    def test_from_the_repo_of_the_model(self):
        # The repo of the original model, its config and weights, with the files of the decision model next to them:
        # `decision_head.json` and the index of its weights, the shards of the model and the file of the head
        torch.manual_seed(0)
        model = self.all_model_classes[0](self.config).eval()
        with tempfile.TemporaryDirectory() as tmp:
            # The weights the backbone has (its model, and the `lm_head` of the heads reading it) are in its file, under
            # its names: `model.layers.0...` for a `ForCausalLM`, `layers.0...` for a base model
            state_dict, backbone = model.state_dict(), self.backbone_class(self.config.text_config)
            names = {
                key: next((name for name in (key, f"model.{key}") if name in state_dict), None)
                for key in backbone.state_dict()
            }
            backbone.load_state_dict({key: state_dict[name] for key, name in names.items() if name}, strict=False)
            backbone.save_pretrained(tmp)
            for component in (self.tokenizer, self.decision_config):
                component.save_pretrained(tmp)
            head = {key: value for key, value in state_dict.items() if key not in names.values()}
            save_file(head, f"{tmp}/head.safetensors", metadata={"format": "pt"})
            index = dict.fromkeys(backbone.state_dict(), "model.safetensors") | dict.fromkeys(head, "head.safetensors")
            Path(tmp, "decision.safetensors.index.json").write_text(
                json.dumps({"metadata": {}, "weight_map": index}), encoding="utf-8"
            )
            config = model.config
            config.transformers_weights = "decision.safetensors.index.json"
            config.to_json_file(f"{tmp}/{DECISION_HEAD_CONFIG_NAME}")
            loaded = (
                self.all_model_classes[0]
                .from_pretrained(tmp, config_file_name=DECISION_HEAD_CONFIG_NAME, attn_implementation="eager")
                .eval()
            )
            decide = pipeline("decision", model=tmp, model_kwargs={"attn_implementation": "eager"})
            answers = decide(self.test["request"])
        self.assertIsInstance(decide.model, self.all_model_classes[0])
        for scores, loaded_scores in zip(self.scores(model), self.scores(loaded)):
            torch.testing.assert_close(loaded_scores, scores)
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
