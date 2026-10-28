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

from huggingface_hub import hf_hub_download
from safetensors.torch import save_file

from transformers import AutoProcessor, is_torch_available, pipeline
from transformers.decision import DecisionConfig, DecisionTemplate
from transformers.testing_utils import require_torch, require_torch_accelerator, require_vision, slow

from ...test_decision_head_common import DecisionHeadTesterMixin, check_decision


if is_torch_available():
    import torch

    from transformers import JevForDecision


@require_torch
class JevForDecisionTest(DecisionHeadTesterMixin, unittest.TestCase):
    all_model_classes = (JevForDecision,) if is_torch_available() else ()
    decision_repo = "ArthurZ/GEV-26B-Decide"
    tiny_sizes = {"hidden_size": 16}

    @require_vision
    def test_chat_template_reproduces_the_model_code(self):
        super().test_chat_template_reproduces_the_model_code()
        # The converted decider heads (`DECIDERS`), whose templates see images
        for repo in DECIDERS:
            with self.subTest(repo):
                test = json.loads(Path(hf_hub_download(repo, "decision_test.json")).read_bytes())
                template = DecisionTemplate(AutoProcessor.from_pretrained(repo), DecisionConfig.from_pretrained(repo))
                prompts = template(test["request"])
                sequences = [{"input_ids": prompt.input_ids, "marks": prompt.marks} for prompt in prompts]
                self.assertEqual(sequences, test["sequences"])

    def test_from_a_vllm_decision_checkpoint(self):
        # A vLLM decision model ships its head (`readout.safetensors`) and settings (`decision_config.json`) next to the
        # weights of its model: read as they are, without any file of ours
        with tempfile.TemporaryDirectory() as tmp:
            torch.manual_seed(0)
            backbone = self.backbone_class(self.config.text_config)
            backbone.save_pretrained(tmp)
            self.tokenizer.save_pretrained(tmp)
            codes, readout = ["A", "B", "C"], torch.randn(3, self.config.text_config.hidden_size)
            settings = {
                "codes": codes,
                "temperature": 2.0,
                "attention_mode": "noncausal_full_attention",
                "prompt": "d3",
            }
            Path(tmp, "decision_config.json").write_text(json.dumps(settings), encoding="utf-8")
            save_file({"weight": readout}, f"{tmp}/readout.safetensors", metadata={"format": "pt"})
            model = JevForDecision.from_pretrained(tmp)
            decide = pipeline("decision", model=tmp)
        self.assertEqual(model.config.num_verbalizers, 3)
        self.assertFalse(model.config.text_config.is_causal)
        torch.testing.assert_close(model.proj.weight, readout)
        self.assertIsInstance(decide.model, JevForDecision)
        self.assertEqual(decide.template.config.temperature, dict.fromkeys(["noul", "choice", "score"], 2.0))


# Code readouts of a Qwen3.5 checkpoint, read with the files of its repo. The answers each converted head gives to the request of its `decision_test.json`. One per prompt and
# attention mode: the other fine-tunes (d3-*, Darwin-*, PhysicalEye) share them
DECIDERS = {
    "ArthurZ/pplx-decider-v1.1-27b": {
        "refund": 0.9993,
        "spam": 0.0003,
        "tone": {"calm": 0.1327, "angry": 0.8002, "sad": 0.0671},
        "urgency": {"0": 0.0008, "1": 0.0058, "2": 0.7808, "3": 0.2126},
    },
    "ArthurZ/pplx-decider-v1-27b": {
        "refund": 0.9946,
        "spam": 0.0208,
        "tone": {"calm": 0.1099, "angry": 0.7974, "sad": 0.0927},
        "urgency": {"0": 0.0034, "1": 0.0051, "2": 0.9301, "3": 0.0614},
    },
    "ArthurZ/d3": {
        "refund": 0.9908,
        "spam": 0.0006,
        "tone": {"calm": 0.1407, "angry": 0.8094, "sad": 0.05},
        "urgency": {"0": 0.0007, "1": 0.0026, "2": 0.724, "3": 0.2727},
    },
}


@slow
@require_torch_accelerator
class JevIntegrationTest(unittest.TestCase):
    def test_jev_9b(self):
        expected = {
            "intent": {"refund": 0.9999, "track": 0.0000, "other": 0.0001},
            "urgent": 0.5419,
            "anger": {"0": 0.0165, "1": 0.1113, "2": 0.5163, "3": 0.3126, "4": 0.0419, "5": 0.0013},
        }
        check_decision(self, "ArthurZ/JEV-9B", "ArthurZ/JEV-9B", expected, atol=1e-2, dtype="bfloat16")

    def test_jev_35b(self):
        # System 1 merged into the weights: its head is rows of the `lm_head`, written to `decision_head.json` next to them
        expected = {
            "intent": {"refund": 0.9981, "track": 0.0001, "other": 0.0018},
            "urgent": 0.7539,
            "anger": {"0": 0.0155, "1": 0.0859, "2": 0.4147, "3": 0.3896, "4": 0.0872, "5": 0.007},
        }
        check_decision(self, "ArthurZ/JEV-35B", "ArthurZ/JEV-35B", expected, atol=1e-2)

    def test_deciders(self):
        for repo, expected in DECIDERS.items():
            with self.subTest(repo):
                check_decision(self, None, repo, expected, atol=1e-2)
