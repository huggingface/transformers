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

from safetensors.torch import load_file, save_file

from transformers import is_torch_available, pipeline
from transformers.testing_utils import require_torch, require_torch_accelerator, slow

from ...test_decision_head_common import DecisionHeadTesterMixin, check_decision


if is_torch_available():
    import torch

    from transformers import ClefForDecision


@require_torch
class ClefForDecisionTest(DecisionHeadTesterMixin, unittest.TestCase):
    all_model_classes = (ClefForDecision,) if is_torch_available() else ()
    decision_repo = "hf-internal-testing/decision-clef"
    # The backbone is wider than the head
    tiny_sizes = {
        "backbone_hidden_size": 24,
        "hidden_size": 16,
        "intermediate_size": 32,
        "num_hidden_layers": 1,
        "num_evidence_layers": 1,
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
    }

    def test_from_a_clef_checkpoint(self):
        # A clef checkpoint ships its head (`joint_head.safetensors`, original names) and its config
        # (`joint_head_config.json`) next to the weights of its model: read as they are, without any file of ours
        torch.manual_seed(0)
        model = ClefForDecision(self.config).eval()
        with tempfile.TemporaryDirectory() as saved, tempfile.TemporaryDirectory() as repo:
            model.save_pretrained(saved)  # under the names of the original checkpoint
            state_dict = load_file(f"{saved}/model.safetensors")
            head = {key: value for key, value in state_dict.items() if not key.startswith(("model.", "lm_head."))}
            save_file(head, f"{repo}/joint_head.safetensors", metadata={"format": "pt"})
            backbone = self.backbone_class(self.config.text_config)
            backbone.load_state_dict({k: v for k, v in state_dict.items() if k not in head}, strict=False)
            backbone.save_pretrained(repo)
            self.tokenizer.save_pretrained(repo)
            config = self.config
            sizes = {
                "hidden_size": config.backbone_hidden_size,
                "width": config.hidden_size,
                "routing_layers": config.num_evidence_layers,
                "layers": config.num_hidden_layers,
                "heads": config.num_attention_heads,
                "feedforward": config.intermediate_size,
            }
            Path(repo, "joint_head_config.json").write_text(json.dumps(sizes), encoding="utf-8")
            loaded = ClefForDecision.from_pretrained(repo, attn_implementation="eager").eval()
            decide = pipeline("decision", model=repo, model_kwargs={"attn_implementation": "eager"})
        for scores, loaded_scores in zip(self.scores(model), self.scores(loaded)):
            torch.testing.assert_close(loaded_scores, scores)
        self.assertIsInstance(decide.model, ClefForDecision)
        self.assertTrue(decide.template.config.sort_choices)


@slow
@require_torch_accelerator
class ClefIntegrationTest(unittest.TestCase):
    def test_clef_flash(self):
        expected = {
            "intent": {"other": 0.0093, "refund": 0.9907},
            "urgent": 0.0817,
            "tone": {"0": 0.9143, "1": 0.0857},
        }
        check_decision(self, "Cloudflare/clef-flash", "ArthurZ/clef-flash", expected, dtype="bfloat16")
