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
from transformers.testing_utils import require_torch, require_torch_accelerator, slow

from ...test_decision_head_common import DecisionHeadTesterMixin, check_decision


if is_torch_available():
    from transformers import ClefForDecision


@require_torch
class ClefForDecisionTest(DecisionHeadTesterMixin, unittest.TestCase):
    all_model_classes = (ClefForDecision,) if is_torch_available() else ()
    decision_repo = "ArthurZ/clef-flash"
    # The backbone is wider than the head, and the tiny model is in float32, not in the dtype of the original head
    tiny_sizes = {
        "dtype": "float32",
        "backbone_hidden_size": 24,
        "hidden_size": 16,
        "intermediate_size": 32,
        "num_hidden_layers": 1,
        "num_evidence_layers": 1,
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
    }

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        # The output embeddings that the head reads are the input embeddings of the tiny model, which the repo of the
        # model holds next to the head
        cls.config.text_config.tie_word_embeddings = cls.config.tie_word_embeddings = True


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
