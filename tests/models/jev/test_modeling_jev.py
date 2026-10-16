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
import unittest
from pathlib import Path

from huggingface_hub import hf_hub_download

from transformers import AutoProcessor, is_torch_available
from transformers.decision import DecisionConfig, DecisionTemplate
from transformers.testing_utils import require_torch, require_torch_accelerator, require_vision, slow

from ...test_decision_head_common import DecisionHeadTesterMixin, check_decision


if is_torch_available():
    from transformers import JevDecisionHead


@require_torch
class JevDecisionHeadTest(DecisionHeadTesterMixin, unittest.TestCase):
    all_model_classes = (JevDecisionHead,) if is_torch_available() else ()
    decision_repo = "ArthurZ/GEV-26B-Decide"
    tiny_sizes = {"hidden_size": 16}

# Code readouts of a Qwen3.5 checkpoint (perplexity-ai/pplx-decider-v1.1-27b, vllm-sr/d3), converted with
# `convert_decider_to_hf.py`: the backbone loads from the original repo
PPLX_DECIDER, D3 = "ArthurZ/pplx-decider-v1.1-27b", "ArthurZ/d3"


@require_torch
@require_vision
class DeciderTemplateTest(unittest.TestCase):
    def test_template_reproduces_the_model_code(self):
        for repo in (PPLX_DECIDER, D3):
            with self.subTest(repo):
                test = json.loads(Path(hf_hub_download(repo, "decision_test.json")).read_bytes())
                template = DecisionTemplate(AutoProcessor.from_pretrained(repo), DecisionConfig.from_pretrained(repo))
                prompts = template(test["request"])
                sequences = [{"input_ids": prompt.input_ids, "marks": prompt.marks} for prompt in prompts]
                self.assertEqual(sequences, test["sequences"])


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
