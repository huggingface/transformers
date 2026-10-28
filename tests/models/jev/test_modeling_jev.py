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


# Code readouts of a Qwen3.5 checkpoint, converted with `convert_decider_to_hf.py`: the backbone loads from the
# original repo. The answers each converted head gives to the request of its `decision_test.json`
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
    "ArthurZ/d3-flash": {
        "refund": 0.9871,
        "spam": 0.0003,
        "tone": {"calm": 0.2152, "angry": 0.5176, "sad": 0.2672},
        "urgency": {"0": 0.0028, "1": 0.0026, "2": 0.6529, "3": 0.3417},
    },
    "ArthurZ/d3-mini": {
        "refund": 0.9793,
        "spam": 0.0003,
        "tone": {"calm": 0.3256, "angry": 0.5122, "sad": 0.1623},
        "urgency": {"0": 0.002, "1": 0.0037, "2": 0.5517, "3": 0.4426},
    },
    "ArthurZ/d3-nano": {
        "refund": 0.9981,
        "spam": 0.0002,
        "tone": {"calm": 0.2152, "angry": 0.6842, "sad": 0.1006},
        "urgency": {"0": 0.0409, "1": 0.0035, "2": 0.3306, "3": 0.625},
    },
    "ArthurZ/d3-edge": {
        "refund": 0.9871,
        "spam": 0.0101,
        "tone": {"calm": 0.5088, "angry": 0.2285, "sad": 0.2627},
        "urgency": {"0": 0.31, "1": 0.0586, "2": 0.2674, "3": 0.364},
    },
    "ArthurZ/d3-lite": {
        "refund": 0.9811,
        "spam": 0.0141,
        "tone": {"calm": 0.3707, "angry": 0.3955, "sad": 0.2338},
        "urgency": {"0": 0.2795, "1": 0.0179, "2": 0.3707, "3": 0.332},
    },
    "ArthurZ/Darwin-27B-ZTC": {
        "refund": 0.9793,
        "spam": 0.0184,
        "tone": {"calm": 0.2687, "angry": 0.6847, "sad": 0.0465},
        "urgency": {"0": 0.0014, "1": 0.0151, "2": 0.9057, "3": 0.0778},
    },
    "ArthurZ/Darwin-27B-ZTC-v2": {
        "refund": 0.9757,
        "spam": 0.0231,
        "tone": {"calm": 0.3083, "angry": 0.632, "sad": 0.0597},
        "urgency": {"0": 0.0025, "1": 0.0104, "2": 0.9457, "3": 0.0414},
    },
    "ArthurZ/Darwin-27B-ZTC-v5": {
        "refund": 0.9569,
        "spam": 0.0171,
        "tone": {"calm": 0.451, "angry": 0.451, "sad": 0.098},
        "urgency": {"0": 0.0019, "1": 0.0205, "2": 0.8876, "3": 0.09},
    },
    "ArthurZ/PhysicalEye-Decide-35B": {
        "refund": 0.9962,
        "spam": 0.0066,
        "tone": {"calm": 0.0321, "angry": 0.9119, "sad": 0.056},
        "urgency": {"0": 0.0074, "1": 0.0098, "2": 0.8503, "3": 0.1324},
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
        # System 1 merged into the weights, converted with `convert_jev_to_hf.py`
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
