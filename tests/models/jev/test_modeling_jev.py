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
    from transformers import JevDecisionHead


@require_torch
class JevDecisionHeadTest(DecisionHeadTesterMixin, unittest.TestCase):
    all_model_classes = (JevDecisionHead,) if is_torch_available() else ()
    decision_repo = "ArthurZ/GEV-26B-Decide"
    tiny_sizes = {"hidden_size": 16}


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
