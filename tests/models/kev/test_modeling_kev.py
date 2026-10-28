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
    from transformers import KevDecisionHead


@require_torch
class KevDecisionHeadTest(DecisionHeadTesterMixin, unittest.TestCase):
    all_model_classes = (KevDecisionHead,) if is_torch_available() else ()
    decision_repo = "ArthurZ/kev-4b"
    tiny_sizes = {"hidden_size": 16, "head_dim": 8}


@slow
@require_torch_accelerator
class KevIntegrationTest(unittest.TestCase):
    def test_kev_4b(self):
        expected = {
            "intent": {"refund": 0.8130, "track": 0.0053, "other": 0.1817},
            "urgent": 0.7992,
            "anger": {"0": 0.0519, "1": 0.3996, "2": 0.5484},
        }
        check_decision(self, "jaredpalmer/kev-4b", "ArthurZ/kev-4b", expected, dtype="float32")
