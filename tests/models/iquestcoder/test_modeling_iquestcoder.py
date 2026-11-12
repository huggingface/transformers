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
"""Testing suite for the PyTorch IQuestCoder model."""

import unittest

from transformers import is_torch_available
from transformers.testing_utils import require_torch

from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester


if is_torch_available():
    from transformers import IQuestCoderModel


class IQuestCoderModelTester(CausalLMModelTester):
    if is_torch_available():
        base_model_class = IQuestCoderModel


@require_torch
class IQuestCoderModelTest(CausalLMModelTest, unittest.TestCase):
    model_tester_class = IQuestCoderModelTester
