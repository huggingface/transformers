# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for IQuestCoder, which uses the Llama implementation."""

import unittest

from huggingface_hub.errors import StrictDataclassClassValidationError

from transformers import IQuestCoderConfig, is_torch_available
from transformers.testing_utils import require_torch

from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester


if is_torch_available():
    from transformers import LlamaModel


class IQuestCoderModelTester(CausalLMModelTester):
    config_class = IQuestCoderConfig
    if is_torch_available():
        base_model_class = LlamaModel


@require_torch
class IQuestCoderModelTest(CausalLMModelTest, unittest.TestCase):
    model_tester_class = IQuestCoderModelTester

    def test_unsupported_options_are_rejected(self):
        # Released checkpoints set these to their disabled values, which must keep loading
        IQuestCoderConfig(clip_qkv=None, use_sliding_window=False, sliding_window=None, max_window_layers=0)
        with self.assertRaises(StrictDataclassClassValidationError):
            IQuestCoderConfig(clip_qkv=8.0)
        with self.assertRaises(StrictDataclassClassValidationError):
            IQuestCoderConfig(use_sliding_window=True, sliding_window=4096)
