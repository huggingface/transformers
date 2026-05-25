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
import unittest

from transformers import Qwen3TTSTokenizerSingleCodebookModel, is_torch_available
from transformers.testing_utils import require_torch


@require_torch
class Qwen3TTSTokenizerSingleCodebookModelTest(unittest.TestCase):
    all_model_classes = (Qwen3TTSTokenizerSingleCodebookModel,) if is_torch_available() else ()

    def test_placeholder(self):
        pass
