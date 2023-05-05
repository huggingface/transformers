# coding=utf-8
# Copyright 2023 HuggingFace Inc.
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

from transformers.tools import TextToSpeechTool
import torch

from .test_tools_common import ToolTesterMixin
import unittest

import unittest

class TextToSpeechToolTester(unittest.TestCase, ToolTesterMixin):
    def setUp(self):
        self.tool = TextToSpeechTool()
        self.tool.setup()

    def test_exact_match_arg(self):
        result = self.tool("hey")
        self.assertTrue(torch.allclose(result[:3], torch.tensor([-0.00022915324370842427, -3.233053212170489e-05, -1.3283072803460527e-05])))

    def test_exact_match_kwarg(self):
        result = self.tool("hey")
        self.assertTrue(torch.allclose(result[:3], torch.tensor([-0.00022915324370842427, -3.233053212170489e-05, -1.3283072803460527e-05])))