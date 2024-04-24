# coding=utf-8
# Copyright 2024 HuggingFace Inc.
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

from transformers import load_tool
from transformers.agents.agent_types import AGENT_TYPE_MAPPING, INSTANCE_TYPE_MAPPING

from .test_tools_common import ToolTesterMixin, output_types


class FinalAnswerToolTester(unittest.TestCase, ToolTesterMixin):
    def setUp(self):
        self.inputs = ['Final answer']
        self.tool = load_tool("final_answer")
        self.tool.setup()

    def test_exact_match_arg(self):
        result = self.tool(*self.inputs)
        self.assertEqual(result, "Final answer")

    def test_exact_match_kwarg(self):
        result = self.tool(args=self.inputs[0])
        self.assertEqual(result, "Final answer")
