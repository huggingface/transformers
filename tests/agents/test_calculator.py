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
from transformers.agents.agent_types import AGENT_TYPE_MAPPING, INSTANCE_TYPE_MAPPING, AgentText

from .test_tools_common import ToolTesterMixin, output_type


class CalculatorToolTester(unittest.TestCase, ToolTesterMixin):
    def setUp(self):
        self.tool = load_tool("calculator")
        self.tool.setup()

    def test_exact_match_arg(self):
        result = self.tool("(2 / 2) * 4")
        self.assertEqual(result, "4.0")

    def test_exact_match_kwarg(self):
        result = self.tool(expression="(2 / 2) * 4")
        self.assertEqual(result, "4.0")

    def test_agent_type_output(self):
        input = '2 * 2'
        output = self.tool(input)
        self.assertTrue(isinstance(output, AGENT_TYPE_MAPPING[self.tool.output_type]))

    def test_agent_types_inputs(self):
        inputs = ['2 * 2']
        _inputs = []

        for _input, expected_input in zip(inputs, self.tool.inputs.values()):
            input_type = expected_input['type']
            _inputs.append(AGENT_TYPE_MAPPING[input_type](_input))

        output = self.tool(*inputs)
        self.assertTrue(isinstance(output, AgentText))

        self.assertTrue(isinstance(output, AGENT_TYPE_MAPPING[self.tool.output_type]))
