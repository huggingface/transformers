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

import unittest

from transformers import load_tool
from transformers.tools.agent_types import AGENT_TYPE_MAPPING

from .test_tools_common import ToolTesterMixin, output_types


class TranslationToolTester(unittest.TestCase, ToolTesterMixin):
    def setUp(self):
        self.tool = load_tool("translation")
        self.tool.setup()
        self.remote_tool = load_tool("translation", remote=True)

    def test_exact_match_arg(self):
        result = self.tool("Hey, what's up?", src_lang="English", tgt_lang="French")
        self.assertEqual(result, "- Hé, comment ça va?")

    def test_exact_match_arg_remote(self):
        result = self.remote_tool("Hey, what's up?", src_lang="English", tgt_lang="French")
        self.assertEqual(result, "- Hé, comment ça va?")

    def test_exact_match_kwarg(self):
        result = self.tool(text="Hey, what's up?", src_lang="English", tgt_lang="French")
        self.assertEqual(result, "- Hé, comment ça va?")

    def test_exact_match_kwarg_remote(self):
        result = self.remote_tool(text="Hey, what's up?", src_lang="English", tgt_lang="French")
        self.assertEqual(result, "- Hé, comment ça va?")

    def test_call(self):
        inputs = ["Hey, what's up?", "English", "Spanish"]
        outputs = self.tool(*inputs)

        # There is a single output
        if len(self.tool.outputs) == 1:
            outputs = [outputs]

        self.assertListEqual(output_types(outputs), self.tool.outputs)

    def test_agent_types_outputs(self):
        inputs = ["Hey, what's up?", "English", "Spanish"]
        outputs = self.tool(*inputs)

        if not isinstance(outputs, list):
            outputs = [outputs]

        self.assertEqual(len(outputs), len(self.tool.outputs))

        for output, output_type in zip(outputs, self.tool.outputs):
            agent_type = AGENT_TYPE_MAPPING[output_type]
            self.assertTrue(isinstance(output, agent_type))

    def test_agent_types_inputs(self):
        inputs = ["Hey, what's up?", "English", "Spanish"]

        _inputs = []

        for _input, input_type in zip(inputs, self.tool.inputs):
            if isinstance(input_type, list):
                _inputs.append([AGENT_TYPE_MAPPING[_input_type](_input) for _input_type in input_type])
            else:
                _inputs.append(AGENT_TYPE_MAPPING[input_type](_input))

        # Should not raise an error
        outputs = self.tool(*inputs)

        if not isinstance(outputs, list):
            outputs = [outputs]

        self.assertEqual(len(outputs), len(self.tool.outputs))
