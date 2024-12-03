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
from transformers.agents.agent_types import AGENT_TYPE_MAPPING

from .test_tools_common import ToolTesterMixin, output_type


class TranslationToolTester(unittest.TestCase, ToolTesterMixin):
    def setUp(self):
        self.tool = load_tool("translation")
        self.tool.setup()
        self.remote_tool = load_tool("translation", remote=True)

    def test_exact_match_arg(self):
        result = self.tool("Hey, what's up?", src_lang="English", tgt_lang="French")
        self.assertEqual(result, "- Hé, comment ça va?")

    def test_exact_match_kwarg(self):
        result = self.tool(text="Hey, what's up?", src_lang="English", tgt_lang="French")
        self.assertEqual(result, "- Hé, comment ça va?")

    def test_call(self):
        inputs = ["Hey, what's up?", "English", "Spanish"]
        output = self.tool(*inputs)

        self.assertEqual(output_type(output), self.tool.output_type)

    def test_agent_type_output(self):
        inputs = ["Hey, what's up?", "English", "Spanish"]
        output = self.tool(*inputs)
        output_type = AGENT_TYPE_MAPPING[self.tool.output_type]
        self.assertTrue(isinstance(output, output_type))

    def test_agent_types_inputs(self):
        example_inputs = {
            "text": "Hey, what's up?",
            "src_lang": "English",
            "tgt_lang": "Spanish",
        }

        _inputs = []
        for input_name in example_inputs.keys():
            example_input = example_inputs[input_name]
            input_description = self.tool.inputs[input_name]
            input_type = input_description["type"]
            _inputs.append(AGENT_TYPE_MAPPING[input_type](example_input))

        # Should not raise an error
        output = self.tool(**example_inputs)
        output_type = AGENT_TYPE_MAPPING[self.tool.output_type]
        self.assertTrue(isinstance(output, output_type))
