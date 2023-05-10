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
