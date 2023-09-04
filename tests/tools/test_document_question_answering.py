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

from datasets import load_dataset

from transformers import load_tool

from .test_tools_common import ToolTesterMixin


class DocumentQuestionAnsweringToolTester(unittest.TestCase, ToolTesterMixin):
    def setUp(self):
        self.tool = load_tool("document-question-answering")
        self.tool.setup()
        self.remote_tool = load_tool("document-question-answering", remote=True)

    def test_exact_match_arg(self):
        dataset = load_dataset("hf-internal-testing/example-documents", split="test")
        document = dataset[0]["image"]

        result = self.tool(document, "When is the coffee break?")
        self.assertEqual(result, "11-14 to 11:39 a.m.")

    def test_exact_match_arg_remote(self):
        dataset = load_dataset("hf-internal-testing/example-documents", split="test")
        document = dataset[0]["image"]

        result = self.remote_tool(document, "When is the coffee break?")
        self.assertEqual(result, "11-14 to 11:39 a.m.")

    def test_exact_match_kwarg(self):
        dataset = load_dataset("hf-internal-testing/example-documents", split="test")
        document = dataset[0]["image"]

        self.tool(document=document, question="When is the coffee break?")

    def test_exact_match_kwarg_remote(self):
        dataset = load_dataset("hf-internal-testing/example-documents", split="test")
        document = dataset[0]["image"]

        result = self.remote_tool(document=document, question="When is the coffee break?")
        self.assertEqual(result, "11-14 to 11:39 a.m.")
