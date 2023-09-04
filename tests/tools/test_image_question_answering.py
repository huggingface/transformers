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
from pathlib import Path

from transformers import is_vision_available, load_tool
from transformers.testing_utils import get_tests_dir

from .test_tools_common import ToolTesterMixin


if is_vision_available():
    from PIL import Image


class ImageQuestionAnsweringToolTester(unittest.TestCase, ToolTesterMixin):
    def setUp(self):
        self.tool = load_tool("image-question-answering")
        self.tool.setup()
        self.remote_tool = load_tool("image-question-answering", remote=True)

    def test_exact_match_arg(self):
        image = Image.open(Path(get_tests_dir("fixtures/tests_samples/COCO")) / "000000039769.png")
        result = self.tool(image, "How many cats are sleeping on the couch?")
        self.assertEqual(result, "2")

    def test_exact_match_arg_remote(self):
        image = Image.open(Path(get_tests_dir("fixtures/tests_samples/COCO")) / "000000039769.png")
        result = self.remote_tool(image, "How many cats are sleeping on the couch?")
        self.assertEqual(result, "2")

    def test_exact_match_kwarg(self):
        image = Image.open(Path(get_tests_dir("fixtures/tests_samples/COCO")) / "000000039769.png")
        result = self.tool(image=image, question="How many cats are sleeping on the couch?")
        self.assertEqual(result, "2")

    def test_exact_match_kwarg_remote(self):
        image = Image.open(Path(get_tests_dir("fixtures/tests_samples/COCO")) / "000000039769.png")
        result = self.remote_tool(image=image, question="How many cats are sleeping on the couch?")
        self.assertEqual(result, "2")
