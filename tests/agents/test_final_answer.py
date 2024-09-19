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
from pathlib import Path

import numpy as np
from PIL import Image

from transformers import is_torch_available
from transformers.agents.agent_types import AGENT_TYPE_MAPPING
from transformers.agents.default_tools import FinalAnswerTool
from transformers.testing_utils import get_tests_dir, require_torch

from .test_tools_common import ToolTesterMixin


if is_torch_available():
    import torch


class FinalAnswerToolTester(unittest.TestCase, ToolTesterMixin):
    def setUp(self):
        self.inputs = {"answer": "Final answer"}
        self.tool = FinalAnswerTool()

    def test_exact_match_arg(self):
        result = self.tool("Final answer")
        self.assertEqual(result, "Final answer")

    def test_exact_match_kwarg(self):
        result = self.tool(answer=self.inputs["answer"])
        self.assertEqual(result, "Final answer")

    def create_inputs(self):
        inputs_text = {"answer": "Text input"}
        inputs_image = {
            "answer": Image.open(Path(get_tests_dir("fixtures/tests_samples/COCO")) / "000000039769.png").resize(
                (512, 512)
            )
        }
        inputs_audio = {"answer": torch.Tensor(np.ones(3000))}
        return {"string": inputs_text, "image": inputs_image, "audio": inputs_audio}

    @require_torch
    def test_agent_type_output(self):
        inputs = self.create_inputs()
        for input_type, input in inputs.items():
            output = self.tool(**input)
            agent_type = AGENT_TYPE_MAPPING[input_type]
            self.assertTrue(isinstance(output, agent_type))

    @require_torch
    def test_agent_types_inputs(self):
        inputs = self.create_inputs()
        for input_type, input in inputs.items():
            output = self.tool(**input)
            agent_type = AGENT_TYPE_MAPPING[input_type]
            self.assertTrue(isinstance(output, agent_type))
