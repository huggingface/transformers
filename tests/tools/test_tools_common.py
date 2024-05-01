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

from pathlib import Path
from typing import List

from transformers import is_torch_available, is_vision_available
from transformers.testing_utils import get_tests_dir, is_tool_test
from transformers.tools.agent_types import AGENT_TYPE_MAPPING, AgentAudio, AgentImage, AgentText


if is_torch_available():
    import torch

if is_vision_available():
    from PIL import Image


authorized_types = ["text", "image", "audio"]


def create_inputs(input_types: List[str]):
    inputs = []

    for input_type in input_types:
        if input_type == "text":
            inputs.append("Text input")
        elif input_type == "image":
            inputs.append(
                Image.open(Path(get_tests_dir("fixtures/tests_samples/COCO")) / "000000039769.png").resize((512, 512))
            )
        elif input_type == "audio":
            inputs.append(torch.ones(3000))
        elif isinstance(input_type, list):
            inputs.append(create_inputs(input_type))
        else:
            raise ValueError(f"Invalid type requested: {input_type}")

    return inputs


def output_types(outputs: List):
    output_types = []

    for output in outputs:
        if isinstance(output, (str, AgentText)):
            output_types.append("text")
        elif isinstance(output, (Image.Image, AgentImage)):
            output_types.append("image")
        elif isinstance(output, (torch.Tensor, AgentAudio)):
            output_types.append("audio")
        else:
            raise ValueError(f"Invalid output: {output}")

    return output_types


@is_tool_test
class ToolTesterMixin:
    def test_inputs_outputs(self):
        self.assertTrue(hasattr(self.tool, "inputs"))
        self.assertTrue(hasattr(self.tool, "outputs"))

        inputs = self.tool.inputs
        for _input in inputs:
            if isinstance(_input, list):
                for __input in _input:
                    self.assertTrue(__input in authorized_types)
            else:
                self.assertTrue(_input in authorized_types)

        outputs = self.tool.outputs
        for _output in outputs:
            self.assertTrue(_output in authorized_types)

    def test_call(self):
        inputs = create_inputs(self.tool.inputs)
        outputs = self.tool(*inputs)

        # There is a single output
        if len(self.tool.outputs) == 1:
            outputs = [outputs]

        self.assertListEqual(output_types(outputs), self.tool.outputs)

    def test_common_attributes(self):
        self.assertTrue(hasattr(self.tool, "description"))
        self.assertTrue(hasattr(self.tool, "default_checkpoint"))
        self.assertTrue(self.tool.description.startswith("This is a tool that"))

    def test_agent_types_outputs(self):
        inputs = create_inputs(self.tool.inputs)
        outputs = self.tool(*inputs)

        if not isinstance(outputs, list):
            outputs = [outputs]

        self.assertEqual(len(outputs), len(self.tool.outputs))

        for output, output_type in zip(outputs, self.tool.outputs):
            agent_type = AGENT_TYPE_MAPPING[output_type]
            self.assertTrue(isinstance(output, agent_type))

    def test_agent_types_inputs(self):
        inputs = create_inputs(self.tool.inputs)

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
