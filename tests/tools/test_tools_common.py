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
from typing import List, Dict

from transformers import is_torch_available, is_vision_available
from transformers.testing_utils import get_tests_dir, is_tool_test
from transformers.agents.agent_types import AGENT_TYPE_MAPPING, AgentAudio, AgentImage, AgentText, INSTANCE_TYPE_MAPPING

if is_torch_available():
    import torch

if is_vision_available():
    from PIL import Image




def is_authorized_type(_type):
    authorized_types = {
        'text': (AgentText, str),
        'audio': (AgentAudio,),
        'image': (AgentImage, Image.Image),
    }

    for authorized_type in authorized_types.values():
        if _type in authorized_type:
            print(_type, 'is in', authorized_type)
            return True
    return False


def create_inputs(tool_inputs: Dict[str, Dict[str | type, str]]):
    input_types = {v['type'] for v in tool_inputs.values()}
    inputs = []
    for input_type in input_types:
        authorized_type = None

        for k, v in authorized_types.items():
            if input_type in v:
                authorized_type = k

        if authorized_type is None:
            raise ValueError(f"Invalid type requested: {input_type}")

        if authorized_type == "text":
            inputs.append("Text input")
        elif authorized_type == "image":
            inputs.append(
                Image.open(Path(get_tests_dir("fixtures/tests_samples/COCO")) / "000000039769.png").resize((512, 512))
            )
        elif authorized_type == "audio":
            inputs.append(torch.ones(3000))
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
        self.assertTrue(hasattr(self.tool, "output_type"))

        inputs = self.tool.inputs
        self.assertTrue(isinstance(inputs, dict))

        for input_name, input_spec in inputs.items():
            self.assertTrue('type' in input_spec)
            self.assertTrue('description' in input_spec)
            self.assertTrue(is_authorized_type(input_spec['type']))
            self.assertTrue(isinstance(input_spec['description'], str))

        output = self.tool.output_type
        self.assertTrue(type(output) == type)

    def test_common_attributes(self):
        self.assertTrue(hasattr(self.tool, "description"))
        self.assertTrue(hasattr(self.tool, "name"))
        self.assertTrue(hasattr(self.tool, "inputs"))
        self.assertTrue(hasattr(self.tool, "output_type"))

    def test_agent_types_outputs(self):
        inputs = create_inputs(self.tool.inputs)
        output = self.tool(*inputs)
        self.assertTrue(isinstance(output, self.tool.output_type))

    def test_agent_types_inputs(self):
        inputs = create_inputs(self.tool.inputs)

        _inputs = []

        for _input, expected_input in zip(inputs, self.tool.inputs.values()):
            input_type = expected_input['type']
            _inputs.append(INSTANCE_TYPE_MAPPING[input_type](_input))

        # Should not raise an error
        output = self.tool(*inputs)

        self.assertTrue(isinstance(output, self.tool.output_type))
