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
from pathlib import Path
from typing import Dict, Union

import numpy as np

from transformers import is_torch_available, is_vision_available
from transformers.agents.agent_types import AGENT_TYPE_MAPPING, AgentAudio, AgentImage, AgentText
from transformers.testing_utils import get_tests_dir, is_agent_test


if is_torch_available():
    import torch

if is_vision_available():
    from PIL import Image


AUTHORIZED_TYPES = ["text", "audio", "image", "any"]


def create_inputs(tool_inputs: Dict[str, Dict[Union[str, type], str]]):
    inputs = {}

    for input_name, input_desc in tool_inputs.items():
        input_type = input_desc["type"]

        if input_type == "text":
            inputs[input_name] = "Text input"
        elif input_type == "image":
            inputs[input_name] = Image.open(
                Path(get_tests_dir("fixtures/tests_samples/COCO")) / "000000039769.png"
            ).resize((512, 512))
        elif input_type == "audio":
            inputs[input_name] = np.ones(3000)
        else:
            raise ValueError(f"Invalid type requested: {input_type}")

    return inputs


def output_type(output):
    if isinstance(output, (str, AgentText)):
        return "text"
    elif isinstance(output, (Image.Image, AgentImage)):
        return "image"
    elif isinstance(output, (torch.Tensor, AgentAudio)):
        return "audio"
    else:
        raise TypeError(f"Invalid output: {output}")


@is_agent_test
class ToolTesterMixin:
    def test_inputs_output(self):
        self.assertTrue(hasattr(self.tool, "inputs"))
        self.assertTrue(hasattr(self.tool, "output_type"))

        inputs = self.tool.inputs
        self.assertTrue(isinstance(inputs, dict))

        for _, input_spec in inputs.items():
            self.assertTrue("type" in input_spec)
            self.assertTrue("description" in input_spec)
            self.assertTrue(input_spec["type"] in AUTHORIZED_TYPES)
            self.assertTrue(isinstance(input_spec["description"], str))

        output_type = self.tool.output_type
        self.assertTrue(output_type in AUTHORIZED_TYPES)

    def test_common_attributes(self):
        self.assertTrue(hasattr(self.tool, "description"))
        self.assertTrue(hasattr(self.tool, "name"))
        self.assertTrue(hasattr(self.tool, "inputs"))
        self.assertTrue(hasattr(self.tool, "output_type"))

    def test_agent_type_output(self):
        inputs = create_inputs(self.tool.inputs)
        output = self.tool(**inputs)
        agent_type = AGENT_TYPE_MAPPING[self.tool.output_type]
        self.assertTrue(isinstance(output, agent_type))

    def test_agent_types_inputs(self):
        inputs = create_inputs(self.tool.inputs)
        _inputs = []
        for _input, expected_input in zip(inputs, self.tool.inputs.values()):
            input_type = expected_input["type"]
            _inputs.append(AGENT_TYPE_MAPPING[input_type](_input))

        output_type = AGENT_TYPE_MAPPING[self.tool.output_type]

        # Should not raise an error
        output = self.tool(**inputs)
        self.assertTrue(isinstance(output, output_type))
