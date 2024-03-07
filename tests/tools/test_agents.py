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
import os
import tempfile
import unittest
import uuid

from transformers.tools.agents import ReactAgent, CodeAgent
from transformers.tools.base import CalculatorTool


def get_new_path(suffix="") -> str:
    directory = tempfile.mkdtemp()
    return os.path.join(directory, str(uuid.uuid4()) + suffix)

def fake_react_llm(prompt: str, stop=None) -> str:
    if "special_marker" not in prompt:
        return """
Thought: I should multiply 2 by 3.6452. special_marker
Action:
{
    "action": "calculator",
    "action_input": {"expression": "2*3.6452"}
}
"""
    else:  # We're at step 2
        return """
Thought: I can now answer the initial question
Action:
{
    "action": "final_answer",
    "action_input": {"answer": "7.2904"}
}
"""

def fake_code_llm(prompt: str, stop=None) -> str:
    return """
Thought: I should multiply 2 by 3.6452. special_marker
Answer:
```py
result = calculator(expression="2*3.6452")
print(result)
```
"""

class AgentTests(unittest.TestCase):
    def test_fake_react_agent(self):
        agent = ReactAgent(fake_react_llm, toolbox=[CalculatorTool()])
        output = agent.run("What is 2 multiplied by 3.6452?")
        assert output == "7.2904"
        assert agent.memory[1] == "Task: What is 2 multiplied by 3.6452?"
        assert agent.memory[3] == "Observation: 7.2904"
        assert agent.memory[4] == """
Thought: I can now answer the initial question
Action:
{
    "action": "final_answer",
    "action_input": {"answer": "7.2904"}
}
"""

    def test_fake_code_agent(self):
        agent = CodeAgent(fake_code_llm, toolbox=[CalculatorTool()])
        output = agent.run("What is 2 multiplied by 3.6452?")
        assert output == "7.2904"

    def test_setup_agent_with_empty_toolbox():
        agent = ReactAgent(fake_react_llm, toolbox=[])
        assert False # TODO: finish this
