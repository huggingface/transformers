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

from transformers.agents.agents import CodeAgent, ReactJSONAgent, ReactCodeAgent
from transformers.agents.default_tools import CalculatorTool


def get_new_path(suffix="") -> str:
    directory = tempfile.mkdtemp()
    return os.path.join(directory, str(uuid.uuid4()) + suffix)

def fake_react_json_llm(messages, stop=None) -> str:
    prompt = str(messages)

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


def fake_react_code_llm(messages, stop=None) -> str:
    prompt = str(messages)
    if "special_marker" not in prompt:
        return """
Thought: I should multiply 2 by 3.6452. special_marker
Code:
```py
result = 2**3.6452
print(result)
```<end_code>
"""
    else:  # We're at step 2
        return """
Thought: I can now answer the initial question
Code:
```py
final_answer(7.2904)
```<end_code>
"""


def fake_code_llm(messages, stop=None) -> str:
    return """
Thought: I should multiply 2 by 3.6452. special_marker
Code:
```py
result = calculator(expression="2*3.6452")
print(result)
```
"""

class AgentTests(unittest.TestCase):
    def test_fake_react_json_agent(self):
        agent = ReactJSONAgent(tools=[CalculatorTool()], llm_engine=fake_react_json_llm)
        output = agent.run("What is 2 multiplied by 3.6452?")
        assert output == "7.2904"
        print(agent.logs[-1])
        assert agent.logs[1] == "Task: What is 2 multiplied by 3.6452?"
        assert agent.logs[3] == "Observation: 7.2904"
        assert agent.logs[4] == """
Thought: I can now answer the initial question
Action:
{
    "action": "final_answer",
    "action_input": {"answer": "7.2904"}
}
"""
    def test_fake_react_code_agent(self):
        agent = ReactCodeAgent(tools=[CalculatorTool()], llm_engine=fake_react_code_llm)
        output = agent.run("What is 2 multiplied by 3.6452?")
        assert output == 7.2904
        assert agent.logs[1] == "Task: What is 2 multiplied by 3.6452?"
        assert agent.logs[3] == "Observation: 7.2904"

    def test_fake_code_agent(self):
        agent = CodeAgent(tools=[CalculatorTool()], llm_engine=fake_code_llm)
        output = agent.run("What is 2 multiplied by 3.6452?")
        assert output == '7.2904'

    def test_setup_agent_with_empty_toolbox(self):
        ReactJSONAgent(llm_engine=fake_react_json_llm, tools=[])
