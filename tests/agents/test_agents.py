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
import os
import tempfile
import unittest
import uuid

import pytest

from transformers.agents.agent_types import AgentText
from transformers.agents.agents import (
    AgentMaxIterationsError,
    CodeAgent,
    ManagedAgent,
    ReactCodeAgent,
    ReactJsonAgent,
    Toolbox,
)
from transformers.agents.default_tools import PythonInterpreterTool
from transformers.testing_utils import require_torch


def get_new_path(suffix="") -> str:
    directory = tempfile.mkdtemp()
    return os.path.join(directory, str(uuid.uuid4()) + suffix)


def fake_react_json_llm(messages, stop_sequences=None, grammar=None) -> str:
    prompt = str(messages)

    if "special_marker" not in prompt:
        return """
Thought: I should multiply 2 by 3.6452. special_marker
Action:
{
    "action": "python_interpreter",
    "action_input": {"code": "2*3.6452"}
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


def fake_react_code_llm(messages, stop_sequences=None, grammar=None) -> str:
    prompt = str(messages)
    if "special_marker" not in prompt:
        return """
Thought: I should multiply 2 by 3.6452. special_marker
Code:
```py
result = 2**3.6452
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


def fake_react_code_llm_error(messages, stop_sequences=None) -> str:
    prompt = str(messages)
    if "special_marker" not in prompt:
        return """
Thought: I should multiply 2 by 3.6452. special_marker
Code:
```py
print = 2
```<end_code>
"""
    else:  # We're at step 2
        return """
Thought: I can now answer the initial question
Code:
```py
final_answer("got an error")
```<end_code>
"""


def fake_react_code_functiondef(messages, stop_sequences=None) -> str:
    prompt = str(messages)
    if "special_marker" not in prompt:
        return """
Thought: Let's define the function. special_marker
Code:
```py
import numpy as np

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w
```<end_code>
"""
    else:  # We're at step 2
        return """
Thought: I can now answer the initial question
Code:
```py
x, w = [0, 1, 2, 3, 4, 5], 2
res = moving_average(x, w)
final_answer(res)
```<end_code>
"""


def fake_code_llm_oneshot(messages, stop_sequences=None, grammar=None) -> str:
    return """
Thought: I should multiply 2 by 3.6452. special_marker
Code:
```py
result = python_interpreter(code="2*3.6452")
final_answer(result)
```
"""


def fake_code_llm_no_return(messages, stop_sequences=None, grammar=None) -> str:
    return """
Thought: I should multiply 2 by 3.6452. special_marker
Code:
```py
result = python_interpreter(code="2*3.6452")
print(result)
```
"""


class AgentTests(unittest.TestCase):
    def test_fake_code_agent(self):
        agent = CodeAgent(tools=[PythonInterpreterTool()], llm_engine=fake_code_llm_oneshot)
        output = agent.run("What is 2 multiplied by 3.6452?")
        assert isinstance(output, str)
        assert output == "7.2904"

    def test_fake_react_json_agent(self):
        agent = ReactJsonAgent(tools=[PythonInterpreterTool()], llm_engine=fake_react_json_llm)
        output = agent.run("What is 2 multiplied by 3.6452?")
        assert isinstance(output, str)
        assert output == "7.2904"
        assert agent.logs[0]["task"] == "What is 2 multiplied by 3.6452?"
        assert agent.logs[1]["observation"] == "7.2904"
        assert agent.logs[1]["rationale"].strip() == "Thought: I should multiply 2 by 3.6452. special_marker"
        assert (
            agent.logs[2]["llm_output"]
            == """
Thought: I can now answer the initial question
Action:
{
    "action": "final_answer",
    "action_input": {"answer": "7.2904"}
}
"""
        )

    def test_fake_react_code_agent(self):
        agent = ReactCodeAgent(tools=[PythonInterpreterTool()], llm_engine=fake_react_code_llm)
        output = agent.run("What is 2 multiplied by 3.6452?")
        assert isinstance(output, float)
        assert output == 7.2904
        assert agent.logs[0]["task"] == "What is 2 multiplied by 3.6452?"
        assert agent.logs[2]["tool_call"] == {
            "tool_arguments": "final_answer(7.2904)",
            "tool_name": "code interpreter",
        }

    def test_react_code_agent_code_errors_show_offending_lines(self):
        agent = ReactCodeAgent(tools=[PythonInterpreterTool()], llm_engine=fake_react_code_llm_error)
        output = agent.run("What is 2 multiplied by 3.6452?")
        assert isinstance(output, AgentText)
        assert output == "got an error"
        assert "Evaluation stopped at line 'print = 2' because of" in str(agent.logs)

    def test_setup_agent_with_empty_toolbox(self):
        ReactJsonAgent(llm_engine=fake_react_json_llm, tools=[])

    def test_react_fails_max_iterations(self):
        agent = ReactCodeAgent(
            tools=[PythonInterpreterTool()],
            llm_engine=fake_code_llm_no_return,  # use this callable because it never ends
            max_iterations=5,
        )
        agent.run("What is 2 multiplied by 3.6452?")
        assert len(agent.logs) == 7
        assert type(agent.logs[-1]["error"]) is AgentMaxIterationsError

    @require_torch
    def test_init_agent_with_different_toolsets(self):
        toolset_1 = []
        agent = ReactCodeAgent(tools=toolset_1, llm_engine=fake_react_code_llm)
        assert (
            len(agent.toolbox.tools) == 1
        )  # when no tools are provided, only the final_answer tool is added by default

        toolset_2 = [PythonInterpreterTool(), PythonInterpreterTool()]
        agent = ReactCodeAgent(tools=toolset_2, llm_engine=fake_react_code_llm)
        assert (
            len(agent.toolbox.tools) == 2
        )  # deduplication of tools, so only one python_interpreter tool is added in addition to final_answer

        toolset_3 = Toolbox(toolset_2)
        agent = ReactCodeAgent(tools=toolset_3, llm_engine=fake_react_code_llm)
        assert (
            len(agent.toolbox.tools) == 2
        )  # same as previous one, where toolset_3 is an instantiation of previous one

        # check that add_base_tools will not interfere with existing tools
        with pytest.raises(KeyError) as e:
            agent = ReactJsonAgent(tools=toolset_3, llm_engine=fake_react_json_llm, add_base_tools=True)
        assert "already exists in the toolbox" in str(e)

        # check that python_interpreter base tool does not get added to code agents
        agent = ReactCodeAgent(tools=[], llm_engine=fake_react_code_llm, add_base_tools=True)
        assert len(agent.toolbox.tools) == 7  # added final_answer tool + 6 base tools (excluding interpreter)

    def test_function_persistence_across_steps(self):
        agent = ReactCodeAgent(
            tools=[], llm_engine=fake_react_code_functiondef, max_iterations=2, additional_authorized_imports=["numpy"]
        )
        res = agent.run("ok")
        assert res[0] == 0.5

    def test_init_managed_agent(self):
        agent = ReactCodeAgent(tools=[], llm_engine=fake_react_code_functiondef)
        managed_agent = ManagedAgent(agent, name="managed_agent", description="Empty")
        assert managed_agent.name == "managed_agent"
        assert managed_agent.description == "Empty"

    def test_agent_description_gets_correctly_inserted_in_system_prompt(self):
        agent = ReactCodeAgent(tools=[], llm_engine=fake_react_code_functiondef)
        managed_agent = ManagedAgent(agent, name="managed_agent", description="Empty")
        manager_agent = ReactCodeAgent(
            tools=[], llm_engine=fake_react_code_functiondef, managed_agents=[managed_agent]
        )
        assert "You can also give requests to team members." not in agent.system_prompt
        assert "<<managed_agents_descriptions>>" not in agent.system_prompt
        assert "You can also give requests to team members." in manager_agent.system_prompt
