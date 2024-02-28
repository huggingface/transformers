#!/usr/bin/env python
# coding=utf-8

# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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
import importlib.util
import json
import os
import time
from dataclasses import dataclass
from typing import Dict, List

import requests
from huggingface_hub import HfFolder, hf_hub_download, list_spaces

from ..models.auto import AutoTokenizer
from ..utils import is_offline_mode, is_openai_available, is_torch_available, logging
from .base import TASK_MAPPING, TOOL_CONFIG_FILE, Tool, load_tool, supports_remote
from .prompts import CHAT_MESSAGE_PROMPT, download_prompt
from .python_interpreter import evaluate


logger = logging.get_logger(__name__)


if is_openai_available():
    import openai

if is_torch_available():
    from ..generation import StoppingCriteria, StoppingCriteriaList
    from ..models.auto import AutoModelForCausalLM
else:
    StoppingCriteria = object

_tools_are_initialized = False


BASE_PYTHON_TOOLS = {
    "print": print,
    "range": range,
    "float": float,
    "int": int,
    "bool": bool,
    "str": str,
}


@dataclass
class PreTool:
    task: str
    description: str
    repo_id: str


HUGGINGFACE_DEFAULT_TOOLS = {}


HUGGINGFACE_DEFAULT_TOOLS_FROM_HUB = [
    "image-transformation",
    "text-download",
    "text-to-image",
    "text-to-video",
]


def get_remote_tools(organization="huggingface-tools"):
    if is_offline_mode():
        logger.info("You are in offline mode, so remote tools are not available.")
        return {}

    spaces = list_spaces(author=organization)
    tools = {}
    for space_info in spaces:
        repo_id = space_info.id
        resolved_config_file = hf_hub_download(repo_id, TOOL_CONFIG_FILE, repo_type="space")
        with open(resolved_config_file, encoding="utf-8") as reader:
            config = json.load(reader)

        task = repo_id.split("/")[-1]
        tools[config["name"]] = PreTool(task=task, description=config["description"], repo_id=repo_id)

    return tools


def _setup_default_tools():
    global HUGGINGFACE_DEFAULT_TOOLS
    global _tools_are_initialized

    if _tools_are_initialized:
        return

    main_module = importlib.import_module("transformers")
    tools_module = main_module.tools

    remote_tools = get_remote_tools()
    for task_name, tool_class_name in TASK_MAPPING.items():
        tool_class = getattr(tools_module, tool_class_name)
        description = tool_class.description
        HUGGINGFACE_DEFAULT_TOOLS[tool_class.name] = PreTool(task=task_name, description=description, repo_id=None)

    if not is_offline_mode():
        for task_name in HUGGINGFACE_DEFAULT_TOOLS_FROM_HUB:
            found = False
            for tool_name, tool in remote_tools.items():
                if tool.task == task_name:
                    HUGGINGFACE_DEFAULT_TOOLS[tool_name] = tool
                    found = True
                    break

            if not found:
                raise ValueError(f"{task_name} is not implemented on the Hub.")

    _tools_are_initialized = True


def parse_json_tool_call(json_blob: str):
    try:
        json_blob = json.loads(json_blob.strip())
    except:
        raise ValueError(f"Invalid JSON blob: {json_blob}")
    if "action" in json_blob and "action_input" in json_blob:
        return json_blob["action"], json_blob["action_input"]
    else:
        raise ValueError(f"Missing keys: {[key for key in ['action', 'action_input'] if key not in json_blob]} in blob {json_blob}")


def get_tool_creation_code(code, toolbox, remote=False):
    code_lines = ["from transformers import load_tool", ""]
    for name, tool in toolbox.items():
        if name not in code or isinstance(tool, Tool):
            continue

        task_or_repo_id = tool.task if tool.repo_id is None else tool.repo_id
        line = f'{name} = load_tool("{task_or_repo_id}"'
        if remote:
            line += ", remote=True"
        line += ")"
        code_lines.append(line)

    return "\n".join(code_lines) + "\n"

class FinalAnswerTool(Tool):
    name = "final_answer"
    description = "Provides a final answer to the given problem"
    inputs = {"answer": str}
    outputs = {"answer": str}

    def __call__(self):
        pass


DEFAULT_REACT_SYSTEM_PROMPT = """Solve the following task as best you can. You have access to the following tools:

{tool_descriptions}

The way you use the tools is by specifying a json blob.
Specifically, this json should have a `action` key (name of the tool to use) and a `action_input` key (input to the tool).

The only values that should be in the "action" field are: {tool_names}

The $ACTION_JSON_BLOB should only contain a SINGLE action and MUST be formatted as markdown, do NOT return a list of multiple actions. Here is an example of a valid $ACTION_JSON_BLOB:

{{
  "action": $TOOL_NAME,
  "action_input": $INPUT
}}


Make sure to have the $INPUT in the right format for the tool you are using, and do not put variable names as input if you can find the right values.

You will be given:

Task: the task you are given.

You should ALWAYS use the following format:

Thought: you should always think about one action to take. Then use the action as follows:
Action:
$ACTION_JSON_BLOB
Observation: the result of the action
... (this Thought/Action/Observation can repeat N times, you should take several steps when needed. The $ACTION_JSON_BLOB must only use a SINGLE action at a time.)

You must always end your output with the following format:

Thought: I now have solved the task.
Action: 
{{
    "action": "final_answer",
    "action_input": {{
        "answer": $ANSWER
    }}
}}
ALWAYS use the final_answer tool to provide the final answer to the task. It is the only way to complete the task, else you will be stuck on a loop.

Now begin!
"""

class Agent:
    """
    Base class for all agents which contains the main API methods.

    Args:
        chat_prompt_template (`str`, *optional*):
            Pass along your own prompt if you want to override the default template for the `chat` method. Can be the
            actual prompt template or a repo ID (on the Hugging Face Hub). The prompt should be in a file named
            `chat_prompt_template.txt` in this repo in this case.
        run_prompt_template (`str`, *optional*):
            Pass along your own prompt if you want to override the default template for the `run` method. Can be the
            actual prompt template or a repo ID (on the Hugging Face Hub). The prompt should be in a file named
            `run_prompt_template.txt` in this repo in this case.
        toolbox ([`Tool`], dictionary with tool values, *optional*:
    """

    def __init__(
            self,
            llm_engine,
            chat_prompt_template=None,
            run_prompt_template=None,
            toolbox=None,
            system_prompt=None
        ):
        agent_name = self.__class__.__name__

        self.llm_engine = llm_engine
        self.chat_prompt_template = download_prompt(chat_prompt_template, agent_name, mode="chat")
        self.run_prompt_template = download_prompt(run_prompt_template, agent_name, mode="run")

        if toolbox is None:
            self._toolbox = _setup_default_tools()
        else:
            self._toolbox = {tool.name: tool for tool in toolbox}

        final_answer_tool = FinalAnswerTool()
        self._toolbox["final_answer"] = final_answer_tool

        self.log = print

        # Init system prompt
        if system_prompt:
            self.system_prompt = system_prompt
        else:
            self.system_prompt = DEFAULT_REACT_SYSTEM_PROMPT
        tool_descriptions = "\n".join([f"- {tool_name}: {tool.description}" for tool_name, tool in self._toolbox.items()])
        self.system_prompt = self.system_prompt.format(
            tool_descriptions=tool_descriptions,
            tool_names=", ".join([tool_name for tool_name in self._toolbox.keys()])
        )
        # Create empty memory
        self.prepare_for_new_chat()

    @property
    def toolbox(self) -> List[Tool]:
        """Get all tool currently available to the agent"""
        return self._toolbox

    def chat(self, *, return_code=False, remote=False, **kwargs):
        """
        Sends a new request to the agent in a chat. Will use the previous ones in its history.
        """
        #TODO: fill this
        while True:
            self.run()


    def run(self, task, *, return_code=False, remote=False, **kwargs):
        """
        Sends a request to the agent.
        """
        self.task = task
        final_answer = None
        while not final_answer:
            final_answer = self.step()
        return final_answer


    def step(self):
        current_prompt = self.system_prompt + "\nTask: " + self.task
        current_prompt += "\n" + "\n".join(self.memory)
        print("=====Calling LLM with this prompt:=====")
        print(current_prompt)
        result = self.llm_engine(current_prompt, stop=["Observation:", "====="])
        self.memory.append(result.strip() + "\n")
        thought, tool_call = result.split("Action:")

        self.log(f"==Thought from the agent==\n{thought}")

        tool_name, arguments = parse_json_tool_call(tool_call)

        if tool_name == "final_answer":
            return arguments

        else:
            self.log("\n\n==Result==")
            observation = self.toolbox[tool_name](**arguments)
            self.memory.append("Observation: " + observation.strip() + "\n")
            return None
        

    def prepare_for_new_chat(self):
        """
        Clears the history of prior calls to [`~Agent.chat`].
        """
        self.memory = []
        self.cached_tools = None #TODO: check if this attribute is useful to keep
