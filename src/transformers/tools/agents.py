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
from dataclasses import dataclass
from typing import Dict

import requests
from huggingface_hub import hf_hub_download, list_spaces

from ..utils import is_offline_mode, is_openai_available, is_torch_available, logging
from .base import TASK_MAPPING, TOOL_CONFIG_FILE, Tool, get_tool_description_with_args, OPENAI_TOOL_DESCRIPTION_TEMPLATE
from .prompts import DEFAULT_REACT_SYSTEM_PROMPT, DEFAULT_CODE_SYSTEM_PROMPT
from .python_interpreter import evaluate as evaluate_python_code


logger = logging.get_logger(__name__)


if is_torch_available():
    from ..generation import StoppingCriteria, StoppingCriteriaList
    from ..models.auto import AutoModelForCausalLM
else:
    StoppingCriteria = object


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


def clean_code_for_chat(result):
    lines = result.split("\n")
    idx = 0
    while idx < len(lines) and not lines[idx].lstrip().startswith("```"):
        idx += 1
    explanation = "\n".join(lines[:idx]).strip()
    if idx == len(lines):
        return explanation, None

    idx += 1
    start_idx = idx
    while not lines[idx].lstrip().startswith("```"):
        idx += 1
    code = "\n".join(lines[start_idx:idx]).strip()

    return explanation, code


def clean_code_for_run(code):
    code = code.strip()
    code_lines = code.split("\n")
    if code_lines[0] in ["```", "```py", "```python"]:
        code_lines = code_lines[1:]
    if code_lines[-1] == "```":
        code_lines = code_lines[:-1]
    code = "\n".join(code_lines)

    return code


def parse_json_tool_call(json_blob: str):
    json_blob = json_blob.strip().replace("```json", "").replace("```", "").replace('\\', "")
    try:
        json_blob = json.loads(json_blob)
    except Exception as e:
        raise ValueError(f"The JSON blob you used is invalid: due to the following error: {e}. Try to correct its formatting.")
    if "action" in json_blob and "action_input" in json_blob:
        return json_blob["action"], json_blob["action_input"]
    else:
        raise ValueError(f"Missing keys: {[key for key in ['action', 'action_input'] if key not in json_blob]} in blob {json_blob}")


class FinalAnswerTool(Tool):
    name = "final_answer"
    description = "Provides a final answer to the given problem"
    inputs = {"answer": str}
    outputs = {"answer": str}

    def __call__(self):
        pass


def format_prompt(toolbox, prompt_template, task):
    tool_descriptions = "\n".join([get_tool_description_with_args(tool) for tool in toolbox.values()])
    prompt = prompt_template.replace("<<tool_descriptions>>", tool_descriptions)
    prompt = prompt.replace("<<task>>", task)
    if "<<tool_names>>" in prompt:
        tool_names = [f"'{tool_name}'" for tool_name in toolbox.keys()]
        prompt = prompt.replace("<<tool_names>>", ", ".join(tool_names))
    return prompt


class Agent:
    def __init__(self, llm_callable,  function_template=None, additional_args=Dict[str,any],toolbox=None):
        self.agent_name = self.__class__.__name__
        self.log = print
        self.llm_callable = llm_callable
        self.function_template=function_template
        if toolbox is None:
            _setup_default_tools()
            self._toolbox = HUGGINGFACE_DEFAULT_TOOLS
        else:
            self._toolbox = {tool.name: tool for tool in toolbox}
        # TODO: allow to specifiy a repo_id str instead of a Tool object in toolbox, and load the corresponding tool from the hub
        self.memory = []
        self.prompt_template = None

    @property
    def toolbox(self) -> Dict[str, Tool]:
        """Get all tools currently available to the agent"""
        return self._toolbox
    
    @property
    def default_function_template(self)-> str:
        """
        This template is taking can desbribe a tool as it is expected by the model
        """
        logger.warning_once(
            "\nNo function template is defined for this tokenizer - using a default function template "
            "that implements the ChatML format (without BOS/EOS tokens!). If the default is not appropriate for "
            "your model, please set `tokenizer.function_template` to an appropriate template. "
        )
        return OPENAI_TOOL_DESCRIPTION_TEMPLATE

    def format_prompt(self, task):
        """Override this for a custom prompt format"""
        return format_prompt(self.toolbox, self.prompt_template, task)


    def parser(self,input_message):
        pass
        # TODO: to continue
        

class CodeAgent(Agent):
    def __init__(self, llm_callable, toolbox=None, run_prompt_template=None, stop_sequences=None, **kwargs):
        super().__init__(llm_callable, toolbox=toolbox)
        self.stop_sequences = stop_sequences
        self.prompt_template = DEFAULT_CODE_SYSTEM_PROMPT


    def clean_code_for_run(self, result):
        """
        Override this method if you want to change the way the code is
        cleaned for the `run` method.
        """
        return clean_code_for_run(result)

    def run(self, task, retrying=False, **kwargs):
        """
        Sends a request to the agent.

        Args:
            task (`str`): The task to perform
            kwargs (additional keyword arguments, *optional*):
                Any keyword argument to send to the agent when evaluating the code.

        Example:

        ```py
        from transformers import CodeAgent

        agent = CodeAgent() # TODO: fill this
        agent.run("Draw me a picture of rivers and lakes")
        ```
        """
        # Run LLM
        prompt = self.format_prompt(task)
        if retrying: # Add memory of the first (failed) run
            prompt += "\n" + "\n".join(self.memory)

        result = self.llm_callable(prompt, stop=["Task:"])

        # Parse
        split_token = "Answer:"
        try:
            result = f"I will use the following {result}"
            explanation, code = result.split(split_token)
        except Exception as e:
            error_msg = f"Error: No '{split_token}' provided. Be sure to add that string to your generation"
            self.log(error_msg)
            if not retrying:
                self.memory.append(error_msg + ". Now let's retry.")
                return self.run(task, retrying=True, **kwargs)
            return None
        self.log(f"==Explanation from the agent==\n{explanation}")
        self.log(f"\n\n==Action generated by the agent==\n{code}")
        
        try: 
            code = self.clean_code_for_run(code)
        except Exception as e:
            error_msg = f"Error in code parsing: {e}. Be sure to provide correct code"
            self.log(error_msg)
            if not retrying:
                self.memory.append(error_msg + ". Now let's retry.")
                return self.run(task, retrying=True, **kwargs)
            return str(e)

        # Execute
        try: 
            self.log("\n\n==Execution==")
            available_tools = {**BASE_PYTHON_TOOLS.copy(), **self.toolbox} 
            # NOTE: The base python tools are not added to toolbox, since they do not have the proper attributes for a description
            return evaluate_python_code(code, available_tools, state=kwargs.copy())
        except Exception as e:
            error_msg = f"Error in execution: {e}. Be sure to provide correct code."
            self.log(error_msg)
            if not retrying:
                self.memory.append(error_msg + ". Now let's retry.")
                return self.run(task, retrying=True, **kwargs)
            return str(e)


class ReactAgent(Agent):
    def __init__(
            self,
            llm_callable,
            toolbox=None,
            system_prompt=None,
            max_iterations=5,
        ):
        super().__init__(llm_callable, toolbox=toolbox)
        # Add final answer to tools
        final_answer_tool = FinalAnswerTool()
        self._toolbox["final_answer"] = final_answer_tool

        # Init system prompt
        if system_prompt:
            self.prompt_template = system_prompt
        else:
            self.prompt_template = DEFAULT_REACT_SYSTEM_PROMPT

        tool_descriptions = "\n".join([get_tool_description_with_args(tool) for tool in self.toolbox.values()])

        self.max_iterations = max_iterations


    def run(self, task):
        """
        Have the agent accomplish a task.
        """
        self.memory = []
        self.task = task
        final_answer = None
        iteration = 0
        while not final_answer and iteration < self.max_iterations:
            final_answer = self.step()
            iteration+=1
        if not final_answer and iteration == self.max_iterations:
            self.log("Failed by reaching max iterations, returning None.")

        return final_answer


    def step(self):
        # Run LLM
        current_prompt = self.format_prompt(self.task)
        current_prompt += "\n" + "\n".join(self.memory)
        self.log("=====Calling LLM with this prompt:=====")
        self.log(current_prompt)

        result = self.llm_callable(current_prompt, stop=["Observation:", "====="]).strip()
        self.log(f"==Model output==\n{result}")
        self.memory.append(result + "\n")

        # Parse
        try:
            explanation, action = result.split("Action:")
        except Exception as e:
            self.memory.append("Error: you did not provide 'Action:': it is mandatory to provide an Action!")
            return None
        self.log(f"==Explanation from the agent==\n{explanation}")
        self.log(f"\n\n==Action generated by the agent==\n{action}")

        try:
            tool_name, arguments = parse_json_tool_call(action)
        except Exception as e:
            self.memory.append(f"Error in json parsing: {e}.")
            self.log("====Error!====")
            self.log(e)
            return None

        if tool_name not in self.toolbox:
            self.memory.append(f"Error: unknown tool {tool_name}, should be instead one of {[tool_name for tool_name in self.toolbox.keys()]}.")
            return None
        
        # Execute
        if tool_name == "final_answer":
            if isinstance(arguments, str):
                return arguments
            return arguments['answer']
        else:
            self.log("\n\n==Result==")
            print("HERE ARE THE AEGGS:", arguments)
            try:
                if isinstance(arguments, str):
                    observation = self.toolbox[tool_name](arguments)
                else:
                    observation = self.toolbox[tool_name](**arguments)
                self.memory.append("Observation: " + observation.strip() + "\n")
                return None
            except Exception as e:
                self.memory.append(
                    f"Error in tool call execution: {e}. Correct the arguments if they are incorrect."
                    f"As a reminder, this tool description is {get_tool_description_with_args(self.toolbox[tool_name])}."
                )
                self.log("====Error!====")
                self.log(e)
                return None
        