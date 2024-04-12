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
import re
from ast import literal_eval
from dataclasses import dataclass
from typing import Dict, Union, List, Optional
from huggingface_hub import hf_hub_download, list_spaces
from math import sqrt

from ..utils import is_offline_mode, logging
from .base import TASK_MAPPING, TOOL_CONFIG_FILE, Tool, get_tool_description_with_args, load_tool, supports_remote, OPENAI_TOOL_DESCRIPTION_TEMPLATE,DEFAULT_TOOL_DESCRIPTION_TEMPLATE
from .prompts import DEFAULT_REACT_SYSTEM_PROMPT, DEFAULT_CODE_SYSTEM_PROMPT
from .python_interpreter import evaluate_python_code

logging.set_verbosity_info()
logger = logging.get_logger(__name__)

_tools_are_initialized = False

def custom_print(*args):
    return ' '.join(map(str, args))


BASE_PYTHON_TOOLS = {
    "print": custom_print,
    "range": range,
    "float": float,
    "int": int,
    "bool": bool,
    "str": str,
    "round": round,
    "sqrt": sqrt,
}


@dataclass
class PreTool:
    name: str
    inputs: Dict[str, type]
    output_type: type
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


class FinalAnswerTool(Tool):
    name = "final_answer"
    description = "Provides a final answer to the given problem"
    inputs = {"answer": {"type": str, "description": "The final answer to the problem"}}
    output_type = str

    def __call__(self):
        pass


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
            config = literal_eval(reader.read())
        task = repo_id.split("/")[-1]
        tools[config["name"]] = PreTool(
            task=task,
            description=config["description"],
            repo_id=repo_id,
            name=task,
            inputs=config["inputs"],
            output_type=config["output_type"],
        )

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
        HUGGINGFACE_DEFAULT_TOOLS[tool_class.name] = PreTool(
            name=tool_class.name,
            inputs=tool_class.inputs,
            output_type=tool_class.output_type,
            task=task_name,
            description=tool_class.description,
            repo_id=None
        )
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


def clean_code_for_run(code):
    code = code.strip()
    code_lines = code.split("\n")
    if code_lines[0] in ["```", "```py", "```python"]:
        code_lines = code_lines[1:]
    if code_lines[-1] == "```":
        code_lines = code_lines[:-1]
    code = "\n".join(code_lines)
    return code
import json
def parse_json_blob(json_blob: str):
    try:
        first_accolade_index =  json_blob.find("{")
        last_accolade_index = [a.start() for a in list(re.finditer('}', json_blob))][-1]
        json_blob = json_blob[first_accolade_index:last_accolade_index+1]
        return literal_eval(json_blob)
    except Exception as e:
        raise ValueError(f"The JSON blob you used is invalid: due to the following error: {e}. Try to correct its formatting.")


def parse_json_tool_call(json_blob: str):
    json_blob = json_blob.replace("```json", "").replace("```", "")
    tool_call = parse_json_blob(json_blob)
    if "action" in tool_call and "action_input" in tool_call:
        return tool_call["action"], tool_call["action_input"]
    else:
        raise ValueError(f"Missing keys: {[key for key in ['action', 'action_input'] if key not in tool_call]} in blob {tool_call}")


def parse_text_tool_call(text: str):
    """
    Expects a text in the format: 'Action:', 'Action input:', 'Observation:'. 'Action input:' contains a json string with input arguments.
    """
    try:
        if "Observation:" in text:
            text = text.split("Observation:")[0]
        if "Action:" in text:
            text = text.split("Action:")[1]
        tool_name, tool_input = text.split("Action input:")
        if '{' in tool_input:
            tool_input = parse_json_blob(tool_input)
        else:
            tool_input = tool_input.strip().replace('"','')
        return tool_name.strip().replace('"','').replace('\\', ''), tool_input
    except Exception as e:
        raise ValueError(f"Error in parsing the text tool call: {e}. Be sure to provide the correct format.")


def format_prompt(toolbox, prompt_template,tool_description_template):
    tool_descriptions = toolbox.show_tool_descriptions(tool_description_template)
    prompt = prompt_template.replace("<<tool_descriptions>>", tool_descriptions)
    if "<<tool_names>>" in prompt:
        tool_names = [f"'{tool_name}'" for tool_name in toolbox.tools.keys()]
        prompt = prompt.replace("<<tool_names>>", ", ".join(tool_names))
    return prompt


def to_text(input: Union[List[Dict[str, str]], Dict[str, str], str]) -> str:
    if isinstance(input, list):
        return "\n".join(map(lambda m: m["content"], input))
    elif isinstance(input, dict):
        return input["content"]
    else:
        return input


class Toolbox():
    def __init__(self, tools: List[Tool], add_base_tools: bool = False):
        self._tools = {tool.name: tool for tool in tools}
        if add_base_tools:
            _setup_default_tools()
            self._tools = self._tools | HUGGINGFACE_DEFAULT_TOOLS.copy()
        self.load_tools_if_needed()

    @property
    def tools(self) -> Dict[str, Tool]:
        """Get all tools currently in the toolbox"""
        return self._tools

    def show_tool_descriptions(self, tool_description_template=None):
        """Returns the description of all tools in the toolbox"""
        return "\n".join([
            get_tool_description_with_args(tool, tool_description_template)
            for tool in self._tools.values()
        ])

    def add_tool(self, tool: Tool):
        """Adds a tool to the toolbox"""
        if tool.name in self._tools:
            raise KeyError(f"Error: tool {tool.name} already exists in the toolbox.")
        self._tools[tool.name] = tool

    def remove_tool(self, tool_name: str):
        """Removes a tool from the toolbox"""
        if tool_name not in self._tools:
            raise KeyError(
                f"Error: tool {tool_name} not found in toolbox for removal,"
                f"should be instead one of {[tool_name for tool_name in self._tools.keys()]}."
            )
        del self._tools[tool_name]

    def update_tool(self, tool: Tool):
        """Updates a tool in the toolbox"""
        if tool.name not in self._tools:
            raise KeyError(f"Error: tool {tool.name} not found in toolbox for update, should be instead one of {[tool_name for tool_name in self._tools.keys()]}.")
        self._tools[tool.name] = tool

    def clear_toolbox(self):
        """Clears the toolbox"""
        self._tools = {}

    def load_tools_if_needed(self, remote=False):
        for name, tool in self._tools.items():
            if not isinstance(tool, Tool):
                task_or_repo_id = tool.task if tool.repo_id is None else tool.repo_id
                _remote = remote and supports_remote(task_or_repo_id)
                self._tools[name] = load_tool(task_or_repo_id, remote=_remote)


class Agent:
    def __init__(
            self, 
            llm_callable,  
            tools: List[Tool],
            system_prompt=DEFAULT_REACT_SYSTEM_PROMPT,
            tool_description_template=None,
            additional_args={},
            max_iterations=1,
            tool_parser=parse_json_tool_call,
            add_base_tools: bool = False,
        ):
        
        self.agent_name = self.__class__.__name__
        self.llm_callable = llm_callable
        self.prompt_template = system_prompt
        self.tool_description_template = tool_description_template if tool_description_template else OPENAI_TOOL_DESCRIPTION_TEMPLATE
        self.additional_args = additional_args
        self.max_iterations = max_iterations
        self.log = logger
        self.tool_parser = tool_parser

        self._toolbox = Toolbox(tools, add_base_tools=add_base_tools)

        self.memory = []
        self.prompt = None



    @property
    def toolbox(self) -> Dict[str, Tool]:
        """Get the toolbox currently available to the agent"""
        return self._toolbox
    
    def show_memory(self):
        self.log.info('\n'.join(self.memory))
    
    def extract_action(self, llm_output: str, split_token: str) -> str:
        """
        Parse action from the LLM output

        Args:
            llm_output (`str`): Output of the LLM 
            split_token (`str`): Separator for the action. Should match the 
                example in the system prompt. 
        """
        try:
            split = llm_output.split(split_token)
            _, action = split[-2], split[-1] # NOTE: using indexes starting from the end solves for when you have more than one split_token in the output
        except Exception as e:
            self.log.error(e, exc_info=1)
            raise RuntimeError(f"Error: No '{split_token}' token provided in your output:///\n{llm_output}\n///. Be sure to include an action, prefaced with '{split_token}'!")
        return action
     
    def execute(self, tool_name: str, arguments: Dict[str, str]) -> None:
        """
        Execute tool with the provided input and append the result to the memory

        Args:
            tool_name (`str`): Name of the Tool to execute (shoulde be one from
                self.toolbox).
            split_token (Any): Arguments passed to the Tool. 
        """

        if tool_name not in self.toolbox.tools:
            error_msg = f"Error: unknown tool {tool_name}, should be instead one of {[tool_name for tool_name in self.toolbox.tools.keys()]}."
            self.log.error(error_msg, exc_info=1)
            raise KeyError(error_msg)
        
        self.log.info("\n\n==Result==")
        try:
            if isinstance(arguments, str):
                observation = self.toolbox.tools[tool_name](arguments)
            else:
                observation = self.toolbox.tools[tool_name](**arguments)
            observation_message = "Observation: " + str(observation).strip()
            self.log.info(observation_message)
            self.memory.append(observation_message)
        except Exception as e:
            raise RuntimeError(
                f"Error in tool call execution: {e}. Make sure to re-read the tool arguments again and adapt your input."
                f"As a reminder, this tool description is the following:\n{get_tool_description_with_args(self.toolbox.tools[tool_name])}"
            )
    
    def run(self, **kwargs):
        """To be implemented in the child class"""
        pass


class CodeAgent(Agent):
    """
    A class for an agent that solves the given task using a single block of code. This is a one-shot agent: it won't be able to act step-by-step.
    """
    def __init__(
            self, 
            llm_callable, 
            system_prompt=DEFAULT_CODE_SYSTEM_PROMPT, 
            tool_description_template=None,
            **kwargs
        ):
        
        super().__init__(
            llm_callable, 
            system_prompt=system_prompt,
            tool_description_template=tool_description_template if tool_description_template else self.default_tool_description_template,
            **kwargs
        )


    @property
    def default_tool_description_template(self)-> str:
        """
        This template is taking can desbribe a tool as it is expected by the model
        """
        logger.warning_once(
            "\nNo tool description template is defined for this tokenizer - using a default tool description template "
            "that implements the ChatML format (without BOS/EOS tokens!). If the default is not appropriate for "
            "your model, please set `tokenizer.tool_description_template` to an appropriate template. "
        )
        return DEFAULT_TOOL_DESCRIPTION_TEMPLATE
    
    def clean_code_for_run(self, result):
        """
        Override this method if you want to change the way the code is
        cleaned for the `run` method.
        """
        return clean_code_for_run(result)
    

    def run(self, task, return_generated_code=False, **kwargs):
        """
        Sends a request to the agent.

        Args:
            task (`str`): The task to perform
            kwargs (additional keyword arguments, *optional*):
                Any keyword argument to send to the agent when evaluating the code.

        Example:

        ```py
        from transformers import CodeAgent
        from transformers.tools.base import CalculatorTool

        calculator = CalculatorTool()
        agent = CodeAgent(toolbox=[CalculatorTool()])
        agent.run("What is the result of 2 power 3.7384?")
        ```
        """
        # Run LLM
        self.system_prompt = format_prompt(self._toolbox, self.prompt_template, self.tool_description_template)

        self.prompt = self.system_prompt + f"\nTask: {task}"
        llm_output = self.llm_callable(self.prompt, stop=["Task:"])
        self.log.info("====Executing with this prompt====")
        self.log.info(self.prompt)

        if return_generated_code:
            return llm_output

        # Parse
        code_action = self.extract_action(
            llm_output=llm_output,
            split_token="Answer:"
        )
        
        try: 
            code_action = self.clean_code_for_run(code_action)
        except Exception as e:
            error_msg = f"Error in code parsing: {e}. Be sure to provide correct code"
            self.log.error(error_msg)
            return error_msg

        # Execute
        try: 
            self.log.info("\n\n==Executing the code below:==")
            self.log.info(code_action)
            available_tools = {**BASE_PYTHON_TOOLS.copy(), **self.toolbox.tools} 
            # NOTE: The base python tools are not added to toolbox, since they do not have the proper attributes for a description
            return evaluate_python_code(code_action, available_tools, state=kwargs.copy())
        except Exception as e:
            error_msg = f"Error in execution: {e}. Be sure to provide correct code."
            self.log.error(error_msg, exc_info=1)
            return error_msg


class ReactAgent(Agent):
    """
    A class for an agent that solves the given task step by step, using the ReAct framework.
    While the objective is not reached, the agent will perform a cycle of thinking and acting.
    The action will be parsed from the LLM output, it will be the call of a tool from the toolbox, with arguments provided by the LLM.
    """
    def __init__(
            self, 
            llm_callable, 
            system_prompt=DEFAULT_REACT_SYSTEM_PROMPT, 
            tool_description_template=None,
            max_iterations=5,
            llm_callable_grammar=None,
            **kwargs
        ):
        
        super().__init__(
            llm_callable, 
            system_prompt=system_prompt,
            tool_description_template=tool_description_template if tool_description_template else self.default_tool_description_template,
            max_iterations=max_iterations,
            **kwargs
        )
        self.llm_callable_grammar = llm_callable_grammar
        self._toolbox.add_tool(FinalAnswerTool())

    @property
    def default_tool_description_template(self)-> str:
        """
        This template is taking can describe a tool as it is expected by the model
        """
        logger.warning_once(
            "\nNo tool description template is defined for this tokenizer - using a default tool description template "
            "that implements the ChatML format (without BOS/EOS tokens!). If the default is not appropriate for "
            "your model, please set `tokenizer.tool_description_template` to an appropriate template. "
        )
        return DEFAULT_TOOL_DESCRIPTION_TEMPLATE
    

    def run(self, task):
        """
        Sends a request to the agent.

        Args:
            task (`str`): The task to perform

        Example:

        ```py
        from transformers import ReactAgent
        from transformers.tools.base import CalculatorTool

        calculator = CalculatorTool()
        agent = ReactAgent(toolbox=[CalculatorTool()])
        agent.run("What is the result of 2 power 3.7384?")
        ```
        """
        self.system_prompt = format_prompt(self._toolbox, self.prompt_template, self.tool_description_template)

        self.memory = [self.system_prompt]

        self.task = task
        task_message = f"Task: {self.task}"
        self.memory.append(task_message)

        self.log.info("=====Initiating LLM with this prompt:=====")
        self.log.info(self.system_prompt)

        final_answer = None
        iteration = 0

        while final_answer is None and iteration < self.max_iterations:
            try:
                final_answer = self.step()
            except Exception as e:
                self.log.error(e)
                error_message = str(e) + "\nNow let's retry: take care not to repeat previous errors! Try to vary strategies as much as you can."
                self.memory.append(error_message)
            finally:
                iteration += 1
        
        if not final_answer and iteration == self.max_iterations:
            self.log.error("Failed by reaching max iterations.")
            final_answer = "Failed by reaching max iterations."

        return final_answer
    

    def step(self):
        """
        Runs agent step with the current prompt (task + state)
        """
        memory_as_text = '\n'.join(self.memory)
        self.prompt = memory_as_text + "\nThought: " # prepend the answer to steer the llm
        print("=====-=====")
        print("NEW STEP")
        print("=====Initiating LLM with this prompt:=====")
        print(self.prompt)

        if self.llm_callable_grammar:
            llm_output = self.llm_callable(self.prompt, stop=["Observation:"], grammar=self.llm_callable_grammar)
        else:
            llm_output = self.llm_callable(self.prompt, stop=["Observation:"])
        self.log.info("=====Output message of the LLM:=====")
        self.memory.append(llm_output)
        self.log.info(llm_output)


        # Parse
        self.log.info("=====Extracting action=====")
        action = self.extract_action(
            llm_output=llm_output,
            split_token="Action:"
        )

        self.log.info("=====Parsing action=====")
        try:
            tool_name, arguments = self.tool_parser(action)
        except Exception as e:
            raise RuntimeError(f"Could not parse the given action: {e}.")
    
        # Execute
        if tool_name == "final_answer":
            if isinstance(arguments, dict):
                return arguments['answer']
            else:
                return arguments
        else:
            self.execute(tool_name, arguments)
            return None