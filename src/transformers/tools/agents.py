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
import json
from enum import Enum
from ast import literal_eval
from dataclasses import dataclass
from typing import Dict, Union, List, Optional
from huggingface_hub import hf_hub_download, list_spaces
from math import sqrt

from ..utils import is_offline_mode, logging
from .base import TASK_MAPPING, TOOL_CONFIG_FILE, Tool, get_tool_description_with_args, load_tool, supports_remote, OPENAI_TOOL_DESCRIPTION_TEMPLATE,DEFAULT_TOOL_DESCRIPTION_TEMPLATE
from .prompts import DEFAULT_REACT_SYSTEM_PROMPT, DEFAULT_CODE_SYSTEM_PROMPT
from .python_interpreter import evaluate_python_code
from PIL import Image

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
    "text-to-image",
]


class FinalAnswerTool(Tool):
    name = "final_answer"
    description = "Provides a final answer to the given problem"
    inputs = {"answer": {"type": str, "description": "The final answer to the problem"}}
    output_type = str

    def __call__(self):
        pass

class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL_CALL = "tool-call"
    TOOL_RESPONSE = "tool-response"

    @classmethod
    def roles(cls):
        return [r.value for r in cls]
    

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


def parse_json_blob(json_blob: str):
    try:
        first_accolade_index =  json_blob.find("{")
        last_accolade_index = [a.start() for a in list(re.finditer('}', json_blob))][-1]
        json_blob = json_blob[first_accolade_index:last_accolade_index+1].replace("\\", "")
        return json.loads(json_blob)
    except Exception as e:
        raise ValueError(f"The JSON blob you used is invalid: due to the following error: {e}. Make sure to correct its formatting.")


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


class AgentError(Exception):
    """Base class for other agent-related exceptions"""
    def __init__(self, message):
        super().__init__(message)
        self.message = message


class AgentParsingError(AgentError):
    """Exception raised for errors in parsing in the agent"""
    pass


class AgentExecutionError(AgentError):
    """Exception raised for errors in execution in the agent"""
    pass


class AgentMaxIterationsError(AgentError):
    """Exception raised for errors in execution in the agent"""
    pass


def get_clean_message_list(message_list: List[Dict[str, str]], role_conversions: Dict[str, str] = {}):
    """
    Subsequent messages with the same role will be concatenated to a single message.

    Args:
        message_list (`List[Dict[str, str]]`): List of chat messages.
    """
    final_message_list = []
    for message in message_list:
        if not set(message.keys()) == {"role", "content"}:
            raise ValueError("Message should contain only 'role' and 'content' keys!")
        
        role = message["role"]
        if role not in MessageRole.roles():
            raise ValueError(f"Incorrect role {role}, only {MessageRole.roles()} are supported for now.")

        if role in role_conversions:
            message["role"] = role_conversions[role]

        if len(final_message_list) > 0 and message["role"] == final_message_list[-1]["role"]:
            final_message_list[-1]["content"] += "\n" + message["content"]
        else:
            final_message_list.append(message)
    return final_message_list


llama_role_conversions = {
    MessageRole.SYSTEM: MessageRole.USER,
    MessageRole.TOOL_RESPONSE: MessageRole.USER,
}

class LLMEngine:
    def __init__(self, client):
        self.client = client

    def call(self, messages: List[Dict[str, str]], stop=["Output:"]) -> str:
        # Get clean message list
        messages = get_clean_message_list(messages, role_conversions=llama_role_conversions)

        # Get answer
        response = self.client.chat_completion(messages, stop=stop, max_tokens=1500)
        response = response.choices[0].message.content

        # Remove stop sequences from the answer
        for stop_seq in stop:
            if response[-len(stop_seq) :] == stop_seq:
                response = response[: -len(stop_seq)]
        return response


class Agent:
    def __init__(
            self, 
            llm_engine,  
            tools: List[Tool],
            system_prompt=DEFAULT_REACT_SYSTEM_PROMPT,
            tool_description_template=None,
            additional_args={},
            max_iterations=1,
            tool_parser=parse_json_tool_call,
            add_base_tools: bool = False,
            verbose=False,
        ):
        
        self.agent_name = self.__class__.__name__
        self.llm_engine = llm_engine
        self.system_prompt_template = system_prompt
        self.tool_description_template = tool_description_template if tool_description_template else OPENAI_TOOL_DESCRIPTION_TEMPLATE
        self.additional_args = additional_args
        self.max_iterations = max_iterations
        self.log = logger
        self.tool_parser = tool_parser

        self._toolbox = Toolbox(tools, add_base_tools=add_base_tools)

        self.system_prompt = format_prompt(self._toolbox, self.system_prompt_template, self.tool_description_template)
        self.messages = []
        self.prompt = None
        self.logs = []

        if verbose:
            logging.set_verbosity_debug()


    @property
    def toolbox(self) -> Dict[str, Tool]:
        """Get the toolbox currently available to the agent"""
        return self._toolbox


    def get_inner_memory_from_logs(self) -> str:
        """
        Reads past llm_outputs, actions, and observations or errors from the logs.
        """
        prompt_message = {
            "role": MessageRole.SYSTEM,
            "content": self.logs[0]["system_prompt"]
        }
        task_message ={
            "role": MessageRole.USER,
            "content": "Task: " + self.logs[0]["task"],
        }
        memory  = [prompt_message, task_message]
        
        for step_log in self.logs[1:]:
            thought_message = {
                "role": MessageRole.ASSISTANT,
                "content": "Thought: " + step_log["llm_output"] + "\n"
            }
            memory.append(thought_message)

            if 'error' in step_log:
                message_content = "Error: " + str(step_log["error"]) + "\nNow let's retry: take care not to repeat previous errors! Try to adopt different approaches if you can.\n"
            else:
                message_content = f"Observation: {step_log['observation']}"
            tool_response_message = {
                "role": MessageRole.TOOL_RESPONSE,
                "content": message_content
            }
            memory.append(tool_response_message)
        return memory


    def show_message_history(self):
        self.log.info('\n'.join(self.messages))



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
            rationale, action = split[-2], split[-1] # NOTE: using indexes starting from the end solves for when you have more than one split_token in the output
        except Exception as e:
            self.log.error(e, exc_info=1)
            raise AgentParsingError(f"Error: No '{split_token}' token provided in your output.\nYour output:\n{llm_output}\n. Be sure to include an action, prefaced with '{split_token}'!")
        return rationale, action
     
    def execute(self, tool_name: str, arguments: Dict[str, str]) -> None:
        """
        Execute tool with the provided input and returns the result.

        Args:
            tool_name (`str`): Name of the Tool to execute (shoulde be one from
                self.toolbox).
            split_token (Any): Arguments passed to the Tool. 
        """

        if tool_name not in self.toolbox.tools:
            error_msg = f"Error: unknown tool {tool_name}, should be instead one of {[tool_name for tool_name in self.toolbox.tools.keys()]}."
            self.log.error(error_msg, exc_info=1)
            raise AgentExecutionError(error_msg)
        
        try:
            if isinstance(arguments, str):
                observation = self.toolbox.tools[tool_name](arguments)
            else:
                for key, value in arguments.items():
                    if value in self.state:
                        arguments[key] = self.state[value]
                observation = self.toolbox.tools[tool_name](**arguments)
            return observation

        except Exception as e:
            raise AgentExecutionError(
                f"Error in tool call execution: {e}.\nYour input was probably incorrect.\n"
                f"As a reminder, this tool's description is the following:\n{get_tool_description_with_args(self.toolbox.tools[tool_name])}"
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
            llm_engine, 
            system_prompt=DEFAULT_CODE_SYSTEM_PROMPT, 
            tool_description_template=None,
            **kwargs
        ):
        
        super().__init__(
            llm_engine, 
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
        self.system_prompt = format_prompt(self._toolbox, self.system_prompt_template, self.tool_description_template)

        self.prompt = self.system_prompt + f"\nTask: {task}"
        self.log.info("====Executing with this prompt====")
        self.log.info(self.prompt)
        llm_output = self.llm_engine(self.prompt, stop=["Task:"])

        task_message = f"Task: {task}"

        self.logs.append({"task": task_message, "system_prompt": self.system_prompt})
        
        memory = self.get_inner_memory_from_logs()

        self.log.info("====Executing with these messages====")
        self.log.info(memory)

        # Run LLM
        llm_output = self.llm_engine(memory, stop=["Task:"])

        if return_generated_code:
            return llm_output["content"]

        # Parse
        _, code_action = self.extract_action(
            llm_output=llm_output["content"],
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
            llm_engine, 
            system_prompt=DEFAULT_REACT_SYSTEM_PROMPT, 
            tool_description_template=None,
            max_iterations=5,
            llm_engine_grammar=None,
            **kwargs
        ):
        
        super().__init__(
            llm_engine, 
            system_prompt=system_prompt,
            tool_description_template=tool_description_template if tool_description_template else self.default_tool_description_template,
            max_iterations=max_iterations,
            **kwargs
        )
        self.llm_engine_grammar = llm_engine_grammar
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
    

    def run(self, task, **kwargs):
        """
        Sends a request to the agent.

        Args:
            task (`str`): The task to perform

        Example:

        ```py
        from transformers import ReactAgent
        from transformers.tools.base import CalculatorTool

        calculator = CalculatorTool()
        agent = ReactAgent(tools=[CalculatorTool()])
        agent.run("What is the result of 2 power 3.7384?")
        ```
        """
        
        self.logs = []
        self.system_prompt = format_prompt(self._toolbox, self.system_prompt_template, self.tool_description_template)

        self.state=kwargs.copy()
        if '<<additional_args>>' in self.system_prompt and len(self.state) > 0:
            self.system_prompt = self.system_prompt.replace(
                '<<additional_args>>',
                f"You have been provided with these initial arguments, that you should absolutely use if needed rather than hallucinating arguments: {str(self.state)}."
            )

        self.log.info("=====New task=====")
        self.log.debug("System prompt is as follows:")
        self.log.debug(self.system_prompt)
        self.logs.append({"system_prompt": self.system_prompt, "task": task})

        final_answer = None
        iteration = 0

        while not final_answer and iteration < self.max_iterations:
            try:
                final_answer = self.step()
            except AgentError as e:
                self.log.error(e)
                self.logs[-1]["error"] = e
            finally:
                iteration += 1
        
        if not final_answer and iteration == self.max_iterations:
            error_message = "Failed by reaching max iterations."
            self.log.error(error_message)
            final_answer = error_message
            self.logs.append({"error": AgentMaxIterationsError(error_message)})

        return final_answer
    

    def step(self):
        """
        Runs agent step with the current prompt (task + state).
        """
        agent_memory = self.get_inner_memory_from_logs()
        self.logs[-1]["agent_memory"] = agent_memory.copy()

        self.prompt = agent_memory
        # self.prompt = agent_memory + "\nThought: " # prepend the answer to steer the llm
        self.log.debug("=====New step=====")

        # Add new step in logs
        self.logs.append({})

        self.log.info("=====Calling LLM with these messages:=====")
        self.log.info(agent_memory)

        if self.llm_engine_grammar:
            llm_output = self.llm_engine(self.prompt, stop=["Observation:"], grammar=self.llm_engine_grammar)
        else:
            llm_output = self.llm_engine(self.prompt, stop=["Observation:"])
        self.log.debug("=====Output message of the LLM:=====")
        self.log.debug(llm_output)
        self.logs[-1]["llm_output"] = llm_output

        # Parse
        self.log.debug("=====Extracting action=====")
        rationale, action = self.extract_action(
            llm_output=llm_output,
            split_token="Action:"
        )

        try:
            tool_name, arguments = self.tool_parser(action)
        except Exception as e:
            raise AgentParsingError(f"Could not parse the given action: {e}.")
        
        self.logs[-1]["rationale"] = rationale
        self.logs[-1]["tool_call"] = {
            "tool_name": tool_name,
            "tool_arguments": arguments
        }
    
        # Execute
        if tool_name == "final_answer":
            if isinstance(arguments, dict):
                answer = arguments['answer']
            else:
                answer = arguments
            if answer in self.state: # if the answer is a state variable, return the value
                answer = self.state[answer]
            return answer
        else:
            observation = self.execute(tool_name, arguments)
            
            observation_type = type(observation)
            if observation_type in [str, int, float, bool]:
                updated_information = str(observation).strip()
            else: # if the execution result is an object, store it
                if observation_type == Image.Image:
                    observation_name = "image.png"
                else:
                    observation_name = "object.object"
                # TODO: improve observation name choice

                self.state[observation_name] = observation
                updated_information = f"Stored '{observation_name}' in memory."
            
            self.log.info(updated_information)
            self.logs[-1]["observation"] = updated_information
            return None