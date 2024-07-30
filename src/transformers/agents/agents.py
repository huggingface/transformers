#!/usr/bin/env python
# coding=utf-8

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
import json
import logging
import re
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

from .. import is_torch_available
from ..utils import logging as transformers_logging
from ..utils.import_utils import is_pygments_available
from .agent_types import AgentAudio, AgentImage, AgentText
from .default_tools import BASE_PYTHON_TOOLS, FinalAnswerTool, setup_default_tools
from .llm_engine import HfEngine, MessageRole
from .prompts import (
    DEFAULT_CODE_SYSTEM_PROMPT,
    DEFAULT_REACT_CODE_SYSTEM_PROMPT,
    DEFAULT_REACT_JSON_SYSTEM_PROMPT,
    PLAN_UPDATE_FINAL_PLAN_REDACTION,
    PROMPTS_FOR_INITIAL_PLAN,
    PROMPTS_FOR_PLAN_UPDATE,
    SUPPORTED_PLAN_TYPES,
    SYSTEM_PROMPT_FACTS,
    SYSTEM_PROMPT_FACTS_UPDATE,
    USER_PROMPT_FACTS_UPDATE,
)
from .python_interpreter import LIST_SAFE_MODULES, evaluate_python_code
from .tools import (
    DEFAULT_TOOL_DESCRIPTION_TEMPLATE,
    Tool,
    get_tool_description_with_args,
    load_tool,
)


if is_pygments_available():
    from pygments import highlight
    from pygments.formatters import Terminal256Formatter
    from pygments.lexers import PythonLexer


class CustomFormatter(logging.Formatter):
    grey = "\x1b[38;20m"
    bold_yellow = "\x1b[33;1m"
    red = "\x1b[31;20m"
    green = "\x1b[32;20m"
    bold_red = "\x1b[31;1m"
    bold_white = "\x1b[37;1m"
    reset = "\x1b[0m"
    format = "%(message)s"

    FORMATS = {
        logging.DEBUG: grey + format + reset,
        logging.INFO: format,
        logging.WARNING: bold_yellow + format + reset,
        31: reset + format + reset,
        32: green + format + reset,
        33: bold_white + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset,
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


logger = transformers_logging.get_logger(__name__)
logger.propagate = False
ch = logging.StreamHandler()
ch.setFormatter(CustomFormatter())
logger.addHandler(ch)


def parse_json_blob(json_blob: str) -> Dict[str, str]:
    try:
        first_accolade_index = json_blob.find("{")
        last_accolade_index = [a.start() for a in list(re.finditer("}", json_blob))][-1]
        json_blob = json_blob[first_accolade_index : last_accolade_index + 1].replace('\\"', "'")
        json_data = json.loads(json_blob, strict=False)
        return json_data
    except json.JSONDecodeError as e:
        place = e.pos
        if json_blob[place - 1 : place + 2] == "},\n":
            raise ValueError(
                "JSON is invalid: you probably tried to provide multiple tool calls in one action. PROVIDE ONLY ONE TOOL CALL."
            )
        raise ValueError(
            f"The JSON blob you used is invalid due to the following error: {e}.\n"
            f"JSON blob was: {json_blob}, decoding failed on that specific part of the blob:\n"
            f"'{json_blob[place-4:place+5]}'."
        )
    except Exception as e:
        raise ValueError(f"Error in parsing the JSON blob: {e}")


def parse_code_blob(code_blob: str) -> str:
    try:
        pattern = r"```(?:py|python)?\n(.*?)\n```"
        match = re.search(pattern, code_blob, re.DOTALL)
        return match.group(1).strip()
    except Exception as e:
        raise ValueError(
            f"""
The code blob you used is invalid: due to the following error: {e}
This means that the regex pattern {pattern} was not respected: make sure to include code with the correct pattern, for instance:
Thoughts: Your thoughts
Code:
```py
# Your python code here
```<end_action>"""
        )


def parse_json_tool_call(json_blob: str) -> Tuple[str, Dict[str, str]]:
    json_blob = json_blob.replace("```json", "").replace("```", "")
    tool_call = parse_json_blob(json_blob)
    if "action" in tool_call and "action_input" in tool_call:
        return tool_call["action"], tool_call["action_input"]
    elif "action" in tool_call:
        return tool_call["action"], None
    else:
        raise ValueError(
            f"Missing keys: {[key for key in ['action', 'action_input'] if key not in tool_call]} in blob {tool_call}"
        )


def parse_text_tool_call(text: str) -> Tuple[str, Union[str, Dict[str, str]]]:
    """
    Expects a text in the format: 'Action:', 'Action input:', 'Observation:'. 'Action input:' contains a json string with input arguments.
    """
    try:
        if "Observation:" in text:
            text = text.split("Observation:")[0]
        if "Action:" in text:
            text = text.split("Action:")[1]
        tool_name, tool_input = text.split("Action input:")
        if "{" in tool_input:
            tool_input = parse_json_blob(tool_input)
        else:
            tool_input = tool_input.strip().replace('"', "")
        return tool_name.strip().replace('"', "").replace("\\", ""), tool_input
    except Exception as e:
        raise ValueError(
            f"Error in parsing the text tool call: {e}. Be sure to provide the correct format. DO NOT repeat your previous incorrect tool call."
        )


def to_text(input: Union[List[Dict[str, str]], Dict[str, str], str]) -> str:
    if isinstance(input, list):
        return "\n".join([m["content"] for m in input])
    elif isinstance(input, dict):
        return input["content"]
    else:
        return input


HUGGINGFACE_DEFAULT_TOOLS = {}
_tools_are_initialized = False


class Toolbox:
    """
    The toolbox contains all tools that the agent can perform operations with, as well as a few methods to
    manage them.

    Args:
        tools (`List[Tool]`):
            The list of tools to instantiate the toolbox with
        add_base_tools (`bool`, defaults to `False`, *optional*, defaults to `False`):
            Whether to add the tools available within `transformers` to the toolbox.
    """

    def __init__(self, tools: List[Tool], add_base_tools: bool = False):
        self._tools = {tool.name: tool for tool in tools}
        if add_base_tools:
            self.add_base_tools()
        self._load_tools_if_needed()

    def add_base_tools(self, add_python_interpreter: bool = False):
        global _tools_are_initialized
        global HUGGINGFACE_DEFAULT_TOOLS
        if not _tools_are_initialized:
            HUGGINGFACE_DEFAULT_TOOLS = setup_default_tools(logger)
            _tools_are_initialized = True
        for tool in HUGGINGFACE_DEFAULT_TOOLS.values():
            if tool.name != "python_interpreter" or add_python_interpreter:
                self.add_tool(tool)
        self._load_tools_if_needed()

    @property
    def tools(self) -> Dict[str, Tool]:
        """Get all tools currently in the toolbox"""
        return self._tools

    def show_tool_descriptions(self, tool_description_template: str = None) -> str:
        """
        Returns the description of all tools in the toolbox

        Args:
            tool_description_template (`str`, *optional*):
                The template to use to describe the tools. If not provided, the default template will be used.
        """
        return "\n".join(
            [get_tool_description_with_args(tool, tool_description_template) for tool in self._tools.values()]
        )

    def add_tool(self, tool: Tool):
        """
        Adds a tool to the toolbox

        Args:
            tool (`Tool`):
                The tool to add to the toolbox.
        """
        if tool.name in self._tools:
            raise KeyError(f"Error: tool '{tool.name}' already exists in the toolbox.")
        self._tools[tool.name] = tool

    def remove_tool(self, tool_name: str):
        """
        Removes a tool from the toolbox

        Args:
            tool_name (`str`):
                The tool to remove from the toolbox.
        """
        if tool_name not in self._tools:
            raise KeyError(
                f"Error: tool {tool_name} not found in toolbox for removal, should be instead one of {list(self._tools.keys())}."
            )
        del self._tools[tool_name]

    def update_tool(self, tool: Tool):
        """
        Updates a tool in the toolbox according to its name.

        Args:
            tool (`Tool`):
                The tool to update to the toolbox.
        """
        if tool.name not in self._tools:
            raise KeyError(
                f"Error: tool {tool.name} not found in toolbox for update, should be instead one of {list(self._tools.keys())}."
            )
        self._tools[tool.name] = tool

    def clear_toolbox(self):
        """Clears the toolbox"""
        self._tools = {}

    def _load_tools_if_needed(self):
        for name, tool in self._tools.items():
            if not isinstance(tool, Tool):
                task_or_repo_id = tool.task if tool.repo_id is None else tool.repo_id
                self._tools[name] = load_tool(task_or_repo_id)

    def __repr__(self):
        toolbox_description = "Toolbox contents:\n"
        for tool in self._tools.values():
            toolbox_description += f"\t{tool.name}: {tool.description}\n"
        return toolbox_description


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


class AgentGenerationError(AgentError):
    """Exception raised for errors in generation in the agent"""

    pass


def format_prompt_with_tools(toolbox: Toolbox, prompt_template: str, tool_description_template: str) -> str:
    tool_descriptions = toolbox.show_tool_descriptions(tool_description_template)
    prompt = prompt_template.replace("<<tool_descriptions>>", tool_descriptions)
    if "<<tool_names>>" in prompt:
        tool_names = [f"'{tool_name}'" for tool_name in toolbox.tools.keys()]
        prompt = prompt.replace("<<tool_names>>", ", ".join(tool_names))
    return prompt


def format_prompt_with_imports(prompt_template: str, authorized_imports: List[str]) -> str:
    if "<<authorized_imports>>" not in prompt_template:
        raise AgentError("Tag '<<authorized_imports>>' should be provided in the prompt.")
    return prompt_template.replace("<<authorized_imports>>", str(authorized_imports))


class Agent:
    def __init__(
        self,
        tools: Union[List[Tool], Toolbox],
        llm_engine: Callable = HfEngine(),
        system_prompt=DEFAULT_REACT_JSON_SYSTEM_PROMPT,
        tool_description_template=None,
        additional_args={},
        max_iterations: int = 6,
        tool_parser=parse_json_tool_call,
        add_base_tools: bool = False,
        verbose: int = 0,
        memory_verbose: bool = False,
    ):
        self.agent_name = self.__class__.__name__
        self.llm_engine = llm_engine
        self.system_prompt_template = system_prompt
        self.tool_description_template = (
            tool_description_template if tool_description_template else DEFAULT_TOOL_DESCRIPTION_TEMPLATE
        )
        self.additional_args = additional_args
        self.max_iterations = max_iterations
        self.logger = logger
        self.tool_parser = tool_parser

        if isinstance(tools, Toolbox):
            self._toolbox = tools
            if add_base_tools:
                if not is_torch_available():
                    raise ImportError("Using the base tools requires torch to be installed.")

                self._toolbox.add_base_tools(add_python_interpreter=(self.__class__ == ReactJsonAgent))
        else:
            self._toolbox = Toolbox(tools, add_base_tools=add_base_tools)
        self._toolbox.add_tool(FinalAnswerTool())

        self.system_prompt = format_prompt_with_tools(
            self._toolbox, self.system_prompt_template, self.tool_description_template
        )
        self.prompt = None
        self.logs = []
        self.task = None
        self.memory_verbose = memory_verbose

        if verbose == 0:
            logger.setLevel(logging.WARNING)
        elif verbose == 1:
            logger.setLevel(logging.INFO)
        elif verbose == 2:
            logger.setLevel(logging.DEBUG)

    @property
    def toolbox(self) -> Toolbox:
        """Get the toolbox currently available to the agent"""
        return self._toolbox

    def initialize_for_run(self):
        self.token_count = 0
        self.system_prompt = format_prompt_with_tools(
            self._toolbox,
            self.system_prompt_template,
            self.tool_description_template,
        )
        if hasattr(self, "authorized_imports"):
            self.system_prompt = format_prompt_with_imports(
                self.system_prompt, list(set(LIST_SAFE_MODULES) | set(self.authorized_imports))
            )
        self.logs = [{"system_prompt": self.system_prompt, "task": self.task}]
        self.logger.warn("======== New task ========")
        self.logger.log(33, self.task)
        self.logger.debug("System prompt is as follows:")
        self.logger.debug(self.system_prompt)

    def write_inner_memory_from_logs(self, summary_mode: Optional[bool] = False) -> List[Dict[str, str]]:
        """
        Reads past llm_outputs, actions, and observations or errors from the logs into a series of messages
        that can be used as input to the LLM.
        """
        prompt_message = {"role": MessageRole.SYSTEM, "content": self.logs[0]["system_prompt"]}
        task_message = {
            "role": MessageRole.USER,
            "content": "Task: " + self.logs[0]["task"],
        }
        if summary_mode:
            memory = [task_message]
        else:
            memory = [prompt_message, task_message]
        for i, step_log in enumerate(self.logs[1:]):
            if "llm_output" in step_log and not summary_mode:
                thought_message = {"role": MessageRole.ASSISTANT, "content": step_log["llm_output"].strip()}
                memory.append(thought_message)
            if "facts" in step_log:
                thought_message = {
                    "role": MessageRole.ASSISTANT,
                    "content": "[FACTS LIST]:\n" + step_log["facts"].strip(),
                }
                memory.append(thought_message)

            if "plan" in step_log and not summary_mode:
                thought_message = {"role": MessageRole.ASSISTANT, "content": "[PLAN]:\n" + step_log["plan"].strip()}
                memory.append(thought_message)

            if "tool_call" in step_log and summary_mode:
                tool_call_message = {
                    "role": MessageRole.ASSISTANT,
                    "content": f"[STEP {i} TOOL CALL]: " + str(step_log["tool_call"]).strip(),
                }
                memory.append(tool_call_message)

            if "task" in step_log:
                tool_call_message = {
                    "role": MessageRole.USER,
                    "content": "New task:\n" + step_log["task"],
                }
                memory.append(tool_call_message)

            if "error" in step_log or "observation" in step_log:
                if "error" in step_log:
                    message_content = (
                        f"[OUTPUT OF STEP {i}] Error: "
                        + str(step_log["error"])
                        + "\nNow let's retry: take care not to repeat previous errors! If you have retried several times, try a completely different approach.\n"
                    )
                elif "observation" in step_log:
                    message_content = f"[OUTPUT OF STEP {i}] Observation:\n{step_log['observation']}"
                tool_response_message = {"role": MessageRole.TOOL_RESPONSE, "content": message_content}
                memory.append(tool_response_message)

        return memory

    def get_succinct_logs(self):
        return [{key: value for key, value in log.items() if key != "agent_memory"} for log in self.logs]

    def extract_action(self, llm_output: str, split_token: str) -> str:
        """
        Parse action from the LLM output

        Args:
            llm_output (`str`): Output of the LLM
            split_token (`str`): Separator for the action. Should match the example in the system prompt.
        """
        try:
            split = llm_output.split(split_token)
            rationale, action = (
                split[-2],
                split[-1],
            )  # NOTE: using indexes starting from the end solves for when you have more than one split_token in the output
        except Exception as e:
            self.logger.error(e, exc_info=1)
            raise AgentParsingError(
                f"Error: No '{split_token}' token provided in your output.\nYour output:\n{llm_output}\n. Be sure to include an action, prefaced with '{split_token}'!"
            )
        return rationale, action

    def execute_tool_call(self, tool_name: str, arguments: Dict[str, str]) -> Any:
        """
        Execute tool with the provided input and returns the result.
        This method replaces arguments with the actual values from the state if they refer to state variables.

        Args:
            tool_name (`str`): Name of the Tool to execute (should be one from self.toolbox).
            arguments (Dict[str, str]): Arguments passed to the Tool.
        """
        if tool_name not in self.toolbox.tools:
            error_msg = f"Error: unknown tool {tool_name}, should be instead one of {list(self.toolbox.tools.keys())}."
            self.logger.error(error_msg, exc_info=1)
            raise AgentExecutionError(error_msg)

        try:
            if isinstance(arguments, str):
                observation = self.toolbox.tools[tool_name](arguments)
            else:
                for key, value in arguments.items():
                    # if the value is the name of a state variable like "image.png", replace it with the actual value
                    if isinstance(value, str) and value in self.state:
                        arguments[key] = self.state[value]
                observation = self.toolbox.tools[tool_name](**arguments)
            return observation
        except Exception as e:
            raise AgentExecutionError(
                f"Error in tool call execution: {e}\nYou should only use this tool with a correct input.\n"
                f"As a reminder, this tool's description is the following:\n{get_tool_description_with_args(self.toolbox.tools[tool_name])}"
            )

    def log_code_action(self, code_action: str) -> None:
        self.logger.warning("==== Agent is executing the code below:")
        if is_pygments_available():
            self.logger.log(
                31, highlight(code_action, PythonLexer(ensurenl=False), Terminal256Formatter(style="nord"))
            )
        else:
            self.logger.log(31, code_action)
        self.logger.warning("====")

    def run(self, **kwargs):
        """To be implemented in the child class"""
        raise NotImplementedError


class CodeAgent(Agent):
    """
    A class for an agent that solves the given task using a single block of code. It plans all its actions, then executes all in one shot.
    """

    def __init__(
        self,
        tools: List[Tool],
        llm_engine: Callable = HfEngine(),
        system_prompt: str = DEFAULT_CODE_SYSTEM_PROMPT,
        tool_description_template: str = DEFAULT_TOOL_DESCRIPTION_TEMPLATE,
        additional_authorized_imports: Optional[List[str]] = None,
        **kwargs,
    ):
        super().__init__(
            tools=tools,
            llm_engine=llm_engine,
            system_prompt=system_prompt,
            tool_description_template=tool_description_template,
            **kwargs,
        )

        if not is_pygments_available():
            transformers_logging.warning_once(
                logger,
                "pygments isn't installed. Installing pygments will enable color syntax highlighting in the "
                "CodeAgent.",
            )

        self.python_evaluator = evaluate_python_code
        self.additional_authorized_imports = additional_authorized_imports if additional_authorized_imports else []
        self.authorized_imports = list(set(LIST_SAFE_MODULES) | set(self.additional_authorized_imports))
        self.system_prompt = self.system_prompt.replace("<<authorized_imports>>", str(self.authorized_imports))

    def parse_code_blob(self, result: str) -> str:
        """
        Override this method if you want to change the way the code is
        cleaned in the `run` method.
        """
        return parse_code_blob(result)

    def run(self, task: str, return_generated_code: bool = False, **kwargs):
        """
        Runs the agent for the given task.

        Args:
            task (`str`): The task to perform
            return_generated_code (`bool`, *optional*, defaults to `False`): Whether to return the generated code instead of running it
            kwargs (additional keyword arguments, *optional*):
                Any keyword argument to send to the agent when evaluating the code.

        Example:

        ```py
        from transformers.agents import CodeAgent, PythonInterpreterTool

        python_interpreter = PythonInterpreterTool()
        agent = CodeAgent(tools=[python_interpreter])
        agent.run("What is the result of 2 power 3.7384?")
        ```
        """
        self.task = task
        if len(kwargs) > 0:
            self.task += f"\nYou have been provided with these initial arguments: {str(kwargs)}."
        self.state = kwargs.copy()
        self.initialize_for_run()

        # Run LLM
        prompt_message = {"role": MessageRole.SYSTEM, "content": self.system_prompt}
        task_message = {
            "role": MessageRole.USER,
            "content": "Task: " + self.task,
        }

        self.prompt = [prompt_message, task_message]
        self.logger.info("====Executing with this prompt====")
        self.logger.info(self.prompt)
        llm_output = self.llm_engine(self.prompt, stop_sequences=["<end_action>"])

        if return_generated_code:
            return llm_output

        # Parse
        try:
            _, code_action = self.extract_action(llm_output=llm_output, split_token="Code:")
        except Exception as e:
            self.logger.debug(
                f"Error in extracting action, trying to parse the whole output as code. Error trace: {e}"
            )
            code_action = llm_output

        try:
            code_action = self.parse_code_blob(code_action)
        except Exception as e:
            error_msg = f"Error in code parsing: {e}. Be sure to provide correct code"
            self.logger.error(error_msg, exc_info=1)
            return error_msg

        # Execute
        self.log_code_action(code_action)
        try:
            available_tools = {**BASE_PYTHON_TOOLS.copy(), **self.toolbox.tools}
            output = self.python_evaluator(
                code_action,
                static_tools=available_tools,
                custom_tools={},
                state=self.state,
                authorized_imports=self.authorized_imports,
            )
            self.logger.info(self.state["print_outputs"])
            return output
        except Exception as e:
            error_msg = f"Error in execution: {e}. Be sure to provide correct code."
            self.logger.error(error_msg, exc_info=1)
            return error_msg


class ReactAgent(Agent):
    """
    This agent that solves the given task step by step, using the ReAct framework:
    While the objective is not reached, the agent will perform a cycle of thinking and acting.
    The action will be parsed from the LLM output: it consists in calls to tools from the toolbox, with arguments chosen by the LLM engine.
    """

    def __init__(
        self,
        tools: List[Tool],
        llm_engine: Callable = HfEngine(),
        system_prompt: str = DEFAULT_REACT_CODE_SYSTEM_PROMPT,
        tool_description_template: str = DEFAULT_TOOL_DESCRIPTION_TEMPLATE,
        plan_type: Literal[tuple(SUPPORTED_PLAN_TYPES)] = SUPPORTED_PLAN_TYPES[0],
        planning_interval: Optional[int] = None,
        **kwargs,
    ):
        assert plan_type in SUPPORTED_PLAN_TYPES, f"plan type {plan_type} is not supported"
        super().__init__(
            tools=tools,
            llm_engine=llm_engine,
            system_prompt=system_prompt,
            tool_description_template=tool_description_template,
            **kwargs,
        )
        self.planning_interval = planning_interval
        self.plan_type = plan_type

    def provide_final_answer(self, task) -> str:
        """
        This method provides a final answer to the task, based on the logs of the agent's interactions.
        """
        self.prompt = [
            {
                "role": MessageRole.SYSTEM,
                "content": "An agent tried to answer an user query but it got stuck and failed to do so. You are tasked with providing an answer instead. Here is the agent's memory:",
            }
        ]
        self.prompt += self.write_inner_memory_from_logs()[1:]
        self.prompt += [
            {
                "role": MessageRole.USER,
                "content": f"Based on the above, please provide an answer to the following user request:\n{task}",
            }
        ]
        try:
            return self.llm_engine(self.prompt)
        except Exception as e:
            return f"Error in generating final llm output: {e}."

    def run(self, task: str, stream: bool = False, reset: bool = True, **kwargs):
        """
        Runs the agent for the given task.

        Args:
            task (`str`): The task to perform

        Example:
        ```py
        from transformers.agents import ReactCodeAgent
        agent = ReactCodeAgent(tools=[])
        agent.run("What is the result of 2 power 3.7384?")
        ```
        """
        self.task = task
        if len(kwargs) > 0:
            self.task += f"\nYou have been provided with these initial arguments: {str(kwargs)}."
        self.state = kwargs.copy()
        if reset:
            self.initialize_for_run()
        else:
            self.logs.append({"task": task})
        if stream:
            return self.stream_run(task)
        else:
            return self.direct_run(task)

    def stream_run(self, task: str):
        """
        Runs the agent in streaming mode, yielding steps as they are executed: should be launched only in the `run` method.
        """
        final_answer = None
        iteration = 0
        while final_answer is None and iteration < self.max_iterations:
            try:
                step_logs = self.step()
                if "final_answer" in step_logs:
                    final_answer = step_logs["final_answer"]
            except AgentError as e:
                self.logger.error(e, exc_info=1)
                self.logs[-1]["error"] = e
            finally:
                iteration += 1
                yield self.logs[-1]

        if final_answer is None and iteration == self.max_iterations:
            error_message = "Reached max iterations."
            final_step_log = {"error": AgentMaxIterationsError(error_message)}
            self.logs.append(final_step_log)
            self.logger.error(error_message, exc_info=1)
            final_answer = self.provide_final_answer(task)
            final_step_log["final_answer"] = final_answer
            yield final_step_log

        yield final_answer

    def direct_run(self, task: str):
        """
        Runs the agent in direct mode, returning outputs only at the end: should be launched only in the `run` method.
        """
        final_answer = None
        iteration = 0
        while final_answer is None and iteration < self.max_iterations:
            try:
                if self.planning_interval is not None and iteration % self.planning_interval == 0:
                    self.planning_step(task, is_first_step=(iteration == 0), iteration=iteration)
                step_logs = self.step()
                if "final_answer" in step_logs:
                    final_answer = step_logs["final_answer"]
            except AgentError as e:
                self.logger.error(e, exc_info=1)
                self.logs[-1]["error"] = e
            finally:
                iteration += 1

        if final_answer is None and iteration == self.max_iterations:
            error_message = "Reached max iterations."
            final_step_log = {"error": AgentMaxIterationsError(error_message)}
            self.logs.append(final_step_log)
            self.logger.error(error_message, exc_info=1)
            final_answer = self.provide_final_answer(task)
            final_step_log["final_answer"] = final_answer

        return final_answer

    def planning_step(self, task, is_first_step: bool = False, iteration: int = None):
        """
        Used periodically by the agent to plan the next steps to reach the objective.

        Args:
            task (`str`): The task to perform
            is_first_step (`bool`): If this step is not the first one, the plan should be an update over a previous plan.
            iteration (`int`): The number of the current step, used as an indication for the LLM.
        """
        if is_first_step:
            message_prompt_facts = {"role": MessageRole.SYSTEM, "content": SYSTEM_PROMPT_FACTS}
            message_prompt_task = {
                "role": MessageRole.USER,
                "content": f"""Here is the task:
```
{task}
```
Now begin!""",
            }

            answer_facts = self.llm_engine([message_prompt_facts, message_prompt_task])

            message_system_prompt_plan = {
                "role": MessageRole.SYSTEM,
                "content": PROMPTS_FOR_INITIAL_PLAN[self.plan_type]["system"],
            }
            message_user_prompt_plan = {
                "role": MessageRole.USER,
                "content": PROMPTS_FOR_INITIAL_PLAN[self.plan_type]["user"].format(
                    task=task,
                    tool_descriptions=self._toolbox.show_tool_descriptions(self.tool_description_template),
                    answer_facts=answer_facts,
                ),
            }
            answer_plan = self.llm_engine(
                [message_system_prompt_plan, message_user_prompt_plan], stop_sequences=["<end_plan>"]
            )

            final_plan_redaction = f"""Here is the plan of action that I will follow to solve the task:
```
{answer_plan}
```"""
            final_facts_redaction = f"""Here are the facts that I know so far:
```
{answer_facts}
```""".strip()
            self.logs.append({"plan": final_plan_redaction, "facts": final_facts_redaction})
            self.logger.debug("===== Initial plan: =====")
            self.logger.debug(final_plan_redaction)
        else:  # update plan
            agent_memory = self.write_inner_memory_from_logs(
                summary_mode=False
            )  # This will not log the plan but will log facts

            # Redact updated facts
            facts_update_system_prompt = {
                "role": MessageRole.SYSTEM,
                "content": SYSTEM_PROMPT_FACTS_UPDATE,
            }
            facts_update_message = {
                "role": MessageRole.USER,
                "content": USER_PROMPT_FACTS_UPDATE,
            }
            facts_update = self.llm_engine([facts_update_system_prompt] + agent_memory + [facts_update_message])

            # Redact updated plan
            plan_update_message = {
                "role": MessageRole.SYSTEM,
                "content": PROMPTS_FOR_PLAN_UPDATE[self.plan_type]["system"].format(task=task),
            }
            plan_update_message_user = {
                "role": MessageRole.USER,
                "content": PROMPTS_FOR_PLAN_UPDATE[self.plan_type]["user"].format(
                    task=task,
                    tool_descriptions=self._toolbox.show_tool_descriptions(self.tool_description_template),
                    facts_update=facts_update,
                    remaining_steps=(self.max_iterations - iteration),
                ),
            }
            plan_update = self.llm_engine(
                [plan_update_message] + agent_memory + [plan_update_message_user], stop_sequences=["<end_plan>"]
            )

            # Log final facts and plan
            final_plan_redaction = PLAN_UPDATE_FINAL_PLAN_REDACTION.format(task=task, plan_update=plan_update)
            final_facts_redaction = f"""Here is the updated list of the facts that I know:
```
{facts_update}
```"""
            self.logs.append({"plan": final_plan_redaction, "facts": final_facts_redaction})
            self.logger.debug("===== Updated plan: =====")
            self.logger.debug(final_plan_redaction)


class ReactJsonAgent(ReactAgent):
    """
    This agent that solves the given task step by step, using the ReAct framework:
    While the objective is not reached, the agent will perform a cycle of thinking and acting.
    The tool calls will be formulated by the LLM in JSON format, then parsed and executed.
    """

    def __init__(
        self,
        tools: List[Tool],
        llm_engine: Callable = HfEngine(),
        system_prompt: str = DEFAULT_REACT_JSON_SYSTEM_PROMPT,
        tool_description_template: str = DEFAULT_TOOL_DESCRIPTION_TEMPLATE,
        planning_interval: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(
            tools=tools,
            llm_engine=llm_engine,
            system_prompt=system_prompt,
            tool_description_template=tool_description_template,
            planning_interval=planning_interval,
            **kwargs,
        )

    def step(self):
        """
        Perform one step in the ReAct framework: the agent thinks, acts, and observes the result.
        The errors are raised here, they are caught and logged in the run() method.
        """
        agent_memory = self.write_inner_memory_from_logs()

        self.prompt = agent_memory
        self.logger.debug("===== New step =====")

        # Add new step in logs
        current_step_logs = {}
        self.logs.append(current_step_logs)
        current_step_logs["agent_memory"] = agent_memory.copy()

        self.logger.info("===== Calling LLM with this last message: =====")
        self.logger.info(self.prompt[-1])

        try:
            llm_output = self.llm_engine(self.prompt, stop_sequences=["<end_action>", "Observation:"])
        except Exception as e:
            raise AgentGenerationError(f"Error in generating llm output: {e}.")
        self.logger.debug("===== Output message of the LLM: =====")
        self.logger.debug(llm_output)
        current_step_logs["llm_output"] = llm_output

        # Parse
        self.logger.debug("===== Extracting action =====")
        rationale, action = self.extract_action(llm_output=llm_output, split_token="Action:")

        try:
            tool_name, arguments = self.tool_parser(action)
        except Exception as e:
            raise AgentParsingError(f"Could not parse the given action: {e}.")

        current_step_logs["rationale"] = rationale
        current_step_logs["tool_call"] = {"tool_name": tool_name, "tool_arguments": arguments}

        # Execute
        self.logger.warning(f"Calling tool: '{tool_name}' with arguments: {arguments}")
        if tool_name == "final_answer":
            if isinstance(arguments, dict):
                if "answer" in arguments:
                    answer = arguments["answer"]
                    if (
                        isinstance(answer, str) and answer in self.state.keys()
                    ):  # if the answer is a state variable, return the value
                        answer = self.state[answer]
                else:
                    answer = arguments
            else:
                answer = arguments
            current_step_logs["final_answer"] = answer
            return current_step_logs
        else:
            observation = self.execute_tool_call(tool_name, arguments)
            observation_type = type(observation)
            if observation_type == AgentText:
                updated_information = str(observation).strip()
            else:
                # TODO: observation naming could allow for different names of same type
                if observation_type == AgentImage:
                    observation_name = "image.png"
                elif observation_type == AgentAudio:
                    observation_name = "audio.mp3"
                else:
                    observation_name = "object.object"

                self.state[observation_name] = observation
                updated_information = f"Stored '{observation_name}' in memory."

            self.logger.info(updated_information)
            current_step_logs["observation"] = updated_information
            return current_step_logs


class ReactCodeAgent(ReactAgent):
    """
    This agent that solves the given task step by step, using the ReAct framework:
    While the objective is not reached, the agent will perform a cycle of thinking and acting.
    The tool calls will be formulated by the LLM in code format, then parsed and executed.
    """

    def __init__(
        self,
        tools: List[Tool],
        llm_engine: Callable = HfEngine(),
        system_prompt: str = DEFAULT_REACT_CODE_SYSTEM_PROMPT,
        tool_description_template: str = DEFAULT_TOOL_DESCRIPTION_TEMPLATE,
        additional_authorized_imports: Optional[List[str]] = None,
        planning_interval: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(
            tools=tools,
            llm_engine=llm_engine,
            system_prompt=system_prompt,
            tool_description_template=tool_description_template,
            planning_interval=planning_interval,
            **kwargs,
        )

        if not is_pygments_available():
            transformers_logging.warning_once(
                logger,
                "pygments isn't installed. Installing pygments will enable color syntax highlighting in the "
                "ReactCodeAgent.",
            )

        self.python_evaluator = evaluate_python_code
        self.additional_authorized_imports = additional_authorized_imports if additional_authorized_imports else []
        self.authorized_imports = list(set(LIST_SAFE_MODULES) | set(self.additional_authorized_imports))
        self.system_prompt = self.system_prompt.replace("<<authorized_imports>>", str(self.authorized_imports))
        self.custom_tools = {}

    def step(self):
        """
        Perform one step in the ReAct framework: the agent thinks, acts, and observes the result.
        The errors are raised here, they are caught and logged in the run() method.
        """
        agent_memory = self.write_inner_memory_from_logs()

        self.prompt = agent_memory.copy()

        self.logger.debug("===== New step =====")

        # Add new step in logs
        current_step_logs = {}
        self.logs.append(current_step_logs)
        current_step_logs["agent_memory"] = agent_memory.copy()

        self.logger.info("===== Calling LLM with these last messages: =====")
        self.logger.info(self.prompt[-2:])

        try:
            llm_output = self.llm_engine(self.prompt, stop_sequences=["<end_action>", "Observation:"])
        except Exception as e:
            raise AgentGenerationError(f"Error in generating llm output: {e}.")

        self.logger.debug("===== Output message of the LLM: =====")
        self.logger.debug(llm_output)
        current_step_logs["llm_output"] = llm_output

        # Parse
        self.logger.debug("===== Extracting action =====")
        try:
            rationale, raw_code_action = self.extract_action(llm_output=llm_output, split_token="Code:")
        except Exception as e:
            self.logger.debug(f"Error in extracting action, trying to parse the whole output. Error trace: {e}")
            rationale, raw_code_action = llm_output, llm_output

        try:
            code_action = parse_code_blob(raw_code_action)
        except Exception as e:
            error_msg = f"Error in code parsing: {e}. Make sure to provide correct code"
            raise AgentParsingError(error_msg)

        current_step_logs["rationale"] = rationale
        current_step_logs["tool_call"] = {"tool_name": "code interpreter", "tool_arguments": code_action}

        # Execute
        self.log_code_action(code_action)
        try:
            result = self.python_evaluator(
                code_action,
                static_tools={
                    **BASE_PYTHON_TOOLS.copy(),
                    **self.toolbox.tools,
                },
                custom_tools=self.custom_tools,
                state=self.state,
                authorized_imports=self.authorized_imports,
            )
            information = self.state["print_outputs"]
            self.logger.warning("Print outputs:")
            self.logger.log(32, information)
            current_step_logs["observation"] = information
        except Exception as e:
            error_msg = f"Code execution failed due to the following error:\n{str(e)}"
            if "'dict' object has no attribute 'read'" in str(e):
                error_msg += "\nYou get this error because you passed a dict as input for one of the arguments instead of a string."
            raise AgentExecutionError(error_msg)
        for line in code_action.split("\n"):
            if line[: len("final_answer")] == "final_answer":
                self.logger.warning(">>> Final answer:")
                self.logger.log(32, result)
                current_step_logs["final_answer"] = result
        return current_step_logs
