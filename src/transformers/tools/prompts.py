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
import re

from ..utils import cached_file


# docstyle-ignore
CHAT_MESSAGE_PROMPT = """
Human: <<task>>

Assistant: """


DEFAULT_PROMPTS_REPO = "huggingface-tools/default-prompts"
PROMPT_FILES = {"chat": "chat_prompt_template.txt", "run": "run_prompt_template.txt"}


def download_prompt(prompt_or_repo_id, agent_name, mode="run"):
    """
    Downloads and caches the prompt from a repo and returns it contents (if necessary)
    """
    if prompt_or_repo_id is None:
        prompt_or_repo_id = DEFAULT_PROMPTS_REPO

    # prompt is considered a repo ID when it does not contain any kind of space
    if re.search("\\s", prompt_or_repo_id) is not None:
        return prompt_or_repo_id

    prompt_file = cached_file(
        prompt_or_repo_id, PROMPT_FILES[mode], repo_type="dataset", user_agent={"agent": agent_name}
    )
    with open(prompt_file, "r", encoding="utf-8") as f:
        return f.read()



DEFAULT_REACT_SYSTEM_PROMPT = """Solve the following task as best you can. You have access to the following tools:

{tool_descriptions}

The way you use the tools is by specifying a json blob.
Specifically, this json should have a `action` key (name of the tool to use) and a `action_input` key (input to the tool).

The value in the "action" field should belong to this list: {tool_names}.

The $ACTION_JSON_BLOB should only contain a SINGLE action, do NOT return a list of multiple actions. It should be formatted in markdown. Do not try to escape scpecial characters. Here is an example of a valid $ACTION_JSON_BLOB:
```json
{{
  "action": $TOOL_NAME,
  "action_input": $INPUT
}}
```
Make sure to have the $INPUT in the right format for the tool you are using, and do not put variable names as input if you can find the right values.

You will be given:

Task: the task you are given.

You should ALWAYS use the following format:

Thought: you should always think about one action to take. Then use the action as follows:
Action:
$ACTION_JSON_BLOB
Observation: the result of the action
... (this Thought/Action/Observation can repeat N times, you should take several steps when needed. The $ACTION_JSON_BLOB must only use a SINGLE action at a time.)

ALWAYS provide a 'Thought:' and an 'Action:' part.
Use the 'final_answer' tool to provide the final answer to the task. It is the only way to complete the task, else you will be stuck on a loop.

Now begin!
"""
