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
from .agent_types import AgentAudio, AgentImage, AgentText, AgentType
from .agents import ReactAgent


def pull_message(step_log: dict):
    try:
        from gradio import ChatMessage
    except ImportError:
        raise ImportError("Gradio should be installed in order to launch a gradio demo.")

    if step_log.get("rationale"):
        yield ChatMessage(role="assistant", content=step_log["rationale"])
    if step_log.get("tool_call"):
        used_code = step_log["tool_call"]["tool_name"] == "code interpreter"
        content = step_log["tool_call"]["tool_arguments"]
        if used_code:
            content = f"```py\n{content}\n```"
        yield ChatMessage(
            role="assistant",
            metadata={"title": f"🛠️ Used tool {step_log['tool_call']['tool_name']}"},
            content=content,
        )
    if step_log.get("observation"):
        yield ChatMessage(role="assistant", content=f"```\n{step_log['observation']}\n```")
    if step_log.get("error"):
        yield ChatMessage(
            role="assistant",
            content=str(step_log["error"]),
            metadata={"title": "💥 Error"},
        )


def stream_from_transformers_agent(agent: ReactAgent, prompt: str):
    """Runs an agent with the given prompt and streams the messages from the agent as ChatMessages."""

    try:
        from gradio import ChatMessage
    except ImportError:
        raise ImportError("Gradio should be installed in order to launch a gradio demo.")

    class Output:
        output: AgentType | str = None

    for step_log in agent.run(prompt, stream=True):
        if isinstance(step_log, dict):
            for message in pull_message(step_log):
                print("message", message)
                yield message

    Output.output = step_log
    if isinstance(Output.output, AgentText):
        yield ChatMessage(role="assistant", content=f"**Final answer:**\n```\n{Output.output.to_string()}\n```")
    elif isinstance(Output.output, AgentImage):
        yield ChatMessage(
            role="assistant",
            content={"path": Output.output.to_string(), "mime_type": "image/png"},
        )
    elif isinstance(Output.output, AgentAudio):
        yield ChatMessage(
            role="assistant",
            content={"path": Output.output.to_string(), "mime_type": "audio/wav"},
        )
    else:
        return ChatMessage(role="assistant", content=Output.output)
