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
from copy import deepcopy
from enum import Enum
from typing import Dict, List

from huggingface_hub import InferenceClient


class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL_CALL = "tool-call"
    TOOL_RESPONSE = "tool-response"

    @classmethod
    def roles(cls):
        return [r.value for r in cls]


def get_clean_message_list(message_list: List[Dict[str, str]], role_conversions: Dict[str, str] = {}):
    """
    Subsequent messages with the same role will be concatenated to a single message.

    Args:
        message_list (`List[Dict[str, str]]`): List of chat messages.
    """
    final_message_list = []
    message_list = deepcopy(message_list)  # Avoid modifying the original list
    for message in message_list:
        if not set(message.keys()) == {"role", "content"}:
            raise ValueError("Message should contain only 'role' and 'content' keys!")

        role = message["role"]
        if role not in MessageRole.roles():
            raise ValueError(f"Incorrect role {role}, only {MessageRole.roles()} are supported for now.")

        if role in role_conversions:
            message["role"] = role_conversions[role]

        if len(final_message_list) > 0 and message["role"] == final_message_list[-1]["role"]:
            final_message_list[-1]["content"] += "\n=======\n" + message["content"]
        else:
            final_message_list.append(message)
    return final_message_list


llama_role_conversions = {
    MessageRole.TOOL_RESPONSE: MessageRole.USER,
}


class HfEngine:
    def __init__(self, model: str = "meta-llama/Meta-Llama-3-8B-Instruct"):
        self.model = model
        self.client = InferenceClient(model=self.model, timeout=120)

    def __call__(self, messages: List[Dict[str, str]], stop_sequences=[]) -> str:
        # Get clean message list
        messages = get_clean_message_list(messages, role_conversions=llama_role_conversions)

        # Get LLM output
        response = self.client.chat_completion(messages, stop=stop_sequences, max_tokens=1500)
        response = response.choices[0].message.content

        # Remove stop sequences from LLM output
        for stop_seq in stop_sequences:
            if response[-len(stop_seq) :] == stop_seq:
                response = response[: -len(stop_seq)]
        return response
