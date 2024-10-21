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
from abc import ABC, abstractmethod
from copy import deepcopy
from enum import Enum
from transformers import AutoTokenizer
from typing import Dict, List, Optional

from huggingface_hub import InferenceClient

from ..pipelines.base import Pipeline


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

class LlmEngine(ABC):
    def __init__(self, tokenizer):
        """Accepts a tokenizer which should have a .encode() method taking as input a string"""
        self.get_token_count = tokenizer
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    @abstractmethod
    def _generate_response(self, messages: List[Dict[str, str]], stop_sequences: List[str], grammar: Optional[str]) -> str:
        pass

    def __call__(self, messages: List[Dict[str, str]], stop_sequences: List[str] = [], grammar: Optional[str] = None) -> str:
        # Count input tokens
        input_text = self._messages_to_text(messages)
        input_tokens = len(self.tokenizer.encode(input_text))
        self.total_input_tokens += input_tokens

        # Generate response
        response = self._generate_response(messages, stop_sequences, grammar)

        # Count output tokens
        output_tokens = len(self.tokenizer.encode(response))
        self.total_output_tokens += output_tokens

        return response

    def _messages_to_text(self, messages: List[Dict[str, str]]) -> str:
        return "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])

    def get_token_usage(self):
        return {
            "input_tokens": self.total_input_tokens,
            "output_tokens": self.total_output_tokens,
            "total_tokens": self.total_input_tokens + self.total_output_tokens
        }


class HfEngine(LlmEngine):
    """Base class for engines that count tokens using an AutoTokenizer"""

    def __init__(self, model: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"):
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        super().__init__(self.tokenizer)


class HfApiEngine(HfEngine):
    """This engine leverages Hugging Face's Inference API service, either serverless or with a dedicated endpoint."""

    def __init__(self, model: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"):
        super().__init__(model)
        self.client = InferenceClient(self.model_name, timeout=120)

    def _generate_response(self, messages: List[Dict[str, str]], stop_sequences: List[str], grammar: Optional[str]) -> str:
        messages = get_clean_message_list(messages, role_conversions=llama_role_conversions)

        if grammar is not None:
            response = self.client.chat_completion(
                messages, stop=stop_sequences, max_tokens=1500, response_format=grammar
            )
        else:
            response = self.client.chat_completion(messages, stop=stop_sequences, max_tokens=1500)

        response = response.choices[0].message.content

        for stop_seq in stop_sequences:
            if response[-len(stop_seq):] == stop_seq:
                response = response[:-len(stop_seq)]
        return response


class TransformersEngine(HfEngine):
    """This engine uses a pre-initialized local text-generation pipeline."""

    def __init__(self, pipeline: Pipeline):
        super().__init__(pipeline.model.config._name_or_path)
        self.pipeline = pipeline

    def _generate_response(self, messages: List[Dict[str, str]], stop_sequences: Optional[List[str]], grammar: Optional[str]) -> str:
        messages = get_clean_message_list(messages, role_conversions=llama_role_conversions)

        output = self.pipeline(
            messages,
            stop_strings=stop_sequences,
            max_length=1500,
            tokenizer=self.pipeline.tokenizer,
        )

        response = output[0]["generated_text"][-1]["content"]

        if stop_sequences is not None:
            for stop_seq in stop_sequences:
                if response[-len(stop_seq):] == stop_seq:
                    response = response[:-len(stop_seq)]
        return response


DEFAULT_JSONAGENT_REGEX_GRAMMAR = {
    "type": "regex",
    "value": 'Thought: .+?\\nAction:\\n\\{\\n\\s{4}"action":\\s"[^"\\n]+",\\n\\s{4}"action_input":\\s"[^"\\n]+"\\n\\}\\n<end_action>',
}

DEFAULT_CODEAGENT_REGEX_GRAMMAR = {
    "type": "regex",
    "value": "Thought: .+?\\nCode:\\n```(?:py|python)?\\n(?:.|\\s)+?\\n```<end_action>",
}
