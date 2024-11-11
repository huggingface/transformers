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
from typing import Dict, List, Optional

from huggingface_hub import InferenceClient

from .. import AutoTokenizer
from ..pipelines.base import Pipeline
from ..utils import logging


logger = logging.get_logger(__name__)


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
    def __init__(self, model_id: Optional[str] = None):
        self.last_input_token_count = None
        self.last_output_token_count = None
        if model_id is None:
            model_id = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
            logger.warning(f"Using default model for token counting: '{model_id}'")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        except Exception as e:
            logger.warning(f"Failed to load tokenizer for model {model_id}: {e}. Loading default tokenizer instead.")
            self.tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-1.7B-Instruct")

    def get_token_counts(self):
        return {
            "input_token_count": self.last_input_token_count,
            "output_token_count": self.last_output_token_count,
        }

    def generate(
        self, messages: List[Dict[str, str]], stop_sequences: Optional[List[str]] = None, grammar: Optional[str] = None
    ):
        raise NotImplementedError

    def __call__(
        self, messages: List[Dict[str, str]], stop_sequences: Optional[List[str]] = None, grammar: Optional[str] = None
    ) -> str:
        if not isinstance(messages, List):
            raise ValueError("Messages should be a list of dictionaries with 'role' and 'content' keys.")
        if stop_sequences is None:
            stop_sequences = []
        response = self.generate(messages, stop_sequences, grammar)
        self.last_input_token_count = len(self.tokenizer.apply_chat_template(messages, tokenize=True))
        self.last_output_token_count = len(self.tokenizer.encode(response))

        # Remove stop sequences from LLM output
        for stop_seq in stop_sequences:
            if response[-len(stop_seq) :] == stop_seq:
                response = response[: -len(stop_seq)]
        return response


class HfApiEngine(HfEngine):
    """This engine leverages Hugging Face's Inference API service, either serverless or with a dedicated endpoint."""

    def __init__(self, model: str = "meta-llama/Meta-Llama-3.1-8B-Instruct", timeout: int = 120):
        super().__init__(model_id=model)
        self.model = model
        self.timeout = timeout
        self.client = InferenceClient(self.model, timeout=self.timeout)

    def generate(
        self, messages: List[Dict[str, str]], stop_sequences: Optional[List[str]] = None, grammar: Optional[str] = None
    ) -> str:
        # Get clean message list
        messages = get_clean_message_list(messages, role_conversions=llama_role_conversions)

        # Get LLM output
        if grammar is not None:
            response = self.client.chat_completion(
                messages, stop=stop_sequences, max_tokens=1500, response_format=grammar
            )
        else:
            response = self.client.chat_completion(messages, stop=stop_sequences, max_tokens=1500)

        response = response.choices[0].message.content
        return response


class TransformersEngine(HfEngine):
    """This engine uses a pre-initialized local text-generation pipeline."""

    def __init__(self, pipeline: Pipeline, model_id: Optional[str] = None):
        super().__init__(model_id)
        self.pipeline = pipeline

    def generate(
        self,
        messages: List[Dict[str, str]],
        stop_sequences: Optional[List[str]] = None,
        grammar: Optional[str] = None,
        max_length: int = 1500,
    ) -> str:
        # Get clean message list
        messages = get_clean_message_list(messages, role_conversions=llama_role_conversions)

        # Get LLM output
        if stop_sequences is not None and len(stop_sequences) > 0:
            stop_strings = stop_sequences
        else:
            stop_strings = None

        output = self.pipeline(
            messages,
            stop_strings=stop_strings,
            max_length=max_length,
            tokenizer=self.pipeline.tokenizer,
        )

        response = output[0]["generated_text"][-1]["content"]
        return response


DEFAULT_JSONAGENT_REGEX_GRAMMAR = {
    "type": "regex",
    "value": 'Thought: .+?\\nAction:\\n\\{\\n\\s{4}"action":\\s"[^"\\n]+",\\n\\s{4}"action_input":\\s"[^"\\n]+"\\n\\}\\n<end_action>',
}

DEFAULT_CODEAGENT_REGEX_GRAMMAR = {
    "type": "regex",
    "value": "Thought: .+?\\nCode:\\n```(?:py|python)?\\n(?:.|\\s)+?\\n```<end_action>",
}
