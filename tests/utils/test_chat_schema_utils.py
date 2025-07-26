# Copyright 2025 The HuggingFace Team. All rights reserved.
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

from transformers.utils.chat_parsing_utils import recursive_parse
from transformers.utils.chat_template_utils import get_json_schema
from transformers import AutoTokenizer

import unittest

basic_template = """
{%- for message in messages %}
    {{- "<|im_start|>" + message["role"] + "\n" }}
    {{- message["content"] }}
    {{- "<|im_end|>\n" }}
{%- endfor %}
""".strip()

tool_template = """
{% if tools %}
    {{- "<|tools|>\n" }}
    {{- tools|tojson(indent=2) }}
    {{- "\n" }}
    {{- "<|endtools|>\n" }}
{% endif %}
{%- for message in messages %}
    {{- "<|im_start|>" + message["role"] + "\n" }}
    {{- message["content"] }}
    {{- "<|im_end|>\n" }}
{%- endfor %}
""".strip()

basic_schema = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "role": {"type": "string", "x-regex": r"<\|im_start\|>(.*?)\n"},
            "content": {"type": "string", "x-regex": r"<\|im_start\|>.*?\n(.*?)<\|im_end\|>\n"}
        },
        "required": ["role", "content"]
    },
    "x-regex-iterator": r"\<\|im_start\|\>.*?\<\|im_end\|\>\n",
}

basic_schema_with_named_groups = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "role": {"type": "string"},
            "content": {"type": "string"}
        },
        "required": ["role", "content"]
    },
    "x-regex-iterator": r"<\|im_start\|>(?P<role>.*?)\n(?P<content>.*?)<\|im_end\|>\n",
}

tools_schema_with_named_groups = {
    "type": "object",
    "properties": {
        "tools": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "type": {"type": "string", "enum": ["function"]},
                    "function": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "description": {"type": "string"},
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "type": {"type": "string", "enum": ["object"]},
                                    "required": {"type": "array", "items": {"type": "string"}},
                                    "properties": {
                                        "type": "object",
                                        "additionalProperties": {
                                            "type": "object",
                                            "properties": {
                                                    "type": {"type": "string"},
                                                    "description": {"type": "string"},
                                                    "enum": {"type": "array", "items": {"type": "string"}},
                                            }
                                        }
                                    },
                                }
                            }
                        }
                    }
                },
            },
            "x-parser": "json",
        },
        "messages": basic_schema_with_named_groups
    },
    "x-regex": r"(?:<\|tools\|>\n(?P<tools>.*?)\n<\|endtools\|>\n)?(?P<messages>.*)",
}

class ChatSchemaParserTest(unittest.TestCase):
    def setUp(self):
        # This tokenizer has no chat template by default
        self.tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-gpt2")

    def test_basic_chat(self):
        chat = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi there! How can I help you today?"},
        ]
        self.tokenizer.chat_template = basic_template
        formatted_chat = self.tokenizer.apply_chat_template(chat, tokenize=False)
        parsed_chat = recursive_parse(formatted_chat, basic_schema)
        self.assertEqual(parsed_chat, chat)

    def test_named_groups(self):
        chat = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi there! How can I help you today?"},
        ]
        self.tokenizer.chat_template = basic_template
        formatted_chat = self.tokenizer.apply_chat_template(chat, tokenize=False)
        parsed_chat = recursive_parse(formatted_chat, basic_schema_with_named_groups)
        self.assertEqual(parsed_chat, chat)

    def test_tool_def_parsing(self):
        def tool(temperature_format: str):
            """
            Test function

            Args:
                temperature_format: The temperature format to use (Choices: ["celsius", "fahrenheit"])


            Returns:
                The temperature
            """
            return -40.0

        chat = [
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi there! How can I help you today?"},
        ]
        self.tokenizer.chat_template = tool_template
        formatted_chat = self.tokenizer.apply_chat_template(chat, tokenize=False, tools=[tool])
        parsed_chat = recursive_parse(formatted_chat, tools_schema_with_named_groups)
        self.assertEqual(parsed_chat['messages'], chat)
        self.assertEqual(parsed_chat['tools'], [get_json_schema(tool)])

    def test_tool_template_with_no_tools(self):
        chat = [
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi there! How can I help you today?"},
        ]
        self.tokenizer.chat_template = tool_template
        formatted_chat = self.tokenizer.apply_chat_template(chat, tokenize=False)
        parsed_chat = recursive_parse(formatted_chat, tools_schema_with_named_groups)
        # Test we still extract messages
        self.assertEqual(parsed_chat['messages'], chat)