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
from typing import Optional, Union

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

cohere_schema = {
    "type": "object",
    "properties": {
        "tools": {
            "type": "array",
            "x-regex": "\n## Available Tools\nHere is a list of tools that you have available to you:((?:\n\n```python\n.*?[\n]+```)+)",
            "x-regex-iterator": "```python\n(.*?)\n+```",
            "items": {
                "type": "object",
                "properties": {
                    "type": {"const": "function"},
                    "function": {
                        "type": "object",
                        "x-scope-vars-dict": {
                            # TODO This is hard because I don't have a way to extract a dict with keys derived from the regex.
                            #      But I need that if I want to do lookups later.
                            #      I think that means x-regex-iterator needs to work on objects if the regex returns "key" and "value" groups.
                            #      Once we have that, we need a way to select from scope-vars-dict by key in other branches of the schema.
                            "argument_types": {
                                "type": "object",

                            }
                        },
                        "properties": {
                            "name": {"type": "string", "x-regex": r"def (\w+)\("},
                            "description": {"type": "string", "x-regex": r"\"\"\"(.*?)(?:\"\"\"|\n+\s*Args:\n)"},
                            "parameters": {
                                "type": "object",
                                "x-regex": r"\n\s*Args:\n\s*(.*?)$",
                                "properties": {

                                }
                            },

                        }
                    }
                }
            }
        },
        "messages": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "role": {"type": "string", "enum": ["user", "assistant", "system"]},
                    "content": {"type": "string"}
                },
                "required": ["role", "content"]
            },
            "x-regex-iterator": "<\|START_OF_TURN_TOKEN\|>(?P<role><\(?SYSTEM|USER|CHATBOT)_TOKEN\|>(?P<content>.*?)(?:<\|END_OF_TURN_TOKEN\|>|\n\n## Available Tools)",
        }
    }
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

    def test_horrific_template(self):
        def simple_tool(temperature_format: str):
            """
            Test function

            Args:
                temperature_format: The temperature format to use (Choices: ["celsius", "fahrenheit"])
            """
            return -40.0

        def tool_with_everything_all_at_once(x: str, y: int, z: float = 43.) -> float:
            """
            Test function with multiple args, and docstring args that we have to strip out.

            Args:
                x: The first input. It's got a big multiline
                   description and also contains
                   (choices: ["a", "b", "c"])

                y: The second input. It's a big list with a single-line description.

                z: The third input. It's some kind of tuple with a default arg.
            """
            return -40.0

        # Command-R is a real workout because it converts tools to Python function defs in the template
        # TODO Move this template to an internal-testing repo and add the schema too
        tokenizer = AutoTokenizer.from_pretrained("CohereLabs/c4ai-command-r-plus")
        chat = [
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi there! How can I help you today?"},
        ]
        formatted_chat = tokenizer.apply_chat_template(chat, tokenize=False, tools=[simple_tool, tool_with_everything_all_at_once], chat_template="tool_use")
        parsed_chat = recursive_parse(formatted_chat, cohere_schema)
        breakpoint()
        self.assertEqual(parsed_chat['messages'], chat)
        self.assertEqual(parsed_chat['tools'], [get_json_schema(simple_tool), tool_with_everything_all_at_once])