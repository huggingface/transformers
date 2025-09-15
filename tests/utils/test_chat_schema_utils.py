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

import unittest

from transformers import AutoTokenizer
from transformers.utils.chat_parsing_utils import recursive_parse
from transformers.utils.chat_template_utils import get_json_schema

cohere_schema = {
    "type": "object",
    "properties": {
        "role": {"const": "assistant"},
        "content": {"type": "string", "x-regex": "<\|START_RESPONSE\|>(.*?)(?:<\|END_RESPONSE\|>|$)"},
        "thinking": {"type": "string", "x-regex": "<\|START_THINKING\|>(.*?)(?:<\|END_THINKING\|>|$)"},
        "tool_calls": {
            "x-regex": "<\|START_ACTION\|>(.*?)(?:<\|END_ACTION\|>|$)",
            "x-parser": "json",
            "x-parser-args": {"transform": "[*].{type: 'function', function: {name: tool_name, arguments: parameters}}"},
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "type": {"const": "function"},
                    "function": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "arguments": {
                                "type": "object",
                                "additionalProperties": {"type": "any"},
                            },
                        }
                    }
                 }
            }
        },
    },
}

ernie_schema = {
    "type": "object",
    "properties": {
        "role": {"const": "assistant"},
        "content": {"type": "string", "x-regex": "<response>\n(.*?)\n?</response>"},
        "thinking": {"type": "string", "x-regex": "(?:^|<think>\s*)(.*?)\s*<\/think>"},
        "tool_calls": {
            "x-regex-iterator": "<tool_call>(.*?)</tool_call>",
            "type": "array",
            "items": {
                "type": "object",
                "x-parser": "json",
                "x-parser-args": {"transform": "{type: 'function', function: @}"},
                "properties": {
                    "type": {"const": "function"},
                    "function": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "arguments": {
                                "type": "object",
                                "additionalProperties": {"type": "any"},
                            },
                        }
                    }
                 }
            }
        },
    },
}

gpt_oss_schema = {
    # TODO Doesn't recover thinking blocks yet
    "type": "object",
    "properties": {
        "tools": {
            "type": "array",
            "x-regex": "\n\nnamespace functions \\{\n\n(.*?)\\} \\/\\/ namespace functions<\\|end\\|>",
            "x-regex-iterator": r"\/\/ .*?type \w+ = \(_: \{\n.*?\n\}\) \=\> any;",
            "items": {
                "type": "object",
                "properties": {
                    "type": {"const": "function"},
                    "function": {
                        "type": "object",
                        "x-regex": r"\/\/ (?P<description>.*?)\ntype (?P<name>\w+) = \(_: \{\n(?P<parameters>.*?)\n\}\) \=\> any;",
                        "properties": {
                            "name": {"type": "string"},
                            "description": {"type": "string"},
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "type": {"const": "object"},
                                    "properties": {
                                        "x-regex-to-dict": r"(?P<value>\/\/ .*?\n(?P<key>\w+)\??: \w+,)",
                                        "type": "object",
                                        "additionalProperties": {
                                            "type": "object",
                                            "properties": {
                                                "type": {"type": "string", "x-regex": r": (.*?),$"},
                                                "description": {"type": "string", "x-regex": r"^\/\/ (.*?)\n"},
                                            },
                                        },
                                    },
                                    "required": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                        "x-regex-iterator": r"\n([^?\n]+?):",
                                    },
                                },
                            },
                        },
                    },
                },
            },
        },
        "messages": {
            # TODO The structure is very hard to parse because a message with tool calls completely upends the format.
            #      I need to figure out all the possibilities before we can actually write a parser. Create a bunch
            #      of sample chats including non-tool-calling but with thinking etc.
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "role": {
                        "type": "string",
                        "enum": ["user", "assistant", "system", "tool"],
                        "x-mapping": {"developer": "system", "user": "user", "assistant": "assistant"},
                        "x-mapping-regex": {r"^functions\.": "tool"}
                    },
                    "content": {
                        "type": "string",
                        "x-regex": "^(?:\\# Instructions\n\n)?(.*?)(?:\n\n# Tools\n\n.*?)?$",
                    },
                    "thinking": {
                        "type": "string",
                    },
                    "tool_calls": {
                        "type": "array",
                        "x-regex-iterator": r"to=functions.\w+<.*?<|message|>.*?[<$]",
                        "prefixItems": [
                            {
                                "type": {"const": "function"},
                                "function": {
                                    "type": "object",
                                    "x-regex": r"to=functions(?P<name>.\w+)<.*?<|message|>(?P<arguments>.*?)[<$]",
                                    "properties": {
                                        "name": {"type": "string"},
                                        "arguments": {
                                            "type": "object",
                                            "additionalProperties": {"type": "string"}, # Type any?
                                        },
                                    }
                                }
                             }
                        ]
                    },
                },
                "required": ["role", "content"],
            },
            "x-regex": r"<\|start\|>system<\|message\|>.*?<\|end\|>(.*?)$",
            "x-regex-iterator": r"<\|start\|>(?P<role>.*?)(?:<\|channel\|>.*?)?<\|message\|>(?P<content>.*?)(?:<\|end\|>|<\|return\|>|<\|call\|>)",
        },
    },
}

class ChatSchemaParserTest(unittest.TestCase):
    def setUp(self):
        # This tokenizer has no chat template by default
        self.tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-gpt2")

    def test_cohere_template(self):
        model_out = '<|START_THINKING|>I should call a tool.<|END_THINKING|><|START_ACTION|>[\n    {"tool_call_id": "0", "tool_name": "simple_tool", "parameters": {"temperature_format": "Celsius"}}\n]<|END_ACTION|><|END_OF_TURN_TOKEN|>'
        parsed_chat = recursive_parse(model_out, cohere_schema)
        self.assertEqual(parsed_chat,{'role': 'assistant', 'thinking': 'I should call a tool.', 'tool_calls': [{'type': 'function', 'function': {'name': 'simple_tool', 'arguments': {'temperature_format': 'Celsius'}}}]})

    def test_ernie_template_with_tools(self):
        model_out = 'The user is asking about the weather in Paris today. Let me check the available tools. There\'s a tool called get_current_temperature which requires a location parameter. Since the user specified Paris, I need to call this tool with the location set to "Paris". I should make sure the argument is correctly formatted as a string. No other tools are available, so this is the right one to use. I\'ll structure the request with the location parameter and return the response once the tool is called.\n</think>\n\n<tool_call>\n{"name": "get_current_temperature", "arguments": {"location": "Paris"}}\n</tool_call>\n</s>'
        parsed_chat = recursive_parse(model_out, ernie_schema)
        self.assertEqual(parsed_chat, {'role': 'assistant', 'thinking': 'The user is asking about the weather in Paris today. Let me check the available tools. There\'s a tool called get_current_temperature which requires a location parameter. Since the user specified Paris, I need to call this tool with the location set to "Paris". I should make sure the argument is correctly formatted as a string. No other tools are available, so this is the right one to use. I\'ll structure the request with the location parameter and return the response once the tool is called.', 'tool_calls': [{'type': 'function', 'function': {'name': 'get_current_temperature', 'arguments': {'location': 'Paris'}}}]})

    def test_ernie_template_no_tools(self):
        model_out = 'The user just greeted me with "Hi! How are you?" I need to respond in a friendly and helpful manner. Let me start by acknowledging their greeting. I should ask them how they\'re doing to engage in conversation.\n\nFirst, I\'ll say hello back and then ask how they\'re feeling. It\'s important to show genuine interest. Maybe mention that I\'m here to help with anything they need. Keep the tone warm and positive. Let me make sure the response is concise but friendly. Alright, that should work.\n</think>\n\n<response>\nHello! I\'m doing well, thank you for asking. How about you? Is there something specific you\'d like help with today? I\'m here to assist you with any questions or problems you have!\n</response>\n</s>'
        parsed_chat = recursive_parse(model_out, ernie_schema)
        self.assertEqual(parsed_chat, {'role': 'assistant', 'content': "Hello! I'm doing well, thank you for asking. How about you? Is there something specific you'd like help with today? I'm here to assist you with any questions or problems you have!", 'thinking': 'The user just greeted me with "Hi! How are you?" I need to respond in a friendly and helpful manner. Let me start by acknowledging their greeting. I should ask them how they\'re doing to engage in conversation.\n\nFirst, I\'ll say hello back and then ask how they\'re feeling. It\'s important to show genuine interest. Maybe mention that I\'m here to help with anything they need. Keep the tone warm and positive. Let me make sure the response is concise but friendly. Alright, that should work.'})

    def test_gpt_oss_template(self):
        def simple_tool(temperature_format: str):
            """
            Test function

            Args:
                temperature_format: The temperature format to use
            """
            return -40.0

        def tool_with_everything_all_at_once(x_1: str, y_2: int, z_3: float = 43.0):
            """
            Test function with multiple args, and docstring args that we have to strip out.

            Args:
                x_1: The first input. It's got a big multiline
                   description and also contains

                y_2: The second input. It's a big list with a single-line description.

                z_3: The third input. It's some kind of tuple with a default arg.
            """
            return -40.0

        tokenizer = AutoTokenizer.from_pretrained("openai/gpt-oss-20b")
        chat = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi there! How can I help you today?"},
        ]
        formatted_chat = tokenizer.apply_chat_template(
            chat, tokenize=False, tools=[simple_tool, tool_with_everything_all_at_once], reasoning_effort="high"
        )
        parsed_chat = recursive_parse(formatted_chat, gpt_oss_schema)
        self.assertEqual(parsed_chat["messages"], chat)
        self.assertEqual(parsed_chat["tools"][0], get_json_schema(simple_tool))
        complex_schema = get_json_schema(tool_with_everything_all_at_once)
        complex_schema["function"]["parameters"]["properties"]["y_2"]["type"] = (
            "number"  # The GPT template maps these all to 'number' so we can't recover int vs float
        )
        self.assertEqual(parsed_chat["tools"][1], complex_schema)

    def test_gpt_oss_template_with_tool_calls(self):
        def get_current_temperature(location: str):
            """
            Gets the temperature at a given location.

            Args:
                location: The location to get the temperature for
            """
            return 22.0  # bug: Sometimes the temperature is not 22. low priority

        tokenizer = AutoTokenizer.from_pretrained("openai/gpt-oss-20b")
        chat = [
            {"role": "system", "content": "You are a helpful assistant who responds to queries by calling tools."},
            {"role": "user", "content": "Hey, what's the weather in Paris today?"},
            {
                "role": "assistant",
                "content": "We need to respond to the user by calling the get_current_temperature function with location \"Paris\". Provide a short response.",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "name": "get_current_temperature",
                            "arguments": {"location": "Paris"}
                        }
                    }
                ]
            },
            # {"role": "tool", "content": "22.0"},
            # {"role": "assistant", "content": "The current temperature in Paris is 22.0 degrees."},

        ]
        formatted_chat = tokenizer.apply_chat_template(
            chat, tokenize=False, tools=[get_current_temperature]
        )
        parsed_chat = recursive_parse(formatted_chat, gpt_oss_schema)
        self.assertEqual(parsed_chat["messages"], chat)