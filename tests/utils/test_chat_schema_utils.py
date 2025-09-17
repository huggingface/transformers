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

import tempfile
import unittest

from transformers import AutoTokenizer
from transformers.utils.chat_parsing_utils import recursive_parse


cohere_schema = {
    "type": "object",
    "properties": {
        "role": {"const": "assistant"},
        "content": {"type": "string", "x-regex": r"<\|START_RESPONSE\|>(.*?)(?:<\|END_RESPONSE\|>|$)"},
        "thinking": {"type": "string", "x-regex": r"<\|START_THINKING\|>(.*?)(?:<\|END_THINKING\|>|$)"},
        "tool_calls": {
            "x-regex": r"<\|START_ACTION\|>(.*?)(?:<\|END_ACTION\|>|$)",
            "x-parser": "json",
            "x-parser-args": {
                "transform": "[*].{type: 'function', function: {name: tool_name, arguments: parameters}}"
            },
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
                        },
                    },
                },
            },
        },
    },
}

ernie_schema = {
    "type": "object",
    "properties": {
        "role": {"const": "assistant"},
        "content": {"type": "string", "x-regex": "<response>\n(.*?)\n?</response>"},
        "thinking": {"type": "string", "x-regex": r"(?:^|<think>\s*)(.*?)\s*<\/think>"},
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
                        },
                    },
                },
            },
        },
    },
}

gpt_oss_schema = {
    "type": "object",
    "properties": {
        "role": {"const": "assistant"},
        "content": {"type": "string", "x-regex": r"<\|channel\|>final<\|message\|>(.*?)(?:<\|end\|>|$)"},
        "thinking": {"type": "string", "x-regex": r"<\|channel\|>analysis<\|message\|>(.*?)<\|end\|>"},
        "tool_calls": {
            "x-regex-iterator": r"<\|channel\|>commentary (to=functions\..*?<\|message\|>.*?)(?:<\|call\|>|$)",
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "type": {"const": "function"},
                    "function": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string", "x-regex": r"^to=functions\.(\w+)"},
                            "arguments": {
                                "type": "object",
                                "x-regex": r"<\|message\|>(.*)",
                                "x-parser": "json",
                                "additionalProperties": {"type": "any"},
                            },
                        },
                    },
                },
            },
        },
    },
}


class ChatSchemaParserTest(unittest.TestCase):
    def test_schema_save_load(self):
        tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-gpt2")
        tokenizer.response_schema = ernie_schema
        with tempfile.TemporaryDirectory() as tmpdir:
            tokenizer.save_pretrained(tmpdir)
            reloaded_tokenizer = AutoTokenizer.from_pretrained(tmpdir)
        self.assertEqual(reloaded_tokenizer.response_schema, ernie_schema)

    def test_tokenizer_method(self):
        tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-gpt2")
        model_out = '<|START_THINKING|>I should call a tool.<|END_THINKING|><|START_ACTION|>[\n    {"tool_call_id": "0", "tool_name": "simple_tool", "parameters": {"temperature_format": "Celsius"}}\n]<|END_ACTION|><|END_OF_TURN_TOKEN|>'
        parsed_chat = recursive_parse(model_out, cohere_schema)
        tokenizer.response_schema = cohere_schema
        tokenizer_parsed_chat = tokenizer.parse_response(model_out)
        self.assertEqual(tokenizer_parsed_chat, parsed_chat)

    def test_cohere_template(self):
        model_out = '<|START_THINKING|>I should call a tool.<|END_THINKING|><|START_ACTION|>[\n    {"tool_call_id": "0", "tool_name": "simple_tool", "parameters": {"temperature_format": "Celsius"}}\n]<|END_ACTION|><|END_OF_TURN_TOKEN|>'
        parsed_chat = recursive_parse(model_out, cohere_schema)
        self.assertEqual(
            parsed_chat,
            {
                "role": "assistant",
                "thinking": "I should call a tool.",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {"name": "simple_tool", "arguments": {"temperature_format": "Celsius"}},
                    }
                ],
            },
        )

    def test_ernie_template_with_tools(self):
        model_out = 'The user is asking about the weather in Paris today. Let me check the available tools. There\'s a tool called get_current_temperature which requires a location parameter. Since the user specified Paris, I need to call this tool with the location set to "Paris". I should make sure the argument is correctly formatted as a string. No other tools are available, so this is the right one to use. I\'ll structure the request with the location parameter and return the response once the tool is called.\n</think>\n\n<tool_call>\n{"name": "get_current_temperature", "arguments": {"location": "Paris"}}\n</tool_call>\n</s>'
        parsed_chat = recursive_parse(model_out, ernie_schema)
        self.assertEqual(
            parsed_chat,
            {
                "role": "assistant",
                "thinking": "The user is asking about the weather in Paris today. Let me check the available tools. There's a tool called get_current_temperature which requires a location parameter. Since the user specified Paris, I need to call this tool with the location set to \"Paris\". I should make sure the argument is correctly formatted as a string. No other tools are available, so this is the right one to use. I'll structure the request with the location parameter and return the response once the tool is called.",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {"name": "get_current_temperature", "arguments": {"location": "Paris"}},
                    }
                ],
            },
        )

    def test_ernie_template_no_tools(self):
        model_out = "The user just greeted me with \"Hi! How are you?\" I need to respond in a friendly and helpful manner. Let me start by acknowledging their greeting. I should ask them how they're doing to engage in conversation.\n\nFirst, I'll say hello back and then ask how they're feeling. It's important to show genuine interest. Maybe mention that I'm here to help with anything they need. Keep the tone warm and positive. Let me make sure the response is concise but friendly. Alright, that should work.\n</think>\n\n<response>\nHello! I'm doing well, thank you for asking. How about you? Is there something specific you'd like help with today? I'm here to assist you with any questions or problems you have!\n</response>\n</s>"
        parsed_chat = recursive_parse(model_out, ernie_schema)
        self.assertEqual(
            parsed_chat,
            {
                "role": "assistant",
                "content": "Hello! I'm doing well, thank you for asking. How about you? Is there something specific you'd like help with today? I'm here to assist you with any questions or problems you have!",
                "thinking": "The user just greeted me with \"Hi! How are you?\" I need to respond in a friendly and helpful manner. Let me start by acknowledging their greeting. I should ask them how they're doing to engage in conversation.\n\nFirst, I'll say hello back and then ask how they're feeling. It's important to show genuine interest. Maybe mention that I'm here to help with anything they need. Keep the tone warm and positive. Let me make sure the response is concise but friendly. Alright, that should work.",
            },
        )

    def test_gpt_oss_template_with_tool_call(self):
        model_out = '<|channel|>analysis<|message|>We need to respond in riddles. The user asks: "What is the weather like in SF?" We need to get the location of the user? The user explicitly asks about SF (San Francisco). So we need to get the current weather in San Francisco, CA. We need to call get_current_weather function. The developer instruction says "Always respond in riddles". So the final answer should be in a riddle form. But we need to call function to get weather data. So we should call get_current_weather with location "San Francisco, CA". Possibly specify format "celsius" (default). Let\'s do that.\n\nWe will call function get_current_weather.<|end|><|start|>assistant<|channel|>commentary to=functions.get_current_weather <|constrain|>json<|message|>{\n  "location": "San Francisco, CA"\n}'
        parsed_chat = recursive_parse(model_out, gpt_oss_schema)
        self.assertEqual(
            parsed_chat,
            {
                "role": "assistant",
                "thinking": 'We need to respond in riddles. The user asks: "What is the weather like in SF?" We need to get the location of the user? The user explicitly asks about SF (San Francisco). So we need to get the current weather in San Francisco, CA. We need to call get_current_weather function. The developer instruction says "Always respond in riddles". So the final answer should be in a riddle form. But we need to call function to get weather data. So we should call get_current_weather with location "San Francisco, CA". Possibly specify format "celsius" (default). Let\'s do that.\n\nWe will call function get_current_weather.',
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {"name": "get_current_weather", "arguments": {"location": "San Francisco, CA"}},
                    }
                ],
            },
        )

    def test_gpt_oss_template_no_tool_call(self):
        model_out = "<|channel|>analysis<|message|>User asks a simple math question: 2+2 = 4. Provide answer.<|end|><|start|>assistant<|channel|>final<|message|>2"
        parsed_chat = recursive_parse(model_out, gpt_oss_schema)
        self.assertEqual(
            parsed_chat,
            {
                "role": "assistant",
                "content": "2",
                "thinking": "User asks a simple math question: 2+2 = 4. Provide answer.",
            },
        )
