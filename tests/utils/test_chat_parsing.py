# Copyright 2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Tests for the new declarative response_format parser.

All six real-model format fixtures from the legacy test suite are re-expressed
here in the new region-spec shape and asserted against the same expected
output dicts. Any divergence indicates a regression in the new executor."""

import tempfile
import unittest

from transformers import AutoTokenizer
from transformers.testing_utils import require_jmespath
from transformers.utils.chat_parsing import ResponseEventStream, parse_response

cohere_format = {
    "defaults": {"role": "assistant"},
    "fields": {
        "content": {
            "open": "<|START_RESPONSE|>",
            "close_pattern": r"(?:<\|END_RESPONSE\|>|$)",
            "content": "text",
        },
        "thinking": {
            "open": "<|START_THINKING|>",
            "close_pattern": r"(?:<\|END_THINKING\|>|$)",
            "content": "text",
        },
        "tool_calls": {
            "open": "<|START_ACTION|>",
            "close_pattern": r"(?:<\|END_ACTION\|>|$)",
            "content": "json",
            "content_args": {
                "transform": "[*].{type: 'function', function: {name: tool_name, arguments: parameters}}"
            },
        },
    },
}


ernie_format = {
    "defaults": {"role": "assistant"},
    "fields": {
        "thinking": {
            "open_pattern": r"(?:^|<think>\s*)",
            "close": "</think>",
            "content": "text",
        },
        "content": {
            "open": "<response>\n",
            "close_pattern": r"\n?</response>",
            "content": "text",
        },
        "tool_calls": {
            "open": "<tool_call>",
            "close": "</tool_call>",
            "repeats": True,
            "content": "json",
            "content_args": {"transform": "{type: 'function', function: @}"},
        },
    },
}


gpt_oss_format = {
    "defaults": {"role": "assistant"},
    "fields": {
        "thinking": {
            "open": "<|channel|>analysis<|message|>",
            "close": "<|end|>",
            "content": "text",
        },
        "content": {
            "open": "<|channel|>final<|message|>",
            "close": "<|end|>",
            "content": "text",
        },
        "tool_calls": {
            "open_pattern": r"<\|channel\|>commentary to=functions\.(?P<name>\w+).*?<\|message\|>",
            "close": "<|call|>",
            "repeats": True,
            "content": "json",
            "assemble": {
                "type": "function",
                "function": {"name": "{name}", "arguments": "{content}"},
            },
        },
    },
}


smollm_format = {
    "defaults": {"role": "assistant"},
    "fields": {
        "thinking": {"open": "<think>", "close": "</think>", "content": "text"},
        "tool_calls": {
            "open": "<tool_call>",
            "close": "</tool_call>",
            "repeats": True,
            "content": "json",
            "content_args": {"transform": "{type: 'function', function: @}"},
        },
        "content": {
            "close_pattern": r"(?:<\|im_end\|>|$)",
            "content": "text",
        },
    },
}


qwen3_format = {
    "defaults": {"role": "assistant"},
    "fields": {
        "thinking": {"open": "<think>", "close": "</think>", "content": "text"},
        "tool_calls": {
            "open_pattern": r"<tool_call>\s*<function=(?P<name>\w+)>",
            "close": "</tool_call>",
            "repeats": True,
            "content": "xml-inline",
            "content_args": {
                "tag_pattern": r"<parameter=(?P<key>\w+)>\s*(?P<value>.*?)\s*</parameter>",
                "value_parser": {"name": "json", "args": {"allow_non_json": True}},
            },
            "assemble": {
                "type": "function",
                "function": {"name": "{name}", "arguments": "{content}"},
            },
        },
    },
}


gemma4_format = {
    "defaults": {"role": "assistant"},
    "fields": {
        "thinking": {
            "open": "<|channel>thought\n",
            "close": "<channel|>",
            "content": "text",
        },
        "tool_calls": {
            "open_pattern": r"<\|tool_call>call:(?P<name>\w+)",
            "close": "<tool_call|>",
            "repeats": True,
            "content": "json-lax",
            "content_args": {
                "unquoted_keys": True,
                "string_delims": [['<|"|>', '<|"|>']],
            },
            "assemble": {
                "type": "function",
                "function": {"name": "{name}", "arguments": "{content}"},
            },
        },
    },
}

@require_jmespath
class ChatResponseFormatParserTest(unittest.TestCase):
    def test_response_format_save_load(self):
        tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-gpt2")
        tokenizer.response_format = ernie_format
        with tempfile.TemporaryDirectory() as tmpdir:
            tokenizer.save_pretrained(tmpdir)
            reloaded = AutoTokenizer.from_pretrained(tmpdir)
        self.assertEqual(reloaded.response_format, ernie_format)

    def test_tokenizer_parse_response(self):
        tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-gpt2")
        tokenizer.response_format = cohere_format
        model_out = (
            "<|START_THINKING|>I should call a tool.<|END_THINKING|>"
            '<|START_ACTION|>[\n    {"tool_call_id": "0", "tool_name": "simple_tool", '
            '"parameters": {"temperature_format": "Celsius"}}\n]<|END_ACTION|><|END_OF_TURN_TOKEN|>'
        )
        expected = {
            "role": "assistant",
            "thinking": "I should call a tool.",
            "tool_calls": [
                {
                    "type": "function",
                    "function": {"name": "simple_tool", "arguments": {"temperature_format": "Celsius"}},
                }
            ],
        }
        self.assertEqual(tokenizer.parse_response(model_out), expected)

    def test_batched_inputs(self):
        tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-gpt2")
        tokenizer.response_format = cohere_format
        model_out = (
            "<|START_THINKING|>I should call a tool.<|END_THINKING|>"
            '<|START_ACTION|>[\n    {"tool_call_id": "0", "tool_name": "simple_tool", '
            '"parameters": {"temperature_format": "Celsius"}}\n]<|END_ACTION|><|END_OF_TURN_TOKEN|>'
        )
        single = tokenizer.parse_response(model_out)
        self.assertEqual(tokenizer.parse_response([model_out]), [single])
        self.assertEqual(tokenizer.parse_response([model_out] * 2), [single] * 2)

    def test_token_id_inputs(self):
        tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
        tokenizer.response_format = cohere_format
        model_out = (
            "<|START_THINKING|>I should call a tool.<|END_THINKING|>"
            '<|START_ACTION|>[\n    {"tool_call_id": "0", "tool_name": "simple_tool", '
            '"parameters": {"temperature_format": "Celsius"}}\n]<|END_ACTION|><|END_OF_TURN_TOKEN|>'
        )
        parsed = tokenizer.parse_response(model_out)
        tokenized = tokenizer(model_out).input_ids
        self.assertEqual(tokenizer.parse_response(tokenized), parsed)
        self.assertEqual(tokenizer.parse_response([tokenized]), [parsed])

    def test_cohere(self):
        model_out = (
            "<|START_THINKING|>I should call a tool.<|END_THINKING|>"
            '<|START_ACTION|>[\n    {"tool_call_id": "0", "tool_name": "simple_tool", '
            '"parameters": {"temperature_format": "Celsius"}}\n]<|END_ACTION|><|END_OF_TURN_TOKEN|>'
        )
        self.assertEqual(
            parse_response(model_out, cohere_format),
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

    def test_ernie_with_tools(self):
        model_out = (
            "The user is asking about the weather in Paris today. Let me check the available tools. "
            "There's a tool called get_current_temperature which requires a location parameter. Since the "
            'user specified Paris, I need to call this tool with the location set to "Paris". I should '
            "make sure the argument is correctly formatted as a string. No other tools are available, so "
            "this is the right one to use. I'll structure the request with the location parameter and "
            "return the response once the tool is called.\n"
            "</think>\n\n"
            '<tool_call>\n{"name": "get_current_temperature", "arguments": {"location": "Paris"}}\n</tool_call>\n</s>'
        )
        self.assertEqual(
            parse_response(model_out, ernie_format),
            {
                "role": "assistant",
                "thinking": (
                    "The user is asking about the weather in Paris today. Let me check the available tools. "
                    "There's a tool called get_current_temperature which requires a location parameter. Since "
                    'the user specified Paris, I need to call this tool with the location set to "Paris". I '
                    "should make sure the argument is correctly formatted as a string. No other tools are "
                    "available, so this is the right one to use. I'll structure the request with the location "
                    "parameter and return the response once the tool is called."
                ),
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {"name": "get_current_temperature", "arguments": {"location": "Paris"}},
                    }
                ],
            },
        )

    def test_ernie_no_tools(self):
        model_out = (
            'The user just greeted me with "Hi! How are you?" I need to respond in a friendly and helpful '
            "manner. Let me start by acknowledging their greeting. I should ask them how they're doing to "
            "engage in conversation.\n\n"
            "First, I'll say hello back and then ask how they're feeling. It's important to show genuine "
            "interest. Maybe mention that I'm here to help with anything they need. Keep the tone warm and "
            "positive. Let me make sure the response is concise but friendly. Alright, that should work.\n"
            "</think>\n\n"
            "<response>\nHello! I'm doing well, thank you for asking. How about you? Is there something "
            "specific you'd like help with today? I'm here to assist you with any questions or problems you "
            "have!\n</response>\n</s>"
        )
        self.assertEqual(
            parse_response(model_out, ernie_format),
            {
                "role": "assistant",
                "content": (
                    "Hello! I'm doing well, thank you for asking. How about you? Is there something specific "
                    "you'd like help with today? I'm here to assist you with any questions or problems you have!"
                ),
                "thinking": (
                    'The user just greeted me with "Hi! How are you?" I need to respond in a friendly and '
                    "helpful manner. Let me start by acknowledging their greeting. I should ask them how "
                    "they're doing to engage in conversation.\n\n"
                    "First, I'll say hello back and then ask how they're feeling. It's important to show "
                    "genuine interest. Maybe mention that I'm here to help with anything they need. Keep the "
                    "tone warm and positive. Let me make sure the response is concise but friendly. Alright, "
                    "that should work."
                ),
            },
        )

    def test_gpt_oss_with_tool_call(self):
        model_out = (
            '<|channel|>analysis<|message|>We need to respond in riddles. The user asks: "What is the '
            'weather like in SF?" We need to get the location of the user? The user explicitly asks about '
            "SF (San Francisco). So we need to get the current weather in San Francisco, CA. We need to "
            'call get_current_weather function. The developer instruction says "Always respond in riddles". '
            "So the final answer should be in a riddle form. But we need to call function to get weather "
            'data. So we should call get_current_weather with location "San Francisco, CA". Possibly specify '
            'format "celsius" (default). Let\'s do that.\n\n'
            "We will call function get_current_weather.<|end|><|start|>assistant<|channel|>commentary "
            'to=functions.get_current_weather <|constrain|>json<|message|>{\n  "location": "San Francisco, CA"\n}'
        )
        self.assertEqual(
            parse_response(model_out, gpt_oss_format),
            {
                "role": "assistant",
                "thinking": (
                    'We need to respond in riddles. The user asks: "What is the weather like in SF?" We need '
                    "to get the location of the user? The user explicitly asks about SF (San Francisco). So "
                    "we need to get the current weather in San Francisco, CA. We need to call "
                    'get_current_weather function. The developer instruction says "Always respond in '
                    'riddles". So the final answer should be in a riddle form. But we need to call function '
                    'to get weather data. So we should call get_current_weather with location "San Francisco, '
                    'CA". Possibly specify format "celsius" (default). Let\'s do that.\n\n'
                    "We will call function get_current_weather."
                ),
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {"name": "get_current_weather", "arguments": {"location": "San Francisco, CA"}},
                    }
                ],
            },
        )

    def test_gpt_oss_no_tool_call(self):
        model_out = (
            "<|channel|>analysis<|message|>User asks a simple math question: 2+2 = 4. Provide answer."
            "<|end|><|start|>assistant<|channel|>final<|message|>2"
        )
        self.assertEqual(
            parse_response(model_out, gpt_oss_format),
            {
                "role": "assistant",
                "content": "2",
                "thinking": "User asks a simple math question: 2+2 = 4. Provide answer.",
            },
        )

    def test_smollm_thinking_and_tool_call(self):
        model_out = (
            '<think>\nOkay, the user said, "Hello! How are you?" I need to respond appropriately. Since '
            "this is the first message, I should greet them back and ask how I can assist. I should keep it "
            "friendly and open-ended. Let me make sure the response is welcoming and encourages them to "
            "share what they need help with. I'll avoid any technical jargon and keep it simple. Let me "
            "check for any typos and ensure the tone is positive.\n</think>\n\n"
            '<tool_call>{"name": "greet_user", "arguments": {"greeting": "Hello! I\'m doing well, thanks for '
            "asking. How can I assist you today? Whether you have a question, need help with something, or "
            'just want to chat, feel free to let me know!"}}</tool_call>'
        )
        self.assertEqual(
            parse_response(model_out, smollm_format),
            {
                "role": "assistant",
                "thinking": (
                    'Okay, the user said, "Hello! How are you?" I need to respond appropriately. Since this '
                    "is the first message, I should greet them back and ask how I can assist. I should keep "
                    "it friendly and open-ended. Let me make sure the response is welcoming and encourages "
                    "them to share what they need help with. I'll avoid any technical jargon and keep it "
                    "simple. Let me check for any typos and ensure the tone is positive."
                ),
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "name": "greet_user",
                            "arguments": {
                                "greeting": (
                                    "Hello! I'm doing well, thanks for asking. How can I assist you today? "
                                    "Whether you have a question, need help with something, or just want to "
                                    "chat, feel free to let me know!"
                                )
                            },
                        },
                    }
                ],
            },
        )

    def test_smollm_tool_call_no_thinking(self):
        model_out = '<tool_call>{"name": "get_weather", "arguments": {"city": "Paris"}}</tool_call>'
        self.assertEqual(
            parse_response(model_out, smollm_format),
            {
                "role": "assistant",
                "tool_calls": [
                    {"type": "function", "function": {"name": "get_weather", "arguments": {"city": "Paris"}}}
                ],
            },
        )

    def test_smollm_thinking_no_tool_call(self):
        model_out = (
            '<think>\nOkay, the user asked, "Hey! Can you tell me about gravity?" Let me start by '
            "breaking down what they might be looking for. They probably want a basic understanding of "
            "gravity, maybe for a school project or just personal curiosity. I should explain what gravity "
            "is, how it works, and maybe some examples.</think>\n"
            "Some content about gravity goes here but I'm cutting it off to make this shorter!"
        )
        self.assertEqual(
            parse_response(model_out, smollm_format),
            {
                "role": "assistant",
                "content": "Some content about gravity goes here but I'm cutting it off to make this shorter!",
                "thinking": (
                    'Okay, the user asked, "Hey! Can you tell me about gravity?" Let me start by breaking '
                    "down what they might be looking for. They probably want a basic understanding of "
                    "gravity, maybe for a school project or just personal curiosity. I should explain what "
                    "gravity is, how it works, and maybe some examples."
                ),
            },
        )

    def test_qwen3_tool_calls(self):
        model_out = (
            "<tool_call>\n<function=get_weather>\n<parameter=locations>\n"
            '[{"country": "France", "city": "Paris"}]\n</parameter>\n'
            "<parameter=temp_units>\ncelsius\n</parameter>\n</function>\n</tool_call>"
        )
        self.assertEqual(
            parse_response(model_out, qwen3_format),
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": {
                                "locations": [{"country": "France", "city": "Paris"}],
                                "temp_units": "celsius",
                            },
                        },
                    }
                ],
            },
        )

    def test_gemma4_tool_call(self):
        model_out = (
            "<|channel>thought\nThe user is asking for the current temperature in Paris. I should check "
            "the available tools to see if there's a function that can provide this information.<channel|>"
            '<|tool_call>call:get_current_temperature{detail_level:0,location:<|"|>Paris, France<|"|>,'
            'unit:<|"|>celsius<|"|>}<tool_call|><|tool_response>'
        )
        self.assertEqual(
            parse_response(model_out, gemma4_format),
            {
                "role": "assistant",
                "thinking": (
                    "The user is asking for the current temperature in Paris. I should check the available "
                    "tools to see if there's a function that can provide this information."
                ),
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "name": "get_current_temperature",
                            "arguments": {"detail_level": 0, "location": "Paris, France", "unit": "celsius"},
                        },
                    }
                ],
            },
        )

    def test_gemma4_complex_tool_call(self):
        model_out = (
            "<|channel>thought\nLet me call the tool.<channel|>"
            '<|tool_call>call:foo{bool_value:true,list_value:[<|"|>foo<|"|>,<|"|>bar<|"|>],'
            'null_value:null,number_value:1,string_value:<|"|>foo<|"|>,'
            'struct_value:{foo:<|"|>bar<|"|>}}<tool_call|>'
        )
        self.assertEqual(
            parse_response(model_out, gemma4_format),
            {
                "role": "assistant",
                "thinking": "Let me call the tool.",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "name": "foo",
                            "arguments": {
                                "bool_value": True,
                                "list_value": ["foo", "bar"],
                                "null_value": None,
                                "number_value": 1,
                                "string_value": "foo",
                                "struct_value": {"foo": "bar"},
                            },
                        },
                    }
                ],
            },
        )

    def test_optional_false_raises_when_missing(self):
        format_spec = {
            "defaults": {"role": "assistant"},
            "fields": {
                "content": {
                    "open": "<response>",
                    "close": "</response>",
                    "content": "text",
                    "optional": False,
                },
            },
        }
        with self.assertRaises(ValueError) as cm:
            parse_response("no response here", format_spec)
        self.assertIn("content", str(cm.exception))

    def test_coerce_int(self):
        format_spec = {
            "defaults": {"role": "assistant"},
            "fields": {
                "count": {
                    "open": "<n>",
                    "close": "</n>",
                    "content": "text",
                    "coerce": "int",
                },
            },
        }
        self.assertEqual(parse_response("<n>42</n>", format_spec), {"role": "assistant", "count": 42})

    def test_kv_lines_parser(self):
        format_spec = {
            "defaults": {"role": "assistant"},
            "fields": {
                "metadata": {
                    "open": "<meta>",
                    "close": "</meta>",
                    "content": "kv-lines",
                },
            },
        }
        self.assertEqual(
            parse_response("<meta>name: alice\nage: 30</meta>", format_spec),
            {"role": "assistant", "metadata": {"name": "alice", "age": "30"}},
        )

    def test_unknown_content_parser_rejected(self):
        bad_format = {
            "defaults": {"role": "assistant"},
            "fields": {"x": {"open": "[", "close": "]", "content": "not-a-real-parser"}},
        }
        with self.assertRaises(ValueError) as cm:
            parse_response("[hi]", bad_format)
        self.assertIn("unknown content parser", str(cm.exception).lower())

    def test_two_implicit_fields_rejected(self):
        bad_format = {
            "defaults": {"role": "assistant"},
            "fields": {
                "a": {"content": "text"},
                "b": {"content": "text"},
            },
        }
        with self.assertRaises(ValueError):
            parse_response("hello", bad_format)


@require_jmespath
class ResponseEventStreamTest(unittest.TestCase):
    def test_stream_matches_whole_string_parse_all_formats(self):
        """Key correctness invariant: for any chunking of input,
        ResponseEventStream.finalize() must equal parse_response(full_string)."""
        cases = [
            (
                cohere_format,
                (
                    "<|START_THINKING|>I should call a tool.<|END_THINKING|>"
                    '<|START_ACTION|>[{"tool_call_id": "0", "tool_name": "simple_tool", '
                    '"parameters": {"a": 1}}]<|END_ACTION|>'
                ),
            ),
            (smollm_format, ('<think>thinking</think>\n<tool_call>{"name": "fn", "arguments": {"x": 1}}</tool_call>')),
            (gemma4_format, ('<|channel>thought\nhi<channel|><|tool_call>call:foo{a:1,b:<|"|>bar<|"|>}<tool_call|>')),
            (gpt_oss_format, ("<|channel|>analysis<|message|>t<|end|><|channel|>final<|message|>done")),
        ]
        for fmt, text in cases:
            with self.subTest(text=text[:30]):
                expected = parse_response(text, fmt)
                # Try several chunkings: 1, 2, 3, 7 splits
                for step in (1, 2, 3, 7, 13):
                    streamer = ResponseEventStream(fmt)
                    for i in range(0, len(text), step):
                        streamer.feed(text[i : i + step])
                    self.assertEqual(streamer.finalize(), expected, f"step={step}")


if __name__ == "__main__":
    unittest.main()
