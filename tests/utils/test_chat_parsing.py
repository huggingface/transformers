# Copyright 2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Tests for the new declarative response_template parser.

All six real-model template fixtures from the legacy test suite are re-expressed
here in the new region-spec shape and asserted against the same expected
output dicts. Any divergence indicates a regression in the new executor."""

import random
import tempfile
import unittest

from transformers import AutoTokenizer
from transformers.testing_utils import require_jmespath
from transformers.utils.chat_parsing import ResponseParser, parse_response


cohere_template = {
    "defaults": {"role": "assistant"},
    "fields": {
        "content": {
            "open": "<|START_RESPONSE|>",
            "close": "<|END_RESPONSE|>",
            "content": "text",
        },
        "thinking": {
            "open": "<|START_THINKING|>",
            "close": "<|END_THINKING|>",
            "content": "text",
        },
        "tool_calls": {
            "open": "<|START_ACTION|>",
            "close": "<|END_ACTION|>",
            "content": "json",
            "content_args": {
                "transform": "[*].{type: 'function', function: {name: tool_name, arguments: parameters}}"
            },
        },
    },
}


ernie_template = {
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


gpt_oss_template = {
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


smollm_template = {
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
            "close": "<|im_end|>",
            "content": "text",
        },
    },
}


qwen3_template = {
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


gemma4_template = {
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
            "content": "json",
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
class ChatResponseTemplateParserTest(unittest.TestCase):
    def test_response_template_save_load(self):
        tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-gpt2")
        tokenizer.response_template = ernie_template
        with tempfile.TemporaryDirectory() as tmpdir:
            tokenizer.save_pretrained(tmpdir)
            reloaded = AutoTokenizer.from_pretrained(tmpdir)
        self.assertEqual(reloaded.response_template, ernie_template)

    def test_tokenizer_parse_response(self):
        tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-gpt2")
        tokenizer.response_template = cohere_template
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
        tokenizer.response_template = cohere_template
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
        tokenizer.response_template = cohere_template
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
            parse_response(model_out, cohere_template),
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
            parse_response(model_out, ernie_template),
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
            parse_response(model_out, ernie_template),
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
            parse_response(model_out, gpt_oss_template),
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
            parse_response(model_out, gpt_oss_template),
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
            parse_response(model_out, smollm_template),
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
            parse_response(model_out, smollm_template),
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
            parse_response(model_out, smollm_template),
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
            parse_response(model_out, qwen3_template),
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
            parse_response(model_out, gemma4_template),
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
            parse_response(model_out, gemma4_template),
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
        template_spec = {
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
            parse_response("no response here", template_spec)
        self.assertIn("content", str(cm.exception))

    def test_int_content_parser(self):
        template_spec = {
            "defaults": {"role": "assistant"},
            "fields": {
                "count": {
                    "open": "<n>",
                    "close": "</n>",
                    "content": "int",
                },
            },
        }
        self.assertEqual(parse_response("<n>42</n>", template_spec), {"role": "assistant", "count": 42})

    def test_kv_lines_parser(self):
        template_spec = {
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
            parse_response("<meta>name: alice\nage: 30</meta>", template_spec),
            {"role": "assistant", "metadata": {"name": "alice", "age": "30"}},
        )

    def test_unknown_content_parser_rejected(self):
        bad_template = {
            "defaults": {"role": "assistant"},
            "fields": {"x": {"open": "[", "close": "]", "content": "not-a-real-parser"}},
        }
        with self.assertRaises(ValueError) as cm:
            parse_response("[hi]", bad_template)
        self.assertIn("unknown content parser", str(cm.exception).lower())

    def test_two_implicit_fields_rejected(self):
        bad_template = {
            "defaults": {"role": "assistant"},
            "fields": {
                "a": {"content": "text"},
                "b": {"content": "text"},
            },
        }
        with self.assertRaises(ValueError):
            parse_response("hello", bad_template)

    def test_literal_list_open_and_close(self):
        """A list of literals matches any one of them, like an alternation."""
        template_spec = {
            "defaults": {"role": "assistant"},
            "fields": {
                "x": {
                    "open": ["<a>", "<bb>"],
                    "close": ["</a>", "</bb>"],
                    "content": "text",
                },
            },
        }
        for opener, closer in (("<a>", "</a>"), ("<bb>", "</bb>"), ("<a>", "</bb>")):
            self.assertEqual(
                parse_response(f"{opener}hi{closer}", template_spec),
                {"role": "assistant", "x": "hi"},
            )

    def test_literal_list_streams_without_64_byte_hold(self):
        """Compared to a regex close, a literal-list close lets the parser
        flush bytes that aren't in the longest-prefix overlap of any literal.
        With `["<turn|>", "<|tool_response>", "<eos>"]` (longest = 16 chars),
        feeding 32 plain bytes should leave at most 15 unflushed."""
        template_spec = {
            "defaults": {"role": "assistant"},
            "fields": {
                "content": {"close": ["<turn|>", "<|tool_response>", "<eos>"], "content": "text"},
            },
        }
        parser = ResponseParser(template_spec)
        plain = "x" * 32
        flushed: list[str] = []
        for ch in parser.feed(plain):
            if ch["type"] == "region_chunk":
                flushed.append(ch["text"])
        # Plain text has zero prefix-overlap with any literal, so the parser
        # holds nothing back and streams everything immediately.
        self.assertEqual("".join(flushed), plain)

    def test_literal_list_defers_prefix_overlapping_literal(self):
        """If a literal is a strict prefix of another in the same list, an
        edge match could still grow with more input — we must defer to be safe."""
        template_spec = {
            "defaults": {"role": "assistant"},
            "fields": {
                "x": {"open": "<x>", "close": ["END", "ENDX"], "content": "text"},
            },
        }
        parser = ResponseParser(template_spec)
        # "<x>hiEND" mid-stream: don't commit the close yet — "ENDX" might be coming.
        events = parser.feed("<x>hiEND")
        self.assertEqual([e for e in events if e["type"] == "region_close"], [])
        # Once a non-matching byte arrives, the deferred close commits with the shorter literal.
        events.extend(parser.feed(" more"))
        message, _ = parser.finalize()
        closes = [e for e in events if e["type"] == "region_close" and e["field"] == "x"]
        self.assertEqual(len(closes), 1)
        self.assertEqual(closes[0]["value"], "hi")
        self.assertEqual(message, {"role": "assistant", "x": "hi"})

    def test_literal_list_rejects_empty_and_non_string(self):
        for bad_open in ([], [""], [1, 2], {"foo": "bar"}):
            spec = {
                "defaults": {"role": "assistant"},
                "fields": {"x": {"open": bad_open, "close": "</x>", "content": "text"}},
            }
            with self.assertRaises(ValueError):
                parse_response("<x>hi</x>", spec)

    def test_literal_list_eos_must_stand_alone(self):
        """The magic `"eos"` literal maps to end-of-stream and is only valid on its own."""
        spec = {
            "defaults": {"role": "assistant"},
            "fields": {"content": {"close": ["eos", "<turn|>"], "content": "text"}},
        }
        with self.assertRaises(ValueError):
            parse_response("hi", spec)


# Fixtures shared by the streaming tests: one representative input per template,
# re-used for both the correctness invariant (any chunking → same dict) and the
# event-shape tests (do we emit the right events in the right order?).
_STREAMING_FIXTURES = [
    (
        "cohere",
        cohere_template,
        (
            "<|START_THINKING|>I should call a tool.<|END_THINKING|>"
            '<|START_ACTION|>[{"tool_call_id": "0", "tool_name": "simple_tool", '
            '"parameters": {"a": 1}}]<|END_ACTION|>'
        ),
    ),
    (
        "ernie",
        ernie_template,
        (
            "<think>some deliberation here</think>\n\n"
            '<tool_call>\n{"name": "get_current_temperature", "arguments": {"location": "Paris"}}\n</tool_call>\n</s>'
        ),
    ),
    (
        "gpt_oss",
        gpt_oss_template,
        "<|channel|>analysis<|message|>thinking chunk<|end|><|channel|>final<|message|>done text",
    ),
    (
        "smollm",
        smollm_template,
        '<think>thinking</think>\n<tool_call>{"name": "fn", "arguments": {"x": 1}}</tool_call>',
    ),
    (
        "qwen3",
        qwen3_template,
        (
            "<think>short thought</think>\n"
            "<tool_call>\n<function=get_weather>\n"
            "<parameter=city>\nParis\n</parameter>\n"
            "</function>\n</tool_call>"
        ),
    ),
    (
        "gemma4",
        gemma4_template,
        '<|channel>thought\nhi<channel|><|tool_call>call:foo{a:1,b:<|"|>bar<|"|>}<tool_call|>',
    ),
]


def _chunk_fixed(text: str, step: int):
    for i in range(0, len(text), step):
        yield text[i : i + step]


def _chunk_random(text: str, rng: random.Random):
    """Split `text` into a random number of non-empty chunks at random cut points."""
    if len(text) <= 1:
        yield text
        return
    num_cuts = rng.randint(0, len(text) - 1)
    cuts = sorted(rng.sample(range(1, len(text)), num_cuts))
    prev = 0
    for c in cuts:
        yield text[prev:c]
        prev = c
    yield text[prev:]


@require_jmespath
class ResponseEventStreamTest(unittest.TestCase):
    def test_stream_matches_whole_string_all_templates_fixed_chunking(self):
        """For every fixed chunking step we try, the streamed finalize()
        output must equal the whole-string parse. Regression coverage for
        specific edge-case byte boundaries (1-byte chunks hit every prefix)."""
        for name, tmpl, text in _STREAMING_FIXTURES:
            expected = parse_response(text, tmpl)
            for step in (1, 2, 3, 5, 7, 13, 31):
                with self.subTest(fixture=name, step=step):
                    streamer = ResponseParser(tmpl)
                    for chunk in _chunk_fixed(text, step):
                        streamer.feed(chunk)
                    message, _ = streamer.finalize()
                    self.assertEqual(message, expected)

    def test_stream_matches_whole_string_all_templates_random_chunking(self):
        """Property-style: for many random chunkings per fixture, the streamed
        finalize() output must equal the whole-string parse. Seeded so failures
        reproduce."""
        rng = random.Random(0xC0DE_5EED)
        for name, tmpl, text in _STREAMING_FIXTURES:
            expected = parse_response(text, tmpl)
            for trial in range(30):
                with self.subTest(fixture=name, trial=trial):
                    streamer = ResponseParser(tmpl)
                    for chunk in _chunk_random(text, rng):
                        streamer.feed(chunk)
                    message, _ = streamer.finalize()
                    self.assertEqual(message, expected)

    def test_events_well_formed_for_every_chunking(self):
        """Every event batch, across every fixture and every chunking, must be
        well-formed: region_open precedes its matching region_close for the
        same field; region_chunk only appears between open and close; no
        region is left open at the end of the stream; and the concatenation
        of all region_chunk payloads for a streamable text-like field equals
        the final value."""
        rng = random.Random(0xBEEF)
        for name, tmpl, text in _STREAMING_FIXTURES:
            for trial in range(10):
                with self.subTest(fixture=name, trial=trial):
                    streamer = ResponseParser(tmpl)
                    all_events: list[dict] = []
                    for chunk in _chunk_random(text, rng):
                        all_events.extend(streamer.feed(chunk))
                    _, final_events = streamer.finalize()
                    all_events.extend(final_events)
                    self._assert_event_stream_well_formed(all_events)

    def _assert_event_stream_well_formed(self, events: list[dict]) -> None:
        open_field: str | None = None
        chunk_accum: dict[str, str] = {}
        close_values: dict[str, object] = {}
        for ev in events:
            t = ev["type"]
            if t == "region_open":
                self.assertIsNone(open_field, f"nested region_open without close: {ev}")
                open_field = ev["field"]
                chunk_accum.setdefault(open_field, "")
            elif t == "region_chunk":
                self.assertEqual(open_field, ev["field"], f"chunk outside its region: {ev}")
                # Every chunk carries a boolean `dirty` flag.
                self.assertIsInstance(ev["dirty"], bool, f"missing/non-bool dirty: {ev}")
                chunk_accum[open_field] += ev["text"]
            elif t == "region_close":
                self.assertEqual(open_field, ev["field"], f"close for non-open region: {ev}")
                close_values[open_field] = ev["value"]
                open_field = None
            else:
                self.fail(f"unexpected event type: {ev!r}")
        self.assertIsNone(open_field, "region left open at end of stream")

    def test_region_chunks_reconstruct_text_regions(self):
        """For text-like regions (`dirty=False`), concatenating chunk texts
        reconstructs the final value reported in region_close. Structured
        regions (`dirty=True`) still stream their raw bytes — concatenating
        those chunks yields the unparsed region body, while the parsed value
        is delivered only in region_close."""
        # Single representative case with a long text region and a JSON region.
        text = _STREAMING_FIXTURES[0][2]  # cohere fixture
        streamer = ResponseParser(cohere_template)
        events: list[dict] = []
        for ch in text:  # 1-byte chunks hit the most anchor boundaries
            events.extend(streamer.feed(ch))
        _, final_events = streamer.finalize()
        events.extend(final_events)

        # Reconstruct per-field.
        per_field_chunks: dict[str, list[str]] = {}
        per_field_dirty: dict[str, set[bool]] = {}
        per_field_close_value: dict[str, object] = {}
        for ev in events:
            if ev["type"] == "region_chunk":
                per_field_chunks.setdefault(ev["field"], []).append(ev["text"])
                per_field_dirty.setdefault(ev["field"], set()).add(ev["dirty"])
            elif ev["type"] == "region_close":
                per_field_close_value[ev["field"]] = ev["value"]

        # `thinking` is text → chunks are clean and concatenate to its value.
        self.assertIn("thinking", per_field_chunks)
        self.assertEqual(per_field_dirty["thinking"], {False})
        self.assertEqual("".join(per_field_chunks["thinking"]), "I should call a tool.")
        self.assertEqual(per_field_close_value["thinking"], "I should call a tool.")
        # `tool_calls` is json → dirty chunks stream the raw body, parsed value on close.
        self.assertIn("tool_calls", per_field_chunks)
        self.assertEqual(per_field_dirty["tool_calls"], {True})
        self.assertEqual(
            "".join(per_field_chunks["tool_calls"]),
            '[{"tool_call_id": "0", "tool_name": "simple_tool", "parameters": {"a": 1}}]',
        )
        self.assertIn("tool_calls", per_field_close_value)

    def test_dirty_flag_marks_structured_regions(self):
        """A template with one text field and one structured field per parser
        family: text/int/float/bool stream chunks with `dirty=False`, while
        json/xml-inline/kv-lines stream chunks with `dirty=True`, and those
        dirty chunks concatenate to the raw region body before parsing."""
        spec = {
            "defaults": {"role": "assistant"},
            "fields": {
                "thinking": {"open": "<t>", "close": "</t>", "content": "text"},
                "score": {"open": "<n>", "close": "</n>", "content": "int"},
                "json_call": {"open": "<j>", "close": "</j>", "content": "json"},
                "xml_call": {
                    "open": "<x>",
                    "close": "</x>",
                    "content": "xml-inline",
                    "content_args": {"tag_pattern": r"<(?P<key>\w+)=(?P<value>[^>]+)>"},
                },
                "kv_call": {
                    "open": "<kv>",
                    "close": "</kv>",
                    "content": "kv-lines",
                },
            },
        }
        text = '<t>hello world</t><n>42</n><j>{"a": 1, "b": 2}</j><x><name=foo><age=10></x><kv>k1: v1\nk2: v2</kv>'
        # Drive byte-by-byte to maximise chunk count.
        streamer = ResponseParser(spec)
        events: list[dict] = []
        for ch in text:
            events.extend(streamer.feed(ch))
        _, final_events = streamer.finalize()
        events.extend(final_events)

        per_field_chunks: dict[str, list[str]] = {}
        per_field_dirty: dict[str, set[bool]] = {}
        for ev in events:
            if ev["type"] == "region_chunk":
                per_field_chunks.setdefault(ev["field"], []).append(ev["text"])
                per_field_dirty.setdefault(ev["field"], set()).add(ev["dirty"])

        # Clean (streamable) regions.
        for field in ("thinking", "score"):
            self.assertEqual(per_field_dirty[field], {False}, f"{field} should be clean")
        # Dirty (structured) regions.
        for field in ("json_call", "xml_call", "kv_call"):
            self.assertEqual(per_field_dirty[field], {True}, f"{field} should be dirty")

        # Dirty chunks reconstruct the raw region body (un-parsed). Clean
        # chunks reconstruct the verbatim body too — stripping happens at close.
        self.assertEqual("".join(per_field_chunks["thinking"]), "hello world")
        self.assertEqual("".join(per_field_chunks["score"]), "42")
        self.assertEqual("".join(per_field_chunks["json_call"]), '{"a": 1, "b": 2}')
        self.assertEqual("".join(per_field_chunks["xml_call"]), "<name=foo><age=10>")
        self.assertEqual("".join(per_field_chunks["kv_call"]), "k1: v1\nk2: v2")

    def test_feed_after_finalize_raises(self):
        streamer = ResponseParser(smollm_template)
        streamer.feed("<think>x</think>")
        streamer.finalize()
        with self.assertRaises(RuntimeError):
            streamer.feed("more")
        with self.assertRaises(RuntimeError):
            streamer.finalize()

    def test_empty_input_streams_cleanly(self):
        streamer = ResponseParser(smollm_template)
        self.assertEqual(streamer.feed(""), [])
        result, final_events = streamer.finalize()
        # Only the default fields should remain; nothing else is required.
        self.assertEqual(result, {"role": "assistant"})
        self.assertEqual(final_events, [])


# ChatML-style anchor used by Qwen / SmolLM templates: assistant header is
# everything from `<|im_start|>` through `assistant\n`. Right-truncating past
# this leaves only the message body, so prefix material (`<|im_start|>`,
# role label) doesn't pollute downstream parsing.
_CHATML_ANCHOR = "<|im_start|>assistant\n"

qwen3_template_with_anchor = {**qwen3_template, "start_anchor": _CHATML_ANCHOR}
smollm_template_with_anchor = {**smollm_template, "start_anchor": _CHATML_ANCHOR}


@require_jmespath
class PrefixAndTruncationTest(unittest.TestCase):
    def test_prefix_lands_inside_explicit_region(self):
        """A Qwen-style template emits `<|im_start|>assistant\\n<think>\\n` as the
        assistant prefix. The model continues from inside the thinking block."""
        prompt = (
            "<|im_start|>system\nYou are helpful<|im_end|>\n"
            "<|im_start|>user\nHi<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n"
        )
        generated = "Let me think...</think>"
        stream = ResponseParser(qwen3_template_with_anchor, prefix=prompt)
        # The region_open for `thinking` surfaces via initial_events; the
        # caller replays it before feeding model output.
        self.assertEqual(
            [(e["type"], e["field"]) for e in stream.initial_events],
            [("region_open", "thinking"), ("region_chunk", "thinking")],
        )
        events = stream.feed(generated)
        result, _ = stream.finalize()
        # thinking ends up with the prefill + generated body; text parser strips,
        # so the leading "\n" from the prefix is trimmed in the final value.
        self.assertEqual(result, {"role": "assistant", "thinking": "Let me think..."})
        # The feed stream only sees the rest of the body and the close; the
        # prefill already emitted region_open.
        self.assertEqual([e["type"] for e in events], ["region_chunk", "region_close"])
        self.assertEqual(events[1]["field"], "thinking")

    def test_prefix_truncated_to_last_anchor(self):
        """Multiple `<|im_start|>assistant\\n` anchors in the prefix (multi-turn
        conversation): only the slice after the LAST anchor matters."""
        prompt = (
            "<|im_start|>system\nA<|im_end|>\n"
            "<|im_start|>user\nB<|im_end|>\n"
            "<|im_start|>assistant\nEarlier reply<|im_end|>\n"
            "<|im_start|>user\nFollowup<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n"
        )
        stream = ResponseParser(qwen3_template_with_anchor, prefix=prompt)
        # We landed inside `thinking` (from the LAST assistant turn's `<think>\n`),
        # not in some earlier-turn artifact.
        opens = [e for e in stream.initial_events if e["type"] == "region_open"]
        self.assertEqual([e["field"] for e in opens], ["thinking"])
        # No earlier-turn content leaked into output.
        stream.feed("done</think>")
        stream.finalize()
        self.assertEqual(stream._output, {"role": "assistant", "thinking": "done"})

    def test_prefix_no_anchor_in_spec_keeps_full_prefix(self):
        """With no `start_anchor` in the spec, the prefix is processed verbatim.
        Useful when the caller has already pre-cut the prefix."""
        prompt = "<think>\n"
        stream = ResponseParser(qwen3_template, prefix=prompt)  # no anchor
        opens = [e for e in stream.initial_events if e["type"] == "region_open"]
        self.assertEqual([e["field"] for e in opens], ["thinking"])
        stream.feed("hi</think>")
        stream.finalize()
        self.assertEqual(stream._output, {"role": "assistant", "thinking": "hi"})

    def test_prefix_anchor_not_found_falls_back(self):
        """Spec has start_anchor but the prefix doesn't contain it: parser
        falls back to processing the entire prefix (with a logged warning)."""
        prompt = "<think>\n"  # no <|im_start|>assistant\n
        stream = ResponseParser(qwen3_template_with_anchor, prefix=prompt)
        opens = [e for e in stream.initial_events if e["type"] == "region_open"]
        self.assertEqual([e["field"] for e in opens], ["thinking"])
        stream.feed("hi</think>")
        stream.finalize()
        self.assertEqual(stream._output, {"role": "assistant", "thinking": "hi"})

    def test_round_trip_equivalence_prefix_vs_concat(self):
        """The load-bearing property: `prefix=p` + chunked `feed(g)` produces
        the same dict as `parse_response(p + g)` and as
        `parse_response(g, prefix=p)`. Auto-truncation makes "one long array"
        equivalent to the explicit two-input form."""
        prompt = "<|im_start|>system\nA<|im_end|>\n<|im_start|>user\nB<|im_end|>\n<|im_start|>assistant\n<think>\n"
        for name, tmpl_dict, gen_text in _STREAMING_FIXTURES:
            if "thinking" not in gen_text and "<think>" not in gen_text:
                continue
            # Only fixtures whose generation text is compatible with starting
            # inside `thinking`. Restrict to qwen3 / smollm shape for clarity.
            if name not in ("qwen3", "smollm"):
                continue
            tmpl_with_anchor = {**tmpl_dict, "start_anchor": _CHATML_ANCHOR}
            # Expected: feed (prefix + gen) without prefix kwarg, but auto-truncate
            # via parse_response should match feeding gen with prefix kwarg.
            via_prefix = parse_response(gen_text, tmpl_with_anchor, prefix=prompt)
            via_concat = parse_response(prompt + gen_text, tmpl_with_anchor)
            self.assertEqual(via_prefix, via_concat, msg=f"fixture={name}")
            # Streaming forms.
            for step in (1, 3, 7, 31):
                with self.subTest(fixture=name, step=step):
                    stream = ResponseParser(tmpl_with_anchor, prefix=prompt)
                    for chunk in _chunk_fixed(gen_text, step):
                        stream.feed(chunk)
                    message, _ = stream.finalize()
                    self.assertEqual(message, via_prefix)

    def test_prefix_with_open_close_inside_truncated_region(self):
        """Prefix opens AND closes a region. The full open/chunk/close event
        sequence is surfaced via initial_events, and the closed region lands
        in the output dict — so renderers can show prefill content."""
        spec = {
            "defaults": {"role": "assistant"},
            "start_anchor": "[BEGIN]",
            "fields": {
                "tag": {"open": "<tag>", "close": "</tag>", "content": "text"},
                "body": {"close_pattern": r"$", "content": "text"},  # implicit
            },
        }
        prefix = "noise[BEGIN]<tag>silently consumed</tag>"
        stream = ResponseParser(spec, prefix=prefix)
        types = [e["type"] for e in stream.initial_events]
        self.assertEqual(types, ["region_open", "region_chunk", "region_close"])
        self.assertTrue(all(e["field"] == "tag" for e in stream.initial_events))
        self.assertEqual(stream.initial_events[-1]["value"], "silently consumed")
        stream.feed("real generated body")
        result, _ = stream.finalize()
        self.assertEqual(result["tag"], "silently consumed")
        self.assertEqual(result["body"], "real generated body")

    def test_prefix_lands_inside_implicit_region(self):
        """Prefix wrote plaintext into the implicit region (e.g. assistant
        prefill before the model continues). The region_open for the implicit
        region must surface via initial_events so consumers don't miss it —
        `_opened` will already be True by the time feed runs."""
        prompt = "<|im_start|>assistant\nSure, here is "
        stream = ResponseParser(smollm_template_with_anchor, prefix=prompt)
        opens = [e for e in stream.initial_events if e["type"] == "region_open"]
        self.assertEqual([e["field"] for e in opens], ["content"])
        events = stream.feed("the answer<|im_end|>")
        # No second region_open from feed — the implicit region was already
        # opened during prefill and surfaced via initial_events.
        self.assertNotIn("region_open", [e["type"] for e in events])

    def test_prefix_partial_pattern_at_boundary(self):
        """Post-truncation prefix ends mid-delimiter. The first feed completes
        the match; initial_events is empty (no region opened within the
        prefix yet) and the open fires from `feed()`."""
        prefix = "<|im_start|>assistant\n<thi"  # incomplete `<think>`
        stream = ResponseParser(qwen3_template_with_anchor, prefix=prefix)
        self.assertEqual(stream.initial_events, [])
        events = stream.feed("nk>real body</think>")
        types = [e["type"] for e in events]
        self.assertIn("region_open", types)  # think opens during feed, not prefill
        stream.finalize()
        self.assertEqual(stream._output, {"role": "assistant", "thinking": "real body"})

    def test_prefix_token_ids_decoded(self):
        """The tokenizer-level helper accepts token IDs as prefix (decoded
        internally), matching how `response` is handled."""
        tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
        tokenizer.response_template = qwen3_template_with_anchor
        prefix_text = "<|im_start|>assistant\n<think>\n"
        prefix_ids = tokenizer(prefix_text).input_ids
        from_str = tokenizer.parse_response("hi</think>", prefix=prefix_text)
        from_ids = tokenizer.parse_response("hi</think>", prefix=prefix_ids)
        self.assertEqual(from_str, from_ids)

    def test_tokenizer_get_response_parser_with_prefix(self):
        """`tokenizer.get_response_parser(prefix=...)` returns a stream that
        is already in the right initial state."""
        tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-gpt2")
        tokenizer.response_template = qwen3_template_with_anchor
        prefix = "<|im_start|>assistant\n<think>\n"
        stream = tokenizer.get_response_parser(prefix=prefix)
        opens = [e for e in stream.initial_events if e["type"] == "region_open"]
        self.assertEqual([e["field"] for e in opens], ["thinking"])
        stream.feed("body</think>")
        result, _ = stream.finalize()
        self.assertEqual(result, {"role": "assistant", "thinking": "body"})


if __name__ == "__main__":
    unittest.main()
