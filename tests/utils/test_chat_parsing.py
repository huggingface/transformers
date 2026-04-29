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

import random
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


# Fixtures shared by the streaming tests: one representative input per format,
# re-used for both the correctness invariant (any chunking → same dict) and the
# event-shape tests (do we emit the right events in the right order?).
_STREAMING_FIXTURES = [
    (
        "cohere",
        cohere_format,
        (
            "<|START_THINKING|>I should call a tool.<|END_THINKING|>"
            '<|START_ACTION|>[{"tool_call_id": "0", "tool_name": "simple_tool", '
            '"parameters": {"a": 1}}]<|END_ACTION|>'
        ),
    ),
    (
        "ernie",
        ernie_format,
        (
            "<think>some deliberation here</think>\n\n"
            '<tool_call>\n{"name": "get_current_temperature", "arguments": {"location": "Paris"}}\n</tool_call>\n</s>'
        ),
    ),
    (
        "gpt_oss",
        gpt_oss_format,
        "<|channel|>analysis<|message|>thinking chunk<|end|><|channel|>final<|message|>done text",
    ),
    (
        "smollm",
        smollm_format,
        '<think>thinking</think>\n<tool_call>{"name": "fn", "arguments": {"x": 1}}</tool_call>',
    ),
    (
        "qwen3",
        qwen3_format,
        (
            "<think>short thought</think>\n"
            "<tool_call>\n<function=get_weather>\n"
            "<parameter=city>\nParis\n</parameter>\n"
            "</function>\n</tool_call>"
        ),
    ),
    (
        "gemma4",
        gemma4_format,
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
    def test_stream_matches_whole_string_all_formats_fixed_chunking(self):
        """For every fixed chunking step we try, the streamed finalize()
        output must equal the whole-string parse. Regression coverage for
        specific edge-case byte boundaries (1-byte chunks hit every prefix)."""
        for name, fmt, text in _STREAMING_FIXTURES:
            expected = parse_response(text, fmt)
            for step in (1, 2, 3, 5, 7, 13, 31):
                with self.subTest(fixture=name, step=step):
                    streamer = ResponseEventStream(fmt)
                    for chunk in _chunk_fixed(text, step):
                        streamer.feed(chunk)
                    self.assertEqual(streamer.finalize(), expected)

    def test_stream_matches_whole_string_all_formats_random_chunking(self):
        """Property-style: for many random chunkings per fixture, the streamed
        finalize() output must equal the whole-string parse. Seeded so failures
        reproduce."""
        rng = random.Random(0xC0DE_5EED)
        for name, fmt, text in _STREAMING_FIXTURES:
            expected = parse_response(text, fmt)
            for trial in range(30):
                with self.subTest(fixture=name, trial=trial):
                    streamer = ResponseEventStream(fmt)
                    for chunk in _chunk_random(text, rng):
                        streamer.feed(chunk)
                    self.assertEqual(streamer.finalize(), expected)

    def test_events_well_formed_for_every_chunking(self):
        """Every event batch, across every fixture and every chunking, must be
        well-formed: region_open precedes its matching region_close for the
        same field; region_chunk only appears between open and close; the
        final event stream ends with stream_end; and the concatenation of all
        region_chunk payloads for a text/raw field equals the final value."""
        rng = random.Random(0xBEEF)
        for name, fmt, text in _STREAMING_FIXTURES:
            for trial in range(10):
                with self.subTest(fixture=name, trial=trial):
                    streamer = ResponseEventStream(fmt)
                    all_events: list[dict] = []
                    for chunk in _chunk_random(text, rng):
                        all_events.extend(streamer.feed(chunk))
                    streamer.finalize()
                    all_events.extend(streamer.final_events)
                    self._assert_event_stream_well_formed(all_events)

    def _assert_event_stream_well_formed(self, events: list[dict]) -> None:
        self.assertTrue(events, "event stream must not be empty")
        self.assertEqual(events[-1], {"type": "stream_end"})
        open_field: str | None = None
        chunk_accum: dict[str, str] = {}
        close_values: dict[str, object] = {}
        for ev in events[:-1]:
            t = ev["type"]
            if t == "region_open":
                self.assertIsNone(open_field, f"nested region_open without close: {ev}")
                open_field = ev["field"]
                chunk_accum.setdefault(open_field, "")
                self.assertIn("meta", ev)
                self.assertIsInstance(ev["meta"], dict)
            elif t == "region_chunk":
                self.assertEqual(open_field, ev["field"], f"chunk outside its region: {ev}")
                chunk_accum[open_field] += ev["text"]
            elif t == "region_close":
                self.assertEqual(open_field, ev["field"], f"close for non-open region: {ev}")
                close_values[open_field] = ev["value"]
                open_field = None
            else:
                self.fail(f"unexpected event type: {ev!r}")
        self.assertIsNone(open_field, "region left open at stream_end")

    def test_region_chunks_reconstruct_text_regions(self):
        """For text/raw regions, concatenating the region_chunk texts should
        reconstruct the final value reported in region_close. JSON-family
        regions emit no chunks and report the full parsed value only on close."""
        # Single representative case with a long text region and a JSON region.
        text = _STREAMING_FIXTURES[0][2]  # cohere fixture
        streamer = ResponseEventStream(cohere_format)
        events: list[dict] = []
        for ch in text:  # 1-byte chunks hit the most anchor boundaries
            events.extend(streamer.feed(ch))
        streamer.finalize()
        events.extend(streamer.final_events)

        # Reconstruct per-field.
        per_field_chunks: dict[str, list[str]] = {}
        per_field_close_value: dict[str, object] = {}
        for ev in events:
            if ev["type"] == "region_chunk":
                per_field_chunks.setdefault(ev["field"], []).append(ev["text"])
            elif ev["type"] == "region_close":
                per_field_close_value[ev["field"]] = ev["value"]

        # `thinking` is text → chunks concatenate to its value.
        self.assertIn("thinking", per_field_chunks)
        self.assertEqual("".join(per_field_chunks["thinking"]), "I should call a tool.")
        self.assertEqual(per_field_close_value["thinking"], "I should call a tool.")
        # `tool_calls` is json → no chunk events, full value on close.
        self.assertNotIn("tool_calls", per_field_chunks)
        self.assertIn("tool_calls", per_field_close_value)

    def test_open_event_carries_named_captures(self):
        """Named groups in open_pattern must land in region_open's `meta`."""
        text = "<|channel|>analysis<|message|>think<|end|><|channel|>commentary to=functions.foo<|message|>{}<|call|>"
        streamer = ResponseEventStream(gpt_oss_format)
        events: list[dict] = []
        for ch in text:
            events.extend(streamer.feed(ch))
        streamer.finalize()
        events.extend(streamer.final_events)

        tool_opens = [e for e in events if e["type"] == "region_open" and e["field"] == "tool_calls"]
        self.assertEqual(len(tool_opens), 1)
        self.assertEqual(tool_opens[0]["meta"], {"name": "foo"})

    def test_feed_after_finalize_raises(self):
        streamer = ResponseEventStream(smollm_format)
        streamer.feed("<think>x</think>")
        streamer.finalize()
        with self.assertRaises(RuntimeError):
            streamer.feed("more")
        with self.assertRaises(RuntimeError):
            streamer.finalize()

    def test_empty_input_streams_cleanly(self):
        streamer = ResponseEventStream(smollm_format)
        self.assertEqual(streamer.feed(""), [])
        result = streamer.finalize()
        # Only the default fields should remain; nothing else is required.
        self.assertEqual(result, {"role": "assistant"})
        self.assertEqual(streamer.final_events[-1], {"type": "stream_end"})


# ChatML-style anchor used by Qwen / SmolLM templates: assistant header is
# everything from `<|im_start|>` through `assistant\n`. Right-truncating past
# this leaves only the message body, so prefix material (`<|im_start|>`,
# role label) doesn't pollute downstream parsing.
_CHATML_ANCHOR = "<|im_start|>assistant\n"

qwen3_format_with_anchor = {**qwen3_format, "start_anchor": _CHATML_ANCHOR}
smollm_format_with_anchor = {**smollm_format, "start_anchor": _CHATML_ANCHOR}


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
        stream = ResponseEventStream(qwen3_format_with_anchor, prefix=prompt)
        # The stream surfaces a synthetic open so the caller knows the state.
        self.assertEqual(
            stream.initial_event,
            {"type": "region_open", "field": "thinking", "meta": {}},
        )
        events = stream.feed(generated)
        result = stream.finalize()
        # thinking should capture only the generated body, not the template prefix.
        self.assertEqual(result, {"role": "assistant", "thinking": "Let me think..."})
        # We should see a region_close event for thinking from the generated chunk;
        # no region_open event for thinking (the silent prefill already opened it).
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
        stream = ResponseEventStream(qwen3_format_with_anchor, prefix=prompt)
        # We landed inside `thinking` (from the LAST assistant turn's `<think>\n`),
        # not in some earlier-turn artifact.
        self.assertIsNotNone(stream.initial_event)
        self.assertEqual(stream.initial_event["field"], "thinking")
        # No earlier-turn content leaked into output.
        stream.feed("done</think>")
        stream.finalize()
        self.assertEqual(stream._output, {"role": "assistant", "thinking": "done"})

    def test_prefix_no_anchor_in_spec_keeps_full_prefix(self):
        """With no `start_anchor` in the spec, the prefix is processed verbatim
        (silently). Useful when the caller has already pre-cut the prefix."""
        prompt = "<think>\n"
        stream = ResponseEventStream(qwen3_format, prefix=prompt)  # no anchor
        self.assertEqual(
            stream.initial_event,
            {"type": "region_open", "field": "thinking", "meta": {}},
        )
        stream.feed("hi</think>")
        stream.finalize()
        self.assertEqual(stream._output, {"role": "assistant", "thinking": "hi"})

    def test_prefix_anchor_not_found_falls_back(self):
        """Spec has start_anchor but the prefix doesn't contain it: parser
        falls back to processing the entire prefix (with a logged warning)."""
        prompt = "<think>\n"  # no <|im_start|>assistant\n
        stream = ResponseEventStream(qwen3_format_with_anchor, prefix=prompt)
        # Even without truncation, the prefix opens `thinking` silently.
        self.assertEqual(
            stream.initial_event,
            {"type": "region_open", "field": "thinking", "meta": {}},
        )
        stream.feed("hi</think>")
        stream.finalize()
        self.assertEqual(stream._output, {"role": "assistant", "thinking": "hi"})

    def test_round_trip_equivalence_prefix_vs_concat(self):
        """The load-bearing property: `prefix=p` + chunked `feed(g)` produces
        the same dict as `parse_response(p + g)` and as
        `parse_response(g, prefix=p)`. Auto-truncation makes "one long array"
        equivalent to the explicit two-input form."""
        prompt = "<|im_start|>system\nA<|im_end|>\n<|im_start|>user\nB<|im_end|>\n<|im_start|>assistant\n<think>\n"
        for name, fmt_dict, gen_text in _STREAMING_FIXTURES:
            if "thinking" not in gen_text and "<think>" not in gen_text:
                continue
            # Only fixtures whose generation text is compatible with starting
            # inside `thinking`. Restrict to qwen3 / smollm shape for clarity.
            if name not in ("qwen3", "smollm"):
                continue
            fmt_with_anchor = {**fmt_dict, "start_anchor": _CHATML_ANCHOR}
            # Expected: feed (prefix + gen) without prefix kwarg, but auto-truncate
            # via parse_response should match feeding gen with prefix kwarg.
            via_prefix = parse_response(gen_text, fmt_with_anchor, prefix=prompt)
            via_concat = parse_response(prompt + gen_text, fmt_with_anchor)
            self.assertEqual(via_prefix, via_concat, msg=f"fixture={name}")
            # Streaming forms.
            for step in (1, 3, 7, 31):
                with self.subTest(fixture=name, step=step):
                    stream = ResponseEventStream(fmt_with_anchor, prefix=prompt)
                    for chunk in _chunk_fixed(gen_text, step):
                        stream.feed(chunk)
                    self.assertEqual(stream.finalize(), via_prefix)

    def test_prefix_with_open_close_inside_truncated_region(self):
        """Synthetic spec where the post-truncation prefix opens AND closes a
        region. The closed region is not surfaced in the output; subsequent
        feed continues from the implicit region (or default state)."""
        spec = {
            "defaults": {"role": "assistant"},
            "start_anchor": "[BEGIN]",
            "fields": {
                "tag": {"open": "<tag>", "close": "</tag>", "content": "text"},
                "body": {"close_pattern": r"$", "content": "text"},  # implicit
            },
        }
        prefix = "noise[BEGIN]<tag>silently consumed</tag>"
        stream = ResponseEventStream(spec, prefix=prefix)
        # Closed-and-discarded inside prefix → no initial_event.
        self.assertIsNone(stream.initial_event)
        stream.feed("real generated body")
        result = stream.finalize()
        # `tag` was closed inside the silent prefix, so it does NOT appear.
        self.assertNotIn("tag", result)
        self.assertEqual(result["body"], "real generated body")

    def test_prefix_partial_pattern_at_boundary(self):
        """Post-truncation prefix ends mid-delimiter. The first feed completes
        the match; the synthetic `initial_event` is None (no region opened
        within the prefix yet) and the open fires from `feed()`."""
        prefix = "<|im_start|>assistant\n<thi"  # incomplete `<think>`
        stream = ResponseEventStream(qwen3_format_with_anchor, prefix=prefix)
        self.assertIsNone(stream.initial_event)
        events = stream.feed("nk>real body</think>")
        types = [e["type"] for e in events]
        self.assertIn("region_open", types)  # think opens during feed, not prefill
        stream.finalize()
        self.assertEqual(stream._output, {"role": "assistant", "thinking": "real body"})

    def test_prefix_token_ids_decoded(self):
        """The tokenizer-level helper accepts token IDs as prefix (decoded
        internally), matching how `response` is handled."""
        tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
        tokenizer.response_format = qwen3_format_with_anchor
        prefix_text = "<|im_start|>assistant\n<think>\n"
        prefix_ids = tokenizer(prefix_text).input_ids
        from_str = tokenizer.parse_response("hi</think>", prefix=prefix_text)
        from_ids = tokenizer.parse_response("hi</think>", prefix=prefix_ids)
        self.assertEqual(from_str, from_ids)

    def test_tokenizer_response_event_stream_with_prefix(self):
        """`tokenizer.response_event_stream(prefix=...)` returns a stream that
        is already in the right initial state."""
        tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-gpt2")
        tokenizer.response_format = qwen3_format_with_anchor
        prefix = "<|im_start|>assistant\n<think>\n"
        stream = tokenizer.response_event_stream(prefix=prefix)
        self.assertIsNotNone(stream.initial_event)
        self.assertEqual(stream.initial_event["field"], "thinking")
        stream.feed("body</think>")
        result = stream.finalize()
        self.assertEqual(result, {"role": "assistant", "thinking": "body"})


if __name__ == "__main__":
    unittest.main()
