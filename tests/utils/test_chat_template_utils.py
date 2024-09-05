# Copyright 2024 The HuggingFace Team. All rights reserved.
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

import json
import os
import tempfile
import unittest
from typing import List, Optional, Tuple, Union

from transformers import AutoTokenizer
from transformers.testing_utils import require_jinja
from transformers.utils import DocstringParsingException, TypeHintParsingException, get_json_schema


class JsonSchemaGeneratorTest(unittest.TestCase):
    def test_simple_function(self):
        def fn(x: int):
            """
            Test function

            Args:
                 x: The input
            """
            return x

        schema = get_json_schema(fn)
        expected_schema = {
            "name": "fn",
            "description": "Test function",
            "parameters": {
                "type": "object",
                "properties": {"x": {"type": "integer", "description": "The input"}},
                "required": ["x"],
            },
        }
        self.assertEqual(schema["function"], expected_schema)

    def test_no_arguments(self):
        def fn():
            """
            Test function
            """
            return True

        schema = get_json_schema(fn)
        expected_schema = {
            "name": "fn",
            "description": "Test function",
            "parameters": {"type": "object", "properties": {}},
        }
        self.assertEqual(schema["function"], expected_schema)

    def test_union(self):
        def fn(x: Union[int, float]):
            """
            Test function

            Args:
                x: The input
            """
            return x

        schema = get_json_schema(fn)
        expected_schema = {
            "name": "fn",
            "description": "Test function",
            "parameters": {
                "type": "object",
                "properties": {"x": {"type": ["integer", "number"], "description": "The input"}},
                "required": ["x"],
            },
        }
        self.assertEqual(schema["function"], expected_schema)

    def test_optional(self):
        def fn(x: Optional[int]):
            """
            Test function

            Args:
                x: The input
            """
            return x

        schema = get_json_schema(fn)
        expected_schema = {
            "name": "fn",
            "description": "Test function",
            "parameters": {
                "type": "object",
                "properties": {"x": {"type": "integer", "description": "The input", "nullable": True}},
                "required": ["x"],
            },
        }
        self.assertEqual(schema["function"], expected_schema)

    def test_default_arg(self):
        def fn(x: int = 42):
            """
            Test function

            Args:
                 x: The input
            """
            return x

        schema = get_json_schema(fn)
        expected_schema = {
            "name": "fn",
            "description": "Test function",
            "parameters": {"type": "object", "properties": {"x": {"type": "integer", "description": "The input"}}},
        }
        self.assertEqual(schema["function"], expected_schema)

    def test_nested_list(self):
        def fn(x: List[List[Union[str, int]]]):
            """
            Test function

            Args:
                x: The input
            """
            return x

        schema = get_json_schema(fn)
        expected_schema = {
            "name": "fn",
            "description": "Test function",
            "parameters": {
                "type": "object",
                "properties": {
                    "x": {
                        "type": "array",
                        "items": {"type": "array", "items": {"type": ["integer", "string"]}},
                        "description": "The input",
                    }
                },
                "required": ["x"],
            },
        }
        self.assertEqual(schema["function"], expected_schema)

    def test_multiple_arguments(self):
        def fn(x: int, y: str):
            """
            Test function

            Args:
                x: The input
                y: Also the input
            """
            return x

        schema = get_json_schema(fn)
        expected_schema = {
            "name": "fn",
            "description": "Test function",
            "parameters": {
                "type": "object",
                "properties": {
                    "x": {"type": "integer", "description": "The input"},
                    "y": {"type": "string", "description": "Also the input"},
                },
                "required": ["x", "y"],
            },
        }
        self.assertEqual(schema["function"], expected_schema)

    def test_multiple_complex_arguments(self):
        def fn(x: List[Union[int, float]], y: Optional[Union[int, str]] = None):
            """
            Test function

            Args:
                x: The input
                y: Also the input
            """
            return x

        schema = get_json_schema(fn)
        expected_schema = {
            "name": "fn",
            "description": "Test function",
            "parameters": {
                "type": "object",
                "properties": {
                    "x": {"type": "array", "items": {"type": ["integer", "number"]}, "description": "The input"},
                    "y": {
                        "type": ["integer", "string"],
                        "nullable": True,
                        "description": "Also the input",
                    },
                },
                "required": ["x"],
            },
        }
        self.assertEqual(schema["function"], expected_schema)

    def test_missing_docstring(self):
        def fn(x: int):
            return x

        with self.assertRaises(DocstringParsingException):
            get_json_schema(fn)

    def test_missing_param_docstring(self):
        def fn(x: int):
            """
            Test function
            """
            return x

        with self.assertRaises(DocstringParsingException):
            get_json_schema(fn)

    def test_missing_type_hint(self):
        def fn(x):
            """
            Test function

            Args:
                 x: The input
            """
            return x

        with self.assertRaises(TypeHintParsingException):
            get_json_schema(fn)

    def test_return_value(self):
        def fn(x: int) -> int:
            """
            Test function

            Args:
                x: The input
            """
            return x

        schema = get_json_schema(fn)
        expected_schema = {
            "name": "fn",
            "description": "Test function",
            "parameters": {
                "type": "object",
                "properties": {"x": {"type": "integer", "description": "The input"}},
                "required": ["x"],
            },
            "return": {"type": "integer"},
        }
        self.assertEqual(schema["function"], expected_schema)

    def test_return_value_docstring(self):
        def fn(x: int) -> int:
            """
            Test function

            Args:
                x: The input


            Returns:
                The output
            """
            return x

        schema = get_json_schema(fn)
        expected_schema = {
            "name": "fn",
            "description": "Test function",
            "parameters": {
                "type": "object",
                "properties": {"x": {"type": "integer", "description": "The input"}},
                "required": ["x"],
            },
            "return": {"type": "integer", "description": "The output"},
        }
        self.assertEqual(schema["function"], expected_schema)

    def test_tuple(self):
        def fn(x: Tuple[int, str]):
            """
            Test function

            Args:
                x: The input


            Returns:
                The output
            """
            return x

        schema = get_json_schema(fn)
        expected_schema = {
            "name": "fn",
            "description": "Test function",
            "parameters": {
                "type": "object",
                "properties": {
                    "x": {
                        "type": "array",
                        "prefixItems": [{"type": "integer"}, {"type": "string"}],
                        "description": "The input",
                    }
                },
                "required": ["x"],
            },
        }
        self.assertEqual(schema["function"], expected_schema)

    def test_single_element_tuple_fails(self):
        def fn(x: Tuple[int]):
            """
            Test function

            Args:
                x: The input


            Returns:
                The output
            """
            return x

        # Single-element tuples should just be the type itself, or List[type] for variable-length inputs
        with self.assertRaises(TypeHintParsingException):
            get_json_schema(fn)

    def test_ellipsis_type_fails(self):
        def fn(x: Tuple[int, ...]):
            """
            Test function

            Args:
                x: The input


            Returns:
                The output
            """
            return x

        # Variable length inputs should be specified with List[type], not Tuple[type, ...]
        with self.assertRaises(TypeHintParsingException):
            get_json_schema(fn)

    def test_enum_extraction(self):
        def fn(temperature_format: str):
            """
            Test function

            Args:
                temperature_format: The temperature format to use (Choices: ["celsius", "fahrenheit"])


            Returns:
                The temperature
            """
            return -40.0

        # Let's see if that gets correctly parsed as an enum
        schema = get_json_schema(fn)
        expected_schema = {
            "name": "fn",
            "description": "Test function",
            "parameters": {
                "type": "object",
                "properties": {
                    "temperature_format": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "The temperature format to use",
                    }
                },
                "required": ["temperature_format"],
            },
        }

        self.assertEqual(schema["function"], expected_schema)

    def test_multiline_docstring_with_types(self):
        def fn(x: int, y: int):
            """
            Test function

            Args:
                x: The first input

                y: The second input. This is a longer description
                   that spans multiple lines with indentation and stuff.

            Returns:
                God knows what
            """
            pass

        schema = get_json_schema(fn)
        expected_schema = {
            "name": "fn",
            "description": "Test function",
            "parameters": {
                "type": "object",
                "properties": {
                    "x": {"type": "integer", "description": "The first input"},
                    "y": {
                        "type": "integer",
                        "description": "The second input. This is a longer description that spans multiple lines with indentation and stuff.",
                    },
                },
                "required": ["x", "y"],
            },
        }

        self.assertEqual(schema["function"], expected_schema)

    def test_everything_all_at_once(self):
        def fn(
            x: str, y: Optional[List[Union[str, int]]], z: Tuple[Union[str, int], str] = (42, "hello")
        ) -> Tuple[int, str]:
            """
            Test function with multiple args, and docstring args that we have to strip out.

            Args:
                x: The first input. It's got a big multiline
                   description and also contains
                   (choices: ["a", "b", "c"])

                y: The second input. It's a big list with a single-line description.

                z: The third input. It's some kind of tuple with a default arg.

            Returns:
                The output. The return description is also a big multiline
                description that spans multiple lines.
            """
            pass

        schema = get_json_schema(fn)
        expected_schema = {
            "name": "fn",
            "description": "Test function with multiple args, and docstring args that we have to strip out.",
            "parameters": {
                "type": "object",
                "properties": {
                    "x": {
                        "type": "string",
                        "enum": ["a", "b", "c"],
                        "description": "The first input. It's got a big multiline description and also contains",
                    },
                    "y": {
                        "type": "array",
                        "items": {"type": ["integer", "string"]},
                        "nullable": True,
                        "description": "The second input. It's a big list with a single-line description.",
                    },
                    "z": {
                        "type": "array",
                        "prefixItems": [{"type": ["integer", "string"]}, {"type": "string"}],
                        "description": "The third input. It's some kind of tuple with a default arg.",
                    },
                },
                "required": ["x", "y"],
            },
            "return": {
                "type": "array",
                "prefixItems": [{"type": "integer"}, {"type": "string"}],
                "description": "The output. The return description is also a big multiline\n    description that spans multiple lines.",
            },
        }
        self.assertEqual(schema["function"], expected_schema)


class ChatTemplateTest(unittest.TestCase):
    def _get_tokenizer(self):
        return AutoTokenizer.from_pretrained("hf-internal-testing/tiny-gpt2-with-chatml-template")

    @require_jinja
    def test_chat_template(self):
        dummy_template = "{% for message in messages %}{{message['role'] + message['content']}}{% endfor %}"
        dummy_conversation = [
            {"role": "system", "content": "system message"},
            {"role": "user", "content": "user message"},
            {"role": "assistant", "content": "assistant message"},
        ]
        expected_output = "systemsystem messageuseruser messageassistantassistant message"
        tokenizer = self._get_tokenizer()
        output = tokenizer.apply_chat_template(
            dummy_conversation, chat_template=dummy_template, tokenize=False, return_dict=False
        )
        self.assertEqual(output, expected_output)  # Test we can pass chat_template arg

        # Check that no error raised when tokenize=True
        output = tokenizer.apply_chat_template(
            dummy_conversation, chat_template=dummy_template, tokenize=True, return_dict=False
        )
        dict_output = tokenizer.apply_chat_template(
            dummy_conversation, chat_template=dummy_template, tokenize=True, return_dict=True
        )
        self.assertEqual(dict_output["input_ids"], output)  # Test return_dict behaviour matches

        tokenizer.chat_template = dummy_template
        self.assertEqual(tokenizer.chat_template, dummy_template)  # Test property setter
        output = tokenizer.apply_chat_template(dummy_conversation, tokenize=False, return_dict=False)
        self.assertEqual(output, expected_output)  # Test chat_template attribute is used if no arg is passed
        # Check that no error raised
        tokenizer.apply_chat_template(dummy_conversation, tokenize=True, return_dict=False)

        with tempfile.TemporaryDirectory() as tmp_dir_name:
            tokenizer.save_pretrained(tmp_dir_name)
            tokenizer = tokenizer.from_pretrained(tmp_dir_name)

        self.assertEqual(tokenizer.chat_template, dummy_template)  # Test template has persisted
        output = tokenizer.apply_chat_template(dummy_conversation, tokenize=False, return_dict=False)
        self.assertEqual(output, expected_output)  # Test output is the same after reloading
        # Check that no error raised
        tokenizer.apply_chat_template(dummy_conversation, tokenize=True, return_dict=False)
        raise ValueError("Test that this test is run, and fails!")

    @require_jinja
    def test_chat_template_batched(self):
        dummy_template = "{% for message in messages %}{{message['role'] + message['content']}}{% endfor %}"
        dummy_conversations = [
            [
                {"role": "system", "content": "system message"},
                {"role": "user", "content": "user message"},
                {"role": "assistant", "content": "assistant message"},
            ],
            [
                {"role": "system", "content": "system message 2"},
                {"role": "user", "content": "user message 2"},
                {"role": "assistant", "content": "assistant message 2"},
            ],
        ]
        tokenizer = self._get_tokenizer()
        output = tokenizer.apply_chat_template(dummy_conversations, chat_template=dummy_template, tokenize=False)
        self.assertEqual(
            output,
            [
                "systemsystem messageuseruser messageassistantassistant message",
                "systemsystem message 2useruser message 2assistantassistant message 2",
            ],
        )
        one_element_output = tokenizer.apply_chat_template(
            dummy_conversations[:1], chat_template=dummy_template, tokenize=False
        )
        self.assertEqual(
            one_element_output, ["systemsystem messageuseruser messageassistantassistant message"]
        )  # Assert that list structure is retained even with one element
        tokenizer.apply_chat_template(
            dummy_conversations, chat_template=dummy_template, tokenize=True
        )  # Check that no error raised

    @require_jinja
    def test_jinja_loopcontrols(self):
        break_template = """
        {%- for message in messages %}
            {{- message.role + " " + message.content }}
            {%- if loop.first %}
                {%- break %}
            {%- endif %}
        {%- endfor %}""".strip()

        dummy_conversation = [
            {"role": "system", "content": "1"},
            {"role": "user", "content": "2"},
            {"role": "assistant", "content": "3"},
        ]

        tokenizer = self._get_tokenizer()
        break_output = tokenizer.apply_chat_template(dummy_conversation, chat_template=break_template, tokenize=False)
        self.assertEqual(break_output, "system 1")  # Loop should break after first iter

    @require_jinja
    def test_jinja_strftime(self):
        strftime_template = """{{- strftime_now("%Y-%m-%d") }}""".strip()

        dummy_conversation = [
            {"role": "system", "content": "1"},
            {"role": "user", "content": "2"},
            {"role": "assistant", "content": "3"},
        ]

        tokenizer = self._get_tokenizer()
        strftime_output = tokenizer.apply_chat_template(
            dummy_conversation, chat_template=strftime_template, tokenize=False
        )

        # Assert that we get a date formatted as expected
        self.assertEqual(len(strftime_output), 10)
        self.assertEqual(len(strftime_output.split("-")), 3)

    @require_jinja
    def test_chat_template_return_assistant_tokens_mask(self):
        dummy_template = (
            "{% for message in messages %}"
            "{% if (message['role'] != 'assistant') %}"
            "{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}"
            "{% elif (message['role'] == 'assistant')%}"
            "{{'<|im_start|>' + message['role'] + '\n'}}"
            "{% generation %}"
            "{{message['content'] + '<|im_end|>'}}"
            "{% endgeneration %}"
            "{{'\n'}}"
            "{% endif %}"
            "{% endfor %}"
        )
        conversations = [
            [
                {"role": "system", "content": "system message"},
                {"role": "user", "content": "user message"},
                {"role": "assistant", "content": "start turn 1 assistant message. end turn 1"},
                {"role": "user", "content": "user message 2"},
                {"role": "assistant", "content": "start turn 2 assistant message. end turn 2"},
            ],
            [
                {"role": "system", "content": "system message 3"},
                {"role": "user", "content": "user message 3"},
                {"role": "assistant", "content": "start turn 3 assistant message. end turn 3"},
                {"role": "user", "content": "user message 4"},
                {"role": "assistant", "content": "start turn 4 assistant message. end turn 4"},
            ],
        ]

        # These are the prefix and suffix strings of all the assistant messages. Used to find the assistant substring
        # in the entire chat string, and then find the corresponding tokens in the tokenized output.
        assistant_prefix_suffix = [
            [("start turn 1", "end turn 1<|im_end|>"), ("start turn 2", "end turn 2<|im_end|>")],
            [("start turn 3", "end turn 3<|im_end|>"), ("start turn 4", "end turn 4<|im_end|>")],
        ]
        tokenizer = self._get_tokenizer()

        # check batched
        output = tokenizer.apply_chat_template(
            conversations,
            chat_template=dummy_template,
            tokenize=True,
            return_assistant_tokens_mask=True,
            return_dict=True,
        )
        for i, conv in enumerate(conversations):
            chat_string = tokenizer.apply_chat_template(conversations[i], tokenize=False, chat_template=dummy_template)
            assistant_start = output.char_to_token(i, chat_string.index(assistant_prefix_suffix[i][0][0]))
            assistant_end = output.char_to_token(
                i,
                chat_string.index(assistant_prefix_suffix[i][0][1]) + len(assistant_prefix_suffix[i][0][1]) - 1,
            )

            assistant_start2 = output.char_to_token(i, chat_string.index(assistant_prefix_suffix[i][1][0]))
            assistant_end2 = output.char_to_token(
                i,
                chat_string.index(assistant_prefix_suffix[i][1][1]) + len(assistant_prefix_suffix[i][1][1]) - 1,
            )

            # assert 1 in first assistant message
            self.assertEqual(
                output["assistant_masks"][i][assistant_start : assistant_end + 1],
                [1] * (assistant_end - assistant_start + 1),
            )
            # assert 1 second assistant message
            self.assertEqual(
                output["assistant_masks"][i][assistant_start2 : assistant_end2 + 1],
                [1] * (assistant_end2 - assistant_start2 + 1),
            )

            # assert 0 in user/system indices
            self.assertEqual(output["assistant_masks"][i][:assistant_start], [0] * assistant_start)
            self.assertEqual(
                output["assistant_masks"][i][assistant_end + 1 : assistant_start2],
                [0] * (assistant_start2 - assistant_end - 1),
            )

        # check not batched
        output = tokenizer.apply_chat_template(
            conversations[0],
            chat_template=dummy_template,
            tokenize=True,
            return_assistant_tokens_mask=True,
            return_dict=True,
        )

        chat_string = tokenizer.apply_chat_template(conversations[0], tokenize=False, chat_template=dummy_template)
        assistant_start = output.char_to_token(0, chat_string.index(assistant_prefix_suffix[0][0][0]))
        assistant_end = output.char_to_token(
            0, chat_string.index(assistant_prefix_suffix[0][0][1]) + len(assistant_prefix_suffix[0][0][1]) - 1
        )
        assistant_start2 = output.char_to_token(0, chat_string.index(assistant_prefix_suffix[0][1][0]))
        assistant_end2 = output.char_to_token(
            0, chat_string.index(assistant_prefix_suffix[0][1][1]) + len(assistant_prefix_suffix[0][1][1]) - 1
        )

        # assert 1 in assistant indices
        self.assertEqual(
            output["assistant_masks"][assistant_start : assistant_end + 1],
            [1] * (assistant_end - assistant_start + 1),
        )
        self.assertEqual(
            output["assistant_masks"][assistant_start2 : assistant_end2 + 1],
            [1] * (assistant_end2 - assistant_start2 + 1),
        )

        # assert 0 in user/system indices
        self.assertEqual(output["assistant_masks"][:assistant_start], [0] * assistant_start)
        self.assertEqual(
            output["assistant_masks"][assistant_end + 1 : assistant_start2],
            [0] * (assistant_start2 - assistant_end - 1),
        )

    @require_jinja
    def test_continue_final_message(self):
        dummy_template = """
        {%- for message in messages %}
            {{- "<|im_start|>" + message['role'] + "\n" + message['content'] + "<|im_end|>" + "\n"}}
        {%- endfor %}"""
        dummy_conversation = [
            {"role": "system", "content": "system message"},
            {"role": "user", "content": "user message"},
            {"role": "assistant", "content": "assistant message"},
        ]
        tokenizer = self._get_tokenizer()
        output = tokenizer.apply_chat_template(
            dummy_conversation, chat_template=dummy_template, tokenize=False, continue_final_message=False
        )
        self.assertEqual(
            output,
            "<|im_start|>system\nsystem message<|im_end|>\n<|im_start|>user\nuser message<|im_end|>\n<|im_start|>assistant\nassistant message<|im_end|>\n",
        )
        prefill_output = tokenizer.apply_chat_template(
            dummy_conversation, chat_template=dummy_template, tokenize=False, continue_final_message=True
        )
        # Assert that the final message is unterminated
        self.assertEqual(
            prefill_output,
            "<|im_start|>system\nsystem message<|im_end|>\n<|im_start|>user\nuser message<|im_end|>\n<|im_start|>assistant\nassistant message",
        )

    @require_jinja
    def test_chat_template_dict(self):
        dummy_template_1 = "{{'a'}}"
        dummy_template_2 = "{{'b'}}"
        dummy_conversation = [
            {"role": "user", "content": "user message"},
        ]
        tokenizer = self._get_tokenizer()
        tokenizer.chat_template = {"template1": dummy_template_1, "template2": dummy_template_2}
        output1 = tokenizer.apply_chat_template(dummy_conversation, chat_template=dummy_template_1, tokenize=False)
        output1_via_dict = tokenizer.apply_chat_template(dummy_conversation, chat_template="template1", tokenize=False)
        self.assertEqual(output1, output1_via_dict)
        output2 = tokenizer.apply_chat_template(dummy_conversation, chat_template=dummy_template_2, tokenize=False)
        output2_via_dict = tokenizer.apply_chat_template(dummy_conversation, chat_template="template2", tokenize=False)
        self.assertEqual(output2, output2_via_dict)

    @require_jinja
    def test_chat_template_dict_saving(self):
        dummy_template_1 = "{{'a'}}"
        dummy_template_2 = "{{'b'}}"
        tokenizer = self._get_tokenizer()
        tokenizer.chat_template = {"template1": dummy_template_1, "template2": dummy_template_2}
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            tokenizer.save_pretrained(tmp_dir_name)
            config_dict = json.load(open(os.path.join(tmp_dir_name, "tokenizer_config.json")))
            # Assert that chat templates are correctly serialized as lists of dictionaries
            self.assertEqual(
                config_dict["chat_template"],
                [{"name": "template1", "template": "{{'a'}}"}, {"name": "template2", "template": "{{'b'}}"}],
            )
            new_tokenizer = tokenizer.from_pretrained(tmp_dir_name)
        # Assert that the serialized list is correctly reconstructed as a single dict
        self.assertEqual(new_tokenizer.chat_template, tokenizer.chat_template)
