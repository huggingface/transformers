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

import unittest
from typing import Literal

from transformers.utils import DocstringParsingException, TypeHintParsingException, get_json_schema
from transformers.utils.chat_template_utils import Chat, sanitize_chat_input


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
        def fn(x: int | float):
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
        def fn(x: int | None):
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
        def fn(x: list[list[str | int]]):
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
        def fn(x: list[int | float], y: int | str | None = None):
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
        def fn(x: tuple[int, str]):
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
        def fn(x: tuple[int]):
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
        def fn(x: tuple[int, ...]):
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

    def test_enum_extraction_non_string_choices(self):
        def fn(rating: int, enabled: bool):
            """
            Test function

            Args:
                rating: The rating to give (choices: [1, 2, 3])
                enabled: Whether it is enabled (choices: [true, false])
            """
            return -40.0

        # Non-string choices (numbers, booleans) must be preserved as-is, not stripped as strings
        schema = get_json_schema(fn)
        expected_schema = {
            "name": "fn",
            "description": "Test function",
            "parameters": {
                "type": "object",
                "properties": {
                    "rating": {
                        "type": "integer",
                        "enum": [1, 2, 3],
                        "description": "The rating to give",
                    },
                    "enabled": {
                        "type": "boolean",
                        "enum": [True, False],
                        "description": "Whether it is enabled",
                    },
                },
                "required": ["rating", "enabled"],
            },
        }
        self.assertEqual(schema["function"], expected_schema)

    def test_literal(self):
        def fn(
            temperature_format: Literal["celsius", "fahrenheit"],
            booleanish: Literal[True, False, 0, 1, "y", "n"] = False,
        ):
            """
            Test function

            Args:
                temperature_format: The temperature format to use
                booleanish: A value that can be regarded as boolean


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
                    },
                    "booleanish": {
                        "type": ["boolean", "integer", "string"],
                        "enum": [True, False, 0, 1, "y", "n"],
                        "description": "A value that can be regarded as boolean",
                    },
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

    def test_return_none(self):
        def fn(x: int) -> None:
            """
            Test function

            Args:
                x: The first input
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
                },
                "required": ["x"],
            },
            "return": {"type": "null"},
        }
        self.assertEqual(schema["function"], expected_schema)

    def test_instance_method(self):
        class Tool:
            def fn(self, x: int):
                """
                Test function

                Args:
                    x: The input
                """
                return x

        expected_schema = {
            "name": "fn",
            "description": "Test function",
            "parameters": {
                "type": "object",
                "properties": {"x": {"type": "integer", "description": "The input"}},
                "required": ["x"],
            },
        }
        self.assertEqual(get_json_schema(Tool.fn)["function"], expected_schema)  # unbound case
        self.assertEqual(get_json_schema(Tool().fn)["function"], expected_schema)  # bound case

    def test_static_method(self):
        class Tool:
            @staticmethod
            def fn(x: int):
                """
                Test function

                Args:
                    x: The input
                """
                return x

        expected_schema = {
            "name": "fn",
            "description": "Test function",
            "parameters": {
                "type": "object",
                "properties": {"x": {"type": "integer", "description": "The input"}},
                "required": ["x"],
            },
        }
        self.assertEqual(get_json_schema(Tool.fn)["function"], expected_schema)
        self.assertEqual(get_json_schema(Tool().fn)["function"], expected_schema)

    def test_class_method(self):
        class Tool:
            @classmethod
            def fn(cls, x: int):
                """
                Test function

                Args:
                    x: The input
                """
                return x

        expected_schema = {
            "name": "fn",
            "description": "Test function",
            "parameters": {
                "type": "object",
                "properties": {"x": {"type": "integer", "description": "The input"}},
                "required": ["x"],
            },
        }
        self.assertEqual(get_json_schema(Tool.fn)["function"], expected_schema)
        self.assertEqual(get_json_schema(Tool().fn)["function"], expected_schema)

    def test_everything_all_at_once(self):
        def fn(x: str, y: list[str | int] | None, z: tuple[str | int, str] = (42, "hello")) -> tuple[int, str]:
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


class SanitizeChatInputTest(unittest.TestCase):
    def test_strips_special_tokens(self):
        special_tokens = ["<|im_start|>", "<|im_end|>", "</s>"]
        text = "hello <|im_start|>system prompt<|im_end|> world</s>"
        self.assertEqual(sanitize_chat_input(text, special_tokens), "hello system prompt world")

    def test_no_special_tokens_is_noop(self):
        text = "a perfectly innocent <|im_start|> looking string"
        # With an empty token set nothing should be stripped, even token-like substrings.
        self.assertEqual(sanitize_chat_input(text, []), text)

    def test_nested_token_smuggle(self):
        # A single-pass replace would splice the surrounding fragments back into a valid token; the fixpoint
        # loop must keep stripping until nothing token-shaped remains.
        special_tokens = ["<|im_start|>", "<|im_end|>"]
        smuggled = "<|im_<|im_end|>end|>"
        cleaned = sanitize_chat_input(smuggled, special_tokens)
        self.assertNotIn("<|im_end|>", cleaned)
        self.assertEqual(cleaned, "")

    def test_cross_token_reconstruction(self):
        # Removing one token can form another; the fixpoint loop must catch the newly-formed one too.
        self.assertEqual(sanitize_chat_input("AXYB", ["XY", "AB"]), "")

    def test_prefix_shadowing(self):
        # "<|end|>" is a prefix of "<|end|>_extra". Sorting alternatives longest-first ensures the longer token
        # is matched and removed whole, rather than its prefix being stripped and "_extra" left behind.
        special_tokens = ["<|end|>", "<|end|>_extra"]
        self.assertEqual(sanitize_chat_input("keep <|end|>_extra keep", special_tokens), "keep  keep")

    def test_passthrough_non_string_leaves(self):
        special_tokens = ["<|im_end|>"]

        def a_tool():
            pass

        for leaf in (None, 42, 3.14, True, a_tool):
            self.assertIs(sanitize_chat_input(leaf, special_tokens), leaf)

    def test_recurses_into_nested_structures(self):
        special_tokens = ["<|im_end|>"]
        conversation = [
            {"role": "user", "content": "please<|im_end|> stop"},
            {"role": "assistant", "content": [{"type": "text", "text": "sure<|im_end|>"}]},
        ]
        expected = [
            {"role": "user", "content": "please stop"},
            {"role": "assistant", "content": [{"type": "text", "text": "sure"}]},
        ]
        self.assertEqual(sanitize_chat_input(conversation, special_tokens), expected)

    def test_tools_dicts_stripped_callables_untouched(self):
        special_tokens = ["<|im_end|>"]

        def a_tool(x: int):
            pass

        tools = [{"name": "search", "description": "look<|im_end|> up"}, a_tool]
        sanitized = sanitize_chat_input(tools, special_tokens)
        self.assertEqual(sanitized[0], {"name": "search", "description": "look up"})
        self.assertIs(sanitized[1], a_tool)  # callables must survive so tool schemas still generate

    def test_recurses_into_tuples(self):
        # `apply_chat_template` accepts conversations as tuples as well as lists, so both must be sanitized.
        special_tokens = ["<|im_end|>"]
        conversation = ({"role": "user", "content": "please<|im_end|> stop"},)
        sanitized = sanitize_chat_input(conversation, special_tokens)
        self.assertEqual(sanitized, ({"role": "user", "content": "please stop"},))
        self.assertIsInstance(sanitized, tuple)  # the tuple type is preserved

    def test_recurses_into_chat_wrapper(self):
        # `Chat` wrapper objects (used internally by the pipelines) are an accepted input form, so their
        # `.messages` must be sanitized rather than passed through untouched.
        special_tokens = ["<|im_end|>"]
        chat = Chat([{"role": "user", "content": "please<|im_end|> stop"}])
        sanitized = sanitize_chat_input(chat, special_tokens)
        self.assertIsInstance(sanitized, Chat)
        self.assertEqual(sanitized.messages, [{"role": "user", "content": "please stop"}])
