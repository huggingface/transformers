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
from typing import List, Optional, Tuple, Union

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
