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

import copy
import dataclasses
import unittest
from enum import Enum
from typing import Literal

import numpy as np

from transformers.testing_utils import require_jinja
from transformers.utils import DocstringParsingException, TypeHintParsingException, get_json_schema
from transformers.utils.chat_template_utils import render_jinja_template


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

    def test_bare_container_types(self):
        def fn(x: list, y: tuple, z: dict):
            """
            Test function

            Args:
                x: The first input
                y: The second input
                z: The third input
            """
            return x

        # Bare builtin containers have no typing origin, so they must be mapped explicitly;
        # they should match their `typing.List` / `typing.Tuple` / `typing.Dict` equivalents.
        schema = get_json_schema(fn)
        expected_schema = {
            "name": "fn",
            "description": "Test function",
            "parameters": {
                "type": "object",
                "properties": {
                    "x": {"type": "array", "description": "The first input"},
                    "y": {"type": "array", "description": "The second input"},
                    "z": {"type": "object", "description": "The third input"},
                },
                "required": ["x", "y", "z"],
            },
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

    def test_union_with_duplicate_types(self):
        def fn(x: list | tuple, y: list | tuple | dict | None = None):
            """
            Test function

            Args:
                x: The first input
                y: The second input
            """
            return x

        # `list` and `tuple` both map to "array", which should appear only once in the schema
        schema = get_json_schema(fn)
        expected_schema = {
            "name": "fn",
            "description": "Test function",
            "parameters": {
                "type": "object",
                "properties": {
                    "x": {"type": "array", "description": "The first input"},
                    "y": {"type": ["array", "object"], "nullable": True, "description": "The second input"},
                },
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

    def test_union_of_complex_types(self):
        def fn(x: str | list[str], y: Literal["a", "b"] | int | None = None):
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
                    "x": {
                        "anyOf": [{"type": "string"}, {"type": "array", "items": {"type": "string"}}],
                        "description": "The input",
                    },
                    "y": {
                        "anyOf": [{"type": "string", "enum": ["a", "b"]}, {"type": "integer"}],
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

    def test_enum_class(self):
        class TemperatureFormat(Enum):
            CELSIUS = "celsius"
            FAHRENHEIT = "fahrenheit"

        class Priority(Enum):
            LOW = 1
            HIGH = 2
            # An alias of LOW, which should not show up as a separate value
            DEFAULT = 1

        def fn(temperature_format: TemperatureFormat, priorities: list[Priority], fallback: Priority | None = None):
            """
            Test function

            Args:
                temperature_format: The temperature format to use
                priorities: The priorities to use
                fallback: The fallback priority
            """
            return -40.0

        # Enum classes should become an enum of their member values, including when they're nested in other hints
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
                    "priorities": {
                        "type": "array",
                        "items": {"type": "integer", "enum": [1, 2]},
                        "description": "The priorities to use",
                    },
                    "fallback": {
                        "type": "integer",
                        "enum": [1, 2],
                        "nullable": True,
                        "description": "The fallback priority",
                    },
                },
                "required": ["temperature_format", "priorities"],
            },
        }

        self.assertEqual(schema["function"], expected_schema)

    def test_enum_class_with_non_json_values_fails(self):
        class Point(Enum):
            ORIGIN = (0, 0)

        def fn(x: Point):
            """
            Test function

            Args:
                x: The input
            """
            return x

        # The model can only generate JSON scalars, so there's no way for it to pick one of these members
        with self.assertRaises(TypeHintParsingException):
            get_json_schema(fn)

    def test_custom_class_fails(self):
        @dataclasses.dataclass
        class Address:
            city: str

        def fn(x: Address):
            """
            Test function

            Args:
                x: The input
            """
            return x

        def nested_fn(x: list[Address]):
            """
            Test function

            Args:
                x: The input
            """
            return x

        # We don't know what JSON a custom class expects, so these should raise rather than guessing "object"
        for tool in (fn, nested_fn):
            with self.assertRaisesRegex(TypeHintParsingException, "Address"):
                get_json_schema(tool)

    def test_custom_class_return_is_omitted(self):
        @dataclasses.dataclass
        class Address:
            city: str

        def fn(x: int) -> Address:
            """
            Test function

            Args:
                x: The input

            Returns:
                The address
            """
            return x

        def nested_fn(x: int) -> list[Address]:
            """
            Test function

            Args:
                x: The input
            """
            return x

        # The model never generates the return value, so a return hint we can't convert should be left out
        # of the schema instead of making the whole tool unusable
        expected_schema = {
            "name": "fn",
            "description": "Test function",
            "parameters": {
                "type": "object",
                "properties": {"x": {"type": "integer", "description": "The input"}},
                "required": ["x"],
            },
        }
        self.assertEqual(get_json_schema(fn)["function"], expected_schema)
        self.assertEqual(get_json_schema(nested_fn)["function"], {**expected_schema, "name": "nested_fn"})

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

    def test_multiline_docstring_with_colons(self):
        def fn(location: str, unit: str = "celsius", source: str = ""):
            """
            Test function

            Args:
                location: The location to get the temperature for.
                    Example: "Paris, France"
                unit: The unit to return the temperature in.
                    Default: "celsius" (choices: ["celsius", "fahrenheit"])
                source: The weather service to query, for example
                    https://api.example.com/v1
            """
            pass

        # Indented continuation lines belong to the description, even when they look like "name: description"
        schema = get_json_schema(fn)
        expected_schema = {
            "name": "fn",
            "description": "Test function",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": 'The location to get the temperature for. Example: "Paris, France"',
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": 'The unit to return the temperature in. Default: "celsius"',
                    },
                    "source": {
                        "type": "string",
                        "description": "The weather service to query, for example https://api.example.com/v1",
                    },
                },
                "required": ["location"],
            },
        }

        self.assertEqual(schema["function"], expected_schema)

    def test_docstring_with_section_after_args(self):
        def fn(x: int):
            """
            Test function

            Args:
                x: The input

            Example:
                fn(1)
            """
            pass

        # A "name:" line indented less than the arguments is not a continuation of the last description
        schema = get_json_schema(fn)
        self.assertEqual(schema["function"]["parameters"]["properties"]["x"]["description"], "The input")

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


# Renders reasoning from the message field named FIELD
REASONING_TEMPLATE = """
{%- for message in messages %}
    {{- "<|" + message.role + "|>\n" }}
    {%- if message.FIELD is defined %}
        {{- "<think>" + message.FIELD + "</think>\n" }}
    {%- endif %}
    {{- message.content + "<|end|>\n" }}
{%- endfor %}"""


@require_jinja
class ReasoningFieldAliasTest(unittest.TestCase):
    def test_reasoning_renders_from_any_field_name(self):
        expected = "<|user|>\nHi<|end|>\n<|assistant|>\n<think>Let me think</think>\nHello!<|end|>\n"
        for template_field in ("reasoning_content", "thinking"):
            for message_field in ("reasoning_content", "thinking", "reasoning"):
                with self.subTest(template_field=template_field, message_field=message_field):
                    chat = [
                        {"role": "user", "content": "Hi"},
                        {"role": "assistant", message_field: "Let me think", "content": "Hello!"},
                    ]
                    original_chat = copy.deepcopy(chat)
                    rendered, _ = render_jinja_template(
                        [chat], chat_template=REASONING_TEMPLATE.replace("FIELD", template_field)
                    )
                    self.assertEqual(rendered[0], expected)
                    # The aliases are added to copies of the messages, not to the caller's chat
                    self.assertEqual(chat, original_chat)

    def test_continue_final_message_on_aliased_reasoning(self):
        expected = "<|user|>\nHi<|end|>\n<|assistant|>\n<think>Let me think"
        for template_field, message_field in (("thinking", "reasoning_content"), ("reasoning_content", "thinking")):
            with self.subTest(template_field=template_field, message_field=message_field):
                chat = [
                    {"role": "user", "content": "Hi"},
                    {"role": "assistant", message_field: "Let me think", "content": "Hello!"},
                ]
                rendered, _ = render_jinja_template(
                    [chat],
                    chat_template=REASONING_TEMPLATE.replace("FIELD", template_field),
                    continue_final_message=message_field,
                )
                self.assertEqual(rendered[0], expected)


@require_jinja
class TemplateSandboxTest(unittest.TestCase):
    def render(self, template, chat=None):
        chat = chat if chat is not None else [{"role": "user", "content": "Hi"}]
        rendered, _ = render_jinja_template([chat], chat_template=template)
        return rendered[0]

    def test_dict_global_is_not_a_type(self):
        self.assertEqual(self.render("{{ dict(a=1) }}"), "{'a': 1}")
        self.assertEqual(self.render("{{ dict.fromkeys(['a']) }}"), "{'a': None}")
        # With the real `dict` type, this would build a `types.GenericAlias`
        self.assertEqual(self.render("{{ dict[1] is defined }}"), "False")

    def test_namespace_rejects_underscore_names(self):
        from jinja2.exceptions import SecurityError

        self.assertEqual(
            self.render("{% set ns = namespace(a=1, self=2) %}{% set ns.b = 3 %}{{ ns.a }}{{ ns.self }}{{ ns.b }}"),
            "123",
        )
        for template in (
            "{% set ns = namespace(_a=1) %}",
            "{% set ns = namespace({'__a': 1}) %}",
            "{% set ns = namespace() %}{% set ns.__a = 1 %}",
        ):
            with self.subTest(template=template), self.assertRaises(SecurityError):
                self.render(template)

    def test_rich_objects_become_placeholders(self):
        class Opaque:
            def method(self):
                pass

        chat = [{"role": "user", "content": [{"type": "image", "image": Opaque()}, {"type": "text", "text": "Hi"}]}]
        template = (
            "{{ messages[0].content[0].image }} {{ messages[0].content[0].image.method is defined }} "
            "{{ tools[0].obj }} {{ documents[0].obj }} {{ extra }}"
        )
        rendered, _ = render_jinja_template(
            [chat], chat_template=template, tools=[{"obj": Opaque()}], documents=[{"obj": Opaque()}], extra=Opaque()
        )
        self.assertEqual(rendered[0], "<Opaque> False <Opaque> <Opaque> <Opaque>")
        # The caller's chat keeps the real object, because processors still need it
        self.assertIsInstance(chat[0]["content"][0]["image"], Opaque)

    def test_scalar_subclasses_become_base_types(self):
        class Role(str, Enum):
            USER = "user"

        rendered = self.render(
            "{{ messages[0].role }} {{ messages[0].role.name is defined }}", [{"role": Role.USER, "content": "Hi"}]
        )
        self.assertEqual(rendered, "user False")

    def test_message_objects_render_like_dicts(self):
        @dataclasses.dataclass
        class Message:
            role: str
            content: str

        self.assertEqual(
            self.render("{{ messages[0].role }}: {{ messages[0].content }}", [Message("user", "Hi")]), "user: Hi"
        )

    def test_opaque_messages_raise(self):
        class Message:
            role = "user"
            content = "Hi"

        with self.assertRaises(TypeError):
            self.render("{{ messages[0].content }}", [Message()])

    def test_numpy_scalars_keep_their_value(self):
        template = "{{ 'on' if flag else 'off' }} {{ count + 1 }}"
        rendered, _ = render_jinja_template(
            [[{"role": "user", "content": "Hi"}]], chat_template=template, flag=np.bool_(False), count=np.int64(1)
        )
        # A placeholder string would be truthy and turn the flag on
        self.assertEqual(rendered[0], "off 2")

    def test_continue_final_message_does_not_modify_chat(self):
        chat = [
            {"role": "user", "content": [{"type": "text", "text": "Hi"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "Hello"}]},
        ]
        original_chat = copy.deepcopy(chat)
        rendered, _ = render_jinja_template(
            [chat],
            chat_template="{% for m in messages %}{{ m.content[0].text }}|{% endfor %}",
            continue_final_message=True,
        )
        self.assertEqual(rendered[0], "Hi|Hello")
        self.assertEqual(chat, original_chat)
