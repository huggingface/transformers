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

from transformers import AutoTokenizer
from transformers.testing_utils import require_jinja, require_tokenizers
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


@require_jinja
@require_tokenizers
class ChatInputSanitizationTest(unittest.TestCase):
    # This template emits exactly one control token of its own per message, so any *additional* one in the
    # encoded output can only have come from the (user-supplied) message content
    template = "{% for message in messages %}{{ bos_token }}{{ message['content'] }}{% endfor %}"

    @classmethod
    def setUpClass(cls):
        cls.tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")
        cls.tokenizer.pad_token = cls.tokenizer.eos_token  # this tokenizer has no pad token of its own
        cls.bos = cls.tokenizer.bos_token
        cls.bos_id = cls.tokenizer.bos_token_id

    def apply(self, chat, template=None, **kwargs):
        return self.tokenizer.apply_chat_template(
            chat, chat_template=template or self.template, return_dict=False, **kwargs
        )

    def test_injected_special_tokens_encode_as_ordinary_text(self):
        chat = [{"role": "user", "content": f"hello {self.bos} world"}]
        # Without sanitization the injected token is encoded as the control token itself
        self.assertEqual(self.apply(chat).count(self.bos_id), 2)

        sanitized = self.apply(chat, sanitize_special_tokens=True)
        # Only the template's own control token remains special...
        self.assertEqual(sanitized.count(self.bos_id), 1)
        # ...and the user's text is preserved rather than stripped or escaped
        self.assertIn(f"hello {self.bos} world", self.tokenizer.decode(sanitized))

    def test_template_may_call_tokenize_securely(self):
        chat = [{"role": "user", "content": f"hello {self.bos} world"}]
        template = "{% for message in messages %}{{ message['content'].tokenize_securely() }}{% endfor %}"
        sanitized = self.apply(chat, template=template, sanitize_special_tokens=True)
        self.assertNotIn(self.bos_id, sanitized)
        self.assertIn(f"hello {self.bos} world", self.tokenizer.decode(sanitized))

    def test_chat_without_special_tokens_is_unaffected(self):
        chat = [{"role": "user", "content": "hello world"}]
        self.assertEqual(self.apply(chat, sanitize_special_tokens=True), self.apply(chat))

    def test_batched_and_padded(self):
        attack = [{"role": "user", "content": f"hello {self.bos} world"}]
        clean = [{"role": "user", "content": "hello world"}]
        batch = self.tokenizer.apply_chat_template(
            [attack, clean], chat_template=self.template, sanitize_special_tokens=True, padding=True
        )
        self.assertEqual(len(batch["input_ids"][0]), len(batch["input_ids"][1]))
        # The clean conversation still encodes exactly as it would on its own
        self.assertEqual(sum(batch["attention_mask"][1]), len(self.apply(clean)))

    def test_truncation(self):
        chat = [{"role": "user", "content": f"hello {self.bos} world"}]
        out = self.tokenizer.apply_chat_template(
            chat, chat_template=self.template, sanitize_special_tokens=True, truncation=True, max_length=5
        )
        self.assertEqual(len(out["input_ids"]), 5)

    def test_unsupported_arguments_raise(self):
        chat = [{"role": "user", "content": "hello"}]
        # The guarantee is a property of the encoding, so it cannot be expressed in string output
        with self.assertRaises(ValueError):
            self.apply(chat, tokenize=False, sanitize_special_tokens=True)
        # Assistant masks need character offsets into the rendered chat
        with self.assertRaises(ValueError):
            self.apply(chat, sanitize_special_tokens=True, return_assistant_tokens_mask=True)
