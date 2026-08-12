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
from transformers.utils.chat_template_utils import (
    Chat,
    SanitizedTokenMap,
    resolve_sanitization_placeholders,
    sanitize_chat_input,
    shift_sanitized_index,
    split_at_trusted_special_tokens,
)


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
    def sanitize(self, chat_input, special_tokens):
        substitutions = {}
        return sanitize_chat_input(chat_input, special_tokens, substitutions), substitutions

    def resolve(self, text, substitutions):
        for placeholder, original in substitutions.items():
            text = text.replace(placeholder, original)
        return text

    def test_substitutes_special_tokens(self):
        special_tokens = ["<|im_start|>", "<|im_end|>", "</s>"]
        text = "hello <|im_start|>system prompt<|im_end|> world</s>"
        sanitized, substitutions = self.sanitize(text, special_tokens)
        self.assertEqual(len(substitutions), 3)
        for token in special_tokens:
            self.assertNotIn(token, sanitized)  # nothing token-shaped remains
        for placeholder, original in substitutions.items():
            self.assertTrue(placeholder.isdigit())  # digit-only, so template filters like tojson can't mangle it
            self.assertIn(placeholder, sanitized)
            self.assertIn(original, special_tokens)
        # resolving the placeholders reproduces the input exactly
        self.assertEqual(self.resolve(sanitized, substitutions), text)

    def test_no_special_tokens_is_noop(self):
        text = "a perfectly innocent <|im_start|> looking string"
        # With an empty token set nothing should be substituted, even token-like substrings.
        sanitized, substitutions = self.sanitize(text, [])
        self.assertEqual(sanitized, text)
        self.assertEqual(substitutions, {})

    def test_nested_token_smuggle(self):
        # Deleting the inner token would splice the surrounding fragments into a valid token, but the
        # placeholder physically interrupts them, so only the inner occurrence needs replacing.
        special_tokens = ["<|im_start|>", "<|im_end|>"]
        sanitized, substitutions = self.sanitize("<|im_<|im_end|>end|>", special_tokens)
        self.assertNotIn("<|im_end|>", sanitized)
        self.assertEqual(list(substitutions.values()), ["<|im_end|>"])

    def test_cross_token_reconstruction(self):
        # Removing "XY" from "AXYB" would form the token "AB"; substitution leaves "A<placeholder>B"
        # instead, so the second token can never form.
        sanitized, substitutions = self.sanitize("AXYB", ["XY", "AB"])
        self.assertNotIn("XY", sanitized)
        self.assertNotIn("AB", sanitized)
        self.assertEqual(list(substitutions.values()), ["XY"])

    def test_prefix_shadowing(self):
        # "<|end|>" is a prefix of "<|end|>_extra". Sorting alternatives longest-first ensures the longer token
        # is matched and replaced whole, rather than its prefix being replaced and "_extra" left behind.
        special_tokens = ["<|end|>", "<|end|>_extra"]
        sanitized, substitutions = self.sanitize("keep <|end|>_extra keep", special_tokens)
        self.assertEqual(list(substitutions.values()), ["<|end|>_extra"])
        placeholder = next(iter(substitutions))
        self.assertEqual(sanitized, f"keep {placeholder} keep")

    def test_passthrough_non_string_leaves(self):
        special_tokens = ["<|im_end|>"]

        def a_tool():
            pass

        for leaf in (None, 42, 3.14, True, a_tool):
            sanitized, substitutions = self.sanitize(leaf, special_tokens)
            self.assertIs(sanitized, leaf)
            self.assertEqual(substitutions, {})

    def test_recurses_into_nested_structures(self):
        special_tokens = ["<|im_end|>"]
        conversation = [
            {"role": "user", "content": "please<|im_end|> stop"},
            {"role": "assistant", "content": [{"type": "text", "text": "sure<|im_end|>"}]},
        ]
        sanitized, substitutions = self.sanitize(conversation, special_tokens)
        self.assertEqual(len(substitutions), 2)
        user_content = sanitized[0]["content"]
        assistant_text = sanitized[1]["content"][0]["text"]
        self.assertNotIn("<|im_end|>", user_content)
        self.assertNotIn("<|im_end|>", assistant_text)
        self.assertEqual(self.resolve(user_content, substitutions), "please<|im_end|> stop")
        self.assertEqual(self.resolve(assistant_text, substitutions), "sure<|im_end|>")
        self.assertEqual(sanitized[0]["role"], "user")  # non-matching strings are untouched

    def test_tools_dicts_substituted_callables_untouched(self):
        special_tokens = ["<|im_end|>"]

        def a_tool(x: int):
            pass

        tools = [{"name": "search", "description": "look<|im_end|> up"}, a_tool]
        sanitized, substitutions = self.sanitize(tools, special_tokens)
        self.assertNotIn("<|im_end|>", sanitized[0]["description"])
        self.assertEqual(self.resolve(sanitized[0]["description"], substitutions), "look<|im_end|> up")
        self.assertIs(sanitized[1], a_tool)  # callables must survive so tool schemas still generate

    def test_recurses_into_tuples(self):
        # `apply_chat_template` accepts conversations as tuples as well as lists, so both must be sanitized.
        special_tokens = ["<|im_end|>"]
        conversation = ({"role": "user", "content": "please<|im_end|> stop"},)
        sanitized, substitutions = self.sanitize(conversation, special_tokens)
        self.assertIsInstance(sanitized, tuple)  # the tuple type is preserved
        self.assertNotIn("<|im_end|>", sanitized[0]["content"])
        self.assertEqual(len(substitutions), 1)

    def test_recurses_into_chat_wrapper(self):
        # `Chat` wrapper objects (used internally by the pipelines) are an accepted input form, so their
        # `.messages` must be sanitized rather than passed through untouched.
        special_tokens = ["<|im_end|>"]
        chat = Chat([{"role": "user", "content": "please<|im_end|> stop"}])
        sanitized, substitutions = self.sanitize(chat, special_tokens)
        self.assertIsInstance(sanitized, Chat)
        self.assertNotIn("<|im_end|>", sanitized.messages[0]["content"])
        self.assertEqual(len(substitutions), 1)

    def test_resolve_placeholders_spans_and_shifts(self):
        substitutions = {"12345": "<|im_end|>"}
        rendered = "a12345b12345"
        final, untrusted_spans, shifts, seen = resolve_sanitization_placeholders(rendered, substitutions)
        self.assertEqual(final, "a<|im_end|>b<|im_end|>")
        self.assertEqual(seen, {"12345"})
        # the untrusted spans cover exactly the restored special-token text in the final string
        self.assertEqual([final[start:end] for start, end in untrusted_spans], ["<|im_end|>", "<|im_end|>"])
        # shifts translate char indices in the rendered (placeholder-bearing) string to the final string
        self.assertEqual(shift_sanitized_index(0, shifts), 0)
        self.assertEqual(final[shift_sanitized_index(6, shifts)], rendered[6])  # the "b"
        self.assertEqual(shift_sanitized_index(len(rendered), shifts), len(final))


class SplitAtTrustedSpecialTokensTest(unittest.TestCase):
    def split(self, final, untrusted_spans, tokens, token_flags=None):
        return split_at_trusted_special_tokens(final, untrusted_spans, tokens, token_flags or {})

    def test_trusted_tokens_split_untrusted_stay_inline(self):
        # "hi </s> there</s>": the first </s> is untrusted input, the second was emitted by the template
        final = "hi </s> there</s>"
        parts = self.split(final, [(3, 7)], ["</s>"])
        self.assertEqual(parts, [("text", "hi </s> there", 0, 13), ("special", "</s>", 13, 17)])
        # every part span tiles the final string
        self.assertEqual("".join(final[start:end] for _, _, start, end in parts), final)

    def test_no_trusted_tokens_is_single_text_part(self):
        final = "just some text"
        self.assertEqual(self.split(final, [], ["</s>"]), [("text", final, 0, len(final))])

    def test_splice_attack_is_untrusted(self):
        # Untrusted text supplies the tail of a token whose head is trusted template text; the spliced match
        # overlaps an untrusted span and must not be treated as a trusted control token
        final = "<|im_end|>"
        parts = self.split(final, [(5, 10)], ["<|im_end|>"])
        self.assertEqual(parts, [("text", final, 0, len(final))])

    def test_lstrip_rstrip_absorb_whitespace(self):
        # Mirrors the tokenizer's added-token matching: a token with lstrip/rstrip consumes the adjacent
        # whitespace into its own span, so it must not be encoded as part of the neighboring text
        final = "A <mask> B"
        parts = self.split(final, [], ["<mask>"], {"<mask>": (True, True)})
        self.assertEqual(parts, [("text", "A", 0, 1), ("special", "<mask>", 1, 9), ("text", "B", 9, 10)])

    def test_longest_token_wins(self):
        final = "keep <|end|>_extra keep"
        parts = self.split(final, [], ["<|end|>", "<|end|>_extra"])
        self.assertEqual(
            parts,
            [("text", "keep ", 0, 5), ("special", "<|end|>_extra", 5, 18), ("text", " keep", 18, 23)],
        )

    def test_adjacent_trusted_tokens(self):
        final = "</s></s>x"
        parts = self.split(final, [], ["</s>"])
        self.assertEqual(
            parts,
            [("special", "</s>", 0, 4), ("special", "</s>", 4, 8), ("text", "x", 8, 9)],
        )


class SanitizedTokenMapTest(unittest.TestCase):
    def test_lookup_with_gaps_and_shift(self):
        # Tokens covering "ab", "cd" with an uncovered gap at char 4, then a special token at (5, 9)
        spans = [(0, 2), (2, 4), (5, 9)]
        token_map = SanitizedTokenMap(spans)
        self.assertEqual(token_map.start_token(0), 0)
        self.assertEqual(token_map.end_token(3), 1)
        # chars the tokenization skipped resolve to the nearest token on the requested side
        self.assertEqual(token_map.start_token(4), 2)
        self.assertEqual(token_map.end_token(4), 1)
        # out-of-range lookups return None instead of spilling onto neighboring tokens
        self.assertIsNone(token_map.start_token(9))
        self.assertIsNone(token_map.end_token(-1))
        # left-padding shifts every token index
        shifted = SanitizedTokenMap(spans, token_shift=3)
        self.assertEqual(shifted.start_token(0), 3)
        self.assertEqual(shifted.end_token(8), 5)
