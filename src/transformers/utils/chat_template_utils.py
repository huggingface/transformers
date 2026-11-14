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

import bisect
import dataclasses
import inspect
import json
import re
import types
from collections.abc import Callable, Iterable, Mapping
from contextlib import contextmanager
from datetime import datetime
from enum import Enum
from functools import lru_cache, partial
from inspect import isfunction
from typing import Any, Literal, Union, get_args, get_origin, get_type_hints, no_type_check

import numpy as np
from packaging import version

from . import logging
from .import_utils import is_jinja_available, is_torch_available, is_vision_available


logger = logging.get_logger(__name__)

if is_jinja_available():
    import jinja2
    import jinja2.exceptions
    import jinja2.ext
    import jinja2.meta
    import jinja2.nodes
    import jinja2.runtime
    import jinja2.visitor
    import markupsafe
    from jinja2.ext import Extension
    from jinja2.sandbox import ImmutableSandboxedEnvironment
else:
    jinja2 = None

if is_vision_available():
    from PIL.Image import Image

ChatType = list[dict[str, Any]]

# Characters that `render_jinja_template` puts around a control token it escapes. Processors decode escaped tokens in
# all the text they encode, so these are noncharacters, which Unicode reserves for internal use, rather than
# private-use characters, which some fonts use for icons
_ESCAPE_START = "﷐"
_ESCAPE_END = "﷑"
# The characters of an escaped token are shifted by this offset, into the plane 16 private-use area
_ESCAPED_CHAR_OFFSET = 0x100000
# Extracts the shifted characters of an escaped control token
_ESCAPED_TOKEN_RE = re.compile(f"{_ESCAPE_START}([\U00100000-\U0010fffd]+){_ESCAPE_END}")


BASIC_TYPES = (int, float, str, bool, Any, type(None), ...)
# Extracts the initial segment of the docstring, containing the function description
description_re = re.compile(r"^(.*?)[\n\s]*(Args:|Returns:|Raises:|\Z)", re.DOTALL)
# Extracts the Args: block from the docstring
args_re = re.compile(r"\n\s*Args:\n(.*?)[\n\s]*(Returns:|Raises:|\Z)", re.DOTALL)
# Splits the Args: block into individual arguments
args_split_re = re.compile(
    r"""
(?:^|\n)  # Match the start of the args block, or a newline
([ \t]*)(\w+):\s*  # Capture the indentation and the argument name
(.*?)\s*  # Capture the argument description, which can span multiple lines, and strip trailing spacing
(?=\n(?!\1[ \t])[ \t]*\w+:|\Z)  # Stop at the next "name:" that is not indented deeper, or the end of the block
""",
    re.DOTALL | re.VERBOSE,
)
# Extracts the Returns: block from the docstring, if present. Note that most chat templates ignore the return type/doc!
returns_re = re.compile(r"\n\s*Returns:\n\s*(.*?)[\n\s]*(Raises:|\Z)", re.DOTALL)


class TypeHintParsingException(Exception):
    """Exception raised for errors in parsing type hints to generate JSON schemas"""


class DocstringParsingException(Exception):
    """Exception raised for errors in parsing docstrings to generate JSON schemas"""


# The Python types that can be written directly as a JSON value, and so can appear in a schema "enum",
# mapped to the name of their JSON schema type
JSON_SCALAR_TYPES = {int: "integer", float: "number", str: "string", bool: "boolean", type(None): "null"}


def _get_json_schema_type(param_type: type) -> dict[str, str] | None:
    type_mapping = {
        int: {"type": "integer"},
        float: {"type": "number"},
        str: {"type": "string"},
        bool: {"type": "boolean"},
        type(None): {"type": "null"},
        list: {"type": "array"},
        tuple: {"type": "array"},
        dict: {"type": "object"},
        Any: {},
    }
    if is_vision_available():
        type_mapping[Image] = {"type": "image"}
    if is_torch_available():
        import torch

        type_mapping[torch.Tensor] = {"type": "audio"}
    return type_mapping.get(param_type)


def _enum_values_to_json_schema(values: list) -> dict:
    # The "type" lists each JSON type that appears in the values once, in the order they first appear
    value_types = list(dict.fromkeys(JSON_SCALAR_TYPES[type(value)] for value in values))
    return {"type": value_types[0] if len(value_types) == 1 else value_types, "enum": values}


def _parse_type_hint(hint: str) -> dict:
    origin = get_origin(hint)
    args = get_args(hint)

    if origin is None and isinstance(hint, type) and issubclass(hint, Enum):
        # An Enum class becomes a JSON "enum" of its member values, since those are what the model has to generate
        values = [member.value for member in hint]
        if not values:
            raise TypeHintParsingException(
                f"The Enum {hint.__name__} has no members, so no value could ever match it."
            )
        if any(type(value) not in JSON_SCALAR_TYPES for value in values):
            raise TypeHintParsingException(
                f"The Enum {hint.__name__} has member values that cannot be written as JSON scalars.\n"
                f"Values: {values}\n"
                "Enum member values must be int, float, str, bool or None to be converted to a JSON schema."
            )
        return _enum_values_to_json_schema(values)

    elif origin is None:
        # Any other bare type is either one of the basic types we know how to map, or a custom class.
        # We can't know what JSON a custom class expects, so we raise instead of guessing
        json_schema_type = _get_json_schema_type(hint)
        if json_schema_type is None:
            raise TypeHintParsingException(
                f"Couldn't convert the type hint {hint!r} to a JSON schema, likely because it is a custom class "
                "or object. Supported type hints are basic types like int, float, str and bool, Enum classes, "
                "typing.Literal, and list, tuple, dict and Union types containing those. If you need a more complex "
                "schema than that, you can write the JSON schema for your tool by hand and pass that instead "
                "of the function."
            )
        return json_schema_type

    elif origin is Union or (hasattr(types, "UnionType") and origin is types.UnionType):
        # Recurse into each of the subtypes in the Union, except None, which is handled separately at the end
        subtypes = [_parse_type_hint(t) for t in args if t is not type(None)]
        if len(subtypes) == 1:
            # A single non-null type can be expressed directly
            return_dict = subtypes[0]
        elif all(subtype.keys() == {"type"} and isinstance(subtype["type"], str) for subtype in subtypes):
            # A union of basic types can be expressed as a list in the schema. Subtypes carrying extra keys
            # (`items`, `enum`, `prefixItems`, ...) must go through `anyOf` so that information is not lost
            # Duplicates are dropped, since different hints (e.g. `list | tuple`) can map to the same type
            subtype_names = sorted({subtype["type"] for subtype in subtypes})
            return_dict = {"type": subtype_names[0] if len(subtype_names) == 1 else subtype_names}
        else:
            # A union of more complex types requires "anyOf"
            return_dict = {"anyOf": subtypes}
        if type(None) in args:
            return_dict["nullable"] = True
        return return_dict

    elif origin is Literal and len(args) > 0:
        if any(type(arg) not in JSON_SCALAR_TYPES for arg in args):
            raise TypeHintParsingException("Only the valid python literals can be listed in typing.Literal.")
        return _enum_values_to_json_schema(list(args))

    elif origin is list:
        if not args:
            return {"type": "array"}
        else:
            # Lists can only have a single type argument, so recurse into it
            return {"type": "array", "items": _parse_type_hint(args[0])}

    elif origin is tuple:
        if not args:
            return {"type": "array"}
        if len(args) == 1:
            raise TypeHintParsingException(
                f"The type hint {str(hint).replace('typing.', '')} is a Tuple with a single element, which "
                "we do not automatically convert to JSON schema as it is rarely necessary. If this input can contain "
                "more than one element, we recommend "
                "using a list[] type instead, or if it really is a single element, remove the tuple[] wrapper and just "
                "pass the element directly."
            )
        if ... in args:
            raise TypeHintParsingException(
                "Conversion of '...' is not supported in Tuple type hints. "
                "Use list[] types for variable-length"
                " inputs instead."
            )
        return {"type": "array", "prefixItems": [_parse_type_hint(t) for t in args]}

    elif origin is dict:
        # The JSON equivalent to a dict is 'object', which mandates that all keys are strings
        # However, we can specify the type of the dict values with "additionalProperties"
        out = {"type": "object"}
        if len(args) == 2:
            out["additionalProperties"] = _parse_type_hint(args[1])
        return out

    raise TypeHintParsingException("Couldn't parse this type hint, likely due to a custom class or object: ", hint)


def _convert_type_hints_to_json_schema(func: Callable) -> dict:
    type_hints = get_type_hints(func)
    signature = inspect.signature(func)
    func_name = getattr(func, "__name__", "operation")
    # For methods, we need to ignore the first "self" or "cls" parameter. Here we assume that if the first parameter
    # is named "self" or "cls" and has no type hint, it is an implicit receiver argument.
    first_param_name = next(iter(signature.parameters), None)
    if (
        first_param_name in {"self", "cls"}
        and signature.parameters[first_param_name].annotation == inspect.Parameter.empty
    ):
        implicit_arg_name = first_param_name
    else:
        implicit_arg_name = None
    required = []
    for param_name, param in signature.parameters.items():
        if param_name == implicit_arg_name:
            continue
        if param.annotation == inspect.Parameter.empty:
            raise TypeHintParsingException(f"Argument {param.name} is missing a type hint in function {func_name}")
        if param.default == inspect.Parameter.empty:
            required.append(param_name)

    properties = {}
    for param_name, param_type in type_hints.items():
        if param_name == implicit_arg_name:
            continue
        if param_name == "return":
            # The model never generates the return value, so a return hint we can't convert shouldn't stop the
            # tool from being used. We leave the return out of the schema rather than describing it wrongly.
            try:
                properties[param_name] = _parse_type_hint(param_type)
            except TypeHintParsingException:
                pass
        else:
            properties[param_name] = _parse_type_hint(param_type)

    schema = {"type": "object", "properties": properties}
    if required:
        schema["required"] = required

    return schema


def parse_google_format_docstring(docstring: str) -> tuple[str | None, dict | None, str | None]:
    """
    Parses a Google-style docstring to extract the function description,
    argument descriptions, and return description.

    Args:
        docstring (str): The docstring to parse.

    Returns:
        The function description, arguments, and return description.
    """

    # Extract the sections
    description_match = description_re.search(docstring)
    args_match = args_re.search(docstring)
    returns_match = returns_re.search(docstring)

    # Clean and store the sections
    description = description_match.group(1).strip() if description_match else None
    docstring_args = args_match.group(1) if args_match else None
    returns = returns_match.group(1).strip() if returns_match else None

    # Parsing the arguments into a dictionary
    if docstring_args is not None:
        docstring_args = "\n".join([line for line in docstring_args.split("\n") if line.strip()])  # Remove blank lines
        matches = args_split_re.findall(docstring_args)
        args_dict = {match[1]: re.sub(r"\s*\n+\s*", " ", match[2].strip()) for match in matches}
    else:
        args_dict = {}

    return description, args_dict, returns


def get_json_schema(func: Callable) -> dict:
    """
    This function generates a JSON schema for a given function, based on its docstring and type hints. This is
    mostly used for passing lists of tools to a chat template. The JSON schema contains the name and description of
    the function, as well as the names, types and descriptions for each of its arguments. `get_json_schema()` requires
    that the function has a docstring, and that each argument has a description in the docstring, in the standard
    Google docstring format shown below. It also requires that all user-facing arguments have valid Python type hints.
    When passing methods, implicit receiver arguments (`self` or `cls`) are ignored.

    Although it is not required, a `Returns` block can also be added, which will be included in the schema. This is
    optional because most chat templates ignore the return value of the function.

    Args:
        func: The function to generate a JSON schema for.

    Returns:
        A dictionary containing the JSON schema for the function.

    Examples:
    ```python
    >>> def multiply(x: float, y: float):
    >>>    '''
    >>>    A function that multiplies two numbers
    >>>
    >>>    Args:
    >>>        x: The first number to multiply
    >>>        y: The second number to multiply
    >>>    '''
    >>>    return x * y
    >>>
    >>> print(get_json_schema(multiply))
    {
        "name": "multiply",
        "description": "A function that multiplies two numbers",
        "parameters": {
            "type": "object",
            "properties": {
                "x": {"type": "number", "description": "The first number to multiply"},
                "y": {"type": "number", "description": "The second number to multiply"}
            },
            "required": ["x", "y"]
        }
    }
    ```

    The general use for these schemas is that they are used to generate tool descriptions for chat templates that
    support them, like so:

    ```python
    >>> from transformers import AutoTokenizer
    >>> from transformers.utils import get_json_schema
    >>>
    >>> def multiply(x: float, y: float):
    >>>    '''
    >>>    A function that multiplies two numbers
    >>>
    >>>    Args:
    >>>        x: The first number to multiply
    >>>        y: The second number to multiply
    >>>    return x * y
    >>>    '''
    >>>
    >>> multiply_schema = get_json_schema(multiply)
    >>> tokenizer = AutoTokenizer.from_pretrained("CohereForAI/c4ai-command-r-v01")
    >>> messages = [{"role": "user", "content": "What is 179 x 4571?"}]
    >>> formatted_chat = tokenizer.apply_chat_template(
    >>>     messages,
    >>>     tools=[multiply_schema],
    >>>     chat_template="tool_use",
    >>>     return_dict=True,
    >>>     return_tensors="pt",
    >>>     add_generation_prompt=True
    >>> )
    >>> # The formatted chat can now be passed to model.generate()
    ```

    Each argument description can also have an optional `(choices: ...)` block at the end, such as
    `(choices: ["tea", "coffee"])`, which will be parsed into an `enum` field in the schema. Note that this will
    only be parsed correctly if it is at the end of the line:

    ```python
    >>> def drink_beverage(beverage: str):
    >>>    '''
    >>>    A function that drinks a beverage
    >>>
    >>>    Args:
    >>>        beverage: The beverage to drink (choices: ["tea", "coffee"])
    >>>    '''
    >>>    pass
    >>>
    >>> print(get_json_schema(drink_beverage))
    ```
    {
        'name': 'drink_beverage',
        'description': 'A function that drinks a beverage',
        'parameters': {
            'type': 'object',
            'properties': {
                'beverage': {
                    'type': 'string',
                    'enum': ['tea', 'coffee'],
                    'description': 'The beverage to drink'
                    }
                },
            'required': ['beverage']
        }
    }
    """
    doc = inspect.getdoc(func)
    func_name = getattr(func, "__name__", "operation")

    if not doc:
        raise DocstringParsingException(f"Cannot generate JSON schema for {func_name} because it has no docstring!")
    doc = doc.strip()
    main_doc, param_descriptions, return_doc = parse_google_format_docstring(doc)

    json_schema = _convert_type_hints_to_json_schema(func)
    if (return_dict := json_schema["properties"].pop("return", None)) is not None:
        if return_doc is not None:  # We allow a missing return docstring since most templates ignore it
            return_dict["description"] = return_doc
    for arg, schema in json_schema["properties"].items():
        if arg not in param_descriptions:
            raise DocstringParsingException(
                f"Cannot generate JSON schema for {func_name} because the docstring has no description for the argument '{arg}'"
            )
        desc = param_descriptions[arg]
        enum_choices = re.search(r"\(choices:\s*(.*?)\)\s*$", desc, flags=re.IGNORECASE)
        if enum_choices:
            schema["enum"] = [c.strip() if isinstance(c, str) else c for c in json.loads(enum_choices.group(1))]
            desc = enum_choices.string[: enum_choices.start()].strip()
        schema["description"] = desc

    output = {"name": func_name, "description": main_doc, "parameters": json_schema}
    if return_dict is not None:
        output["return"] = return_dict
    return {"type": "function", "function": output}


@lru_cache
@no_type_check
def _get_template_variables(chat_template: str | None) -> frozenset[str]:
    """Return the set of undeclared variables referenced by a chat template.

    Uses ``jinja2.meta.find_undeclared_variables`` so that callers can
    automatically distinguish template-level kwargs from processor kwargs
    without maintaining a manual allowlist. Needed only to support BC as we
    allowed all `kwargs` to be merged into one in the past
    """
    if chat_template is None:
        return frozenset()
    compiled = _compile_jinja_template(chat_template)
    ast = compiled.environment.parse(chat_template)
    return frozenset(jinja2.meta.find_undeclared_variables(ast))


@lru_cache
@no_type_check
def _get_template_literal_text(chat_template: str) -> str:
    """
    Return the literal strings in a chat template (its raw template text, plus every string constant), joined with null
    characters so that a substring search can't match across two of them.

    Added tokens found in this text are treated as control tokens by `sanitize_control_tokens`. This misses tokens that
    the template builds from pieces, like Phi-4-mini's `'<|' + message['role'] + '|>'`. Special tokens are always
    control tokens, so only non-special tokens can be missed, and tokenizers whose templates build them should list
    their control tokens in `tokenizer.chat_control_tokens` instead.
    """
    compiled = _compile_jinja_template(chat_template)
    ast = compiled.environment.parse(chat_template)
    literals = {node.data for node in ast.find_all(jinja2.nodes.TemplateData)}
    literals |= {node.value for node in ast.find_all(jinja2.nodes.Const) if isinstance(node.value, str)}
    return "\0".join(literals)


@lru_cache
def _get_control_tokens_re(control_tokens: tuple[str, ...]) -> re.Pattern:
    # Matches any of the control tokens. Longer tokens come first, so a token takes priority over the tokens it contains
    longest_first = sorted(control_tokens, key=len, reverse=True)
    return re.compile("|".join(map(re.escape, longest_first)) or "(?!)")  # (?!) never matches


class _TemplateText(str):
    """
    A string that remembers which of its characters a chat template wrote, and which came from the template's inputs.
    Chats are rendered into this for `sanitize_control_tokens`, which has to tell the control tokens the template
    writes from the ones in messages, tools and documents.

    The template's own strings (its raw text and its string constants) are `_TemplateText` with no input spans, and
    everything the caller passes in is a plain `str`. Joining the two records where the plain parts ended up. String
    methods that aren't overridden here return a plain `str`, so any text we lose track of counts as input, and never
    as the template's.
    """

    __slots__ = ("input_spans",)
    input_spans: tuple[tuple[int, int], ...]

    def __new__(cls, text="", input_spans=None):
        self = super().__new__(cls, text)
        # Jinja and Python sometimes rebuild a string with `type(string)(...)`. We can't tell where that text came
        # from, so a string made without spans is all input
        self.input_spans = ((0, len(self)),) if input_spans is None else input_spans
        return self

    def __str__(self):
        # Compiled templates call `str()` on everything they output, which would otherwise drop the spans
        return self

    def __add__(self, other):
        if not isinstance(other, str):
            return NotImplemented
        if hasattr(other, "__html__"):
            # `other` is Markup, from the `safe` filter, and adding to Markup HTML-escapes the other side
            return _as_markup(_concat_traced((_escape_html_traced(self), other)))
        return _concat_traced((self, other))

    def __radd__(self, other):
        if not isinstance(other, str):
            return NotImplemented
        return _concat_traced((other, self))

    def __getitem__(self, key):
        text = str.__getitem__(self, key)
        if not isinstance(key, slice) or key.step not in (None, 1):
            return text
        # A plain slice keeps the input spans that fall inside it. Empty spans mark where an empty input was joined
        # in, and we keep those too, because control tokens mustn't match across them
        start, stop, _ = key.indices(len(self))
        stop = max(start, stop)
        input_spans = []
        for span_start, span_end in self.input_spans:
            clipped_start, clipped_end = max(span_start, start), min(span_end, stop)
            if clipped_start < clipped_end or (span_start == span_end and start <= span_start <= stop):
                input_spans.append((clipped_start - start, clipped_end - start))
        return _TemplateText(text, tuple(input_spans))

    def __mul__(self, count):
        if not isinstance(count, int):
            return NotImplemented
        return _concat_traced([self] * count)

    __rmul__ = __mul__

    def lstrip(self, chars=None):
        return self[len(self) - len(str.lstrip(self, chars)) :]

    def rstrip(self, chars=None):
        return self[: len(str.rstrip(self, chars))]

    def strip(self, chars=None):
        return self.lstrip(chars).rstrip(chars)

    def join(self, iterable):
        pieces = []
        for index, item in enumerate(iterable):
            if not isinstance(item, str):
                raise TypeError(f"sequence item {index}: expected str instance, {type(item).__name__} found")
            pieces += [self, item] if index else [item]
        return _concat_traced(pieces)

    def replace(self, old, new, count=-1):
        if not old:
            return str.replace(self, old, new, count)
        # We cut the string at each match, and join the pieces back up with `new`, so everything keeps its input spans
        pieces, position = [], 0
        while count != 0 and (index := str.find(self, old, position)) != -1:
            pieces += [self[position:index], new]
            position = index + len(old)
            count -= 1
        pieces.append(self[position:])
        return _concat_traced(pieces)


if is_jinja_available():

    class _TemplateMarkup(markupsafe.Markup):
        # The `safe` filter turns a string into Markup, and adding anything to Markup HTML-escapes the other side.
        # Chat templates do use the filter, so we need Markup that keeps its input spans when it's added to. Anything
        # else it does is plain Markup, which gives a string without spans, i.e. all input

        __slots__ = ("input_spans",)
        input_spans: tuple[tuple[int, int], ...]

        def __new__(cls, text="", input_spans=None):
            self = super().__new__(cls, text)
            self.input_spans = ((0, len(self)),) if input_spans is None else input_spans
            return self

        def __str__(self):
            return self

        def __add__(self, other):
            if not isinstance(other, str):
                return NotImplemented
            return _as_markup(_concat_traced((self, _escape_html_traced(other))))

        def __radd__(self, other):
            if not isinstance(other, str):
                return NotImplemented
            return _as_markup(_concat_traced((_escape_html_traced(other), self)))


def _get_input_spans(value: Any) -> tuple[tuple[int, int], ...] | None:
    # The input spans of `_TemplateText` and `_TemplateMarkup`, and None for anything else. We only look at strings,
    # because other objects in a template can object to being asked, like Jinja's undefined variables
    return getattr(value, "input_spans", None) if isinstance(value, str) else None


def _as_markup(text):
    # The `safe` filter, for templates that trace their inputs
    return _TemplateMarkup(text, _get_input_spans(text))


def _escape_html_traced(text: str) -> str:
    # HTML-escape a string like Markup does to anything added to it. Escaping works one character at a time, so we
    # can escape the template's text and the input text separately, and keep track of which is which
    if hasattr(text, "__html__"):
        return text
    plain_text = "".join([text])
    input_spans = _get_input_spans(text)
    if input_spans is None:
        input_spans = ((0, len(plain_text)),)
    pieces, position = [], 0
    for start, end in input_spans:
        pieces.append(_TemplateText(markupsafe.escape(plain_text[position:start]), ()))
        pieces.append(str(markupsafe.escape(plain_text[start:end])))
        position = end
    pieces.append(_TemplateText(markupsafe.escape(plain_text[position:]), ()))
    return _concat_traced(pieces)


def _concat_traced(pieces: Iterable) -> _TemplateText:
    # Join the pieces like `"".join()`, keeping track of the text that came from the template's inputs. Jinja joins
    # everything that a template, macro or block outputs with this, and we use it for `~` as well
    texts, input_spans, position = [], [], 0
    for piece in pieces:
        piece_spans = _get_input_spans(piece)
        if piece_spans is None:
            # Anything that isn't `_TemplateText` came from the inputs
            piece = str(piece)
            piece_spans = ((0, len(piece)),)
        for start, end in piece_spans:
            if input_spans and input_spans[-1][1] == start + position:
                input_spans[-1] = (input_spans[-1][0], end + position)
            else:
                input_spans.append((start + position, end + position))
        texts.append(piece)
        position += len(piece)
    return _TemplateText("".join(texts), tuple(input_spans))


def _wrap_traced_str_format(jinja_env, value):
    # The sandbox swaps `str.format` for a wrapper of its own, which rebuilds the result with `type(string)(...)`.
    # Some templates build their control tokens with `format`, like `'<|{}|>'.format(suffix)`, so we keep track of the
    # simple case: when the template wrote the format string and all of its arguments, it wrote the result
    wrapper = ImmutableSandboxedEnvironment.wrap_str_format(jinja_env, value)
    if wrapper is None:
        return None
    format_string, is_format_map = value.__self__, value.__name__ == "format_map"

    def traced_wrapper(*args, **kwargs):
        result = wrapper(*args, **kwargs)
        arguments = list(args[0].values()) if is_format_map else [*args, *kwargs.values()]
        if all(_get_input_spans(argument) == () for argument in [format_string, *arguments]):
            return _TemplateText(result, ())
        return result

    return traced_wrapper


@lru_cache
@no_type_check
def _compile_traced_jinja_template(chat_template):
    """
    Compile a chat template so that it renders `_TemplateText`, which records the text that came from its inputs.

    Three things change compared to `_compile_jinja_template`:
      - Every string the template contains is loaded as `_TemplateText`. Templates use their strings to search and
        compare as well as to output, and `_TemplateText` is equal to the plain string for all of that.
      - Everything that joins strings keeps track of the inputs: the output of the template and of its macros and
        blocks (`concat`), and the `~` and `+` operators.
      - `format` and the `safe` filter, which templates use on their control tokens, don't lose track of them.
    """
    # We take a fresh environment with all our usual settings, and change it
    jinja_env = _cached_compile_jinja_template(chat_template).environment
    jinja_env.concat = _concat_traced
    jinja_env.wrap_str_format = partial(_wrap_traced_str_format, jinja_env)
    jinja_env.filters["safe"] = _as_markup
    jinja_env.globals["_concat_traced"] = _concat_traced
    literals = {}

    class LiteralTracer(jinja2.visitor.NodeTransformer):
        def trace(self, text, node):
            # Each literal becomes a variable that holds it as `_TemplateText`
            name = literals.setdefault(text, f"_template_text_{len(literals)}")
            return jinja2.nodes.Name(name, "load", lineno=node.lineno)

        def visit_TemplateData(self, node):
            return self.trace(node.data, node)

        def visit_Const(self, node):
            return self.trace(node.value, node) if type(node.value) is str else node

        def visit_Concat(self, node):
            # `~` compiles to a join that returns a plain string, so we call ours
            self.generic_visit(node)
            join = jinja2.nodes.Name("_concat_traced", "load", lineno=node.lineno)
            return jinja2.nodes.Call(join, [jinja2.nodes.List(node.nodes)], [], None, None, lineno=node.lineno)

    template_ast = LiteralTracer().visit(jinja_env.parse(chat_template))
    template_ast.set_environment(jinja_env)
    jinja_env.globals.update({name: _TemplateText(text, ()) for text, name in literals.items()})
    return jinja2.Template.from_code(jinja_env, jinja_env.compile(template_ast), jinja_env.make_globals(None))


def _escape_input_control_tokens(rendered_chat: _TemplateText, control_tokens_re: re.Pattern) -> str:
    # Escape the control tokens in a rendered chat that the template didn't write itself. A token counts as the
    # template's only when the template wrote all of it, so tokens that the inputs complete are escaped as well, and
    # so are tokens that only appear once the template joins two inputs
    text = "".join([rendered_chat])
    span_starts = [start for start, _ in rendered_chat.input_spans]
    span_ends = [end for _, end in rendered_chat.input_spans]
    pieces, position = [], 0
    for match in control_tokens_re.finditer(text):
        start, end = match.span()
        # The spans are sorted and don't overlap, so only the first one that ends after the token starts can overlap
        # it. An empty span is where the template joined in an empty input, which a token can't match across either
        index = bisect.bisect_right(span_ends, start)
        if index == len(span_ends) or span_starts[index] >= end:
            continue
        token = match.group()
        if any(ord(char) > 0xFFFD for char in token):
            raise ValueError(
                f"Can't sanitize the control token {token!r}: it has characters outside the Basic Multilingual Plane, "
                "which can't be escaped. If the token is listed in `tokenizer.chat_control_tokens` and doesn't need "
                "sanitizing, you can remove it from there."
            )
        escaped = "".join(chr(ord(char) + _ESCAPED_CHAR_OFFSET) for char in token)
        pieces += [text[position:start], f"{_ESCAPE_START}{escaped}{_ESCAPE_END}"]
        position = end
    pieces.append(text[position:])
    return "".join(pieces)


def _parse_escaped_control_tokens(text: str) -> tuple[str, list[tuple[int, int]]]:
    # Undo `_escape_input_control_tokens`. Returns the text with its escaped tokens restored, and the character spans
    # of those tokens in it. Splitting on the escaped tokens gives [text, escaped token, text, escaped token, ..., text]
    pieces = _ESCAPED_TOKEN_RE.split(text)
    restored, escaped_spans = pieces[0], []
    for escaped, piece in zip(pieces[1::2], pieces[2::2]):
        token = "".join(chr(ord(char) - _ESCAPED_CHAR_OFFSET) for char in escaped)
        escaped_spans.append((len(restored), len(restored) + len(token)))
        restored += token + piece
    return restored, escaped_spans


def _encode_stretch(tokenizer, text: str, start: int, end: int) -> list[int]:
    # Encode `text[start:end]` as plain text, with the steps of the backend tokenizer that come after it has found the
    # added tokens. We run them on the stretch where it is in the text, and not on a copy of it, because some steps
    # treat the start of the text differently (like SentencePiece's prefix space)
    from tokenizers import PreTokenizedString

    if start == end:
        return []
    backend = tokenizer.backend_tokenizer
    pretokenized = PreTokenizedString(text)
    pretokenized.split(lambda _, normalized: [normalized[start:end]])
    if backend.normalizer is not None:
        pretokenized.normalize(backend.normalizer.normalize)
    if backend.pre_tokenizer is not None:
        backend.pre_tokenizer.pre_tokenize(pretokenized)
    pretokenized.tokenize(backend.model.tokenize)
    return [token.id for _, _, tokens in pretokenized.get_splits() for token in tokens]


def _encode_escaped_spans_as_text(
    tokenizer,
    text: str,
    escaped_spans: list[tuple[int, int]],
    input_ids: list[int],
    offsets: list[tuple[int, int]],
    special_tokens_mask: list[int],
) -> tuple[list[int], list[int | None]]:
    """
    Fix the tokens of a text that `_parse_escaped_control_tokens` restored and the tokenizer then encoded normally, so
    that its escaped tokens are encoded as plain text.

    The tokenizer finds added tokens first, and encodes the text between them one stretch at a time. So when an added
    token overlaps an escaped span, we drop it, and encode the whole stretch it was in again, between the added tokens
    that remain on either side. This is what the tokenizer would do if the escaped token wasn't an added token at all.
    All other tokens are left as they are.

    Args:
        tokenizer: The tokenizer that encoded the text.
        text: The restored text.
        escaped_spans: The character spans of the escaped tokens in `text`.
        input_ids: The token ids of `text`, encoded normally.
        offsets: The character span of each token in `text`.
        special_tokens_mask: Which tokens the tokenizer added around the text, like BOS.

    Returns:
        The new token ids, and for each of them the index of the token it was copied from, or `None` if it was encoded
        again.
    """
    # First, we find the tokens that were matched as added tokens. We can't go by their ids alone, because the model
    # also outputs ids like the unknown token's for text that it can't encode
    added_tokens = tokenizer.added_tokens_decoder
    is_added = [
        not is_special and token_id in added_tokens and added_tokens[token_id].content in text[start:end]
        for token_id, (start, end), is_special in zip(input_ids, offsets, special_tokens_mask)
    ]
    # The escaped spans are sorted and don't overlap, so an added token overlaps one of them exactly when it overlaps
    # the first span that ends after the token starts
    span_starts = [span_start for span_start, _ in escaped_spans]
    span_ends = [span_end for _, span_end in escaped_spans]
    dropped = [
        is_added[i] and (span := bisect.bisect_right(span_ends, start)) < len(span_ends) and span_starts[span] < end
        for i, (start, end) in enumerate(offsets)
    ]
    if not any(dropped):
        return list(input_ids), list(range(len(input_ids)))

    # Next, we rebuild the tokens one stretch at a time. A stretch ends at each added token that we keep, and at each
    # token that the tokenizer added around the text. (Added tokens with `normalized=True` are found after the text
    # is normalized, so the tokenizer really normalizes the text on both sides of them in one go. Ending the stretch
    # there only differs for normalizers that prepend to their text, when the stretch also has an escaped token)
    boundaries = [i for i in range(len(input_ids)) if special_tokens_mask[i] or (is_added[i] and not dropped[i])]
    new_ids, origins = [], []
    first_token, stretch_start = 0, 0
    for boundary in [*boundaries, len(input_ids)]:
        if any(dropped[first_token:boundary]):
            # The stretch had an escaped token in it, so we encode its text again, without added tokens
            ends_text = boundary == len(input_ids) or special_tokens_mask[boundary]
            stretch_end = len(text) if ends_text else offsets[boundary][0]
            stretch_ids = _encode_stretch(tokenizer, text, stretch_start, stretch_end)
            for token_id in {input_ids[i] for i in range(first_token, boundary) if dropped[i]} & set(stretch_ids):
                raise ValueError(
                    f"Can't sanitize the control token {added_tokens[token_id].content!r}: the tokenizer encodes plain "
                    "text as this token too. If the token is listed in `tokenizer.chat_control_tokens` and doesn't "
                    "need sanitizing, you can remove it from there."
                )
            new_ids += stretch_ids
            origins += [None] * len(stretch_ids)
        else:
            # Nothing to fix in this stretch
            new_ids += input_ids[first_token:boundary]
            origins += range(first_token, boundary)
        if boundary < len(input_ids):
            new_ids.append(input_ids[boundary])
            origins.append(boundary)
            if not special_tokens_mask[boundary]:
                stretch_start = offsets[boundary][1]
        first_token = boundary + 1
    return new_ids, origins


def encode_escaped_chat(tokenizer, text: str | list[str], **kwargs):
    """
    Encode text like `tokenizer(text, **kwargs)`, except that escaped control tokens are encoded as plain text.

    Chats rendered with `sanitize_control_tokens=True` escape the control tokens that came from messages, tools and
    documents, so that neither tokenizers nor processors mistake them for the control tokens the chat template wrote.
    Tokenizers encode their chats with this function, and processors encode all their text with it. Text without
    escaped tokens is passed straight to the tokenizer.

    Text with escaped tokens is restored and encoded normally first, and then `_encode_escaped_spans_as_text` encodes
    the escaped tokens again as plain text. That changes the number of tokens, so we truncate and pad at the end.

    Args:
        tokenizer ([`PreTrainedTokenizerBase`]):
            The tokenizer to encode with. Text with escaped tokens needs a fast tokenizer.
        text (`str` or `list[str]`):
            The text to encode.
        kwargs:
            Arguments for the tokenizer's `__call__`. Pairs, targets, pretokenized text, overflowing tokens, offsets and
            lengths aren't supported for text with escaped tokens.

    Example:

    ```python
    >>> from transformers import AutoTokenizer
    >>> from transformers.utils.chat_template_utils import encode_escaped_chat, render_jinja_template

    >>> tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
    >>> chat = [{"role": "user", "content": "What does <|endoftext|> do?"}]
    >>> template = "{{ messages[0]['content'] }}<|endoftext|>"
    >>> # The control token in the message is escaped, and the one that the template writes is not
    >>> (escaped,), _ = render_jinja_template([chat], chat_template=template, control_tokens=("<|endoftext|>",))
    >>> input_ids = encode_escaped_chat(tokenizer, escaped)["input_ids"]
    >>> input_ids.count(tokenizer.eos_token_id)
    1
    ```
    """
    is_batched = not isinstance(text, str)
    texts = text if is_batched else [text]
    if not isinstance(texts, (list, tuple)) or not any(
        isinstance(sample, str) and _ESCAPED_TOKEN_RE.search(sample) for sample in texts
    ):
        return tokenizer(text, **kwargs)

    from ..tokenization_utils_base import TruncationStrategy

    if not tokenizer.is_fast:
        raise ValueError(
            "The text has control tokens escaped by `sanitize_control_tokens=True`, which only fast tokenizers can "
            f"encode, but got a {type(tokenizer).__name__}."
        )
    unsupported = [
        "text_pair",
        "text_target",
        "text_pair_target",
        "is_split_into_words",
        "stride",
        "return_overflowing_tokens",
        "return_offsets_mapping",
        "return_length",
    ]
    if used := [key for key in unsupported if kwargs.get(key)]:
        raise ValueError(
            "The text has control tokens escaped by `sanitize_control_tokens=True`, which can't be encoded with "
            f"{', '.join(used)}."
        )
    kwargs = {key: value for key, value in kwargs.items() if key not in unsupported}

    padding = kwargs.pop("padding", False)
    truncation = kwargs.pop("truncation", None)
    max_length = kwargs.pop("max_length", None)
    pad_to_multiple_of = kwargs.pop("pad_to_multiple_of", None)
    padding_side = kwargs.pop("padding_side", None)
    return_tensors = kwargs.pop("return_tensors", None)
    return_attention_mask = kwargs.pop("return_attention_mask", None)
    return_special_tokens_mask = kwargs.pop("return_special_tokens_mask", False)
    verbose = kwargs.pop("verbose", True)

    # First, we restore the text and encode it normally. `pad` adds the attention mask at the end, and the lengths
    # are only right once the tokens are fixed, so the tokenizer shouldn't warn about them here
    restored = [_parse_escaped_control_tokens(sample) for sample in texts]
    encodings = tokenizer(
        [restored_text for restored_text, _ in restored],
        return_offsets_mapping=True,
        return_special_tokens_mask=True,
        return_attention_mask=False,
        verbose=False,
        **kwargs,
    )

    # Next, we fix the tokens of the escaped spans, and line the other outputs up with the new tokens
    per_token_keys = [key for key in ("token_type_ids", "special_tokens_mask") if key in encodings]
    outputs = {key: [] for key in ["input_ids", *per_token_keys]}
    for i, (restored_text, escaped_spans) in enumerate(restored):
        input_ids, origins = _encode_escaped_spans_as_text(
            tokenizer,
            restored_text,
            escaped_spans,
            encodings["input_ids"][i],
            encodings["offset_mapping"][i],
            encodings["special_tokens_mask"][i],
        )
        outputs["input_ids"].append(input_ids)
        for key in per_token_keys:
            # Tokens that were encoded again are text, so they're in the first segment and aren't special
            values = encodings[key][i]
            outputs[key].append([values[origin] if origin is not None else 0 for origin in origins])

    # Finally, we truncate and pad. Like the tokenizer, we only truncate the text, and keep the special tokens that
    # were added around it
    padding_strategy, truncation_strategy, max_length, _ = tokenizer._get_padding_truncation_strategies(
        padding=padding,
        truncation=truncation,
        max_length=max_length,
        pad_to_multiple_of=pad_to_multiple_of,
        verbose=verbose,
    )
    if truncation_strategy != TruncationStrategy.DO_NOT_TRUNCATE and max_length is not None:
        for i, special_tokens_mask in enumerate(outputs["special_tokens_mask"]):
            num_tokens = len(special_tokens_mask)
            num_before = next((n for n, is_special in enumerate(special_tokens_mask) if not is_special), num_tokens)
            num_after = next((n for n, is_special in enumerate(reversed(special_tokens_mask)) if not is_special), 0)
            num_removed = min(num_tokens - max_length, num_tokens - num_before - num_after)
            if num_removed <= 0:
                continue
            if tokenizer.truncation_side == "right":
                removed = slice(num_tokens - num_after - num_removed, num_tokens - num_after)
            else:
                removed = slice(num_before, num_before + num_removed)
            for rows in outputs.values():
                del rows[i][removed]
    for input_ids in outputs["input_ids"]:
        tokenizer._eventual_warn_about_too_long_sequence(input_ids, max_length, verbose)
    if not return_special_tokens_mask:
        del outputs["special_tokens_mask"]
    if not is_batched and not return_tensors:
        outputs = {key: rows[0] for key, rows in outputs.items()}
    return tokenizer.pad(
        outputs,
        padding=padding_strategy,
        max_length=max_length,
        pad_to_multiple_of=pad_to_multiple_of,
        padding_side=padding_side,
        return_attention_mask=return_attention_mask,
        return_tensors=return_tensors,
        verbose=verbose,
    )


def _render_with_assistant_indices(
    compiled_template, messages, tools, documents, add_generation_prompt, **template_kwargs
):
    rendered_blocks = []
    generation_indices = []
    with compiled_template.environment.activate_tracker(rendered_blocks, generation_indices):
        for block in compiled_template.generate(
            messages=messages,
            tools=tools,
            documents=documents,
            add_generation_prompt=add_generation_prompt,
            **template_kwargs,
        ):
            rendered_blocks.append(block)
        rendered_chat = "".join(rendered_blocks)
    return rendered_chat, generation_indices


@lru_cache
def _compile_jinja_template(chat_template):
    return _cached_compile_jinja_template(chat_template)


@no_type_check
def _cached_compile_jinja_template(chat_template):
    if not is_jinja_available():
        raise ImportError(
            "apply_chat_template requires jinja2 to be installed. Please install it using `pip install jinja2`."
        )

    class AssistantTracker(Extension):
        # This extension is used to track the indices of assistant-generated tokens in the rendered chat
        tags = {"generation"}

        def __init__(self, environment: ImmutableSandboxedEnvironment):
            # The class is only initiated by jinja.
            super().__init__(environment)
            environment.extend(activate_tracker=self.activate_tracker)
            self._rendered_blocks = None
            self._generation_indices = None

        def parse(self, parser: jinja2.parser.Parser) -> jinja2.nodes.CallBlock:
            lineno = next(parser.stream).lineno
            body = parser.parse_statements(["name:endgeneration"], drop_needle=True)
            return jinja2.nodes.CallBlock(self.call_method("_generation_support"), [], [], body).set_lineno(lineno)

        @jinja2.pass_eval_context
        def _generation_support(self, context: jinja2.nodes.EvalContext, caller: jinja2.runtime.Macro) -> str:
            rv = caller()
            if self.is_active():
                # Only track generation indices if the tracker is active
                start_index = len("".join(self._rendered_blocks))
                end_index = start_index + len(rv)
                self._generation_indices.append((start_index, end_index))
            return rv

        def is_active(self) -> bool:
            return self._rendered_blocks is not None or self._generation_indices is not None

        @contextmanager
        def activate_tracker(self, rendered_blocks: list[int], generation_indices: list[int]):
            try:
                if self.is_active():
                    raise ValueError("AssistantTracker should not be reused before closed")
                self._rendered_blocks = rendered_blocks
                self._generation_indices = generation_indices

                yield
            finally:
                self._rendered_blocks = None
                self._generation_indices = None

    if version.parse(jinja2.__version__) < version.parse("3.1.6"):
        # Older versions have known sandbox escapes, and chat templates from the Hub are untrusted code
        raise ImportError(
            f"apply_chat_template requires jinja2>=3.1.6 to be installed. Your version is {jinja2.__version__}. "
            "You can upgrade it with `pip install -U jinja2`."
        )

    def raise_exception(message):
        raise jinja2.exceptions.TemplateError(message)

    def tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
        # We override the built-in tojson filter because Jinja's default filter escapes HTML characters
        # We also expose some options like custom indents and separators
        return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)

    def strftime_now(format):
        return datetime.now().strftime(format)

    class TemplateDict:
        # Make a limited subclass so that users can't break out with dictionary attribute access
        # or type-hackery
        def __call__(self, *args, **kwargs):
            return dict(*args, **kwargs)

        def fromkeys(self, *args, **kwargs):
            return dict.fromkeys(*args, **kwargs)

    class TemplateNamespace(jinja2.utils.Namespace):
        # Namespaces reject any dunder attributes to prevent access to CPython internals
        def __init__(*args, **kwargs):
            # Same signature trick as Jinja's Namespace, so that `namespace(self=...)` still works
            self, args = args[0], args[1:]
            attrs = dict(*args, **kwargs)
            for name in attrs:
                TemplateNamespace._check_name(name)
            jinja2.utils.Namespace.__init__(self, attrs)

        def __setitem__(self, name, value):
            TemplateNamespace._check_name(name)
            jinja2.utils.Namespace.__setitem__(self, name, value)

        @staticmethod
        def _check_name(name):
            if isinstance(name, str) and name.startswith("_"):
                raise jinja2.exceptions.SecurityError(
                    f"Chat templates cannot store underscore-prefixed attributes on a namespace, but got {name!r}. "
                    "Please rename the attribute."
                )

    jinja_env = ImmutableSandboxedEnvironment(
        trim_blocks=True, lstrip_blocks=True, extensions=[AssistantTracker, jinja2.ext.loopcontrols]
    )
    jinja_env.filters["tojson"] = tojson
    jinja_env.globals["raise_exception"] = raise_exception
    jinja_env.globals["strftime_now"] = strftime_now
    jinja_env.globals["dict"] = TemplateDict()
    jinja_env.globals["namespace"] = TemplateNamespace
    return jinja_env.from_string(chat_template)


# Names a message's reasoning is passed under. The first is the standard one that chat templates read.
_REASONING_FIELDS = ("reasoning_content", "thinking", "reasoning")


def _alias_reasoning_fields(chat: ChatType) -> ChatType:
    # Fill `reasoning_content` from older names for it, and fill `thinking` for gpt-oss templates, which read that
    aliased_chat = []
    for message in chat:
        if isinstance(message, dict):
            reasoning = next((message[key] for key in _REASONING_FIELDS if message.get(key) is not None), None)
            if reasoning is not None:
                message = {"thinking": reasoning, **message, "reasoning_content": reasoning}
        aliased_chat.append(message)
    return aliased_chat


def _plain_template_data(value: Any) -> Any:
    # Chat templates come from the Hub, but callers often put rich objects in their inputs, like the decoded images,
    # arrays and tensors in multimodal content blocks. Those objects have methods that the sandbox doesn't know are
    # dangerous (some of them write files!), so we rebuild everything the template sees out of plain JSON types,
    # and replace anything else with an inert placeholder. Templates only check whether media keys are present,
    # and never read their values, so this doesn't change what they render.
    if value is None or type(value) in (str, int, float, bool):
        # Exact JSON scalars are safe as-is
        return value
    elif isinstance(value, str):
        # Subclasses (enums, numpy scalars, ...) can carry extra methods, so we strip them back to the base type
        return str.__str__(value)
    elif isinstance(value, int):
        return int.__int__(value)
    elif isinstance(value, float):
        return float.__float__(value)
    elif isinstance(value, Mapping):
        return {_plain_template_data(key): _plain_template_data(item) for key, item in value.items()}
    elif isinstance(value, list):
        return [_plain_template_data(item) for item in value]
    elif isinstance(value, tuple):
        return tuple(_plain_template_data(item) for item in value)
    elif isinstance(value, np.generic):
        # `np.bool_(False)` would otherwise become a truthy placeholder string
        return _plain_template_data(value.item())
    elif dataclasses.is_dataclass(value) and not isinstance(value, type):
        # Fields are read one level at a time, so arrays and images inside are not deep-copied
        return {field.name: _plain_template_data(getattr(value, field.name)) for field in dataclasses.fields(value)}
    elif callable(getattr(value, "model_dump", None)):
        # Pydantic models
        return _plain_template_data(value.model_dump())
    else:
        # Anything else becomes a placeholder naming its type, e.g. "<ndarray>"
        return f"<{type(value).__name__}>"


def render_jinja_template(
    conversations: list[ChatType],
    tools: list[dict | Callable] | None = None,
    documents: ChatType | None = None,
    chat_template: str | None = None,
    return_assistant_tokens_mask: bool = False,
    continue_final_message: bool | str = False,
    add_generation_prompt: bool = False,
    control_tokens: tuple[str, ...] | None = None,
    special_tokens_map: dict[str, Any] | None = None,
    **kwargs,
) -> str:
    """
    Render conversations with a chat template.

    Args:
        conversations: The conversations to render, each a list of messages.
        tools: Tools the model can call, as JSON schemas or as functions, which are converted to schemas.
        documents: Documents for RAG models, as dicts.
        chat_template: The Jinja template to render with.
        return_assistant_tokens_mask: Whether to also return the character spans of the template's `generation` blocks.
        continue_final_message: Whether to cut the render off at the end of the final message's content, or the name
            of the field of the final message to cut it off at.
        add_generation_prompt: Passed to the template.
        control_tokens: If set, these tokens are escaped wherever the template didn't write them itself, i.e. where
            they came from messages, tools, documents or `kwargs`. `encode_escaped_chat` then encodes them as plain
            text. This is how `apply_chat_template` implements `sanitize_control_tokens`.
        special_tokens_map: The tokenizer's special tokens, which templates read as variables like `bos_token`. With
            `control_tokens`, they count as text that the template wrote.
        kwargs: Other variables for the template. They overwrite special tokens of the same name.

    Returns:
        The rendered conversations, and for each one the spans of its `generation` blocks, if they were asked for.
    """
    if return_assistant_tokens_mask and not re.search(r"\{\%-?\s*generation\s*-?\%\}", chat_template):
        logger.warning_once(
            "return_assistant_tokens_mask==True but chat template does not contain `{% generation %}` keyword."
        )

    # Compilation function uses a cache to avoid recompiling the same template
    if control_tokens is not None:
        compiled_template = _compile_traced_jinja_template(chat_template)
    else:
        compiled_template = _compile_jinja_template(chat_template)

    # We accept either JSON schemas or functions for tools. If we get functions, we convert them to schemas
    if tools is not None:
        tool_schemas = []
        for tool in tools:
            if isinstance(tool, dict):
                tool_schemas.append(tool)
            elif isfunction(tool) or inspect.ismethod(tool):
                tool_schemas.append(get_json_schema(tool))
            else:
                raise ValueError(
                    "Tools should either be a JSON schema, or a callable function with type hints "
                    "and a docstring suitable for auto-conversion to a schema."
                )
    else:
        tool_schemas = None

    if documents is not None:
        for document in documents:
            if not isinstance(document, dict):
                raise TypeError("Documents should be a list of dicts with 'title' and 'text' keys!")

    # Everything the template can see goes through `_plain_template_data`. The conversations are converted below.
    tool_schemas = _plain_template_data(tool_schemas)
    documents = _plain_template_data(documents)
    kwargs = _plain_template_data(kwargs)
    special_tokens_map = _plain_template_data(special_tokens_map or {})
    if control_tokens is not None:
        # The special tokens are the tokenizer's own, so they are as good as written into the template
        special_tokens_map = {
            name: _TemplateText(token, ()) if isinstance(token, str) else token
            for name, token in special_tokens_map.items()
        }
    kwargs = {**special_tokens_map, **kwargs}

    rendered = []
    all_generation_indices = []
    continue_final_message_tag = "CONTINUE_FINAL_MESSAGE_TAG "
    for chat in conversations:
        if hasattr(chat, "messages"):
            # Indicates it's a Conversation object
            chat = chat.messages
        # The converted chat is a fresh copy, so the edits below never touch the caller's messages
        chat = _plain_template_data(chat)
        for message in chat:
            if not isinstance(message, dict):
                raise TypeError(f"Chat messages should be dicts, dataclasses or pydantic models, but got {message!r}.")
        chat = _alias_reasoning_fields(chat)
        if control_tokens is not None:
            for message in chat:
                # Some templates build control tokens out of roles, like `'<|' + message['role'] + '|>'`, so a role
                # counts as the template's text, as long as it is a single word like the roles that templates expect
                if isinstance(message.get("role"), str) and re.fullmatch(r"\w+", message["role"]):
                    message["role"] = _TemplateText(message["role"], ())
        if continue_final_message:
            if not chat:
                raise ValueError("continue_final_message is set but the conversation is empty!")
            continue_final_message = continue_final_message if isinstance(continue_final_message, str) else "content"

            if (final_message := chat[-1].get(continue_final_message)) is None:
                raise ValueError(
                    f'continue_final_message is set but the final message has no "{continue_final_message}" to continue!'
                )
            continue_fields = [continue_final_message]
            if continue_final_message in _REASONING_FIELDS:
                # Reasoning is copied under several names, so tag every copy and accept whichever one the template reads
                continue_fields = [field for field in _REASONING_FIELDS if chat[-1].get(field) == final_message]
            if not any(field in chat_template for field in continue_fields):
                raise ValueError(
                    f'continue_final_message is set to "{continue_final_message}" but this is not an accepted field in the chat_template'
                )

            elif isinstance(final_message, (list, tuple)):
                for content_block in reversed(final_message):
                    if "text" in content_block:
                        # Pick the last text block in the message (the first one we hit while iterating in reverse)
                        final_message = content_block["text"]
                        content_block["text"] = content_block["text"] + continue_final_message_tag
                        break
                else:
                    raise ValueError(
                        "continue_final_message is set but we could not find any text to continue in the final message!"
                    )
            else:
                for field in continue_fields:
                    chat[-1][field] = chat[-1][field] + continue_final_message_tag
        if return_assistant_tokens_mask:
            rendered_chat, generation_indices = _render_with_assistant_indices(
                compiled_template=compiled_template,
                messages=chat,
                tools=tool_schemas,
                documents=documents,
                add_generation_prompt=add_generation_prompt,
                **kwargs,
            )
            all_generation_indices.append(generation_indices)
        else:
            rendered_chat = compiled_template.render(
                messages=chat,
                tools=tool_schemas,
                documents=documents,
                add_generation_prompt=add_generation_prompt,
                **kwargs,
            )
        if continue_final_message:
            if (final_message.strip() not in rendered_chat) or (
                continue_final_message_tag.strip() not in rendered_chat
            ):
                raise ValueError(
                    "continue_final_message is set but the final message does not appear in the chat after "
                    "applying the chat template! This can happen if the chat template deletes portions of "
                    "the final message. Please verify the chat template and final message in your chat to "
                    "ensure they are compatible."
                    f"Final message to continue: {final_message.strip()}\nRendered chat:\n{rendered_chat}"
                )
            tag_loc = rendered_chat.rindex(continue_final_message_tag.strip())
            if rendered_chat[tag_loc : tag_loc + len(continue_final_message_tag)] == continue_final_message_tag:
                # The template preserves spacing, so things are simple
                rendered_chat = rendered_chat[:tag_loc]
            else:
                # The message has trailing spacing that was trimmed, so we must be more cautious
                rendered_chat = rendered_chat[:tag_loc].rstrip()
        if control_tokens is not None:
            rendered_chat = _escape_input_control_tokens(rendered_chat, _get_control_tokens_re(control_tokens))
        rendered.append(rendered_chat)

    return rendered, all_generation_indices


def is_valid_message(message):
    """
    Check that input is a valid message in a chat, namely a dict with "role" and "content" keys.
    """
    if not isinstance(message, dict):
        return False
    if not ("role" in message and "content" in message):
        return False
    return True


class Chat:
    """This class is intended to just be used internally for pipelines and not exposed to users. We convert chats
    to this format because the rest of the pipeline code tends to assume that lists of messages are
    actually a batch of samples rather than messages in the same conversation."""

    def __init__(self, messages: dict):
        for message in messages:
            if not is_valid_message(message):
                raise ValueError("When passing chat dicts as input, each dict must have a 'role' and 'content' key.")
        self.messages = messages
