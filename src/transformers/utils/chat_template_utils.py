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

import dataclasses
import inspect
import json
import re
import types
from collections.abc import Callable, Iterable, Mapping
from contextlib import contextmanager
from datetime import datetime
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
args_re = re.compile(r"\n\s*Args:\n\s*(.*?)[\n\s]*(Returns:|Raises:|\Z)", re.DOTALL)
# Splits the Args: block into individual arguments
args_split_re = re.compile(
    r"""
(?:^|\n)  # Match the start of the args block, or a newline
\s*(\w+):\s*  # Capture the argument name and strip spacing
(.*?)\s*  # Capture the argument description, which can span multiple lines, and strip trailing spacing
(?=\n\s*\w+:|\Z)  # Stop when you hit the next argument or the end of the block
""",
    re.DOTALL | re.VERBOSE,
)
# Extracts the Returns: block from the docstring, if present. Note that most chat templates ignore the return type/doc!
returns_re = re.compile(r"\n\s*Returns:\n\s*(.*?)[\n\s]*(Raises:|\Z)", re.DOTALL)


class TypeHintParsingException(Exception):
    """Exception raised for errors in parsing type hints to generate JSON schemas"""


class DocstringParsingException(Exception):
    """Exception raised for errors in parsing docstrings to generate JSON schemas"""


def _get_json_schema_type(param_type: type) -> dict[str, str]:
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
    return type_mapping.get(param_type, {"type": "object"})


def _parse_type_hint(hint: str) -> dict:
    origin = get_origin(hint)
    args = get_args(hint)

    if origin is None:
        try:
            return _get_json_schema_type(hint)
        except KeyError:
            raise TypeHintParsingException(
                "Couldn't parse this type hint, likely due to a custom class or object: ", hint
            )

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
        LITERAL_TYPES = (int, float, str, bool, type(None))
        args_types = []
        for arg in args:
            if type(arg) not in LITERAL_TYPES:
                raise TypeHintParsingException("Only the valid python literals can be listed in typing.Literal.")
            arg_type = _get_json_schema_type(type(arg)).get("type")
            if arg_type is not None and arg_type not in args_types:
                args_types.append(arg_type)
        return {
            "type": args_types.pop() if len(args_types) == 1 else list(args_types),
            "enum": list(args),
        }

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
    docstring_args = args_match.group(1).strip() if args_match else None
    returns = returns_match.group(1).strip() if returns_match else None

    # Parsing the arguments into a dictionary
    if docstring_args is not None:
        docstring_args = "\n".join([line for line in docstring_args.split("\n") if line.strip()])  # Remove blank lines
        matches = args_split_re.findall(docstring_args)
        args_dict = {match[0]: re.sub(r"\s*\n+\s*", " ", match[1].strip()) for match in matches}
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
    A string that knows which of its characters the chat template wrote: `trusted` has a 1 for each of them. The
    template's own strings are fully trusted and everything passed in is a plain `str`. String methods that aren't
    overridden here return a plain `str`, so text we lose track of always counts as input, never as the template's.
    """

    __slots__ = ("trusted",)

    def __new__(cls, text="", trusted=b""):
        self = super().__new__(cls, text)
        # Jinja sometimes rebuilds a string with `type(string)(...)`, which we can't trace, so that is all input
        self.trusted = trusted if len(trusted) == len(self) else bytes(len(self))
        return self

    def __str__(self):
        # Compiled templates call `str()` on everything they output
        return self

    def __add__(self, other):
        # Adding to Markup (from the `safe` filter) HTML-escapes, so we leave it to Markup, untraced
        if not isinstance(other, str) or hasattr(other, "__html__"):
            return NotImplemented
        return _concat_traced((self, other))

    def __radd__(self, other):
        return _concat_traced((other, self)) if isinstance(other, str) else NotImplemented

    def __mul__(self, count):
        return (
            _TemplateText(str.__mul__(self, count), self.trusted * count) if isinstance(count, int) else NotImplemented
        )

    __rmul__ = __mul__

    def __getitem__(self, key):
        trusted = self.trusted[key]
        return _TemplateText(str.__getitem__(self, key), bytes([trusted]) if isinstance(key, int) else trusted)

    def lstrip(self, chars=None):
        return self[len(self) - len(str.lstrip(self, chars)) :]

    def rstrip(self, chars=None):
        return self[: len(str.rstrip(self, chars))]

    def strip(self, chars=None):
        return self.lstrip(chars).rstrip(chars)

    def join(self, iterable):
        items = list(iterable)
        str.join(self, items)  # Raises like `str.join` for items that aren't strings
        return _concat_traced([piece for i, item in enumerate(items) for piece in ((self, item) if i else (item,))])

    def replace(self, old, new, count=-1):
        if not old:
            return str.replace(self, old, new, count)
        pieces, position = [], 0
        while count != 0 and (index := str.find(self, old, position)) != -1:
            pieces += [self[position:index], new]
            position, count = index + len(old), count - 1
        return _concat_traced([*pieces, self[position:]])


def _trusted(text: str) -> _TemplateText:
    return _TemplateText(text, b"\1" * len(text))


def _concat_traced(pieces: Iterable) -> _TemplateText:
    # `"".join()` that keeps track of the trusted characters. Jinja joins all template output with this, and `~` too
    pieces = [piece if isinstance(piece, str) else str(piece) for piece in pieces]
    trusted = [piece.trusted if isinstance(piece, _TemplateText) else bytes(len(piece)) for piece in pieces]
    return _TemplateText("".join(pieces), b"".join(trusted))


def _wrap_traced_str_format(jinja_env, value):
    # The sandbox swaps `str.format` for its own wrapper. Templates build control tokens like `'<|{}|>'.format('x')`,
    # so the result is trusted when the template wrote the format string and all of its arguments
    wrapper = ImmutableSandboxedEnvironment.wrap_str_format(jinja_env, value)
    if wrapper is None:
        return None

    def traced_wrapper(*args, **kwargs):
        result = wrapper(*args, **kwargs)
        strings = [value.__self__, *args, *kwargs.values()]
        return _trusted(result) if all(isinstance(x, _TemplateText) and all(x.trusted) for x in strings) else result

    return traced_wrapper


@lru_cache
@no_type_check
def _compile_traced_jinja_template(chat_template):
    """
    Compile a chat template so that it renders `_TemplateText`: every string literal in the template is trusted
    `_TemplateText`, and the output join (`concat`), `~` and `format` keep track of it.
    """
    jinja_env = _cached_compile_jinja_template(chat_template).environment
    jinja_env.concat = _concat_traced
    jinja_env.wrap_str_format = partial(_wrap_traced_str_format, jinja_env)
    jinja_env.globals["_concat_traced"] = _concat_traced
    literals = {}

    class LiteralTracer(jinja2.visitor.NodeTransformer):
        def trace(self, text, node):
            # Each literal becomes a variable that holds it as trusted `_TemplateText`
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
    jinja_env.globals.update({name: _trusted(text) for text, name in literals.items()})
    return jinja2.Template.from_code(jinja_env, jinja_env.compile(template_ast), jinja_env.make_globals(None))


def _escape_input_control_tokens(rendered_chat: _TemplateText, control_tokens_re: re.Pattern) -> str:
    # Escape the control tokens that the template didn't write all of itself, so tokens that inputs complete or that
    # only appear once the template joins two inputs are escaped as well
    text, pieces, position = str.__str__(rendered_chat), [], 0
    for match in control_tokens_re.finditer(text):
        start, end = match.span()
        if all(rendered_chat.trusted[start:end]):
            continue
        token = match.group()
        if any(ord(char) > 0xFFFD for char in token):
            raise ValueError(f"Can't sanitize the control token {token!r}: it has characters outside the BMP.")
        escaped = "".join(chr(ord(char) + _ESCAPED_CHAR_OFFSET) for char in token)
        pieces += [text[position:start], f"{_ESCAPE_START}{escaped}{_ESCAPE_END}"]
        position = end
    return "".join([*pieces, text[position:]])


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


def _encode_stretch(tokenizer, text: str, start: int, end: int):
    # Encode `text[start:end]` as plain text, with the steps of the backend tokenizer that come after it has found the
    # added tokens. We run them on the stretch where it is in the text, and not on a copy of it, because some steps
    # treat the start of the text differently (like SentencePiece's prefix space)
    from tokenizers import PreTokenizedString

    backend = tokenizer.backend_tokenizer
    if start == end:
        return backend.encode("", add_special_tokens=False)
    pretokenized = PreTokenizedString(text)
    pretokenized.split(lambda _, normalized: [normalized[start:end]])
    if backend.normalizer is not None:
        pretokenized.normalize(backend.normalizer.normalize)
    if backend.pre_tokenizer is not None:
        backend.pre_tokenizer.pre_tokenize(pretokenized)
    pretokenized.tokenize(backend.model.tokenize)
    return pretokenized.to_encoding()


def encode_escaped_text(tokenizer, text: str, add_special_tokens: bool):
    """
    Encode a chat rendered with `sanitize_control_tokens=True` into a backend `Encoding`. The added tokens that the
    tokenizer finds are kept, except the ones that overlap a control token escaped by `_escape_input_control_tokens`:
    the stretch of text between the kept tokens is encoded again without added tokens, like the tokenizer would if
    those weren't added tokens. Truncation, special tokens and padding are then applied by the backend as usual.
    """
    from tokenizers import Encoding

    text, escaped_spans = _parse_escaped_control_tokens(text)
    backend, added_tokens = tokenizer.backend_tokenizer, tokenizer.added_tokens_decoder
    pieces, dropped, stretch_ids, start = [], set(), set(), 0
    encoding = backend.encode(text, add_special_tokens=False)
    for token_id, (token_start, token_end) in zip(encoding.ids, encoding.offsets):
        # The model can output ids like the unknown token's for plain text, so we check the text too
        if token_id not in added_tokens or added_tokens[token_id].content not in text[token_start:token_end]:
            continue
        if any(span_start < token_end and token_start < span_end for span_start, span_end in escaped_spans):
            dropped.add(token_id)
            continue
        token = backend.encode(added_tokens[token_id].content, add_special_tokens=False)
        pieces += [_encode_stretch(tokenizer, text, start, token_start), token]
        stretch_ids.update(pieces[-2].ids)
        start = token_end
    pieces.append(_encode_stretch(tokenizer, text, start, len(text)))
    stretch_ids.update(pieces[-1].ids)
    if dropped & stretch_ids:
        token = added_tokens[min(dropped & stretch_ids)].content
        raise ValueError(f"Can't sanitize the control token {token!r}: the tokenizer encodes plain text as it too.")
    return backend.post_process(Encoding.merge(pieces, growing_offsets=False), None, add_special_tokens)


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
            they came from messages, tools, documents or `kwargs`. the tokenizer then encodes them as plain
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
            name: _trusted(token) if isinstance(token, str) else token for name, token in special_tokens_map.items()
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
                    message["role"] = _trusted(message["role"])
        if continue_final_message:
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
