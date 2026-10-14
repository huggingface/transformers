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
import inspect
import json
import re
import types
from collections.abc import Callable
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime
from functools import lru_cache
from inspect import isfunction
from typing import Any, Literal, Union, get_args, get_origin, get_type_hints, no_type_check

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
    from jinja2.ext import Extension
    from jinja2.sandbox import ImmutableSandboxedEnvironment
else:
    jinja2 = None

if is_vision_available():
    from PIL.Image import Image

ChatType = list[dict[str, Any]]

# Private-use characters that `mark_control_tokens` wraps around control tokens in user input
_CONTROL_TOKEN_START = "\U000f0000"
_CONTROL_TOKEN_END = "\U000f0001"
# Matches a marked control token, or a lone marker left by string methods in the chat template
_MARKED_TOKEN_RE = re.compile(
    f"{_CONTROL_TOKEN_START}([^{_CONTROL_TOKEN_START}{_CONTROL_TOKEN_END}]*){_CONTROL_TOKEN_END}"
    f"|[{_CONTROL_TOKEN_START}{_CONTROL_TOKEN_END}]"
)
# `escape_marked_control_tokens` shifts the characters of marked tokens by this offset, into the plane 16 private-use
# area, and `_parse_escaped_control_tokens` finds them with this regex, which also matches lone markers
_ESCAPED_CHAR_OFFSET = 0x100000
_ESCAPED_TOKEN_RE = re.compile(
    f"{_CONTROL_TOKEN_START}([\U00100000-\U0010fffd]*){_CONTROL_TOKEN_END}"
    f"|[{_CONTROL_TOKEN_START}{_CONTROL_TOKEN_END}]"
)


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
    """
    Return a regex that matches any of `control_tokens`. Longer tokens are matched first, so a token takes priority over
    the tokens it contains.
    """
    longest_first = sorted(control_tokens, key=len, reverse=True)
    return re.compile("|".join(map(re.escape, longest_first)) or "(?!)")  # (?!) never matches


def mark_control_tokens(obj: Any, control_tokens_re: re.Pattern) -> Any:
    """
    Wrap the control tokens in user-supplied strings with private-use markers, so that `escape_marked_control_tokens`
    can tell them apart from the control tokens the chat template writes itself.

    Args:
        obj: A string, or a list or dict (like messages, tools or documents) whose values are marked recursively. Dict
            keys aren't marked.
        control_tokens_re: A regex that matches any control token, from `_get_control_tokens_re`.
    """
    if isinstance(obj, str):
        # Drop any markers already in the string first, so they can't be used to forge a marked token
        obj = obj.replace(_CONTROL_TOKEN_START, "").replace(_CONTROL_TOKEN_END, "")
        return control_tokens_re.sub(f"{_CONTROL_TOKEN_START}\\g<0>{_CONTROL_TOKEN_END}", obj)
    if isinstance(obj, dict):
        return {key: mark_control_tokens(value, control_tokens_re) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [mark_control_tokens(value, control_tokens_re) for value in obj]
    return obj


def _encode_chat_piece(tokenizer, piece: str, newline_ids: list[int]) -> list[int]:
    """Encode a piece from the middle of a chat as plain text, without matching any added tokens."""
    encode = tokenizer._encode_without_added_tokens
    # Encode after a newline, so SentencePiece doesn't add the prefix space it adds at the start of a string, then
    # remove the newline again
    # FIXME: This is a trick to avoid SentencePiece hacks, so we can remove it if we get tokenizer support
    #        for encoding with an offset, or something similar
    piece_ids = encode("\n" + piece)
    if piece_ids[: len(newline_ids)] == newline_ids:
        return piece_ids[len(newline_ids) :]
    return encode(piece)


def escape_marked_control_tokens(chat: str) -> str:
    """
    Escape the control tokens marked by `mark_control_tokens` in a rendered chat, by shifting their characters into a
    private-use area. Processors search the chat for their multimodal tokens, like `<image>`, before tokenizing it, and
    this stops them from finding the ones in user input. Tokenizers called with `escaped_control_tokens=True` shift the
    characters back and encode the tokens as plain text. Lone markers are kept, so tokens still can't match across them.
    """
    return _MARKED_TOKEN_RE.sub(_escape_marked_control_token, chat)


def _escape_marked_control_token(match: re.Match) -> str:
    token = match.group(1)
    if token is None:
        return match.group(0)  # A lone marker
    if any(ord(char) > 0xFFFD for char in token):
        raise ValueError(f"Can't sanitize {token!r}: only characters in the Basic Multilingual Plane can be escaped")
    escaped = "".join(chr(ord(char) + _ESCAPED_CHAR_OFFSET) for char in token)
    return f"{_CONTROL_TOKEN_START}{escaped}{_CONTROL_TOKEN_END}"


def _contains_escaped_control_tokens(text: Any) -> bool:
    """Whether `text`, a string or a list of strings, contains markers from `escape_marked_control_tokens`."""
    texts = [text] if isinstance(text, str) else text
    return isinstance(texts, (list, tuple)) and any(
        isinstance(t, str) and (_CONTROL_TOKEN_START in t or _CONTROL_TOKEN_END in t) for t in texts
    )


def _parse_escaped_control_tokens(text: str) -> tuple[str, list[tuple[int, int]]]:
    """
    Undo `escape_marked_control_tokens`. Returns the text with the escaped tokens restored and the markers removed, and
    the character spans of the escaped tokens in it. Lone markers give empty spans, so tokens still can't match across
    them.
    """
    # Splitting on the escaped tokens gives [text, escaped token, text, escaped token, ..., text], with None in place
    # of the token for a lone marker
    pieces = _ESCAPED_TOKEN_RE.split(text)
    restored, escaped_spans = pieces[0], []
    for escaped, piece in zip(pieces[1::2], pieces[2::2]):
        token = "".join(chr(ord(char) - _ESCAPED_CHAR_OFFSET) for char in escaped or "")
        escaped_spans.append((len(restored), len(restored) + len(token)))
        restored += token + piece
    return restored, escaped_spans


def _encode_escaped_spans_as_text(
    tokenizer,
    text: str,
    escaped_spans: list[tuple[int, int]],
    input_ids: list[int],
    offsets: list[tuple[int, int]],
    added_token_ids: set[int],
) -> tuple[list[int], list[int | None]]:
    """
    Fix the tokens of a text restored by `_parse_escaped_control_tokens` and then encoded normally, so that its escaped
    tokens are encoded as plain text. Added tokens that overlap an escaped span are dropped (an empty span, from a lone
    marker, overlaps the tokens that cross it), and the text between the added tokens on either side of them is encoded
    again without added tokens. The escaped tokens are then tokenized as if they weren't added tokens, and all other
    tokens are unchanged.

    Args:
        tokenizer: The tokenizer that encoded the text.
        text: The restored text.
        escaped_spans: The character spans of the escaped tokens and lone markers in `text`.
        input_ids: The token ids of `text`, encoded normally.
        offsets: The character span of each token in `text`.
        added_token_ids: The ids of the tokenizer's added tokens.

    Returns:
        The new token ids, and for each of them the index of the token it was copied from, or `None` if it was encoded
        again.
    """
    # The spans are sorted and don't overlap, so a token overlaps one of them exactly when it overlaps the first span
    # that ends after the token starts. Special tokens added around the text, like BOS, have empty spans and are always
    # kept
    span_starts = [span_start for span_start, _ in escaped_spans]
    span_ends = [span_end for _, span_end in escaped_spans]
    dropped = [
        token_id in added_token_ids
        and start != end
        and (span := bisect.bisect_right(span_ends, start)) < len(span_ends)
        and span_starts[span] < end
        for token_id, (start, end) in zip(input_ids, offsets)
    ]
    if not any(dropped):
        return list(input_ids), list(range(len(input_ids)))
    kept_added = [i for i, token_id in enumerate(input_ids) if token_id in added_token_ids and not dropped[i]]
    newline_ids = tokenizer.encode("\n", add_special_tokens=False)

    new_ids, origins = [], []
    run_start = previous_end = 0
    for index in [*kept_added, len(input_ids)]:
        # The tokens in input_ids[run_start:index] are between two kept added tokens, so if any of them were dropped,
        # the text between those two added tokens is encoded again
        if any(dropped[run_start:index]):
            if index < len(input_ids) and offsets[index][0] != offsets[index][1]:
                next_start = offsets[index][0]
            else:
                next_start = len(text)
            piece = text[previous_end:next_start]
            if previous_end == 0:
                piece_ids = tokenizer._encode_without_added_tokens(piece)
            else:
                piece_ids = _encode_chat_piece(tokenizer, piece, newline_ids)
            for token_id in {input_ids[i] for i in range(run_start, index) if dropped[i]} & set(piece_ids):
                token = tokenizer.convert_ids_to_tokens(token_id)
                raise ValueError(f"Can't sanitize {token!r}: the tokenizer produces it from plain text")
            new_ids += piece_ids
            origins += [None] * len(piece_ids)
        else:
            new_ids += input_ids[run_start:index]
            origins += range(run_start, index)
        if index < len(input_ids):
            new_ids.append(input_ids[index])
            origins.append(index)
            previous_end = max(previous_end, offsets[index][1])
        run_start = index + 1
    return new_ids, origins


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

    if version.parse(jinja2.__version__) < version.parse("3.1.0"):
        raise ImportError(
            f"apply_chat_template requires jinja2>=3.1.0 to be installed. Your version is {jinja2.__version__}."
        )

    def raise_exception(message):
        raise jinja2.exceptions.TemplateError(message)

    def tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
        # We override the built-in tojson filter because Jinja's default filter escapes HTML characters
        # We also expose some options like custom indents and separators
        return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)

    def strftime_now(format):
        return datetime.now().strftime(format)

    jinja_env = ImmutableSandboxedEnvironment(
        trim_blocks=True, lstrip_blocks=True, extensions=[AssistantTracker, jinja2.ext.loopcontrols]
    )
    jinja_env.filters["tojson"] = tojson
    jinja_env.globals["raise_exception"] = raise_exception
    jinja_env.globals["strftime_now"] = strftime_now
    return jinja_env.from_string(chat_template)


def render_jinja_template(
    conversations: list[ChatType],
    tools: list[dict | Callable] | None = None,
    documents: ChatType | None = None,
    chat_template: str | None = None,
    return_assistant_tokens_mask: bool = False,
    continue_final_message: bool | str = False,
    add_generation_prompt: bool = False,
    control_tokens: tuple[str, ...] | None = None,
    **kwargs,
) -> str:
    if return_assistant_tokens_mask and not re.search(r"\{\%-?\s*generation\s*-?\%\}", chat_template):
        logger.warning_once(
            "return_assistant_tokens_mask==True but chat template does not contain `{% generation %}` keyword."
        )

    # Compilation function uses a cache to avoid recompiling the same template
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

    # With `sanitize_control_tokens`, the control tokens in the user's strings are wrapped with private markers, and
    # escaped after rendering
    control_tokens_re = _get_control_tokens_re(control_tokens) if control_tokens is not None else None
    if control_tokens_re is not None:
        tool_schemas = mark_control_tokens(tool_schemas, control_tokens_re)
        documents = mark_control_tokens(documents, control_tokens_re)

    rendered = []
    all_generation_indices = []
    continue_final_message_tag = "CONTINUE_FINAL_MESSAGE_TAG "
    for chat in conversations:
        if hasattr(chat, "messages"):
            # Indicates it's a Conversation object
            chat = chat.messages
        if control_tokens_re is not None:
            chat = mark_control_tokens(chat, control_tokens_re)
        if continue_final_message:
            chat = deepcopy(chat)
            continue_final_message = continue_final_message if isinstance(continue_final_message, str) else "content"

            if (final_message := chat[-1].get(continue_final_message)) is None:
                raise ValueError(
                    f'continue_final_message is set but the final message has no "{continue_final_message}" to continue!'
                )
            if continue_final_message not in chat_template:
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
                chat[-1][continue_final_message] = chat[-1][continue_final_message] + continue_final_message_tag
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
        if control_tokens_re is not None:
            rendered_chat = escape_marked_control_tokens(rendered_chat)
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
