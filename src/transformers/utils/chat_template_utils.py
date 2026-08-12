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

import inspect
import json
import re
import secrets
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
        elif all("type" in subtype and isinstance(subtype["type"], str) for subtype in subtypes):
            # A union of basic types can be expressed as a list in the schema
            return_dict = {"type": sorted([subtype["type"] for subtype in subtypes])}
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

    rendered = []
    all_generation_indices = []
    continue_final_message_tag = "CONTINUE_FINAL_MESSAGE_TAG "
    for chat in conversations:
        if hasattr(chat, "messages"):
            # Indicates it's a Conversation object
            chat = chat.messages
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


@lru_cache
def _compile_special_token_pattern(all_special_tokens: tuple[str, ...]) -> re.Pattern | None:
    if not all_special_tokens:
        return None
    # Match longest-first to catch cases where one token is a substring of another
    escaped = sorted((re.escape(token) for token in all_special_tokens if token), key=len, reverse=True)
    return re.compile("|".join(escaped))


def sanitize_chat_input(chat_input: Any, all_special_tokens: list[str], substitutions: dict[str, str]) -> Any:
    """Sanitize special tokens out of chat inputs by replacing each occurrence with a unique placeholder,
    recorded in `substitutions` as `{placeholder: original_text}`. After the chat template has been rendered,
    the placeholders can be resolved back to the original text with `resolve_sanitization_placeholders` and
    encoded as ordinary (non-special) tokens."""
    return _sanitize_chat_input(chat_input, _compile_special_token_pattern(tuple(all_special_tokens)), substitutions)


def _sanitize_chat_input(chat_input: Any, pattern: re.Pattern | None, substitutions: dict[str, str]) -> Any:
    if pattern is None:
        return chat_input
    if hasattr(chat_input, "messages"):
        # catches Chat objects
        chat_input.messages = _sanitize_chat_input(chat_input.messages, pattern, substitutions)
        return chat_input
    if isinstance(chat_input, dict):
        return {key: _sanitize_chat_input(value, pattern, substitutions) for key, value in chat_input.items()}
    elif isinstance(chat_input, list):
        return [_sanitize_chat_input(item, pattern, substitutions) for item in chat_input]
    elif isinstance(chat_input, tuple):
        return tuple(_sanitize_chat_input(item, pattern, substitutions) for item in chat_input)
    elif isinstance(chat_input, str):

        def replacement(match: re.Match) -> str:
            # Pure-digit placeholders survive template filters like tojson/trim/upper unchanged
            placeholder = str(int.from_bytes(secrets.token_bytes(10), "big"))
            substitutions[placeholder] = match.group(0)
            return placeholder

        # Substitution cannot splice fragments of nested tokens into new ones (the placeholder physically
        # interrupts them), but we re-check to fixpoint anyway as cheap insurance
        while pattern.search(chat_input):
            chat_input = pattern.sub(replacement, chat_input)
        return chat_input
    else:
        return chat_input


def sanitization_special_tokens(tokenizer) -> list[str]:
    """The token strings protected by the `sanitize_special_tokens` argument of `apply_chat_template`: every
    added token flagged as special, plus the named special tokens. Note that this is a superset of
    `all_special_tokens`, which only contains named special tokens and `additional_special_tokens` — chat
    control tokens like `<|start_header_id|>` are frequently special added tokens without a name."""
    protected = {token.content for token in tokenizer.added_tokens_decoder.values() if token.special}
    protected.update(str(token) for token in tokenizer.all_special_tokens)
    return sorted(protected)


def resolve_sanitization_placeholders(
    rendered: str, substitutions: dict[str, str]
) -> tuple[str, list[tuple[int, int]], list[tuple[int, int]], set[str]]:
    """Restore the original special-token text at the placeholders inserted by `sanitize_chat_input` in a
    rendered chat string. Returns `(final, untrusted_spans, shifts, seen)`:

    - `final`: the final chat string, with every placeholder replaced by its original text.
    - `untrusted_spans`: `(start, end)` character spans in `final` covering the restored text, i.e. the
      regions that came from untrusted chat input and must not encode as control tokens.
    - `shifts`: `(position, delta)` pairs recording how character indices in `rendered` map to indices in
      `final` (see `shift_sanitized_index`), since placeholders and their originals differ in length.
    - `seen`: the placeholder keys that were found, so callers can detect placeholders that a template mangled.
    """
    pattern = re.compile("|".join(map(re.escape, substitutions)))
    pieces, untrusted_spans, shifts, seen = [], [], [], set()
    last = final_length = 0
    for match in pattern.finditer(rendered):
        original = substitutions[match.group(0)]
        seen.add(match.group(0))
        pieces.append(rendered[last : match.start()])
        final_length += match.start() - last
        untrusted_spans.append((final_length, final_length + len(original)))
        pieces.append(original)
        final_length += len(original)
        shifts.append((match.end(), len(original) - (match.end() - match.start())))
        last = match.end()
    pieces.append(rendered[last:])
    return "".join(pieces), untrusted_spans, shifts, seen


def shift_sanitized_index(index: int, shifts: list[tuple[int, int]]) -> int:
    """Translate a character index in a placeholder-bearing rendered chat string (as produced by rendering
    inputs processed by `sanitize_chat_input`) to the corresponding index in the final string in which the
    placeholders have been resolved back to their original text."""
    return index + sum(delta for position, delta in shifts if position <= index)


def split_at_trusted_special_tokens(
    final: str,
    untrusted_spans: list[tuple[int, int]],
    protected_tokens: list[str],
    token_flags: dict[str, tuple[bool, bool]],
) -> list[tuple[str, str, int, int]]:
    """Split a sanitized chat string at the special tokens the chat template itself emitted. Returns a list
    of `(kind, text, char_start, char_end)` parts whose spans tile `final`:

    - `("special", token, start, end)`: a trusted special-token occurrence, to be converted directly to its
      token id. The span may be wider than the token itself when its `lstrip`/`rstrip` flags (looked up in
      `token_flags`) absorb adjacent whitespace, mirroring the tokenizer's own added-token matching.
    - `("text", text, start, end)`: everything else, to be encoded *without* special-token matching (as with
      `split_special_tokens=True`). Untrusted special-token text stays inline in these parts, so it is
      tokenized in context as ordinary text rather than as control tokens.

    A special-token match only counts as trusted when it lies entirely outside `untrusted_spans`: a match that
    overlaps untrusted text, including one spliced together from trusted and untrusted characters, stays
    inline as ordinary text.
    """
    pattern = _compile_special_token_pattern(tuple(protected_tokens))
    parts = []
    last = 0
    for match in pattern.finditer(final) if pattern is not None else ():
        if match.start() < last:
            # Inside whitespace already absorbed by the previous token's rstrip
            continue
        if any(match.start() < span_end and span_start < match.end() for span_start, span_end in untrusted_spans):
            continue
        start, end = match.start(), match.end()
        lstrip, rstrip = token_flags.get(match.group(0), (False, False))
        while lstrip and start > last and final[start - 1].isspace():
            start -= 1
        while rstrip and end < len(final) and final[end].isspace():
            end += 1
        if last < start:
            parts.append(("text", final[last:start], last, start))
        parts.append(("special", match.group(0), start, end))
        last = end
    if last < len(final):
        parts.append(("text", final[last:], last, len(final)))
    return parts


class SanitizedTokenMap:
    """Maps character indices in a sanitized chat string to token indices in its sanitized encoding, standing
    in for `BatchEncoding.char_to_token` (which has no offsets for an encoding assembled from parts).
    `token_spans` holds one `(char_start, char_end)` span per token; `token_shift` accounts for tokens
    prepended to the sequence after the spans were recorded (i.e. left-padding)."""

    def __init__(self, token_spans: list[tuple[int, int]], token_shift: int = 0):
        self.token_spans = token_spans
        self.token_shift = token_shift

    def start_token(self, char: int) -> int | None:
        """The first token that covers `char`, or begins after it (for characters the tokenization skipped)."""
        for index, (start, end) in enumerate(self.token_spans):
            if end > char and end > start:
                return index + self.token_shift
        return None

    def end_token(self, char: int) -> int | None:
        """The last token that covers `char`, or ends before it (for characters the tokenization skipped)."""
        for index in range(len(self.token_spans) - 1, -1, -1):
            start, end = self.token_spans[index]
            if start <= char and end > start:
                return index + self.token_shift
        return None


def encode_sanitized_chats(
    tokenizer,
    batch_parts: list[list[tuple[str, str, int, int]]],
    batch_untrusted_spans: list[list[tuple[int, int]]],
    padding=False,
    truncation: bool = False,
    max_length: int | None = None,
    return_tensors=None,
    return_token_maps: bool = False,
    **tokenizer_kwargs,
):
    """Encode chats that have been split into parts by `split_at_trusted_special_tokens`, using only the
    public tokenizer API: `"special"` parts are converted directly to their token ids, while `"text"` parts
    are encoded without special-token matching (as with `split_special_tokens=True`), so that special-token
    text from untrusted chat input can never act as control tokens.

    Since trusted special tokens are split points in the tokenizer's own pipeline as well, the concatenated
    ids match encoding the full chat string directly, except that some SentencePiece tokenizers
    (`legacy=False`) only prepend their dummy space to the first split of a call and may therefore gain an
    extra space token where a text part directly follows a special token.

    Sanitization is all-or-nothing: if untrusted special-token text would still produce special token ids
    even when encoded as ordinary text (some vocabularies can assemble special tokens from ordinary pieces),
    a `ValueError` is raised rather than deleting or rewriting the offending text.

    Returns `(batch_encoding, token_maps)`, where `token_maps` (one `SanitizedTokenMap` per chat, or `None`
    unless `return_token_maps=True`) supports the character-to-token lookups needed for assistant masks."""
    verbose = tokenizer_kwargs.pop("verbose", True)
    padding_side = tokenizer_kwargs.pop("padding_side", None)
    pad_to_multiple_of = tokenizer_kwargs.pop("pad_to_multiple_of", None)
    return_attention_mask = tokenizer_kwargs.pop("return_attention_mask", None)
    if tokenizer_kwargs:
        # The encoding is assembled from per-part calls, so arbitrary per-call kwargs don't compose. Reject
        # them rather than let a security-relevant call silently diverge from what was asked for.
        raise ValueError(
            f"`sanitize_special_tokens=True` does not support these tokenizer kwargs: {sorted(tokenizer_kwargs)}"
        )
    padding_strategy, truncation_strategy, max_length, _ = tokenizer._get_padding_truncation_strategies(
        padding=padding,
        truncation=truncation,
        max_length=max_length,
        pad_to_multiple_of=pad_to_multiple_of,
        verbose=verbose,
    )

    # Encode every text part across the batch in a single call, without special-token matching. Offsets are
    # requested whenever the backend can produce them: they drive both the assistant-mask token maps and the
    # precise protected-id check below.
    texts = [text for parts in batch_parts for kind, text, _, _ in parts if kind == "text"]
    with_offsets = getattr(tokenizer, "backend", None) == "tokenizers" or return_token_maps
    encoded_texts = None
    if texts:
        try:
            encoded_texts = tokenizer(
                texts,
                add_special_tokens=False,
                split_special_tokens=True,
                return_offsets_mapping=with_offsets,
                return_attention_mask=False,
            )
        except (TypeError, ValueError, NotImplementedError) as exc:
            raise NotImplementedError(
                f"`sanitize_special_tokens` requires standard text encoding, which {type(tokenizer).__name__} "
                "does not support."
            ) from exc
    text_ids = iter(encoded_texts["input_ids"]) if encoded_texts is not None else iter(())
    text_offsets = iter(encoded_texts["offset_mapping"]) if encoded_texts is not None and with_offsets else None

    protected_ids = {token_id for token_id, token in tokenizer.added_tokens_decoder.items() if token.special}
    protected_ids.update(tokenizer.all_special_ids)
    # Out-of-vocabulary text legitimately encodes to the unknown token, exactly as it would without
    # sanitization, so it must not be treated as a protected control token
    protected_ids.discard(tokenizer.unk_token_id)

    batch_ids, batch_spans = [], []
    for parts, untrusted_spans in zip(batch_parts, batch_untrusted_spans):
        ids, spans = [], []
        for kind, text, char_start, char_end in parts:
            if kind == "special":
                ids.append(tokenizer.convert_tokens_to_ids(text))
                spans.append((char_start, char_end))
                continue
            part_ids = next(text_ids)
            part_offsets = next(text_offsets) if text_offsets is not None else None
            # A protected id must never come from untrusted text, which is possible when the vocabulary can
            # assemble a special token out of ordinary pieces. Protected ids arising from trusted text are
            # native tokenizer behavior (e.g. wav2vec2's word delimiter) and pass through. With offsets each
            # id is attributed to its characters exactly; without them (Python backends), any protected id in
            # a part that overlaps untrusted text is conservatively treated as unsafe.
            if part_offsets is not None:
                unsafe = any(
                    token_id in protected_ids
                    and any(
                        char_start + start < span_end and span_start < char_start + end
                        for span_start, span_end in untrusted_spans
                    )
                    for token_id, (start, end) in zip(part_ids, part_offsets)
                )
            else:
                unsafe = any(
                    char_start < span_end and span_start < char_end for span_start, span_end in untrusted_spans
                ) and bool(protected_ids.intersection(part_ids))
            if unsafe:
                raise ValueError(
                    "`sanitize_special_tokens=True` cannot safely encode these chat inputs: special-token "
                    "text from the inputs would still produce special token ids even when encoded as "
                    "ordinary text. Rather than delete or rewrite the offending text, sanitization refuses "
                    "to encode the chat."
                )
            ids.extend(part_ids)
            if return_token_maps and part_offsets is not None:
                spans.extend((char_start + start, char_start + end) for start, end in part_offsets)
        batch_ids.append(ids)
        batch_spans.append(spans)

    if truncation_strategy.value != "do_not_truncate" and max_length is not None:
        for i, ids in enumerate(batch_ids):
            if len(ids) > max_length:
                keep = slice(-max_length, None) if tokenizer.truncation_side == "left" else slice(None, max_length)
                batch_ids[i] = ids[keep]
                batch_spans[i] = batch_spans[i][keep]
    for ids in batch_ids:
        tokenizer._eventual_warn_about_too_long_sequence(ids, max_length, verbose)

    encoded_inputs = {"input_ids": batch_ids}
    if "token_type_ids" in tokenizer.model_input_names:
        encoded_inputs["token_type_ids"] = [[0] * len(ids) for ids in batch_ids]
    batch_encoding = tokenizer.pad(
        encoded_inputs,
        padding=padding_strategy.value,
        max_length=max_length,
        pad_to_multiple_of=pad_to_multiple_of,
        padding_side=padding_side,
        return_attention_mask=return_attention_mask,
        return_tensors=return_tensors,
        verbose=verbose,
    )

    token_maps = None
    if return_token_maps:
        token_maps = []
        for i, spans in enumerate(batch_spans):
            padded_length = len(batch_encoding["input_ids"][i])
            left_pad = padded_length - len(spans) if (padding_side or tokenizer.padding_side) == "left" else 0
            token_maps.append(SanitizedTokenMap(spans, left_pad))
    return batch_encoding, token_maps
