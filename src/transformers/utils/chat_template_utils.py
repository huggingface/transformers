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
import inspect
import json
import re
import types
import uuid
from collections.abc import Callable
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime
from functools import lru_cache
from inspect import isfunction
from typing import Any, Literal, Union, get_args, get_origin, get_type_hints, no_type_check

from packaging import version

from . import logging
from .generic import to_py_obj
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
) -> tuple[list[str], list[list[tuple[int, int]]]]:
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


def escape_special_tokens(
    data: Any, special_tokens: list[str], nonce: str | None = None
) -> tuple[Any, dict[str, str]]:
    """
    Escapes special tokens in user data (conversations, tools, documents) to prevent special token injection attacks.

    Args:
        data (`Any`): The conversation messages, tools, or documents structure to escape.
        special_tokens (`list[str]`): List of special token strings to escape.
        nonce (`str`, *optional*): An optional pre-generated random nonce string.

    Returns:
        `tuple[Any, dict[str, str]]`: A tuple containing the escaped data structure and a mapping of
        placeholder strings back to the original special token strings.
    """
    if not special_tokens or not data:
        return data, {}

    valid_tokens = sorted(
        [t for t in special_tokens if isinstance(t, str) and t],
        key=len,
        reverse=True,
    )
    if not valid_tokens:
        return data, {}

    if nonce is None:
        nonce = uuid.uuid4().hex
    prefix = f"__HF_ESC_{nonce}_"

    token_to_placeholder = {}
    for i, token in enumerate(valid_tokens):
        token_to_placeholder[token] = f"{prefix}{i}__"

    # Only the placeholders that were actually substituted are reported back, so that callers can
    # take the plain (and much cheaper) code path when the payload contains no special token at all.
    placeholder_to_token: dict[str, str] = {}

    def _escape_val(val: Any) -> Any:
        if isinstance(val, str):
            res = val
            if prefix in res:
                res = res.replace(prefix, f"__SAN_{nonce}_")
            for token, placeholder in token_to_placeholder.items():
                if token in res:
                    res = res.replace(token, placeholder)
                    placeholder_to_token[placeholder] = token
            return res
        elif hasattr(val, "messages"):
            escaped_messages = _escape_val(val.messages)
            try:
                val_copy = copy.copy(val)
                val_copy.messages = escaped_messages
                return val_copy
            except (AttributeError, TypeError):
                return Chat(escaped_messages)
        elif isinstance(val, dict):
            return {k: _escape_val(v) for k, v in val.items()}
        elif isinstance(val, list):
            return [_escape_val(v) for v in val]
        elif isinstance(val, tuple):
            return tuple(_escape_val(v) for v in val)
        return val

    escaped = _escape_val(data)
    return escaped, placeholder_to_token


def unpack_special_tokens(rendered_text: str | list[str], placeholder_to_token: dict[str, str]) -> str | list[str]:
    """
    Unpacks escaped special token placeholders back to their original literal special token string representations.

    Args:
        rendered_text (`Union[str, list[str]]`): Rendered Jinja template string or list of rendered strings.
        placeholder_to_token (`dict[str, str]`): Mapping from placeholder strings to original special tokens.

    Returns:
        `Union[str, list[str]]`: Unpacked rendered text with original special token strings restored.
    """
    if not placeholder_to_token or not rendered_text:
        return rendered_text

    if isinstance(rendered_text, str):
        res = rendered_text
        for placeholder, token in placeholder_to_token.items():
            res = res.replace(placeholder, token)
        return res
    elif isinstance(rendered_text, list):
        return [unpack_special_tokens(item, placeholder_to_token) for item in rendered_text]
    return rendered_text


def split_on_placeholders(
    rendered_text: str, placeholder_to_token: dict[str, str]
) -> list[tuple[str, bool, int, int]]:
    """
    Splits a rendered template string into segments on escaped special token placeholders.

    Args:
        rendered_text (`str`): Rendered Jinja template string, possibly containing placeholders.
        placeholder_to_token (`dict[str, str]`): Mapping from placeholder strings to original special tokens.

    Returns:
        `list[tuple[str, bool, int, int]]`: A list of `(text, is_user_content, start, end)` segments, where `start`
        and `end` are the character offsets of the segment inside `rendered_text` (i.e. offsets of the placeholder
        itself for user content). Segments with `is_user_content=True` contain the original special token text that
        came from user input and must be encoded with `split_special_tokens=True` so that they cannot become
        control tokens. Concatenating all segment texts reproduces the rendered text with placeholders substituted
        back.
    """
    if not placeholder_to_token or not rendered_text:
        return [(rendered_text, False, 0, len(rendered_text))]

    # Longest first so that no placeholder is a prefix of another during matching
    pattern = re.compile("|".join(re.escape(p) for p in sorted(placeholder_to_token, key=len, reverse=True)))

    segments = []
    pos = 0
    for match in pattern.finditer(rendered_text):
        if match.start() > pos:
            segments.append((rendered_text[pos : match.start()], False, pos, match.start()))
        segments.append((placeholder_to_token[match.group(0)], True, match.start(), match.end()))
        pos = match.end()
    if pos < len(rendered_text):
        segments.append((rendered_text[pos:], False, pos, len(rendered_text)))
    return segments or [("", False, 0, 0)]


def neutralize_special_tokens(rendered_text: str | list[str], placeholder_to_token: dict[str, str]) -> str | list[str]:
    """
    Replaces escaped placeholders with a neutralized rendering of the original special token.

    A zero-width space (`U+200B`) is inserted after the first character of the token so that it remains visually
    identical while no longer matching the tokenizer's special token vocabulary. This is used on the
    `tokenize=False` path, where returning the literal token text would let user content be re-tokenized as
    control tokens by any downstream call.

    Args:
        rendered_text (`Union[str, list[str]]`): Rendered Jinja template string or list of rendered strings.
        placeholder_to_token (`dict[str, str]`): Mapping from placeholder strings to original special tokens.

    Returns:
        `Union[str, list[str]]`: Rendered text with placeholders replaced by neutralized special tokens.
    """
    if not placeholder_to_token or not rendered_text:
        return rendered_text

    if isinstance(rendered_text, list):
        return [neutralize_special_tokens(item, placeholder_to_token) for item in rendered_text]

    res = rendered_text
    for placeholder, token in placeholder_to_token.items():
        neutralized = f"{token[:1]}\u200b{token[1:]}" if token else token
        res = res.replace(placeholder, neutralized)
    return res


def replace_placeholders_in_sequences(
    sequence: list[int],
    attention_mask: list[int] | None,
    assistant_mask: list[int] | None,
    target_ids: list[int],
    replacement_ids: list[int],
) -> tuple[list[int], list[int] | None, list[int] | None, bool]:
    """
    Replaces occurrences of target_ids with replacement_ids in sequence, while maintaining
    positional alignment with attention_mask and assistant_mask.

    Args:
        sequence (`list[int]`): The input token ID sequence.
        attention_mask (`list[int]`, *optional*): The attention mask sequence.
        assistant_mask (`list[int]`, *optional*): The assistant token mask sequence.
        target_ids (`list[int]`): The target sub-sequence of token IDs to replace.
        replacement_ids (`list[int]`): The replacement sub-sequence of token IDs.

    Returns:
        `tuple[list[int], list[int] | None, list[int] | None, bool]`: A tuple containing the updated sequence,
        updated attention_mask, updated assistant_mask, and whether at least one match was replaced. The returned
        sequences are always plain Python lists, even when the inputs were framework tensors.
    """
    # The inputs may be framework tensors (e.g. when `return_tensors` was passed to the processor), for which
    # slice comparison would return an elementwise tensor instead of a bool. Work on plain lists instead.
    sequence = to_py_obj(sequence)
    attention_mask = to_py_obj(attention_mask) if attention_mask is not None else None
    assistant_mask = to_py_obj(assistant_mask) if assistant_mask is not None else None
    target_ids = to_py_obj(target_ids)
    replacement_ids = to_py_obj(replacement_ids)

    if not target_ids or len(target_ids) > len(sequence):
        return sequence, attention_mask, assistant_mask, False

    found = False
    res_sequence = []
    res_attention = [] if attention_mask is not None else None
    res_assistant = [] if assistant_mask is not None else None
    i = 0
    seq_len = len(sequence)
    target_len = len(target_ids)
    rep_len = len(replacement_ids)

    while i < seq_len:
        if sequence[i : i + target_len] == target_ids:
            found = True
            res_sequence.extend(replacement_ids)
            if attention_mask is not None:
                att_val = attention_mask[i] if i < len(attention_mask) else 1
                res_attention.extend([att_val] * rep_len)
            if assistant_mask is not None:
                ast_val = assistant_mask[i] if i < len(assistant_mask) else 0
                res_assistant.extend([ast_val] * rep_len)
            i += target_len
        else:
            res_sequence.append(sequence[i])
            if attention_mask is not None:
                res_attention.append(attention_mask[i] if i < len(attention_mask) else 1)
            if assistant_mask is not None:
                res_assistant.append(assistant_mask[i] if i < len(assistant_mask) else 0)
            i += 1

    return res_sequence, res_attention, res_assistant, found
