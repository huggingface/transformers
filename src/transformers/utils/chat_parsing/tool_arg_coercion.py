# Copyright 2026 The HuggingFace Team. All rights reserved.
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
"""Cast tool-call arguments captured as text to the types their JSON Schema declares."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any


# Container args arrive as one JSON blob, which a `value_parser` already decodes.
_CONTAINER_TYPES = frozenset({"array", "object"})


def _schema_types(schema: Any) -> tuple[str, ...]:
    """Candidate type names for `schema`, recursing into `anyOf` / `oneOf` / `allOf`.

    `$ref` counts as `object`. Empty when no type can be determined.
    """
    if not isinstance(schema, dict):
        return ()
    types: list[str] = []
    type_value = schema.get("type")
    if isinstance(type_value, str):
        types.append(type_value)
    elif isinstance(type_value, list):
        types.extend(t for t in type_value if isinstance(t, str))
    for choice_field in ("anyOf", "oneOf", "allOf"):
        for choice in schema.get(choice_field) or []:
            types.extend(_schema_types(choice))
    if "$ref" in schema:
        types.append("object")
    return tuple(types)


def build_tool_type_index(tools: Any) -> dict[str, dict[str, tuple[str, ...]]]:
    """Map tool name -> parameter name -> candidate types. Untyped parameters are omitted."""
    out: dict[str, dict[str, tuple[str, ...]]] = {}
    for tool in tools or []:
        fn = tool.get("function", tool) if isinstance(tool, dict) else None
        if not isinstance(fn, dict):
            continue
        name = fn.get("name")
        if not name:
            continue
        params = fn.get("parameters")
        props = params.get("properties") if isinstance(params, dict) else None
        param_types: dict[str, tuple[str, ...]] = {}
        if isinstance(props, dict):
            for param_name, schema in props.items():
                types = _schema_types(schema)
                if types:
                    param_types[param_name] = types
        out[name] = param_types
    return out


def raw_text_params(param_types: dict[str, Sequence[str]]) -> frozenset[str]:
    """Parameters that must stay raw text for `coerce_arguments` to type them.

    One declared scalar type is enough, since a `value_parser` that reads `1.5` as a float
    puts it beyond a cast's reach. Container-only parameters are better served by the parser.
    """
    return frozenset(name for name, types in param_types.items() if not _CONTAINER_TYPES.issuperset(types))


def _coerce_scalar(raw: str, type_name: Any) -> Any:
    """Coerce `raw` to one type, returning it unchanged on failure.

    Booleans follow the `bool` content parser; `object` / `array` need JSON of that shape.
    """
    try:
        if type_name == "integer":
            return int(raw)
        if type_name == "number":
            value = float(raw)
            if value != value or value in (float("inf"), float("-inf")):
                # Reject NaN / inf: not valid JSON numbers.
                return raw
            # Preserve ints when the source text had no fractional part.
            return int(value) if value.is_integer() and "." not in raw else value
        if type_name == "boolean":
            lowered = raw.strip().lower()
            if lowered in ("true", "1"):
                return True
            if lowered in ("false", "0"):
                return False
            return raw
        if type_name == "object":
            decoded = json.loads(raw)
            return decoded if isinstance(decoded, dict) else raw
        if type_name == "array":
            decoded = json.loads(raw)
            return decoded if isinstance(decoded, list) else raw
        if type_name == "null":
            return None if raw.strip() in ("null", "None") else raw
    except (ValueError, TypeError, json.JSONDecodeError):
        return raw
    # "string" and unknown/absent types pass through unchanged.
    return raw


def _coerce_value(raw: str, type_names: Sequence[str]) -> Any:
    """Try each candidate type in turn, keeping the first that changes `raw`.

    Skipping no-op types lets a `[string, integer]` union still recover the integer.
    """
    for type_name in type_names:
        coerced = _coerce_scalar(raw, type_name)
        if coerced is not raw:
            return coerced
    return raw


def _coerce_argument(value: Any, type_names: Sequence[str]) -> Any:
    """Coerce one argument. Lists (from `merge_duplicates`) are cast element-wise."""
    if isinstance(value, str):
        return _coerce_value(value, type_names)
    if isinstance(value, list):
        return [_coerce_value(item, type_names) if isinstance(item, str) else item for item in value]
    return value


def coerce_arguments(arguments: dict, param_types: dict[str, Sequence[str]]) -> dict:
    """Cast string entries of `arguments`. Anything the schema omits is left untouched."""
    if not param_types:
        return arguments
    return {
        key: (_coerce_argument(value, param_types[key]) if key in param_types else value)
        for key, value in arguments.items()
    }


def tool_call_param_types(value: Any, tool_types: dict[str, dict[str, tuple[str, ...]]]) -> dict | None:
    """Declared types for `value`, or `None` unless it is a call to a known tool.

    Requiring the full OpenAI shape keeps fields that merely capture a `name` out of this path.
    """
    if not isinstance(value, dict):
        return None
    fn = value.get("function")
    if not isinstance(fn, dict) or not isinstance(fn.get("arguments"), dict):
        return None
    name = fn.get("name")
    return tool_types.get(name) if isinstance(name, str) else None


def coerce_tool_calls(value: Any, tool_types: dict[str, dict[str, tuple[str, ...]]]) -> Any:
    """Coerce a parsed tool call, or a list of them, in place against the type index.

    Values that don't match the OpenAI tool-call shape are returned unchanged.
    """
    if not tool_types:
        return value
    if isinstance(value, list):
        return [_coerce_call(item, tool_types) for item in value]
    return _coerce_call(value, tool_types)


def _coerce_call(call: Any, tool_types: dict[str, dict[str, tuple[str, ...]]]) -> Any:
    param_types = tool_call_param_types(call, tool_types)
    if param_types:
        fn = call["function"]
        fn["arguments"] = coerce_arguments(fn["arguments"], param_types)
    return call


__all__ = [
    "build_tool_type_index",
    "coerce_arguments",
    "coerce_tool_calls",
    "raw_text_params",
    "tool_call_param_types",
]
