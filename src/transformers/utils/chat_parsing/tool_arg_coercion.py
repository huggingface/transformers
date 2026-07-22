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
"""Schema-driven typing of tool-call arguments.

Some response templates capture tool arguments as raw text: an `xml-inline` field
with no `value_parser` yields all strings (e.g. `{"hour": "7", "enabled": "true"}`),
even when the tool's JSON Schema declares `integer` / `boolean` / etc. When
OpenAI-style `tools` definitions are available, `coerce_tool_calls` casts each
string-valued argument to its declared type, using the tool named by the parsed
call itself (`function.name`).

Coercion is a pure post-parse pass over the canonical tool-call value produced by a
field's `transform`; it never reaches into the generic content parsers. It is
conservative: already-typed values and arguments the schema does not describe are
left untouched, and any cast that fails falls back to the raw string, so passing
`tools=` can only add type information, never corrupt a parsed call.
"""

from __future__ import annotations

import json
from typing import Any


def _schema_types(schema: Any) -> list[str]:
    """Candidate JSON Schema type names for `schema`.

    Handles `type` as a string or list, plus `anyOf` / `oneOf` / `allOf` (e.g.
    `Optional[int]` -> `anyOf: [integer, null]`) recursively, and `$ref` (nested
    models) treated as `object`. Returns an empty list when nothing can be determined.
    """
    if not isinstance(schema, dict):
        return []
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
    return types


def _coerce_scalar(raw: str, type_name: Any) -> Any:
    """Single-type coercion. Returns the original `raw` object on any failure.

    Recognized booleans match the `bool` content parser (`true`/`false`/`1`/`0`,
    case-insensitive); anything else stays a string. `object` / `array` must be valid
    JSON of the matching shape, so a `{...}` body is never accepted for an `array`
    parameter (or vice versa)."""
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


def _coerce_value(raw: str, schema: Any) -> Any:
    """Coerce `raw` to a declared JSON Schema type in `schema`; return `raw` on failure.

    Tries each candidate type in turn. Types that leave the value unchanged
    (notably `string`) are skipped so a union like `[string, integer]` can still
    recover the integer, while a plain `string` schema leaves the value as a string.
    """
    for type_name in _schema_types(schema):
        coerced = _coerce_scalar(raw, type_name)
        if coerced is not raw:
            return coerced
    return raw


def _properties_by_tool_name(tools: Any) -> dict[str, dict]:
    """Map tool name -> JSON Schema `properties` dict from OpenAI tool specs."""
    out: dict[str, dict] = {}
    for tool in tools or []:
        fn = tool.get("function", tool) if isinstance(tool, dict) else None
        if not isinstance(fn, dict):
            continue
        name = fn.get("name")
        if not name:
            continue
        params = fn.get("parameters")
        props = params.get("properties") if isinstance(params, dict) else None
        out[name] = props if isinstance(props, dict) else {}
    return out


def coerce_arguments(arguments: dict, properties: Any) -> dict:
    """Cast string-valued entries of `arguments` to their declared JSON Schema types.

    `properties` is a single tool's JSON Schema `parameters.properties` dict (argument
    name -> schema). Entries with no matching key, or whose value is not a string, are
    left untouched.
    """
    if not properties:
        return arguments
    return {
        key: (_coerce_value(value, properties[key]) if key in properties and isinstance(value, str) else value)
        for key, value in arguments.items()
    }


def coerce_tool_calls(value: Any, tools: Any) -> Any:
    """Coerce the arguments of a parsed tool-call value against `tools`.

    Accepts either a single tool call (`{"function": {"name", "arguments"}, ...}`, as
    produced by a `repeats` field) or a list of them (as produced by `transform_each`).
    Values that don't match the OpenAI tool-call shape are returned unchanged. Matching
    calls are mutated in place and returned.
    """
    if not tools:
        return value
    if isinstance(value, list):
        return [_coerce_one(item, tools) for item in value]
    return _coerce_one(value, tools)


def _coerce_one(call: Any, tools: Any) -> Any:
    if not isinstance(call, dict):
        return call
    fn = call.get("function")
    if not isinstance(fn, dict):
        return call
    args = fn.get("arguments")
    if not isinstance(args, dict):
        return call
    props = _properties_by_tool_name(tools).get(fn.get("name"))
    if props:
        fn["arguments"] = coerce_arguments(args, props)
    return call
