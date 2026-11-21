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
"""This file contains the parsers used by chat response parsing. Each parser takes a chunk of captured text
and parses it into a single key in the output message dictionary. Functions are generally boilerplate."""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from typing import Any, NoReturn

import regex as re


def _text(text: str, args: dict) -> str:
    return text.strip() if args.get("strip", True) else text


def _int(text: str, args: dict) -> int:
    return int(_text(text, args))


def _float(text: str, args: dict) -> float:
    return float(_text(text, args))


def _bool(text: str, args: dict) -> bool:
    return _text(text, args).lower() in ("true", "1")


# Sentinel characters for lax-JSON string pre-extraction — ASCII control chars
# that should never appear in real LLM output.
_LAX_OPEN, _LAX_CLOSE = "\x01", "\x02"


def _json(text: str, args: dict) -> Any:
    """JSON parser with optional dialect knobs for LLM-emitted quirks.

    `args`:
      - `unquoted_keys` (bool): quote bare-identifier keys before parsing.
      - `string_delims` ([[open, close], ...]): strings delimited by these
        custom markers are pre-extracted, then restored as standard JSON strings.
      - `allow_non_json` (bool): return stripped text if parsing fails.
    """
    string_delims = args.get("string_delims", [])
    unquoted_keys = args.get("unquoted_keys", False)

    if string_delims and (_LAX_OPEN in text or _LAX_CLOSE in text):
        raise ValueError("json: input contains reserved sentinel characters (\\x01/\\x02); cannot parse safely.")

    working = text
    captured: list[str] = []
    for open_d, close_d in string_delims:
        pattern = re.escape(open_d) + r"(.*?)" + re.escape(close_d)

        def _capture(m: Any) -> str:
            captured.append(m.group(1))
            return f"{_LAX_OPEN}{len(captured) - 1}{_LAX_CLOSE}"

        working = re.sub(pattern, _capture, working, flags=re.DOTALL)

    if unquoted_keys:
        # The first alternative swallows whole quoted strings (escapes included) and puts them back untouched,
        # so that key-like text inside a string value is never rewritten. Only the second one is a bare key.
        working = re.sub(
            r'"(?:\\.|[^"\\])*"|(?<=[{,])(\s*)(\w+)(\s*):',
            lambda m: m.group(0) if m.group(2) is None else f'{m.group(1)}"{m.group(2)}"{m.group(3)}:',
            working,
        )

    for i, s in enumerate(captured):
        working = working.replace(f"{_LAX_OPEN}{i}{_LAX_CLOSE}", json.dumps(s))

    try:
        return json.loads(working)
    except json.JSONDecodeError as e:
        if args.get("allow_non_json"):
            return _text(text, args)
        if working == text:
            raise ValueError(f"json parser could not parse region as JSON.\nContent: {text!r}\nError: {e}") from e
        raise ValueError(
            f"json: could not parse after dialect transforms.\n"
            f"Original: {text!r}\nTransformed: {working!r}\nError: {e}"
        ) from e


def _sub_parse(raw: str, value_parser: dict | None) -> Any:
    if value_parser is None:
        return raw
    return parse_content(raw, value_parser.get("name", "text"), value_parser.get("args", {}))


def _xml_inline(text: str, args: dict) -> dict:
    """Parse shallow XML-ish tags into a dict. `tag_pattern` regex must have named
    groups `key` and `value`. Optional `value_parser` recurses; `merge_duplicates`
    collects duplicate keys into a list."""
    tag_pattern = args.get("tag_pattern")
    if tag_pattern is None:
        raise ValueError("xml-inline: 'tag_pattern' content_arg is required")
    value_parser = args.get("value_parser")
    merge = args.get("merge_duplicates", False)

    out: dict[str, Any] = {}
    merged: set[str] = set()
    for m in re.finditer(tag_pattern, text, flags=re.DOTALL):
        groups = m.groupdict()
        key = groups.get("key")
        if key is None:
            raise ValueError(f"xml-inline: tag_pattern must have a named group 'key'. Pattern: {tag_pattern}")
        value = _sub_parse(groups.get("value", ""), value_parser)
        if key in out and merge:
            if key not in merged:
                out[key] = [out[key]]
                merged.add(key)
            out[key].append(value)
        else:
            out[key] = value
    return out


def _kv_lines(text: str, args: dict) -> dict:
    """Parse line-delimited `key<sep>value` pairs into a dict."""
    line_sep = args.get("line_sep", "\n")
    kv_sep = args.get("kv_sep", ":")
    value_parser = args.get("value_parser")

    out: dict[str, Any] = {}
    for line in text.split(line_sep):
        line = _text(line, args)
        if not line or kv_sep not in line:
            continue
        k, v = line.split(kv_sep, 1)
        k, v = _text(k, args), _text(v, args)
        out[k] = _sub_parse(v, value_parser)
    return out


_CALL_LIST_WHITESPACE = re.compile(r"[ \t\n\r]*")
_CALL_LIST_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(?:[-.][A-Za-z0-9_]+)*")
_CALL_LIST_NUMBER = re.compile(r"-?(?:0|[1-9][0-9]*)(\.[0-9]+)?([eE][+-]?[0-9]+)?")
_CALL_LIST_STRING = re.compile(r"""'((?:[^'\\\n\r]|\\.)*)'|"((?:[^"\\\n\r]|\\.)*)\"""", flags=re.DOTALL)
_CALL_LIST_UNICODE_ESCAPE = re.compile(r"u([0-9a-fA-F]{4})")
_CALL_LIST_SURROGATE = re.compile(r"[\ud800-\udfff]")
_CALL_LIST_LOW_SURROGATE = re.compile(r"\\u([dD][c-fC-F][0-9a-fA-F]{2})")
_CALL_LIST_ESCAPES = {"\\": "\\", "'": "'", '"': '"', "/": "/", "b": "\b", "f": "\f", "n": "\n", "r": "\r", "t": "\t"}
_CALL_LIST_CONSTANTS = {"True": True, "true": True, "False": False, "false": False, "None": None, "null": None}
_CALL_LIST_MAX_INTEGER = 2**53 - 1
_CALL_LIST_MAX_DEPTH = 100


def _python_call_list(text: str, args: dict) -> list[dict]:
    """Parse a list of keyword-argument calls, e.g. `[get_weather(city="Paris")]`, into
    `[{"name": ..., "arguments": {...}}]`. The grammar is specified in `docs/source/en/chat_response_parsing.md`
    and is deliberately strict, so that implementations in other languages can match it exactly."""
    position = 0

    def fail(message: str) -> NoReturn:
        raise ValueError(f"python-call-list: {message} at position {position}.\nContent: {text!r}")

    def skip_whitespace() -> None:
        nonlocal position
        position = _CALL_LIST_WHITESPACE.match(text, position).end()

    def at(char: str) -> bool:
        skip_whitespace()
        return text.startswith(char, position)

    def expect(char: str) -> None:
        nonlocal position
        if not at(char):
            fail(f"expected {char!r}")
        position += 1

    def items(close: str, item: Callable[[], Any]) -> list:
        nonlocal position
        out = []
        if at(close):
            position += 1
            return out
        while True:
            out.append(item())
            if at(close):
                position += 1
                return out
            expect(",")

    def unique(pairs: list[tuple[str, Any]]) -> dict:
        out = {}
        for key, value in pairs:
            if key in out:
                fail(f"repeated name {key!r}")
            out[key] = value
        return out

    def string() -> str:
        nonlocal position
        match = _CALL_LIST_STRING.match(text, position)
        if match is None:
            fail("invalid string")
        body = match[1] if match[1] is not None else match[2]
        if _CALL_LIST_SURROGATE.search(body) is not None:
            fail("lone surrogate in the string starting")
        out, start = [], 0
        while (backslash := body.find("\\", start)) != -1:
            out.append(body[start:backslash])
            start = backslash + 2
            if body[backslash + 1] in _CALL_LIST_ESCAPES:
                out.append(_CALL_LIST_ESCAPES[body[backslash + 1]])
                continue
            escape = _CALL_LIST_UNICODE_ESCAPE.match(body, backslash + 1)
            if escape is None:
                fail(f"unsupported escape {body[backslash : backslash + 2]!r} in the string starting")
            code, start = int(escape[1], 16), escape.end()
            if 0xD800 <= code <= 0xDBFF and (low := _CALL_LIST_LOW_SURROGATE.match(body, start)) is not None:
                code, start = 0x10000 + ((code - 0xD800) << 10) + (int(low[1], 16) - 0xDC00), low.end()
            if 0xD800 <= code <= 0xDFFF:
                fail("lone surrogate escape in the string starting")
            out.append(chr(code))
        position = match.end()
        return "".join(out) + body[start:]

    def value(depth: int) -> Any:
        # `depth` is the number of lists and dicts that enclose this value.
        nonlocal position
        if at("'") or at('"'):
            return string()
        if at("[") or at("{"):
            if depth >= _CALL_LIST_MAX_DEPTH:
                fail(f"lists and dicts nested more than {_CALL_LIST_MAX_DEPTH} levels deep")
            position += 1
            if text[position - 1] == "[":
                return items("]", lambda: value(depth + 1))
            return unique(items("}", lambda: pair(depth + 1)))
        if (match := _CALL_LIST_NUMBER.match(text, position)) is not None:
            is_integer = match[1] is None and match[2] is None
            # Check the digit count first so that huge literals never reach `int()`.
            if is_integer and (len(match[0]) > 17 or abs(int(match[0])) > _CALL_LIST_MAX_INTEGER):
                fail("integer outside the range of -(2^53 - 1) to 2^53 - 1")
            if not is_integer and math.isinf(float(match[0])):
                fail("number too large")
            position = match.end()
            return int(match[0]) if is_integer else float(match[0])
        if (match := _CALL_LIST_NAME.match(text, position)) is not None and match[0] in _CALL_LIST_CONSTANTS:
            position = match.end()
            return _CALL_LIST_CONSTANTS[match[0]]
        fail("expected a string, number, list, dict, True, False or None")

    def pair(depth: int) -> tuple[str, Any]:
        if not (at("'") or at('"')):
            fail("dict keys must be strings")
        key = string()
        expect(":")
        return key, value(depth)

    def name(what: str) -> str:
        nonlocal position
        skip_whitespace()
        match = _CALL_LIST_NAME.match(text, position)
        if match is None:
            fail(f"expected {what}")
        position = match.end()
        return match[0]

    def argument() -> tuple[str, Any]:
        nonlocal position
        key = name("a keyword argument (positional arguments and trailing commas are not supported)")
        if not at("="):
            fail("expected '=' (positional arguments are not supported)")
        position += 1
        return key, value(0)

    def call() -> dict:
        function_name = name("a function name")
        expect("(")
        return {"name": function_name, "arguments": unique(items(")", argument))}

    expect("[")
    calls = items("]", call)
    skip_whitespace()
    if position != len(text):
        fail("unexpected content after the closing ']'")
    return calls


CONTENT_PARSERS = {
    "text": _text,
    "int": _int,
    "float": _float,
    "bool": _bool,
    "json": _json,
    "xml-inline": _xml_inline,
    "kv-lines": _kv_lines,
    "python-call-list": _python_call_list,
}

# Parsers whose output is the verbatim body text (modulo whitespace) — chunks
# from these fields stream with `dirty=False` because each chunk is part of
# the final value. Structured parsers (`json`, `xml-inline`, `kv-lines`, `python-call-list`) only
# produce a meaningful value on close, so their chunks stream raw bytes
# flagged `dirty=True` while the parsed value is delivered in `region_close`.
STREAMABLE_PARSERS = frozenset({"text", "int", "float", "bool"})

# Structured parsers whose syntax already distinguishes quoted strings from typed
# literals (numbers, booleans, null/None, lists, dicts). Schema-driven string
# coercion in `ResponseParser` skips these so explicit string arguments are preserved.
TYPED_STRUCTURED_PARSERS = frozenset({"json", "python-call-list"})


def parse_content(text: str, name: str, args: dict) -> Any:
    return CONTENT_PARSERS[name](text, args)


_PLACEHOLDER = re.compile(r"\{(\w+(?:\.\w+)*)\}")


def _apply_transform(transform: Any, scope: dict) -> Any:
    """Recursively walk a transform template, which is used to restructure
    parsed output into the actual shape we want. A dotted placeholder like
    `{content.args}` descends into keys of the looked-up value."""
    if isinstance(transform, dict):
        return {k: _apply_transform(v, scope) for k, v in transform.items()}
    if isinstance(transform, list):
        return [_apply_transform(v, scope) for v in transform]
    if not isinstance(transform, str):
        return transform
    whole = _PLACEHOLDER.fullmatch(transform)
    if not whole:
        return transform
    path = whole.group(1)
    root, *keys = path.split(".")
    if root not in scope:
        raise KeyError(f"transform placeholder '{{{path}}}' is not defined. Available: {sorted(scope)}")
    value = scope[root]
    for key in keys:
        if not isinstance(value, dict):
            raise ValueError(f"transform placeholder '{{{path}}}' cannot index into {type(value).__name__} at '{key}'")
        if key not in value:
            raise ValueError(f"transform placeholder '{{{path}}}' is missing key '{key}'. Available: {sorted(value)}")
        value = value[key]
    return value


def validate_transform_strings(scope: str, transform: Any) -> None:
    """Walk a transform template and reject any string that mixes a `{name}`
    placeholder with literal text. Only whole-string placeholders (e.g.
    `"{content}"`) and plain literals are supported. Called from the template
    loader so authors get a clear error at load time, not at parse time."""
    if isinstance(transform, dict):
        for v in transform.values():
            validate_transform_strings(scope, v)
        return
    if isinstance(transform, list):
        for v in transform:
            validate_transform_strings(scope, v)
        return
    if not isinstance(transform, str):
        return
    if _PLACEHOLDER.search(transform) and not _PLACEHOLDER.fullmatch(transform):
        raise ValueError(
            f"{scope}: transform string {transform!r} mixes a {{placeholder}} with literal text. "
            'Use either a whole-string placeholder (e.g. "{content}") or a plain literal; '
            "string interpolation is not supported."
        )


def process_field(body: str, field, captures: dict) -> Any:
    """Run `body` through the field's content parser, then optionally apply the
    transform template. When `transform_each` is set, the parsed content must
    be a list and the template is applied to each element (with the element's
    keys unpacked into the template scope, alongside any regex captures).

    `field` is a `spec.Field`; typed via duck-typing to avoid a cyclic import."""
    value = parse_content(body, field.content, field.content_args)
    if field.transform is None:
        return value
    if field.transform_each:
        if not isinstance(value, list):
            raise ValueError(
                f"Field '{field.name}': transform_each requires the parsed content to be a list, "
                f"got {type(value).__name__}."
            )
        out = []
        for item in value:
            if not isinstance(item, dict):
                raise ValueError(
                    f"Field '{field.name}': transform_each requires each list element to be a dict, "
                    f"got {type(item).__name__}."
                )
            out.append(_apply_transform(field.transform, {**captures, **item}))
        return out
    return _apply_transform(field.transform, {**captures, "content": value})


__all__ = [
    "CONTENT_PARSERS",
    "STREAMABLE_PARSERS",
    "TYPED_STRUCTURED_PARSERS",
    "parse_content",
    "process_field",
    "validate_transform_strings",
]
