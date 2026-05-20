# Copyright 2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""This file contains the parsers used by chat response parsing. Each parser takes a chunk of captured text
and parses it into a single key in the output message dictionary. Functions are generally boilerplate and as
a result this is mostly agent-written."""

from __future__ import annotations

import json
import re
from typing import Any


def _text(text: str, args: dict) -> str:
    return text.strip() if args.get("strip", True) else text


def _int(text: str, args: dict) -> int:
    return int(text.strip())


def _float(text: str, args: dict) -> float:
    return float(text.strip())


def _bool(text: str, args: dict) -> bool:
    return text.strip().lower() in ("true", "1")


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
    string_delims = args.get("string_delims") or []
    unquoted_keys = args.get("unquoted_keys", False)

    if string_delims and (_LAX_OPEN in text or _LAX_CLOSE in text):
        raise ValueError("json: input contains reserved sentinel characters (\\x01/\\x02); cannot parse safely.")

    working = text
    captured: list[str] = []
    for open_d, close_d in string_delims:
        pattern = re.escape(open_d) + r"(.*?)" + re.escape(close_d)

        def _capture(m: re.Match) -> str:
            captured.append(m.group(1))
            return f"{_LAX_OPEN}{len(captured) - 1}{_LAX_CLOSE}"

        working = re.sub(pattern, _capture, working, flags=re.DOTALL)

    if unquoted_keys:
        working = re.sub(r"(?<=[{,])(\w+):", r'"\1":', working)

    for i, s in enumerate(captured):
        working = working.replace(f"{_LAX_OPEN}{i}{_LAX_CLOSE}", json.dumps(s))

    try:
        return json.loads(working)
    except json.JSONDecodeError as e:
        if args.get("allow_non_json"):
            return text.strip()
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
    for m in re.finditer(tag_pattern, text, flags=re.DOTALL):
        groups = m.groupdict()
        key = groups.get("key")
        if key is None:
            raise ValueError(f"xml-inline: tag_pattern must have a named group 'key'. Pattern: {tag_pattern}")
        value = _sub_parse(groups.get("value", ""), value_parser)
        if key in out and merge:
            if not isinstance(out[key], list):
                out[key] = [out[key]]
            out[key].append(value)
        else:
            out[key] = value
    return out


def _kv_lines(text: str, args: dict) -> dict:
    """Parse line-delimited `key<sep>value` pairs into a dict."""
    line_sep = args.get("line_sep", "\n")
    kv_sep = args.get("kv_sep", ":")
    value_parser = args.get("value_parser")
    strip = args.get("strip", True)

    out: dict[str, Any] = {}
    for line in text.split(line_sep):
        if strip:
            line = line.strip()
        if not line or kv_sep not in line:
            continue
        k, v = line.split(kv_sep, 1)
        if strip:
            k, v = k.strip(), v.strip()
        out[k] = _sub_parse(v, value_parser)
    return out


CONTENT_PARSERS = {
    "text": _text,
    "int": _int,
    "float": _float,
    "bool": _bool,
    "json": _json,
    "xml-inline": _xml_inline,
    "kv-lines": _kv_lines,
}

# Parsers whose output is the verbatim body text (modulo whitespace) — chunks
# from these fields stream with `dirty=False` because each chunk is part of
# the final value. Structured parsers (`json`, `xml-inline`, `kv-lines`) only
# produce a meaningful value on close, so their chunks stream raw bytes
# flagged `dirty=True` while the parsed value is delivered in `region_close`.
STREAMABLE_PARSERS = frozenset({"text", "int", "float", "bool"})


def parse_content(text: str, name: str, args: dict) -> Any:
    return CONTENT_PARSERS[name](text, args)


_PLACEHOLDER = re.compile(r"\{(\w+)\}")


def _apply_transform(transform: Any, scope: dict) -> Any:
    """Recursively walk a transform template. A string matching the whole-string
    placeholder pattern `{name}` is replaced with the value of `name` in
    `scope`, with the value's type preserved (so `"{content}"` resolving to a
    dict stays a dict). Strings without any placeholder are literals. Mixing
    a placeholder with literal text in the same string is rejected at
    template load time; see `validate_transform_strings`."""
    if isinstance(transform, dict):
        return {k: _apply_transform(v, scope) for k, v in transform.items()}
    if isinstance(transform, list):
        return [_apply_transform(v, scope) for v in transform]
    if not isinstance(transform, str):
        return transform
    whole = _PLACEHOLDER.fullmatch(transform)
    if not whole:
        return transform
    key = whole.group(1)
    if key not in scope:
        raise KeyError(f"transform placeholder '{{{key}}}' is not defined. Available: {sorted(scope)}")
    return scope[key]


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


def process_field(body: str, fld, captures: dict) -> Any:
    """Run `body` through the field's content parser, then optionally apply the
    transform template. When `transform_each` is set, the parsed content must
    be a list and the template is applied to each element (with the element's
    keys unpacked into the template scope, alongside any regex captures).

    `fld` is a `spec.Field`; typed via duck-typing to avoid a cyclic import."""
    value = parse_content(body, fld.content, fld.content_args)
    if fld.transform is None:
        return value
    if fld.transform_each:
        if not isinstance(value, list):
            raise ValueError(
                f"Field '{fld.name}': transform_each requires the parsed content to be a list, "
                f"got {type(value).__name__}."
            )
        out = []
        for item in value:
            if not isinstance(item, dict):
                raise ValueError(
                    f"Field '{fld.name}': transform_each requires each list element to be a dict, "
                    f"got {type(item).__name__}."
                )
            out.append(_apply_transform(fld.transform, {**captures, **item}))
        return out
    return _apply_transform(fld.transform, {**captures, "content": value})


__all__ = [
    "CONTENT_PARSERS",
    "STREAMABLE_PARSERS",
    "parse_content",
    "process_field",
    "validate_transform_strings",
]
