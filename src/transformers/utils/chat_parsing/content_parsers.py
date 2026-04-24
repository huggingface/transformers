# Copyright 2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Content pipeline: body string → parsed value → assembled → coerced.

Three stages compose into `process_field(body, fld, captures)`:
  1. `parse_content` dispatches through the closed `CONTENT_PARSERS` registry.
  2. `_apply_assemble` optionally wraps the parsed value in a template
     (using `{name}` placeholders filled from open-pattern captures).
  3. `_apply_coerce` optionally casts the result to int/float/bool.

The registry is closed — schemas select parsers by name but cannot ship code."""

from __future__ import annotations

import json
import re
from typing import Any


def _text(text: str, args: dict) -> str:
    return text.strip() if args.get("strip", True) else text


def _raw(text: str, args: dict) -> str:
    return text


def _jmespath(value: Any, transform: str | None) -> Any:
    if transform is None:
        return value
    from ...utils import is_jmespath_available

    if not is_jmespath_available():
        raise ImportError(
            "response_format uses a jmespath 'transform', but jmespath is not installed. "
            "Install with `pip install jmespath`."
        )
    import jmespath

    return jmespath.search(transform, value)


def _json(text: str, args: dict) -> Any:
    try:
        value = json.loads(text)
    except json.JSONDecodeError as e:
        if args.get("allow_non_json"):
            return text.strip()
        raise ValueError(f"json parser could not parse region as JSON.\nContent: {text!r}\nError: {e}") from e
    return _jmespath(value, args.get("transform"))


# Sentinel characters for lax-JSON string pre-extraction — ASCII control chars
# that should never appear in real LLM output.
_LAX_OPEN, _LAX_CLOSE = "\x01", "\x02"


def _json_lax(text: str, args: dict) -> Any:
    """JSON with optional dialect knobs for LLM-emitted quirks.

    `args`:
      - `unquoted_keys` (bool): quote bare-identifier keys before parsing.
      - `string_delims` ([[open, close], ...]): strings delimited by these
        custom markers are pre-extracted, then restored as standard JSON strings.
      - `transform` (str): jmespath applied after parsing.
      - `allow_non_json` (bool): return stripped text if parsing fails.
    """
    if _LAX_OPEN in text or _LAX_CLOSE in text:
        raise ValueError("json-lax: input contains reserved sentinel characters (\\x01/\\x02); cannot parse safely.")

    working = text
    captured: list[str] = []
    for open_d, close_d in args.get("string_delims") or []:
        pattern = re.escape(open_d) + r"(.*?)" + re.escape(close_d)

        def _capture(m: re.Match) -> str:
            captured.append(m.group(1))
            return f"{_LAX_OPEN}{len(captured) - 1}{_LAX_CLOSE}"

        working = re.sub(pattern, _capture, working, flags=re.DOTALL)

    if args.get("unquoted_keys"):
        working = re.sub(r"(?<=[{,])(\w+):", r'"\1":', working)

    for i, s in enumerate(captured):
        working = working.replace(f"{_LAX_OPEN}{i}{_LAX_CLOSE}", json.dumps(s))

    try:
        value = json.loads(working)
    except json.JSONDecodeError as e:
        if args.get("allow_non_json"):
            return text.strip()
        raise ValueError(
            f"json-lax: could not parse after dialect transforms.\n"
            f"Original: {text!r}\nTransformed: {working!r}\nError: {e}"
        ) from e
    return _jmespath(value, args.get("transform"))


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
    "raw": _raw,
    "json": _json,
    "json-lax": _json_lax,
    "xml-inline": _xml_inline,
    "kv-lines": _kv_lines,
}


def parse_content(text: str, name: str, args: dict) -> Any:
    return CONTENT_PARSERS[name](text, args)


_PLACEHOLDER = re.compile(r"\{(\w+)\}")


def _apply_assemble(assemble: Any, captures: dict, content: Any) -> Any:
    if assemble is None:
        return content
    if isinstance(assemble, dict):
        return {k: _apply_assemble(v, captures, content) for k, v in assemble.items()}
    if isinstance(assemble, list):
        return [_apply_assemble(v, captures, content) for v in assemble]
    if not isinstance(assemble, str):
        return assemble

    def _lookup(key: str) -> Any:
        if key == "content":
            return content
        if key in captures:
            return captures[key]
        raise KeyError(
            f"assemble template '{assemble}' references unknown capture '{key}'. "
            f"Available: {sorted(list(captures) + ['content'])}"
        )

    whole = _PLACEHOLDER.fullmatch(assemble)
    if whole:
        return _lookup(whole.group(1))
    return _PLACEHOLDER.sub(lambda m: str(_lookup(m.group(1))), assemble)


def _apply_coerce(value: Any, coerce: str | None) -> Any:
    if coerce is None:
        return value
    if coerce == "int":
        return int(value)
    if coerce == "float":
        return float(value)
    if coerce == "bool":
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in ("true", "1", "yes")
        return bool(value)
    raise ValueError(f"Unknown coerce: {coerce}")


def process_field(body: str, fld, captures: dict) -> Any:
    """Run `body` through the field's content parser, assembler, and coercer.

    `fld` is a `spec.Field`; typed via duck-typing to avoid a cyclic import."""
    value = parse_content(body, fld.content, fld.content_args)
    value = _apply_assemble(fld.assemble, captures, value)
    value = _apply_coerce(value, fld.coerce)
    return value


__all__ = ["CONTENT_PARSERS", "parse_content", "process_field"]
