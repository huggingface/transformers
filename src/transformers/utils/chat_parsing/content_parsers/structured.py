# Copyright 2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Structured content parsers that turn shallow XML-ish or line-KV regions
into dicts. Both support a recursive `value_parser` to further parse each value."""

from __future__ import annotations

import re
from typing import Any


def _parse_value(raw: str, value_parser_spec: dict | None) -> Any:
    if value_parser_spec is None:
        return raw
    # Avoid a top-level circular import: content_parsers.__init__ imports this module.
    from . import parse_content

    sub_name = value_parser_spec.get("name", "text")
    sub_args = value_parser_spec.get("args", {})
    return parse_content(raw, sub_name, sub_args)


class XmlInlineParser:
    """Parse shallow XML-ish tags into a dict.

    `content_args`:
      - `tag_pattern` (str, required): regex with named groups `key` and `value`.
      - `value_parser` ({"name": ..., "args": {...}}, optional): recursive parser
        applied to each value before assembly.
      - `merge_duplicates` (bool, default False): if True, duplicate keys become a list.
    """

    name = "xml-inline"

    def parse(self, text: str, args: dict) -> dict:
        tag_pattern = args.get("tag_pattern")
        if tag_pattern is None:
            raise ValueError("xml-inline: 'tag_pattern' content_arg is required")
        value_parser = args.get("value_parser")
        merge_duplicates = args.get("merge_duplicates", False)

        out: dict[str, Any] = {}
        for m in re.finditer(tag_pattern, text, flags=re.DOTALL):
            groups = m.groupdict()
            key = groups.get("key")
            value = groups.get("value", "")
            if key is None:
                raise ValueError(f"xml-inline: tag_pattern must have a named group 'key'. Pattern: {tag_pattern}")
            parsed = _parse_value(value, value_parser)
            if key in out and merge_duplicates:
                existing = out[key]
                if not isinstance(existing, list):
                    out[key] = [existing]
                out[key].append(parsed)
            else:
                out[key] = parsed
        return out


class KvLinesParser:
    """Parse a region of line-delimited `key<sep>value` pairs into a dict.

    `content_args`:
      - `line_sep` (str, default "\\n"): line separator.
      - `kv_sep` (str, default ":"): key/value separator (first occurrence splits).
      - `value_parser` ({"name": ..., "args": {...}}, optional).
      - `strip` (bool, default True): strip whitespace from keys and values.
    """

    name = "kv-lines"

    def parse(self, text: str, args: dict) -> dict:
        line_sep = args.get("line_sep", "\n")
        kv_sep = args.get("kv_sep", ":")
        value_parser = args.get("value_parser")
        strip = args.get("strip", True)

        out: dict[str, Any] = {}
        for line in text.split(line_sep):
            if strip:
                line = line.strip()
            if not line:
                continue
            if kv_sep not in line:
                continue
            k, v = line.split(kv_sep, 1)
            if strip:
                k, v = k.strip(), v.strip()
            out[k] = _parse_value(v, value_parser)
        return out
