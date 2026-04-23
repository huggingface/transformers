# Copyright 2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Content-parser registry. Users select a parser by name in their spec; the registry
is closed (no dynamic registration from user data — schemas are data, not code)."""

from __future__ import annotations

from typing import Any, Protocol


class ContentParser(Protocol):
    name: str

    def parse(self, text: str, args: dict) -> Any: ...


from .json_parser import JsonLaxParser, JsonParser
from .structured import KvLinesParser, XmlInlineParser
from .text_parser import RawParser, TextParser


CONTENT_PARSERS: dict[str, ContentParser] = {
    "text": TextParser(),
    "raw": RawParser(),
    "json": JsonParser(),
    "json-lax": JsonLaxParser(),
    "xml-inline": XmlInlineParser(),
    "kv-lines": KvLinesParser(),
}


def parse_content(text: str, parser_name: str, args: dict) -> Any:
    """Parse `text` using the named content parser with the given args."""
    return CONTENT_PARSERS[parser_name].parse(text, args)


__all__ = ["CONTENT_PARSERS", "ContentParser", "parse_content"]
