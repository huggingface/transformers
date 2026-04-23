# Copyright 2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""JSON content parsers. `json` is strict; `json-lax` handles dialect knobs
(unquoted keys, custom string delimiters) that LLMs sometimes emit."""

from __future__ import annotations

import json
import re
from typing import Any


def _apply_transform(value: Any, transform: str | None) -> Any:
    if transform is None:
        return value
    from ....utils import is_jmespath_available

    if not is_jmespath_available():
        raise ImportError(
            "response_format uses a jmespath 'transform', but jmespath is not installed. "
            "Install with `pip install jmespath`."
        )
    import jmespath

    return jmespath.search(transform, value)


class JsonParser:
    name = "json"

    def parse(self, text: str, args: dict) -> Any:
        transform = args.get("transform")
        allow_non_json = args.get("allow_non_json", False)
        try:
            value = json.loads(text)
        except json.JSONDecodeError as e:
            if allow_non_json:
                return text.strip() if isinstance(text, str) else text
            raise ValueError(f"json parser could not parse region as JSON.\nContent: {text!r}\nError: {e}") from e
        return _apply_transform(value, transform)


# Placeholder sentinel for lax-JSON string-pre-extraction. Using characters
# extremely unlikely to appear in LLM output (ASCII control chars).
_LAX_OPEN = "\x01"
_LAX_CLOSE = "\x02"


class JsonLaxParser:
    """JSON with optional dialect knobs for LLM-emitted quirks.

    Supported `content_args`:
      - `unquoted_keys` (bool): quote bare-identifier keys before parsing.
      - `string_delims` ([[open, close], ...]): strings delimited by these
        custom markers are pre-extracted, then restored as properly-escaped
        standard JSON strings.
      - `transform` (str): jmespath transform applied after parsing.
      - `allow_non_json` (bool): return raw text if parsing fails.
    """

    name = "json-lax"

    def parse(self, text: str, args: dict) -> Any:
        transform = args.get("transform")
        allow_non_json = args.get("allow_non_json", False)
        unquoted_keys = args.get("unquoted_keys", False)
        string_delims = args.get("string_delims") or []

        working = text
        if _LAX_OPEN in working or _LAX_CLOSE in working:
            raise ValueError(
                "json-lax: input contains reserved sentinel characters (\\x01/\\x02); cannot parse safely."
            )

        captured_strings: list[str] = []
        for pair in string_delims:
            if not (isinstance(pair, (list, tuple)) and len(pair) == 2):
                raise ValueError(f"json-lax: each entry in string_delims must be a [open, close] pair; got {pair!r}")
            open_d, close_d = pair
            pattern = re.escape(open_d) + r"(.*?)" + re.escape(close_d)

            def _capture(m: re.Match) -> str:
                captured_strings.append(m.group(1))
                return f"{_LAX_OPEN}{len(captured_strings) - 1}{_LAX_CLOSE}"

            working = re.sub(pattern, _capture, working, flags=re.DOTALL)

        if unquoted_keys:
            working = re.sub(r"(?<=[{,])(\w+):", r'"\1":', working)

        for i, s in enumerate(captured_strings):
            working = working.replace(f"{_LAX_OPEN}{i}{_LAX_CLOSE}", json.dumps(s))

        try:
            value = json.loads(working)
        except json.JSONDecodeError as e:
            if allow_non_json:
                return text.strip() if isinstance(text, str) else text
            raise ValueError(
                f"json-lax: could not parse after dialect transforms.\n"
                f"Original: {text!r}\nTransformed: {working!r}\nError: {e}"
            ) from e
        return _apply_transform(value, transform)
