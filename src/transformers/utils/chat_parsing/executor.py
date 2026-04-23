# Copyright 2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Region executor: a stream-first state machine that turns text into a
message dict according to a response_format spec.

V1 execution model: whole-string (buffer all text, then run one pass). The
state machine is structured so that V2 streaming events can later be emitted
incrementally against the same core loop.
"""

from __future__ import annotations

import re
from typing import Any

from .content_parsers import parse_content
from .spec import load_spec


def parse_response(text: str, response_format: dict) -> dict:
    """Whole-string parse: return the assembled message dict."""
    return _execute(load_spec(response_format), text)


class ResponseEventStream:
    """Stateful event-driven parser.

    V1 emits no per-chunk events; it buffers text fed via `feed()` and produces
    the full message dict on `finalize()`. V2 will emit region_open / region_chunk
    / region_close events incrementally.
    """

    def __init__(self, response_format: dict):
        self._spec = load_spec(response_format)
        self._buffer: list[str] = []
        self._finalized = False

    def feed(self, text: str) -> list[dict]:
        if self._finalized:
            raise RuntimeError("ResponseEventStream already finalized")
        self._buffer.append(text)
        return []  # V2: streaming events

    def finalize(self) -> dict:
        if self._finalized:
            raise RuntimeError("ResponseEventStream already finalized")
        self._finalized = True
        return _execute(self._spec, "".join(self._buffer))


def _execute(spec: dict, text: str) -> dict:
    pos = 0
    output: dict[str, Any] = dict(spec["defaults"])
    fields = spec["fields"]
    implicit_name = spec["implicit"]
    implicit_buffer = ""

    while pos < len(text):
        prev_pos = pos
        # Find the next event: either an explicit field opening, or the implicit field closing.
        candidates = []
        for fld in fields.values():
            if fld["open_re"] is not None:
                m = fld["open_re"].search(text, pos)
                if m is not None:
                    candidates.append((m.start(), -(m.end() - m.start()), "open", fld, m))
        if implicit_name is not None and fields[implicit_name]["close_re"] is not None:
            impl = fields[implicit_name]
            m = impl["close_re"].search(text, pos)
            if m is not None:
                candidates.append((m.start(), -(m.end() - m.start()), "implicit_close", impl, m))

        if not candidates:
            if implicit_name is not None:
                implicit_buffer += text[pos:]
            break

        candidates.sort()
        _, _, event, fld, m = candidates[0]

        # Pre-event bytes belong to the implicit region (if any).
        if implicit_name is not None and m.start() > pos:
            implicit_buffer += text[pos : m.start()]

        if event == "implicit_close":
            _emit(output, fld, implicit_buffer, {})
            implicit_buffer = ""
            pos = m.end()
        else:  # "open"
            if implicit_name is not None and implicit_buffer:
                _emit(output, fields[implicit_name], implicit_buffer, {})
                implicit_buffer = ""
            captures = {k: v for k, v in m.groupdict().items() if v is not None}
            close = fld["close_re"].search(text, m.end()) if fld["close_re"] is not None else None
            if close is None:
                body, pos = text[m.end() :], len(text)
            else:
                body, pos = text[m.end() : close.start()], close.end()
            _emit(output, fld, body, captures)

        if pos <= prev_pos:
            # Safety net against schemas with zero-width open/close matches.
            raise ValueError(
                f"Parser made no progress at position {pos}. Check response_format for zero-width open/close patterns."
            )

    if implicit_name is not None and implicit_buffer:
        _emit(output, fields[implicit_name], implicit_buffer, {})

    missing = [n for n, f in fields.items() if not f["optional"] and n not in output]
    if missing:
        raise ValueError(f"Required response_format fields missing from parsed output: {missing}")
    defaults = spec["defaults"]
    return {
        k: v
        for k, v in output.items()
        if k in defaults or not (v is None or (isinstance(v, (list, dict, str)) and not v))
    }


def _emit(output: dict, field: dict, body: str, captures: dict) -> None:
    value = parse_content(body, field["content"], field["content_args"])
    value = _apply_assemble(field["assemble"], captures, value)
    value = _apply_coerce(value, field["coerce"])
    if field["repeats"]:
        output.setdefault(field["name"], []).append(value)
    else:
        output[field["name"]] = value


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

    # If the whole string is a single `{placeholder}`, pass the value through as-is
    # (structured content like dicts/lists must not be stringified).
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
