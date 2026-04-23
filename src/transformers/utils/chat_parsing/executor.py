# Copyright 2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Region executor: stream-first state machine that turns text into a message dict
according to a response_format spec.

V1 execution model: whole-string (buffer all text, then run one pass). The state
machine is structured so that V2 streaming events (region_open / region_chunk /
region_close) can be emitted incrementally against the same core loop."""

from __future__ import annotations

import re
from typing import Any

from .content_parsers import parse_content
from .spec import FieldSpec, ResponseFormatSpec, load_spec


def parse_response(text: str, response_format: dict | ResponseFormatSpec) -> dict:
    """Whole-string parse: execute `response_format` against `text` and return the
    assembled message dict."""
    spec = response_format if isinstance(response_format, ResponseFormatSpec) else load_spec(response_format)
    return _execute(spec, text)


class ResponseEventStream:
    """Stateful event-driven parser.

    V1 emits no per-chunk events; it buffers text fed via `feed()` and produces
    the full message dict on `finalize()`. V2 will emit region_open / region_chunk
    / region_close events incrementally.
    """

    def __init__(self, response_format: dict | ResponseFormatSpec):
        self._spec = response_format if isinstance(response_format, ResponseFormatSpec) else load_spec(response_format)
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


def _execute(spec: ResponseFormatSpec, text: str) -> dict:
    pos = 0
    output: dict[str, Any] = dict(spec.defaults)
    implicit_name = spec.implicit_field_name
    implicit_buffer = ""

    while pos < len(text):
        prev_pos = pos
        candidates: list[tuple[int, int, str, str, FieldSpec, tuple[int, int, dict]]] = []
        for name, fld in spec.fields.items():
            if fld.is_implicit:
                continue
            m = _match_open(text, pos, fld)
            if m is not None:
                candidates.append((m[0], -(m[1] - m[0]), "open", name, fld, m))

        if implicit_name is not None:
            impl_fld = spec.fields[implicit_name]
            close_m = _match_close(text, pos, impl_fld)
            if close_m is not None:
                candidates.append(
                    (close_m[0], -(close_m[1] - close_m[0]), "implicit_close", implicit_name, impl_fld, close_m)
                )

        if not candidates:
            # No further events; any remainder goes to the implicit field.
            if implicit_name is not None:
                implicit_buffer += text[pos:]
            pos = len(text)
            break

        candidates.sort()
        _, _, event, name, fld, m = candidates[0]
        m_start, m_end, captures = m

        # Pre-event bytes belong to the implicit region (if any).
        if implicit_name is not None and m_start > pos:
            implicit_buffer += text[pos:m_start]

        if event == "implicit_close":
            _flush_implicit(output, fld, implicit_buffer)
            implicit_buffer = ""
            pos = m_end
        else:  # open
            if implicit_name is not None and implicit_buffer:
                _flush_implicit(output, spec.fields[implicit_name], implicit_buffer)
                implicit_buffer = ""
            close_m = _match_close(text, m_end, fld)
            if close_m is None:
                body = text[m_end:]
                pos = len(text)
            else:
                c_start, c_end, _ = close_m
                body = text[m_end:c_start]
                pos = c_end

            parsed = parse_content(body, fld.content, fld.content_args)
            value = _apply_assemble(fld.assemble, captures, parsed)
            value = _apply_coerce(value, fld.coerce)

            if fld.repeats:
                output.setdefault(name, [])
                output[name].append(value)
            else:
                output[name] = value

        if pos <= prev_pos:
            # Safety net: a schema that would otherwise loop forever on zero-width matches.
            raise ValueError(
                f"Parser made no progress at position {pos}. Check response_format for zero-width open/close patterns."
            )

    if implicit_name is not None and implicit_buffer:
        _flush_implicit(output, spec.fields[implicit_name], implicit_buffer)

    _check_required(output, spec)
    return _filter_empty(output, spec)


def _match_open(text: str, pos: int, field: FieldSpec) -> tuple[int, int, dict] | None:
    if field.open_literal is not None:
        idx = text.find(field.open_literal, pos)
        if idx < 0:
            return None
        return (idx, idx + len(field.open_literal), {})
    if field.open_pattern is not None:
        m = field.open_pattern.search(text, pos)
        if m is None:
            return None
        return (m.start(), m.end(), {k: v for k, v in m.groupdict().items() if v is not None})
    return None


def _match_close(text: str, pos: int, field: FieldSpec) -> tuple[int, int, dict] | None:
    if field.close_literal is not None:
        if field.close_literal == "eos":
            return (len(text), len(text), {})
        idx = text.find(field.close_literal, pos)
        if idx < 0:
            return None
        return (idx, idx + len(field.close_literal), {})
    if field.close_pattern is not None:
        m = field.close_pattern.search(text, pos)
        if m is None:
            return None
        return (m.start(), m.end(), {k: v for k, v in m.groupdict().items() if v is not None})
    return None


def _flush_implicit(output: dict, field: FieldSpec, buffer: str) -> None:
    parsed = parse_content(buffer, field.content, field.content_args)
    value = _apply_assemble(field.assemble, {}, parsed)
    value = _apply_coerce(value, field.coerce)
    if field.repeats:
        output.setdefault(field.name, [])
        output[field.name].append(value)
    else:
        output[field.name] = value


_SINGLE_PLACEHOLDER = re.compile(r"^\{(\w+)\}$")
_MULTI_PLACEHOLDER = re.compile(r"\{(\w+)\}")


def _apply_assemble(assemble: Any, captures: dict, content: Any) -> Any:
    if assemble is None:
        return content
    if isinstance(assemble, dict):
        return {k: _apply_assemble(v, captures, content) for k, v in assemble.items()}
    if isinstance(assemble, list):
        return [_apply_assemble(v, captures, content) for v in assemble]
    if isinstance(assemble, str):
        return _substitute_template(assemble, captures, content)
    return assemble


def _substitute_template(template: str, captures: dict, content: Any) -> Any:
    single = _SINGLE_PLACEHOLDER.fullmatch(template)
    if single:
        key = single.group(1)
        if key == "content":
            return content
        if key in captures:
            return captures[key]
        raise KeyError(
            f"assemble template '{template}' references unknown capture '{key}'. "
            f"Available: {sorted(list(captures) + ['content'])}"
        )

    def _sub(m: re.Match) -> str:
        key = m.group(1)
        if key == "content":
            return str(content)
        if key in captures:
            return str(captures[key])
        raise KeyError(
            f"assemble template '{template}' references unknown capture '{key}'. "
            f"Available: {sorted(list(captures) + ['content'])}"
        )

    return _MULTI_PLACEHOLDER.sub(_sub, template)


def _apply_coerce(value: Any, coerce: str | None) -> Any:
    if coerce is None:
        return value
    if coerce == "int":
        return int(value) if not isinstance(value, bool) else int(value)
    if coerce == "float":
        return float(value)
    if coerce == "bool":
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in ("true", "1", "yes")
        return bool(value)
    raise ValueError(f"Unknown coerce: {coerce}")


def _check_required(output: dict, spec: ResponseFormatSpec) -> None:
    missing = [name for name, fld in spec.fields.items() if not fld.optional and name not in output]
    if missing:
        raise ValueError(f"Required response_format fields missing from parsed output: {missing}")


def _filter_empty(output: dict, spec: ResponseFormatSpec) -> dict:
    """Always keep `defaults`. Drop explicit fields that ended up None or empty."""
    keep: dict[str, Any] = {}
    for k, v in output.items():
        if k in spec.defaults:
            keep[k] = v
            continue
        if v is None:
            continue
        if isinstance(v, (list, dict, str)) and not v:
            continue
        keep[k] = v
    return keep
