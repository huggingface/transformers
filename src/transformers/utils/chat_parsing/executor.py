# Copyright 2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Region executor: a stream-first state machine that turns text into a
message dict according to a response_format spec.

The executor advances a single pointer through a growing buffer. At any point
there is a "current region" -- either the implicit/sink field (if the spec
defined one), or an explicit field whose open pattern just matched -- and a
watchlist of regex patterns whose match would end or redirect it. On each
`feed()` call it commits every byte it can safely classify and holds back any
trailing bytes that could still be the prefix of an upcoming delimiter, so
events fire incrementally without waiting for the whole response.

`parse_response()` is a thin wrapper that feeds the whole string at once; the
correctness invariant is that streamed + finalize() equals whole-string parse
for any chunking of the input."""

from __future__ import annotations

import re
from typing import Any

from .content_parsers import process_field
from .spec import Field, load_spec


# When a delimiter is specified as a regex, we can't reason exactly about how
# many trailing bytes might still complete a match, so we conservatively hold
# back a fixed window. 64 chars covers every delimiter regex currently in use.
_REGEX_STREAM_LOOKBACK = 64


def parse_response(text: str, response_format: dict) -> dict:
    """Whole-string parse: returns the assembled message dict.

    Implemented as a single-chunk feed through `ResponseEventStream` so both
    paths share one state machine."""
    stream = ResponseEventStream(response_format)
    stream.feed(text)
    return stream.finalize()


class ResponseEventStream:
    """Incremental parser over a `response_format` spec.

    Usage:
        streamer = ResponseEventStream(response_format)
        for chunk in model_text_stream:
            for event in streamer.feed(chunk):
                handle(event)
        message = streamer.finalize()
        for event in streamer.final_events:
            handle(event)  # region_close for any EOS-bounded region, then stream_end

    Event types (all dicts):
      - {"type": "region_open",  "field": name, "meta": {<named captures>}}
      - {"type": "region_chunk", "field": name, "text": <committed text>}  # only for text/raw
      - {"type": "region_close", "field": name, "value": <parsed+assembled value>}
      - {"type": "stream_end"}

    Correctness invariant (property-tested): for any chunking of the input,
    finalize() returns the same dict as the whole-string parse_response().
    """

    def __init__(self, response_format: dict):
        self._spec = load_spec(response_format)
        self._buffer: str = ""
        self._pos: int = 0
        self._output: dict[str, Any] = dict(self._spec.defaults)
        self._implicit_name: str | None = self._spec.implicit
        # Unified current-region state: starts in the implicit region (or a
        # null sink if none was declared), and returns there after every close.
        # For explicit regions `_opened` flips to True eagerly on the open
        # match; for the implicit region it flips lazily on the first byte.
        self._current: str | None = self._implicit_name
        self._captures: dict[str, str] = {}
        self._body: str = ""
        self._opened: bool = False
        self._finalized: bool = False
        self.final_events: list[dict] = []

    def feed(self, text: str) -> list[dict]:
        if self._finalized:
            raise RuntimeError("ResponseEventStream already finalized")
        if text:
            self._buffer += text
        events: list[dict] = []
        self._process(events, eos=False)
        return events

    def finalize(self) -> dict:
        if self._finalized:
            raise RuntimeError("ResponseEventStream already finalized")
        self._process(self.final_events, eos=True)
        missing = [n for n, f in self._spec.fields.items() if not f.optional and n not in self._output]
        if missing:
            raise ValueError(f"Required response_format fields missing from parsed output: {missing}")
        defaults = self._spec.defaults
        self._output = {k: v for k, v in self._output.items() if k in defaults or not _is_empty(v)}
        self._finalized = True
        self.final_events.append({"type": "stream_end"})
        return self._output

    # --- state machine ---

    def _process(self, events: list[dict], eos: bool) -> None:
        while True:
            watch = self._watchlist()
            best = self._best_match(watch)
            # Mid-stream, a match that reaches the end of the current buffer
            # is ambiguous: `$`-alts are zero-width at len(buffer), and regex
            # quantifiers like `\w+` may still extend if more input arrives.
            # Literal non-zero-width matches can never extend, so we always
            # commit those. At EOS we commit whatever we have and rely on the
            # no-progress guard below to stop re-firing zero-width no-ops.
            if (
                best is not None
                and not eos
                and _should_defer(best[1], best[2], len(self._buffer), is_open=(best[0] == "open"))
            ):
                best = None

            if best is not None:
                kind, fld, m = best
                if m.start() > self._pos:
                    self._accumulate(events, self._buffer[self._pos : m.start()])
                self._pos = m.end()
                if kind == "open":
                    self._close_current(events)
                    self._open_explicit(events, fld, m)
                else:  # "close" (always the implicit region's close here,
                    #   since explicit regions only expose their own close)
                    had_content = self._opened
                    self._close_current(events)
                    # Zero-width close on an already-empty region would just
                    # re-fire next iteration -- bail out to make progress.
                    if not had_content and m.start() == m.end():
                        break
                continue

            # No match in the current buffer.
            if eos:
                if self._pos < len(self._buffer):
                    self._accumulate(events, self._buffer[self._pos :])
                    self._pos = len(self._buffer)
                self._close_current(events)
                break
            if not watch:
                # No termination patterns active (explicit region whose
                # close_re is None, or an implicit region with no close and
                # no explicit opens). Safe to stream everything buffered.
                if self._pos < len(self._buffer):
                    self._accumulate(events, self._buffer[self._pos :])
                    self._pos = len(self._buffer)
                break
            hold = self._max_hold(watch)
            safe_end = len(self._buffer) - hold
            if safe_end > self._pos:
                self._accumulate(events, self._buffer[self._pos : safe_end])
                self._pos = safe_end
            break

    # --- watchlist helpers ---

    def _watchlist(self) -> list[tuple[str, Field]]:
        """Patterns we care about right now: the close of the currently-open
        explicit region, or -- if we're in the implicit/null region -- every
        explicit open plus the implicit's own close (if any)."""
        if self._current is not None and self._current != self._implicit_name:
            fld = self._spec.fields[self._current]
            return [("close", fld)] if fld.close_re is not None else []
        watch: list[tuple[str, Field]] = []
        for fld in self._spec.fields.values():
            if fld.open_re is not None:
                watch.append(("open", fld))
        if self._implicit_name is not None:
            impl = self._spec.fields[self._implicit_name]
            if impl.close_re is not None:
                watch.append(("close", impl))
        return watch

    def _best_match(self, watch: list[tuple[str, Field]]) -> tuple[str, Field, re.Match] | None:
        """Earliest-starting (longest on ties, opens before closes) match."""
        best_key: tuple | None = None
        best: tuple[str, Field, re.Match] | None = None
        for kind, fld in watch:
            regex = fld.open_re if kind == "open" else fld.close_re
            m = regex.search(self._buffer, self._pos)
            if m is None:
                continue
            key = (m.start(), -(m.end() - m.start()), 0 if kind == "open" else 1, fld.name)
            if best_key is None or key < best_key:
                best_key, best = key, (kind, fld, m)
        return best

    def _max_hold(self, watch: list[tuple[str, Field]]) -> int:
        hold = 0
        for kind, fld in watch:
            literal = fld.open_lit if kind == "open" else fld.close_lit
            hold = max(hold, _pattern_hold(self._buffer, self._pos, literal))
        return hold

    # --- region lifecycle ---

    def _accumulate(self, events: list[dict], text: str) -> None:
        """Route `text` into the currently active region. When the current
        region is the null sink (no implicit declared, no explicit open), we
        silently discard."""
        if not text or self._current is None:
            return
        fld = self._spec.fields[self._current]
        if not self._opened:
            events.append({"type": "region_open", "field": self._current, "meta": dict(self._captures)})
            self._opened = True
        self._body += text
        if fld.content in ("text", "raw"):
            events.append({"type": "region_chunk", "field": self._current, "text": text})

    def _open_explicit(self, events: list[dict], fld: Field, m: re.Match) -> None:
        self._current = fld.name
        self._captures = {k: v for k, v in m.groupdict().items() if v is not None}
        self._body = ""
        events.append({"type": "region_open", "field": fld.name, "meta": dict(self._captures)})
        self._opened = True

    def _close_current(self, events: list[dict]) -> None:
        """Close the current region and reset to the implicit/null region.
        Skipped (aside from the reset) when the current region never opened --
        avoids vacuous open/close pairs at every explicit boundary."""
        if self._current is None or not self._opened:
            self._reset_to_implicit()
            return
        fld = self._spec.fields[self._current]
        value = process_field(self._body, fld, self._captures)
        if fld.repeats:
            self._output.setdefault(self._current, []).append(value)
        else:
            self._output[self._current] = value
        events.append({"type": "region_close", "field": self._current, "value": value})
        self._reset_to_implicit()

    def _reset_to_implicit(self) -> None:
        self._current = self._implicit_name
        self._captures = {}
        self._body = ""
        self._opened = False


def _is_empty(v: Any) -> bool:
    return v is None or (isinstance(v, (list, dict, str)) and not v)


def _should_defer(fld: Field, m: re.Match, buf_len: int, is_open: bool) -> bool:
    """Whether a match that already succeeded should nonetheless be deferred
    until more input (or EOS) arrives. A match is "ambiguous at the edge" when
    it ends at the current buffer end and either (a) it was zero-width (`$`-alt
    style) or (b) the delimiter was specified as a regex, which may still
    extend with more input. Literal non-zero-width matches never extend."""
    if m.end() != buf_len:
        return False
    literal = fld.open_lit if is_open else fld.close_lit
    return literal is None or m.start() == m.end()


def _pattern_hold(buffer: str, start: int, literal: str | None) -> int:
    """How many trailing bytes of `buffer[start:]` must stay held because they
    might be a partial match for the delimiter. For literal delimiters this is
    the longest-trailing-prefix of the literal; for regex delimiters (literal
    is None) we use a conservative constant window. Caller must have already
    verified that the delimiter's regex has no full match from `start` onward."""
    avail = len(buffer) - start
    if avail <= 0:
        return 0
    if literal is not None:
        max_k = min(len(literal) - 1, avail)
        for k in range(max_k, 0, -1):
            if buffer.endswith(literal[:k]):
                return k
        return 0
    return min(avail, _REGEX_STREAM_LOOKBACK)
