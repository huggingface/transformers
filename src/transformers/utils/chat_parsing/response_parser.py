# Copyright 2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Response parsing walks a single pointer through a growing buffer. At any point
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

from .content_parsers import STREAMABLE_PARSERS, process_field
from .response_templates import ResponseTemplate, ResponseTemplateField, load_response_template


def parse_response(text: str, response_template: dict | ResponseTemplate, *, prefix: str | None = None) -> dict:
    """A convenience function for response parsing when you don't want streaming. Takes a whole output + text
    and parses it without streaming any events, then returns the parsed message.
    """
    response_template = load_response_template(response_template)
    stream = ResponseParser(response_template, prefix=prefix)
    stream.feed(text)
    message, _events = stream.finalize()
    return message


class ResponseParser:
    """This class implements a streaming parser with a `response_template`. If you don't need streaming and
    just want to parse a complete message, use the `parse_response` function above. Streaming parsing emits
    events indicating when regions (message fields) are opened and closed, with the model writing to the region
    that is currently open.

    Usage:
        parser = ResponseParser(response_template, prefix=chat_prompt)
        for event in parser.initial_events:
            handle(event)
        for chunk in model_text_stream:
            for event in parser.feed(chunk):
                handle(event)
        message, final_events = parser.finalize()
        for event in final_events:
            handle(event)

    Events can be either "region_open", "region_chunk", or "region_close".

    ResponseParser requires the chat `prefix` (i.e. the chat history, the prefill before the current generation).
    This is because chat templates or assistant prefills can sometimes write part of the message, and if we
    only see the model output, and not the template, then we can't reliably parse the message in those cases.
    Any events produced while consuming the prefix are exposed as `initial_events`, so renderers can show
    prefill regions before the model writes anything; closed prefill regions also land in the output dict.
    """

    def __init__(self, response_template: dict | ResponseTemplate, prefix: str | None = None):
        self._spec = load_response_template(response_template)
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
        self.initial_events: list[dict] = []
        if prefix:
            self._consume_prefix(prefix)

    def _consume_prefix(self, prefix: str) -> None:
        """Loads the prefix (the chat prefill sent to the model), right-truncates it to the start of the
        assistant message (as determined by start_anchor) and then runs the remainder through the parser.
        Events produced while processing the prefix are stashed on `initial_events` so callers can replay
        them into a renderer before feeding model output.

        Think of this as the "get the parser up to speed on the story so far" method.
        """
        truncated = self._spec.truncate_past_last_anchor(prefix)
        if not truncated:
            return
        self._buffer = truncated
        self._process(self.initial_events, eos=False)

    def feed(self, text: str) -> list[dict]:
        """Feeds more text/tokens from the model output into the tokenizer, and returns any events that result
        (regions entered or left). This is the method you want to call after each generation step."""
        if self._finalized:
            raise RuntimeError("ResponseParser already finalized")
        if text:
            self._buffer += text
        events: list[dict] = []
        self._process(events, eos=False)
        return events

    def finalize(self) -> tuple[dict, list[dict]]:
        """Close the stream and return the final message dict together with
        any finalization events. This is necessary because some regions may only
        end at the end of the sequence, so you won't see the event telling you they're
        ready until the sequence is finalized."""

        def _is_empty(v: Any) -> bool:
            return v is None or (isinstance(v, (list, dict, str)) and not v)

        if self._finalized:
            raise RuntimeError("ResponseParser already finalized")
        events: list[dict] = []
        self._process(events, eos=True)
        missing = [n for n, f in self._spec.fields.items() if not f.optional and n not in self._output]
        if missing:
            raise ValueError(f"Required response_template fields missing from parsed output: {missing}")
        defaults = self._spec.defaults
        self._output = {k: v for k, v in self._output.items() if k in defaults or not _is_empty(v)}
        self._finalized = True
        return self._output, events

    # Matt: The private methods below cover the internals of the class and are mostly agent-written.

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

    def _watchlist(self) -> list[tuple[str, ResponseTemplateField]]:
        """Patterns we care about right now: the close of the currently-open
        explicit region, or -- if we're in the implicit/null region -- every
        explicit open plus the implicit's own close (if any)."""
        if self._current is not None and self._current != self._implicit_name:
            fld = self._spec.fields[self._current]
            return [("close", fld)] if fld.close_re is not None else []
        watch: list[tuple[str, ResponseTemplateField]] = []
        for fld in self._spec.fields.values():
            if fld.open_re is not None:
                watch.append(("open", fld))
        if self._implicit_name is not None:
            impl = self._spec.fields[self._implicit_name]
            if impl.close_re is not None:
                watch.append(("close", impl))
        return watch

    def _best_match(
        self, watch: list[tuple[str, ResponseTemplateField]]
    ) -> tuple[str, ResponseTemplateField, re.Match] | None:
        """Earliest-starting (longest on ties, opens before closes) match."""
        best_key: tuple | None = None
        best: tuple[str, ResponseTemplateField, re.Match] | None = None
        for kind, fld in watch:
            regex = fld.open_re if kind == "open" else fld.close_re
            assert regex is not None  # Should always be correct, and keeps ty happy
            m = regex.search(self._buffer, self._pos)
            if m is None:
                continue
            key = (m.start(), -(m.end() - m.start()), 0 if kind == "open" else 1, fld.name)
            if best_key is None or key < best_key:
                best_key, best = key, (kind, fld, m)
        return best

    def _max_hold(self, watch: list[tuple[str, ResponseTemplateField]]) -> int:
        hold = 0
        for kind, fld in watch:
            literals = fld.open_lits if kind == "open" else fld.close_lits
            hold = max(hold, _pattern_hold(self._buffer, self._pos, literals))
        return hold

    def _accumulate(self, events: list[dict], text: str) -> None:
        """Route `text` into the currently active region. When the current
        region is the null sink (no implicit declared, no explicit open), we
        silently discard. Every routed chunk emits a `region_chunk` event so
        consumers can render live; `dirty=True` flags chunks from structured
        parsers (json, xml-inline, kv-lines) whose raw bytes will only be
        parsed into the final value on close."""
        if not text or self._current is None:
            return
        fld = self._spec.fields[self._current]
        if not self._opened:
            events.append({"type": "region_open", "field": self._current})
            self._opened = True
        self._body += text
        dirty = fld.content not in STREAMABLE_PARSERS
        events.append({"type": "region_chunk", "field": self._current, "text": text, "dirty": dirty})

    def _open_explicit(self, events: list[dict], fld: ResponseTemplateField, m: re.Match) -> None:
        self._current = fld.name
        self._captures = {k: v for k, v in m.groupdict().items() if v is not None}
        self._body = ""
        self._opened = True
        events.append({"type": "region_open", "field": fld.name})

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


def _should_defer(fld: ResponseTemplateField, m: re.Match, buf_len: int, is_open: bool) -> bool:
    """Whether a match that already succeeded should nonetheless be deferred
    until more input (or EOS) arrives. A match is "ambiguous at the edge" when
    it ends at the current buffer end and either (a) it was zero-width (`$`-alt
    style), (b) the delimiter was specified as a regex (which may still extend
    with more input), or (c) the delimiter is a literal-alternation where one
    literal is a prefix of another (so a short match could yet grow). Otherwise,
    a non-zero-width literal match cannot be extended and is safe to commit."""
    if m.end() != buf_len:
        return False
    if is_open:
        literals, can_extend = fld.open_lits, fld.open_lit_can_extend
    else:
        literals, can_extend = fld.close_lits, fld.close_lit_can_extend
    return literals is None or can_extend or m.start() == m.end()


def _pattern_hold(buffer: str, start: int, literals: list[str] | None) -> int:
    """How many trailing bytes of `buffer[start:]` must stay held because they
    might be a partial match for the delimiter. For literal delimiters this is
    the longest-trailing-prefix across all literals; for regex delimiters
    (literals is None) we use a conservative constant window. Caller must have
    already verified that the delimiter's regex has no full match from `start`
    onward."""
    avail = len(buffer) - start
    if avail <= 0:
        return 0
    if literals is not None:
        best = 0
        for literal in literals:
            max_k = min(len(literal) - 1, avail)
            for k in range(max_k, best, -1):
                if buffer.endswith(literal[:k]):
                    best = k
                    break
        return best
    return min(avail, 64)  # 64 chosen as a safe default for now, but later we might consider making this configurable
