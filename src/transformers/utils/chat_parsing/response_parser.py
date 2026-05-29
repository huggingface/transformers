# Copyright 2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

from __future__ import annotations

from typing import Any

import regex as re

from .content_parsers import STREAMABLE_PARSERS, process_field
from .response_templates import ResponseTemplate, ResponseTemplateField, load_response_template


def parse_response(text: str, response_template: dict | ResponseTemplate, *, prefix: str | None = None) -> dict:
    """The main function for response parsing when you don't want streaming. Takes generated output
    and the prompt prefix and parses them  without streaming any events, then returns the parsed message.
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
            best, hold_start = self._scan(watch, eos)

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

            # No committable match in the current buffer.
            if eos:
                if self._pos < len(self._buffer):
                    self._accumulate(events, self._buffer[self._pos :])
                    self._pos = len(self._buffer)
                self._close_current(events)
                break
            # Stream everything up to the earliest still-pending delimiter. When
            # nothing is pending `hold_start == len(self._buffer)`, so this flushes
            # the whole buffer; otherwise we hold the (possibly partial) delimiter
            # bytes back until more input resolves them.
            if hold_start > self._pos:
                self._accumulate(events, self._buffer[self._pos : hold_start])
                self._pos = hold_start
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

    def _scan(
        self, watch: list[tuple[str, ResponseTemplateField]], eos: bool
    ) -> tuple[tuple[str, ResponseTemplateField, re.Match] | None, int]:
        """Single pass over the watched delimiters, using the `regex` module's
        partial matching to decide -- per delimiter -- whether it can be committed
        now or must be held. Returns `(best, hold_start)`:

        * `best` is the earliest-starting delimiter we can safely commit *now*
          (longest on ties, opens before closes), or `None`.
        * `hold_start` is the leftmost buffer position occupied by a still-pending
          match: a partial (incomplete) delimiter, or a complete one ending at the
          buffer edge that more input could still grow. Bytes before it are safe to
          emit; bytes from it onward must be held. It stays `len(self._buffer)` when
          nothing is pending, letting the caller flush the whole buffer.

        A complete match is committable only if it starts strictly before
        `hold_start` -- otherwise an earlier (or co-located) pending delimiter could
        turn out to be the real one. At EOS nothing can grow, so partial matching is
        skipped and every complete match is committable.

        (The `regex` module always reports the empty string as a live prefix, so a
        partial search with no real match returns a zero-width match at the buffer
        end; that lands in the pending branch with `start == len(self._buffer)`, a
        no-op for `hold_start`.)
        """
        best_key: tuple | None = None
        best: tuple[str, ResponseTemplateField, re.Match] | None = None
        hold_start = len(self._buffer)
        for kind, fld in watch:
            pattern = fld.open_re if kind == "open" else fld.close_re
            assert pattern is not None  # watched delimiters always carry a regex
            if eos:
                m = pattern.search(self._buffer, self._pos)
            else:
                m = pattern.search(self._buffer, self._pos, partial=True)
            if m is None:
                continue
            if not eos and (m.partial or self._can_grow(kind, fld, m)):
                # Pending: can't commit, and blocks emitting from its start onward.
                hold_start = min(hold_start, m.start())
                continue
            key = (m.start(), -(m.end() - m.start()), 0 if kind == "open" else 1, fld.name)
            if best_key is None or key < best_key:
                best_key, best = key, (kind, fld, m)
        # A committable match co-located with or after a pending one must wait too:
        # the pending delimiter starts no later and might be the one that fires.
        if best is not None and best[2].start() >= hold_start:
            best = None
        return best, hold_start

    def _can_grow(self, kind: str, fld: ResponseTemplateField, m: re.Match) -> bool:
        r"""Whether a *complete* match ending at the current buffer edge could still
        change as more input arrives -- in which case we defer rather than commit. A
        match ending before the edge has already seen its terminating byte and is
        final. At the edge: zero-width matches (`$` / `\Z`) are only real at true
        EOS; a fully-present literal that no other literal in its set extends cannot
        grow (the fast path that keeps literal delimiters zero-latency); anything
        else (regex delimiters, prefix-overlapping literal lists) might."""
        if m.end() != len(self._buffer):
            return False
        if m.start() == m.end():
            return True
        lits, can_extend = (
            (fld.open_lits, fld.open_lit_can_extend) if kind == "open" else (fld.close_lits, fld.close_lit_can_extend)
        )
        return lits is None or can_extend

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
