# Copyright 2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Spec loading and validation for response_format dicts."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from ...utils import logging
from .content_parsers import CONTENT_PARSERS


logger = logging.get_logger(__name__)


_ALLOWED_FIELD_KEYS = {
    "open", "open_pattern", "close", "close_pattern",
    "content", "content_args", "repeats", "optional", "assemble", "coerce",
}  # fmt: skip
_ALLOWED_TOP_KEYS = {"version", "defaults", "fields", "start_anchor", "start_anchor_pattern"}
_COERCE_OPTS = {"int", "float", "bool"}
_EOS_RE = re.compile(r"\Z")


@dataclass
class Field:
    name: str
    open_re: re.Pattern | None
    open_lit: str | None
    close_re: re.Pattern | None
    close_lit: str | None
    content: str
    content_args: dict
    repeats: bool
    optional: bool
    assemble: Any
    coerce: str | None


@dataclass
class Spec:
    defaults: dict
    fields: dict[str, Field]
    implicit: str | None = None
    start_anchor_re: re.Pattern | None = None
    start_anchor_lit: str | None = None

    def truncate_past_last_anchor(self, text: str, *, log_if_missing: bool = True) -> str:
        """Right-truncate `text` to the suffix that begins immediately after
        the last `start_anchor` match. The intent: when callers pass a chat
        prompt (system + user + ... + assistant header) as a prefix, only
        the bytes after the assistant header — i.e., the actual message
        body — flow into the state machine.

        Returns `text` unchanged when no anchor is configured. Returns it
        unchanged (warning unless suppressed) when an anchor is configured
        but not found."""
        if self.start_anchor_re is None:
            return text
        if self.start_anchor_lit is not None:
            idx = text.rfind(self.start_anchor_lit)
            if idx < 0:
                if log_if_missing:
                    logger.warning_once(
                        "response_format defines start_anchor but the anchor was not found "
                        "in the prefix; the parser will process the entire prefix instead."
                    )
                return text
            return text[idx + len(self.start_anchor_lit) :]
        last_end: int | None = None
        for m in self.start_anchor_re.finditer(text):
            last_end = m.end()
        if last_end is None:
            if log_if_missing:
                logger.warning_once(
                    "response_format defines start_anchor_pattern but the anchor was not "
                    "found in the prefix; the parser will process the entire prefix instead."
                )
            return text
        return text[last_end:]


def _compile_anchor(scope: str, fd: dict, lit_key: str, pat_key: str) -> tuple[re.Pattern | None, str | None]:
    """Return (compiled, literal). `literal` is populated only when the anchor
    was given as a literal string (not "eos" and not a regex); the streaming
    executor uses it to compute safe-to-emit boundaries. `scope` prefixes
    error messages — e.g. `"Field 'thinking'"` or `"response_format"`."""
    if lit_key in fd and pat_key in fd:
        raise ValueError(f"{scope}: cannot specify both '{lit_key}' and '{pat_key}'")
    if lit_key in fd:
        lit = fd[lit_key]
        if lit == "eos":
            return _EOS_RE, None
        return re.compile(re.escape(lit), re.DOTALL), lit
    if pat_key in fd:
        try:
            return re.compile(fd[pat_key], re.DOTALL), None
        except re.error as e:
            raise ValueError(f"{scope}: invalid {pat_key} regex: {e}") from e
    return None, None


def load_spec(spec: dict | Spec) -> Spec:
    """Validate and normalize a response_format dict. Idempotent: a `Spec`
    passed in is returned as-is."""
    if isinstance(spec, Spec):
        return spec
    if not isinstance(spec, dict):
        raise ValueError(f"response_format must be a dict, got {type(spec).__name__}")
    unknown = set(spec) - _ALLOWED_TOP_KEYS
    if unknown:
        raise ValueError(f"Unknown keys in response_format: {sorted(unknown)}")
    version = spec.get("version", 1)
    if version != 1:
        raise ValueError(f"Unsupported response_format version: {version}")
    defaults = spec.get("defaults", {})
    if not isinstance(defaults, dict):
        raise ValueError("response_format.defaults must be a dict")
    fields_raw = spec.get("fields", {})
    if not isinstance(fields_raw, dict) or not fields_raw:
        raise ValueError("response_format.fields must be a non-empty dict")

    fields: dict[str, Field] = {}
    implicit_fields: list[str] = []
    for name, fd in fields_raw.items():
        if not isinstance(fd, dict):
            raise ValueError(f"Field '{name}' must be a dict")
        bad = set(fd) - _ALLOWED_FIELD_KEYS
        if bad:
            raise ValueError(f"Field '{name}': unknown keys {sorted(bad)}")
        content = fd.get("content", "text")
        if content not in CONTENT_PARSERS:
            raise ValueError(
                f"Field '{name}': unknown content parser '{content}'. Available: {sorted(CONTENT_PARSERS)}"
            )
        coerce = fd.get("coerce")
        if coerce is not None and coerce not in _COERCE_OPTS:
            raise ValueError(f"Field '{name}': coerce must be one of {sorted(_COERCE_OPTS)}")
        open_re, open_lit = _compile_anchor(f"Field '{name}'", fd, "open", "open_pattern")
        close_re, close_lit = _compile_anchor(f"Field '{name}'", fd, "close", "close_pattern")
        if open_re is None:
            implicit_fields.append(name)
        fields[name] = Field(
            name=name,
            open_re=open_re,
            open_lit=open_lit,
            close_re=close_re,
            close_lit=close_lit,
            content=content,
            content_args=fd.get("content_args", {}),
            repeats=fd.get("repeats", False),
            optional=fd.get("optional", True),
            assemble=fd.get("assemble"),
            coerce=coerce,
        )
    if len(implicit_fields) > 1:
        raise ValueError(
            "At most one field may omit 'open'/'open_pattern' (that field becomes the "
            f"implicit-open / leftover sink). Found: {', '.join(implicit_fields)}"
        )
    start_anchor_re, start_anchor_lit = _compile_anchor(
        "response_format", spec, "start_anchor", "start_anchor_pattern"
    )
    return Spec(
        defaults=dict(defaults),
        fields=fields,
        implicit=implicit_fields[0] if implicit_fields else None,
        start_anchor_re=start_anchor_re,
        start_anchor_lit=start_anchor_lit,
    )


__all__ = ["Field", "Spec", "load_spec"]
